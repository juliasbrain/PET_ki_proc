import glob
import scipy.integrate
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from datetime import datetime
from fpdf import FPDF
from nilearn.plotting import plot_roi
from sklearn.linear_model import LinearRegression

from pet_ki_proc.app.services.config import ATLAS_NAMES
from pet_ki_proc.app.utils.helpers import extract_frame_info, get_thresh, get_thresh_method
from pet_ki_proc.app.utils.pathnames import pathnames

## PATLAK FITTING
def patlak_fit(subject, session, denoising, gm_mask, **kwargs):
    """
    **PET ki Patlak Fit**

    Calculate FDOPA influx constant (ki) for time frame 20 to 60 min post-FDOPA injection using Patlak model fitting
    in the whole striatum and subregions, with the cerebellum as the reference region.

    **Input:**
    - denoised and motion corrected PET image for frames from 4 -70 min (output from *pet_denoising*).
    - cerebellum mask in native PET space binarized and thresholded (output from *mask_registration*).
    - striatum masks in native PET space binarized and thresholded (output from *mask_registration*).

    **Output:**
    - Excel results sheet: *{subject}/ses-{session}/pet/patlak_fit/{subject}_ses-{session}_thr0.4_PET_ki.xlsx *. 
    - ki map: *{subject}/ses-{session}/pet/patlak_fit/patlak/{subject}_ses-{session}_ki_map_thr0.4.nii*.
    - r2 map: *{subject}/ses-{session}/pet/patlak_fit/patlak/{subject}_ses-{session}_r2_map_thr0.4.nii*.
    
    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - denoising (bool, optional): Apply denoising, default=True.
    - gm_mask (bool, optional): Restict striatum atlas to GM mask, default=True.
    - atlas (optional): Atlas for striatum and subregions, default='OGI_3'.
    - thr_striatum (optional): Threshold below this number, default=0.4.
    - thrp_striatum (optional): Threshold below this percentage.
    - thr_cerebellum (optional): Threshold below this number, default=0.9.
    - thrp_cerebellum (optional): Threshold below this percentage.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """
    
    # Parameters
    atlas = kwargs.get('atlas', 'OGI_3')
    atlas_name = ATLAS_NAMES.get(atlas, 'Unknown atlas')
    thr_striatum = kwargs.get('thr_striatum', 0.4)
    thrp_striatum = kwargs.get('thrp_striatum', None)
    thr_cerebellum = kwargs.get('thr_cerebellum', 0.9)
    thrp_cerebellum = kwargs.get('thrp_cerebellum', None)

    # Set path
    Cfg = pathnames(subject, session, atlas)
    os.makedirs(Cfg['patlakDir'], exist_ok=True)

    striatum_mask_files = [m for m in glob.glob(os.path.join(Cfg['atlas'], '*.nii.gz')) if 'cerebellum' not in m]
    striatum_masks = [os.path.splitext(os.path.splitext(os.path.basename(mask))[0])[0] for mask in striatum_mask_files]

    thresh_striatum, mask_thresh_striatum = get_thresh(thr_striatum, thrp_striatum, ('thr', 0.4))
    thresh_cerebellum, mask_thresh_cerebellum = get_thresh(thr_cerebellum, thrp_cerebellum, ('thr', 0.9))

    # Load PET image
    pet_img_path = os.path.join(Cfg['motioncorrDir'], f"r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5" + ('_denoised.nii.gz' if denoising else '.nii.gz')) 
    pet_img = nib.load(pet_img_path)
    pet_data = pet_img.get_fdata()

    # Load masks
    mask_images = {}
    for mask_name in striatum_masks:
        suffix = '_gm' if gm_mask else ''
        mask_img_path = os.path.join(Cfg['maskDir'],f"{thresh_striatum}{mask_thresh_striatum}{suffix}_native_{subject}_ses-{session}_{mask_name}.nii.gz")
        mask_images[mask_name] = nib.load(mask_img_path)

    cerebellum_img_path = os.path.join(Cfg['maskDir'], f'{thresh_cerebellum}{mask_thresh_cerebellum}_native_{subject}_ses-{session}_cerebellum.nii.gz')
    mask_images['cerebellum'] = nib.load(cerebellum_img_path)

    # Read frame duration from json file
    json_file = os.path.join(Cfg['petDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.json')
    _, _, frame_onsets, _, total_scan_duration = extract_frame_info(json_file)
    frames_tacs = frame_onsets[frame_onsets >= 5] # starting from 5 min post injection

    time_start = total_scan_duration - 50 # Selection for kicer (last 50 min to last 10 min) 
    time_end = total_scan_duration - 10
    frames_of_interest = np.where((frames_tacs >= time_start) & (frames_tacs <= time_end))[0]

    # Compute TACs (Time activity Curves)
    tacs = {}
    for mask_name, mask_img in {**mask_images}.items():
        mask_data = mask_img.get_fdata()
        tacs[mask_name] = np.array([np.mean(pet_data[..., j][mask_data > 0]) for j in range(len(frames_tacs))])

    # Plot TACs
    fig, ax = plt.subplots()
    ax.plot(frames_tacs, tacs['striatum'], 'r-', label='striatum')
    ax.plot(frames_tacs, tacs['cerebellum'], 'b-', label='cerebellum')
    ax.legend(loc='lower left')
    ax.set_xlabel('min')
    ax.set_ylabel('Bq/ml')
    fig.savefig(os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_{atlas}_striatum_and_cerebellum_TACs_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised' if denoising else ''}.png"), format='png')
    plt.close(fig)

    # Patlak Model Fitting - ROI-based
    ki_ROIbased = {}
    r2_ROIbased = {}
    x_var = np.array([scipy.integrate.cumulative_trapezoid(tacs['cerebellum'][:j+1], frames_tacs[:j+1], initial=0)[-1] / tacs['cerebellum'][j] for j in frames_of_interest])
    y_vars = {mask_name: np.array([tacs[mask_name][j] / tacs['cerebellum'][j] for j in frames_of_interest]) for mask_name in striatum_masks}
    for mask_name, y_var in y_vars.items():
        model = LinearRegression().fit(x_var.reshape(-1, 1), y_var)
        ki_ROIbased[mask_name] = model.coef_[0] # ki is beta1 of the regression model
        r2_ROIbased[mask_name] = model.score(x_var.reshape(-1, 1), y_var) # r2 quantifies the goodness of model fit

    # Plot Patlak plot
    fig, ax = plt.subplots()
    ax.plot(x_var, y_vars['striatum'])
    fig.savefig(os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_{atlas}_Patlak_plot_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised' if denoising else ''}.png"), format='png')
    plt.close(fig)

    # Patlak Model Fitting voxel-wise
    y_var_map = np.array([pet_data[..., j] / tacs['cerebellum'][j] for j in frames_of_interest]).transpose(1, 2, 3, 0)
    r2_map = np.zeros(pet_img.shape[:3])
    ki_map = np.zeros(pet_img.shape[:3])

    striatum_data = mask_images['striatum'].get_fdata()
    voxels_for_fitting = np.nonzero(striatum_data > 0) # Restrict fitting to striatum voxels
    voxels_for_fitting_tuples = list(zip(*voxels_for_fitting))

    for position in voxels_for_fitting_tuples:
        y_var_voxel = y_var_map[position].reshape(-1, 1)
        model = LinearRegression().fit(x_var.reshape(-1, 1), y_var_voxel)
        ki_map[position] = model.coef_[0, 0]
        r2_map[position] = model.score(x_var.reshape(-1, 1), y_var_voxel)

    # Calculate mean ki within masks
    ki_VoxelWise = {mask_name: np.mean(ki_map[mask_images[mask_name].get_fdata() > 0]) for mask_name in striatum_masks}
    r2_VoxelWise = {mask_name: np.mean(r2_map[mask_images[mask_name].get_fdata() > 0]) for mask_name in striatum_masks}

    ## SAVE RESULTS 
    kiname = f"{subject}_ses-{session}_{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised_PET_ki' if denoising else '_PET_ki'}"
    striatum_mask_thresh_name = f'{thresh_striatum}_{mask_thresh_striatum}'
    cerebellum_mask_thresh_name = f'{thresh_cerebellum}_{mask_thresh_cerebellum}'

    # save values in xslx
    kiheader = [f"ki_{name}" for name in striatum_masks]
    r2header = [f"r2_{name}" for name in striatum_masks]
    header = ['Subject_ID', 'Session', 'Atlas','striatum_mask_threshold', 'cerebellum_mask_threshold'] + kiheader + r2header
    ROIbased = [subject, session, atlas_name, striatum_mask_thresh_name, cerebellum_mask_thresh_name] + list(ki_ROIbased.values()) + list(r2_ROIbased.values())
    VoxelWise = [subject, session, atlas_name, striatum_mask_thresh_name, cerebellum_mask_thresh_name] + list(ki_VoxelWise.values()) + list(r2_VoxelWise.values())

    with pd.ExcelWriter(os.path.join(Cfg['analysisDir'], f"{kiname}.xlsx"), engine='xlsxwriter') as writer:
        ROIbased_df = pd.DataFrame([ROIbased], columns=header)
        ROIbased_df.to_excel(writer, sheet_name='ROIbased', index=False)
        VoxelWise_df = pd.DataFrame([VoxelWise], columns=header)
        VoxelWise_df.to_excel(writer, sheet_name='VoxelWise', index=False)

    # save maps as nifti
    ki_map_img = nib.Nifti1Image(ki_map, pet_img.affine, pet_img.header)
    ki_map_name = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_ki_map_{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised.nii' if denoising else '.nii'}")
    nib.save(ki_map_img, ki_map_name)

    r2_map_img = nib.Nifti1Image(r2_map, pet_img.affine, pet_img.header)
    r2_map_name = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_r2_map_{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised.nii' if denoising else '.nii'}")
    nib.save(r2_map_img, r2_map_name)

## SUMMARY PDF
def patlak_pdf(subject, session, denoising=True, gm_mask=True, **kwargs): 
    """
    **PET Summary Pdf**

    Create summary pdf of FDOPA-PET ki processing.

    **Input:**
    - Outputs from *motion_correction*, *pet_denoising*, *mask_prep*, *patlak_fit*.
     
    **Output:**
    - Summary pdf: *{subject}/ses-{session}/pet/patlak_fit/{subject}_ses-{session}_{thresh}{mask_thresh}_PET_ki.pdf*.
   
    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - denoising (bool, optional): Apply denoising, default=True.
    - gm_mask (bool, optional): Restict striatum atlas to GM mask, default=True.
    - atlas (optional): Atlas for striatum and subregions, default='OGI_3'.
    - weight (optional): Denoising weight, default=100.
    - num_iter (optional): Maximum number of iterations for denoising, default=1000.
    - thr_striatum (optional): Threshold below this number, default=0.4.
    - thrp_striatum (optional): Threshold below this percentage.
    - thr_cerebellum (optional): Threshold below this number, default=0.9.
    - thrp_cerebellum (optional): Threshold below this percentage.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Settings
    ref_frame = kwargs.get('ref_frame', 'last_frame') 
    #ref_frame = (kwargs.get('ref_frame', last_frame)) -1
    weight = kwargs.get('weight', 100)  
    num_iter = kwargs.get('num_iter', 1000)
    atlas = kwargs.get('atlas', 'OGI_3')
    atlas_name = ATLAS_NAMES.get(atlas, 'Unknown atlas')
    thr_striatum = kwargs.get('thr_striatum', None)
    thrp_striatum = kwargs.get('thrp_striatum', None)
    thr_cerebellum = kwargs.get('thr_cerebellum', None)
    thrp_cerebellum = kwargs.get('thrp_cerebellum', None)

    # Set path
    Cfg = pathnames(subject,session, atlas)

    striatum_mask_files = [mask for mask in glob.glob(os.path.join(Cfg['atlas'], '*.nii.gz')) if 'cerebellum' not in mask]
    striatum_masks = [os.path.splitext(os.path.splitext(os.path.basename(mask))[0])[0] for mask in striatum_mask_files]
    num_masks = len(striatum_masks)
    
    thresh_striatum, mask_thresh_striatum = get_thresh(thr_striatum, thrp_striatum, ('thr', 0.4))
    thresh_cerebellum, mask_thresh_cerebellum = get_thresh(thr_cerebellum, thrp_cerebellum, ('thr', 0.9))
    
    thresh_method = get_thresh_method(thresh_striatum)
    cerebellum_thresh_method = get_thresh_method(thresh_cerebellum)

    # Load Ki data
    Ki_file = os.path.join(Cfg['analysisDir'], f"{subject}_ses-{session}_{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised_PET_ki.xlsx' if denoising else '_PET_ki.xlsx'}")
    if os.path.exists(Ki_file):
        ki_table = pd.read_excel(Ki_file)
        ki_indices = list(range(5, 5 + num_masks))
        r2_indices = list(range(5 + num_masks, 5 + 2 * num_masks))
        ki_values = {mask: round(ki_table.iloc[0, idx], 5) for mask, idx in zip(striatum_masks, ki_indices)}
        r2_values = {mask: round(ki_table.iloc[0, idx], 3) for mask, idx in zip(striatum_masks, r2_indices)}
    else:
        raise FileNotFoundError(f"The file {Ki_file} does not exist.")

    # Mean FD
    fd_file = os.path.join(Cfg['motioncorrDir'], f'FD_r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.txt')
    if os.path.exists(fd_file):
        fd_array = np.loadtxt(fd_file)
        mean_fd = round(np.mean(fd_array), 2)
    else:
        raise FileNotFoundError(f"The file {fd_file} does not exist.")

    # Load figures
    tac_fig = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_{atlas}_striatum_and_cerebellum_TACs_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised.png' if denoising else '.png'}")
    patlak_fig = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_{atlas}_Patlak_plot_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised.png' if denoising else '.png'}")
    motion_fig = os.path.join(Cfg['motioncorrDir'], f'r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.png')

    # Load images
    striatum_img = nib.load(os.path.join(Cfg['maskDir'], f"{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}_native_{subject}_ses-{session}_striatum.nii.gz"))
    cerebellum_img = nib.load(os.path.join(Cfg['maskDir'], f'{thresh_cerebellum}{mask_thresh_cerebellum}_native_{subject}_ses-{session}_cerebellum.nii.gz'))
    T1_img = nib.load(os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w.nii.gz'))

    pet_img_path = os.path.join(Cfg['motioncorrDir'], f"mean_r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5{'_denoised.nii.gz' if denoising else '.nii.gz'}")
    pet_img = nib.load(pet_img_path)   

    # Create screenshots 
    pet_striatum = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_striatum_on_pet_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised.png' if denoising else '.png'}")
    plot_roi(striatum_img, cmap='spring', alpha=0.3, bg_img=pet_img, draw_cross=False, annotate=False, output_file=pet_striatum)

    anat_striatum = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_striatum_on_T1_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}.png")
    plot_roi(striatum_img, cmap='spring', alpha=0.3, bg_img=T1_img, output_file=anat_striatum, draw_cross=False, annotate=False)

    anat_cerebellum = os.path.join(Cfg['patlakDir'], f"{subject}_ses-{session}_cerebellum_on_T1_{thresh_cerebellum}{mask_thresh_cerebellum}.png")
    plot_roi(cerebellum_img, cmap='winter', alpha=0.3, bg_img=T1_img, output_file=anat_cerebellum, draw_cross=False, annotate=False)

    # Create PDF
    def add_text(pdf, text, x, y, size=12, style='B'):
        pdf.set_xy(x, y)
        pdf.set_font('Arial', style, size)
        pdf.cell(0, 5, text, 0, 1, 'L')

    class PDF(FPDF):
        def header(self):
            self.set_y(5)
            self.set_font('Arial', '', 10)
            self.set_text_color(110, 110, 110)  # change color (gray)
            self.cell(0, 6, 'F-DOPA PET ki processing (github.com/juliasbrain/PET_ki_proc)', 0, 1, 'C')
            self.line(10, 15, 200, 15)
            self.set_text_color(0, 0, 0)  # reset color

        def footer(self):
            self.set_y(-11)
            self.set_font('Arial', 'I', 8)
            current_date = datetime.now().strftime("%Y-%m-%d")
            self.cell(0, 10, current_date, 0, 0, 'L') # add current date
            page_number = f'Page {self.page_no()}'
            self.cell(0, 10, page_number, 0, 0, 'R') # add page number

        def chapter_body(self, body):
            self.set_font('Arial', '', 8)
            self.multi_cell(0, 4, body)
            self.ln()

    pdf = PDF('P', 'mm', 'A4')

    # PAGE I
    pdf.add_page()

    # HEADING
    pdf.set_font('Arial', '', 4)
    pdf.cell(0, 10, '', 0, 1) 
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, f'F-DOPA PET: {subject} ses-{session}', 0, 1,)

    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, f'Processing summary:', 0, 1,)

    json_file = os.path.join(Cfg['petDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.json') 
    _, _, _, last_frame, _ = extract_frame_info(json_file) # Read frame duration from json file
    ref_frame = (kwargs.get('ref_frame', last_frame)) -1

    ref_frame_text = 'last' if ref_frame == (last_frame - 1) else str(ref_frame)

    text = (
        "Dynamic FDOPA PET images were processed using the PET_ki_proc pipeline (github.com/juliasbrain/PET_ki_proc).\n"
        f"Motion correction was applied to the PET data from time frame 5 min post-FDOPA injection with the {ref_frame_text} frame as reference, using FSL mcflirt. "
        + ( f"The motion-corrected PET data were denoised using Chambolle's total variation denoising algorithm (weight={weight}, num_it={num_iter}), implemented via skimage package in Python.\n" if denoising else "" ) +
        f"Striatum and cerebellum masks from the {atlas_name} were registered to the native PET space utilizing FSL fnirt and flirt. "
        + ( "The striatal mask was additionally restricted to the gray matter mask.\n" if gm_mask else "" ) +
        f"The striatum mask was thresholded using a(n) {thresh_method} threshold of {mask_thresh_striatum}, "
        f"the cerebellum mask was thresholded with a(n) {cerebellum_thresh_method} threshold of {mask_thresh_cerebellum}.\n"
        "FDOPA influx constant (ki) was calculated for the time frame 20 to 60 minutes post-FDOPA injection, encompassing the whole striatum, and its subregions, with the cerebellum as the reference region."
    )
    pdf.chapter_body(text)

#   PATLAK FIT
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, 'Patlak fit:', 0, 1) 

    first_row = striatum_masks[:4]  # First 4 masks
    second_row = striatum_masks[4:]  # Remaining masks (if any)

    pdf.set_font('Arial', '', 9)

    for mask in first_row:
        pdf.cell(0, 5, f'{mask}', 0, 1)
    pdf.set_y(pdf.get_y() - 20)  # Move back to start of annotation section
    pdf.set_x(48)
    for mask in first_row:
        pdf.cell(0, 5, f'ki: {ki_values[mask]}', 0, 6, 'L')

    pdf.set_y(pdf.get_y() - 20)  
    pdf.set_x(70)
    for mask in first_row:
        pdf.cell(0, 5, f'r2: {r2_values[mask]}', 0, 6, 'L')

    if second_row:
        pdf.set_y(pdf.get_y() - 20)
        for mask in second_row:
            pdf.set_x(110)
            pdf.cell(0, 5, f'{mask}', 0, 1)

        pdf.set_y(pdf.get_y() - 20)  
        pdf.set_x(148)
        for mask in second_row:
            pdf.cell(0, 5, f'ki: {ki_values[mask]}', 0, 6, 'L')

        pdf.set_y(pdf.get_y() - 20)  
        pdf.set_x(170)
        for mask in second_row:
            pdf.cell(0, 5, f'r2: {r2_values[mask]}', 0, 6, 'L')

    # Add images
    add_text(pdf, 'TAC:', 10, 107, size=10, style='')
    pdf.image(tac_fig, x=15, y=pdf.get_y(), w=85)
    add_text(pdf, 'Patlak:', 110, 107, size=10, style='')
    pdf.image(patlak_fig, x=105, y=pdf.get_y(), w=85)
    pdf.set_y(pdf.get_y() + 85)  # Adjust y position after adding images

    # MOTION CORR
    pdf.set_font('Arial', 'B', 10)
    pdf.set_y(pdf.get_y() - 20)
    pdf.cell(0, 6, 'Motion parameters:', 0, 1) 
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 6, f'mean frame-wise displacement: {mean_fd}', 0,4, 'L')
    pdf.image(motion_fig, x=25, y=pdf.get_y(), w=155)

    # PAGE II
    pdf.add_page()

    # MASK FIT
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, '', 0, 1)  # Add some space
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, 'Mask fit:', 0, 1) 

    add_text(pdf, 'striatum mask on mean PET image:', 10, 28, size=10, style='') 
    pdf.image(pet_striatum, x=15, y=pdf.get_y(), w=120)
    add_text(pdf, 'striatum mask on T1w image:', 10, 102, size=10, style='') 
    pdf.image(anat_striatum, x=15, y=pdf.get_y(), w=120)
    add_text(pdf, 'cerebellum mask on T1w image:', 10, 176, size=10, style='') 
    pdf.image(anat_cerebellum, x=15, y=pdf.get_y(), w=120)

    # SAVE PDF
    pdf_name = os.path.join(Cfg['analysisDir'],f"{subject}_ses-{session}_{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised_PET_ki.pdf' if denoising else '_PET_ki.pdf'}")
    pdf.output(pdf_name, 'F')
