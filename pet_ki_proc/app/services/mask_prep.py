import glob
import os
import nibabel as nib
from fsl.data.image import Image
from fsl.wrappers.fast import fast
from fsl.wrappers.flirt import flirt
from fsl.wrappers.fnirt import applywarp, invwarp, fnirt
from fsl.wrappers.fslmaths import fslmaths
from fsl.wrappers.misc import fslreorient2std

from pet_ki_proc.app.services.thresholding import threshold_and_binarize
from pet_ki_proc.app.utils.helpers import extract_frame_info, get_thresh
from pet_ki_proc.app.utils.pathnames import pathnames

## T1W TO PET and MNI-152 REGISTRATION
def T1w_registration(subject, session, denoising=True, **kwargs):
    """
    **T1w registration**

    Registration of T1w image into native PET and MNI-152 space using fsl fnirt and flirt. 

    **Input:**
    - motion corrected PET image for frames (output from *motion_correction*).
    - T1w image of subject *{subject}/ses-{session}/anat/{subject}_ses-{session}_T1w.nii*.

    **Output:**
    - T1w in native PET space: *{subject}/ses-{session}/pet/patlak_fit/T1w/r2PET_C003_ses-01_T1w.nii.gz*.

    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - denoising (bool, optional): Apply denoising, default=True.
    - ref_frame (optional): Reference frame for motion correction, default=last-frame.
    
    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Settings
    Cfg = pathnames(subject,session)
    os.makedirs(Cfg['anatDir'], exist_ok=True)

    json_file = os.path.join(Cfg['petDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.json') 
    _, start_frame_idx, _, last_frame, _ = extract_frame_info(json_file) # Read frame duration from json file
    ref_frame = (kwargs.get('ref_frame', last_frame)) -1
    refvol = ref_frame - start_frame_idx

    # Load data
    T1w_img_raw = Image(os.path.join(Cfg['anatsourceDir'], f'{subject}_ses-{session}_T1w.nii.gz'))
    pet_img_path = os.path.join(Cfg['motioncorrDir'], f"r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5" + ('_denoised.nii.gz' if denoising else '.nii.gz')) 
    pet_img = nib.load(pet_img_path)
    pet_data = pet_img.get_fdata()

    # Get the reference frame of PET data
    pet_frame = pet_data[..., refvol]
    pet_frame_img = nib.Nifti1Image(pet_frame, pet_img.affine, pet_img.header)

    # Register native T1w to MNI-152 standard space
    T1w_img = os.path.join(Cfg['anatsourceDir'], f'{subject}_ses-{session}_2std_T1w.nii.gz')
    fslreorient2std(T1w_img_raw, T1w_img)

    # Register native T1w to native PET space
    T1w_pet_img = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w.nii.gz')
    T1w_pet_omat = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w.mat')
    flirt(T1w_img, ref=pet_frame_img, out=T1w_pet_img, omat=T1w_pet_omat, cost='corratio', interp='trilinear')

    # Register T1w-in-PET-Space to MNI-152 standard space
    # Flirt affine from structural to standard mni-152 space
    mni = os.path.join(Cfg['fslDir'], 'data/standard/MNI152_T1_2mm.nii.gz')
    T1w_pet_mni_img = os.path.join(Cfg['anatDir'], f'flirt_r2PET2mni_{subject}_ses-{session}_T1w.nii.gz')
    T1w_pet_mni_omat = os.path.join(Cfg['anatDir'], f'flirt_r2PET2mni_{subject}_ses-{session}_T1w.mat')
    flirt(T1w_pet_img, ref=mni, out=T1w_pet_mni_img , omat=T1w_pet_mni_omat , dof=12, interp='trilinear')

    # Fnirt with affine matrix from flirt
    config = os.path.join(Cfg['fslDir'], 'etc/flirtsch/T1_2_MNI152_2mm.cnf')
    T1w_pet_mni_fnirt_img = os.path.join(Cfg['anatDir'], f'fnirt_r2PET2mni_{subject}_ses-{session}_T1w.nii.gz')
    T1w_pet_mni_fnirt_warp = os.path.join(Cfg['anatDir'], f'r2PET2mni_{subject}_ses-{session}_T1w_warpcoef.nii.gz')
    fnirt(T1w_pet_img, ref=mni, aff=T1w_pet_mni_omat, config=config, iout=T1w_pet_mni_fnirt_img, cout=T1w_pet_mni_fnirt_warp)

    # Inverse warp from fnirt
    T1w_pet_mni_fnirt_inwarp = os.path.join(Cfg['anatDir'], f'r2PET2mni_{subject}_ses-{session}_T1w_inwarpcoef.nii.gz')
    invwarp(T1w_pet_mni_fnirt_warp, ref=T1w_pet_img, out=T1w_pet_mni_fnirt_inwarp)

## MASK TO PET SPACE REGISTRATION
def mask_registration(subject, session, **kwargs):
    """
    **Mask registration**

    Registration of striatum and cerebellum masks into native PET space using fsl fnirt. 

    **Input:**
    - T1w in native PET space: *{subject}/ses-{session}/pet/patlak_fit/T1w/r2PET_C003_ses-01_T1w.nii.gz*. (output from *T1w_registration*)
    - striatum and cerebellum masks in MNI space (default under: *PET/PET_ki_proc/masks/*). 

    **Output:**
    - masks in native PET space: *{subject}/ses-{session}/pet/patlak_fit/masks/native_{subject}_ses-{session}_{mask_name}.nii.gz*.

    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - atlas (optional): Atlas for striatum and subregions, default='OGI_3'.
    
    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Settings
    atlas = kwargs.get('atlas', 'OGI_3')
    Cfg = pathnames(subject,session, atlas)
    os.makedirs(os.path.join(Cfg['maskDir']), exist_ok=True)

    mask_files = glob.glob(os.path.join(Cfg['atlas'], '*.nii.gz'))
    T1w_pet_img = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w.nii.gz')
    T1w_pet_mni_fnirt_inwarp = os.path.join(Cfg['anatDir'], f'r2PET2mni_{subject}_ses-{session}_T1w_inwarpcoef.nii.gz')
    
    for mask_file in mask_files:
        mask_name = os.path.basename(mask_file).replace('.nii.gz', '')
        mask_native = os.path.join(Cfg['maskDir'], f'native_{subject}_ses-{session}_{mask_name}.nii.gz')
        applywarp(mask_file, ref=T1w_pet_img, out=mask_native, w=T1w_pet_mni_fnirt_inwarp)

## RESTRICT MASKS TO GM
def mask_restrict_gm(subject, session, **kwargs):
    """
    **Mask restriction to GM**

    Restriction of striatum atlas to native GM mask using fsl fast. 

    **Input:**
    - T1w in native PET space: *{subject}/ses-{session}/pet/patlak_fit/T1w/r2PET_C003_ses-01_T1w.nii.gz* (output from *T1w_registration*).
    - striatum masks in native space  (output from *mask_registration*).

    **Output:**
    - masks in native PET space restricted to GM: *{subject}/ses-{session}/pet/patlak_fit/masks/gm_native_{subject}_ses-{session}_{mask_name}.nii.gz*.

    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - gm_mask (bool, optional): Restict striatum atlas to GM mask, default=True.
    
    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Settings
    atlas = kwargs.get('atlas', 'OGI_3')
    Cfg = pathnames(subject, session, atlas)

    # Segment T1w-in-PET-Space
    T1w_pet_img = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w.nii.gz')
    T1w_pet_seg = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w')
    fast(T1w_pet_img, 'g', out=T1w_pet_seg)

    # Threshold GM mask
    T1w_pet_gm = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w_pve_2.nii.gz')
    T1w_pet_gm_thr = os.path.join(Cfg['anatDir'], f'r2PET_{subject}_ses-{session}_T1w_pve_2_thr.nii.gz')
    fslmaths(T1w_pet_gm).thr('0.9').bin().run(T1w_pet_gm_thr)

    # Multiply the GM mask with striatum masks:
    striatum_mask_files = [mask for mask in glob.glob(os.path.join(Cfg['atlas'], '*.nii.gz')) if 'cerebellum' not in mask]
    striatum_masks = [os.path.splitext(os.path.splitext(os.path.basename(mask))[0])[0] for mask in striatum_mask_files]

    for mask_name in striatum_masks:
        mask_native = os.path.join(Cfg['maskDir'], f'native_{subject}_ses-{session}_{mask_name}.nii.gz')
        mask_gm_native = os.path.join(Cfg['maskDir'], f'gm_native_{subject}_ses-{session}_{mask_name}.nii.gz')
        fslmaths(mask_native).mul(T1w_pet_gm_thr).run(mask_gm_native)

## MASK THRESHOLDING AND BINARIZATION
def mask_thresh_bin(subject, session, gm_mask, **kwargs): 
    """
    **Mask threholsing and binarization**
    
    Thresholding and binarization of striatum and cerebellum masks in native PET space. 

    **Input:**
    - Masks in native PET space (output from *mask_registration*)

    **Output:**
    - Binarized and thresholded masks in native PET space: *{subject}/ses-{session}/pet/patlak_fit/masks/{threshold}_bin_native_{subject}_ses-{session}_{mask_name}.nii.gz*.
    
    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - gm_mask (bool, optional): Restict striatum atlas to GM mask, default=True.
    - atlas (optional): Atlas for striatum and subregions, default='OGI_3'.
    - thr_striatum (optional): Threshold below this number, default=0.4.
    - thrp_striatum (optional): Threshold below this percentage.
    - thr_cerebellum (optional): Threshold below this number, default=0.9.
    - thrp_cerebellum (optional): Threshold below this percentage.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Settings
    atlas = kwargs.get('atlas', 'OGI_3')
    Cfg = pathnames(subject,session,atlas)

    thr_striatum = kwargs.get('thr_striatum', None)
    thrp_striatum = kwargs.get('thrp_striatum', None)
    thr_cerebellum = kwargs.get('thr_cerebellum', None)
    thrp_cerebellum = kwargs.get('thrp_cerebellum', None)

    striatum_mask_files = [mask for mask in glob.glob(os.path.join(Cfg['atlas'], '*.nii.gz')) if 'cerebellum' not in mask]
    striatum_masks = [os.path.splitext(os.path.splitext(os.path.basename(mask))[0])[0] for mask in striatum_mask_files]

    # Get thresholds
    thresh_striatum, mask_thresh_striatum = get_thresh(thr_striatum, thrp_striatum, ('thr', 0.4))
    thresh_cerebellum, mask_thresh_cerebellum = get_thresh(thr_cerebellum, thrp_cerebellum, ('thr', 0.9))

    # Threshold striatum masks
    for mask_name in striatum_masks:
        mask_native = os.path.join(Cfg['maskDir'], f'native_{subject}_ses-{session}_{mask_name}.nii.gz')
        if gm_mask:
            mask_native = os.path.join(Cfg['maskDir'], f'gm_native_{subject}_ses-{session}_{mask_name}.nii.gz')
            threshold_and_binarize(mask_native, thresh_striatum, mask_thresh_striatum, Cfg['maskDir'], f'gm_native_{subject}_ses-{session}_{mask_name}')
        else:
            mask_native = os.path.join(Cfg['maskDir'], f'native_{subject}_ses-{session}_{mask_name}.nii.gz')
            threshold_and_binarize(mask_native, thresh_striatum, mask_thresh_striatum, Cfg['maskDir'], f'native_{subject}_ses-{session}_{mask_name}')

    # Threshold cerebellum mask
    cerebellum_native = os.path.join(Cfg['maskDir'], f'native_{subject}_ses-{session}_cerebellum.nii.gz')
    threshold_and_binarize(cerebellum_native, thresh_cerebellum, mask_thresh_cerebellum, Cfg['maskDir'], f'native_{subject}_ses-{session}_cerebellum')