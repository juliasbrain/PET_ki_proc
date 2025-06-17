import glob
import os
import numpy as np
from fsl.data.image import Image
from fsl.utils.image.roi import roi
from fsl.wrappers.avwutils import fslmerge, fslsplit
from fsl.wrappers.flirt import mcflirt
from fsl.wrappers.fslmaths import fslmaths
from fsl.wrappers.fslstats import fslstats
import matplotlib.pylab as plt
    
from pet_ki_proc.app.utils.helpers import extract_frame_info    
from pet_ki_proc.app.utils.pathnames import pathnames

## MOTION CORRECTION
def motion_correction(subject, session, **kwargs):
    """
    **PET motion correction**

    Motion correction of dynamic FDOPA-PET data using FSL mcflirt.

    **Input:** 
    - Decay corrected dynamic FDOPA-PET image: *{subject}/ses-{session}/pet/{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.nii*.

    **Output:**
    - in <{subject}/ses-{session}/pet/patlak_fit/motion_corr/
    - decay corrected dynamic PET image for frames from 5-70 min: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii.gz*.
    - motion corrected PET image for frames from 4-70 min: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii.gz*.
    - mean of motion corrected PET data: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/mean_r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii.gz*.
    - motion correction parameter: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.par*.

    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - ref_frame (optional): Reference frame for motion correction, default=30.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Set path
    Cfg = pathnames(subject,session)
    os.makedirs(Cfg['motioncorrDir'], exist_ok=True)

    # Settings
    json_file = os.path.join(Cfg['petDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.json') 
    frame_at_5min, start_frame_idx, _, last_frame, _ = extract_frame_info(json_file) # Read frame duration from json file
    ref_frame = (kwargs.get('ref_frame', last_frame)) -1
    refvol = ref_frame - start_frame_idx
    
    # Load PET image
    pet_img = Image(os.path.join(Cfg['petDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.nii.gz'))

    # Split PET image in single frames
    split_path = os.path.join(Cfg['motioncorrDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet')
    fslsplit(pet_img, out=split_path, dim='t')

    frames = [os.path.join(Cfg['motioncorrDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet{frame_num:04}.nii.gz')
        for frame_num in range(frame_at_5min, last_frame)]
    
    # crop neck
    last_frame_img = os.path.join(Cfg['motioncorrDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet00{last_frame - 1}.nii.gz')
    center = fslstats(last_frame_img).C.run()
    z_coord = center[2]
    lower_z = int(z_coord - 30)
    for frame in frames:
        frame_img = Image(frame)
        frame_name = os.path.basename(frame)
        output_path = os.path.join(Cfg['motioncorrDir'], f'c_{frame_name}')
        cropped = roi(frame_img, [(0, 344), (0, 344), (lower_z, 127)])
        Image.save(cropped, output_path)

    # Merge frames starting at 5 minutes
    cropped_frames = [os.path.join(Cfg['motioncorrDir'], f'c_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet{frame_num:04}.nii.gz')
        for frame_num in range(frame_at_5min, last_frame)]
    fslmerge('t', (os.path.join(Cfg['motioncorrDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii')), *cropped_frames)

    # Clean up split files
    os.system(f'rm -rf {split_path}00*')
    for file_path in cropped_frames:
        os.remove(file_path)
    
    # Motion correction with FSL mcflirt
    input_file = os.path.join(Cfg['motioncorrDir'], f'{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii.gz')
    output_file = os.path.join(Cfg['motioncorrDir'], f'r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5')
    mcflirt(input_file, out=output_file, cost='mutualinfo', refvol=refvol, dof='6', plots=True)
    # INFO: -dof 6: rigid body, -cost: mutualinfo=any modalities (including PET)
    
    # Compute and save mean image
    mean_img_path = os.path.join(Cfg['motioncorrDir'], f'mean_r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii.gz')
    fslmaths(output_file).Tmean().run(output=mean_img_path)

## MOTION PARAMETERS
def motion_params(subject, session):
    """
    **PET motion parameters**

    Create graphs of framewise-displacement after motion correction.
    
    **Input:**
    - motion correction parameter (output from *motion_correction*)

    **Output:**
    - framewise-displacement values: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/FD_r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.txt*
    - motion parameter plots: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.png*

    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Set path
    Cfg = pathnames(subject,session)

#    Load the .par file
    par_file = glob.glob(os.path.join(Cfg['motioncorrDir'], '*.par'))[0]
    motion_params = np.loadtxt(par_file)

    # Calculate framewise displacement
    diff = np.diff(motion_params, axis=0)
    translations = diff[:, 3:]
    rotations = diff[:, :3] * 50  # Convert rotations from radians to mm (standard brain radius: 50mm)
    fd = np.sum(np.abs(translations) + np.abs(rotations), axis=1)  # Framewise displacement
    fd = np.append(fd, 0)  # Pad last frame with 0

    # Save framewise displacement
    basename = os.path.basename(par_file).removesuffix('.par')
    np.savetxt(os.path.join(Cfg['motioncorrDir'], f'FD_{basename}.txt'), fd, comments='', fmt='%.6f')

    # Create plots
    timepoints = motion_params.shape[0]
    x = np.arange(timepoints)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Motion parameters
    ax[0].plot(x, motion_params[:, :3], label=['Pitch (X)', 'Yaw (Y)', 'Roll (Z)'])
    ax[0].plot(x, motion_params[:, 3:], linestyle='--', label=['X Translation', 'Y Translation', 'Z Translation'])
    ax[0].set_title('Motion Parameters')
    ax[0].set_ylabel('Radians / mm')
    ax[0].legend(loc='upper right')
    
    # Framewise displacement
    ax[1].plot(x, fd, color='red', label='Frame-Wise Displacement')
    ax[1].set_title('Frame-Wise Displacement')
    ax[1].set_xlabel('Time Points')
    ax[1].set_ylabel('Displacement (mm)')
    ax[1].legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(os.path.join(Cfg['motioncorrDir'], f'{basename}.png'))