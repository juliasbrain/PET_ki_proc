import os
import nibabel as nib
import numpy as np
import skimage.restoration as restoration
from fsl.wrappers.fslmaths import fslmaths

from pet_ki_proc.app.utils.pathnames import pathnames

## DENOISING
def pet_denoising(subject, session, **kwargs): 
    """
    **PET denoising**

    Denoising of motion corrected dynamic PET images using Chambolles algorithm of total variation denoising from scikit-image.
    
    **Input:**
    - motion corrected PET image for frames from 4-70 min (output from *motion_correction*)

    **Output:**
    - denoised pet image: *{subject}/ses-{session}/pet/patlak_fit/motion_corr/r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5_denoised.nii.gz*.
   
    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - weight (optional): Denoising weight, default=100.
    - num_iter (optional): Maximum number of iterations for denoising, default=1000.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # Set path
    Cfg = pathnames(subject,session)
    
    # Get denoising parameters
    weight = kwargs.get('weight', 100) 
    num_iter = kwargs.get('num_iter', 1000) 
    
    # Load PET image
    pet_img = nib.load(os.path.join(Cfg['motioncorrDir'], f'r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5.nii.gz') )
    pet_data = pet_img.get_fdata()
    
    # Denoise PET data
    pet_denoised = np.zeros_like(pet_data)
    for frame in range(pet_data.shape[-1]):
        pet_denoised[..., frame] = restoration.denoise_tv_chambolle(
            pet_data[..., frame], weight=weight, max_num_iter=num_iter, channel_axis=None)

    # Save denoised image
    denoised_img_path = os.path.join(Cfg['motioncorrDir'], f'r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5_denoised.nii.gz')
    pet_denoised_img = nib.Nifti1Image(pet_denoised, pet_img.affine, pet_img.header)
    nib.save(pet_denoised_img, denoised_img_path)

    # Compute and save mean image
    mean_img_path = os.path.join(Cfg['motioncorrDir'], f'mean_r_{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet_from5_denoised.nii.gz')
    fslmaths(pet_denoised_img).Tmean().run(output=mean_img_path)