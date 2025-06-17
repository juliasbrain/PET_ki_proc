import os
import nibabel as nib
import numpy as np
from fsl.wrappers import fslmaths

def threshold_and_binarize(mask_native, thresh_type, thresh_value, output_path, mask_name):
    """
    Apply absolute or percentile thresholding to a NIfTI mask and save the binary mask.

    Parameters:
        mask_native (str): Path to native space mask.
        thresh_type (str): 'thr' for absolute or 'thrp' for percentile thresholding.
        thresh_value (float): Threshold value.
        output_path (str): Directory where output mask will be saved.
        mask_name (str): Name for the output file.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """
    
    if thresh_type == 'thr':
        out_file = os.path.join(output_path, f'thr{thresh_value}_{mask_name}.nii.gz')
        fslmaths(mask_native).thr(thresh_value).bin().run(out_file)

    elif thresh_type == 'thrp':
        mask_img = nib.load(mask_native)
        data = mask_img.get_fdata()

        non_zero_indices = np.nonzero(data)
        non_zero_values = data[non_zero_indices]
        sorted_indices = np.argsort(non_zero_values)
        sorted_values = non_zero_values[sorted_indices]
        sorted_non_zero_indices = tuple(idx[sorted_indices] for idx in non_zero_indices)
        num_voxels = len(sorted_values)
        num_to_remove = int(thresh_value * num_voxels)

        remaining_indices = (sorted_non_zero_indices[0][num_to_remove:], 
                             sorted_non_zero_indices[1][num_to_remove:], 
                             sorted_non_zero_indices[2][num_to_remove:])
        remaining_values = sorted_values[num_to_remove:]

        mask_thrp = np.zeros(data.shape)
        for i in range(len(remaining_values)):
            mask_thrp[remaining_indices[0][i], remaining_indices[1][i], remaining_indices[2][i]] = remaining_values[i]
        mask_thrp[mask_thrp > 0] = 1

        out_file = os.path.join(output_path, f'thrp{thresh_value}_bin_{mask_name}.nii.gz')
        mask_thrp_img = nib.Nifti1Image(mask_thrp, mask_img.affine, mask_img.header)
        nib.save(mask_thrp_img, out_file)

    else:
        raise ValueError("Invalid threshold type. Use 'thr' or 'thrp'.")