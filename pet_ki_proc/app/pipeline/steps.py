from pet_ki_proc.app.services.denoising import *
from pet_ki_proc.app.services.mask_prep import *
from pet_ki_proc.app.services.motion_corr import *
from pet_ki_proc.app.services.patlak_fit import *
from pet_ki_proc.app.utils.helpers import file_exists, zip_nii

# Run motion correction steps
def run_motion_correction(subject, session, Cfg, **kwargs):
    if file_exists(Cfg['motioncorrDir'], '.png'):
        print("      ✔ Motion correction already done")
        return
    print("      ➤ Starting motion correction")
    motion_correction(subject, session, **kwargs)
    motion_params(subject, session)
    return

# Run denoising steps
def run_denoising(subject, session, Cfg, denoising, **kwargs):
    if not denoising:
        print("      ⚠ Denoising skipped")
        return

    if file_exists(Cfg['motioncorrDir'], 'denoised.nii.gz'):
        print("      ✔ Denoising already done")
    else:
        print("      ➤ Starting denoising")
        pet_denoising(subject, session, **kwargs)
    return

# Run mask preparation steps
def run_mask_preparation(subject, session, Cfg, denoising, gm_mask, thresh_striatum, mask_thresh_striatum, **kwargs):
    if not file_exists(Cfg['anatDir'], 'inwarpcoef.nii.gz'):
        print("      ➤ Starting T1 registration")
        zip_nii(Cfg['anatsourceDir'], 'T1w')
        T1w_registration(subject, session, denoising, **kwargs)
    else:
        print("      ✔ T1 registration already done")
    
    if not file_exists(Cfg['maskDir'], '.nii.gz'):
        print("      ➤ Starting mask registration")
        mask_registration(subject, session, **kwargs)
    else:
        print("      ✔ Mask registration already done")

    if gm_mask:
        if not file_exists(Cfg['maskDir'], 'gm_'):
            print("      ➤ Starting GM masking")
            mask_restrict_gm(subject, session, **kwargs)
        else:
            print("      ✔ GM masking already done")
    else:
        print("      ⚠ GM masking skipped")

    mask_name = f"{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else '_'}"
    if not file_exists(Cfg['maskDir'], mask_name):
        print("      ➤ Starting mask thresholding")
        mask_thresh_bin(subject, session, gm_mask, **kwargs)
    else:
        print("      ✔ Mask thresholding already done")
    return

# Run Patlak fit steps
def run_patlak(subject, session, Cfg, denoising, gm_mask, thresh_striatum, mask_thresh_striatum, **kwargs):
    atlas = kwargs.get('atlas', 'OGI_3')
    patlak_name = f"{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised.nii' if denoising else '.nii'}"
    
    if not file_exists(Cfg['patlakDir'], patlak_name):
        print("      ➤ Starting Patlak fitting")
        patlak_fit(subject, session, denoising, gm_mask, **kwargs)
    else:
        print("      ✔ Patlak fitting already done")

    print("      ➤ Generating PET-ki-proc PDF")
    patlak_pdf(subject, session, denoising, gm_mask, **kwargs)
    return
