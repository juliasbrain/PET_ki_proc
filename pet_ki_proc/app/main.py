from pathlib import Path
from pet_ki_proc.app.pipeline.steps import run_denoising, run_mask_preparation, run_motion_correction, run_patlak
from pet_ki_proc.app.utils.helpers import file_exists, get_thresh, update_set_pathnames, zip_nii
from pet_ki_proc.app.utils.pathnames import pathnames

def PET_ki_proc(subject=None, session=None, denoising=True, gm_mask=True, subject_path=None, **kwargs):
    """
    FDOPA-PET ki processing pipeline.

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    # --- 0. Set path to subject directory ---
    if subject_path:
        update_set_pathnames(subject_path)
        print(f"Set subject path to: {subject_path}")
        #return
    
    # --- 1. Set pathnames ---
    if not subject or not session:
        raise ValueError("subject and session are required.")

    atlas = kwargs.get("atlas", "OGI_3")
    Cfg = pathnames(subject, session, atlas)
    print(f"[INFO] Using subject path: {Cfg['preprocDir']}")

    # --- 2. Configurations ---
    thr_striatum = kwargs.get('thr_striatum')
    thrp_striatum = kwargs.get('thrp_striatum')
    thresh_striatum, mask_thresh_striatum = get_thresh(thr_striatum, thrp_striatum, ('thr', 0.4))
    atlas = kwargs.get('atlas', 'OGI_3')
    pdf_name = f"{atlas}_{thresh_striatum}{mask_thresh_striatum}{'_gm' if gm_mask else ''}{'_denoised_PET_ki.pdf' if denoising else '_PET_ki.pdf'}"

    # --- 3. Check if subject is already processed ---
    if file_exists(Cfg['analysisDir'], pdf_name):
        print(f"[SKIP] PET ki processing already done for {subject} ses-{session}!")
        return
    
    # --- 4. Preparations ---
    zip_nii(Cfg['petDir'], 'acdyn_pet') 
    Path(Cfg['analysisDir']).mkdir(parents=True, exist_ok=True)

    if not file_exists(Cfg['petDir'], 'acdyn_pet.nii.gz'):
        print(f"[ERROR] No PET data for {subject} ses-{session}")
        return

    print(f"[START] PET ki processing for {subject} ses-{session}")

    # --- 5. Motion Correction ---
    run_motion_correction(subject, session, Cfg, **kwargs)
        
    # --- 6. Denoising ---
    run_denoising(subject, session, Cfg, denoising, **kwargs)

    # --- 7. Mask Preparation ---
    run_mask_preparation(subject, session, Cfg, denoising, gm_mask, thresh_striatum, mask_thresh_striatum, **kwargs)

    # --- 8. Patlak Fit ---
    run_patlak(subject, session, Cfg, denoising, gm_mask, thresh_striatum, mask_thresh_striatum, **kwargs)

    print(f"[DONE] PET Ki processing complete for {subject} ses-{session}")
    return True