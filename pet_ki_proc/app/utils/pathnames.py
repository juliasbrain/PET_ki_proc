## SET PATHNAMES
def pathnames(subject, session, atlas='OGI_3'):
    """
    Set pathnames for PET ki processing.

    **Args:** 
    - subject: Subject ID, e.g., 'sub-001'.
    - session: Session ID, e.g., '01'.
    - atlas: Name of the atlas folder under 'app/data/atlas/' (default 'OGI_3').

    Julia Schulz, julia.a.schulz@tum.de, github.com/juliasbrain/PET_ki_proc, 04.06.2025
    """

    import os
    Cfg = {}

    # Base directory
    file_dir = os.path.dirname(os.path.abspath(__file__))
    Cfg['appDir'] = os.path.dirname(file_dir)

    # Atlas path
    Cfg['atlas'] = os.path.join(Cfg['appDir'], 'data', 'atlas', atlas)

    # Preproc locations
    Cfg['preprocDir'] = '/app/preprocData'
    Cfg['petDir'] = os.path.join(Cfg['preprocDir'], subject, f'ses-{session}', 'pet')
    Cfg['analysisDir'] = os.path.join(Cfg['petDir'], 'patlak_fit')
    Cfg['motioncorrDir'] = os.path.join(Cfg['analysisDir'], 'motion_corr')
    Cfg['anatsourceDir'] = os.path.join(Cfg['preprocDir'], subject, f'ses-{session}', 'anat')
    Cfg['anatDir'] = os.path.join(Cfg['analysisDir'], 'T1w')
    Cfg['maskDir'] = os.path.join(Cfg['analysisDir'], 'masks', atlas)
    Cfg['patlakDir'] = os.path.join(Cfg['analysisDir'], 'patlak')

    # FSL directory
    Cfg['fslDir'] = os.getenv('FSLDIR')

    return Cfg
