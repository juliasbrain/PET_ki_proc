import argparse
from pet_ki_proc.app.main import PET_ki_proc

def main():
    parser = argparse.ArgumentParser(
        description="""
        🧠 Run PET-ki-proc processing pipeline for F-DOPA PET data analysis and ki estimation.

        Input: 
        - in BIDS format.
        - Decay corrected dynamic FDOPA-PET image: *{subject}/ses-{session}/pet/{subject}_ses-{session}_trc-18FDOPA_rec-acdyn_pet.nii*.
        - T1w image of subject *{subject}/ses-{session}/anat/{subject}_ses-{session}_T1w.nii*.

        Output:
        - in <{subject}/ses-{session}/pet/patlak_fit/
        - Results sheet: patlak/{subject}_ses-{session}_OGI_3_thr0.6_denoised_PET_ki.xslx *. 
        - Summary PDF: patlak/{subject}_ses-{session}_OGI_3_thr0.6_denoised_PET_ki.pdf *. 
        - ki map: patlak/{subject}_ses-{session}_ki_map_thr0.4.nii*.
        - r2 map: patlak/{subject}_ses-{session}_r2_map_thr0.4.nii*.

        This application performs motion correction, optional denoising, anatomical mask registration,
        thresholding, and Patlak model fitting to calculate FDOPA influx constant (Ki).
        """,

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=
        """
        Example usage:
        run_PET_ki_proc.py -subject_path=/path/to/preprocData

        run_PET_ki_proc.py -subject sub-001 -session 01 -denoising=True -gm_mask=True -ref_frame=30

        Available Atlases:
        - OGI_3: Oxford-GSK-Imanova Striatal Connectivity Atlas (3 subregions)
        - OGI_7: OGI with 7 subregions
        - HO: Harvard-Oxford striatal atlas
        - CIT: CIT168 subcortical atlas

        📎 GitHub: https://github.com/juliasbrain/PET_ki_proc
        """
    )

    parser.add_argument('-subject', type=str, metavar='', help='Subject ID (e.g., sub-001).')
    parser.add_argument('-session', type=str, metavar='', help='Session ID (e.g., 01).')
    parser.add_argument('-subject_path', type=str, metavar='', help='Set Subject Path.')
    parser.add_argument('-denoising', choices=['True', 'False'], default='True', help='Apply denoising.')
    parser.add_argument('-weight', type=int, default=100, metavar='', help='Denoising weight.')
    parser.add_argument('-num_iter', type=int, default=1000, metavar='', help='Max iterations for denoising.')
    parser.add_argument('-ref_frame', type=int, default=30, metavar='', help='Reference frame for motion correction.')
    parser.add_argument('-atlas', type=str, default='OGI_3', help='Atlas used for masks.')
    parser.add_argument('-gm_mask', choices=['True', 'False'], default='True', help='Restict striatum atlas to GM mask.')
    parser.add_argument('-thr_striatum', type=float, metavar='', help='Threshold for striatum mask.')
    parser.add_argument('-thrp_striatum', type=float, metavar='', help='Percentage threshold for striatum mask.')
    parser.add_argument('-thr_cerebellum', type=float, metavar='', help='Threshold for cerebellum mask.')
    parser.add_argument('-thrp_cerebellum', type=float, metavar='', help='Percentage threshold for cerebellum mask.')

    args = parser.parse_args()
    kwargs = vars(args)

    subject = kwargs.pop('subject', None)
    session = kwargs.pop('session', None)
    subject_path = kwargs.pop('subject_path', None)
    denoising = kwargs.pop('denoising', 'True') == 'True'
    gm_mask = kwargs.pop('gm_mask', 'True') == 'True'

    PET_ki_proc(subject=subject, session=session, denoising=denoising, gm_mask=gm_mask, subject_path=subject_path, **kwargs)

if __name__ == '__main__':
    main()
