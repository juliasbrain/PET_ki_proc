import os
import subprocess
import argparse
from pet_ki_proc.app.utils.pathnames import pathnames

def loop_through_subs(session, preprocDir, **kwargs):
    subject_list_path = os.path.join(preprocDir, 'subjectlist.txt')
        
    with open(subject_list_path, 'r') as file:
        subjects = file.read().splitlines()

    # Build CLI argument string from kwargs
    kwargs_str = ' '.join([f'-{key} {value}' for key, value in kwargs.items() if value is not None])

    for subject in subjects:
        try:
            run_pet_ki_proc_command = (
                f'python run_PET_ki_proc.py -subject {subject} -session {session} {kwargs_str}'
            )
            subprocess.run(run_pet_ki_proc_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] processing {subject} ses-{session}: {e}")
        except Exception as e:
                print(f"[ERROR] for {subject} ses-{session}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run PET-ki-proc for all subjects.')
        
    parser.add_argument('session', type=str, metavar='', help='Session ID (e.g., 01)')
    parser.add_argument('-denoising', choices=['True', 'False'], default='True', help='Apply denoising.')
    parser.add_argument('-ref_frame', type=int, default=30, metavar='', help='Reference frame for motion-correction.')
    parser.add_argument('-weight', type=int, default=100, metavar='', help='Denoising weight.')
    parser.add_argument('-num_iter', type=int, default=1000, metavar='', help='Max. number of iterations for denoising.')
    parser.add_argument('-atlas', type=str, default='OGI_3', metavar='', help='Atlas for striatum and subregions.')
    parser.add_argument('-gm_mask', choices=['True', 'False'], default='True', help='Apply GM masking.')
    parser.add_argument('-thr_striatum', type=float, metavar='', help='Threshold below this number for striatum.')
    parser.add_argument('-thrp_striatum', type=float, metavar='', help='Threshold below this percentage for striatum.')
    parser.add_argument('-thr_cerebellum', type=float, metavar='', help='Threshold below this number for cerebellum.')
    parser.add_argument('-thrp_cerebellum', type=float, metavar='', help='Threshold below this percentage for cerebellum.')

    args = parser.parse_args()
    session = args.session

    # Get path
    Cfg = pathnames('sub-001', session)  # Dummy subject to get base path
    subject_list_path = os.path.join(Cfg['preprocDir'], 'subjectlist.txt')

    # Create subjectlist.txt in preprocDir
    with open(subject_list_path, 'w') as file:
        for sub in os.listdir(Cfg['preprocDir']):
            if os.path.isdir(os.path.join(Cfg['preprocDir'], sub)):
                file.write(f"{sub}\n")

    # Build kwargs dictionary
    kwargs = {
        'denoising': args.denoising,
        'ref_frame': args.ref_frame,
        'weight': args.weight,
        'num_iter': args.num_iter,
        'atlas': args.atlas,
        'gm_mask':args.gm_mask,
        'thr_striatum': args.thr_striatum,
        'thrp_striatum': args.thrp_striatum,
        'thr_cerebellum': args.thr_cerebellum,
        'thrp_cerebellum': args.thrp_cerebellum
    }

    # Start processing
    loop_through_subs(session, Cfg['preprocDir'], **kwargs)

if __name__ == '__main__':
     main()
