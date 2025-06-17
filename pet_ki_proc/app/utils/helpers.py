
import gzip
import json
import os
import shutil
import numpy as np
from pathlib import Path

def extract_frame_info(json_file, threshold_sec=300):
    with open(json_file, "r") as file:
        data = json.load(file)

    frame_start = np.array(data["FrameTimesStart"])  # in seconds
    frame_duration = np.array(data["FrameDuration"]) / 60  # convert to minutes
    total_scan_duration = np.sum(frame_duration)
    frame_onsets = np.cumsum(np.insert(frame_duration[:-1], 0, 0))  # start of each frame in minutes
    target_time = 300  # 5 minutes in seconds

    # Find first frame at or after 5 minutes
    frame_at_5min = np.argmax(frame_start >= target_time)
    start_frame_idx = frame_at_5min  # same meaning here
    last_frame = len(frame_start)

    return frame_at_5min, start_frame_idx, frame_onsets, last_frame, total_scan_duration

def file_exists(directory, pattern):
    return any(Path(directory).glob(f'*{pattern}*'))

def get_thresh(thr, thrp, default_thresh):
    if thr is not None and thrp is not None:
        raise ValueError("Please specify only one threshold method: 'thr' or 'thrp'")
    return ('thr', thr) if thr is not None else ('thrp', thrp) if thrp else default_thresh

def get_thresh_method(thresh):
    return 'absolute' if thresh == 'thr' else 'percentage'

def update_set_pathnames(subject_path):
    path_file = Path('pet_ki_proc/app/utils/pathnames.py')
    lines = path_file.read_text().splitlines()
    for i, line in enumerate(lines):
        if "Cfg['preprocDir']" in line:
            lines[i] = f"    Cfg['preprocDir'] = '{subject_path}'"
            break
    path_file.write_text('\n'.join(lines) + '\n')

def zip_nii(directory, file_pattern):
    directory = Path(directory)
    for filename in os.listdir(directory):
        if filename.endswith(f"{file_pattern}.nii") and not filename.endswith('.nii.gz'):
            nii_path = directory / filename
            gz_path = nii_path.with_suffix('.nii.gz')
            with open(nii_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            nii_path.unlink()