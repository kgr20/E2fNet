from pathlib import Path
from tqdm.auto import tqdm
from typing import List

import h5py
import numpy as np

import matplotlib.pyplot as plt
import time

def load_h5_from_list(data_root: str, individual_list: List[str]):
    eeg_data = None
    fmri_data = None

    pbar = tqdm(individual_list, leave=True)
    for individual_name in pbar:
        pbar.set_description(f'Individual {individual_name}')
        h5_path = Path(data_root) / f'{individual_name}.h5'

        if not h5_path.exists():
            print(f"[Warning] File {h5_path} does not exist. Skipping.")
            continue

        with h5py.File(h5_path, 'r') as f:
            eeg_indv = np.array(f['eeg'][:])
            fmri_indv = np.array(f['fmri'][:])

            if eeg_data is None:
                eeg_data = eeg_indv
            else:
                eeg_data = np.concatenate([eeg_data, eeg_indv], axis=0)

            if fmri_data is None:
                fmri_data = fmri_indv
            else:
                fmri_data = np.concatenate([fmri_data, fmri_indv], axis=0)

    if eeg_data is None or fmri_data is None:
        raise ValueError("No data loaded.")
    
    return eeg_data, fmri_data

def visual_results(img_a: np.ndarray, img_b: np.ndarray, fig_size=(6, 8)):
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    img_list = [img_a, img_b]
    
    # visualizing the results
    for ax, img in zip(axs.ravel(), img_list):
        ax.imshow(img, cmap='gray')
    
    plt.tight_layout()
    plt.show()

def normalize_data(data: np.ndarray, base_range=None):
    """Normalize data based on min_val & max_val
    Note: This is NOT normalizing to range [min_val, max_val]
    Args:
        base_range (tuple): min, max value of the whole dataset
    """
    if base_range is None:
        # normalize to range [0, 1]
        min_val, max_val = data.min(), data.max()
    else:
        min_val, max_val = base_range
        assert min_val < max_val, f"Min must be less than max, got {min_val=}, {max_val=}"
    
    return (data - min_val) / (max_val - min_val + 1e-10)

def get_current_timestr():
    timestr = time.strftime("%y%m%d_%H%M%S")

    return timestr

def get_individual_data(dataset: str, data_list: list):
    idv_list = []
    if dataset == "NODDI":
        idv_list =  data_list
    elif dataset == "Oddball":
        tasks = [1, 2]
        runs = [1, 2, 3]
        for individual in data_list:
            for task in tasks:
                for run in runs:
                    idv_list.append(f"{individual}/task{task:03}_run{run:03}") # sub00x/task00x_run00x
    elif dataset == "CNEPFL":
        runs = [1] # can be [1, 2, 3, 4, 5, 6] but only use run-001 in this paper
        for individual in data_list:
            for run in runs:
                idv_list.append(f"{individual}/{individual}_run-{run:03}") # sub-xx/sub-xx_run-xxx

    return idv_list