import os
import glob
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import zscore
from scipy.fft import fft
from scipy.fftpack import dctn, idctn
from scipy.io import loadmat
from nilearn import image as niimage
import mne

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
from typing import List, Tuple, Dict

import warnings
warnings.simplefilter("ignore", RuntimeWarning)

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

def visualize_results(img_a: np.ndarray, img_b: np.ndarray, fig_size: Tuple=(6, 8)):
    fig, axs = plt.subplots(1, 2, figsize=fig_size)

    img_list = [img_a, img_b]
    
    # visualizing the results
    for ax, img in zip(axs.ravel(), img_list):
        ax.imshow(img, cmap='gray')
    
    plt.tight_layout()
    plt.show()

def normalize_data(data: np.ndarray, base_range: Tuple=None):
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

def get_individual_list(dataset: str, data_list: List):
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

def correct_vhdr_path(vhdr_path: Path):
    """Check and modify metadata of the EEG (vhdr) (if not correct)
    Args:
        vhdr_path (Path): path to vhdr file
    """
    # marker file
    vmrk_path = vhdr_path.parent/f"{vhdr_path.stem}.vmrk"
    
    # EEG file
    eeg_path = vhdr_path.parent/f"{vhdr_path.stem}.eeg"
    if not os.path.isfile(str(eeg_path)):
        eeg_path = vhdr_path.parent/f"{vhdr_path.stem}.dat"

    assert os.path.isfile(str(vmrk_path))
    assert os.path.isfile(str(eeg_path))
    
    data_filename = eeg_path.name
    vmrk_filename = vmrk_path.name

    # read vhdr file
    with open(vhdr_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    is_modify = False
    for i, line in enumerate(lines):
        if line.startswith('DataFile='):
            if data_filename not in line:
                lines[i] = f'DataFile={data_filename}\n'
                is_modify = True
        elif line.startswith('MarkerFile='):
            if vmrk_filename not in line:
                lines[i] = f'MarkerFile={vmrk_filename}\n'
                is_modify = True

    if is_modify:
        print(f"Fixed {vhdr_path}")
        # write the modified lines back to the file
        with open(vhdr_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

def compute_fft(signal_1d: np.ndarray, limit: bool=True, f_limit: int=250):
    """Compute Fast Fourier Transforms
    Args:
        signal_1d (np.ndarray): 1D EEG signal from a channel
        limit (bool): apply a frequency limit cut
        f_limit (int): frequency limit value
    Returns:
        transformed signal (np.ndarray)
    """
    N = len(signal_1d)

    fft1 = fft(signal_1d)

    if(limit):
        if(fft1.shape[0] > f_limit):
            return fft1[:f_limit]
        print(f"WARNING: {fft1.shape} is lower than {f_limit}")
        # pad signal with zeros
        return np.append(fft1, np.zeros((f_limit - fft1.shape[0],), dtype=np.complex128))

    return np.abs(fft1[:N//2])

def stft(eeg, channel: int=0, window_size: int=2, fs: int=250, limit=True, 
         f_limit: int=250, start_time: int=None, stop_time: int=None):
    """Short-time Fourier Transform on a 1D EEG signal
    Args:
        eeg: multi-channel EEG data
        channel (int): EEG channel index
        window_size (int): EEG window size to sample (in seconds)
        fs (int): frequency sampling rate
        limit (bool): apply a frequency limit cut
        f_limit (int): frequency limit value
        start_time (int): where to start calculating STFT
        stop_time (int): where to stop calculating STFT
    Returns:
        Transformed signal (np.ndarray)
    """

    # signal is 1D data
    signal = eeg[channel][:]

    if(type(signal) is tuple):
        signal, _ = signal
        signal = signal.reshape((signal.shape[1]))
    else:
        signal = signal.reshape((signal.shape[0]))

    if(start_time == None):
        start_time = 0
    if(stop_time == None):
        stop_time = len(signal)

    signal = signal[start_time: stop_time]

    t = []
    Z = []
    seconds = 0

    fs_window_size = int(window_size*fs)
    sample_range = list(range(start_time, stop_time, fs_window_size))

    # remove the last signal part if smaller than fs_window_size
    if (stop_time - start_time) % fs_window_size != 0:
        sample_range = sample_range[:-1]

    for time in sample_range:
        fft1 = compute_fft(signal[time: time + fs_window_size], limit=limit, f_limit=f_limit)

        N = len(signal[time: time + fs_window_size])/2
        f = np.linspace (0, len(fft1), int(N/2))

        # average
        Z += [list(abs(fft1[1:]))] # remove the Direct Current (DC) component
        t += [seconds]
        seconds += window_size

    return f[1:], np.transpose(np.array(Z)), t

### Oddball
def get_fmri_instance_Oddball(data_root: Path, individual_name: str="sub001", task: int=1, run: int=1):
    """Load fMRI data from a given individual
    """
    task_run = f"task{task:03}_run{run:03}"
    
    fmri_data_path = data_root/individual_name/"BOLD"/task_run/"bold.nii.gz"
    
    # load fMRI data
    fmri_data = niimage.load_img(fmri_data_path)
    
    return fmri_data

def get_eeg_instance_Oddball(data_root: Path, individual_name: str="sub001", task: int=1, run: int=1):
    """Load EEG data from a given individual
    """
    task_run = f"task{task:03}_run{run:03}"
    eeg_data_path = data_root/individual_name/"EEG"/task_run/"EEG_noGA.mat"
    
    eeg_data = loadmat(eeg_data_path)

    # Follow Calhas et al., used first 43 channels (ref: https://github.com/DCalhas/eeg_to_fmri)
    return eeg_data['data_noGA'][:43, :]

def get_data_Oddball(data_root: Path, individual_name: str, task: int, run: int, 
                     bold_shift: int=6, f_resample: float=2.0, 
                     eeg_limit: bool=True, eeg_f_limit: float=250, 
                     standardize_eeg: bool=True, n_volumes: int=164):
    """Get pair EEG & fMRI data from an individual
    """

    # process fMRI data
    fmri_data = get_fmri_instance_Oddball(data_root, individual_name=individual_name, task=task, run=run)
    fmri_data = fmri_data.get_fdata()

    # get the last slices (ignore first slices)
    recording_time = min(n_volumes, fmri_data.shape[-1])
    fmri_data = fmri_data[:, :, :, bold_shift: recording_time + bold_shift] # [64, 64, 32, N]
    
    # normalize each fMRI voxel
    min_vals = np.min(fmri_data, axis=(0, 1, 2), keepdims=True)
    max_vals = np.max(fmri_data, axis=(0, 1, 2), keepdims=True)

    # normalize the data to [0, 1]
    fmri_data = (fmri_data - min_vals) / (max_vals - min_vals + 1e-10)
    fmri_data = fmri_data.transpose(3, 0, 1, 2) # [N, 64, 64, 32]

    # process EEG data
    eeg_data = get_eeg_instance_Oddball(data_root, individual_name=individual_name, task=task, run=run)
        
    # frequency sample: 1000.0 Hz
    fs_sample = 1000
    # num. channels: 43
    len_channels = len(eeg_data)
    
    x_instance = []
    for channel in range(len_channels):
        _, Zxx, _ = stft(eeg_data, channel=channel, window_size=f_resample, 
                         fs=fs_sample, limit=eeg_limit, f_limit=eeg_f_limit)
        x_instance += [Zxx]
        
    if(standardize_eeg):
        eeg_data = zscore(np.array(x_instance))
    else:
        eeg_data = np.array(x_instance)
    
    eeg_data = eeg_data.transpose(2, 0, 1)[bold_shift: recording_time + bold_shift] # [N, 43, 249]

    return eeg_data, fmri_data

## NODDI
def get_fmri_instance_NODDI(data_root: Path, individual_name: str):
    """Get fMRI data from a given individual
    """
    
    fmri_data_path = sorted(glob.glob(str(data_root/'fMRI'/individual_name/'*_cross.nii.gz'), recursive=True))
    assert len(fmri_data_path) == 1

    fmri_data_path = fmri_data_path[0]
    
    # load fMRI data
    fmri_data = niimage.load_img(fmri_data_path)
    
    return fmri_data

def get_eeg_instance_NODDI(data_root: Path, individual_name: str):
    """Get EEG data from a given individual
    """
    individual_path = data_root/'EEG'/individual_name
    individual_path = individual_path/'export'

    vhdr_path = glob.glob(str(individual_path/'*.vhdr'))
    assert len(vhdr_path) == 1, f'Found multiple vhdr files in {individual_path}'
    
    vhdr_path = vhdr_path[0]

    # correct EEG metadata if needed
    correct_vhdr_path(Path(vhdr_path))
    
    return mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=0)

def get_data_NODDI(data_root: Path, individual_name: str, bold_shift: int=6, 
                   f_resample: float=2.160, eeg_limit: bool=True, eeg_f_limit: int=250, 
                   standardize_eeg: bool=True, n_volumes: int=294):
    """Get pair EEG & fMRI data from an individual
    """

    # process fMRI data
    fmri_data = get_fmri_instance_NODDI(data_root=data_root, individual_name=individual_name)
    fmri_data = fmri_data.get_fdata()

    # get the last slices (ignore first slices)
    recording_time = min(n_volumes, fmri_data.shape[-1])
    fmri_data = fmri_data[:, :, :, bold_shift: recording_time + bold_shift] # [64, 64, 30, N]

    # normalize each fMRI voxel
    min_vals = np.min(fmri_data, axis=(0, 1, 2), keepdims=True)
    max_vals = np.max(fmri_data, axis=(0, 1, 2), keepdims=True)

    # normalize the data to [0, 1]
    fmri_data = (fmri_data - min_vals) / (max_vals - min_vals + 1e-10)
    fmri_data = fmri_data.transpose(3, 0, 1, 2) # [N, 64, 64, 30]

    # process EEG data
    eeg_data = get_eeg_instance_NODDI(data_root=data_root, individual_name=individual_name)
        
    # frequency sample: 250.0 Hz
    fs_sample = eeg_data.info['sfreq']
    # num. channels: 64
    len_channels = len(eeg_data.ch_names)
    
    x_instance = []
    for channel in range(len_channels):
        _, Zxx, _ = stft(eeg_data, channel=channel, window_size=f_resample, 
                         fs=fs_sample, limit=eeg_limit, f_limit=eeg_f_limit)
        x_instance += [Zxx]

    if(standardize_eeg):
        eeg_data = zscore(np.array(x_instance))
    else:
        eeg_data = np.array(x_instance)

    eeg_data = eeg_data.transpose(2, 0, 1)[bold_shift: recording_time + bold_shift] # [N, 64, 294]

    return eeg_data, fmri_data

## CN-EPFL
def get_fmri_instance_CNEPFL(data_root: Path, individual_name: str="sub-02", 
                              run: int=1, downsample=True, downsample_shape=(64, 64, 30)):
    """Load fMRI data from a given individual
    """
    
    fmri_data_dir = data_root/individual_name/"ses-001/func"
    fmri_data_paths = glob.glob(str(fmri_data_dir/"**bold.nii.gz"))

    run_id = f"task-main_run-{run:03}_bold.nii.gz"

    fmri_data_path = None
    for file_path in fmri_data_paths:
        if file_path.endswith(run_id):
            fmri_data_path = file_path
            break
    if fmri_data_path is None:
        raise Exception("No fMRI file found!")
    
    # load fMRI data
    fmri_data = niimage.load_img(fmri_data_path) # [108, 108, 64, N]

    if(downsample):
        dct_coeffs = dctn(fmri_data.get_fdata(), axes=(0, 1, 2), norm="ortho") # [108, 108, 64, N]

        # retain only the low-frequency coefficients (64, 64, 30)
        H, W, D = downsample_shape
        truncated_dct = dct_coeffs[:H, :W, :D, :] # [64, 64, 30, N]

        # perform the inverse DCT-III to reconstruct in reduced space
        new_data = idctn(truncated_dct, axes=(0, 1, 2), norm='ortho') # [64, 64, 30, N]
        
        fmri_data = niimage.new_img_like(fmri_data, new_data)
    
    return fmri_data # [64, 64, 30, N]

def get_eeg_instance_CNEPFL(data_root: Path, individual_name: str="sub-02", run: int=1):
    """Get EEG data from a given individual
    """
    eeg_data_dir = data_root/individual_name/"ses-001/eeg"
    vhdr_paths = glob.glob(str(eeg_data_dir/"**_eeg.vhdr"))

    run_id = f"task-main_run-{run:03}_eeg.vhdr"

    vhdr_data_path = None
    for file_path in vhdr_paths:
        if file_path.endswith(run_id):
            vhdr_data_path = file_path
            break
    if vhdr_data_path is None:
        raise Exception("No VHDR file found!")

    # correct EEG metadata if needed
    correct_vhdr_path(Path(vhdr_data_path))
    return mne.io.read_raw_brainvision(vhdr_data_path, preload=False, verbose=0)

def get_data_CNEPFL(data_root:Path, individual_name: str, run: int, bold_shift: int=6, 
                    f_resample: float=1.280, eeg_limit: bool=True, eeg_f_limit: int=250, 
                    standardize_eeg: bool=True, n_volumes: int=364):
    """Get pair EEG & fMRI data from an individual
    """

    # process fMRI data
    fmri_data = get_fmri_instance_CNEPFL(data_root=data_root, individual_name=individual_name, run=run)
    fmri_data = fmri_data.get_fdata()

    # get the last slices (ignore first slices)
    recording_time = min(n_volumes, fmri_data.shape[-1])
    fmri_data = fmri_data[:, :, :, bold_shift: recording_time + bold_shift] # [64, 64, 30, N]

    # normalize each fMRI voxel
    min_vals = np.min(fmri_data, axis=(0, 1, 2), keepdims=True)
    max_vals = np.max(fmri_data, axis=(0, 1, 2), keepdims=True)

    # normalize the data to [0, 1]
    fmri_data = (fmri_data - min_vals) / (max_vals - min_vals + 1e-10)
    fmri_data = fmri_data.transpose(3, 0, 1, 2) # [N, 64, 64, 30]
    
    # process EEG data
    eeg_data = get_eeg_instance_CNEPFL(data_root=data_root, individual_name=individual_name, run=run)
        
    # frequency sample: 5000.0 Hz
    fs_sample = eeg_data.info['sfreq']
    # num. channels: 64
    len_channels = len(eeg_data.ch_names)
    
    x_instance = []
    for channel in range(len_channels):
        _, Zxx, _ = stft(eeg_data, channel=channel, window_size=f_resample, 
                         fs=fs_sample, limit=eeg_limit, f_limit=eeg_f_limit)
        x_instance += [Zxx]
        
    if(standardize_eeg):
        eeg_data = zscore(np.array(x_instance))
    else:
        eeg_data = np.array(x_instance)
        
    eeg_data = eeg_data.transpose(2, 0, 1)[bold_shift: recording_time + bold_shift] # [N, 64, 249]

    return eeg_data, fmri_data

def create_eeg_bold_pairs(eeg_data: np.ndarray, fmri_data: np.ndarray, interval_eeg: int, n_volumes: int):
    """Copy from Calhas et al., (ref: https://github.com/DCalhas/eeg_to_fmri)
    """
    x_eeg = np.empty((n_volumes - interval_eeg, ) + eeg_data.shape[1:] + (interval_eeg, ))
    x_bold = np.empty((n_volumes - interval_eeg, ) + fmri_data.shape[1:])
        
    for index_volume in range(0, n_volumes - interval_eeg):
        if(np.transpose(eeg_data[index_volume: index_volume + interval_eeg], (1, 2, 0)).shape[-1] != interval_eeg):
            continue

        x_eeg[index_volume] = np.transpose(eeg_data[index_volume: index_volume + interval_eeg], (1,2,0))
        x_bold[index_volume] = fmri_data[index_volume + interval_eeg]
    
    return x_eeg, x_bold

def save_h5_data_NODDI(individuals: List, processing_cfg: Dict, 
                       data_dir: Path, dest_dir: Path, data_name: str="NODDI"):
    """Save h5 data for NODDI dataset
    Args:
        individuals (List): List of individual names
        processing_cfg (Dict): Data pre-processing config (see data_cfg.py)
        data_dir (Path): Raw dataset directory
        dest_dir (Path): Where to save h5 files
        data_name (str): Dataset name (NODDI)
    """

    individuals = sorted(individuals)
    pbar = tqdm(individuals, leave=True)

    # get processing configs
    bold_shift = processing_cfg['bold_shift']
    eeg_limit = processing_cfg['eeg_limit']
    eeg_f_limit = processing_cfg['eeg_f_limit']
    interval_eeg = processing_cfg['interval_eeg']

    n_volumes = processing_cfg[data_name]['n_volumes']
    f_resample = processing_cfg[data_name]['f_resample']

    for individual_name in pbar:
        pbar.set_description(individual_name)

        eeg_data, fmri_data = get_data_NODDI(data_root=data_dir, 
                                             individual_name=individual_name, 
                                             bold_shift=bold_shift,
                                             f_resample=f_resample, eeg_limit=eeg_limit, 
                                             eeg_f_limit=eeg_f_limit, n_volumes=n_volumes)

        eeg_data, fmri_data = create_eeg_bold_pairs(eeg_data, fmri_data, 
                                                    interval_eeg=interval_eeg, 
                                                    n_volumes=n_volumes)

        eeg_data = eeg_data.transpose(0, 3, 1, 2) # [N, 20, C, F]
        fmri_data = fmri_data.transpose(0, 3, 1, 2) # [N, D, W, H]

        assert len(eeg_data) == len(fmri_data), \
            f'EEG not same length as fMRI, got {len(eeg_data) and len(fmri_data)}'
        
        h5_path = f"{Path(dest_dir)/individual_name}.h5"
        hf = h5py.File(h5_path, 'w')
        hf.create_dataset('eeg', data=eeg_data)
        hf.create_dataset('fmri', data=fmri_data)
        hf.close()

def save_h5_data_Oddball(individuals: List, tasks: List, runs: List, 
                         processing_cfg: Dict, data_dir: Path, 
                         dest_dir: Path, data_name: str="Oddball"):
    """Save h5 data for Oddball dataset
    Args:
        individuals (List): List of individual names
        tasks (List): Oddball has 2 tasks [1, 2]
        runs (List): Each task has 3 runs [1, 2, 3]
        processing_cfg (Dict): Data pre-processing config (see data_cfg.py)
        data_dir (Path): Raw dataset directory
        dest_dir (Path): Where to save h5 files
        data_name (str): Dataset names (Oddball)
    """
    individuals = sorted(individuals)
    pbar = tqdm(individuals, leave=True)

    # get processing configs
    bold_shift = processing_cfg['bold_shift']
    eeg_limit = processing_cfg['eeg_limit']
    eeg_f_limit = processing_cfg['eeg_f_limit']
    interval_eeg = processing_cfg['interval_eeg']

    n_volumes = processing_cfg[data_name]['n_volumes']
    f_resample = processing_cfg[data_name]['f_resample']

    for individual_name in pbar:
        save_dir = dest_dir/individual_name
        os.makedirs(save_dir, exist_ok=True)
        
        pbar.set_description(individual_name)
        for task in tasks:
            for run in runs:
                task_run = f"task{task:03}_run{run:03}"
                eeg_data, fmri_data = get_data_Oddball(data_root=data_dir, 
                                                       individual_name=individual_name, 
                                                       task=task, run=run, bold_shift=bold_shift, 
                                                       f_resample=f_resample, eeg_limit=eeg_limit, 
                                                       eeg_f_limit=eeg_f_limit, n_volumes=n_volumes)

                eeg_data, fmri_data = create_eeg_bold_pairs(eeg_data, fmri_data, 
                                                            interval_eeg=interval_eeg, 
                                                            n_volumes=n_volumes)

                eeg_data = eeg_data.transpose(0, 3, 1, 2) # [N, 20, C, F]
                fmri_data = fmri_data.transpose(0, 3, 1, 2) # [N, D, W, H]

                assert len(eeg_data) == len(fmri_data), \
                    f'EEG not same length as fMRI, got {len(eeg_data) and len(fmri_data)}'
                
                h5_path = f"{save_dir/task_run}.h5"
                hf = h5py.File(h5_path, 'w')
                hf.create_dataset('eeg', data=eeg_data)
                hf.create_dataset('fmri', data=fmri_data)
                hf.close()

def save_h5_data_CNEPFL(individuals: List, runs: List, processing_cfg: Dict, 
                        data_dir: Path, dest_dir: Path, data_name: str="CNEPFL"):
    """Save h5 data for CNEPFL dataset
    Args:
        individuals (List): List of individual names
        runs (List): CNEPFL has total of 6 runs [1, 2, 3, 4, 5, 6]
        processing_cfg (Dict): Data pre-processing config (see data_cfg.py)
        data_dir (Path): Raw dataset directory
        dest_dir (Path): Where to save h5 files
        data_name (str): Dataset names (CNEPFL)
    """
    individuals = sorted(individuals)
    pbar = tqdm(individuals, leave=True)

    # get processing configs
    bold_shift = processing_cfg['bold_shift']
    eeg_limit = processing_cfg['eeg_limit']
    eeg_f_limit = processing_cfg['eeg_f_limit']
    interval_eeg = processing_cfg['interval_eeg']

    n_volumes = processing_cfg[data_name]['n_volumes']
    f_resample = processing_cfg[data_name]['f_resample']

    for individual_name in pbar:
        pbar.set_description(individual_name)
        for run in runs:
            eeg_data, fmri_data = get_data_CNEPFL(data_root=data_dir, 
                                                  individual_name=individual_name, 
                                                  run=run, bold_shift=bold_shift, 
                                                  f_resample=f_resample, eeg_limit=eeg_limit, 
                                                  eeg_f_limit=eeg_f_limit, n_volumes=n_volumes)

            eeg_data, fmri_data = create_eeg_bold_pairs(eeg_data, fmri_data, 
                                                        interval_eeg=interval_eeg, 
                                                        n_volumes=n_volumes)

            eeg_data = eeg_data.transpose(0, 3, 1, 2) # [N, 20, C, F]
            fmri_data = fmri_data.transpose(0, 3, 1, 2) # [N, D, W, H]

            assert len(eeg_data) == len(fmri_data), \
                f'EEG not same length as fMRI, got {len(eeg_data) and len(fmri_data)}'
            
            h5_path = f"{Path(dest_dir)/individual_name}_run-{run:03}.h5"
            hf = h5py.File(h5_path, 'w')
            hf.create_dataset('eeg', data=eeg_data)
            hf.create_dataset('fmri', data=fmri_data)
            hf.close()