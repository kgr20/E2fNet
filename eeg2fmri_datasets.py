import numpy as np
import torch
from torch.utils.data import Dataset

class EEG2fMRIDataset(Dataset):
    def __init__(self, eeg_data: np.ndarray, fmri_data: np.ndarray):
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.fmri_data = torch.tensor(fmri_data, dtype=torch.float32)

        assert len(self.eeg_data) == len(self.fmri_data), f'EEG and fMRI not the same length!'

    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.fmri_data[idx]