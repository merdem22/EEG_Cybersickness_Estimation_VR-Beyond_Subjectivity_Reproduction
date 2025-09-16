"""
loader.py
---------
Purpose:
    - Load processed EEG forecasting dataset and create DataLoaders for training.

Usage:
    from src.loader import EEGDataset, get_dataloader
    dataset = EEGDataset('data/processed_eeg_forecasting_dataset.npz', input_windows=5, target_windows=5)
    loader = get_dataloader('data/processed_eeg_forecasting_dataset.npz', batch_size=32, input_windows=5, target_windows=5)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class EEGDataset(Dataset):
    """
    EEG forecasting dataset with support for different target strategies.
    
    Args:
        npz_path: Path to the processed .npz file
        input_windows: Number of consecutive windows to use as input
        target_windows: Number of consecutive windows to use as target
        target_strategy: 'average', 'max', 'multi', 'next_average', 'next_max', or 'next_multi'
    """
    def __init__(self, npz_path, input_windows=5, target_windows=1, target_strategy='next_average'):
        data = np.load(npz_path, allow_pickle=True)
        self.eeg_windows = torch.tensor(data['eeg_windows'], dtype=torch.float32)
        self.cs_labels = torch.tensor(data['cs_labels'], dtype=torch.float32)
        self.participant_ids = data['participant_ids']
        
        self.input_windows = input_windows
        self.target_windows = target_windows
        self.target_strategy = target_strategy

        
        # Calculate valid indices based on strategy
        self.valid_indices = self._calculate_valid_indices()
        
        print(f"Dataset: {len(self.valid_indices)} samples, input={input_windows}w, target={target_windows}w ({target_strategy})")
        
    def _calculate_valid_indices(self):
        """Calculate valid indices based on target strategy."""
        if self.target_strategy in ['multi', 'average', 'max']:
            return self._get_overlapping_indices()
        else:
            return self._get_standard_indices()
    
    def _get_standard_indices(self):
        """Get indices for standard strategies (next_average, next_max, next_multi)."""
        valid_indices = []
        for i in range(len(self.eeg_windows) - self.input_windows - self.target_windows + 1):
            # Check if all windows in the sequence belong to the same participant
            start_idx = i
            end_idx = i + self.input_windows + self.target_windows
            participant_sequence = self.participant_ids[start_idx:end_idx]
            if len(np.unique(participant_sequence)) == 1:
                valid_indices.append(i)
        return valid_indices
    
    def _get_overlapping_indices(self):
        """Get indices for overlapping strategies (multi, average, max)."""
        valid_indices = []
        unique_participants = np.unique(self.participant_ids)
        
        for participant in unique_participants:
            # Get all indices for this participant
            participant_mask = self.participant_ids == participant
            participant_indices = np.where(participant_mask)[0]
            
            if len(participant_indices) < self.input_windows:
                continue
                
            # Create overlapping sequences for this participant
            for i in range(len(participant_indices) - self.input_windows + 1):
                start_idx = participant_indices[i]
                # Check if we have enough consecutive windows
                end_idx = start_idx + self.input_windows
                if end_idx <= len(self.eeg_windows):
                    valid_indices.append(start_idx)
        
        return valid_indices
    
    def _get_targets_average(self, start_idx, input_end):
        """Get targets for 'average' strategy - average of overlapping windows."""
        return self.cs_labels[start_idx:start_idx + self.input_windows].mean()
    
    def _get_targets_max(self, start_idx, input_end):
        """Get targets for 'max' strategy - max of overlapping windows."""
        return self.cs_labels[start_idx:start_idx + self.input_windows].max()
    
    def _get_targets_multi(self, start_idx, input_end):
        """Get targets for 'multi' strategy - individual labels of overlapping windows."""
        return self.cs_labels[start_idx:start_idx + self.input_windows]
    
    def _get_targets_next_average(self, start_idx, input_end):
        """Get targets for 'next_average' strategy."""
        target_start = input_end
        target_end = target_start + self.target_windows
        target_ratings = self.cs_labels[target_start:target_end]
        return target_ratings.mean()
    
    def _get_targets_next_max(self, start_idx, input_end):
        """Get targets for 'next_max' strategy."""
        target_start = input_end
        target_end = target_start + self.target_windows
        target_ratings = self.cs_labels[target_start:target_end]
        return target_ratings.max()
    
    def _get_targets_next_multi(self, start_idx, input_end):
        """Get targets for 'next_multi' strategy."""
        target_start = input_end
        target_end = target_start + self.target_windows
        return self.cs_labels[target_start:target_end]
    
    def _get_targets(self, start_idx, input_end):
        """Get targets based on strategy."""
        if self.target_strategy == 'average':
            return self._get_targets_average(start_idx, input_end)
        elif self.target_strategy == 'max':
            return self._get_targets_max(start_idx, input_end)
        elif self.target_strategy == 'multi':
            return self._get_targets_multi(start_idx, input_end)
        elif self.target_strategy == 'next_average':
            return self._get_targets_next_average(start_idx, input_end)
        elif self.target_strategy == 'next_max':
            return self._get_targets_next_max(start_idx, input_end)
        elif self.target_strategy == 'next_multi':
            return self._get_targets_next_multi(start_idx, input_end)
        else:
            raise ValueError(f"Unknown target_strategy: {self.target_strategy}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Input: consecutive windows
        input_start = start_idx
        input_end = start_idx + self.input_windows
        inputs = self.eeg_windows[input_start:input_end]
        
        # Get targets based on strategy
        targets = self._get_targets(start_idx, input_end)
        
        # Get participant ID (all windows in sequence belong to same participant)
        participant_id = self.participant_ids[start_idx]
        
        return inputs, targets, participant_id

class EEGDatasetSubset(Dataset):
    """
    A wrapper for EEGDataset that handles Subset properly.
    This ensures valid_indices work correctly when the dataset is subsetted.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # Map the subset index to the original dataset's valid_indices
        original_idx = self.dataset.valid_indices[self.indices[idx]]
        
        # Get the data using the original dataset's __getitem__
        start_idx = original_idx
        
        # Input: consecutive windows
        input_start = start_idx
        input_end = start_idx + self.dataset.input_windows
        inputs = self.dataset.eeg_windows[input_start:input_end]
        
        # Get targets using the helper method
        targets = self.dataset._get_targets(start_idx, input_end)
        
        # Get participant ID
        participant_id = self.dataset.participant_ids[start_idx]
        
        return inputs, targets, participant_id

def get_dataloader(npz_path, batch_size=32, shuffle=True, drop_last=False, 
                  input_windows=5, target_windows=1, target_strategy='next_average'):
    """
    Returns a PyTorch DataLoader for the EEG forecasting dataset.
    
    Args:
        npz_path: Path to the processed .npz file
        batch_size: Batch size for training/evaluation
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        input_windows: Number of consecutive windows to use as input
        target_windows: Number of consecutive windows to use as target
        target_strategy: 'average', 'max', 'multi', 'next_average', 'next_max', or 'next_multi'
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = EEGDataset(npz_path, input_windows, target_windows, target_strategy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) 