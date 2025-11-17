import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import glob


class JointAngleDataset(Dataset):
    """
    Dataset class for loading joint angle sequences from CSV files.
    Each CSV contains 2 rows (q1 and q2) with variable length sequences.
    """
    
    def __init__(
        self, 
        root_dir: str,
        obs_frames: int = 25,
        pred_frames: int = 100,
        subjects: List[str] = None,
        normalize: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing S01, S02, ... folders
            obs_frames: Number of frames to observe (H in the paper)
            pred_frames: Number of frames to predict (F in the paper)
            subjects: List of subject folders to include (e.g., ['S01', 'S05'])
                     If None, all subjects are included
            normalize: Whether to normalize the data
        """
        self.root_dir = root_dir
        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.total_frames = obs_frames + pred_frames
        self.normalize = normalize
        
        # Load all sequences
        self.sequences = []
        self.load_data(subjects)
        
        # Compute normalization statistics
        if self.normalize:
            self.compute_statistics()
    
    def load_data(self, subjects: List[str] = None):
        """
        Load all CSV files from specified subject folders.
        """
        # Get all subject folders
        if subjects is None:
            subject_folders = sorted(glob.glob(os.path.join(self.root_dir, 'S*')))
        else:
            subject_folders = [os.path.join(self.root_dir, s) for s in subjects]
        
        print(f"Loading data from {len(subject_folders)} subject folders...")
        
        for subject_folder in subject_folders:
            if not os.path.isdir(subject_folder):
                continue
                
            # Get all CSV files in this subject folder
            csv_files = glob.glob(os.path.join(subject_folder, '*.csv'))
            
            for csv_file in csv_files:
                try:
                    # Read CSV with no header (data is transposed)
                    df = pd.read_csv(csv_file, header=None)
                    
                    # Transpose to get shape (time_steps, 2) where columns are q1, q2
                    data = df.T.values.astype(np.float32)
                    
                    # Extract valid sequences with sliding window
                    self.extract_sequences(data, csv_file)
                    
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
        
        print(f"Loaded {len(self.sequences)} sequences in total")
    
    def extract_sequences(self, data: np.ndarray, source_file: str):
        """
        Extract all possible sequences of length (obs_frames + pred_frames)
        from a single motion capture file using a sliding window.
        
        Args:
            data: Array of shape (time_steps, 2) containing q1 and q2
            source_file: Source file name for tracking
        """
        seq_length = data.shape[0]
        
        # Check if sequence is long enough
        if seq_length < self.total_frames:
            return
        
        # Extract all possible windows with stride=1
        for start_idx in range(seq_length - self.total_frames + 1):
            end_idx = start_idx + self.total_frames
            sequence = data[start_idx:end_idx, :]  # Shape: (total_frames, 2)
            
            self.sequences.append({
                'data': sequence,
                'source': os.path.basename(source_file),
                'start_idx': start_idx
            })
    
    def compute_statistics(self):
        """
        Compute mean and std for normalization across all sequences.
        """
        all_data = np.concatenate([seq['data'] for seq in self.sequences], axis=0)
        self.mean = np.mean(all_data, axis=0, keepdims=True)  # Shape: (1, 2)
        self.std = np.std(all_data, axis=0, keepdims=True) + 1e-8  # Shape: (1, 2)
        
        print(f"Data statistics - Mean: {self.mean.flatten()}, Std: {self.std.flatten()}")
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using precomputed statistics."""
        return (data - self.mean) / self.std
    
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return data * self.std + self.mean
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            obs: Observation sequence of shape (obs_frames, 2)
            target: Target sequence of shape (total_frames, 2)
        """
        sequence = self.sequences[idx]['data'].copy()
        
        # Normalize if required
        if self.normalize:
            sequence = self.normalize_data(sequence)
        
        # Split into observation and full sequence
        obs = sequence[:self.obs_frames, :]  # Shape: (obs_frames, 2)
        target = sequence  # Shape: (total_frames, 2)
        
        return torch.from_numpy(obs), torch.from_numpy(target)


def create_dataloaders(
    root_dir: str,
    train_subjects: List[str],
    test_subjects: List[str],
    obs_frames: int = 25,
    pred_frames: int = 100,
    batch_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and testing dataloaders.
    
    Args:
        root_dir: Root directory containing subject folders
        train_subjects: List of training subjects (e.g., ['S01', 'S02', ...])
        test_subjects: List of testing subjects (e.g., ['S14', 'S15'])
        obs_frames: Number of observation frames
        pred_frames: Number of prediction frames
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = JointAngleDataset(
        root_dir=root_dir,
        obs_frames=obs_frames,
        pred_frames=pred_frames,
        subjects=train_subjects,
        normalize=True
    )
    
    test_dataset = JointAngleDataset(
        root_dir=root_dir,
        obs_frames=obs_frames,
        pred_frames=pred_frames,
        subjects=test_subjects,
        normalize=True
    )
    
    # Use training statistics for test set normalization
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Define your data directory
    # data_dir = "/path/to/your/data"
    data_dir = "data"
    
    # Define train and test subjects
    train_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 
                      'S06', 'S07', 'S08', 'S09', 'S10']
    test_subjects = ['S11', 'S12', 'S13', 'S14', 'S15']
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        root_dir=data_dir,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        obs_frames=25,
        pred_frames=100,
        batch_size=64
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    for obs, target in train_loader:
        print(f"Observation shape: {obs.shape}")  # (batch_size, 25, 2)
        print(f"Target shape: {target.shape}")  # (batch_size, 125, 2)
        break
