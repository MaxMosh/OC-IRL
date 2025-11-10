import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F

class SequenceDataset(Dataset):
    def __init__(self, root_dir="../data", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []

        # Run into S01 to S20 folders
        for subject_dir in sorted(os.listdir(root_dir)):
            full_dir = os.path.join(root_dir, subject_dir)
            if os.path.isdir(full_dir):
                for file in os.listdir(full_dir):           # list .csv from current SXX folder
                    if file.endswith(".csv"):
                        self.files.append(os.path.join(full_dir, file))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        seq = pd.read_csv(file_path, header=None).values    # load csv
        seq = seq.transpose()
        seq = torch.tensor(seq, dtype=torch.float32)        # convert to pytorch tensor

        if self.transform:
            seq = self.transform(seq)

        return seq