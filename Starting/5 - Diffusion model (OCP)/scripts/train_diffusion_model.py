import torch
import torch.nn as nn
import numpy as np
import os
# from torch.nn.utils.rnn import pad_sequence

from datetime import datetime

# Keeping date, hour and minutes (format : str YYYY-MM-DD_HH:MM)
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d %H:%M").replace(" ","_")


# Data loading, with normalization
DATA_PATH = "Starting/5 - Diffusion model/data/trajectories_dataset_train.npy"
data = np.load(DATA_PATH, allow_pickle=True)

# Saving the stats
all_traj = np.concatenate([d["future"] for d in data], axis=0)
mean = all_traj.mean(axis=0)
std = all_traj.std(axis=0)
np.save("stats.npy", {"mean": mean, "std": std})

MAX_LEN = max(x["future"].shape[0] for x in data)
print(f"Longueur max détectée : {MAX_LEN}")


# Initializing pytorch dataset properly
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data, max_len, mean, std):
        self.data = data
        self.max_len = max_len
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        traj = torch.tensor(self.data[i]["future"], dtype=torch.float32)
        traj = (traj - self.mean) / (self.std + 1e-8)

        L = traj.shape[0]
        dim = traj.shape[1]

        # padding
        if L < self.max_len:
            pad = torch.zeros((self.max_len - L, dim))
            traj_padded = torch.cat([traj, pad], dim=0)
        else:
            traj_padded = traj[:self.max_len]

        mask = torch.zeros(self.max_len)
        mask[:L] = 1
        return traj_padded, mask


dataset = TrajectoryDataset(data, MAX_LEN, mean, std)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Time embedding and GRU model
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)      # MODIFIED: ADDED .to(device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TemporalDiffusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(input_size=dim, hidden_size=128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(128 + dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x_t, t):
        t_emb = timestep_embedding(t, x_t.shape[-1]).to(x_t.device)
        t_emb = t_emb[:, None, :].repeat(1, x_t.shape[1], 1)
        h, _ = self.gru(x_t)
        inp = torch.cat([h, t_emb], dim=-1)
        return self.mlp(inp)


# Diffusion parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps = 1000
betas = torch.linspace(1e-5, 1e-3, timesteps).to(device)                            # MODIFIED: .to(device) added
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
model = TemporalDiffusion(dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
n_epochs = 200
for epoch in range(n_epochs):
    total_loss = 0.0
    for x0, mask in loader:
        x0 = x0.to(device)
        mask = mask.to(device)

        bsz, L, dim = x0.shape
        t = torch.randint(0, timesteps, (bsz,), device=device)
        eps = torch.randn_like(x0)

        a_bar = alphas_cumprod[t].view(-1, 1, 1)#.to(device)         # MODIFIED: .to(device) deleted
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps

        # Freezing a random prefix
        prefix_len = torch.randint(10, L - 10, (1,)).item()
        x_t[:, :prefix_len] = x0[:, :prefix_len]

        pred_eps = model(x_t, t)
        mask = mask.unsqueeze(-1)
        loss = (((eps - pred_eps) ** 2) * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {total_loss/len(loader):.6f}")


torch.save(model.state_dict(), f"Starting/5 - Diffusion model/trained_models/trained_diffusion_model_{formatted_now}.pt")