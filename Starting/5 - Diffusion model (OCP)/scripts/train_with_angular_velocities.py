import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Keeping date, hour and minutes (format : str YYYY-MM-DD_HH:MM)
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d_%H:%M")


# Data loading, with normalization
# DATA_PATH = "Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities.npy"
# data = np.load(DATA_PATH, allow_pickle=True)
# Data loading
DATA_PATH_TRAIN = "Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities_train.npy"
DATA_PATH_VAL = "Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities_val.npy"

data_train = np.load(DATA_PATH_TRAIN, allow_pickle=True)
data_val = np.load(DATA_PATH_VAL, allow_pickle=True)

print(f"Train dataset size: {len(data_train)}")
print(f"Val dataset size: {len(data_val)}")

# Stats over the 4 inputs (q1, q2, dq1, dq2)
# TODO: I HAVE TO CHECK WHY I SHOULD ONLY GIVE THE STATS OVER FUTURE DATA AND NOT PREFIX AND FUTURE
all_traj = np.concatenate([d["future"] for d in data_train], axis=0)
# all_traj = np.concatenate([d["future"] for d in data], axis=0)
# all_traj = np.concatenate([d["prefix"] for d in data] + [d["future"] for d in data], axis=0)
mean = all_traj.mean(axis=0)  # shape (4,)
std = all_traj.std(axis=0)    # shape (4,)

# Saving the stats
np.save("Starting/5 - Diffusion model/data/stats_with_velocities_train.npy", 
        {"mean": mean, "std": std})
print(f"Computed means: {mean}")
print(f"Computed stds: {std}")

# Computing maximum length of subsequences (for padding)
# MAX_LEN = max(x["future"].shape[0] for x in data)
MAX_LEN = max(max(x["future"].shape[0] for x in data_train), max(x["future"].shape[0] for x in data_val))


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
        dim = traj.shape[1]  # should be 4

        # Padding
        if L < self.max_len:
            pad = torch.zeros((self.max_len - L, dim))
            traj_padded = torch.cat([traj, pad], dim=0)
        else:
            traj_padded = traj[:self.max_len]

        mask = torch.zeros(self.max_len)
        mask[:L] = 1
        return traj_padded, mask


# dataset = TrajectoryDataset(data, MAX_LEN, mean, std)
# batch_size = 32                                                                    # INITIALLY 32
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(f"DataLoader created: batch_size={batch_size}, {len(loader)} batches")

dataset_train = TrajectoryDataset(data_train, MAX_LEN, mean, std)
dataset_val = TrajectoryDataset(data_val, MAX_LEN, mean, std)
batch_size = 32                                                                 # INITIALLY 32
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=False)
print(f"Train DataLoader created: batch_size={batch_size}, {len(loader_train)} batches")
print(f"Val DataLoader created: batch_size={batch_size}, {len(loader_val)} batches")


# Time embedding and GRU model
def timestep_embedding(timesteps, dim):
    """Sinusoidal embedding for diffusion timestep"""
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)      # MODIFIED: ADDED .to(device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TemporalDiffusion(nn.Module):
    """Temporal diffusion model for trajectories with GRU"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # GRU to capture temporal dependencies
        self.gru = nn.GRU(input_size=dim, hidden_size=128, batch_first=True)
        # MLP to predict noise
        self.mlp = nn.Sequential(
            nn.Linear(128 + dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x_t, t):
        # Embedding of the diffusion timestep
        t_emb = timestep_embedding(t, x_t.shape[-1]).to(x_t.device)
        t_emb = t_emb[:, None, :].repeat(1, x_t.shape[1], 1)
        
        # Time-based processing with GRU
        h, _ = self.gru(x_t)
        
        # Concatenation and prediction
        inp = torch.cat([h, t_emb], dim=-1)
        return self.mlp(inp)


# Diffusion parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps = 1000                                                    # INITIALLY 1000
betas = torch.linspace(1e-5, 1e-3, timesteps).to(device)            # MODIFIED: .to(device) added
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

model = TemporalDiffusion(dim=4).to(device)  # dim=4 for [q1, q2, dq1, dq2]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Number of parameters of the used model: {sum(p.numel() for p in model.parameters())}")



# Validation function
@torch.no_grad()
def validate(model, loader_val, alphas_cumprod, device):
    model.eval()
    total_loss = 0.0
    
    for x0, mask in loader_val:
        x0 = x0.to(device)
        mask = mask.to(device)
        
        bsz, L, dim = x0.shape
        t = torch.randint(0, timesteps, (bsz,), device=device)
        eps = torch.randn_like(x0)
        
        a_bar = alphas_cumprod[t].view(-1, 1, 1)
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
        
        prefix_len = torch.randint(10, L - 10, (1,)).item()
        x_t[:, :prefix_len] = x0[:, :prefix_len]
        
        pred_eps = model(x_t, t)
        mask = mask.unsqueeze(-1)
        loss = (((eps - pred_eps) ** 2) * mask).sum() / mask.sum()
        
        total_loss += loss.item()
    
    model.train()
    return total_loss / len(loader_val)


# Train
n_epochs = 200                                                      # MODIFIED: INITIALLY 200
print(f"\nTraining on {n_epochs} epochs\n")

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    total_loss = 0.0
    # for x0, mask in loader:
    for x0, mask in loader_train:
        x0 = x0.to(device)
        mask = mask.to(device)

        bsz, L, dim = x0.shape
        
        # Sample a random diffusion timestep
        t = torch.randint(0, timesteps, (bsz,), device=device)
        
        # Sampling noise
        eps = torch.randn_like(x0)

        # Forward diffusion process
        a_bar = alphas_cumprod[t].view(-1, 1, 1)#.to(device)         # MODIFIED: .to(device) deleted
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps

        # Freezing a random prefix (conditioning)
        prefix_len = torch.randint(10, L - 10, (1,)).item()
        x_t[:, :prefix_len] = x0[:, :prefix_len]

        # Noise prediction
        pred_eps = model(x_t, t)
        
        # Hidden loss (ignores padding)
        mask = mask.unsqueeze(-1)
        loss = (((eps - pred_eps) ** 2) * mask).sum() / mask.sum()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # avg_loss = total_loss / len(loader)
    avg_train_loss = total_loss / len(loader_train)
    train_losses.append(avg_train_loss)

    avg_val_loss = validate(model, loader_val, alphas_cumprod, device)
    val_losses.append(avg_val_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        # print(f"Epoch {epoch+1:3d}/{n_epochs} - Loss: {avg_loss:.6f}")
        print(f"Epoch {epoch+1:3d}/{n_epochs} - Train Loss: {avg_train_loss:.6f}")
        print(f"Epoch {epoch+1:3d}/{n_epochs} - Val Loss: {avg_val_loss:.6f}")



# Plotting the losses
fig, ax = plt.subplots(figsize=(10, 6))
epochs_range = range(1, n_epochs + 1)

ax.plot(epochs_range, train_losses, label='Train Loss', linewidth=2)
ax.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Evolution des losses pendant l\'entraînement', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# Saving losses graphs
plt.savefig(f"Starting/5 - Diffusion model/results/training_losses_{formatted_now}.png", dpi=150)
plt.show()


# Saving of the model
model_save_path = f"Starting/5 - Diffusion model/trained_models/trained_diffusion_model_with_velocities_{formatted_now}.pt"
torch.save(model.state_dict(), model_save_path)

print(f"Saved model at the following path: {model_save_path}")