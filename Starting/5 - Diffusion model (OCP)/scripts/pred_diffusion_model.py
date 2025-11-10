import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Functions
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)          # MODIFIED: ADDED .to(timesteps.device)
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


# Load stats and model
model_path = "Starting/5 - Diffusion model/trained_models/trained_diffusion_model_2025-10-30_13:54.pt"
stats = np.load("Starting/5 - Diffusion model/data/stats.npy", allow_pickle=True).item()
mean, std = stats["mean"], stats["std"]

device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps = 1000
betas = torch.linspace(1e-5, 1e-3, timesteps).to(device)                                    # MODIFIED: ADDED .to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TemporalDiffusion(dim=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Sampling with prefix
@torch.no_grad()
def sample_with_prefix(model, prefix, n_total, n_samples=50, T=1000):
    prefix = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(device)
    n_known = prefix.shape[1]

    all_samples = []
    for s in range(n_samples):
        x = torch.randn((1, n_total, 2), device=device)
        x[:, :n_known] = prefix

        for t in reversed(range(T)):
            a = alphas[t]
            a_bar = alphas_cumprod[t]
            t_tensor = torch.tensor([t], device=device)
            eps = model(x, t_tensor)
            z = torch.randn_like(x) if t > 0 else 0
            x = (1/torch.sqrt(a)) * (x - (1 - a)/torch.sqrt(1 - a_bar) * eps) + torch.sqrt(betas[t]) * z
            x[:, :n_known] = prefix
        all_samples.append(x.cpu().squeeze(0).numpy())

    return np.array(all_samples)


# Using on a trajectory
data = np.load("Starting/5 - Diffusion model/data/trajectories_dataset.npy", allow_pickle=True)
example_traj = data[0]["future"]
example_traj = (example_traj - mean) / (std + 1e-8)  # normalizing with same mean and std as in the training

n_prefix = 30
prefix = example_traj[:n_prefix]
n_total = len(example_traj)

print("Génération en cours...")
samples = sample_with_prefix(model, prefix, n_total=n_total, n_samples=50)
samples = samples * std + mean  # denormalizing
mean_traj = samples.mean(axis=0)
std_traj = samples.std(axis=0)

example_traj = example_traj * std + mean  # true trajectory

# Visualization
time = np.arange(n_total)
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
titles = [r"$q_1(t)$", r"$q_2(t)$"]

for i, ax in enumerate(axs):
    ax.fill_between(
        time[n_prefix:],
        mean_traj[n_prefix:, i] - 2 * std_traj[n_prefix:, i],
        mean_traj[n_prefix:, i] + 2 * std_traj[n_prefix:, i],
        color="orange", alpha=0.3, label="Probable zone (+/-2$\sigma$)"
    )
    ax.plot(time[n_prefix:], mean_traj[n_prefix:, i], color="orange", label="Generated mean")
    ax.plot(time[:n_prefix], prefix[:, i]*std[i]+mean[i], color="blue", linewidth=2, label="Given prefix")
    ax.plot(time, example_traj[:, i], "--", color="black", alpha=0.6, label="DOC trajectory")
    ax.axvline(x=n_prefix, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel(titles[i])
    ax.legend()

axs[-1].set_xlabel("Time (discretized)")
plt.suptitle("Trajectory completion with time-based diffusion (GRU)")
plt.tight_layout()
plt.show()
