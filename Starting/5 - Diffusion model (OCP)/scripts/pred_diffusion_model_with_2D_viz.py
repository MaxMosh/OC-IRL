import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Functions
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
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


# 2 DoFs robot direct kinematic function
def forward_kinematics_2R(q, L1=1.0, L2=1.0):
    """
    Calculates the position of the effector for a 2 DoFs planar robot
    q: array of shape (N, 2) with [q1, q2] for N timesteps
    L1, L2: lenghts of the arms
    Returns: positions (N, 2) of effectors [x, y]
    """
    q1 = q[:, 0]
    q2 = q[:, 1]
    
    # Position du coude (joint 1)
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    
    # Position de l'effecteur (bout du bras 2)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    
    return np.column_stack([x1, y1]), np.column_stack([x2, y2])


# Loading stats and model
model_path = "Starting/5 - Diffusion model/trained_models/trained_diffusion_model_2025-10-29_11:02.pt"
stats = np.load("Starting/5 - Diffusion model/data/stats.npy", allow_pickle=True).item()
mean, std = stats["mean"], stats["std"]

timesteps = 1000
betas = torch.linspace(1e-5, 1e-3, timesteps)
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


# Trajectory generation
data = np.load("Starting/5 - Diffusion model/data/trajectories_dataset.npy", allow_pickle=True)
example_traj = data[0]["future"]
example_traj_norm = (example_traj - mean) / (std + 1e-8)

n_prefix = 80
prefix = example_traj_norm[:n_prefix]
n_total = len(example_traj)

print("Generating trajectory...")
samples = sample_with_prefix(model, prefix, n_total=n_total, n_samples=50)
samples = samples * std + mean  # denormalizing
mean_traj = samples.mean(axis=0)
std_traj = samples.std(axis=0)
print("End of generation...")

# Visualization
print("Graphs generation...")
time = np.arange(n_total)
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
titles = [r"$q_1(t)$", r"$q_2(t)$"]

for i, ax in enumerate(axs):
    ax.fill_between(
        time[n_prefix:],
        mean_traj[n_prefix:, i] - 2 * std_traj[n_prefix:, i],
        mean_traj[n_prefix:, i] + 2 * std_traj[n_prefix:, i],
        color="orange", alpha=0.3, label=r"Probable zone (+/-2$\sigma$)"
    )
    ax.plot(time[n_prefix:], mean_traj[n_prefix:, i], color="orange", label="Generated mean")
    ax.plot(time[:n_prefix], prefix[:, i]*std[i]+mean[i], color="blue", linewidth=2, label="Given prefix")
    ax.plot(time, example_traj[:, i], "--", color="black", alpha=0.6, label="DOC trajectory")
    ax.axvline(x=n_prefix, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel(titles[i])
    ax.legend()

axs[-1].set_xlabel("Time (discretized)")
plt.suptitle("Trajecotry completion using time-based diffusion (GRU)")
plt.tight_layout()
plt.savefig("Starting/5 - Diffusion model/results/temporal_plots.png", dpi=150)
plt.show()

# 2D visualization
# Computing robot positions
elbow_true, end_true = forward_kinematics_2R(example_traj)
elbow_gen, end_gen = forward_kinematics_2R(mean_traj)
elbow_prefix, end_prefix = forward_kinematics_2R(prefix * std + mean)

fig, ax = plt.subplots(figsize=(8, 8))

# End-effector trajectories
ax.plot(end_true[:, 0], end_true[:, 1], '--', color='black', alpha=0.6, 
        linewidth=2, label='DOC trajectory')
ax.plot(end_prefix[:, 0], end_prefix[:, 1], color='blue', 
        linewidth=3, label='Given prefix')
ax.plot(end_gen[n_prefix:, 0], end_gen[n_prefix:, 1], color='orange', 
        linewidth=2, label='Generated mean')

# Display several robot configuration through time
step = max(1, n_total // 10)
for i in range(0, n_total, step):
    alpha_val = 0.3 if i < n_prefix else 0.5
    color = 'blue' if i < n_prefix else 'orange'
    
    # Arms drawing
    ax.plot([0, elbow_gen[i, 0], end_gen[i, 0]], 
            [0, elbow_gen[i, 1], end_gen[i, 1]], 
            'o-', color=color, alpha=alpha_val, markersize=4)

# Initial and final configuration
ax.plot([0, elbow_gen[0, 0], end_gen[0, 0]], 
        [0, elbow_gen[0, 1], end_gen[0, 1]], 
        'o-', color='green', linewidth=2, markersize=8, label='Start')
ax.plot([0, elbow_gen[-1, 0], end_gen[-1, 0]], 
        [0, elbow_gen[-1, 1], end_gen[-1, 1]], 
        'o-', color='red', linewidth=2, markersize=8, label='End')

ax.plot(0, 0, 'ko', markersize=10, label='Base')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('2D robot trajectory')
plt.tight_layout()
plt.savefig("Starting/5 - Diffusion model/results/robot_2D_trajectory.png", dpi=150)
plt.show()

# Robot animation

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# Objects to animate
line_robot, = ax.plot([], [], 'o-', color='orange', linewidth=3, markersize=8)
line_trail, = ax.plot([], [], '-', color='orange', alpha=0.5, linewidth=1)
line_prefix, = ax.plot(end_prefix[:, 0], end_prefix[:, 1], 
                        color='blue', linewidth=2, label='Prefix')
line_doc, = ax.plot(end_true[:, 0], end_true[:, 1], 
                     '--', color='black', alpha=0.4, label='DOC')
ax.plot(0, 0, 'ko', markersize=10, label='Base')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
ax.legend(loc='upper right')

trail_x, trail_y = [], []

def init():
    line_robot.set_data([], [])
    line_trail.set_data([], [])
    time_text.set_text('')
    return line_robot, line_trail, time_text

def animate(frame):
    # Robot configuration at the current frame
    x_vals = [0, elbow_gen[frame, 0], end_gen[frame, 0]]
    y_vals = [0, elbow_gen[frame, 1], end_gen[frame, 1]]
    line_robot.set_data(x_vals, y_vals)
    
    # Trail of end-effector
    trail_x.append(end_gen[frame, 0])
    trail_y.append(end_gen[frame, 1])
    line_trail.set_data(trail_x, trail_y)
    
    # Color change after prefix
    if frame < n_prefix:
        line_robot.set_color('blue')
        time_text.set_text(f'Frame: {frame}/{n_total} (Prefix)')
    else:
        line_robot.set_color('orange')
        time_text.set_text(f'Frame: {frame}/{n_total} (Generated)')
    
    return line_robot, line_trail, time_text

# Animation creation
frames = range(0, n_total, 2)
anim = FuncAnimation(fig, animate, init_func=init, frames=frames, 
                     interval=50, blit=True, repeat=True)

# Saving the GIF animation
writer = PillowWriter(fps=20)
anim.save("Starting/5 - Diffusion model/results/robot_animation.gif", writer=writer)

plt.show()

print("All graphs generated.")