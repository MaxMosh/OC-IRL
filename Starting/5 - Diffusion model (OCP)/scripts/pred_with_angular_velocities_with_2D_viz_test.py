import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Functions
def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)           # MODIFIED: ADDED .to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TemporalDiffusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.gru = nn.GRU(input_size=dim, hidden_size=128, batch_first=True)
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
        t_emb = timestep_embedding(t, x_t.shape[-1]).to(x_t.device)
        t_emb = t_emb[:, None, :].repeat(1, x_t.shape[1], 1)
        h, _ = self.gru(x_t)
        inp = torch.cat([h, t_emb], dim=-1)
        return self.mlp(inp)


def forward_kinematics_2R(q, L1=1.0, L2=1.0):
    """Direct kinematics of the 2 DoFs planar robot"""
    q1 = q[:, 0]
    q2 = q[:, 1]
    
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    
    return np.column_stack([x1, y1]), np.column_stack([x2, y2])


# Load stats and model
# TODO: change the path of the model and the stats used
model_path = "Starting/5 - Diffusion model/trained_models/trained_diffusion_model_with_velocities_2025-10-30_16:29.pt"

stats = np.load("Starting/5 - Diffusion model/data/stats_with_velocities.npy", allow_pickle=True).item()
mean, std = stats["mean"], stats["std"]

print(f"Loaded stats:")
print(f"Mean : {mean}")
print(f"Std  : {std}")

device = "cuda" if torch.cuda.is_available() else "cpu"

timesteps = 1000                                                                            # MODIFIED: INITIALLY 1000
betas = torch.linspace(1e-5, 1e-3, timesteps).to(device)                                    # MODIFIED: ADDED .to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# device = "cuda" if torch.cuda.is_available() else "cpu"
model = TemporalDiffusion(dim=4).to(device)  # dim=4 for [q1, q2, dq1, dq2]
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model, using the device: {device}")


# Sampling with prefix
n_samples = 50 #TODO: ADDED, NEED TO SEE IF IT WORKS WITH ELSE THAN 50                      # MODIFIED: INITIALLY 50
@torch.no_grad()
def sample_with_prefix(model, prefix, n_total, n_samples=n_samples, T=timesteps):           # MODIFIED: T from 1000 to timesteps
    """Generates n_samples trajectories completing the given prefix"""
    prefix = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(device)
    n_known = prefix.shape[1]

    all_samples = []
    for s in range(n_samples):
        # Initialization with Gaussian noise
        x = torch.randn((1, n_total, 4), device=device)
        x[:, :n_known] = prefix

        # Reverse diffusion process
        for t in reversed(range(T)):
            a = alphas[t]
            a_bar = alphas_cumprod[t]
            t_tensor = torch.tensor([t], device=device)
            
            # Noise prediction
            eps = model(x, t_tensor)
            
            # Denoising
            z = torch.randn_like(x) if t > 0 else 0
            x = (1/torch.sqrt(a)) * (x - (1 - a)/torch.sqrt(1 - a_bar) * eps) + torch.sqrt(betas[t]) * z
            
            # Keep the prefix constant
            x[:, :n_known] = prefix
            
        all_samples.append(x.cpu().squeeze(0).numpy())

    return np.array(all_samples)


# Trajectories generation
data = np.load("Starting/5 - Diffusion model/data/trajectories_dataset_with_velocities_test.npy", 
               allow_pickle=True)
example_traj = data[0]["future"]  # shape (N, 4)

# Normalization
example_traj_norm = (example_traj - mean) / (std + 1e-8)

n_prefix = 30                                                                               # MODIFIED, INITIALLY 30
prefix = example_traj_norm[:n_prefix]
n_total = len(example_traj)

print(f"Prefix: {n_prefix} timesteps")
print(f"Total to generate: {n_total} timesteps")
print(f"\nGeneration of {n_samples} samples...")

samples = sample_with_prefix(model, prefix, n_total=n_total, n_samples=50)
samples = samples * std + mean  # denormalizing

print(f"\nEnd of trajectory generation...")

mean_traj = samples.mean(axis=0)
std_traj = samples.std(axis=0)

# Visualization
print("Graphs generation...")
time = np.arange(n_total)
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
titles = [r"$q_1(t)$", r"$q_2(t)$", r"$\dot{q}_1(t)$", r"$\dot{q}_2(t)$"]
colors = ['orange', 'green', 'purple', 'brown']

for i, (ax, title, color) in enumerate(zip(axs, titles, colors)):
    # Zone d'incertitude
    ax.fill_between(
        time[n_prefix:],
        mean_traj[n_prefix:, i] - 2 * std_traj[n_prefix:, i],
        mean_traj[n_prefix:, i] + 2 * std_traj[n_prefix:, i],
        color=color, alpha=0.3, label=r"Probable zone (+/-2$\sigma$)"
    )
    
    # Trajectoire générée (moyenne)
    ax.plot(time[n_prefix:], mean_traj[n_prefix:, i], 
            color=color, linewidth=2, label="Generated mean")
    
    # Given prefix
    ax.plot(time[:n_prefix], prefix[:, i]*std[i]+mean[i], 
            color='blue', linewidth=2.5, label="Given prefix")
    
    # DOC trajectory
    ax.plot(time, example_traj[:, i], "--", 
            color="black", alpha=0.6, linewidth=1.5, label="DOC trajectory")
    
    # Vertical line to separate prefix and generated trajectories
    ax.axvline(x=n_prefix, color="gray", linestyle="--", alpha=0.7)
    
    ax.set_ylabel(title, fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

axs[-1].set_xlabel("Time (discretized)", fontsize=12)
plt.suptitle("Trajectory completion with angles and angles velocity", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("Starting/5 - Diffusion model/results/temporal_plots_with_velocities.png", dpi=150)
plt.show()


# Visualization of the trajectory in 2D
# Angles
angles_true = example_traj[:, :2]
angles_gen = mean_traj[:, :2]
angles_prefix = (prefix * std + mean)[:, :2]

# Computing robot position
elbow_true, end_true = forward_kinematics_2R(angles_true)
elbow_gen, end_gen = forward_kinematics_2R(angles_gen)
elbow_prefix, end_prefix = forward_kinematics_2R(angles_prefix)

fig, ax = plt.subplots(figsize=(10, 10))

# End-effector trajectory
ax.plot(end_true[:, 0], end_true[:, 1], '--', color='black', 
        alpha=0.6, linewidth=2, label='DOC trajectory')
ax.plot(end_prefix[:, 0], end_prefix[:, 1], color='blue', 
        linewidth=3, label='Given prefix')
ax.plot(end_gen[n_prefix:, 0], end_gen[n_prefix:, 1], color='orange', 
        linewidth=2, label='Generated mean')

# Display serveral robot configuration through time
step = max(1, n_total // 10)
for i in range(0, n_total, step):
    alpha_val = 0.3 if i < n_prefix else 0.5
    color = 'blue' if i < n_prefix else 'orange'
    
    ax.plot([0, elbow_gen[i, 0], end_gen[i, 0]], 
            [0, elbow_gen[i, 1], end_gen[i, 1]], 
            'o-', color=color, alpha=alpha_val, markersize=4)

# Initial and final configurations
ax.plot([0, elbow_gen[0, 0], end_gen[0, 0]], 
        [0, elbow_gen[0, 1], end_gen[0, 1]], 
        'o-', color='green', linewidth=2.5, markersize=10, label='Start')
ax.plot([0, elbow_gen[-1, 0], end_gen[-1, 0]], 
        [0, elbow_gen[-1, 1], end_gen[-1, 1]], 
        'o-', color='red', linewidth=2.5, markersize=10, label='End')

ax.plot(0, 0, 'ko', markersize=12, label='Base', zorder=10)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Robot trajectories (angles and angles velocities)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("Starting/5 - Diffusion model/results/robot_2D_trajectory_with_velocities.png", dpi=150)
plt.show()


# GIF creation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Robot animation
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 2.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('x (m)', fontsize=11)
ax1.set_ylabel('y (m)', fontsize=11)
ax1.set_title('Robot configuration', fontsize=12, fontweight='bold')

line_robot, = ax1.plot([], [], 'o-', color='orange', linewidth=3, markersize=10)
line_trail, = ax1.plot([], [], '-', color='orange', alpha=0.5, linewidth=1.5)
line_prefix, = ax1.plot(end_prefix[:, 0], end_prefix[:, 1], 
                         color='blue', linewidth=2, label='Prefix')
line_doc, = ax1.plot(end_true[:, 0], end_true[:, 1], 
                      '--', color='black', alpha=0.4, label='DOC')
ax1.plot(0, 0, 'ko', markersize=12, label='Base', zorder=10)
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                      fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax1.legend(loc='upper right', fontsize=9)

# Angular velocities animation
ax2.set_xlim(0, n_total)
ax2.set_ylim(mean_traj[:, 2:4].min() - 0.5, mean_traj[:, 2:4].max() + 0.5)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Time (discretized)', fontsize=11)
ax2.set_ylabel('Angular velocity (rad/s)', fontsize=11)
ax2.set_title('Real time angular velocities', fontsize=12, fontweight='bold')

line_dq1, = ax2.plot([], [], color='purple', linewidth=2, label=r'$\dot{q}_1$')
line_dq2, = ax2.plot([], [], color='brown', linewidth=2, label=r'$\dot{q}_2$')
ax2.axvline(x=n_prefix, color="gray", linestyle="--", alpha=0.7, label='Prefix end')
ax2.legend(loc='upper right', fontsize=9)

trail_x, trail_y = [], []
time_history = []
dq1_history, dq2_history = [], []

def init():
    line_robot.set_data([], [])
    line_trail.set_data([], [])
    line_dq1.set_data([], [])
    line_dq2.set_data([], [])
    time_text.set_text('')
    return line_robot, line_trail, line_dq1, line_dq2, time_text

def animate(frame):
    # Robot animation
    x_vals = [0, elbow_gen[frame, 0], end_gen[frame, 0]]
    y_vals = [0, elbow_gen[frame, 1], end_gen[frame, 1]]
    line_robot.set_data(x_vals, y_vals)
    
    trail_x.append(end_gen[frame, 0])
    trail_y.append(end_gen[frame, 1])
    line_trail.set_data(trail_x, trail_y)
    
    # Color depending on prefix or generated
    if frame < n_prefix:
        line_robot.set_color('blue')
        time_text.set_text(f'Frame: {frame}/{n_total} (Prefix)')
    else:
        line_robot.set_color('orange')
        time_text.set_text(f'Frame: {frame}/{n_total} (Generated)')
    
    # Angular velocities
    time_history.append(frame)
    dq1_history.append(mean_traj[frame, 2])
    dq2_history.append(mean_traj[frame, 3])
    
    line_dq1.set_data(time_history, dq1_history)
    line_dq2.set_data(time_history, dq2_history)
    
    return line_robot, line_trail, line_dq1, line_dq2, time_text

# Animation creation
frames = range(0, n_total, 2)
anim = FuncAnimation(fig, animate, init_func=init, frames=frames, 
                     interval=50, blit=True, repeat=True)

# Saving the GIF animation
writer = PillowWriter(fps=20)
anim.save("Starting/5 - Diffusion model/results/robot_animation_with_velocities.gif", 
          writer=writer)

plt.show()

print("All graphs generated.")