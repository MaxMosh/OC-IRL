import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== CONFIG ====================
TRAJECTORY_DIM = 2  # q1 and q2
MAX_LENGTH = 200  # Maximum trajectory length
NOISE_STEPS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== MODEL ARCHITECTURE (same as training) ====================
class RNNBlock(nn.Module):
    """
    RNN block with residual connection and time embedding.
    """
    def __init__(self, dim, time_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=dim, hidden_size=dim, batch_first=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, t):
        """
        Args:
            x: (batch, seq_len, dim)
            t: (batch, time_dim)
        """
        # Add time embedding
        t_emb = self.time_mlp(t)  # (batch, dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, dim)
        
        # Apply RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, dim)
        
        # Residual connection
        h = self.norm(rnn_out)
        h = h + t_emb  # Add time embedding to RNN output
        return x + h

class DiffusionRNN(nn.Module):
    """
    RNN-based diffusion model for trajectory generation.
    """
    def __init__(self, trajectory_dim=TRAJECTORY_DIM, max_length=MAX_LENGTH, 
                 hidden_dim=256, num_layers=6, time_dim=128):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.max_length = max_length
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(trajectory_dim, hidden_dim)
        
        # RNN blocks
        self.blocks = nn.ModuleList([RNNBlock(hidden_dim, time_dim) for _ in range(num_layers)])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, trajectory_dim)
        )
        
    def pos_encoding(self, t, channels):
        """
        Positional encoding for timesteps.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, seq_len, trajectory_dim) - noisy trajectory
            t: (batch,) - timestep
            
        Returns:
            (batch, seq_len, trajectory_dim) - predicted noise
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Process through network
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        for block in self.blocks:
            h = block(h, t_emb)
        
        output = self.output_proj(h)  # (batch, seq_len, trajectory_dim)
        
        return output

# ==================== DIFFUSION PROCESS ====================
class Diffusion:
    def __init__(self, noise_steps=NOISE_STEPS, beta_start=0.0001, beta_end=0.02, 
                 max_length=MAX_LENGTH, device='cuda'):
        """
        Initialize the Diffusion process.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.max_length = max_length
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        """Move tensors to device."""
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self

    def forward_diffusion(self, x, t):
        """
        Add noise to trajectories according to timestep t.
        
        Args:
            x: (batch, max_length, trajectory_dim) - clean trajectory
            t: (batch,) - timestep values
            
        Returns:
            noised_trajectory, noise
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1)
        
        noise = torch.randn_like(x, device=self.device)
        noised_trajectory = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        return noised_trajectory, noise

    def reverse_diffusion(self, model, x, verbose=True):
        """
        Reverse diffusion process to generate trajectories from noise.
        
        Args:
            model: The trained model
            x: Initial noise (batch, max_length, trajectory_dim)
            verbose: Whether to show progress bar
            
        Returns:
            Denoised trajectory
        """
        model.eval()
        
        iterator = reversed(range(1, self.noise_steps))
        if verbose:
            iterator = tqdm(list(iterator), desc="Reverse diffusion")
        
        with torch.no_grad():
            for i in iterator:
                t = torch.full((x.shape[0],), i, device=self.device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x, t)
                
                # Denoise
                sqrt_alpha = torch.sqrt(self.alpha[i])
                one_minus_alpha = 1.0 - self.alpha[i]
                sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[i])
                
                x = (1 / sqrt_alpha) * (x - one_minus_alpha * predicted_noise / sqrt_one_minus_alpha_hat)
                
                # Add noise (except for last step)
                if i > 1:
                    noise = torch.randn_like(x, device=self.device)
                    x = x + torch.sqrt(self.beta[i]) * noise
        
        return x

# ==================== LOAD MODEL ====================
def load_trained_model(model_path, device=DEVICE):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        
    Returns:
        model, config, diffusion
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    
    # Initialize model
    model = DiffusionRNN(
        trajectory_dim=config['trajectory_dim'],
        max_length=config['max_length'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        time_dim=config['time_dim']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize diffusion
    diffusion = Diffusion(
        noise_steps=config['noise_steps'],
        max_length=config['max_length'],
        device=device
    )
    
    print(f"Model loaded successfully!")
    print(f"Configuration: {config}")
    
    return model, config, diffusion

# ==================== LOAD REAL TRAJECTORY ====================
def load_real_trajectory(file_path, max_length=MAX_LENGTH):
    """
    Load a single trajectory from CSV file.
    
    Args:
        file_path: Path to CSV file
        max_length: Maximum length for padding/truncation
        
    Returns:
        trajectory tensor (1, max_length, 2)
    """
    # Load CSV without header, transpose to get (time_steps, 2)
    df = pd.read_csv(file_path, header=None)
    trajectory = df.T.values  # Shape: (time_steps, 2)
    
    original_length = len(trajectory)
    
    # Pad or truncate
    if original_length < max_length:
        padding = np.zeros((max_length - original_length, TRAJECTORY_DIM))
        trajectory = np.vstack([trajectory, padding])
    else:
        trajectory = trajectory[:max_length]
    
    # Convert to tensor and add batch dimension
    trajectory_tensor = torch.from_numpy(trajectory).float().unsqueeze(0)
    
    return trajectory_tensor, original_length

# ==================== TEST 1: GENERATION FROM GAUSSIAN NOISE ====================
def test_generation_from_noise(model, diffusion, num_samples=6, device=DEVICE):
    """
    Test generation of trajectories from Gaussian white noise.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        num_samples: Number of trajectories to generate
        device: Device to run on
    """
    print("\n" + "="*50)
    print("TEST 1: Generation from Gaussian White Noise")
    print("="*50)
    
    model.eval()
    generated_trajectories = []
    
    # Generate trajectories
    with torch.no_grad():
        for i in range(num_samples):
            print(f"\nGenerating trajectory {i+1}/{num_samples}...")
            
            # Start with Gaussian white noise
            x = torch.randn(1, MAX_LENGTH, TRAJECTORY_DIM, device=device)
            print(f"Initial noise - mean: {x.mean():.4f}, std: {x.std():.4f}")
            
            # Reverse diffusion to generate trajectory
            generated = diffusion.reverse_diffusion(model, x, verbose=True)
            
            # Store result
            generated_trajectories.append(generated.squeeze().cpu().numpy())
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        traj = generated_trajectories[i]
        
        axes[i].plot(traj[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[i].plot(traj[:, 1], label='q2', alpha=0.7, linewidth=2)
        axes[i].set_title(f'Generated Trajectory {i+1}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Joint Angle')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Generated Trajectories from Gaussian White Noise', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs("test_plots", exist_ok=True)
    save_path = f"test_plots/generation_from_noise_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    return generated_trajectories

# ==================== TEST 2: DENOISING REAL TRAJECTORY ====================
def test_denoising_real_trajectory(model, diffusion, trajectory_path, device=DEVICE):
    """
    Test denoising process on a real trajectory from the dataset.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        trajectory_path: Path to a real trajectory CSV file
        device: Device to run on
    """
    print("\n" + "="*50)
    print("TEST 2: Denoising Real Trajectory")
    print("="*50)
    
    # Load real trajectory
    print(f"\nLoading trajectory from: {trajectory_path}")
    real_trajectory, original_length = load_real_trajectory(trajectory_path, MAX_LENGTH)
    real_trajectory = real_trajectory.to(device)
    
    print(f"Original trajectory length: {original_length}")
    print(f"Trajectory shape: {real_trajectory.shape}")
    
    # Test denoising at different noise levels
    noise_levels = [20, 40, 60, 80, 99]  # Different timesteps to add noise
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(len(noise_levels), 3, figsize=(15, 4*len(noise_levels)))
    
    model.eval()
    
    for idx, t_value in enumerate(noise_levels):
        print(f"\nTesting noise level t={t_value}...")
        
        # Create timestep tensor
        t = torch.tensor([t_value], device=device)
        
        # Forward diffusion: add noise to real trajectory
        noised_trajectory, true_noise = diffusion.forward_diffusion(real_trajectory, t)
        print(f"Noised trajectory - mean: {noised_trajectory.mean():.4f}, std: {noised_trajectory.std():.4f}")
        
        # Reverse diffusion: denoise the trajectory
        with torch.no_grad():
            denoised = noised_trajectory.clone()
            
            for i in tqdm(reversed(range(1, t_value + 1)), desc=f"Denoising from t={t_value}"):
                t_reverse = torch.tensor([i], device=device)
                predicted_noise = model(denoised, t_reverse)
                
                sqrt_alpha = torch.sqrt(diffusion.alpha[i])
                one_minus_alpha = 1.0 - diffusion.alpha[i]
                sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - diffusion.alpha_hat[i])
                
                denoised = (1 / sqrt_alpha) * (denoised - one_minus_alpha * predicted_noise / sqrt_one_minus_alpha_hat)
                
                if i > 1:
                    noise_term = torch.randn_like(denoised, device=device)
                    denoised = denoised + torch.sqrt(diffusion.beta[i]) * noise_term
        
        # Convert to numpy for plotting
        original = real_trajectory[0].cpu().numpy()
        noised = noised_trajectory[0].cpu().numpy()
        denoised_np = denoised[0].cpu().numpy()
        
        # Calculate reconstruction error
        mse = np.mean((original - denoised_np) ** 2)
        print(f"Reconstruction MSE: {mse:.6f}")
        
        # Plot original trajectory
        axes[idx, 0].plot(original[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[idx, 0].plot(original[:, 1], label='q2', alpha=0.7, linewidth=2)
        axes[idx, 0].set_title('Original Trajectory', fontweight='bold')
        axes[idx, 0].set_ylabel(f't={t_value}', fontweight='bold', fontsize=11)
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot noised trajectory
        axes[idx, 1].plot(noised[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[idx, 1].plot(noised[:, 1], label='q2', alpha=0.7, linewidth=2)
        axes[idx, 1].set_title(f'Noised (t={t_value})', fontweight='bold')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Plot denoised trajectory
        axes[idx, 2].plot(denoised_np[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[idx, 2].plot(denoised_np[:, 1], label='q2', alpha=0.7, linewidth=2)
        axes[idx, 2].set_title(f'Denoised (MSE={mse:.4f})', fontweight='bold')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Denoising Test on Real Trajectory', fontsize=14, fontweight='bold')
    
    # Set common x-labels for bottom row
    for ax in axes[-1]:
        ax.set_xlabel('Time Step')
    
    plt.tight_layout()
    
    # Save plot
    save_path = f"test_plots/denoising_real_trajectory_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()

# ==================== LOAD DATASET ====================
def load_dataset_trajectories(base_dir="data", max_length=MAX_LENGTH, max_trajectories=None):
    """
    Load all trajectories from the dataset.
    
    Args:
        base_dir: Base directory containing S01-S20 folders
        max_length: Maximum length for padding/truncation
        max_trajectories: Maximum number of trajectories to load (None for all)
        
    Returns:
        List of trajectory tensors
    """
    trajectories = []
    file_paths = []
    
    # Collect all file paths
    for i in range(1, 21):  # S01 to S20
        subject_dir = os.path.join(base_dir, f"S{i:02d}")
        
        if not os.path.exists(subject_dir):
            continue
        
        csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_paths.append(os.path.join(subject_dir, csv_file))
    
    # Limit number of trajectories if specified
    if max_trajectories is not None:
        file_paths = file_paths[:max_trajectories]
    
    print(f"Loading {len(file_paths)} trajectories from dataset...")
    
    # Load trajectories
    for path in tqdm(file_paths, desc="Loading"):
        try:
            df = pd.read_csv(path, header=None)
            trajectory = df.T.values
            
            original_length = len(trajectory)
            
            if original_length < max_length:
                padding = np.zeros((max_length - original_length, TRAJECTORY_DIM))
                trajectory = np.vstack([trajectory, padding])
            else:
                trajectory = trajectory[:max_length]
            
            trajectories.append(torch.from_numpy(trajectory).float())
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return trajectories

def compute_nearest_neighbor(generated_traj, dataset_trajs):
    """
    Find the nearest neighbor in the dataset for a generated trajectory.
    
    Args:
        generated_traj: Generated trajectory (max_length, 2)
        dataset_trajs: List of dataset trajectories
        
    Returns:
        nearest_traj, min_distance, nearest_idx
    """
    min_distance = float('inf')
    nearest_traj = None
    nearest_idx = -1
    
    for idx, dataset_traj in enumerate(dataset_trajs):
        # Compute MSE distance
        distance = torch.mean((generated_traj - dataset_traj) ** 2).item()
        
        if distance < min_distance:
            min_distance = distance
            nearest_traj = dataset_traj
            nearest_idx = idx
    
    return nearest_traj, min_distance, nearest_idx

# ==================== TEST 3: NEAREST NEIGHBOR ANALYSIS ====================
def test_nearest_neighbors(model, diffusion, dataset_trajs, num_samples=6, device=DEVICE):
    """
    Generate trajectories and compare with nearest neighbors in the dataset.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        dataset_trajs: List of trajectories from training dataset
        num_samples: Number of trajectories to generate
        device: Device to run on
    """
    print("\n" + "="*50)
    print("TEST 3: Nearest Neighbor Analysis")
    print("="*50)
    
    model.eval()
    generated_trajectories = []
    nearest_neighbors = []
    distances = []
    
    # Generate trajectories
    with torch.no_grad():
        for i in range(num_samples):
            print(f"\nGenerating trajectory {i+1}/{num_samples}...")
            
            # Start with Gaussian white noise
            x = torch.randn(1, MAX_LENGTH, TRAJECTORY_DIM, device=device)
            
            # Reverse diffusion
            generated = diffusion.reverse_diffusion(model, x, verbose=True)
            generated_np = generated.squeeze().cpu()
            
            # Find nearest neighbor
            print("Finding nearest neighbor in dataset...")
            nearest_traj, min_dist, nn_idx = compute_nearest_neighbor(generated_np, dataset_trajs)
            
            generated_trajectories.append(generated_np.numpy())
            nearest_neighbors.append(nearest_traj.numpy())
            distances.append(min_dist)
            
            print(f"Nearest neighbor: trajectory #{nn_idx}, MSE distance: {min_dist:.6f}")
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        gen_traj = generated_trajectories[i]
        nn_traj = nearest_neighbors[i]
        
        # Plot generated trajectory
        axes[i, 0].plot(gen_traj[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[i, 0].plot(gen_traj[:, 1], label='q2', alpha=0.7, linewidth=2)
        axes[i, 0].set_title(f'Generated Trajectory {i+1}', fontsize=12, fontweight='bold')
        axes[i, 0].set_ylabel('Joint Angle')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot nearest neighbor
        axes[i, 1].plot(nn_traj[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[i, 1].plot(nn_traj[:, 1], label='q2', alpha=0.7, linewidth=2)
        axes[i, 1].set_title(f'Nearest Neighbor (MSE={distances[i]:.4f})', 
                            fontsize=12, fontweight='bold')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    # Set x-labels for bottom row
    axes[-1, 0].set_xlabel('Time Step')
    axes[-1, 1].set_xlabel('Time Step')
    
    plt.suptitle('Generated vs Nearest Neighbor in Training Dataset', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs("test_plots", exist_ok=True)
    save_path = f"test_plots/nearest_neighbors_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print("\n" + "-"*50)
    print("Nearest Neighbor Statistics:")
    print(f"Mean MSE distance: {np.mean(distances):.6f}")
    print(f"Std MSE distance: {np.std(distances):.6f}")
    print(f"Min MSE distance: {np.min(distances):.6f}")
    print(f"Max MSE distance: {np.max(distances):.6f}")
    print("-"*50)

# ==================== TEST 4: INTERPOLATION IN NOISE SPACE ====================
def test_noise_interpolation(model, diffusion, dataset_trajs, num_interpolations=5, device=DEVICE):
    """
    Test interpolation between two fully noised trajectories.
    
    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        dataset_trajs: List of trajectories from training dataset
        num_interpolations: Number of interpolation steps (including endpoints)
        device: Device to run on
    """
    print("\n" + "="*50)
    print("TEST 4: Noise Space Interpolation")
    print("="*50)
    
    # Randomly select two trajectories from dataset
    idx1, idx2 = np.random.choice(len(dataset_trajs), size=2, replace=False)
    traj1 = dataset_trajs[idx1].unsqueeze(0).to(device)
    traj2 = dataset_trajs[idx2].unsqueeze(0).to(device)
    
    print(f"\nSelected trajectory 1: index {idx1}")
    print(f"Selected trajectory 2: index {idx2}")
    
    # Fully noise both trajectories (t = noise_steps - 1)
    t_max = torch.tensor([diffusion.noise_steps - 1], device=device)
    
    print(f"\nNoising trajectories to t={diffusion.noise_steps - 1}...")
    noised_traj1, noise1 = diffusion.forward_diffusion(traj1, t_max)
    noised_traj2, noise2 = diffusion.forward_diffusion(traj2, t_max)
    
    print(f"Noised traj 1 - mean: {noised_traj1.mean():.4f}, std: {noised_traj1.std():.4f}")
    print(f"Noised traj 2 - mean: {noised_traj2.mean():.4f}, std: {noised_traj2.std():.4f}")
    
    # Create interpolations in noise space
    alphas = np.linspace(0, 1, num_interpolations)
    interpolated_trajectories = []
    
    model.eval()
    
    for i, alpha in enumerate(alphas):
        print(f"\n{'='*40}")
        print(f"Interpolation {i+1}/{num_interpolations} (alpha={alpha:.2f})")
        print(f"{'='*40}")
        
        # Linear interpolation in noise space
        # To ensure the result is centered and normalized, we need to renormalize
        interpolated_noise = alpha * noised_traj1 + (1 - alpha) * noised_traj2
        
        # Renormalize to ensure the interpolated noise has standard Gaussian properties
        # This is important to maintain the distribution expected by the model
        noise_mean = interpolated_noise.mean()
        noise_std = interpolated_noise.std()
        
        # Standardize: (x - mean) / std to get N(0,1) distribution
        interpolated_noise = (interpolated_noise - noise_mean) / (noise_std + 1e-8)
        
        print(f"Interpolated noise - mean: {interpolated_noise.mean():.4f}, std: {interpolated_noise.std():.4f}")
        
        # Denoise the interpolated noise
        with torch.no_grad():
            denoised = diffusion.reverse_diffusion(model, interpolated_noise, verbose=True)
        
        interpolated_trajectories.append(denoised.squeeze().cpu().numpy())
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a large figure with all interpolations
    fig, axes = plt.subplots(num_interpolations, 1, figsize=(12, 4*num_interpolations))
    
    if num_interpolations == 1:
        axes = [axes]
    
    for i, (traj, alpha) in enumerate(zip(interpolated_trajectories, alphas)):
        axes[i].plot(traj[:, 0], label='q1', alpha=0.7, linewidth=2)
        axes[i].plot(traj[:, 1], label='q2', alpha=0.7, linewidth=2)
        
        if i == 0:
            title = f'Start: Trajectory A (alpha=0.00)'
        elif i == num_interpolations - 1:
            title = f'End: Trajectory B (alpha=1.00)'
        else:
            title = f'Interpolation {i} (alpha={alpha:.2f})'
        
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Joint Angle')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    
    plt.suptitle('Interpolation in Fully Noised Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = f"test_plots/noise_interpolation_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()
    
    # Also create a comparison plot showing original trajectories
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot original trajectory 1
    orig1 = traj1.squeeze().cpu().numpy()
    axes2[0].plot(orig1[:, 0], label='q1', alpha=0.7, linewidth=2)
    axes2[0].plot(orig1[:, 1], label='q2', alpha=0.7, linewidth=2)
    axes2[0].set_title(f'Original Trajectory A (idx={idx1})', fontsize=12, fontweight='bold')
    axes2[0].set_xlabel('Time Step')
    axes2[0].set_ylabel('Joint Angle')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Plot original trajectory 2
    orig2 = traj2.squeeze().cpu().numpy()
    axes2[1].plot(orig2[:, 0], label='q1', alpha=0.7, linewidth=2)
    axes2[1].plot(orig2[:, 1], label='q2', alpha=0.7, linewidth=2)
    axes2[1].set_title(f'Original Trajectory B (idx={idx2})', fontsize=12, fontweight='bold')
    axes2[1].set_xlabel('Time Step')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.suptitle('Original Trajectories Used for Interpolation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path2 = f"test_plots/noise_interpolation_originals_{timestamp}.png"
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Original trajectories plot saved to: {save_path2}")
    plt.show()

# ==================== MAIN TEST FUNCTION ====================
def main():
    """
    Main test function - runs all tests.
    """
    print("="*50)
    print("DIFFUSION MODEL TESTING")
    print("="*50)
    print(f"Device: {DEVICE}")
    
    # ========== LOAD MODEL ==========
    # Update this path to your trained model
    model_path = "trained_models/diffusion_mlp_20251110_161646.pth"  # UPDATE THIS!
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please update the model_path variable with your trained model path.")
        return
    
    model, config, diffusion = load_trained_model(model_path, device=DEVICE)
    
    # ========== LOAD DATASET FOR TESTS 3 & 4 ==========
    print("\n" + "="*50)
    print("Loading dataset for nearest neighbor and interpolation tests...")
    print("="*50)
    dataset_trajs = load_dataset_trajectories(base_dir="data", max_length=MAX_LENGTH, 
                                               max_trajectories=500)  # Limit for speed
    print(f"Loaded {len(dataset_trajs)} trajectories from dataset")
    
    # ========== TEST 1: GENERATION FROM NOISE ==========
    generated_trajectories = test_generation_from_noise(
        model, diffusion, num_samples=6, device=DEVICE
    )
    
    # ========== TEST 2: DENOISING REAL TRAJECTORY ==========
    # Update this path to a real trajectory file from your dataset
    trajectory_path = "data/S01/Trial00.csv"  # UPDATE THIS!
    
    if not os.path.exists(trajectory_path):
        print(f"\nWARNING: Trajectory file not found at {trajectory_path}")
        print("Skipping denoising test. Please update the trajectory_path variable.")
    else:
        test_denoising_real_trajectory(
            model, diffusion, trajectory_path, device=DEVICE
        )
    
    # ========== TEST 3: NEAREST NEIGHBOR ANALYSIS ==========
    test_nearest_neighbors(
        model, diffusion, dataset_trajs, num_samples=6, device=DEVICE
    )
    
    # ========== TEST 4: NOISE SPACE INTERPOLATION ==========
    test_noise_interpolation(
        model, diffusion, dataset_trajs, num_interpolations=5, device=DEVICE
    )
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print("\nGenerated plots saved in: test_plots/")

if __name__ == "__main__":
    main()