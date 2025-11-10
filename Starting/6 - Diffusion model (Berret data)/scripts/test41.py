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
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print("\nGenerated plots saved in: test_plots/")

if __name__ == "__main__":
    main()