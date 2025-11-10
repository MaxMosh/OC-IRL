import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== CONFIG ====================
TRAJECTORY_DIM = 2  # q1 and q2
MAX_LENGTH = 200  # Maximum trajectory length (adjust based on your data)
NOISE_STEPS = 100
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== DATA LOADING ====================
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, max_length=MAX_LENGTH):
        """
        Load trajectory data from multiple CSV files.
        
        Args:
            data_paths: List of paths to CSV files
            max_length: Maximum length for padding/truncation
        """
        self.trajectories = []
        self.lengths = []
        self.max_length = max_length
        
        for path in tqdm(data_paths, desc="Loading trajectories"):
            try:
                # Load CSV without header, transpose to get (time_steps, 2)
                df = pd.read_csv(path, header=None)
                trajectory = df.T.values  # Shape: (time_steps, 2)
                
                original_length = len(trajectory)
                
                # Pad or truncate
                if original_length < max_length:
                    # Pad with zeros
                    padding = np.zeros((max_length - original_length, TRAJECTORY_DIM))
                    trajectory = np.vstack([trajectory, padding])
                    actual_length = original_length
                else:
                    # Truncate
                    trajectory = trajectory[:max_length]
                    actual_length = max_length
                
                self.trajectories.append(trajectory)
                self.lengths.append(actual_length)
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        self.trajectories = np.array(self.trajectories, dtype=np.float32)
        self.lengths = np.array(self.lengths, dtype=np.int32)
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.trajectories[idx]),
            torch.tensor(self.lengths[idx])
        )

def load_all_trajectories(base_dir="data"):
    """
    Load all trajectory files from S01 to S20 directories.
    
    Args:
        base_dir: Base directory containing S01-S20 folders
        
    Returns:
        List of file paths
    """
    file_paths = []
    
    for i in range(1, 21):  # S01 to S20
        subject_dir = os.path.join(base_dir, f"S{i:02d}")
        
        if not os.path.exists(subject_dir):
            print(f"Warning: {subject_dir} does not exist")
            continue
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_paths.append(os.path.join(subject_dir, csv_file))
    
    print(f"Found {len(file_paths)} trajectory files")
    return file_paths

# ==================== RNN MODEL ====================
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

    def reverse_diffusion(self, model, x, start_t=None, num_steps_to_return=None):
        """
        Reverse diffusion process to generate trajectories.
        
        Args:
            model: The trained model
            x: Initial noise
            start_t: Starting timestep (defaults to noise_steps - 1)
            num_steps_to_return: Number of intermediate steps to return
            
        Returns:
            List of denoised trajectories at different steps
        """
        model.eval()
        denoised_samples = []
        
        if start_t is None:
            start_t = self.noise_steps - 1
        if num_steps_to_return is None:
            num_steps_to_return = 1
        
        step_size = max(1, start_t // num_steps_to_return)
        
        with torch.no_grad():
            for i in reversed(range(1, start_t + 1)):
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
                
                # Save intermediate results
                if i % step_size == 0 or i == 1:
                    denoised_samples.append(x.cpu().clone())
        
        return denoised_samples

# ==================== TRAINING ====================
def train(model, diffusion, dataloader, device=DEVICE, epochs=NUM_EPOCHS, learning_rate=LR):
    """
    Train the diffusion model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    result_loss = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                          desc=f"Epoch {epoch + 1}/{epochs}")
        
        for i, (trajectories, lengths) in progress_bar:
            trajectories = trajectories.to(device)
            
            # Generate random diffusion steps for each trajectory in batch
            t = torch.randint(0, diffusion.noise_steps, (trajectories.shape[0],), device=device)
            
            # Perform forward diffusion
            noised_trajectories, true_noise = diffusion.forward_diffusion(trajectories, t)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through model to predict noise
            predicted_noise = model(noised_trajectories, t)
            
            # Calculate loss
            loss = criterion(predicted_noise, true_noise)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            result_loss.append(loss.item())
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))
    
    return model, result_loss


# ==================== TESTING NOISE/DENOISE ====================
def test_noise_denoise(model, diffusion, dataloader, device=DEVICE):
    """
    Test the noise and denoise process on a real trajectory from the dataset.
    """
    # Get one real trajectory from the dataset
    real_trajectory, length = next(iter(dataloader))
    real_trajectory = real_trajectory[0:1].to(device)  # Take first trajectory, keep batch dim
    
    print(f"Original trajectory shape: {real_trajectory.shape}")
    print(f"Original trajectory length: {length[0].item()}")
    
    # Noise the trajectory at different timesteps
    timesteps_to_test = [100, 300, 500, 700, 900]
    
    fig, axes = plt.subplots(len(timesteps_to_test), 3, figsize=(15, 4*len(timesteps_to_test)))
    
    for idx, t_value in enumerate(timesteps_to_test):
        # Create timestep tensor
        t = torch.tensor([t_value], device=device)
        
        # Add noise (forward diffusion)
        noised_trajectory, noise = diffusion.forward_diffusion(real_trajectory, t)
        
        # Denoise (reverse diffusion from this timestep)
        model.eval()
        with torch.no_grad():
            denoised = noised_trajectory.clone()
            for i in reversed(range(1, t_value + 1)):
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
        
        # Plot original
        axes[idx, 0].plot(original[:, 0], label='q1', alpha=0.7)
        axes[idx, 0].plot(original[:, 1], label='q2', alpha=0.7)
        axes[idx, 0].set_title(f'Original Trajectory')
        axes[idx, 0].set_ylabel(f't={t_value}')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        # Plot noised
        axes[idx, 1].plot(noised[:, 0], label='q1', alpha=0.7)
        axes[idx, 1].plot(noised[:, 1], label='q2', alpha=0.7)
        axes[idx, 1].set_title(f'Noised Trajectory (t={t_value})')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
        
        # Plot denoised
        axes[idx, 2].plot(denoised_np[:, 0], label='q1', alpha=0.7)
        axes[idx, 2].plot(denoised_np[:, 1], label='q2', alpha=0.7)
        axes[idx, 2].set_title(f'Denoised Trajectory')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'noise_denoise_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
    plt.show()
    
    print("\nTest complete! The plot shows:")
    print("- Left column: Original trajectory")
    print("- Middle column: Noised trajectory at different timesteps")
    print("- Right column: Denoised trajectory (should reconstruct the original)")
    print("\nAs timestep increases, noise increases, making reconstruction harder.")



# ==================== GENERATION ====================
def generate_trajectories(model, diffusion, num_trajectories=10, num_steps_to_return=5):
    """
    Generate new trajectories using the trained model.
    """
    model.eval()
    generated_trajectories = []
    all_steps = []
    
    with torch.no_grad():
        for i in range(num_trajectories):
            # Start with random noise
            x = torch.randn(1, MAX_LENGTH, TRAJECTORY_DIM, device=diffusion.device)
            
            # Reverse diffusion
            trajectories_at_steps = diffusion.reverse_diffusion(
                model, x, num_steps_to_return=num_steps_to_return
            )
            
            all_steps.append([t.squeeze().cpu().numpy() for t in trajectories_at_steps])
            generated_trajectories.append(trajectories_at_steps[0].squeeze().cpu().numpy())
    
    return generated_trajectories, all_steps

# ==================== VISUALIZATION ====================
def plot_trajectories(trajectories, title="Generated Trajectories", max_plots=6):
    """
    Plot multiple trajectories.
    """
    num_plots = min(len(trajectories), max_plots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_plots):
        traj = trajectories[i]
        axes[i].plot(traj[:, 0], label='q1', alpha=0.7)
        axes[i].plot(traj[:, 1], label='q2', alpha=0.7)
        axes[i].set_title(f'Trajectory {i+1}')
        axes[i].set_xlabel('Time step')
        axes[i].set_ylabel('Angle')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_loss(losses):
    """
    Plot training loss.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()

# ==================== MAIN ====================
def main():
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\n=== Loading Data ===")
    file_paths = load_all_trajectories("data")
    dataset = TrajectoryDataset(file_paths, max_length=MAX_LENGTH)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Loaded {len(dataset)} trajectories")
    print(f"Trajectory shape: ({MAX_LENGTH}, {TRAJECTORY_DIM})")
    
    # Initialize model and diffusion
    print("\n=== Initializing Model ===")
    model = DiffusionRNN(
        trajectory_dim=TRAJECTORY_DIM,
        max_length=MAX_LENGTH,
        hidden_dim=256,
        num_layers=6,
        time_dim=128
    ).to(DEVICE)
    
    diffusion = Diffusion(
        noise_steps=NOISE_STEPS,
        max_length=MAX_LENGTH,
        device=DEVICE
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Train model
    print("\n=== Training ===")
    trained_model, losses = train(model, diffusion, dataloader, 
                                  device=DEVICE, epochs=NUM_EPOCHS, learning_rate=LR)
    
    # Plot training loss
    plot_loss(losses)
    
    # Save model
    print("\n=== Saving Model ===")
    os.makedirs("trained_models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"trained_models/diffusion_mlp_{timestamp}.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'trajectory_dim': TRAJECTORY_DIM,
            'max_length': MAX_LENGTH,
            'noise_steps': NOISE_STEPS,
            'hidden_dim': 256,
            'num_layers': 6,
            'time_dim': 128
        },
        'losses': losses
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Generate samples
    print("\n=== Generating Samples ===")
    generated_trajectories, all_steps = generate_trajectories(
        trained_model, diffusion, num_trajectories=6, num_steps_to_return=5
    )
    
    # Plot generated trajectories
    plot_trajectories(generated_trajectories, title="Generated Trajectories")
    
    # Generate samples
    print("\n=== Generating Samples ===")
    generated_trajectories, all_steps = generate_trajectories(
        trained_model, diffusion, num_trajectories=6, num_steps_to_return=5
    )
    
    # Plot generated trajectories
    plot_trajectories(generated_trajectories, title="Generated Trajectories")
    
    # TEST NOISE/DENOISE - ADD THESE LINES
    print("\n=== Testing Noise/Denoise Process ===")
    test_noise_denoise(trained_model, diffusion, dataloader, device=DEVICE)
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    main()