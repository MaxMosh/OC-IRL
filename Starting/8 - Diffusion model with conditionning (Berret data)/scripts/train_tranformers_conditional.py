import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# ==================== CONFIG ====================
TRAJECTORY_DIM = 2  # q1 and q2
MAX_LENGTH = 200  # Maximum trajectory length (adjust based on your data)
NOISE_STEPS = 1000
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

# ==================== TRANSFORMER MODEL ====================
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence positions.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward layers.
    Includes time embedding conditioning.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb):
        """
        Args:
            x: (batch, seq_len, d_model)
            t_emb: (batch, d_model) - time embedding
        """
        # Add time embedding to input
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (batch, 1, d_model)
        x = x + t_emb
        
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x

class DiffusionTransformer(nn.Module):
    """
    Transformer-based diffusion model for trajectory generation.
    """
    def __init__(self, trajectory_dim=TRAJECTORY_DIM, max_length=MAX_LENGTH, 
                 d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, 
                 dropout=0.1, time_dim=128):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.max_length = max_length
        self.d_model = d_model
        self.time_dim = time_dim
        
        # Input projection: project trajectory_dim to d_model
        self.input_proj = nn.Linear(trajectory_dim, d_model)
        
        # Positional encoding for sequence positions
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_length)
        
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection: project d_model back to trajectory_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, trajectory_dim)
        )
        
    def timestep_embedding(self, t, channels):
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: (batch,) tensor of timesteps
            channels: number of channels for embedding
            
        Returns:
            (batch, channels) tensor of embeddings
        """
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if channels % 2 == 1:  # Zero pad if odd number of channels
            emb = F.pad(emb, (0, 1))
            
        return emb
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, seq_len, trajectory_dim) - noisy trajectory
            t: (batch,) - timestep
            
        Returns:
            (batch, seq_len, trajectory_dim) - predicted noise
        """
        batch_size, seq_len, _ = x.shape
        
        # Create time embedding
        t_emb = self.timestep_embedding(t.float(), self.time_dim)  # (batch, time_dim)
        t_emb = self.time_mlp(t_emb)  # (batch, d_model)
        
        # Project input to d_model dimension
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        h = self.pos_encoding(h)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            h = block(h, t_emb)
        
        # Project back to trajectory dimension
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

    def reverse_diffusion(self, model, x, start_t=None, num_steps_to_return=None, context=None):
        """
        Reverse diffusion process to generate trajectories conditionnées.
        
        Args:
            model: The trained model
            x: Initial noise
            start_t: Starting timestep (defaults to noise_steps - 1)
            num_steps_to_return: Number of intermediate steps to return
            context: début de trajectoire (optionnel)
            
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
                
                # Si context est fourni, on remplace la partie contextuelle à chaque étape
                if context is not None:
                    context_length = context.shape[1]
                    x[:, :context_length, :] = context
                
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

def generate_conditioned_trajectories(model, diffusion, context, num_trajectories=6, num_steps_to_return=5):
    """
    Génère plusieurs suites de trajectoires à partir d'un début (context).
    Args:
        model: modèle entraîné
        diffusion: objet diffusion
        context: torch.Tensor (1, context_length, trajectory_dim)
        num_trajectories: nombre de suites à générer
        num_steps_to_return: nombre d'étapes intermédiaires à retourner
    Returns:
        generated_trajectories, all_steps
    """
    model.eval()
    generated_trajectories = []
    all_steps = []
    context_length = context.shape[1]
    for i in range(num_trajectories):
        # Initialiser le bruit pour la partie à générer
        gen_length = diffusion.max_length - context_length
        noise = torch.randn(1, gen_length, diffusion.trajectory_dim, device=diffusion.device)
        # Concaténer le contexte et le bruit
        x = torch.cat([context, noise], dim=1)
        # Reverse diffusion conditionné
        trajectories_at_steps = diffusion.reverse_diffusion(
            model, x, num_steps_to_return=num_steps_to_return, context=context
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
    model = DiffusionTransformer(
        trajectory_dim=TRAJECTORY_DIM,
        max_length=MAX_LENGTH,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
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
    model_path = f"trained_models/diffusion_transformer_{timestamp}.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'trajectory_dim': TRAJECTORY_DIM,
            'max_length': MAX_LENGTH,
            'noise_steps': NOISE_STEPS,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
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
    
    # TEST NOISE/DENOISE
    print("\n=== Testing Noise/Denoise Process ===")
    test_noise_denoise(trained_model, diffusion, dataloader, device=DEVICE)
    
    print("\n=== Complete ===")

if __name__ == "__main__":
    main()