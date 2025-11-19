# CONDITIONAL DIFFUSION MODEL TESTING SCRIPT
# Tests conditional generation on S18 Trial00.csv
# Given first 20 frames, generates 10 possible continuations for next 80 frames

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== CONFIG ====================
TRAJECTORY_DIM = 2
CONTEXT_LENGTH = 20
PREDICTION_LENGTH = 80
MAX_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH  # 100
NOISE_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== MODEL ARCHITECTURE ====================
class ConditionalRNNBlock(nn.Module):
    """
    RNN block with residual connection, time embedding, and context conditioning.
    """
    def __init__(self, dim, time_dim, context_dim):
        super().__init__()
        self.rnn = nn.RNN(input_size=dim, hidden_size=dim, batch_first=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.GELU()
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, dim),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, t, c):
        """
        Args:
            x: (batch, seq_len, dim)
            t: (batch, time_dim)
            c: (batch, context_dim)
        """
        t_emb = self.time_mlp(t).unsqueeze(1)
        c_emb = self.context_mlp(c).unsqueeze(1)
        
        rnn_out, _ = self.rnn(x)
        
        h = self.norm(rnn_out)
        h = h + t_emb + c_emb
        return x + h

class ConditionalDiffusionRNN(nn.Module):
    """
    RNN-based diffusion model with conditioning on context frames.
    """
    def __init__(self, trajectory_dim=TRAJECTORY_DIM, 
                 context_length=CONTEXT_LENGTH,
                 prediction_length=PREDICTION_LENGTH,
                 hidden_dim=256, num_layers=6, time_dim=128):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.time_dim = time_dim
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_length * trajectory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(trajectory_dim, hidden_dim)
        
        # RNN blocks
        self.blocks = nn.ModuleList([
            ConditionalRNNBlock(hidden_dim, time_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
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
    
    def forward(self, x_noisy, context, t):
        """
        Args:
            x_noisy: (batch, prediction_length, trajectory_dim)
            context: (batch, context_length, trajectory_dim)
            t: (batch,)
            
        Returns:
            (batch, prediction_length, trajectory_dim) - predicted noise
        """
        batch_size = x_noisy.shape[0]
        
        # Encode context
        context_flat = context.view(batch_size, -1)
        context_emb = self.context_encoder(context_flat)
        
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Process through network
        h = self.input_proj(x_noisy)
        
        for block in self.blocks:
            h = block(h, t_emb, context_emb)
        
        output = self.output_proj(h)
        
        return output

# ==================== DIFFUSION PROCESS ====================
class ConditionalDiffusion:
    def __init__(self, noise_steps=NOISE_STEPS, beta_start=0.0001, beta_end=0.02, 
                 prediction_length=PREDICTION_LENGTH, device='cuda'):
        """
        Initialize the Conditional Diffusion process.
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_length = prediction_length
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

    def reverse_diffusion(self, model, context, x=None, verbose=True, num_samples=1):
        """
        Reverse diffusion to generate multiple predictions.
        
        Args:
            model: The trained model
            context: (batch, context_length, trajectory_dim)
            x: Initial noise (if None, starts from Gaussian)
            verbose: Show progress bar
            num_samples: Number of predictions to generate
            
        Returns:
            List of generated predictions
        """
        model.eval()
        
        batch_size = context.shape[0]
        all_predictions = []
        
        for sample_idx in range(num_samples):
            # Start with random noise
            if x is None:
                x_pred = torch.randn(batch_size, self.prediction_length, 
                                    context.shape[2], device=self.device)
            else:
                x_pred = x.clone()
            
            iterator = reversed(range(1, self.noise_steps))
            if verbose:
                iterator = tqdm(list(iterator), desc=f"Generating sample {sample_idx+1}/{num_samples}")
            
            with torch.no_grad():
                for i in iterator:
                    t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                    
                    # Predict noise
                    predicted_noise = model(x_pred, context, t)
                    
                    # Denoise
                    sqrt_alpha = torch.sqrt(self.alpha[i])
                    one_minus_alpha = 1.0 - self.alpha[i]
                    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[i])
                    
                    x_pred = (1 / sqrt_alpha) * (x_pred - one_minus_alpha * predicted_noise / sqrt_one_minus_alpha_hat)
                    
                    # Add noise (except last step)
                    if i > 1:
                        noise = torch.randn_like(x_pred, device=self.device)
                        x_pred = x_pred + torch.sqrt(self.beta[i]) * noise
            
            all_predictions.append(x_pred.cpu().clone())
        
        return all_predictions

# ==================== LOAD MODEL ====================
def load_trained_model(model_path, device=DEVICE):
    """
    Load trained conditional model from checkpoint.
    
    Args:
        model_path: Path to saved model
        device: Device to load on
        
    Returns:
        model, config, diffusion
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    
    # Initialize model
    model = ConditionalDiffusionRNN(
        trajectory_dim=config['trajectory_dim'],
        context_length=config['context_length'],
        prediction_length=config['prediction_length'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        time_dim=config['time_dim']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize diffusion
    diffusion = ConditionalDiffusion(
        noise_steps=config['noise_steps'],
        prediction_length=config['prediction_length'],
        device=device
    )
    
    print(f"Model loaded successfully!")
    print(f"Configuration: {config}")
    
    return model, config, diffusion

# ==================== LOAD TEST TRAJECTORY ====================
def load_test_trajectory(file_path, context_length=CONTEXT_LENGTH, 
                        prediction_length=PREDICTION_LENGTH):
    """
    Load test trajectory and split into context and ground truth prediction.
    
    Args:
        file_path: Path to CSV file
        context_length: Length of context (20)
        prediction_length: Length of prediction (80)
        
    Returns:
        context, true_prediction, full_trajectory
    """
    print(f"Loading test trajectory from: {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path, header=None)
    trajectory = df.T.values  # (time_steps, 2)
    
    total_length = context_length + prediction_length
    
    if len(trajectory) < total_length:
        raise ValueError(f"Trajectory too short: {len(trajectory)} < {total_length}")
    
    # Extract context and prediction
    context = trajectory[:context_length]
    true_prediction = trajectory[context_length:total_length]
    full_trajectory = trajectory[:total_length]
    
    # Convert to tensors
    context_tensor = torch.from_numpy(context).float().unsqueeze(0)
    true_pred_tensor = torch.from_numpy(true_prediction).float().unsqueeze(0)
    full_traj_tensor = torch.from_numpy(full_trajectory).float()
    
    print(f"Context shape: {context_tensor.shape}")
    print(f"True prediction shape: {true_pred_tensor.shape}")
    
    return context_tensor, true_pred_tensor, full_traj_tensor

# ==================== TEST: CONDITIONAL GENERATION ====================
def test_conditional_generation(model, diffusion, context, true_prediction, 
                               num_samples=10, device=DEVICE):
    """
    Test conditional generation: given context, generate multiple predictions.
    
    Args:
        model: Trained conditional diffusion model
        diffusion: Diffusion process
        context: Context frames (1, 20, 2)
        true_prediction: Ground truth continuation (1, 80, 2)
        num_samples: Number of predictions to generate (default: 10)
        device: Device to run on
    """
    print("\n" + "="*50)
    print("TEST: Conditional Generation")
    print("="*50)
    print(f"Generating {num_samples} possible continuations...")
    
    context = context.to(device)
    
    # Generate multiple predictions
    predictions = diffusion.reverse_diffusion(
        model, context, verbose=True, num_samples=num_samples
    )
    
    # Plot results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Figure 1: All predictions on one plot
    fig1, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    # Plot context
    context_np = context.squeeze().cpu().numpy()
    time_context = np.arange(CONTEXT_LENGTH)
    ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=3, 
           label='Context q1', markersize=5)
    ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=3, 
           label='Context q2', markersize=5)
    
    # Plot ground truth
    true_pred_np = true_prediction.squeeze().cpu().numpy()
    time_pred = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)
    ax.plot(time_pred, true_pred_np[:, 0], '--', color='blue', linewidth=3, 
           alpha=0.7, label='Ground Truth q1')
    ax.plot(time_pred, true_pred_np[:, 1], '--', color='red', linewidth=3, 
           alpha=0.7, label='Ground Truth q2')
    
    # Plot all generated predictions
    for i, pred in enumerate(predictions):
        pred_np = pred.squeeze().cpu().numpy()
        alpha = 0.4
        label_q1 = 'Generated q1' if i == 0 else None
        label_q2 = 'Generated q2' if i == 0 else None
        ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1.5, 
               alpha=alpha, label=label_q1)
        ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1.5, 
               alpha=alpha, label=label_q2)
    
    # Vertical line separator
    ax.axvline(x=CONTEXT_LENGTH, color='black', linestyle=':', linewidth=2, 
              label='Prediction starts')
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Joint Angle', fontsize=12)
    ax.set_title(f'Conditional Generation: {num_samples} Predictions from Same Context', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("test_plots", exist_ok=True)
    save_path1 = f"test_plots/conditional_generation_all_{timestamp}.png"
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path1}")
    plt.show()
    
    # Figure 2: Individual predictions in grid
    rows = (num_samples + 2) // 3  # 3 columns
    fig2, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i, pred in enumerate(predictions):
        if i >= len(axes):
            break
            
        pred_np = pred.squeeze().cpu().numpy()
        
        # Plot context
        axes[i].plot(time_context, context_np[:, 0], 'o-', color='blue', 
                    linewidth=2, label='Context q1', markersize=4)
        axes[i].plot(time_context, context_np[:, 1], 'o-', color='red', 
                    linewidth=2, label='Context q2', markersize=4)
        
        # Plot ground truth
        axes[i].plot(time_pred, true_pred_np[:, 0], '--', color='blue', 
                    linewidth=2, alpha=0.5, label='True q1')
        axes[i].plot(time_pred, true_pred_np[:, 1], '--', color='red', 
                    linewidth=2, alpha=0.5, label='True q2')
        
        # Plot this prediction
        axes[i].plot(time_pred, pred_np[:, 0], '-', color='cyan', 
                    linewidth=2, label='Generated q1')
        axes[i].plot(time_pred, pred_np[:, 1], '-', color='orange', 
                    linewidth=2, label='Generated q2')
        
        axes[i].axvline(x=CONTEXT_LENGTH, color='black', linestyle=':', linewidth=1.5)
        axes[i].set_title(f'Prediction {i+1}', fontweight='bold')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Joint Angle')
        axes[i].legend(loc='best', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Individual Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path2 = f"test_plots/conditional_generation_grid_{timestamp}.png"
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Grid plot saved to: {save_path2}")
    plt.show()
    
    # Compute statistics
    print("\n" + "-"*50)
    print("Statistics:")
    
    # MSE with ground truth
    mse_values = []
    for pred in predictions:
        pred_np = pred.squeeze().cpu().numpy()
        mse = np.mean((pred_np - true_pred_np) ** 2)
        mse_values.append(mse)
    
    print(f"MSE with ground truth:")
    print(f"  Mean: {np.mean(mse_values):.6f}")
    print(f"  Std:  {np.std(mse_values):.6f}")
    print(f"  Min:  {np.min(mse_values):.6f}")
    print(f"  Max:  {np.max(mse_values):.6f}")
    
    # Diversity: pairwise distances between predictions
    pairwise_distances = []
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            pred_i = predictions[i].squeeze().cpu().numpy()
            pred_j = predictions[j].squeeze().cpu().numpy()
            dist = np.mean((pred_i - pred_j) ** 2)
            pairwise_distances.append(dist)
    
    print(f"\nPairwise MSE between predictions (diversity):")
    print(f"  Mean: {np.mean(pairwise_distances):.6f}")
    print(f"  Std:  {np.std(pairwise_distances):.6f}")
    print(f"  Min:  {np.min(pairwise_distances):.6f}")
    print(f"  Max:  {np.max(pairwise_distances):.6f}")
    print("-"*50)
    
    return predictions

# ==================== MAIN TEST FUNCTION ====================
def main():
    """
    Main test function for conditional generation on S18 Trial00.csv
    """
    print("="*50)
    print("CONDITIONAL DIFFUSION MODEL TESTING")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Context length: {CONTEXT_LENGTH} frames")
    print(f"Prediction length: {PREDICTION_LENGTH} frames")
    
    # ========== LOAD MODEL ==========
    model_path = "trained_models/conditional_diffusion_20251119_epochs_10000.pth"  # UPDATE THIS!
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please update the model_path variable with your trained model path.")
        return
    
    model, config, diffusion = load_trained_model(model_path, device=DEVICE)
    
    # ========== LOAD TEST TRAJECTORY (S18 Trial00.csv) ==========
    test_trajectory_path = "data/S18/Trial00.csv"
    
    if not os.path.exists(test_trajectory_path):
        print(f"\nERROR: Test trajectory not found at {test_trajectory_path}")
        print("Please ensure the file exists.")
        return
    
    context, true_prediction, full_trajectory = load_test_trajectory(
        test_trajectory_path, 
        context_length=CONTEXT_LENGTH, 
        prediction_length=PREDICTION_LENGTH
    )
    
    # ========== TEST: CONDITIONAL GENERATION ==========
    predictions = test_conditional_generation(
        model, diffusion, context, true_prediction,
        num_samples=10, device=DEVICE
    )
    
    print("\n" + "="*50)
    print("TEST COMPLETED")
    print("="*50)
    print(f"Generated {len(predictions)} predictions from context")
    print("Results saved in: test_plots/")

if __name__ == "__main__":
    main()