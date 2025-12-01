# CONDITIONAL DIFFUSION TRANSFORMER MODEL TESTING SCRIPT WITH GIF ANIMATION
# Tests conditional generation on S18 Trial00, Trial05, Trial10
# Given first 20 frames, generates 10 possible continuations for next 80 frames
# Creates animated GIFs showing predictions over time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import math

# PARAMETERS
TRAJECTORY_DIM = 2
CONTEXT_LENGTH = 20
PREDICTION_LENGTH = 80
MAX_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH  # 100
NOISE_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL ARCHITECTURE
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
        return x + self.pe[:, :x.size(1)]

class ConditionalTransformerBlock(nn.Module):
    """
    Transformer block with self-attention, time embedding, and context conditioning.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.context_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, t_emb, c_emb):
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        c_emb = self.context_mlp(c_emb).unsqueeze(1)
        x = x + t_emb + c_emb
        
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x

class ConditionalDiffusionTransformer(nn.Module):
    """
    Transformer-based diffusion model with conditioning on context frames.
    """
    def __init__(self, trajectory_dim=TRAJECTORY_DIM, 
                 context_length=CONTEXT_LENGTH,
                 prediction_length=PREDICTION_LENGTH,
                 d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, 
                 dropout=0.1, time_dim=128):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.time_dim = time_dim
        
        # Context encoder
        self.context_input_proj = nn.Linear(trajectory_dim, d_model)
        self.context_pos_encoding = PositionalEncoding(d_model, max_len=context_length)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=2
        )
        self.context_pooling = nn.Sequential(
            nn.Linear(d_model * context_length, d_model),
            nn.GELU()
        )
        
        # Input projection
        self.input_proj = nn.Linear(trajectory_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=prediction_length)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ConditionalTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, trajectory_dim)
        )
        
    def timestep_embedding(self, t, channels):
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if channels % 2 == 1:
            emb = F.pad(emb, (0, 1))
            
        return emb
    
    def forward(self, x_noisy, context, t):
        batch_size = x_noisy.shape[0]
        
        # Encode context
        context_h = self.context_input_proj(context)
        context_h = self.context_pos_encoding(context_h)
        context_h = self.context_encoder(context_h)
        context_flat = context_h.view(batch_size, -1)
        context_emb = self.context_pooling(context_flat)
        
        # Time embedding
        t_emb = self.timestep_embedding(t.float(), self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Process through network
        h = self.input_proj(x_noisy)
        h = self.pos_encoding(h)
        
        for block in self.transformer_blocks:
            h = block(h, t_emb, context_emb)
        
        output = self.output_proj(h)
        
        return output

# DIFFUSION PROCESS
class ConditionalDiffusion:
    def __init__(self, noise_steps=NOISE_STEPS, beta_start=0.0001, beta_end=0.02, 
                 prediction_length=PREDICTION_LENGTH, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_length = prediction_length
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self

    def reverse_diffusion(self, model, context, x=None, verbose=True, num_samples=1):
        model.eval()
        
        batch_size = context.shape[0]
        all_predictions = []
        
        for sample_idx in range(num_samples):
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
                    
                    predicted_noise = model(x_pred, context, t)
                    
                    sqrt_alpha = torch.sqrt(self.alpha[i])
                    one_minus_alpha = 1.0 - self.alpha[i]
                    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[i])
                    
                    x_pred = (1 / sqrt_alpha) * (x_pred - one_minus_alpha * predicted_noise / sqrt_one_minus_alpha_hat)
                    
                    if i > 1:
                        noise = torch.randn_like(x_pred, device=self.device)
                        x_pred = x_pred + torch.sqrt(self.beta[i]) * noise
            
            all_predictions.append(x_pred.cpu().clone())
        
        return all_predictions

# LOAD MODEL
def load_trained_model(model_path, device=DEVICE):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    
    model = ConditionalDiffusionTransformer(
        trajectory_dim=config['trajectory_dim'],
        context_length=config['context_length'],
        prediction_length=config['prediction_length'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        time_dim=config['time_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = ConditionalDiffusion(
        noise_steps=config['noise_steps'],
        prediction_length=config['prediction_length'],
        device=device
    )
    
    print(f"Model loaded successfully!")
    print(f"Configuration: {config}")
    
    return model, config, diffusion

# LOAD TEST TRAJECTORY
def load_test_trajectory_multi_start(file_path, context_length=CONTEXT_LENGTH, 
                                      prediction_length=PREDICTION_LENGTH,
                                      num_starts=5):
    """
    Load test trajectory and extract multiple subsequences at different starting points.
    """
    print(f"Loading test trajectory from: {file_path}")
    
    df = pd.read_csv(file_path, header=None)
    trajectory = df.T.values
    
    total_length = context_length + prediction_length
    
    if len(trajectory) < total_length:
        raise ValueError(f"Trajectory too short: {len(trajectory)} < {total_length}")
    
    max_start = len(trajectory) - total_length
    if num_starts == 1:
        start_indices = [0]
    else:
        # NOTE: THE LINE BELOW HAS BEEN MODIFIED FOR GETTING ALL FRAMES
        # start_indices = np.linspace(0, max_start, num_starts, dtype=int)
        start_indices = np.linspace(0, max_start, max_start, dtype=int)
    
    subsequences = []
    
    for start_idx in start_indices:
        end_idx = start_idx + total_length
        
        context = trajectory[start_idx:start_idx + context_length]
        true_prediction = trajectory[start_idx + context_length:end_idx]
        full_trajectory = trajectory  # Keep full trajectory for visualization
        
        context_tensor = torch.from_numpy(context).float().unsqueeze(0)
        true_pred_tensor = torch.from_numpy(true_prediction).float().unsqueeze(0)
        
        subsequences.append((context_tensor, true_pred_tensor, int(start_idx), full_trajectory))
    
    print(f"Extracted {len(subsequences)} subsequences at frames: {start_indices.tolist()}")
    
    return subsequences

# TEST WITH FULL TRAJECTORY VISUALIZATION
def test_conditional_generation_full_trajectory(model, diffusion, subsequences, 
                                                num_samples=10, device=DEVICE, trial_name="Trial"):
    """
    Test conditional generation showing the full trajectory with highlighted context and prediction zones.
    """
    print("\n" + "="*50)
    print(f"TEST: Full Trajectory Visualization - {trial_name}")
    print("="*50)
    print(f"Testing at {len(subsequences)} different starting points")
    print(f"Generating {num_samples} predictions per starting point...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    num_starts = len(subsequences)
    fig, axes = plt.subplots(num_starts, 1, figsize=(18, 5*num_starts))
    
    if num_starts == 1:
        axes = [axes]
    
    all_stats = []
    
    for idx, (context, true_prediction, start_idx, full_trajectory) in enumerate(subsequences):
        print(f"\n--- Starting point {idx+1}/{num_starts} (frame {start_idx}) ---")
        
        context = context.to(device)
        
        # Generate predictions
        predictions = diffusion.reverse_diffusion(
            model, context, verbose=(idx==0), num_samples=num_samples
        )
        
        ax = axes[idx]
        
        # Plot full trajectory in light gray as background
        time_full = np.arange(len(full_trajectory))
        ax.plot(time_full, full_trajectory[:, 0], '-', color='lightgray', 
               linewidth=1, alpha=0.5, label='Full trajectory q1')
        ax.plot(time_full, full_trajectory[:, 1], '-', color='lightgray', 
               linewidth=1, alpha=0.5, label='Full trajectory q2')
        
        # Highlight context zone with vertical spans
        context_start = start_idx
        context_end = start_idx + CONTEXT_LENGTH
        ax.axvspan(context_start, context_end, alpha=0.2, color='blue', label='Context zone')
        ax.axvline(x=context_start, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=context_end, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        # Highlight prediction zone with vertical spans
        pred_start = context_end
        pred_end = start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH
        ax.axvspan(pred_start, pred_end, alpha=0.2, color='green', label='Prediction zone')
        ax.axvline(x=pred_start, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=pred_end, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        # Plot context
        context_np = context.squeeze().cpu().numpy()
        time_context = np.arange(start_idx, start_idx + CONTEXT_LENGTH)
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=3, 
               label='Context q1', markersize=4, zorder=5)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=3, 
               label='Context q2', markersize=4, zorder=5)
        
        # Plot ground truth prediction
        true_pred_np = true_prediction.squeeze().cpu().numpy()
        time_pred = np.arange(start_idx + CONTEXT_LENGTH, 
                             start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH)
        ax.plot(time_pred, true_pred_np[:, 0], '--', color='blue', linewidth=3, 
               alpha=0.8, label='Ground Truth q1', zorder=4)
        ax.plot(time_pred, true_pred_np[:, 1], '--', color='red', linewidth=3, 
               alpha=0.8, label='Ground Truth q2', zorder=4)
        
        # Plot generated predictions
        for i, pred in enumerate(predictions):
            pred_np = pred.squeeze().cpu().numpy()
            alpha = 0.35
            label_q1 = 'Generated q1' if i == 0 else None
            label_q2 = 'Generated q2' if i == 0 else None
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1.5, 
                   alpha=alpha, label=label_q1, zorder=3)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1.5, 
                   alpha=alpha, label=label_q2, zorder=3)
        
        ax.set_xlabel('Time Step (absolute frame)', fontsize=11)
        ax.set_ylabel('Joint Angle', fontsize=11)
        ax.set_title(f'Start at frame {start_idx}: {num_samples} Predictions', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Compute statistics
        mse_values = []
        for pred in predictions:
            pred_np = pred.squeeze().cpu().numpy()
            mse = np.mean((pred_np - true_pred_np) ** 2)
            mse_values.append(mse)
        
        pairwise_distances = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                pred_i = predictions[i].squeeze().cpu().numpy()
                pred_j = predictions[j].squeeze().cpu().numpy()
                dist = np.mean((pred_i - pred_j) ** 2)
                pairwise_distances.append(dist)
        
        stats = {
            'start_idx': start_idx,
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'diversity_mean': np.mean(pairwise_distances),
            'diversity_std': np.std(pairwise_distances)
        }
        all_stats.append(stats)
        
        print(f"  MSE with GT: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
        print(f"  Diversity: {stats['diversity_mean']:.6f} ± {stats['diversity_std']:.6f}")
    
    plt.suptitle(f'{trial_name} - Predictions with Full Trajectory Context', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs("test_plots", exist_ok=True)
    save_path = f"test_plots/{trial_name}_full_trajectory_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFull trajectory plot saved to: {save_path}")
    plt.show()
    
    return all_stats

# CREATE ANIMATED GIF
def create_animated_gif(model, diffusion, subsequences, device=DEVICE, 
                       trial_name="Trial", num_samples=5, fps=5):
    """
    Create an animated GIF showing predictions evolving over the full trajectory.
    The animation shows the sliding window moving through the trajectory.
    """
    print("\n" + "="*50)
    print(f"Creating animated GIF for {trial_name}")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get full trajectory from first subsequence
    _, _, _, full_trajectory = subsequences[0]
    
    # Prepare figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    def update_frame(frame_idx):
        """Update function for animation."""
        ax.clear()
        
        context, true_prediction, start_idx, _ = subsequences[frame_idx]
        context = context.to(device)
        
        # Generate predictions for this frame
        print(f"Generating frame {frame_idx + 1}/{len(subsequences)}...")
        predictions = diffusion.reverse_diffusion(
            model, context, verbose=False, num_samples=num_samples
        )
        
        # Plot full trajectory in light gray
        time_full = np.arange(len(full_trajectory))
        ax.plot(time_full, full_trajectory[:, 0], '-', color='lightgray', 
               linewidth=1, alpha=0.5, label='Full trajectory q1')
        ax.plot(time_full, full_trajectory[:, 1], '-', color='lightgray', 
               linewidth=1, alpha=0.5, label='Full trajectory q2')
        
        # Highlight context zone
        context_start = start_idx
        context_end = start_idx + CONTEXT_LENGTH
        ax.axvspan(context_start, context_end, alpha=0.2, color='blue', label='Context zone')
        ax.axvline(x=context_start, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=context_end, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        
        # Highlight prediction zone
        pred_start = context_end
        pred_end = start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH
        ax.axvspan(pred_start, pred_end, alpha=0.2, color='green', label='Prediction zone')
        ax.axvline(x=pred_start, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=pred_end, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        # Plot context
        context_np = context.squeeze().cpu().numpy()
        time_context = np.arange(start_idx, start_idx + CONTEXT_LENGTH)
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=3, 
               label='Context q1', markersize=4, zorder=5)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=3, 
               label='Context q2', markersize=4, zorder=5)
        
        # Plot ground truth
        true_pred_np = true_prediction.squeeze().cpu().numpy()
        time_pred = np.arange(start_idx + CONTEXT_LENGTH, 
                             start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH)
        ax.plot(time_pred, true_pred_np[:, 0], '--', color='blue', linewidth=3, 
               alpha=0.8, label='Ground Truth q1', zorder=4)
        ax.plot(time_pred, true_pred_np[:, 1], '--', color='red', linewidth=3, 
               alpha=0.8, label='Ground Truth q2', zorder=4)
        
        # Plot generated predictions
        for i, pred in enumerate(predictions):
            pred_np = pred.squeeze().cpu().numpy()
            alpha_val = 0.4
            label_q1 = 'Generated q1' if i == 0 else None
            label_q2 = 'Generated q2' if i == 0 else None
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1.5, 
                   alpha=alpha_val, label=label_q1, zorder=3)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1.5, 
                   alpha=alpha_val, label=label_q2, zorder=3)
        
        ax.set_xlabel('Time Step (absolute frame)', fontsize=12)
        ax.set_ylabel('Joint Angle', fontsize=12)
        ax.set_title(f'{trial_name} - Frame {start_idx}: {num_samples} Predictions', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        ax.set_xlim(0, len(full_trajectory))
        y_min = min(full_trajectory[:, 0].min(), full_trajectory[:, 1].min()) - 0.2
        y_max = max(full_trajectory[:, 0].max(), full_trajectory[:, 1].max()) + 0.2
        ax.set_ylim(y_min, y_max)
    
    # Create animation
    print("Creating animation...")
    anim = FuncAnimation(fig, update_frame, frames=len(subsequences), 
                        interval=1000//fps, repeat=True)
    
    # Save as GIF
    os.makedirs("test_plots", exist_ok=True)
    gif_path = f"test_plots/{trial_name}_animation_{timestamp}.gif"
    
    print(f"Saving GIF to {gif_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    
    plt.close(fig)
    
    print(f"GIF saved successfully!")
    print(f"Number of frames: {len(subsequences)}")
    print(f"Frame rate: {fps} fps")
    print(f"Duration: {len(subsequences)/fps:.1f} seconds")
    
    return gif_path

# MAIN TEST FUNCTION
def main():
    """
    Main test function with full trajectory visualization and GIF animation.
    """
    print("="*50)
    print("CONDITIONAL DIFFUSION TRANSFORMER - FULL TRAJECTORY & GIF")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Context length: {CONTEXT_LENGTH} frames")
    print(f"Prediction length: {PREDICTION_LENGTH} frames")
    
    # Load model
    model_path = "trained_models/conditional_diffusion_transformer_20251121_epochs_1000.pth"  # UPDATE THIS!
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please update the model_path variable with your trained model path.")
        return
    
    model, config, diffusion = load_trained_model(model_path, device=DEVICE)
    
    # Load test trajectories
    test_files = ["Trial00.csv", "Trial05.csv", "Trial10.csv"]
    test_base_path = "data/S18"
    num_starting_points = 5  # Test at 5 different starting points per trial
    
    all_test_results = []
    
    for trial_file in test_files:
        test_trajectory_path = os.path.join(test_base_path, trial_file)
        
        if not os.path.exists(test_trajectory_path):
            print(f"\nWARNING: Test trajectory not found at {test_trajectory_path}")
            print("Skipping this file.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing on: {trial_file}")
        print(f"{'='*60}")
        
        # Load trajectory with multiple starting points
        subsequences = load_test_trajectory_multi_start(
            test_trajectory_path, 
            context_length=CONTEXT_LENGTH, 
            prediction_length=PREDICTION_LENGTH,
            num_starts=num_starting_points
        )
        
        # Test 1: Full trajectory visualization with static plot
        trial_stats = test_conditional_generation_full_trajectory(
            model, diffusion, subsequences,
            num_samples=10, device=DEVICE, 
            trial_name=trial_file.replace('.csv', '')
        )
        
        # Test 2: Create animated GIF
        gif_path = create_animated_gif(
            model, diffusion, subsequences,
            device=DEVICE,
            trial_name=trial_file.replace('.csv', ''),
            num_samples=5,  # Use fewer samples for faster GIF generation
            fps=2  # 2 frames per second (slower animation)
        )
        
        all_test_results.append({
            'trial': trial_file,
            'stats': trial_stats,
            'gif_path': gif_path
        })
    
    # Summary
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print(f"\nGenerated visualizations for {len(all_test_results)} trials:")
    for result in all_test_results:
        print(f"  - {result['trial']}: Static plot + Animated GIF")
        print(f"    GIF location: {result['gif_path']}")
    print("\nAll results saved in: test_plots/")

if __name__ == "__main__":
    main()
