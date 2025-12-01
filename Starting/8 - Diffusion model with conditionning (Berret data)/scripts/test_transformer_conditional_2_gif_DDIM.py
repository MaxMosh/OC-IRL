# CONDITIONAL DIFFUSION TRANSFORMER MODEL TESTING SCRIPT WITH GIF ANIMATION & DDIM SAMPLING
# Tests conditional generation on S18 Trial00, Trial05, Trial10
# Uses DDIM (Denoising Diffusion Implicit Models) for faster sampling (50 and 20 steps)
# Includes Execution Time Metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time
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
NOISE_STEPS = 1000 # Original training noise steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL ARCHITECTURE (UNCHANGED)
class PositionalEncoding(nn.Module):
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
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.context_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
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
    def __init__(self, trajectory_dim=TRAJECTORY_DIM, context_length=CONTEXT_LENGTH,
                 prediction_length=PREDICTION_LENGTH, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, time_dim=128):
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.time_dim = time_dim
        
        self.context_input_proj = nn.Linear(trajectory_dim, d_model)
        self.context_pos_encoding = PositionalEncoding(d_model, max_len=context_length)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers=2
        )
        self.context_pooling = nn.Sequential(nn.Linear(d_model * context_length, d_model), nn.GELU())
        self.input_proj = nn.Linear(trajectory_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=prediction_length)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.transformer_blocks = nn.ModuleList([
            ConditionalTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, trajectory_dim)
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
        context_h = self.context_input_proj(context)
        context_h = self.context_pos_encoding(context_h)
        context_h = self.context_encoder(context_h)
        context_flat = context_h.view(batch_size, -1)
        context_emb = self.context_pooling(context_flat)
        
        t_emb = self.timestep_embedding(t.float(), self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        h = self.input_proj(x_noisy)
        h = self.pos_encoding(h)
        
        for block in self.transformer_blocks:
            h = block(h, t_emb, context_emb)
        
        output = self.output_proj(h)
        return output

# DIFFUSION PROCESS WITH DDIM
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

    def ddim_reverse_diffusion(self, model, context, ddim_steps=50, x=None, verbose=True, num_samples=1, eta=0.0):
        """
        DDIM Sampling.
        ddim_steps: Number of inference steps (e.g., 50 or 20).
        eta: 0.0 for deterministic DDIM, 1.0 for DDPM-like stochasticity.
        """
        model.eval()
        batch_size = context.shape[0]
        all_predictions = []

        # Create time steps sequence for DDIM
        # We want to select 'ddim_steps' evenly spaced points from [0, noise_steps-1]
        times = torch.linspace(0, self.noise_steps - 1, steps=ddim_steps + 1).long().to(self.device)
        # Reverse to go from T to 0: e.g. [999, 979, ..., 0]
        times = torch.flip(times, [0])
        
        # Create pairs (t_now, t_next) e.g., (999, 979), (979, 959)...
        time_pairs = list(zip(times[:-1], times[1:]))

        for sample_idx in range(num_samples):
            if x is None:
                # Start from pure noise
                x_pred = torch.randn(batch_size, self.prediction_length, 
                                    context.shape[2], device=self.device)
            else:
                x_pred = x.clone()

            iterator = time_pairs
            if verbose:
                iterator = tqdm(iterator, desc=f"DDIM ({ddim_steps} steps) Sample {sample_idx+1}/{num_samples}")

            with torch.no_grad():
                for t, t_next in iterator:
                    # Broadcast time t to batch
                    t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # 1. Predict noise epsilon_theta
                    predicted_noise = model(x_pred, context, t_tensor)
                    
                    # 2. Get alpha values for current t and next t
                    alpha_hat_t = self.alpha_hat[t]
                    alpha_hat_t_next = self.alpha_hat[t_next]
                    
                    # 3. Predict x0 (clean data)
                    # x_t = sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * eps
                    # => x0 = (x_t - sqrt(1 - alpha_hat_t) * eps) / sqrt(alpha_hat_t)
                    x0_pred = (x_pred - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)
                    
                    # 4. Compute direction pointing to x_t
                    # (Standard DDIM formula with eta=0 makes sigma_t = 0)
                    sigma_t = eta * torch.sqrt((1 - alpha_hat_t_next) / (1 - alpha_hat_t) * (1 - alpha_hat_t / alpha_hat_t_next))
                    
                    # Direction to x_t
                    pred_dir_xt = torch.sqrt(1 - alpha_hat_t_next - sigma_t**2) * predicted_noise
                    
                    # 5. Compute x_{t-1} (or x_{t_next})
                    x_prev = torch.sqrt(alpha_hat_t_next) * x0_pred + pred_dir_xt
                    
                    # Add noise if eta > 0 (Standard DDIM usually eta=0)
                    if sigma_t > 0:
                        noise = torch.randn_like(x_pred)
                        x_prev = x_prev + sigma_t * noise
                        
                    x_pred = x_prev

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
    return model, config, diffusion

# LOAD TEST TRAJECTORY
def load_test_trajectory_multi_start(file_path, context_length=CONTEXT_LENGTH, 
                                      prediction_length=PREDICTION_LENGTH,
                                      num_starts=5):
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
        start_indices = np.linspace(0, max_start, num_starts, dtype=int)
        # start_indices = np.linspace(0, max_start, max_start//3, dtype=int)
    
    subsequences = []
    for start_idx in start_indices:
        end_idx = start_idx + total_length
        context = trajectory[start_idx:start_idx + context_length]
        true_prediction = trajectory[start_idx + context_length:end_idx]
        full_trajectory = trajectory
        context_tensor = torch.from_numpy(context).float().unsqueeze(0)
        true_pred_tensor = torch.from_numpy(true_prediction).float().unsqueeze(0)
        subsequences.append((context_tensor, true_pred_tensor, int(start_idx), full_trajectory))
    
    print(f"Extracted {len(subsequences)} subsequences.")
    return subsequences

# TEST WITH FULL TRAJECTORY VISUALIZATION & TIMING METRICS
def test_conditional_generation_full_trajectory_ddim(model, diffusion, subsequences, 
                                                ddim_steps=50, num_samples=10, 
                                                device=DEVICE, trial_name="Trial"):
    """
    Test conditional generation using DDIM with timing metrics.
    """
    print("\n" + "="*50)
    print(f"TEST: DDIM {ddim_steps} Steps - Full Trajectory - {trial_name}")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_starts = len(subsequences)
    fig, axes = plt.subplots(num_starts, 1, figsize=(18, 5*num_starts))
    if num_starts == 1: axes = [axes]
    
    all_stats = []
    
    for idx, (context, true_prediction, start_idx, full_trajectory) in enumerate(subsequences):
        print(f"\n--- Starting point {idx+1}/{num_starts} (frame {start_idx}) ---")
        context = context.to(device)
        
        # --- TIMING START ---
        start_time = time.time()
        
        # Generate predictions using DDIM
        predictions = diffusion.ddim_reverse_diffusion(
            model, context, ddim_steps=ddim_steps, verbose=(idx==0), num_samples=num_samples
        )
        
        end_time = time.time()
        # --- TIMING END ---
        
        total_gen_time = end_time - start_time
        avg_time_per_sample = total_gen_time / num_samples
        
        ax = axes[idx]
        
        # Plot full trajectory background
        time_full = np.arange(len(full_trajectory))
        ax.plot(time_full, full_trajectory[:, 0], '-', color='lightgray', linewidth=1, alpha=0.5)
        ax.plot(time_full, full_trajectory[:, 1], '-', color='lightgray', linewidth=1, alpha=0.5)
        
        # Zones
        context_end = start_idx + CONTEXT_LENGTH
        pred_end = context_end + PREDICTION_LENGTH
        
        ax.axvspan(start_idx, context_end, alpha=0.1, color='blue')
        ax.axvspan(context_end, pred_end, alpha=0.1, color='green')
        
        # Context
        context_np = context.squeeze().cpu().numpy()
        time_context = np.arange(start_idx, context_end)
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=3, markersize=3)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=3, markersize=3)
        
        # GT
        true_pred_np = true_prediction.squeeze().cpu().numpy()
        time_pred = np.arange(context_end, pred_end)
        ax.plot(time_pred, true_pred_np[:, 0], '--', color='blue', linewidth=2, alpha=0.6)
        ax.plot(time_pred, true_pred_np[:, 1], '--', color='red', linewidth=2, alpha=0.6)
        
        # Predictions
        for pred in predictions:
            pred_np = pred.squeeze().cpu().numpy()
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1, alpha=0.4)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1, alpha=0.4)
        
        # Add Timing Text Box
        time_text = (f"DDIM Steps: {ddim_steps}\n"
                     f"Total Time ({num_samples} traj): {total_gen_time:.4f}s\n"
                     f"Avg Time/Traj: {avg_time_per_sample:.4f}s")
        
        # Place text in top left corner of axes
        ax.text(0.02, 0.95, time_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Start frame {start_idx} (DDIM {ddim_steps})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Stats logic (MSE)
        mse_values = [np.mean((p.squeeze().cpu().numpy() - true_pred_np)**2) for p in predictions]
        print(f"  Avg Time per Sample: {avg_time_per_sample:.4f}s")
        print(f"  MSE with GT: {np.mean(mse_values):.6f}")

    plt.suptitle(f'{trial_name} - DDIM {ddim_steps} Steps Predictions', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs("test_plots_ddim", exist_ok=True)
    save_path = f"test_plots_ddim/{trial_name}_DDIM{ddim_steps}_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    plt.close()
    
    return all_stats

# CREATE ANIMATED GIF (DDIM VERSION)
def create_animated_gif_ddim(model, diffusion, subsequences, ddim_steps=50, 
                             device=DEVICE, trial_name="Trial", num_samples=5, fps=5):
    print("\n" + "="*50)
    print(f"Creating animated GIF for {trial_name} (DDIM {ddim_steps})")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _, _, full_trajectory = subsequences[0]
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    def update_frame(frame_idx):
        ax.clear()
        context, true_prediction, start_idx, _ = subsequences[frame_idx]
        context = context.to(device)
        
        # Use DDIM for GIF generation too
        predictions = diffusion.ddim_reverse_diffusion(
            model, context, ddim_steps=ddim_steps, verbose=False, num_samples=num_samples
        )
        
        # Background
        ax.plot(full_trajectory[:, 0], '-', color='lightgray', alpha=0.5, label='Traj q1')
        ax.plot(full_trajectory[:, 1], '-', color='lightgray', alpha=0.5, label='Traj q2')
        
        # Zones
        context_end = start_idx + CONTEXT_LENGTH
        pred_end = context_end + PREDICTION_LENGTH
        ax.axvspan(start_idx, context_end, alpha=0.2, color='blue')
        ax.axvspan(context_end, pred_end, alpha=0.2, color='green')
        
        # Context
        context_np = context.squeeze().cpu().numpy()
        time_context = np.arange(start_idx, context_end)
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=3)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=3)
        
        # Predictions
        time_pred = np.arange(context_end, pred_end)
        for pred in predictions:
            pred_np = pred.squeeze().cpu().numpy()
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', alpha=0.5)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', alpha=0.5)
            
        ax.set_title(f'{trial_name} (DDIM {ddim_steps}) - Frame {start_idx}', fontsize=14)
        ax.set_xlim(0, len(full_trajectory))
        y_min = min(full_trajectory.min(), full_trajectory.min()) - 0.5
        y_max = max(full_trajectory.max(), full_trajectory.max()) + 0.5
        ax.set_ylim(y_min, y_max)
    
    anim = FuncAnimation(fig, update_frame, frames=len(subsequences), interval=1000//fps)
    
    os.makedirs("test_plots_ddim", exist_ok=True)
    gif_path = f"test_plots_ddim/{trial_name}_DDIM{ddim_steps}_anim_{timestamp}.gif"
    
    print(f"Saving GIF to {gif_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)
    return gif_path

def main():
    print("="*50)
    print("CONDITIONAL DIFFUSION - DDIM TESTING (50 vs 20 steps)")
    print("="*50)
    
    # UPDATE PATH HERE
    model_path = "trained_models/conditional_diffusion_transformer_20251121_epochs_1000.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model, config, diffusion = load_trained_model(model_path, device=DEVICE)
    
    test_files = ["Trial00.csv", "Trial05.csv", "Trial10.csv"]
    test_base_path = "data/S18"
    
    # DDIM Configurations to test
    # NOTE: CHANGE NUMBER OF DDIM STEPS BELOW
    # ddim_configs = [50, 20]
    ddim_configs = [10, 5]
    
    for trial_file in test_files:
        test_path = os.path.join(test_base_path, trial_file)
        if not os.path.exists(test_path): continue
        
        subsequences = load_test_trajectory_multi_start(test_path, num_starts=5)
        trial_name = trial_file.replace('.csv', '')
        
        for steps in ddim_configs:
            print(f"\n>>> PROCESSING {trial_name} WITH DDIM STEPS: {steps} <<<")
            
            # 1. Static Plot with Time Metrics
            test_conditional_generation_full_trajectory_ddim(
                model, diffusion, subsequences, ddim_steps=steps, 
                num_samples=10, device=DEVICE, trial_name=trial_name
            )
            
            # 2. Animated GIF
            create_animated_gif_ddim(
                model, diffusion, subsequences, ddim_steps=steps,
                device=DEVICE, trial_name=trial_name, num_samples=5, fps=2
            )

if __name__ == "__main__":
    main()