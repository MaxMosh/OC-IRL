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

# PARAMETERS
TRAJECTORY_DIM = 2
CONTEXT_LENGTH = 20
PREDICTION_LENGTH = 80
MAX_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH  # 100
NOISE_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL ARCHITECTURE
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

# DIFFUSION PROCESS
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

# LOAD MODEL
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

# LOAD TEST TRAJECTORY WITH MULTIPLE STARTING POINTS
def load_test_trajectory_multi_start(file_path, context_length=CONTEXT_LENGTH, 
                                      prediction_length=PREDICTION_LENGTH,
                                      num_starts=5):
    """
    Load test trajectory and extract multiple subsequences at different starting points.
    This allows testing conditional generation at different moments in the trajectory.
    
    Args:
        file_path: Path to CSV file
        context_length: Length of context (20)
        prediction_length: Length of prediction (80)
        num_starts: Number of different starting points to extract
        
    Returns:
        List of tuples (context, true_prediction, start_idx) for each starting point
    """
    print(f"Loading test trajectory from: {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path, header=None)
    trajectory = df.T.values  # (time_steps, 2)
    
    total_length = context_length + prediction_length
    
    if len(trajectory) < total_length:
        raise ValueError(f"Trajectory too short: {len(trajectory)} < {total_length}")
    
    # Calculate starting points evenly spaced throughout the trajectory
    max_start = len(trajectory) - total_length
    if num_starts == 1:
        start_indices = [0]
    else:
        start_indices = np.linspace(0, max_start, num_starts, dtype=int)
    
    subsequences = []
    
    for start_idx in start_indices:
        end_idx = start_idx + total_length
        
        context = trajectory[start_idx:start_idx + context_length]
        true_prediction = trajectory[start_idx + context_length:end_idx]
        
        # Convert to tensors
        context_tensor = torch.from_numpy(context).float().unsqueeze(0)
        true_pred_tensor = torch.from_numpy(true_prediction).float().unsqueeze(0)
        
        subsequences.append((context_tensor, true_pred_tensor, int(start_idx)))
    
    print(f"Extracted {len(subsequences)} subsequences at frames: {start_indices.tolist()}")
    
    return subsequences

# TEST: CONDITIONAL GENERATION WITH MULTIPLE STARTS
def test_conditional_generation_multi_start(model, diffusion, subsequences, 
                                            num_samples=10, device=DEVICE, trial_name="Trial"):
    """
    Test conditional generation at multiple starting points in the trajectory.
    
    Args:
        model: Trained conditional diffusion model
        diffusion: Diffusion process
        subsequences: List of (context, true_prediction, start_idx) tuples
        num_samples: Number of predictions to generate per context
        device: Device to run on
        trial_name: Name of the trial for file naming
    """
    print("\n" + "="*50)
    print(f"TEST: Multi-Start Conditional Generation - {trial_name}")
    print("="*50)
    print(f"Testing at {len(subsequences)} different starting points")
    print(f"Generating {num_samples} predictions per starting point...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with one subplot per starting point
    num_starts = len(subsequences)
    fig, axes = plt.subplots(num_starts, 1, figsize=(16, 5*num_starts))
    
    if num_starts == 1:
        axes = [axes]
    
    all_stats = []
    
    for idx, (context, true_prediction, start_idx) in enumerate(subsequences):
        print(f"\n--- Starting point {idx+1}/{num_starts} (frame {start_idx}) ---")
        
        context = context.to(device)
        
        # Generate multiple predictions
        predictions = diffusion.reverse_diffusion(
            model, context, verbose=(idx==0), num_samples=num_samples
        )
        
        # Plot on corresponding subplot
        ax = axes[idx]
        
        # Plot context
        context_np = context.squeeze().cpu().numpy()
        time_context = np.arange(start_idx, start_idx + CONTEXT_LENGTH)
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=3, 
               label='Context q1', markersize=4)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=3, 
               label='Context q2', markersize=4)
        
        # Plot ground truth
        true_pred_np = true_prediction.squeeze().cpu().numpy()
        time_pred = np.arange(start_idx + CONTEXT_LENGTH, 
                             start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH)
        ax.plot(time_pred, true_pred_np[:, 0], '--', color='blue', linewidth=3, 
               alpha=0.7, label='Ground Truth q1')
        ax.plot(time_pred, true_pred_np[:, 1], '--', color='red', linewidth=3, 
               alpha=0.7, label='Ground Truth q2')
        
        # Plot all generated predictions
        for i, pred in enumerate(predictions):
            pred_np = pred.squeeze().cpu().numpy()
            alpha = 0.35
            label_q1 = 'Generated q1' if i == 0 else None
            label_q2 = 'Generated q2' if i == 0 else None
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1.5, 
                   alpha=alpha, label=label_q1)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1.5, 
                   alpha=alpha, label=label_q2)
        
        # Vertical line separator
        ax.axvline(x=start_idx + CONTEXT_LENGTH, color='black', linestyle=':', 
                  linewidth=2, label='Prediction starts')
        
        ax.set_xlabel('Time Step (absolute frame)', fontsize=11)
        ax.set_ylabel('Joint Angle', fontsize=11)
        ax.set_title(f'Start at frame {start_idx}: {num_samples} Predictions', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Compute statistics for this starting point
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
    
    plt.suptitle(f'{trial_name} - Predictions at Multiple Starting Points', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs("test_plots", exist_ok=True)
    save_path = f"test_plots/{trial_name}_multistart_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nMulti-start plot saved to: {save_path}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "-"*60)
    print(f"Summary Statistics for {trial_name}:")
    print("-"*60)
    print(f"{'Frame':<10} {'MSE Mean':<15} {'MSE Std':<15} {'Diversity':<15}")
    print("-"*60)
    for stat in all_stats:
        print(f"{stat['start_idx']:<10} {stat['mse_mean']:<15.6f} "
              f"{stat['mse_std']:<15.6f} {stat['diversity_mean']:<15.6f}")
    print("-"*60)
    
    return all_stats

# CREATE SUMMARY COMPARISON PLOT
def create_summary_plot(all_test_results):
    """
    Create a summary plot showing statistics across all trials and starting points.
    
    Args:
        all_test_results: List of dicts with trial results and statistics
    """
    print("\n" + "="*50)
    print("Creating summary statistics plot...")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data for plotting
    trial_names = []
    all_mse_means = []
    all_diversity_means = []
    
    for result in all_test_results:
        trial_name = result['trial'].replace('.csv', '')
        stats = result['stats']
        
        for stat in stats:
            trial_names.append(f"{trial_name}\n@{stat['start_idx']}")
            all_mse_means.append(stat['mse_mean'])
            all_diversity_means.append(stat['diversity_mean'])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(trial_names))
    
    # Plot 1: MSE across all trials and starting points
    axes[0].bar(x_pos, all_mse_means, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Trial @ Starting Frame', fontsize=12)
    axes[0].set_ylabel('Mean MSE with Ground Truth', fontsize=12)
    axes[0].set_title('Prediction Accuracy Across Trials', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(trial_names, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Diversity across all trials and starting points
    axes[1].bar(x_pos, all_diversity_means, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Trial @ Starting Frame', fontsize=12)
    axes[1].set_ylabel('Mean Pairwise MSE (Diversity)', fontsize=12)
    axes[1].set_title('Prediction Diversity Across Trials', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(trial_names, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('S18 - Summary Statistics Across All Tests', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = f"test_plots/summary_statistics_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary statistics plot saved to: {save_path}")
    plt.show()

# MAIN TEST FUNCTION
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
    
    # LOAD MODEL
    model_path = "trained_models/conditional_diffusion_XXXXXXXX_XXXXXX.pth"  # UPDATE THIS!
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please update the model_path variable with your trained model path.")
        return
    
    model, config, diffusion = load_trained_model(model_path, device=DEVICE)
    
    # LOAD TEST TRAJECTORIES (S18 Trial00, Trial05, Trial10)
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
        
        # TEST: CONDITIONAL GENERATION AT MULTIPLE STARTS
        trial_stats = test_conditional_generation_multi_start(
            model, diffusion, subsequences,
            num_samples=10, device=DEVICE, 
            trial_name=trial_file.replace('.csv', '')
        )
        
        all_test_results.append({
            'trial': trial_file,
            'stats': trial_stats
        })
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print(f"Tested on {len(all_test_results)} trajectories")
    print(f"Each trajectory tested at {num_starting_points} starting points")
    print("Results saved in: test_plots/")

if __name__ == "__main__":
    main()
