# CONDITIONAL DIFFUSION TRANSFORMER MODEL TESTING SCRIPT
# Tests conditional generation on S18 Trial00, Trial05, Trial10
# Given first 20 frames, generates 10 possible continuations for next 80 frames

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import seaborn as sns

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
    Modified to return attention weights for visualization.
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
        
    def forward(self, x, t_emb, c_emb, return_attention=False):
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        c_emb = self.context_mlp(c_emb).unsqueeze(1)
        x = x + t_emb + c_emb
        
        attn_out, attn_weights = self.self_attn(x, x, x, average_attn_weights=False)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        if return_attention:
            return x, attn_weights
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
        self.nhead = nhead
        
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
    
    def forward(self, x_noisy, context, t, return_attention=False):
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
        
        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                h, attn = block(h, t_emb, context_emb, return_attention=True)
                attention_weights.append(attn)
            else:
                h = block(h, t_emb, context_emb, return_attention=False)
        
        output = self.output_proj(h)
        
        if return_attention:
            return output, attention_weights
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
                    
                    predicted_noise = model(x_pred, context, t, return_attention=False)
                    
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

# LOAD TRAINING DATASET
def load_train_dataset(base_dir="data", subjects=None, context_length=CONTEXT_LENGTH, 
                       prediction_length=PREDICTION_LENGTH, max_trajectories=500):
    """
    Load trajectories from training dataset (S01-S14).
    
    Returns:
        List of (context, prediction, full_trajectory) tuples
    """
    if subjects is None:
        subjects = range(1, 15)  # S01 to S14 (training set)
    
    print(f"\nLoading training dataset from subjects {list(subjects)}...")
    
    file_paths = []
    for i in subjects:
        subject_dir = os.path.join(base_dir, f"S{i:02d}")
        
        if not os.path.exists(subject_dir):
            continue
        
        csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_paths.append(os.path.join(subject_dir, csv_file))
    
    # Limit number of trajectories
    if max_trajectories is not None and len(file_paths) > max_trajectories:
        file_paths = file_paths[:max_trajectories]
    
    train_data = []
    total_length = context_length + prediction_length
    
    for path in tqdm(file_paths, desc="Loading training data"):
        try:
            df = pd.read_csv(path, header=None)
            trajectory = df.T.values
            
            if len(trajectory) >= total_length:
                context = trajectory[:context_length]
                prediction = trajectory[context_length:total_length]
                full_traj = trajectory[:total_length]
                
                train_data.append((context, prediction, full_traj))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"Loaded {len(train_data)} trajectories from training set")
    return train_data

# COMPUTE NEAREST NEIGHBOR IN TRAINING SET
def find_nearest_neighbor(generated_pred, context, train_data):
    """
    Find the nearest neighbor in the training set for a generated prediction.
    Compare based on full trajectory (context + prediction).
    
    Args:
        generated_pred: Generated prediction (80, 2)
        context: Context used for generation (20, 2)
        train_data: List of (context, prediction, full_trajectory) from training set
        
    Returns:
        nearest_full_traj, nearest_context, nearest_pred, min_distance, idx
    """
    # Concatenate context and generated prediction
    full_generated = np.vstack([context, generated_pred])
    
    min_distance = float('inf')
    nearest_idx = -1
    nearest_full_traj = None
    nearest_context = None
    nearest_pred = None
    
    for idx, (train_context, train_pred, train_full) in enumerate(train_data):
        # Compare full trajectories
        distance = np.mean((full_generated - train_full) ** 2)
        
        if distance < min_distance:
            min_distance = distance
            nearest_idx = idx
            nearest_full_traj = train_full
            nearest_context = train_context
            nearest_pred = train_pred
    
    return nearest_full_traj, nearest_context, nearest_pred, min_distance, nearest_idx

# LOAD TEST TRAJECTORY WITH MULTIPLE STARTING POINTS
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
        start_indices = np.linspace(0, max_start, num_starts, dtype=int)
    
    subsequences = []
    
    for start_idx in start_indices:
        end_idx = start_idx + total_length
        
        context = trajectory[start_idx:start_idx + context_length]
        true_prediction = trajectory[start_idx + context_length:end_idx]
        
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
    """
    print("\n" + "="*50)
    print(f"TEST: Multi-Start Conditional Generation - {trial_name}")
    print("="*50)
    print(f"Testing at {len(subsequences)} different starting points")
    print(f"Generating {num_samples} predictions per starting point...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    num_starts = len(subsequences)
    fig, axes = plt.subplots(num_starts, 1, figsize=(16, 5*num_starts))
    
    if num_starts == 1:
        axes = [axes]
    
    all_stats = []
    
    for idx, (context, true_prediction, start_idx) in enumerate(subsequences):
        print(f"\n--- Starting point {idx+1}/{num_starts} (frame {start_idx}) ---")
        
        context = context.to(device)
        
        predictions = diffusion.reverse_diffusion(
            model, context, verbose=(idx==0), num_samples=num_samples
        )
        
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
        
        # Plot generated predictions
        for i, pred in enumerate(predictions):
            pred_np = pred.squeeze().cpu().numpy()
            alpha = 0.35
            label_q1 = 'Generated q1' if i == 0 else None
            label_q2 = 'Generated q2' if i == 0 else None
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1.5, 
                   alpha=alpha, label=label_q1)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1.5, 
                   alpha=alpha, label=label_q2)
        
        ax.axvline(x=start_idx + CONTEXT_LENGTH, color='black', linestyle=':', 
                  linewidth=2, label='Prediction starts')
        
        ax.set_xlabel('Time Step (absolute frame)', fontsize=11)
        ax.set_ylabel('Joint Angle', fontsize=11)
        ax.set_title(f'Start at frame {start_idx}: {num_samples} Predictions', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
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
    
    plt.suptitle(f'{trial_name} - Predictions at Multiple Starting Points', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    
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

# TEST: NEAREST NEIGHBOR ANALYSIS
def test_nearest_neighbors(model, diffusion, subsequences, train_data, 
                           num_samples=6, device=DEVICE, trial_name="Trial"):
    """
    Generate predictions and find their nearest neighbors in the training set.
    """
    print("\n" + "="*60)
    print(f"TEST: Nearest Neighbor Analysis - {trial_name}")
    print("="*60)
    print(f"Generating {num_samples} predictions and finding nearest neighbors...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Take first context for this test
    context, true_prediction, start_idx = subsequences[0]
    context = context.to(device)
    context_np = context.squeeze().cpu().numpy()
    
    # Generate predictions
    predictions = diffusion.reverse_diffusion(model, context, verbose=True, num_samples=num_samples)
    
    # Find nearest neighbors
    print("\nFinding nearest neighbors in training set...")
    nearest_neighbors = []
    distances = []
    
    for i, pred in enumerate(tqdm(predictions, desc="Finding NNs")):
        pred_np = pred.squeeze().cpu().numpy()
        nn_full, nn_context, nn_pred, dist, idx = find_nearest_neighbor(pred_np, context_np, train_data)
        nearest_neighbors.append((nn_full, nn_context, nn_pred))
        distances.append(dist)
        print(f"  Prediction {i+1}: NN index {idx}, MSE distance: {dist:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        pred_np = predictions[i].squeeze().cpu().numpy()
        nn_full, nn_context, nn_pred = nearest_neighbors[i]
        
        # Plot generated trajectory (context + prediction)
        ax = axes[i, 0]
        time_context = np.arange(CONTEXT_LENGTH)
        time_pred = np.arange(CONTEXT_LENGTH, MAX_LENGTH)
        
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=2, 
               label='Context q1', markersize=3)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=2, 
               label='Context q2', markersize=3)
        ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=2, 
               label='Generated q1')
        ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=2, 
               label='Generated q2')
        ax.axvline(x=CONTEXT_LENGTH, color='black', linestyle=':', linewidth=1.5)
        ax.set_title(f'Generated Trajectory {i+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot nearest neighbor from training set
        ax = axes[i, 1]
        ax.plot(time_context, nn_context[:, 0], 'o-', color='blue', linewidth=2, 
               label='Context q1', markersize=3)
        ax.plot(time_context, nn_context[:, 1], 'o-', color='red', linewidth=2, 
               label='Context q2', markersize=3)
        ax.plot(time_pred, nn_pred[:, 0], '-', color='cyan', linewidth=2, 
               label='Prediction q1')
        ax.plot(time_pred, nn_pred[:, 1], '-', color='orange', linewidth=2, 
               label='Prediction q2')
        ax.axvline(x=CONTEXT_LENGTH, color='black', linestyle=':', linewidth=1.5)
        ax.set_title(f'Nearest Neighbor (MSE={distances[i]:.4f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{trial_name} - Generated Trajectories vs Nearest Neighbors in Training Set', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = f"test_plots/{trial_name}_nearest_neighbors_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nNearest neighbor plot saved to: {save_path}")
    plt.show()
    
    # Compute RMSE on context
    print("\n" + "-"*60)
    print("Context RMSE Analysis:")
    print("-"*60)
    context_rmses = []
    for i, (nn_full, nn_context, nn_pred) in enumerate(nearest_neighbors):
        context_rmse = np.sqrt(np.mean((context_np - nn_context) ** 2))
        context_rmses.append(context_rmse)
        print(f"  Prediction {i+1}: Context RMSE = {context_rmse:.6f}")
    
    print(f"\nMean Context RMSE: {np.mean(context_rmses):.6f}")
    print(f"Std Context RMSE:  {np.std(context_rmses):.6f}")
    print("-"*60)
    
    # Statistics
    print("\n" + "-"*60)
    print("Nearest Neighbor Distance Statistics:")
    print(f"  Mean MSE: {np.mean(distances):.6f}")
    print(f"  Std MSE:  {np.std(distances):.6f}")
    print(f"  Min MSE:  {np.min(distances):.6f}")
    print(f"  Max MSE:  {np.max(distances):.6f}")
    print("-"*60)

# TEST: ATTENTION VISUALIZATION
def visualize_attention(model, diffusion, context, device=DEVICE, trial_name="Trial", 
                       layers_to_visualize=[0, 2, 5]):
    """
    Visualize attention weights from selected transformer layers.
    
    Args:
        model: Trained model
        diffusion: Diffusion process
        context: Context tensor (1, 20, 2)
        device: Device
        trial_name: Name for saving
        layers_to_visualize: Which layers to visualize (0-indexed)
    """
    print("\n" + "="*60)
    print(f"TEST: Attention Visualization - {trial_name}")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model.eval()
    context = context.to(device)
    
    # Start with random noise for prediction
    x_pred = torch.randn(1, PREDICTION_LENGTH, TRAJECTORY_DIM, device=device)
    
    # Get attention at middle timestep
    t = torch.tensor([NOISE_STEPS // 2], device=device)
    
    print(f"Extracting attention weights at timestep t={t.item()}...")
    
    with torch.no_grad():
        _, attention_weights = model(x_pred, context, t, return_attention=True)
    
    # Filter layers to visualize
    layers_to_visualize = [l for l in layers_to_visualize if l < len(attention_weights)]
    num_layers = len(layers_to_visualize)
    
    if num_layers == 0:
        print("No valid layers to visualize.")
        return
    
    print(f"Visualizing attention for {num_layers} layers: {layers_to_visualize}")
    
    # Create figure with subplots for each layer
    fig, axes = plt.subplots(1, num_layers, figsize=(6*num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for idx, layer_idx in enumerate(layers_to_visualize):
        attn = attention_weights[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        
        # Average over heads
        attn_avg = attn.mean(dim=1).squeeze(0).cpu().numpy()  # (seq_len, seq_len)
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(attn_avg, cmap='viridis', aspect='auto')
        ax.set_title(f'Layer {layer_idx} Attention', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'{trial_name} - Transformer Attention Weights (averaged over heads)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = f"test_plots/{trial_name}_attention_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attention visualization saved to: {save_path}")
    plt.show()
    
    # Also visualize attention for all heads of one layer
    print(f"\nVisualizing all heads for layer {layers_to_visualize[0]}...")
    
    attn_layer = attention_weights[layers_to_visualize[0]].squeeze(0).cpu().numpy()  # (num_heads, seq_len, seq_len)
    num_heads = attn_layer.shape[0]
    
    # Create grid for all heads
    cols = 4
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        im = ax.imshow(attn_layer[head_idx], cmap='viridis', aspect='auto')
        ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{trial_name} - Layer {layers_to_visualize[0]} All Attention Heads', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    save_path = f"test_plots/{trial_name}_attention_all_heads_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"All heads visualization saved to: {save_path}")
    plt.show()

# CREATE SUMMARY COMPARISON PLOT
def create_summary_plot(all_test_results):
    """
    Create a summary plot showing statistics across all trials and starting points.
    """
    print("\n" + "="*50)
    print("Creating summary statistics plot...")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x_pos = np.arange(len(trial_names))
    
    # Plot 1: MSE
    axes[0].bar(x_pos, all_mse_means, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Trial @ Starting Frame', fontsize=12)
    axes[0].set_ylabel('Mean MSE with Ground Truth', fontsize=12)
    axes[0].set_title('Prediction Accuracy Across Trials', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(trial_names, rotation=45, ha='right', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Diversity
    axes[1].bar(x_pos, all_diversity_means, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Trial @ Starting Frame', fontsize=12)
    axes[1].set_ylabel('Mean Pairwise MSE (Diversity)', fontsize=12)
    axes[1].set_title('Prediction Diversity Across Trials', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(trial_names, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('S18 - Summary Statistics Across All Tests', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = f"test_plots/summary_statistics_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Summary statistics plot saved to: {save_path}")
    plt.show()

# MAIN TEST FUNCTION
def main():
    """
    Main test function for conditional generation on S18 Trial00, Trial05, Trial10
    """
    print("="*50)
    print("CONDITIONAL DIFFUSION TRANSFORMER MODEL TESTING")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Context length: {CONTEXT_LENGTH} frames")
    print(f"Prediction length: {PREDICTION_LENGTH} frames")
    
    # LOAD MODEL
    model_path = "trained_models/conditional_diffusion_transformer_20251121_epochs_1000.pth"  # UPDATE THIS!
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please update the model_path variable with your trained model path.")
        return
    
    model, config, diffusion = load_trained_model(model_path, device=DEVICE)
    
    # LOAD TRAINING DATASET FOR NEAREST NEIGHBOR ANALYSIS
    print("\n" + "="*60)
    print("Loading training dataset for nearest neighbor analysis...")
    print("="*60)
    train_data = load_train_dataset(base_dir="data", subjects=range(1, 15), 
                                    max_trajectories=500)
    
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
        
        # TEST 1: Conditional generation at multiple starts
        trial_stats = test_conditional_generation_multi_start(
            model, diffusion, subsequences,
            num_samples=10, device=DEVICE, 
            trial_name=trial_file.replace('.csv', '')
        )
        
        # TEST 2: Nearest neighbor analysis
        test_nearest_neighbors(
            model, diffusion, subsequences, train_data,
            num_samples=6, device=DEVICE,
            trial_name=trial_file.replace('.csv', '')
        )
        
        # TEST 3: Attention visualization (only for first trial)
        if trial_file == "Trial00.csv":
            context, _, _ = subsequences[0]
            visualize_attention(
                model, diffusion, context, device=DEVICE,
                trial_name=trial_file.replace('.csv', ''),
                layers_to_visualize=[0, 2, 5]  # First, middle, last layers
            )
        
        all_test_results.append({
            'trial': trial_file,
            'stats': trial_stats,
            'subsequences': subsequences
        })
    
    # Summary plots
    if len(all_test_results) > 0:
        create_summary_plot(all_test_results)
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print(f"Tested on {len(all_test_results)} trajectories")
    print(f"Each trajectory tested at {num_starting_points} starting points")
    print("\nTests performed:")
    print("  1. Multi-start conditional generation (5 starting points per trial)")
    print("  2. Nearest neighbor analysis (comparing with training set)")
    print("  3. Attention visualization (transformer attention patterns)")
    print("\nResults saved in: test_plots/")

if __name__ == "__main__":
    main()
