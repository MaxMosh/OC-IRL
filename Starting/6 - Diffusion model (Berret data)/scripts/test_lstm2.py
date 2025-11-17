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
CONTEXT_LENGTH = 20  # Number of timesteps to use as context for prediction
PREDICTION_LENGTH = 10  # Number of timesteps to predict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== LSTM MODEL ====================
class TrajectoryLSTM(nn.Module):
    """
    LSTM model for trajectory prediction.
    Takes a sequence of past timesteps and predicts future timesteps.
    """
    def __init__(self, input_dim=TRAJECTORY_DIM, hidden_dim=256, num_layers=3, 
                 dropout=0.2, prediction_length=PREDICTION_LENGTH):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, predict_steps=None):
        """
        Args:
            x: (batch, seq_len, input_dim) - context sequence
            predict_steps: Number of steps to predict (defaults to prediction_length)
            
        Returns:
            (batch, predict_steps, input_dim) - predicted sequence
        """
        if predict_steps is None:
            predict_steps = self.prediction_length
            
        batch_size = x.shape[0]
        
        # Process context through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state to start prediction
        predictions = []
        
        # Get the last output from context
        current_input = lstm_out[:, -1:, :]  # (batch, 1, hidden_dim)
        
        # Autoregressive prediction
        for _ in range(predict_steps):
            # Project to output dimension
            pred = self.output_proj(current_input)  # (batch, 1, input_dim)
            predictions.append(pred)
            
            # Use prediction as input for next step
            next_input, (hidden, cell) = self.lstm(pred, (hidden, cell))
            current_input = next_input
        
        # Concatenate all predictions
        predictions = torch.cat(predictions, dim=1)  # (batch, predict_steps, input_dim)
        
        return predictions

# ==================== LOAD MODEL ====================
def load_trained_model(model_path, device=DEVICE):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        
    Returns:
        model, config
    """
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    
    # Initialize model
    model = TrajectoryLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        prediction_length=config['prediction_length']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Configuration: {config}")
    
    if 'final_val_loss' in checkpoint:
        print(f"Training validation loss: {checkpoint['final_val_loss']:.6f}")
    
    return model, config

# ==================== LOAD TEST DATA ====================
def load_test_trajectories(test_dir="data/S21"):
    """
    Load all trajectory files from test directory (S21).
    
    Args:
        test_dir: Directory containing test trajectories
        
    Returns:
        List of file paths
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    file_paths = []
    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        file_paths.append(os.path.join(test_dir, csv_file))
    
    print(f"Found {len(file_paths)} test trajectory files in {test_dir}")
    return file_paths

def load_trajectory(file_path):
    """
    Load a single trajectory from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        trajectory: (time_steps, 2) numpy array
    """
    df = pd.read_csv(file_path, header=None)
    trajectory = df.T.values  # Shape: (time_steps, 2)
    return trajectory

def create_test_samples(trajectory, context_length, prediction_length):
    """
    Create test samples from a trajectory using sliding window.
    
    Args:
        trajectory: (time_steps, 2) numpy array
        context_length: Number of timesteps for context
        prediction_length: Number of timesteps to predict
        
    Returns:
        List of (context, target) tuples
    """
    samples = []
    total_window_size = context_length + prediction_length
    
    if len(trajectory) >= total_window_size:
        for start_idx in range(len(trajectory) - total_window_size + 1):
            context = trajectory[start_idx:start_idx + context_length]
            target = trajectory[start_idx + context_length:start_idx + total_window_size]
            samples.append((context, target))
    
    return samples

# ==================== PREDICTION ====================
def predict_trajectory(model, context, num_steps, device=DEVICE):
    """
    Predict future trajectory given a context sequence.
    
    Args:
        model: Trained LSTM model
        context: (seq_len, trajectory_dim) - context sequence
        num_steps: Number of steps to predict
        device: Device to run on
        
    Returns:
        predictions: (num_steps, trajectory_dim) - predicted trajectory
    """
    model.eval()
    
    # Convert to tensor and add batch dimension
    if not isinstance(context, torch.Tensor):
        context = torch.from_numpy(context).float()
    context = context.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(context, predict_steps=num_steps)
    
    return predictions.squeeze(0).cpu().numpy()

def predict_long_term(model, context, num_steps, device=DEVICE):
    """
    Predict long-term trajectory by iteratively using predictions as context.
    
    Args:
        model: Trained LSTM model
        context: (context_length, trajectory_dim) - initial context
        num_steps: Total number of steps to predict
        device: Device to run on
        
    Returns:
        full_trajectory: (context_length + num_steps, trajectory_dim)
    """
    model.eval()
    
    context_np = context if isinstance(context, np.ndarray) else context.numpy()
    full_trajectory = [context_np]
    
    current_context = torch.from_numpy(context_np).float()
    
    with torch.no_grad():
        steps_predicted = 0
        while steps_predicted < num_steps:
            # Predict next chunk
            steps_to_predict = min(PREDICTION_LENGTH, num_steps - steps_predicted)
            predictions = predict_trajectory(model, current_context, steps_to_predict, device)
            
            full_trajectory.append(predictions)
            steps_predicted += steps_to_predict
            
            # Update context: use last CONTEXT_LENGTH steps
            current_context = torch.from_numpy(
                np.vstack([current_context.numpy(), predictions])[-CONTEXT_LENGTH:]
            ).float()
    
    return np.vstack(full_trajectory)

# ==================== EVALUATION ====================
def evaluate_test_set(model, test_paths, context_length, prediction_length, device=DEVICE):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained LSTM model
        test_paths: List of test file paths
        context_length: Context length
        prediction_length: Prediction length
        device: Device to run on
        
    Returns:
        results dictionary
    """
    model.eval()
    
    all_mses = []
    mse_per_step = np.zeros(prediction_length)
    num_samples = 0
    total_window_size = context_length + prediction_length
    
    skipped_files = 0
    trajectory_lengths = []
    
    print(f"\nRequired trajectory length: {total_window_size} (context={context_length} + pred={prediction_length})")
    print("Evaluating on test set...")
    
    for file_path in tqdm(test_paths, desc="Processing test files"):
        trajectory = load_trajectory(file_path)
        trajectory_lengths.append(len(trajectory))
        
        if len(trajectory) < total_window_size:
            skipped_files += 1
            continue
            
        samples = create_test_samples(trajectory, context_length, prediction_length)
        
        for context, target in samples:
            # Predict
            predictions = predict_trajectory(model, context, prediction_length, device)
            
            # Calculate MSE
            mse = np.mean((predictions - target) ** 2)
            all_mses.append(mse)
            
            # MSE per step
            for t in range(prediction_length):
                step_mse = np.mean((predictions[t] - target[t]) ** 2)
                mse_per_step[t] += step_mse
            
            num_samples += 1
    
    # Check if we have any samples
    if num_samples == 0:
        print("\n" + "!"*70)
        print("WARNING: No test samples could be created!")
        print("!"*70)
        print(f"\nTrajectory length statistics:")
        print(f"  Min length: {np.min(trajectory_lengths)}")
        print(f"  Max length: {np.max(trajectory_lengths)}")
        print(f"  Mean length: {np.mean(trajectory_lengths):.1f}")
        print(f"  Required length: {total_window_size}")
        print(f"  Files skipped (too short): {skipped_files}/{len(test_paths)}")
        print(f"\nSuggestions:")
        print(f"  1. Use shorter CONTEXT_LENGTH and PREDICTION_LENGTH")
        print(f"  2. Test on a different dataset with longer trajectories")
        print(f"  3. Use only trajectories longer than {total_window_size} timesteps")
        
        return None
    
    # Average MSE per step
    mse_per_step = mse_per_step / num_samples
    
    results = {
        'avg_mse': np.mean(all_mses),
        'std_mse': np.std(all_mses),
        'min_mse': np.min(all_mses),
        'max_mse': np.max(all_mses),
        'mse_per_step': mse_per_step,
        'num_samples': num_samples,
        'skipped_files': skipped_files,
        'total_files': len(test_paths),
        'trajectory_lengths': trajectory_lengths
    }
    
    if skipped_files > 0:
        print(f"\nNote: {skipped_files}/{len(test_paths)} files were too short (< {total_window_size} timesteps)")
    
    return results

# ==================== VISUALIZATION ====================
def plot_test_predictions(model, test_paths, context_length, prediction_length, 
                         num_samples=6, device=DEVICE):
    """
    Plot predictions on test samples.
    """
    model.eval()
    
    # Collect samples from test set
    all_samples = []
    total_window_size = context_length + prediction_length
    
    for file_path in test_paths:
        trajectory = load_trajectory(file_path)
        if len(trajectory) >= total_window_size:  # Only use trajectories long enough
            samples = create_test_samples(trajectory, context_length, prediction_length)
            all_samples.extend(samples)
    
    if len(all_samples) == 0:
        print(f"No test samples available for plotting (need trajectories >= {total_window_size} timesteps)")
        return None
    
    # Randomly select samples
    selected_indices = np.random.choice(len(all_samples), size=min(num_samples, len(all_samples)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(selected_indices):
        context, target = all_samples[sample_idx]
        
        # Predict
        predictions = predict_trajectory(model, context, prediction_length, device)
        
        # Calculate MSE
        mse = np.mean((predictions - target) ** 2)
        
        # Create time axis
        context_time = np.arange(len(context))
        pred_time = np.arange(len(context), len(context) + prediction_length)
        
        # Plot q1
        axes[idx].plot(context_time, context[:, 0], 'b-', label='Context q1', linewidth=2)
        axes[idx].plot(pred_time, target[:, 0], 'g-', label='True q1', linewidth=2)
        axes[idx].plot(pred_time, predictions[:, 0], 'g--', label='Pred q1', linewidth=2, alpha=0.7)
        
        # Plot q2
        axes[idx].plot(context_time, context[:, 1], 'r-', label='Context q2', linewidth=2)
        axes[idx].plot(pred_time, target[:, 1], 'orange', label='True q2', linewidth=2)
        axes[idx].plot(pred_time, predictions[:, 1], 'orange', linestyle='--', label='Pred q2', 
                      linewidth=2, alpha=0.7)
        
        # Add vertical line to separate context and prediction
        axes[idx].axvline(x=len(context)-0.5, color='black', linestyle=':', linewidth=1.5)
        
        axes[idx].set_title(f'Test Sample {idx+1} (MSE={mse:.6f})', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time Step')
        axes[idx].set_ylabel('Joint Angle')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Test Set Predictions (S21 - Unseen Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_mse_per_step(mse_per_step, title="MSE per Prediction Step"):
    """
    Plot MSE for each prediction step.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    steps = np.arange(1, len(mse_per_step) + 1)
    ax.bar(steps, mse_per_step, alpha=0.7, color='steelblue')
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_long_term_predictions(model, test_paths, context_length, num_predictions=3, 
                               prediction_steps=50, device=DEVICE):
    """
    Plot long-term predictions on test trajectories.
    """
    model.eval()
    
    total_length_needed = context_length + prediction_steps
    
    # Collect trajectories
    trajectories = []
    for path in test_paths:
        traj = load_trajectory(path)
        if len(traj) >= total_length_needed:
            trajectories.append(traj)
    
    if len(trajectories) == 0:
        print(f"No suitable trajectories for long-term prediction (need length >= {total_length_needed})!")
        return None
    
    # Select random trajectories
    selected_indices = np.random.choice(len(trajectories), size=min(num_predictions, len(trajectories)), replace=False)
    
    fig, axes = plt.subplots(num_predictions, 2, figsize=(15, 5*num_predictions))
    if num_predictions == 1:
        axes = axes.reshape(1, -1)
    
    for idx, traj_idx in enumerate(selected_indices):
        trajectory = trajectories[traj_idx]
        
        # Use first context_length as context
        context = trajectory[:context_length]
        ground_truth = trajectory[:total_length_needed]
        
        # Predict
        predicted = predict_long_term(model, context, prediction_steps, device)
        
        # Calculate MSE on prediction part
        mse = np.mean((predicted[context_length:] - ground_truth[context_length:]) ** 2)
        
        # Plot q1
        time_steps = np.arange(len(predicted))
        axes[idx, 0].plot(time_steps[:context_length], predicted[:context_length, 0], 
                         'b-', label='Context', linewidth=2)
        axes[idx, 0].plot(time_steps[context_length:], ground_truth[context_length:, 0], 
                         'g-', label='Ground Truth', linewidth=2)
        axes[idx, 0].plot(time_steps[context_length:], predicted[context_length:, 0], 
                         'g--', label='Prediction', linewidth=2, alpha=0.7)
        axes[idx, 0].axvline(x=context_length-0.5, color='red', linestyle='--', 
                           linewidth=1.5, alpha=0.5)
        axes[idx, 0].set_title(f'Long-term Prediction {idx+1} - q1 (MSE={mse:.6f})', 
                              fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('Time Step')
        axes[idx, 0].set_ylabel('Joint Angle')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot q2
        axes[idx, 1].plot(time_steps[:context_length], predicted[:context_length, 1], 
                         'r-', label='Context', linewidth=2)
        axes[idx, 1].plot(time_steps[context_length:], ground_truth[context_length:, 1], 
                         'orange', label='Ground Truth', linewidth=2)
        axes[idx, 1].plot(time_steps[context_length:], predicted[context_length:, 1], 
                         'orange', linestyle='--', label='Prediction', linewidth=2, alpha=0.7)
        axes[idx, 1].axvline(x=context_length-0.5, color='red', linestyle='--', 
                           linewidth=1.5, alpha=0.5)
        axes[idx, 1].set_title(f'Long-term Prediction {idx+1} - q2 (MSE={mse:.6f})', 
                              fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel('Time Step')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Long-term Predictions ({prediction_steps} steps) on Test Set', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# ==================== MAIN ====================
def main():
    """
    Main test function.
    """
    print("="*70)
    print("LSTM TRAJECTORY PREDICTION - TEST ON S21")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # ========== LOAD MODEL ==========
    model_path = "trained_models/lstm_trajectory_20251112_124811.pth"  # UPDATE THIS!
    
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        print("Please update the model_path variable with your trained model path.")
        return
    
    model, config = load_trained_model(model_path, device=DEVICE)
    
    context_length = config['context_length']
    prediction_length = config['prediction_length']
    
    # ========== LOAD TEST DATA ==========
    print("\n" + "="*70)
    print("Loading test data from S21...")
    print("="*70)
    
    try:
        test_paths = load_test_trajectories("data/S21")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure the data/S21 directory exists with test trajectories.")
        return
    
    if len(test_paths) == 0:
        print("No test files found!")
        return
    
    # ========== EVALUATE ON TEST SET ==========
    print("\n" + "="*70)
    print("TEST 1: Quantitative Evaluation")
    print("="*70)
    
    results = evaluate_test_set(model, test_paths, context_length, prediction_length, device=DEVICE)
    
    # Check if evaluation was successful
    if results is None:
        print("\nCannot proceed with visualization - no test samples available.")
        return
    
    print(f"\nTest Set Results (S21 - Unseen Data):")
    print(f"  Files processed: {results['total_files']}")
    print(f"  Files skipped (too short): {results['skipped_files']}")
    print(f"  Test samples created: {results['num_samples']}")
    print(f"  Average MSE: {results['avg_mse']:.6f} ± {results['std_mse']:.6f}")
    print(f"  Min MSE: {results['min_mse']:.6f}")
    print(f"  Max MSE: {results['max_mse']:.6f}")
    
    # ========== VISUALIZE PREDICTIONS ==========
    print("\n" + "="*70)
    print("TEST 2: Short-term Predictions")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("test_plots", exist_ok=True)
    
    pred_fig = plot_test_predictions(model, test_paths, context_length, prediction_length, 
                                     num_samples=6, device=DEVICE)
    if pred_fig:
        save_path = f"test_plots/test_predictions_{timestamp}.png"
        pred_fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
        plt.show()
    
    # ========== MSE PER STEP ==========
    print("\n" + "="*70)
    print("TEST 3: Error Analysis per Step")
    print("="*70)
    
    mse_fig = plot_mse_per_step(results['mse_per_step'], 
                                title="MSE per Prediction Step (Test Set - S21)")
    save_path = f"test_plots/test_mse_per_step_{timestamp}.png"
    mse_fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    plt.show()
    
    # ========== LONG-TERM PREDICTIONS ==========
    print("\n" + "="*70)
    print("TEST 4: Long-term Predictions (50 steps)")
    print("="*70)
    
    long_fig = plot_long_term_predictions(model, test_paths, context_length, 
                                         num_predictions=3, prediction_steps=50, device=DEVICE)
    if long_fig:
        save_path = f"test_plots/test_long_term_{timestamp}.png"
        long_fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
        plt.show()
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\nAll plots saved in: test_plots/")
    print(f"\nSummary:")
    print(f"  Test files processed: {len(test_paths)}")
    print(f"  Test samples evaluated: {results['num_samples']}")
    print(f"  Average test MSE: {results['avg_mse']:.6f}")

if __name__ == "__main__":
    main()