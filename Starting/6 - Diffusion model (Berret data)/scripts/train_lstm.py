# TRYING TO SOLVE THE PREDICTION OF MOVEMENT IN A MORE CLASSICAL WAY, WITH A LSTM
# DATA USED: BERRET'S RECORDING OF Q1 AND Q2 (ARM REACHING IN A 2D SPACE, FIXED FINAL ABSCISSE)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# PARAMETERS OF THE MODEL
TRAJECTORY_DIM = 2                                                                  # q1 and q2
CONTEXT_LENGTH = 10  # Number of timesteps to use as context for prediction         # context given to the model to guess the end of the trajectory ; INITIALLY 20
PREDICTION_LENGTH = 40  # Number of timesteps to predict                            # lenght guessed by th model ; INITIALLY 10
BATCH_SIZE = 32
NUM_EPOCHS = 10                                                                     # INITIALLY 1000, THEN 10
LR = 1e-3
VALIDATION_SPLIT = 0.2                                                              # 20% for validation dataset, the rest is for the training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING TBE DATA
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, context_length=CONTEXT_LENGTH, prediction_length=PREDICTION_LENGTH):
        """
        Load trajectory data from multiple CSV files.
        Creates samples by sliding window over each trajectory.
        IMPORTANT: our data is strangely arranged in two lines, the first is q1 and the second is q2. Thus, the data is transposed during the loading.
        
        Args:
            data_paths: List of paths to CSV files
            context_length: Number of timesteps to use as input
            prediction_length: Number of timesteps to predict
        """
        self.context_samples = []
        self.target_samples = []
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        total_window_size = context_length + prediction_length
        
        for path in tqdm(data_paths, desc="Loading trajectories"):
            try:
                df = pd.read_csv(path, header=None)                             # load CSV without header (IMPORTANT, otherwise we lose q1)
                trajectory = df.T.values                                        # transposing the data
                
                # Create sliding windows if trajectory is long enough
                if len(trajectory) >= total_window_size:
                    # Create multiple samples from this trajectory using sliding window
                    # NOTE: could be changed to get less subtrajectories
                    for start_idx in range(len(trajectory) - total_window_size + 1):
                        context = trajectory[start_idx:start_idx + context_length]
                        target = trajectory[start_idx + context_length:start_idx + total_window_size]
                        
                        self.context_samples.append(context)
                        self.target_samples.append(target)
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        self.context_samples = np.array(self.context_samples, dtype=np.float32)
        self.target_samples = np.array(self.target_samples, dtype=np.float32)
        
        print(f"Created {len(self.context_samples)} training samples")
        
    def __len__(self):
        return len(self.context_samples)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.context_samples[idx]),
            torch.from_numpy(self.target_samples[idx])
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
    
    for i in range(1, 16):                                                                  # INITIALLY 1, 21, SHORTENED TO KEEP A TEST DATASET
        subject_dir = os.path.join(base_dir, f"S{i:02d}")
        
        if not os.path.exists(subject_dir):
            print(f"Warning: {subject_dir} does not exist")
            continue
        
        # Listing all CSV file in the subdirectory SXX
        csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_paths.append(os.path.join(subject_dir, csv_file))
    
    print(f"Found {len(file_paths)} trajectory files")
    return file_paths

def create_train_val_split(file_paths, validation_split=VALIDATION_SPLIT, random_seed=42):
    """
    Split file paths into train and validation sets.
    
    Args:
        file_paths: List of all file paths
        validation_split: Fraction of data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        train_paths, val_paths
    """
    train_paths, val_paths = train_test_split(
        file_paths, 
        test_size=validation_split, 
        random_state=random_seed
    )
    
    print(f"Train set: {len(train_paths)} files")
    print(f"Validation set: {len(val_paths)} files")
    
    return train_paths, val_paths

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

# TRAINING THE MODEL
def train_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [Train]", leave=False)
    
    for context, target in progress_bar:
        # print(f"Shape of context sequence: {context.shape}")                            # ADDED
        context = context.to(device)
        target = target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(context)
        # print(f"Shape of prediction sequence: {predictions.shape}")                         # ADDED
        
        # Calculate loss
        # MODIFIED: THE SECOND TERM HAS BEEN ADDED TO CORRECT NON-SMOOTHNESS OF THE TRANSITION BETWEEN CONTEXT AND PREDICTION
        loss = criterion(predictions, target) + criterion(context[:,-1,:], predictions[:,0,:])
        # loss = criterion(predictions, target) + criterion(context[:,-2:,:], predictions[:,:2,:])
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * context.shape[0]
        
        # Update progress bar
        current_avg_loss = running_loss / ((progress_bar.n + 1) * context.shape[0])
        progress_bar.set_postfix(loss=f"{current_avg_loss:.6f}")
    
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device, epoch_num):
    """
    Validate for one epoch.
    """
    model.eval()
    running_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num} [Val]  ", leave=False)
    
    with torch.no_grad():
        for context, target in progress_bar:
            context = context.to(device)
            target = target.to(device)
            
            # Forward pass
            predictions = model(context)
            
            # Calculate loss
            loss = criterion(predictions, target)
            
            running_loss += loss.item() * context.shape[0]
            
            # Update progress bar
            current_avg_loss = running_loss / ((progress_bar.n + 1) * context.shape[0])
            progress_bar.set_postfix(loss=f"{current_avg_loss:.6f}")
    
    return running_loss / len(dataloader.dataset)

def train(model, train_loader, val_loader, device=DEVICE, epochs=NUM_EPOCHS, learning_rate=LR):
    """
    Train the LSTM model with validation.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10         # INITIALLY 30 FOR 1000 MAX_EPOCHS
    
    print(f"\nStarting training for up to {epochs} epochs...")
    print(f"Early stopping patience: {early_stop_patience} epochs")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch + 1)
        val_losses.append(val_loss)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            status = "-> NEW BEST"
        else:
            patience_counter += 1
            status = f"(patience: {patience_counter}/{early_stop_patience})"
        
        if new_lr < old_lr:
            status += f" | LR: {old_lr:.2e} → {new_lr:.2e}"
        
        print(f"Epoch {epoch + 1:4d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {status}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n{'='*70}")
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"{'='*70}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses

# EVALUATE THE MODEL
def evaluate_model(model, dataloader, device=DEVICE):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: Trained LSTM model
        dataloader: DataLoader for evaluation
        device: Device to run on
        
    Returns:
        avg_loss, avg_mse_per_step
    """
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    mse_per_step = np.zeros(PREDICTION_LENGTH)
    num_samples = 0
    
    with torch.no_grad():
        for context, target in tqdm(dataloader, desc="Evaluating"):
            context = context.to(device)
            target = target.to(device)
            
            predictions = model(context)
            loss = criterion(predictions, target)
            
            total_loss += loss.item() * context.shape[0]
            
            # Calculate MSE for each prediction step
            for t in range(PREDICTION_LENGTH):
                step_mse = torch.mean((predictions[:, t, :] - target[:, t, :]) ** 2).item()
                mse_per_step[t] += step_mse * context.shape[0]
            
            num_samples += context.shape[0]
    
    avg_loss = total_loss / num_samples
    mse_per_step = mse_per_step / num_samples
    
    return avg_loss, mse_per_step

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
    
    # Add batch dimension
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
    
    context_np = context.numpy() if isinstance(context, torch.Tensor) else context
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

# ==================== VISUALIZATION ====================
def plot_predictions(model, dataset, num_samples=6, device=DEVICE):
    """
    Plot predictions vs ground truth for visualization.
    """
    model.eval()
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(indices):
        context, target = dataset[sample_idx]
        
        # Predict
        predictions = predict_trajectory(model, context, PREDICTION_LENGTH, device)
        
        # Convert to numpy
        context_np = context.numpy()
        target_np = target.numpy()
        
        # Create time axis
        context_time = np.arange(len(context_np))
        pred_time = np.arange(len(context_np), len(context_np) + PREDICTION_LENGTH)
        
        # Plot q1
        axes[idx].plot(context_time, context_np[:, 0], 'b-', label='Context q1', linewidth=2)
        axes[idx].plot(pred_time, target_np[:, 0], 'g-', label='True q1', linewidth=2)
        axes[idx].plot(pred_time, predictions[:, 0], 'g--', label='Pred q1', linewidth=2, alpha=0.7)
        
        # Plot q2
        axes[idx].plot(context_time, context_np[:, 1], 'r-', label='Context q2', linewidth=2)
        axes[idx].plot(pred_time, target_np[:, 1], 'orange', label='True q2', linewidth=2)
        axes[idx].plot(pred_time, predictions[:, 1], 'orange', linestyle='--', label='Pred q2', 
                      linewidth=2, alpha=0.7)
        
        # Add vertical line to separate context and prediction
        axes[idx].axvline(x=len(context_np)-0.5, color='black', linestyle=':', linewidth=1.5)
        
        axes[idx].set_title(f'Sample {idx+1}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time Step')
        axes[idx].set_ylabel('Joint Angle')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Trajectory Predictions (Context | Prediction)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_training_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    
    # Find best validation loss
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = np.min(val_losses)
    
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best Val (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_mse_per_step(mse_per_step):
    """
    Plot MSE for each prediction step.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    steps = np.arange(1, len(mse_per_step) + 1)
    ax.bar(steps, mse_per_step, alpha=0.7, color='steelblue')
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error per Prediction Step (Validation Set)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

# MAIN
def main():
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("\nLoading Data...")
    file_paths = load_all_trajectories("data")
    
    # Create train/val split
    # NOTE: train and val data are not splitted by subject
    train_paths, val_paths = create_train_val_split(file_paths, VALIDATION_SPLIT)
    
    # Create datasets
    print("\nCreating Datasets...")
    train_dataset = TrajectoryDataset(
        train_paths, 
        context_length=CONTEXT_LENGTH, 
        prediction_length=PREDICTION_LENGTH
    )
    val_dataset = TrajectoryDataset(
        val_paths, 
        context_length=CONTEXT_LENGTH, 
        prediction_length=PREDICTION_LENGTH
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Context length: {CONTEXT_LENGTH}, Prediction length: {PREDICTION_LENGTH}")
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = TrajectoryLSTM(
        input_dim=TRAJECTORY_DIM,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2,
        prediction_length=PREDICTION_LENGTH
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Train model
    print("\n=== Training ===")
    trained_model, train_losses, val_losses = train(
        model, train_loader, val_loader, device=DEVICE, epochs=NUM_EPOCHS, learning_rate=LR
    )
    
    # Evaluate model on validation set
    print("\n=== Final Validation ===")
    val_loss, mse_per_step = evaluate_model(trained_model, val_loader, device=DEVICE)
    print(f"Final validation loss: {val_loss:.6f}")
    print(f"MSE per step: {mse_per_step}")
    
    # Plot training curves
    print("\n=== Plotting Results ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    train_fig = plot_training_curves(train_losses, val_losses)
    train_fig.savefig(f'training_validation_curves_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot MSE per step
    mse_fig = plot_mse_per_step(mse_per_step)
    mse_fig.savefig(f'mse_per_step_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot predictions on validation set
    pred_fig = plot_predictions(trained_model, val_dataset, num_samples=6, device=DEVICE)
    pred_fig.savefig(f'validation_predictions_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save model
    print("\n=== Saving Model ===")
    os.makedirs("trained_models", exist_ok=True)
    model_path = f"trained_models/lstm_trajectory_{timestamp}.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'input_dim': TRAJECTORY_DIM,
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.2,
            'prediction_length': PREDICTION_LENGTH,
            'context_length': CONTEXT_LENGTH
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'mse_per_step': mse_per_step.tolist(),
        'final_val_loss': val_loss
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Test long-term prediction
    print("\n=== Testing Long-term Prediction ===")
    test_context, _ = val_dataset[0]
    long_pred = predict_long_term(trained_model, test_context, num_steps=50, device=DEVICE)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(long_pred[:, 0], label='q1', linewidth=2)
    axes[0].axvline(x=CONTEXT_LENGTH-0.5, color='red', linestyle='--', 
                   label='Context end', linewidth=2)
    axes[0].set_title('Long-term Prediction - q1')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Joint Angle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(long_pred[:, 1], label='q2', linewidth=2, color='orange')
    axes[1].axvline(x=CONTEXT_LENGTH-0.5, color='red', linestyle='--', 
                   label='Context end', linewidth=2)
    axes[1].set_title('Long-term Prediction - q2')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Joint Angle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Long-term Trajectory Prediction (50 steps)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'long_term_prediction_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== Complete ===")
    print(f"\nFinal Results:")
    print(f"  Best Validation Loss: {min(val_losses):.6f}")
    print(f"  Final Validation Loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()