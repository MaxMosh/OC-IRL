# CONDITIONAL DIFFUSION TRAINING SCRIPT
# MODEL USED FOR LEARNING NOISE: RNN, CONDITIONED TO A PREFIX
# CONDITION ENCODING: MLP OVER THE 20 FRAMES PREFIX
# DATA USED: BERRET'S RECORDING OF Q1 AND Q2 (ARM REACHING IN A 2D SPACE, FIXED FINAL ABSCISSE)
# GENERATED DATA: NEXT 80 FRAMES OF THE MOVEMENT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# PARAMETERS
TRAJECTORY_DIM = 2                                  # joints q1 and q2
CONTEXT_LENGTH = 20                                 # number of conditioning frames
PREDICTION_LENGTH = 80                              # number of frames to predict
MAX_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH     # total length = 100
NOISE_STEPS = 1000                                  # number of noise steps for diffusion
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING THE DATA WITH SUBSEQUENCE AUGMENTATION
class ConditionalTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, context_length=CONTEXT_LENGTH, 
                 prediction_length=PREDICTION_LENGTH, use_subsequences=True,
                 subsequence_stride=10):
        """
        Load trajectory data for conditional generation with subsequence augmentation.
        Each trajectory can be split into multiple subsequences to increase data diversity.
        
        Args:
            data_paths: list of paths to CSV files
            context_length: number of frames to use as context (20)
            prediction_length: number of frames to predict (80)
            use_subsequences: if True, extract multiple subsequences from each trajectory
            subsequence_stride: stride for sliding window when extracting subsequences
            TODO: A COMPARISON SHOULD BE PERFORMED BETWEEN OVERLAPPING AND NON-OVERLAPPING SUBSAMPLING
        
        Attributes:
            contexts: array of context trajectories (first 20 frames)
            predictions: array of prediction trajectories (next 80 frames)
            full_trajectories: array of complete trajectories (for visualization)
            TODO: A TEST SHOULD BE PERFORMED ON A LONGER CONTEXT
        """
        self.contexts = []                                      # then redefined as an array
        self.predictions = []                                   # then redefined as an array
        self.full_trajectories = []                             # then redefined as an array
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length
        self.use_subsequences = use_subsequences
        self.subsequence_stride = subsequence_stride
        
        for path in tqdm(data_paths, desc="Loading trajectories"):
            try:
                # NOTE: Berret's data are suprisingly stored in two lines q1 and q2, we thus transpose the data
                df = pd.read_csv(path, header=None)     # header=None, otherwise we lose q1
                trajectory = df.T.values                # transpose to get (time_steps, 2)
                
                if use_subsequences:
                    # Extract multiple subsequences from each trajectory (data augmentation)
                    # NOTE: as the task asked to be performed during the experiment is very simple,
                    # subsampling here allows us to predict the end of the trajectory not only from
                    # the beginning of the arm reaching (taking only the beginning of the sequence would
                    # be very poor in diversity)
                    for start_idx in range(0, len(trajectory) - self.total_length + 1, subsequence_stride):
                        end_idx = start_idx + self.total_length
                        
                        if end_idx <= len(trajectory):
                            subseq = trajectory[start_idx:end_idx]
                            
                            context = subseq[:context_length]
                            prediction = subseq[context_length:self.total_length]
                            
                            self.contexts.append(context)
                            self.predictions.append(prediction)
                            self.full_trajectories.append(subseq)
                else:
                    # Original behavior: only use the beginning of each trajectory
                    if len(trajectory) >= self.total_length:
                        context = trajectory[:context_length]
                        prediction = trajectory[context_length:self.total_length]
                        full_traj = trajectory[:self.total_length]
                        
                        self.contexts.append(context)
                        self.predictions.append(prediction)
                        self.full_trajectories.append(full_traj)
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        self.contexts = np.array(self.contexts, dtype=np.float32)
        self.predictions = np.array(self.predictions, dtype=np.float32)
        self.full_trajectories = np.array(self.full_trajectories, dtype=np.float32)
        
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.contexts[idx]),
            torch.from_numpy(self.predictions[idx]),
            torch.from_numpy(self.full_trajectories[idx])
        )

def load_trajectories_by_subjects(base_dir="data", subjects=None):
    """
    Load trajectory files from specific subject directories.
    
    Args:
        base_dir: Base directory containing S01-S20 folders
        subjects: List of subject numbers to load (e.g., [1, 2, 3] for S01, S02, S03)
        
    Returns:
        List of CSV file paths
    """
    file_paths = []
    
    if subjects is None:
        subjects = range(1, 21)  # All subjects S01 to S20
    
    for i in subjects:
        subject_dir = os.path.join(base_dir, f"S{i:02d}")
        
        if not os.path.exists(subject_dir):
            print(f"Warning: {subject_dir} does not exist")
            continue
        
        csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_paths.append(os.path.join(subject_dir, csv_file))
    
    print(f"Found {len(file_paths)} trajectory files for subjects {subjects}")
    return file_paths

# CONDITIONAL RNN MODEL
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
            x: (batch, seq_len, dim) - noisy prediction sequence
            t: (batch, time_dim) - time embedding
            c: (batch, context_dim) - context embedding
        """
        # Add time and context embeddings
        t_emb = self.time_mlp(t).unsqueeze(1)  # (batch, 1, dim)
        c_emb = self.context_mlp(c).unsqueeze(1)  # (batch, 1, dim)
        
        # Apply RNN
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, dim)
        
        # Residual connection with time and context conditioning
        h = self.norm(rnn_out)
        h = h + t_emb + c_emb
        return x + h

class ConditionalDiffusionRNN(nn.Module):
    """
    RNN-based diffusion model with conditioning on context frames.
    Predicts noise for the prediction part only, conditioned on the context.
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
        
        # Context encoder - processes the conditioning frames
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
        
        # Input projection for prediction part
        self.input_proj = nn.Linear(trajectory_dim, hidden_dim)
        
        # RNN blocks with context conditioning
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
            x_noisy: (batch, prediction_length, trajectory_dim) - noisy prediction part
            context: (batch, context_length, trajectory_dim) - clean context frames
            t: (batch,) - timestep
            
        Returns:
            (batch, prediction_length, trajectory_dim) - predicted noise
        """
        batch_size = x_noisy.shape[0]
        
        # Encode context
        context_flat = context.view(batch_size, -1)  # (batch, context_length * trajectory_dim)
        context_emb = self.context_encoder(context_flat)  # (batch, hidden_dim)
        
        # Time embedding
        t = t.unsqueeze(-1).float()
        t_emb = self.pos_encoding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Process noisy prediction through network
        h = self.input_proj(x_noisy)  # (batch, prediction_length, hidden_dim)
        
        for block in self.blocks:
            h = block(h, t_emb, context_emb)
        
        output = self.output_proj(h)  # (batch, prediction_length, trajectory_dim)
        
        return output

# CONDITIONAL DIFFUSION PROCESS
class ConditionalDiffusion:
    def __init__(self, noise_steps=NOISE_STEPS, beta_start=0.0001, beta_end=0.02, 
                 prediction_length=PREDICTION_LENGTH, device='cuda'):
        """
        Initialize the Conditional Diffusion process.
        Only the prediction part is noised, the context remains clean.
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

    def forward_diffusion(self, x_pred, t):
        """
        Add noise to the prediction part only (context remains clean).
        
        Args:
            x_pred: (batch, prediction_length, trajectory_dim) - clean prediction part
            t: (batch,) - timestep values
            
        Returns:
            noised_prediction, noise
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t]).view(-1, 1, 1)
        
        noise = torch.randn_like(x_pred, device=self.device)
        noised_prediction = sqrt_alpha_hat * x_pred + sqrt_one_minus_alpha_hat * noise
        
        return noised_prediction, noise

    def reverse_diffusion(self, model, context, x=None, verbose=True, num_samples=1):
        """
        Reverse diffusion process to generate multiple prediction continuations.
        
        Args:
            model: The trained model
            context: (batch, context_length, trajectory_dim) - conditioning frames
            x: Initial noise for prediction part (if None, starts from Gaussian noise)
            verbose: Whether to show progress bar
            num_samples: Number of different predictions to generate for each context
            
        Returns:
            List of denoised predictions, one for each sample
        """
        model.eval()
        
        batch_size = context.shape[0]
        all_predictions = []
        
        # Generate multiple samples for the same context
        for sample_idx in range(num_samples):
            # Start with random noise for prediction part
            if x is None:
                x_pred = torch.randn(batch_size, self.prediction_length, 
                                    context.shape[2], device=self.device)
            else:
                x_pred = x.clone()
            
            iterator = reversed(range(1, self.noise_steps))
            if verbose and sample_idx == 0:  # Only show progress for first sample
                iterator = tqdm(list(iterator), desc=f"Reverse diffusion (sample {sample_idx+1}/{num_samples})")
            
            with torch.no_grad():
                for i in iterator:
                    t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                    
                    # Predict noise conditioned on context
                    predicted_noise = model(x_pred, context, t)
                    
                    # Denoise step
                    sqrt_alpha = torch.sqrt(self.alpha[i])
                    one_minus_alpha = 1.0 - self.alpha[i]
                    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[i])
                    
                    x_pred = (1 / sqrt_alpha) * (x_pred - one_minus_alpha * predicted_noise / sqrt_one_minus_alpha_hat)
                    
                    # Add noise (except for last step)
                    if i > 1:
                        noise = torch.randn_like(x_pred, device=self.device)
                        x_pred = x_pred + torch.sqrt(self.beta[i]) * noise
            
            all_predictions.append(x_pred.cpu().clone())
        
        return all_predictions

# TRAINING THE MODEL
def train(model, diffusion, train_loader, val_loader, device=DEVICE, 
          epochs=NUM_EPOCHS, learning_rate=LR):
    """
    Train the conditional diffusion model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                          desc=f"Epoch {epoch + 1}/{epochs}")
        
        for i, (contexts, predictions, _) in progress_bar:
            contexts = contexts.to(device)
            predictions = predictions.to(device)
            
            # Generate random diffusion steps
            t = torch.randint(0, diffusion.noise_steps, (contexts.shape[0],), device=device)
            
            # Forward diffusion on prediction part only
            noised_predictions, true_noise = diffusion.forward_diffusion(predictions, t)
            
            optimizer.zero_grad()
            
            # Predict noise conditioned on context
            predicted_noise = model(noised_predictions, contexts, t)
            
            # Calculate loss
            loss = criterion(predicted_noise, true_noise)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for contexts, predictions, _ in val_loader:
                contexts = contexts.to(device)
                predictions = predictions.to(device)
                
                t = torch.randint(0, diffusion.noise_steps, (contexts.shape[0],), device=device)
                noised_predictions, true_noise = diffusion.forward_diffusion(predictions, t)
                predicted_noise = model(noised_predictions, contexts, t)
                
                loss = criterion(predicted_noise, true_noise)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {running_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
    
    return model, train_losses, val_losses

# VISUALIZATION
def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def visualize_conditional_generation(model, diffusion, val_dataset, num_contexts=3, 
                                     num_samples=10, device=DEVICE):
    """
    Visualize conditional generation: show context and multiple possible continuations.
    """
    model.eval()
    
    # Select random examples from validation set
    indices = np.random.choice(len(val_dataset), num_contexts, replace=False)
    
    fig, axes = plt.subplots(num_contexts, 1, figsize=(15, 5*num_contexts))
    if num_contexts == 1:
        axes = [axes]
    
    for idx, data_idx in enumerate(indices):
        context, true_prediction, full_traj = val_dataset[data_idx]
        context = context.unsqueeze(0).to(device)
        
        # Generate multiple predictions
        predictions = diffusion.reverse_diffusion(model, context, verbose=False, 
                                                 num_samples=num_samples)
        
        # Plot
        ax = axes[idx]
        
        # Plot context (first 20 frames)
        context_np = context.squeeze().cpu().numpy()
        time_context = np.arange(CONTEXT_LENGTH)
        ax.plot(time_context, context_np[:, 0], 'o-', color='blue', linewidth=2, 
               label='Context q1', markersize=3)
        ax.plot(time_context, context_np[:, 1], 'o-', color='red', linewidth=2, 
               label='Context q2', markersize=3)
        
        # Plot ground truth continuation
        true_pred_np = true_prediction.cpu().numpy()
        time_pred = np.arange(CONTEXT_LENGTH, CONTEXT_LENGTH + PREDICTION_LENGTH)
        ax.plot(time_pred, true_pred_np[:, 0], '--', color='blue', linewidth=2, 
               alpha=0.5, label='True q1')
        ax.plot(time_pred, true_pred_np[:, 1], '--', color='red', linewidth=2, 
               alpha=0.5, label='True q2')
        
        # Plot generated predictions
        for i, pred in enumerate(predictions):
            pred_np = pred.squeeze().cpu().numpy()
            alpha = 0.3 if i > 0 else 0.6
            label_q1 = 'Generated q1' if i == 0 else None
            label_q2 = 'Generated q2' if i == 0 else None
            ax.plot(time_pred, pred_np[:, 0], '-', color='cyan', linewidth=1, 
                   alpha=alpha, label=label_q1)
            ax.plot(time_pred, pred_np[:, 1], '-', color='orange', linewidth=1, 
                   alpha=alpha, label=label_q2)
        
        # Add vertical line to separate context and prediction
        ax.axvline(x=CONTEXT_LENGTH, color='black', linestyle=':', linewidth=2, 
                  label='Prediction start')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle')
        ax.set_title(f'Conditional Generation Example {idx+1}: {num_samples} Predictions')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# MAIN
def main():
    print(f"Using device: {DEVICE}")
    print(f"Context length: {CONTEXT_LENGTH}, Prediction length: {PREDICTION_LENGTH}")
    
    # Load data - S01 to S14 for training, S15 for validation
    print("\nLoading Data...")
    print("Training with subsequence augmentation enabled (stride=10)")
    
    train_paths = load_trajectories_by_subjects("data", subjects=range(1, 15))  # S01-S14
    val_paths = load_trajectories_by_subjects("data", subjects=[15])  # S15
    
    # Training dataset with subsequence augmentation
    train_dataset = ConditionalTrajectoryDataset(
        train_paths, 
        use_subsequences=True,
        subsequence_stride=10  # Extract subsequence every 10 frames
    )
    
    # Validation dataset without augmentation (only beginning of trajectories)
    val_dataset = ConditionalTrajectoryDataset(
        val_paths,
        use_subsequences=False
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training trajectories: {len(train_dataset)}")
    print(f"Validation trajectories: {len(val_dataset)}")
    
    # Initialize model and diffusion
    print("\nInitializing Model...")
    model = ConditionalDiffusionRNN(
        trajectory_dim=TRAJECTORY_DIM,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        hidden_dim=256,
        num_layers=6,
        time_dim=128
    ).to(DEVICE)
    
    diffusion = ConditionalDiffusion(
        noise_steps=NOISE_STEPS,
        prediction_length=PREDICTION_LENGTH,
        device=DEVICE
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Train model
    print("\nTraining...")
    trained_model, train_losses, val_losses = train(
        model, diffusion, train_loader, val_loader,
        device=DEVICE, epochs=NUM_EPOCHS, learning_rate=LR
    )
    
    # Plot losses
    plot_losses(train_losses, val_losses, save_path='conditional_training_losses.png')
    
    # Save model
    print("\nSaving Model...")
    os.makedirs("trained_models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"trained_models/conditional_diffusion_{timestamp}.pth"
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': {
            'trajectory_dim': TRAJECTORY_DIM,
            'context_length': CONTEXT_LENGTH,
            'prediction_length': PREDICTION_LENGTH,
            'noise_steps': NOISE_STEPS,
            'hidden_dim': 256,
            'num_layers': 6,
            'time_dim': 128
        },
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Visualize results
    print("\nGenerating Visualizations...")
    fig = visualize_conditional_generation(
        trained_model, diffusion, val_dataset, 
        num_contexts=3, num_samples=10, device=DEVICE
    )
    fig.savefig(f'conditional_generation_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nTraining Complete!")

if __name__ == "__main__":
    main()
