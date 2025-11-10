import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Configuration
class Config:
    data_dirs = [f"data/S{i:02d}" for i in range(1, 21)]
    sequence_length = 100  # Fixed length for all sequences
    batch_size = 32
    num_epochs = 1000
    lr = 1e-3
    T = 1000  # Number of diffusion steps
    beta_start = 1e-4
    beta_end = 0.02
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "trained_models"

# Create save directory if it doesn't exist
os.makedirs(Config.save_dir, exist_ok=True)

# Custom Dataset with padding/truncation
class TrajectoryDataset(Dataset):
    def __init__(self, data_dirs, sequence_length):
        self.sequence_length = sequence_length
        self.data = self.load_data(data_dirs)
        
    def load_data(self, data_dirs):
        all_data = []
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                continue
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(data_dir, file)
                    df = pd.read_csv(file_path, header=None)
                    trajectory = df.values.astype(np.float32)
                    
                    # Handle variable length sequences
                    if len(trajectory) > self.sequence_length:
                        # Truncate
                        trajectory = trajectory[:self.sequence_length]
                    elif len(trajectory) < self.sequence_length:
                        # Pad with zeros
                        pad_length = self.sequence_length - len(trajectory)
                        trajectory = np.pad(trajectory, ((0, pad_length), (0, 0)), mode='constant')
                    
                    all_data.append(trajectory)
        
        return torch.tensor(np.array(all_data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# MLP Model
class MLPDiffusion(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 256, 128], T=1000):
        super().__init__()
        self.T = T
        
        # Time embedding
        self.time_embed = nn.Embedding(T, 32)
        
        layers = []
        prev_dim = input_dim + 32  # Concatenate time embedding
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        # x shape: (batch_size, seq_len, 2)
        # t shape: (batch_size,)
        
        batch_size, seq_len, input_dim = x.shape
        
        # Embed time
        t_embed = self.time_embed(t)  # (batch_size, 32)
        t_embed = t_embed.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, 32)
        
        # Concatenate time embedding
        x = torch.cat([x, t_embed], dim=-1)  # (batch_size, seq_len, input_dim + 32)
        
        # Flatten sequence dimension
        x = x.view(batch_size * seq_len, -1)
        
        # Pass through network
        out = self.net(x)
        
        # Reshape back to sequence
        out = out.view(batch_size, seq_len, input_dim)
        
        return out

# Diffusion process
class Diffusion:
    def __init__(self, T, beta_start, beta_end, device):
        self.T = T
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def forward_diffusion(self, x_0, t):
        """Add noise to input at timestep t"""
        batch_size = x_0.shape[0]
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Get alpha_bar for current timestep
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        
        # Noisy sample
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t, noise
    
    def reverse_diffusion(self, model, x_t, t):
        """Reverse diffusion step using model prediction"""
        with torch.no_grad():
            # Predict noise
            pred_noise = model(x_t, t)
            
            # Get parameters for current timestep
            alpha_t = self.alphas[t].view(-1, 1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
            beta_t = self.betas[t].view(-1, 1, 1)
            
            # Reverse diffusion step
            if t[0] > 0:
                z = torch.randn_like(x_t)
            else:
                z = 0
                
            x_t_minus_1 = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * z
            
            return x_t_minus_1

# Training function
def train_diffusion():
    # Initialize components
    dataset = TrajectoryDataset(Config.data_dirs, Config.sequence_length)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    input_dim = 2  # q1 and q2
    model = MLPDiffusion(input_dim=input_dim, T=Config.T).to(Config.device)
    diffusion = Diffusion(Config.T, Config.beta_start, Config.beta_end, Config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(Config.num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")
        
        for batch in progress_bar:
            batch = batch.to(Config.device)
            
            # Sample random timesteps
            t = torch.randint(0, Config.T, (batch.size(0),), device=Config.device)
            
            # Forward diffusion
            x_t, noise = diffusion.forward_diffusion(batch, t)
            
            # Predict noise
            pred_noise = model(x_t, t)
            
            # Calculate loss
            loss = criterion(pred_noise, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Save model periodically
        if (epoch + 1) % 100 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(Config.save_dir, f"diffusion_mlp_{timestamp}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataloader),
            }, model_path)
            
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(Config.save_dir, f"diffusion_mlp_{timestamp}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    return model

# Generate samples function
def generate_samples(model, diffusion, num_samples, sequence_length):
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(num_samples, sequence_length, 2).to(Config.device)
        
        # Reverse diffusion process
        for t in tqdm(reversed(range(diffusion.T)), desc="Generating samples"):
            t_batch = torch.full((num_samples,), t, device=Config.device, dtype=torch.long)
            x = diffusion.reverse_diffusion(model, x, t_batch)
        
        return x.cpu().numpy()

if __name__ == "__main__":
    # Train the model
    trained_model = train_diffusion()
    
    # Example of generating new trajectories
    diffusion = Diffusion(Config.T, Config.beta_start, Config.beta_end, Config.device)
    generated_trajectories = generate_samples(trained_model, diffusion, num_samples=10, sequence_length=Config.sequence_length)
    print(f"Generated trajectories shape: {generated_trajectories.shape}")