import torch
import torch.nn as nn
import math

class TrajectoryEncoder(nn.Module):
    """
    Encodes a trajectory of variable length (Batch, Channels, Length) 
    into a fixed-size latent condition vector.
    
    input_channels=4 corresponds to: [q1, q2, dq1, dq2]
    """
    def __init__(self, input_channels=4, embedding_dim=128):
        super().__init__()
        
        # Convolutional Feature Extractor
        # We use a series of convolutions to extract temporal features.
        self.net = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(), 
            
            # Layer 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )

        # Adaptive Pooling allows handling variable sequence lengths.
        # It forces the time dimension to become 1, regardless of input length.
        # MaxPool is good for detecting "presence" of features (e.g. "did we reach this velocity?").
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection to embedding dimension
        self.project = nn.Linear(128, embedding_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        # Input shape: (Batch, input_channels, Length_Variable)
        
        features = self.net(x)          # (Batch, 128, Length_Variable)
        pooled = self.global_pool(features) # (Batch, 128, 1)
        pooled = pooled.flatten(1)      # (Batch, 128)
        
        return self.act(self.project(pooled))

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Classic Transformers positional embeddings to encode time steps t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DenoisingNetwork(nn.Module):
    """
    The main diffusion model (MLP).
    Input: Noisy w (Batch, w_dim) + Time t + Condition (Trajectory Embedding)
    Output: Predicted Noise (Batch, w_dim)
    """
    def __init__(self, w_dim=15, cond_dim=128, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Input to the MLP: w + condition + time_embedding
        input_dim = w_dim + cond_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.SiLU(),

            nn.Linear(256, w_dim) # Output matches w dimension
        )

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        # Concatenate inputs along feature dimension
        x_input = torch.cat([x, cond, t_emb], dim=1)
        return self.net(x_input)

class ConditionalDiffusionModel(nn.Module):
    """
    Wrapper class combining the Encoder and the Denoiser.
    """
    def __init__(self, w_dim=15, input_channels=4, embedding_dim=128):
        super().__init__()
        self.encoder = TrajectoryEncoder(input_channels=input_channels, embedding_dim=embedding_dim)
        self.denoiser = DenoisingNetwork(w_dim=w_dim, cond_dim=embedding_dim)

    def forward(self, w_noisy, t, trajectory):
        # 1. Encode the variable-length trajectory into a fixed vector
        cond = self.encoder(trajectory)
        
        # 2. Predict noise
        return self.denoiser(w_noisy, t, cond)