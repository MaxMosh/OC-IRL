import torch
import torch.nn as nn
import math

class TrajectoryEncoder(nn.Module):
    """
    Encodes the trajectory (Batch, Channels, 50) into a latent condition vector.
    
    input_channels=4 corresponds to: [q1, q2, dq1, dq2]
    """
    def __init__(self, input_channels=4, sequence_length=50, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Capture local features
            nn.Conv1d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.SiLU(), # Swish activation
            nn.MaxPool1d(2), # Reduces 50 -> 25

            # Layer 2: Deeper features
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.MaxPool1d(2), # Reduces 25 -> 12

            # Flatten and project to embedding dimension
            nn.Flatten(),
            nn.Linear(32 * 12, embedding_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # Expects input shape: (Batch, input_channels, 50)
        return self.net(x)

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
    def __init__(self, w_dim=5, cond_dim=64, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # We concatenate: w (w_dim) + condition (64) + time_embedding (32)
        input_dim = w_dim + cond_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 256),
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
    Allows dynamic configuration of w_dim and input_channels.
    """
    def __init__(self, w_dim=5, input_channels=4):
        super().__init__()
        # Pass input_channels to the encoder (e.g., 4 for q1, q2, dq1, dq2)
        self.encoder = TrajectoryEncoder(input_channels=input_channels)
        self.denoiser = DenoisingNetwork(w_dim=w_dim)

    def forward(self, w_noisy, t, trajectory):
        cond = self.encoder(trajectory)
        return self.denoiser(w_noisy, t, cond)