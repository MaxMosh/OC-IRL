import torch
import torch.nn as nn
import math

class TrajectoryEncoder(nn.Module):
    """
    Encodes the trajectory (Batch, 2, 50) into a latent condition vector.
    Uses 1D Convolutions since data is sequential/temporal.
    """
    def __init__(self, input_channels=2, sequence_length=50, embedding_dim=64):
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
        # Expects input shape: (Batch, 2, 50)
        return self.net(x)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Classic Transformers positional embeddings to encode time steps 't'.
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
    The main diffusion model.
    Input: Noisy w (Batch, 3) + Time t + Condition (Trajectory Embedding)
    Output: Predicted Noise (Batch, 3)
    """
    def __init__(self, w_dim=3, cond_dim=64, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # We concatenate: w (3) + condition (64) + time_embedding (32)
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
    """
    def __init__(self):
        super().__init__()
        self.encoder = TrajectoryEncoder()
        self.denoiser = DenoisingNetwork()

    def forward(self, w_noisy, t, trajectory):
        cond = self.encoder(trajectory)
        return self.denoiser(w_noisy, t, cond)