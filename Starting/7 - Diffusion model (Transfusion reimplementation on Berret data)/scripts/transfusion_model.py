import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        # Global average pooling over sequence dimension
        squeeze = x.mean(dim=1)  # (batch, channels)
        
        # Excitation
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = torch.sigmoid(excitation)
        
        # Apply attention
        excitation = excitation.unsqueeze(1)  # (batch, 1, channels)
        return x * excitation


class SETransformerBlock(nn.Module):
    """
    Transformer block with Squeeze-and-Excitation attention.
    """
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Squeeze-and-Excitation block
        self.se_block = SqueezeExcitation(d_model)
        
        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # SE block
        x = self.se_block(x) + x
        
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        
        # Feed-forward with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class DCTTransform:
    """
    Discrete Cosine Transform utilities for motion sequences.
    """
    @staticmethod
    def get_dct_matrix(N: int, L: int = None):
        """
        Generate DCT matrix.
        
        Args:
            N: Total sequence length
            L: Number of DCT coefficients to keep (if None, keep all)
        
        Returns:
            DCT matrix of shape (L, N) or (N, N)
        """
        if L is None:
            L = N
        
        dct_m = np.zeros((L, N))
        for k in range(L):
            for i in range(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / N)
        
        return torch.from_numpy(dct_m).float()
    
    @staticmethod
    def dct(x, dct_matrix):
        """
        Apply DCT transform.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            dct_matrix: DCT matrix of shape (L, seq_len)
        
        Returns:
            Transformed tensor of shape (batch, L, features)
        """
        # Ensure dct_matrix has the same dtype as x
        dct_matrix = dct_matrix.to(x.device, dtype=x.dtype)
        return torch.matmul(dct_matrix, x)
    
    @staticmethod
    def idct(y, dct_matrix):
        """
        Apply inverse DCT transform.
        
        Args:
            y: Input tensor of shape (batch, L, features)
            dct_matrix: DCT matrix of shape (L, seq_len)
        
        Returns:
            Transformed tensor of shape (batch, seq_len, features)
        """
        # Ensure dct_matrix has the same dtype as y
        dct_matrix = dct_matrix.to(y.device, dtype=y.dtype)
        return torch.matmul(dct_matrix.T, y)


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TransFusion(nn.Module):
    """
    TransFusion: Transformer-based Diffusion Model for Joint Angle Prediction.
    """
    def __init__(
        self,
        input_dim: int = 2,  # q1, q2
        obs_frames: int = 25,
        pred_frames: int = 100,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 9,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_dct_coeffs: int = 20,
        T: int = 1000  # Number of diffusion steps
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.obs_frames = obs_frames
        self.pred_frames = pred_frames
        self.total_frames = obs_frames + pred_frames
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_dct_coeffs = num_dct_coeffs
        self.T = T
        
        # DCT matrices
        self.register_buffer(
            'dct_matrix',
            DCTTransform.get_dct_matrix(self.total_frames, num_dct_coeffs)
        )
        
        # Input projection for noisy DCT coefficients
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Condition projection (for observed motion)
        self.cond_proj = nn.Linear(input_dim, d_model)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Positional encoding for sequence positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_dct_coeffs + 1, d_model) * 0.02
        )
        
        # Transformer blocks with skip connections
        self.transformer_blocks = nn.ModuleList([
            SETransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Skip connection projections (for long skip connections)
        self.skip_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) if i > 0 else None
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, input_dim)
        
        # Initialize noise schedule
        self.register_noise_schedule()
    
    def register_noise_schedule(self):
        """
        Register the noise schedule for diffusion process using cosine schedule.
        """
        # Cosine schedule
        steps = torch.arange(self.T + 1)
        alphas_cumprod = torch.cos(((steps / self.T) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
    
    def forward(self, y_t, condition, t):
        """
        Predict noise in the diffusion process.
        
        Args:
            y_t: Noisy DCT coefficients, shape (batch, L, input_dim)
            condition: Condition from observation, shape (batch, L, input_dim)
            t: Diffusion timestep, shape (batch,)
        
        Returns:
            Predicted noise, shape (batch, L, input_dim)
        """
        batch_size = y_t.shape[0]
        
        # Project inputs to d_model dimension
        y_embed = self.input_proj(y_t)  # (batch, L, d_model)
        cond_embed = self.cond_proj(condition)  # (batch, L, d_model)
        
        # Timestep embedding
        t_embed = self.time_embed(t)  # (batch, d_model)
        t_embed = t_embed.unsqueeze(1)  # (batch, 1, d_model)
        
        # Combine condition with timestep
        cond_token = (cond_embed.mean(dim=1, keepdim=True) + t_embed)  # (batch, 1, d_model)
        
        # Concatenate condition token with noisy coefficients
        x = torch.cat([cond_token, y_embed], dim=1)  # (batch, L+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Pass through transformer blocks with skip connections
        skip_connections = []
        for i, (block, skip_proj) in enumerate(zip(self.transformer_blocks, self.skip_projs)):
            x = block(x)
            
            # Store output for skip connection
            if i < len(self.transformer_blocks) // 2:
                skip_connections.append(x)
            
            # Apply skip connection from earlier layer
            elif len(skip_connections) > 0:
                skip_idx = len(self.transformer_blocks) - 1 - i
                if skip_idx >= 0 and skip_idx < len(skip_connections):
                    skip = skip_connections[skip_idx]
                    x = torch.cat([x, skip], dim=-1)
                    x = skip_proj(x)
        
        # Remove condition token and project to output
        x = x[:, 1:, :]  # (batch, L, d_model)
        noise_pred = self.output_proj(x)  # (batch, L, input_dim)
        
        return noise_pred
    
    def compute_loss(self, x_0, obs):
        """
        Compute training loss for the diffusion model.
        
        Args:
            x_0: Ground truth motion sequence, shape (batch, total_frames, input_dim)
            obs: Observed motion sequence, shape (batch, obs_frames, input_dim)
        
        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        
        # Transform to frequency domain
        y_0 = DCTTransform.dct(x_0, self.dct_matrix)  # (batch, L, input_dim)
        
        # Prepare condition (pad observation and transform)
        obs_padded = F.pad(
            obs,
            (0, 0, 0, self.pred_frames),
            mode='replicate'
        )  # (batch, total_frames, input_dim)
        condition = DCTTransform.dct(obs_padded, self.dct_matrix)  # (batch, L, input_dim)
        
        # Sample random timestep
        t = torch.randint(0, self.T, (batch_size,), device=x_0.device).long()
        
        # Sample noise
        noise = torch.randn_like(y_0)
        
        # Add noise to data
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        y_t = sqrt_alpha_cumprod_t * y_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        # Predict noise
        noise_pred = self.forward(y_t, condition, t)
        
        # Compute MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
