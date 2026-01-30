import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding for Transformers.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a long enough P.E. matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return x

class TrajectoryTransformerEncoder(nn.Module):
    """
    Encodes the trajectory (Batch, Length, Channels) into a sequence of latent features.
    Unlike the previous CNN which pooled everything, this preserves the sequence
    to allow Cross-Attention.
    """
    def __init__(self, input_channels=4, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 1. Project input features (q, dq) to d_model dimension
        self.input_proj = nn.Linear(input_channels, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm usually stabilizes training
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        x: (Batch, Input_Channels, Length) -> needs permutation
        src_key_padding_mask: (Batch, Length) - True where value is padding
        """
        # Permute to (Batch, Length, Input_Channels)
        x = x.transpose(1, 2)
        
        # Project and add position info
        x = self.input_proj(x) # (B, L, d_model)
        x = self.pos_encoder(x)
        
        # Apply Transformer
        # Output: (B, L, d_model)
        memory = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

class TransformerDenoiser(nn.Module):
    """
    Denoises the weights using a Transformer Decoder architecture.
    Treats the weights matrix (5, 3) as a sequence of length 3.
    """
    # def __init__(self, w_features=5, w_phases=3, d_model=128, nhead=4, num_layers=4, dropout=0.1):
    # def __init__(self, w_features=4, w_phases=3, d_model=128, nhead=4, num_layers=4, dropout=0.1):
    def __init__(self, w_features=4, w_phases=1, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.w_features = w_features
        self.w_phases = w_phases
        self.d_model = d_model

        # 1. Embeddings for Time (Diffusion Step)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 2. Embedding for Weights
        # We project the 5 cost components to d_model
        self.weight_proj = nn.Linear(w_features, d_model)
        
        # 3. Positional Embedding for Phases (Start, Mid, End)
        self.phase_embedding = nn.Embedding(w_phases, d_model)

        # 4. Transformer Decoder
        # It will Self-Attend to weights (phases) and Cross-Attend to Trajectory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 5. Final Output Projection
        self.output_head = nn.Linear(d_model, w_features)

    def forward(self, w_noisy, t, trajectory_memory, trajectory_mask=None):
        """
        w_noisy: (Batch, 15) -> Flattened weights
        t: (Batch,) -> Timesteps
        trajectory_memory: (Batch, Seq_Len, d_model) -> From Encoder
        trajectory_mask: (Batch, Seq_Len) -> Mask for padding
        """
        batch_size = w_noisy.shape[0]

        # A. Reshape flattened weights to (Batch, 3 Phases, 5 Features)
        w_seq = w_noisy.view(batch_size, self.w_phases, self.w_features)
        
        # B. Embed Weights
        x = self.weight_proj(w_seq) # (B, 3, d_model)

        # C. Add Phase Embeddings (0, 1, 2)
        phase_ids = torch.arange(self.w_phases, device=w_noisy.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.phase_embedding(phase_ids)

        # D. Add Time Embeddings
        # Time is global for the sample, so we add it to all tokens
        t_emb = self.time_mlp(t.unsqueeze(-1).float()) # (B, d_model)
        x = x + t_emb.unsqueeze(1)

        # E. Transformer Processing
        # tgt = x (Weights), memory = trajectory_memory
        # memory_key_padding_mask handles the variable length of trajectories
        output = self.transformer_decoder(
            tgt=x, 
            memory=trajectory_memory,
            memory_key_padding_mask=trajectory_mask
        )

        # F. Project back to weight space
        pred = self.output_head(output) # (B, 3, 5)

        # Flatten back to (B, 15)
        return pred.view(batch_size, -1)

class ConditionalDiffusionModel(nn.Module):
    """
    Main Model Class.
    Combines TrajectoryTransformerEncoder and TransformerDenoiser.
    """
    # def __init__(self, w_dim=15, input_channels=4, d_model=256, nhead=8, num_layers=6):
    # def __init__(self, w_dim=12, input_channels=4, d_model=256, nhead=8, num_layers=6):
    def __init__(self, w_dim=4, input_channels=4, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        
        # Configuration
        # w_dim must be divisible by 3 (phases)
        # assert w_dim % 3 == 0, "w_dim must be divisible by 3 (3 phases)"
        # w_features = w_dim // 3
        n_phases = 1 
        assert w_dim % n_phases == 0, f"w_dim must be divisible by {n_phases}"
        w_features = w_dim // n_phases
        
        self.encoder = TrajectoryTransformerEncoder(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers // 2 # Encoder gets half layers
        )
        
        self.denoiser = TransformerDenoiser(
            w_features=w_features,
            # w_phases=3,
            w_phases=n_phases,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers # Decoder gets full layers
        )

    def forward(self, w_noisy, t, trajectory, trajectory_mask=None):
        """
        w_noisy: (Batch, 15)
        t: (Batch,)
        trajectory: (Batch, 4, Length)
        trajectory_mask: (Batch, Length) - Boolean mask (True = Padding)
        """
        # 1. Encode Trajectory (Cross-Attention Key/Values)
        traj_memory = self.encoder(trajectory, src_key_padding_mask=trajectory_mask)
        
        # 2. Denoise Weights (Query)
        noise_pred = self.denoiser(w_noisy, t, traj_memory, trajectory_mask=trajectory_mask)
        
        return noise_pred