import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
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

class TransformerDiffusionModel(nn.Module):
    def __init__(self, 
                 seq_len=50, 
                 w_dim=3, 
                 cond_dim=3, # (q1, q2, mask)
                 d_model=128, 
                 nhead=4, 
                 num_layers=4, 
                 dim_feedforward=512):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Projection d'entrée
        # On concatène w_noisy (3) + condition (3) = 6 features par pas de temps
        input_dim = w_dim + cond_dim
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Encodage du Temps (Diffusion Step)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 3. Encodage de Position (Séquence 0..50)
        # Paramètre apprenable pour savoir où on est dans la séquence
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        # 4. Le Transformer (Backbone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True,
            activation="gelu",
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Tête de sortie
        self.output_head = nn.Linear(d_model, w_dim)

    def forward(self, w_noisy, t, trajectory_cond):
        """
        w_noisy: (Batch, 50, 3)
        t: (Batch,)
        trajectory_cond: (Batch, 50, 3) -> Contient [q1, q2, mask]
        """
        batch_size, seq_len, _ = w_noisy.shape

        # --- A. Fusion des entrées ---
        # (Batch, 50, 6)
        x = torch.cat([w_noisy, trajectory_cond], dim=-1)
        
        # Projection vers d_model
        x = self.input_proj(x) # (Batch, 50, d_model)

        # --- B. Ajout des Embeddings ---
        # 1. Time Embedding (Ajouté globalement à tous les tokens de la séquence)
        t_emb = self.time_mlp(t) # (Batch, d_model)
        t_emb = t_emb.unsqueeze(1) # (Batch, 1, d_model)
        
        # 2. Positional Embedding (Ajouté pour distinguer t=0 de t=50)
        x = x + self.pos_embedding + t_emb

        # --- C. Transformer ---
        # Le masque d'attention n'est pas nécessaire ici car on veut que tous les points
        # temporels se voient (Self-Attention bidirectionnelle)
        x = self.transformer(x)

        # --- D. Prédiction du Bruit ---
        output = self.output_head(x) # (Batch, 50, 3)
        
        return output