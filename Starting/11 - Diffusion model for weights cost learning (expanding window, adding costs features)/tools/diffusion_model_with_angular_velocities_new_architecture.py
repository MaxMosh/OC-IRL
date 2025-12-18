import torch
import torch.nn as nn
import math

class TransformerTrajectoryEncoder(nn.Module):
    """
    Encodeur basé sur un Transformer (Encoder-only).
    Idéal pour capturer les dépendances globales et les singularités (coudes)
    dans la trajectoire.
    """
    def __init__(self, input_channels=4, sequence_length=50, embedding_dim=64, nhead=4, num_layers=3):
        super().__init__()
        
        self.seq_len = sequence_length
        self.emb_dim = embedding_dim
        
        # 1. Projection d'entrée (Feature mixing) : 4 -> 64
        self.input_proj = nn.Linear(input_channels, embedding_dim)
        
        # 2. Positional Embedding (Apprenable)
        # Indispensable pour que le modèle sache que t=0 est différent de t=50
        self.pos_embedding = nn.Parameter(torch.randn(1, sequence_length + 1, embedding_dim))
        
        # 3. Token [CLS] (comme dans BERT)
        # Ce token va "résumer" toute la trajectoire
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # 4. Le Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True # Important: (Batch, Seq, Feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Fusion finale avec les conditions aux limites (Robustesse Solver)
        # On concatène : [CLS_Embedding] + [Start_State] + [End_State]
        # Dimensions : 64 + 4 + 4 = 72
        combined_dim = embedding_dim + input_channels * 2
        
        self.fc_out = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        # x shape initial: (Batch, input_channels, seq_len) -> ex: (32, 4, 50)
        B = x.shape[0]
        
        # Extraction des conditions aux limites AVANT transformation
        start_state = x[:, :, 0]  # (Batch, 4)
        end_state = x[:, :, -1]   # (Batch, 4)

        # Permutation pour le Transformer : (Batch, Seq, Channels)
        x = x.permute(0, 2, 1) # -> (32, 50, 4)
        
        # Projection linéaire
        x = self.input_proj(x) # -> (32, 50, 64)
        
        # Ajout du CLS token au début de la séquence
        # On passe de longueur 50 à 51
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Ajout des embeddings de position
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Passage dans le Transformer
        x = self.transformer(x)
        
        # On récupère uniquement le token [CLS] (le premier de la séquence)
        # C'est lui qui contient le résumé contextuel de toute la courbe
        cls_out = x[:, 0, :] # (Batch, 64)
        
        # Concaténation explicite pour aider le modèle
        final_vec = torch.cat([cls_out, start_state, end_state], dim=1)
        
        return self.fc_out(final_vec)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Embeddings positionnels pour le temps 't' dans le processus de diffusion.
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
    Le réseau de débruitage (MLP ResNet-ish).
    """
    def __init__(self, w_dim=5, cond_dim=64, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

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

            nn.Linear(256, w_dim)
        )

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x, cond, t_emb], dim=1)
        return self.net(x_input)

class ConditionalDiffusionModel(nn.Module):
    """
    Wrapper utilisant le TransformerEncoder.
    """
    def __init__(self, w_dim=5, input_channels=4):
        super().__init__()
        # On utilise le Transformer ici
        self.encoder = TransformerTrajectoryEncoder(
            input_channels=input_channels,
            sequence_length=50,
            embedding_dim=64
        )
        self.denoiser = DenoisingNetwork(w_dim=w_dim)

    def forward(self, w_noisy, t, trajectory):
        cond = self.encoder(trajectory)
        return self.denoiser(w_noisy, t, cond)