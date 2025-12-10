import torch
import torch.nn as nn

class DAN_WeightEstimator(nn.Module):
    """
    Data Assimilation Network for Cost Weight Estimation.
    Based on the framework by Boudier et al. (2020).
    
    Structure:
    - Propagator (b): S -> S (Dynamics of the internal belief/memory)
    - Analyzer (a): S x Y -> S (Update belief based on observation)
    - Procoder (c): S -> W (Decode belief into physical weights)
    """
    def __init__(self, input_dim=2, hidden_dim=64, w_dim=3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # --- 1. Propagator (b) ---
        # Represents the natural evolution of the weights (prior prediction).
        # In the paper, this is b: S -> S
        self.propagator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim) # Residual connection handled in forward
        )
        
        # --- 2. Analyzer (a) ---
        # Integrates the new observation Y (trajectory) into the belief S.
        # In the paper, this is a: S x Y -> S
        # We use a GRUCell to act as the recurrence mechanism for the Analyzer.
        self.analyzer_fusion = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        self.analyzer_gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # --- 3. Procoder (c) ---
        # Decodes the latent state S into the estimated weights W.
        # In the paper, this is c: S -> P_x (here, point estimate of weights)
        self.procoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, w_dim),
            nn.Softplus() # Enforce positive weights (physical constraint)
        )

    def forward(self, trajectory_batch):
        """
        Args:
            trajectory_batch: Tensor of shape (Batch, Seq_Len, Input_Dim) -> (B, 50, 2)
        Returns:
            w_sequence: Tensor of shape (Batch, Seq_Len, W_Dim) -> (B, 50, 3)
        """
        batch_size, seq_len, _ = trajectory_batch.size()
        device = trajectory_batch.device
        
        # Initialize memory (s_0)
        s_curr = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        w_estimates = []
        
        for t in range(seq_len):
            obs_t = trajectory_batch[:, t, :] # (Batch, 2)
            
            # --- A. Propagation Step (Forecast) ---
            # s_prior = b(s_prev)
            # We add a residual connection to stabilize training
            s_prop = self.propagator(s_curr)
            s_prior = s_curr + s_prop 
            
            # --- B. Analysis Step (Update) ---
            # s_posterior = a(s_prior, observation)
            
            # Pre-fusion of prior belief and current observation
            fusion_input = torch.cat([s_prior, obs_t], dim=1)
            gru_input = self.analyzer_fusion(fusion_input)
            
            # Recurrent update
            s_posterior = self.analyzer_gru(gru_input, s_prior)
            
            # --- C. Procoding Step (Decode) ---
            # w_t = c(s_posterior)
            w_t = self.procoder(s_posterior)
            w_estimates.append(w_t.unsqueeze(1))
            
            # Update state for next step
            s_curr = s_posterior
            
        # Concatenate all time steps
        return torch.cat(w_estimates, dim=1)