import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------
# Même MLP que l'entraînement
# -----------------------
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        batch, seq_len, dim = x.shape
        x_flat = x.reshape(batch*seq_len, dim)
        out = self.model(x_flat)
        return out.reshape(batch, seq_len, dim)

# -----------------------
# DDPM sampler
# -----------------------
class DDPM:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
    
    @torch.no_grad()
    def sample(self, seq_len=50, device='cpu'):
        x = torch.randn(1, seq_len, 2).to(device)  # start from noise
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((1,), t, dtype=torch.long, device=device).view(-1,1,1)
            beta_t = self.betas[t].to(device)
            alpha_t = self.alphas[t].to(device)
            alpha_cumprod_t = self.alpha_cumprod[t].to(device)
            alpha_cumprod_prev_t = self.alpha_cumprod_prev[t].to(device)
            
            pred_noise = self.model(x)
            coef1 = 1 / alpha_t.sqrt()
            coef2 = (1 - alpha_t) / (1 - alpha_cumprod_t).sqrt()
            x = coef1 * (x - coef2 * pred_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = (beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)).sqrt()
                x += sigma_t * noise
        return x.squeeze(0).cpu().numpy()  # shape (seq_len, 2)

# -----------------------
# Charger modèle et générer
# -----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MLP().to(device)

# Charger les poids du modèle entraîné
model.load_state_dict(torch.load('trained_models/ddpm_mlp_20251107_150549.pt', map_location=device))
model.eval()

ddpm = DDPM(model)

# Générer une trajectoire de longueur 50
generated_traj = ddpm.sample(seq_len=50, device=device)
print("Trajectoire générée :", generated_traj)

print(f"Size of generated trajectory: {generated_traj.shape}")
plt.plot(generated_traj[:,0])
plt.show()

plt.plot(generated_traj[:,1])
plt.show()