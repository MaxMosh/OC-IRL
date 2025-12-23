import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SIMULATEUR PHYSIQUE (Génération de données) ---
def simulate_spring_damper(k, c, x0=1.0, v0=0.0, length=100, dt=0.1):
    """
    Simule un système masse-ressort-amortisseur: m*x'' + c*x' + k*x = 0 (m=1)
    """
    t = np.linspace(0, length*dt, length)
    # Solution analytique ou Euler simple pour la démo
    trajectory = []
    x, v = x0, v0
    for _ in range(length):
        trajectory.append(x)
        a = -c * v - k * x # F = ma
        v += a * dt
        x += v * dt
    return np.array(trajectory, dtype=np.float32)

def create_dataset(n_samples=1000, length=64):
    data = []
    # On génère des k et c aléatoires
    ks = np.random.uniform(0.5, 2.0, n_samples)
    cs = np.random.uniform(0.0, 0.5, n_samples)
    
    for i in range(n_samples):
        traj = simulate_spring_damper(ks[i], cs[i], length=length)
        
        # Normalisation simple pour aider le réseau (important !)
        traj = traj / 2.0  # On sait que x0=1, donc ça reste borné
        
        # Création du tenseur joint [3, Length]
        # Canal 0: Trajectoire
        # Canal 1: k (répété)
        # Canal 2: c (répété)
        sample = np.zeros((3, length), dtype=np.float32)
        sample[0, :] = traj
        sample[1, :] = (ks[i] - 1.25) / 0.75 # Normalisation centrée approx
        sample[2, :] = (cs[i] - 0.25) / 0.25 
        data.append(sample)
        
    return torch.tensor(data)

# --- 2. MODÈLE DE DIFFUSION (Architecture simple) ---

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # Sinusoidal embedding classique
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

class Simple1DDiffusionNet(nn.Module):
    def __init__(self, n_channels=3, length=64):
        super().__init__()
        self.time_embed = TimeEmbedding(32)
        
        # Un U-Net 1D très simplifié
        self.inc = nn.Conv1d(n_channels, 64, 3, padding=1)
        self.mid = nn.Conv1d(64, 64, 3, padding=1)
        self.out = nn.Conv1d(64, n_channels, 3, padding=1)
        
        self.t_proj = nn.Linear(32, 64)
        self.act = nn.SiLU()

    def forward(self, x, t):
        # x: [Batch, 3, Length]
        t_emb = self.act(self.t_proj(self.time_embed(t))) # [Batch, 64]
        t_emb = t_emb.unsqueeze(-1) # Broadcast pour additionner aux convs
        
        h = self.act(self.inc(x))
        h = h + t_emb # Injection du temps
        h = self.act(self.mid(h))
        return self.out(h)

# --- 3. PARAMÈTRES DDPM (Noise Schedule) ---
n_steps = 100
betas = torch.linspace(1e-4, 0.02, n_steps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def q_sample(x_0, t, noise=None):
    """Bruite les données à l'instant t"""
    if noise is None: noise = torch.randn_like(x_0)
    sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

# --- 4. ENTRAÎNEMENT ---
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = create_dataset(n_samples=2000).to(device)
model = Simple1DDiffusionNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Début de l'entraînement...")
for epoch in range(501):
    # Batch simple (tout le dataset pour la démo)
    x_0 = dataset
    t = torch.randint(0, n_steps, (x_0.shape[0],)).to(device)
    
    # On bruite TOUT : trajectoire ET paramètres
    x_t, noise = q_sample(x_0, t)
    
    # Le modèle essaie de prédire le bruit
    noise_pred = model(x_t, t.view(-1, 1))
    
    loss = nn.MSELoss()(noise_pred, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")

# --- 5. INFERENCE : PRÉDICTION DE PARAMÈTRES (INPAINTING) ---
# Scénario : On observe une trajectoire réelle, on veut retrouver k et c.

# Prenons une "Vérité Terrain" inconnue du modèle
true_k, true_c = 1.8, 0.1
obs_traj_np = simulate_spring_damper(true_k, true_c, length=64) / 2.0
obs_traj = torch.tensor(obs_traj_np, dtype=torch.float32).to(device)

# Initialisation : Bruit pur aléatoire pour les 3 canaux
x = torch.randn(1, 3, 64).to(device) 

print("\nInférence (Recherche des paramètres)...")
model.eval()
with torch.no_grad():
    for i in reversed(range(n_steps)):
        t = torch.tensor([i]).to(device)
        
        # 1. Prédire le bruit
        noise_pred = model(x, t.view(-1, 1))
        
        # 2. Étape de débruitage (Maths standard DDPM)
        beta = betas[i]
        sqrt_recip_alpha = 1.0 / torch.sqrt(alphas[i])
        sigma = torch.sqrt(beta) if i > 0 else 0
        
        # Formule x_{t-1}
        x = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alphas_cumprod[i] * noise_pred)
        if i > 0:
            x += sigma * torch.randn_like(x)
            
        # 3. L'ASTUCE (Inpainting) :
        # On connait la trajectoire (Canal 0), donc on la remet en place !
        # Mais attention : on doit remettre la version BRUITÉE de la trajectoire observée à l'étape i-1
        if i > 0:
            noise_for_traj = torch.randn_like(obs_traj)
            obs_traj_noisy = sqrt_alphas_cumprod[i-1] * obs_traj + sqrt_one_minus_alphas_cumprod[i-1] * noise_for_traj
            x[:, 0, :] = obs_traj_noisy # On force la trajectoire connue
        else:
            x[:, 0, :] = obs_traj # À la fin, on met la vraie sans bruit

# --- RÉSULTATS ---
# On récupère les paramètres prédits (moyenne sur la dimension temporelle)
pred_k_norm = x[0, 1, :].mean().item()
pred_c_norm = x[0, 2, :].mean().item()

# Dénormalisation
pred_k = pred_k_norm * 0.75 + 1.25
pred_c = pred_c_norm * 0.25 + 0.25

print(f"--- Résultats ---")
print(f"Vrai k: {true_k:.3f} | Prédit k: {pred_k:.3f}")
print(f"Vrai c: {true_c:.3f} | Prédit c: {pred_c:.3f}")

# Visualisation
plt.figure(figsize=(10, 4))
plt.plot(obs_traj_np, label="Trajectoire Observée (Input)", color='black', linewidth=2)
# On simule avec les paramètres prédits pour vérifier
recons_traj = simulate_spring_damper(pred_k, pred_c, length=64) / 2.0
plt.plot(recons_traj, label="Trajectoire avec Params Prédits", linestyle='--')
plt.legend()
plt.title(f"Identification Système via Diffusion : k={pred_k:.2f}, c={pred_c:.2f}")
plt.show()