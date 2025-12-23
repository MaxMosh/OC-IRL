import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 1. Création du Dataset (La "Vérité Terrain")
# -----------------------------------------------------------------------------
# Au lieu d'avoir la fonction mathématique, en Deep Learning, on a des données.
# Nous créons des points groupés autour des 3 "minimums" de l'exemple précédent.
def generate_training_data(n_samples=5000):
    # Centres des puits (correspondant à l'exemple précédent)
    centers = [
        [2.0, 2.0],   # Puits 1
        [-2.0, -2.0], # Puits 2
        [2.0, -2.0]   # Puits 3
    ]
    # On choisit un centre au hasard pour chaque point
    indices = np.random.choice(len(centers), n_samples)
    data = []
    for idx in indices:
        center = centers[idx]
        # On ajoute un peu de bruit pour faire des "clusters"
        point = center + np.random.randn(2) * 0.5 
        data.append(point)
    
    return torch.tensor(np.array(data), dtype=torch.float32).to(DEVICE)

# -----------------------------------------------------------------------------
# 2. Le Modèle de Diffusion (Réseau de Neurones)
# -----------------------------------------------------------------------------
class ScoreNet(nn.Module):
    """
    Un réseau de neurones simple qui apprend à prédire le 'Score' (le gradient).
    Input: Une position (x, y) et un temps (t)
    Output: Le vecteur de direction (dx, dy) pour aller vers les données.
    """
    def __init__(self):
        super().__init__()
        # On encode le temps pour aider le réseau à savoir à quel niveau de bruit il est
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16)
        )
        
        # Le réseau principal qui traite la position (x,y)
        self.main = nn.Sequential(
            nn.Linear(2 + 16, 64), # 2 pour (x,y) + 16 pour le temps
            nn.Softplus(),         # Activation douce souvent utilisée en physique/score
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 2)       # Sortie : vecteur 2D (le score estimé)
        )

    def forward(self, x, t):
        # x: [batch, 2], t: [batch, 1]
        t_emb = self.time_embedding(t)
        # On concatène la position et l'info de temps
        input_data = torch.cat([x, t_emb], dim=1)
        return self.main(input_data)

# -----------------------------------------------------------------------------
# 3. Entraînement (Denoising Score Matching)
# -----------------------------------------------------------------------------
def train_model(model, dataset, n_epochs=100000, batch_size=1024, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    print(f"Entraînement sur {DEVICE}...")
    
    for epoch in range(n_epochs):
        # 1. Préparer un batch de données réelles x0
        indices = torch.randint(0, len(dataset), (batch_size,))
        x0 = dataset[indices]
        
        # 2. Echantillonner un temps t aléatoire entre 0 (peu de bruit) et 1 (beaucoup de bruit)
        # Pour simplifier, sigma(t) = t. Donc t est directement l'écart-type du bruit.
        t = torch.rand(batch_size, 1, device=DEVICE) * 4.0 + 0.1 # t entre 0.1 et 4.1
        
        # 3. Ajouter du bruit (Forward Process)
        noise = torch.randn_like(x0)
        xt = x0 + t * noise # x bruité
        
        # 4. Prédire le bruit (ou le score)
        # Le score théorique est -noise / t
        # On entraîne le réseau à prédire : Score * t = -noise
        # C'est plus stable numériquement de prédire le bruit directement ou -score*t
        
        output_score = model(xt, t)
        
        # Objectif : Le réseau doit retrouver la direction vers x0.
        # La cible pour le score est -(xt - x0) / (t^2) = -noise / t
        target_score = -noise / t
        
        # Loss : MSE pondérée par t^2 (standard dans les papiers de diffusion)
        # Loss = || output_score - target_score ||^2 * t^2
        # Ce qui simplifie en || output_score * t + noise ||^2
        
        loss = torch.mean(torch.sum((output_score * t + noise)**2, dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            losses.append(loss.item())
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    return losses

# -----------------------------------------------------------------------------
# 4. Sampling avec le Modèle Appris
# -----------------------------------------------------------------------------
def generate_samples(model, n_samples=1000, n_steps=200):
    """
    Génère des nouveaux points en utilisant le réseau entraîné.
    """
    model.eval()
    # On commence avec du bruit aléatoire (comme avant)
    z = torch.randn(n_samples, 2, device=DEVICE) * 4.0
    
    trajectory = [z.detach().cpu().numpy()]
    
    # Programme de bruit décroissant (de 4.0 à 0.1)
    time_steps = np.linspace(4.0, 0.1, n_steps)
    step_size_base = 0.05
    
    with torch.no_grad():
        for i, t_val in enumerate(time_steps):
            t_batch = torch.ones(n_samples, 1, device=DEVICE) * t_val
            
            # --- 1. Prédiction du Score par le Réseau ---
            score = model(z, t_batch)
            
            # --- 2. Mise à jour (Langevin) ---
            # Le pas dépend du temps (plus petit à la fin)
            dt = step_size_base * (t_val / time_steps[0])
            
            noise = torch.randn_like(z)
            
            # Formule standard Langevin : x_new = x + score * dt + bruit
            z = z + score * dt + np.sqrt(2 * dt) * noise
            
            if i % (n_steps // 10) == 0:
                trajectory.append(z.cpu().numpy())
                
    return z.cpu().numpy(), trajectory

# -----------------------------------------------------------------------------
# 5. Visualisation
# -----------------------------------------------------------------------------
def plot_results_dl(dataset, final_samples, trajectory, losses):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    dataset = dataset.cpu().numpy()
    
    # 1. Données d'entraînement
    axes[0].scatter(dataset[:, 0], dataset[:, 1], s=5, alpha=0.5, label='Training Data')
    axes[0].set_title("1. Données d'entraînement (Objectif)")
    axes[0].set_xlim(-6, 6)
    axes[0].set_ylim(-6, 6)
    axes[0].legend()
    
    # 2. Courbe d'apprentissage
    axes[1].plot(losses)
    axes[1].set_title("2. Perte (Loss) pendant l'entraînement")
    axes[1].set_xlabel("Epochs (x200)")
    
    # 3. Résultat généré par le réseau
    start_pts = trajectory[0]
    axes[2].scatter(start_pts[:, 0], start_pts[:, 1], c='gray', alpha=0.3, s=5, label='Bruit Initial')
    axes[2].scatter(final_samples[:, 0], final_samples[:, 1], c='red', alpha=0.6, s=10, label='Généré par IA')
    axes[2].set_title("3. Génération via Réseau de Neurones")
    axes[2].set_xlim(-6, 6)
    axes[2].set_ylim(-6, 6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

# A. Préparer les données
data = generate_training_data()

# B. Créer et entraîner le modèle
score_model = ScoreNet().to(DEVICE)
loss_history = train_model(score_model, data)

# C. Générer de nouveaux échantillons
final_samples, trajectory = generate_samples(score_model)

# D. Afficher
plot_results_dl(data, final_samples, trajectory, loss_history)