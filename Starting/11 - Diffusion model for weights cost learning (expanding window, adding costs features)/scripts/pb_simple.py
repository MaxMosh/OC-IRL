import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour la reproductibilité
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Définition de la fonction de coût (L'Énergie)
# -----------------------------------------------------------------------------
def energy_function(z):
    """
    Une fonction R^2 -> R avec plusieurs minimums locaux.
    C'est notre 'problème inverse' : trouver les x,y qui minimisent cette fonction.
    On utilise une somme de Gaussiennes inversées pour créer des 'puits'.
    """
    x = z[:, 0]
    y = z[:, 1]
    
    # Création de 3 puits (minimums) distincts
    # Puits 1
    u1 = -1.0 * torch.exp(-((x - 2)**2 + (y - 2)**2) / 1.0)
    # Puits 2
    u2 = -1.0 * torch.exp(-((x + 2)**2 + (y + 2)**2) / 1.0)
    # Puits 3
    u3 = -0.8 * torch.exp(-((x - 2)**2 + (y + 2)**2) / 1.0) # Un peu moins profond
    
    # On ajoute un terme quadratique léger pour confiner les particules au centre
    confining = 0.05 * (x**2 + y**2)
    
    return u1 + u2 + u3 + confining

def get_score(z):
    """
    Calcule la 'Score Function' : le gradient du log de la densité.
    Si p(x) = exp(-E(x)), alors log p(x) = -E(x).
    Donc le score = - grad(E(x)).
    C'est la direction qui pousse les particules vers les minimums de l'énergie.
    """
    z = z.detach().requires_grad_(True)
    e = energy_function(z)
    # On veut le gradient de la somme des énergies par rapport à z
    grad_e = torch.autograd.grad(e.sum(), z)[0]
    return -grad_e  # Le score est l'opposé du gradient de l'énergie

# -----------------------------------------------------------------------------
# 2. Le Processus de Diffusion (Langevin Dynamics)
# -----------------------------------------------------------------------------
def annealed_langevin_dynamics(n_samples, n_steps, step_size, noise_scale_start, noise_scale_end):
    """
    Simule le processus inverse : part du bruit blanc et 'refroidit' le système
    pour qu'il se concentre dans les minimums de la fonction.
    """
    # Initialisation : Bruit blanc (Distribution Normale Standard)
    # C'est l'équivalent du temps T dans un modèle de diffusion
    z = torch.randn(n_samples, 2) * 4 # *4 pour étaler le bruit initialement
    
    trajectory = [z.detach().cpu().numpy()]
    
    # On fait décroître le niveau de bruit (Annealing)
    # Cela permet d'explorer l'espace au début, puis de affiner à la fin.
    noise_scales = np.linspace(noise_scale_start, noise_scale_end, n_steps)
    
    print("Début du processus de diffusion...")
    for i, noise_scale in enumerate(noise_scales):
        # Calcul du score (la direction vers les minimums)
        score = get_score(z)
        
        # Ajout de bruit aléatoire (Mouvement Brownien)
        noise = torch.randn_like(z)
        
        # Mise à jour de Langevin :
        # z_new = z_old + (step_size * score) + (sqrt(step_size) * noise)
        # Note: Le coefficient exact dépend de la formulation, ici c'est une version simplifiée
        # efficace pour l'optimisation.
        
        # Le terme de score pousse vers le bas de la pente
        # Le terme de bruit permet de sauter hors des petits minimums locaux indésirables (si on voulait)
        
        # Ajustement du pas en fonction du niveau de bruit (optionnel mais aide la convergence)
        current_lr = step_size * (noise_scale**2 / noise_scales[-1]**2)
        
        z = z + (current_lr * score) + np.sqrt(2 * current_lr) * noise * noise_scale
        
        if i % (n_steps // 10) == 0:
            trajectory.append(z.detach().cpu().numpy())
            
    print("Diffusion terminée.")
    return z.detach().cpu().numpy(), trajectory

# -----------------------------------------------------------------------------
# 3. Visualisation
# -----------------------------------------------------------------------------
def plot_results(final_samples, trajectory):
    # Création de la grille pour afficher la fonction d'énergie réelle (Ground Truth)
    grid_size = 100
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Conversion en tenseur pour passer dans notre fonction energy_function
    grid_tensor = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    Z = energy_function(grid_tensor).detach().numpy().reshape(grid_size, grid_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1 : La trajectoire au début (Bruit) vs Fin
    axes[0].contourf(X, Y, Z, levels=20, cmap='viridis_r', alpha=0.6)
    axes[0].set_title("Distribution Initiale (Points gris) vs Finale (Points rouges)")
    
    # Points de départ (Bruit)
    start_pts = trajectory[0]
    axes[0].scatter(start_pts[:, 0], start_pts[:, 1], c='gray', alpha=0.5, s=10, label='Initial (Bruit)')
    
    # Points finaux (Minimums trouvés)
    axes[0].scatter(final_samples[:, 0], final_samples[:, 1], c='red', alpha=0.8, s=20, label='Final (Minimums)')
    axes[0].legend()
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)

    # Plot 2 : Densité estimée des points finaux
    axes[1].contourf(X, Y, Z, levels=20, cmap='gray', alpha=0.3) # Fond léger
    sns.kdeplot(x=final_samples[:, 0], y=final_samples[:, 1], fill=True, cmap="Reds", alpha=0.7, ax=axes[1])
    axes[1].set_title("Densité des solutions trouvées (KDE)")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

# Paramètres
N_SAMPLES = 1000       # Nombre de particules à simuler
N_STEPS = 500          # Nombre de pas de temps de diffusion
STEP_SIZE = 0.05       # Taille du pas de base
NOISE_START = 1.0      # Niveau de bruit au début (exploration)
NOISE_END = 0.01       # Niveau de bruit à la fin (précision)

# Lancer la diffusion
final_samples, trajectory = annealed_langevin_dynamics(
    N_SAMPLES, N_STEPS, STEP_SIZE, NOISE_START, NOISE_END
)

# Afficher
plot_results(final_samples, trajectory)