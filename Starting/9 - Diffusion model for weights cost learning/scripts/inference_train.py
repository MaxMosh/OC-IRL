import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tools.diffusion_model import ConditionalDiffusionModel

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTEPS = 1000
N_SAMPLES_TO_GENERATE = 50 # How many w guesses for one single trajectory?

# --- 1. Load Resources ---
# Load Data (Just to pick a test sample)
try:
    traj_data = np.load("data/array_results_angles_101.npy")
    w_true_data = np.load("data/array_w_101.npy")
except FileNotFoundError:
    print("Error: Data files not found.")
    exit()

# Load Scaler (Essential to denormalize output)
with open('scaler_w.pkl', 'rb') as f:
    scaler_w = pickle.load(f)

# Load Model
model = ConditionalDiffusionModel().to(DEVICE)
model.load_state_dict(torch.load("diffusion_model.pth", map_location=DEVICE))
model.eval()

# --- 2. Diffusion Schedule (Must match training) ---
beta = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)
alpha_bar_prev = torch.cat([torch.tensor([1.]).to(DEVICE), alpha_hat[:-1]])
posterior_variance = beta * (1. - alpha_bar_prev) / (1. - alpha_hat)

def sample_diffusion(model, condition_trajectory, n_samples):
    """
    Performs the reverse diffusion process: Noise -> Data
    condition_trajectory: Shape (1, 2, 50)
    """
    model.eval()
    with torch.no_grad():
        # Repeat the condition for N samples (we generate N hypotheses for 1 trajectory)
        cond_repeated = condition_trajectory.repeat(n_samples, 1, 1)

        # Start from pure Gaussian noise
        w_current = torch.randn(n_samples, 3).to(DEVICE)

        # Iterate backwards: T -> 0
        for i in reversed(range(TIMESTEPS)):
            t = torch.full((n_samples,), i, device=DEVICE, dtype=torch.long)

            # Predict noise
            predicted_noise = model(w_current, t, cond_repeated)

            # Mathematical coefficients for DDPM sampling
            alpha_t = alpha[i]
            alpha_hat_t = alpha_hat[i]
            beta_t = beta[i]

            if i > 0:
                noise = torch.randn_like(w_current)
            else:
                noise = torch.zeros_like(w_current)

            # Update step: x_{t-1} = 1/sqrt(alpha) * (x_t - ...) + sigma * z
            # This formula removes a small part of the noise
            w_current = (1 / torch.sqrt(alpha_t)) * (
                w_current - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise # Simple sigma option

    return w_current

# --- 3. Run Inference ---

# Pick a random index from the dataset to test
test_idx = np.random.randint(0, len(traj_data))
traj_sample = traj_data[test_idx] # (50, 2)
w_truth = w_true_data[test_idx]   # (3,)

print(f"Testing on sample index: {test_idx}")
print(f"Ground Truth w: {w_truth}")

# Prepare input tensor (Batch=1, Channels=2, Length=50)
traj_tensor = torch.FloatTensor(traj_sample).unsqueeze(0).transpose(1, 2).to(DEVICE)

# Generate multiple potential solutions for this single trajectory
print(f"Generating {N_SAMPLES_TO_GENERATE} potential solutions...")
generated_w_normalized = sample_diffusion(model, traj_tensor, N_SAMPLES_TO_GENERATE)

# Denormalize back to original scale
generated_w = scaler_w.inverse_transform(generated_w_normalized.cpu().numpy())

# --- 4. Analyze Results ---
mean_pred = np.mean(generated_w, axis=0)
std_pred = np.std(generated_w, axis=0)

print("-" * 30)
print(f"Mean Prediction: {mean_pred}")
print(f"Uncertainty (Std): {std_pred}")
print("-" * 30)

# Optional: Simple Plot
# Assumes w has 3 dimensions. We plot histograms of the distribution for each dim.
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
param_names = ['w1', 'w2', 'w3']

for i in range(3):
    ax[i].hist(generated_w[:, i], bins=15, alpha=0.7, color='skyblue', label='Predictions')
    ax[i].axvline(w_truth[i], color='red', linestyle='dashed', linewidth=2, label='Ground Truth')
    ax[i].set_title(f"Distribution of {param_names[i]}")
    ax[i].legend()

plt.suptitle(f"Inference for Sample {test_idx}")
plt.tight_layout()
plt.show()