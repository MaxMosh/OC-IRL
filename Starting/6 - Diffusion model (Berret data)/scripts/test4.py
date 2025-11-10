import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def test_gaussian_noise_and_denoise(model, diffusion, dataloader, device):
    """
    Test the noise and denoise process on a synthetic Gaussian noise trajectory.
    """
    # Create synthetic Gaussian noise trajectory (initial random noise)
    noise_trajectory = torch.randn(1, MAX_LENGTH, TRAJECTORY_DIM, device=device)
    
    print(f"Initial Gaussian noise shape: {noise_trajectory.shape}")
    
    # Add noise (forward diffusion) at various timesteps
    timesteps_to_test = [100, 300, 500, 700, 900]
    
    fig, axes = plt.subplots(len(timesteps_to_test), 3, figsize=(15, 4 * len(timesteps_to_test)))
    
    for idx, t_value in enumerate(timesteps_to_test):
        t = torch.tensor([t_value], device=device)  # timestep tensor
        
        # Add noise using forward diffusion process
        noised_trajectory, true_noise = diffusion.forward_diffusion(noise_trajectory, t)
        
        # Denoise (reverse diffusion process) from the noisy trajectory
        denoised_trajectory = noised_trajectory.clone()
        
        # Perform reverse diffusion step by step
        for i in reversed(range(1, t_value + 1)):
            t_reverse = torch.tensor([i], device=device)
            predicted_noise = model(denoised_trajectory, t_reverse)
            
            sqrt_alpha = torch.sqrt(diffusion.alpha[i])
            one_minus_alpha = 1.0 - diffusion.alpha[i]
            sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - diffusion.alpha_hat[i])
            
            denoised_trajectory = (1 / sqrt_alpha) * (denoised_trajectory - one_minus_alpha * predicted_noise / sqrt_one_minus_alpha_hat)
            
            if i > 1:
                noise_term = torch.randn_like(denoised_trajectory, device=device)
                denoised_trajectory = denoised_trajectory + torch.sqrt(diffusion.beta[i]) * noise_term
        
        # Convert to numpy for plotting
        original = noise_trajectory[0].cpu().numpy()
        noised = noised_trajectory[0].cpu().numpy()
        denoised_np = denoised_trajectory[0].cpu().numpy()
        
        # Plot original, noised, and denoised trajectories
        axes[idx, 0].plot(original[:, 0], label='q1', alpha=0.7)
        axes[idx, 0].plot(original[:, 1], label='q2', alpha=0.7)
        axes[idx, 0].set_title(f'Original Trajectory')
        axes[idx, 0].set_ylabel(f't={t_value}')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        axes[idx, 1].plot(noised[:, 0], label='q1', alpha=0.7)
        axes[idx, 1].plot(noised[:, 1], label='q2', alpha=0.7)
        axes[idx, 1].set_title(f'Noised Trajectory (t={t_value})')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
        
        axes[idx, 2].plot(denoised_np[:, 0], label='q1', alpha=0.7)
        axes[idx, 2].plot(denoised_np[:, 1], label='q2', alpha=0.7)
        axes[idx, 2].set_title(f'Denoised Trajectory')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True)
    
    plt.tight_layout()

    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'gaussian_noise_denoise_test_{timestamp}.png'
    plt.savefig(plot_filename, dpi=150)
    plt.show()
    
    print(f"Plot saved as {plot_filename}")
    print("\nTest complete!")

