import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class DDIMSampler:
    """
    DDIM sampler for faster inference.
    """
    def __init__(self, model, num_inference_steps: int = 100):
        self.model = model
        self.num_inference_steps = num_inference_steps
        
        # Create inference timesteps
        self.inference_timesteps = torch.linspace(
            model.T - 1, 0, num_inference_steps
        ).long()
    
    @torch.no_grad()
    def sample(self, obs, num_samples: int = 1, use_noisy_guidance: bool = True):
        """
        Generate predictions using DDIM sampling.
        
        Args:
            obs: Observed motion, shape (batch, obs_frames, input_dim)
            num_samples: Number of samples to generate per observation
            use_noisy_guidance: Whether to use noisy observation guidance
        
        Returns:
            Predicted motions, shape (batch, num_samples, total_frames, input_dim)
        """
        device = obs.device
        batch_size = obs.shape[0]
        
        # Prepare condition
        obs_padded = F.pad(
            obs,
            (0, 0, 0, self.model.pred_frames),
            mode='replicate'
        )
        condition = DCTTransform.dct(obs_padded, self.model.dct_matrix)
        
        # Initialize from noise
        y_T = torch.randn(
            batch_size, num_samples, self.model.num_dct_coeffs, self.model.input_dim,
            device=device
        )
        
        # Iterative denoising
        y_t = y_T
        for i, t in enumerate(tqdm(self.inference_timesteps, desc="Sampling")):
            t_batch = torch.full((batch_size * num_samples,), t, device=device).long()
            print(y_t.shape)
            
            # Reshape for model
            y_t_flat = y_t.view(batch_size * num_samples, -1, self.model.input_dim)
            cond_flat = condition.unsqueeze(1).repeat(1, num_samples, 1, 1).view(
                batch_size * num_samples, -1, self.model.input_dim
            )
            
            # Predict noise
            noise_pred = self.model(y_t_flat, cond_flat, t_batch)
            noise_pred = noise_pred.view(batch_size, num_samples, -1, self.model.input_dim)
            
            # DDIM update
            if i < len(self.inference_timesteps) - 1:
                t_next = self.inference_timesteps[i + 1]
            else:
                t_next = torch.tensor(-1)
            
            y_t = self.ddim_step(y_t, noise_pred, t, t_next)
            
            # Apply noisy observation guidance
            if use_noisy_guidance and i < len(self.inference_timesteps) - 1:
                y_t = self.apply_noisy_guidance(y_t, condition, t_next, obs.shape[1])
        
        # Transform back to time domain
        y_0 = y_t
        y_0_flat = y_0.view(batch_size * num_samples, -1, self.model.input_dim)
        x_pred = DCTTransform.idct(y_0_flat, self.model.dct_matrix)
        x_pred = x_pred.view(batch_size, num_samples, -1, self.model.input_dim)
        
        return x_pred
    
    def ddim_step(self, y_t, noise_pred, t, t_next):
        """
        Perform one DDIM denoising step.
        """
        alpha_t = self.model.alphas_cumprod[t]
        
        if t_next >= 0:
            alpha_t_next = self.model.alphas_cumprod[t_next]
        else:
            alpha_t_next = torch.tensor(1.0)
        
        # Predict x_0
        y_0_pred = (y_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Sample y_{t-1}
        if t_next >= 0:
            noise = torch.randn_like(y_t)
            y_t_next = (
                torch.sqrt(alpha_t_next) * y_0_pred +
                torch.sqrt(1 - alpha_t_next) * noise_pred
            )
        else:
            y_t_next = y_0_pred
        
        return y_t_next
    
    def apply_noisy_guidance(self, y_denoised, condition, t, obs_len):
        """
        Apply noisy observation guidance during sampling.
        """
        # Create mask in frequency domain
        # This is a simplified version - you may need to adjust based on DCT properties
        batch_size, num_samples = y_denoised.shape[:2]
        
        # Sample noisy observation
        alpha_t = self.model.alphas_cumprod[t]
        noise_obs = torch.randn_like(condition)
        y_obs_noisy = (
            torch.sqrt(alpha_t) * condition +
            torch.sqrt(1 - alpha_t) * noise_obs
        )
        
        # Mix observation and prediction
        # Use lower frequency components from observation
        mix_ratio = 0.5  # Can be tuned
        y_denoised[:, :, :self.model.num_dct_coeffs // 2, :] = (
            mix_ratio * y_obs_noisy[:, None, :self.model.num_dct_coeffs // 2, :] +
            (1 - mix_ratio) * y_denoised[:, :, :self.model.num_dct_coeffs // 2, :]
        )
        
        return y_denoised


def train_epoch(model, dataloader, optimizer, device, classifier_free_prob=0.2):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    for obs, target in tqdm(dataloader, desc="Training"):
        obs = obs.to(device)
        target = target.to(device)
        
        # Classifier-free guidance: randomly drop condition
        if np.random.rand() < classifier_free_prob:
            obs_masked = torch.zeros_like(obs)
        else:
            obs_masked = obs
        
        # Compute loss
        loss = model.compute_loss(target, obs_masked)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device, num_samples=50):
    """
    Evaluate the model on test set.
    """
    model.eval()
    sampler = DDIMSampler(model, num_inference_steps=100)
    
    all_predictions = []
    all_targets = []
    all_observations = []
    
    for obs, target in tqdm(dataloader, desc="Evaluating"):
        obs = obs.to(device)
        target = target.to(device)
        
        # Generate predictions
        predictions = sampler.sample(obs, num_samples=num_samples)
        
        all_predictions.append(predictions.cpu())
        all_targets.append(target.cpu())
        all_observations.append(obs.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_observations = torch.cat(all_observations, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets, all_observations)
    
    return metrics, (all_predictions, all_targets, all_observations)


def compute_metrics(predictions, targets, observations):
    """
    Compute evaluation metrics: ADE, FDE, APD.
    
    Args:
        predictions: (num_samples, num_predictions, total_frames, input_dim)
        targets: (num_samples, total_frames, input_dim)
        observations: (num_samples, obs_frames, input_dim)
    """
    num_samples, num_preds = predictions.shape[:2]
    obs_len = observations.shape[1]
    
    # Focus on prediction horizon only
    pred_only = predictions[:, :, obs_len:, :]  # (num_samples, num_preds, pred_frames, input_dim)
    target_only = targets[:, obs_len:, :]  # (num_samples, pred_frames, input_dim)
    
    # Average Displacement Error (ADE) - best of many
    distances = torch.norm(
        pred_only - target_only.unsqueeze(1), dim=-1
    )  # (num_samples, num_preds, pred_frames)
    ade_per_pred = distances.mean(dim=-1)  # (num_samples, num_preds)
    ade_best = ade_per_pred.min(dim=1)[0].mean().item()
    ade_median = ade_per_pred.median(dim=1)[0].mean().item()
    ade_worst = ade_per_pred.max(dim=1)[0].mean().item()
    
    # Final Displacement Error (FDE) - best of many
    fde_per_pred = distances[:, :, -1]  # (num_samples, num_preds)
    fde_best = fde_per_pred.min(dim=1)[0].mean().item()
    fde_median = fde_per_pred.median(dim=1)[0].mean().item()
    fde_worst = fde_per_pred.max(dim=1)[0].mean().item()
    
    # Average Pairwise Distance (APD) - diversity measure
    apd_list = []
    for i in range(num_samples):
        preds_i = pred_only[i]  # (num_preds, pred_frames, input_dim)
        pairwise_dists = []
        for j in range(num_preds):
            for k in range(j + 1, num_preds):
                dist = torch.norm(preds_i[j] - preds_i[k], dim=-1).mean().item()
                pairwise_dists.append(dist)
        if pairwise_dists:
            apd_list.append(np.mean(pairwise_dists))
    apd = np.mean(apd_list) if apd_list else 0.0
    
    metrics = {
        'ADE_best': ade_best,
        'ADE_median': ade_median,
        'ADE_worst': ade_worst,
        'FDE_best': fde_best,
        'FDE_median': fde_median,
        'FDE_worst': fde_worst,
        'APD': apd
    }
    
    return metrics


def visualize_predictions(observations, predictions, targets, idx=0, num_to_show=10):
    """
    Visualize predictions for a single sample.
    
    Args:
        observations: (num_samples, obs_frames, 2)
        predictions: (num_samples, num_predictions, total_frames, 2)
        targets: (num_samples, total_frames, 2)
        idx: Index of sample to visualize
        num_to_show: Number of predictions to show
    """
    obs = observations[idx].numpy()
    preds = predictions[idx, :num_to_show].numpy()
    target = targets[idx].numpy()
    
    obs_len = obs.shape[0]
    total_len = target.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot q1
    axes[0].plot(range(obs_len), obs[:, 0], 'k-', linewidth=2, label='Observation')
    axes[0].plot(range(obs_len, total_len), target[obs_len:, 0], 'g-', linewidth=2, label='Ground Truth')
    for i in range(num_to_show):
        axes[0].plot(range(obs_len, total_len), preds[i, obs_len:, 0], 'r-', alpha=0.3)
    axes[0].axvline(x=obs_len, color='b', linestyle='--', label='Prediction Start')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Angle (rad)')
    axes[0].set_title('Joint Angle q1')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot q2
    axes[1].plot(range(obs_len), obs[:, 1], 'k-', linewidth=2, label='Observation')
    axes[1].plot(range(obs_len, total_len), target[obs_len:, 1], 'g-', linewidth=2, label='Ground Truth')
    for i in range(num_to_show):
        axes[1].plot(range(obs_len, total_len), preds[i, obs_len:, 1], 'r-', alpha=0.3)
    axes[1].axvline(x=obs_len, color='b', linestyle='--', label='Prediction Start')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].set_title('Joint Angle q2')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'predictions_visualization.png'")


# Import DCTTransform from model file
from transfusion_model import DCTTransform
