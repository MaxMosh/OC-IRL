"""
Test script to generate trajectory predictions from a given observation prefix.

This script loads a trained TransFusion model and generates multiple
trajectory predictions from a CSV file in the test set (e.g., from S18 folder).
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from pathlib import Path

# Import custom modules
# from transfusion_model import TransFusion, DCTTransform
# from transfusion_inference import DDIMSampler


class TransFusionPredictor:
    """
    Wrapper class for generating predictions with a trained TransFusion model.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration from checkpoint if available
        # Otherwise use default values
        self.obs_frames = 25
        self.pred_frames = 100
        self.total_frames = self.obs_frames + self.pred_frames
        
        # Create model
        from transfusion_model import TransFusion
        self.model = TransFusion(
            input_dim=2,
            obs_frames=self.obs_frames,
            pred_frames=self.pred_frames,
            d_model=512,
            nhead=8,
            num_layers=9,
            dim_feedforward=2048,
            dropout=0.1,
            num_dct_coeffs=20,
            T=1000
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Load normalization statistics if available
        if 'normalization_stats' in checkpoint:
            self.mean = checkpoint['normalization_stats']['mean'].astype(np.float32)
            self.std = checkpoint['normalization_stats']['std'].astype(np.float32)
            print(f"Loaded normalization stats - Mean: {self.mean}, Std: {self.std}")
        else:
            # Use default normalization (you should save these during training)
            self.mean = np.array([0.0, 0.0], dtype=np.float32)
            self.std = np.array([1.0, 1.0], dtype=np.float32)
            print("Warning: No normalization stats found in checkpoint, using defaults")
        
        # Create DDIM sampler
        from transfusion_inference import DDIMSampler
        self.sampler = DDIMSampler(self.model, num_inference_steps=100)
    
    def load_csv_sequence(self, csv_path):
        """
        Load a sequence from a CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Array of shape (time_steps, 2) containing q1 and q2
        """
        print(f"Loading sequence from {csv_path}...")
        df = pd.read_csv(csv_path, header=None)
        
        # Transpose to get shape (time_steps, 2)
        data = df.T.values.astype(np.float32)  # Ensure float32
        
        print(f"Loaded sequence with {data.shape[0]} time steps")
        return data
    
    def normalize_data(self, data):
        """Normalize data using saved statistics."""
        # Ensure float32 for consistency
        return ((data - self.mean) / self.std).astype(np.float32)
    
    def denormalize_data(self, data):
        """Denormalize data back to original scale."""
        # Ensure float32 for consistency
        return (data * self.std + self.mean).astype(np.float32)
    
    def generate_predictions(
        self, 
        observation, 
        num_samples=50,
        use_noisy_guidance=True
    ):
        """
        Generate multiple trajectory predictions from an observation.
        
        Args:
            observation: Observation sequence of shape (obs_frames, 2)
            num_samples: Number of trajectory samples to generate
            use_noisy_guidance: Whether to use noisy observation guidance
            
        Returns:
            predictions: Array of shape (num_samples, total_frames, 2)
        """
        # Normalize observation
        obs_normalized = self.normalize_data(observation)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.from_numpy(obs_normalized).unsqueeze(0).to(self.device)
        
        # Generate predictions
        print(f"Generating {num_samples} trajectory predictions...")
        with torch.no_grad():
            predictions = self.sampler.sample(
                obs_tensor, 
                num_samples=num_samples,
                use_noisy_guidance=use_noisy_guidance
            )
        
        # Remove batch dimension and convert to numpy
        predictions = predictions.squeeze(0).cpu().numpy()
        
        # Denormalize predictions
        predictions_denorm = np.zeros_like(predictions)
        for i in range(num_samples):
            predictions_denorm[i] = self.denormalize_data(predictions[i])
        
        return predictions_denorm
    
    def extract_observation_from_sequence(self, sequence, start_idx=0):
        """
        Extract an observation window from a full sequence.
        
        Args:
            sequence: Full sequence array of shape (time_steps, 2)
            start_idx: Starting index for observation window
            
        Returns:
            observation: Array of shape (obs_frames, 2)
            ground_truth: Array of shape (total_frames, 2) or None if not enough data
        """
        if start_idx + self.obs_frames > len(sequence):
            raise ValueError(f"Not enough frames for observation. Need at least {self.obs_frames} frames from index {start_idx}")
        
        observation = sequence[start_idx:start_idx + self.obs_frames]
        
        # Extract ground truth if available
        if start_idx + self.total_frames <= len(sequence):
            ground_truth = sequence[start_idx:start_idx + self.total_frames]
        else:
            ground_truth = None
            print("Warning: Not enough frames for full ground truth")
        
        return observation, ground_truth


def visualize_predictions_advanced(
    observation,
    predictions,
    ground_truth=None,
    num_to_show=20,
    save_path='test_predictions.png',
    show_confidence=True
):
    """
    Advanced visualization of predictions with confidence bands.
    
    Args:
        observation: Observation sequence (obs_frames, 2)
        predictions: Predicted trajectories (num_samples, total_frames, 2)
        ground_truth: Ground truth trajectory (total_frames, 2) or None
        num_to_show: Number of individual predictions to show
        save_path: Path to save the figure
        show_confidence: Whether to show confidence bands
    """
    obs_len = observation.shape[0]
    total_len = predictions.shape[1]
    num_samples = predictions.shape[0]
    
    # Create figure with 2 rows and 2 columns
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Individual joint angles
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Bottom row: 2D trajectory and statistics
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    time_obs = np.arange(obs_len)
    time_pred = np.arange(obs_len, total_len)
    time_all = np.arange(total_len)
    
    # Plot q1
    ax1.plot(time_obs, observation[:, 0], 'b-', linewidth=3, label='Observation', marker='o')
    if ground_truth is not None:
        ax1.plot(time_pred, ground_truth[obs_len:, 0], 'g-', linewidth=3, 
                label='Ground Truth', marker='s', markersize=4)
    
    # Plot individual predictions
    for i in range(min(num_to_show, num_samples)):
        alpha = 0.15 if num_to_show > 10 else 0.3
        ax1.plot(time_pred, predictions[i, obs_len:, 0], 'r-', alpha=alpha, linewidth=1)
    
    # Plot confidence bands
    if show_confidence and num_samples > 1:
        pred_mean = predictions[:, obs_len:, 0].mean(axis=0)
        pred_std = predictions[:, obs_len:, 0].std(axis=0)
        ax1.plot(time_pred, pred_mean, 'r-', linewidth=2, label='Mean Prediction')
        ax1.fill_between(time_pred, pred_mean - pred_std, pred_mean + pred_std,
                        color='r', alpha=0.2, label='±1 std')
        ax1.fill_between(time_pred, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                        color='r', alpha=0.1, label='±2 std')
    
    ax1.axvline(x=obs_len, color='k', linestyle='--', linewidth=2, label='Prediction Start')
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Angle (rad)', fontsize=12)
    ax1.set_title('Joint Angle q1', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot q2
    ax2.plot(time_obs, observation[:, 1], 'b-', linewidth=3, label='Observation', marker='o')
    if ground_truth is not None:
        ax2.plot(time_pred, ground_truth[obs_len:, 1], 'g-', linewidth=3,
                label='Ground Truth', marker='s', markersize=4)
    
    for i in range(min(num_to_show, num_samples)):
        alpha = 0.15 if num_to_show > 10 else 0.3
        ax2.plot(time_pred, predictions[i, obs_len:, 1], 'r-', alpha=alpha, linewidth=1)
    
    if show_confidence and num_samples > 1:
        pred_mean = predictions[:, obs_len:, 1].mean(axis=0)
        pred_std = predictions[:, obs_len:, 1].std(axis=0)
        ax2.plot(time_pred, pred_mean, 'r-', linewidth=2, label='Mean Prediction')
        ax2.fill_between(time_pred, pred_mean - pred_std, pred_mean + pred_std,
                        color='r', alpha=0.2, label='±1 std')
        ax2.fill_between(time_pred, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
                        color='r', alpha=0.1, label='±2 std')
    
    ax2.axvline(x=obs_len, color='k', linestyle='--', linewidth=2, label='Prediction Start')
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Angle (rad)', fontsize=12)
    ax2.set_title('Joint Angle q2', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 2D trajectory (q1 vs q2)
    ax3.plot(observation[:, 0], observation[:, 1], 'b-', linewidth=3,
            label='Observation', marker='o', markersize=6)
    ax3.plot(observation[0, 0], observation[0, 1], 'go', markersize=12,
            label='Start', zorder=5)
    
    if ground_truth is not None:
        ax3.plot(ground_truth[obs_len:, 0], ground_truth[obs_len:, 1], 'g-',
                linewidth=3, label='Ground Truth', marker='s', markersize=4)
        ax3.plot(ground_truth[-1, 0], ground_truth[-1, 1], 'gs', markersize=12,
                label='GT End', zorder=5)
    
    for i in range(min(num_to_show, num_samples)):
        alpha = 0.15 if num_to_show > 10 else 0.3
        ax3.plot(predictions[i, obs_len:, 0], predictions[i, obs_len:, 1],
                'r-', alpha=alpha, linewidth=1)
        ax3.plot(predictions[i, -1, 0], predictions[i, -1, 1],
                'rx', alpha=alpha, markersize=8)
    
    ax3.set_xlabel('q1 (rad)', fontsize=12)
    ax3.set_ylabel('q2 (rad)', fontsize=12)
    ax3.set_title('2D Trajectory (q1 vs q2)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot statistics
    if ground_truth is not None:
        # Compute errors for each prediction
        errors = []
        final_errors = []
        for i in range(num_samples):
            error = np.linalg.norm(predictions[i, obs_len:] - ground_truth[obs_len:], axis=1)
            errors.append(error.mean())
            final_errors.append(error[-1])
        
        errors = np.array(errors)
        final_errors = np.array(final_errors)
        
        # Create box plots
        ax4.boxplot([errors, final_errors], labels=['ADE', 'FDE'])
        ax4.set_ylabel('Error (rad)', fontsize=12)
        ax4.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        stats_text = f"ADE: {errors.mean():.4f} ± {errors.std():.4f}\n"
        stats_text += f"FDE: {final_errors.mean():.4f} ± {final_errors.std():.4f}\n"
        stats_text += f"Min ADE: {errors.min():.4f}\n"
        stats_text += f"Max ADE: {errors.max():.4f}\n"
        stats_text += f"Samples: {num_samples}"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Show diversity statistics if no ground truth
        diversity_q1 = predictions[:, obs_len:, 0].std(axis=0).mean()
        diversity_q2 = predictions[:, obs_len:, 1].std(axis=0).mean()
        
        # Plot diversity over time
        time_pred_range = np.arange(len(time_pred))
        div_q1_over_time = predictions[:, obs_len:, 0].std(axis=0)
        div_q2_over_time = predictions[:, obs_len:, 1].std(axis=0)
        
        ax4.plot(time_pred_range, div_q1_over_time, 'b-', linewidth=2, label='q1 diversity')
        ax4.plot(time_pred_range, div_q2_over_time, 'r-', linewidth=2, label='q2 diversity')
        ax4.set_xlabel('Prediction Frame', fontsize=12)
        ax4.set_ylabel('Std Deviation (rad)', fontsize=12)
        ax4.set_title('Prediction Diversity Over Time', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        stats_text = f"Avg diversity q1: {diversity_q1:.4f}\n"
        stats_text += f"Avg diversity q2: {diversity_q2:.4f}\n"
        stats_text += f"Samples: {num_samples}"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'TransFusion Predictions ({num_samples} samples, showing {min(num_to_show, num_samples)})',
                fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.show()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test TransFusion model on a CSV sequence')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file for testing (e.g., data/S18/sequence_001.csv)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting frame index for observation window')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of trajectory samples to generate')
    parser.add_argument('--num_to_show', type=int, default=20,
                       help='Number of trajectories to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--save_path', type=str, default='test_predictions.png',
                       help='Path to save visualization')
    parser.add_argument('--use_noisy_guidance', action='store_true',
                       help='Use noisy observation guidance during sampling')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        
        # Try to find CSV files in S18 folder
        data_dir = Path(args.csv_path).parent.parent
        s18_path = data_dir / 'S18'
        if s18_path.exists():
            csv_files = list(s18_path.glob('*.csv'))
            if csv_files:
                print(f"\nFound {len(csv_files)} CSV files in S18 folder:")
                for i, csv_file in enumerate(csv_files[:5]):
                    print(f"  {i+1}. {csv_file}")
                if len(csv_files) > 5:
                    print(f"  ... and {len(csv_files) - 5} more")
        return
    
    print("="*60)
    print("TransFusion Test Script - Generate Trajectory Predictions")
    print("="*60)
    
    # Initialize predictor
    predictor = TransFusionPredictor(args.checkpoint, device=args.device)
    
    # Load sequence from CSV
    sequence = predictor.load_csv_sequence(args.csv_path)
    
    # Extract observation and ground truth
    try:
        observation, ground_truth = predictor.extract_observation_from_sequence(
            sequence, start_idx=args.start_idx
        )
        print(f"\nObservation shape: {observation.shape}")
        if ground_truth is not None:
            print(f"Ground truth shape: {ground_truth.shape}")
        else:
            print("No ground truth available (sequence too short)")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Sequence length: {len(sequence)} frames")
        print(f"Required: at least {predictor.obs_frames} frames from index {args.start_idx}")
        return
    
    # Generate predictions
    print("\n" + "="*60)
    predictions = predictor.generate_predictions(
        observation,
        num_samples=args.num_samples,
        use_noisy_guidance=args.use_noisy_guidance
    )
    print(f"Generated predictions shape: {predictions.shape}")
    
    # Compute statistics if ground truth is available
    if ground_truth is not None:
        print("\n" + "="*60)
        print("Prediction Statistics:")
        print("="*60)
        
        # Compute errors
        obs_len = observation.shape[0]
        errors = []
        final_errors = []
        
        for i in range(args.num_samples):
            error = np.linalg.norm(predictions[i, obs_len:] - ground_truth[obs_len:], axis=1)
            errors.append(error.mean())
            final_errors.append(error[-1])
        
        errors = np.array(errors)
        final_errors = np.array(final_errors)
        
        print(f"ADE (Average Displacement Error):")
        print(f"  Best:   {errors.min():.6f}")
        print(f"  Median: {np.median(errors):.6f}")
        print(f"  Mean:   {errors.mean():.6f} ± {errors.std():.6f}")
        print(f"  Worst:  {errors.max():.6f}")
        
        print(f"\nFDE (Final Displacement Error):")
        print(f"  Best:   {final_errors.min():.6f}")
        print(f"  Median: {np.median(final_errors):.6f}")
        print(f"  Mean:   {final_errors.mean():.6f} ± {final_errors.std():.6f}")
        print(f"  Worst:  {final_errors.max():.6f}")
    
    # Compute diversity
    print("\nDiversity Statistics:")
    obs_len = observation.shape[0]
    diversity_q1 = predictions[:, obs_len:, 0].std(axis=0).mean()
    diversity_q2 = predictions[:, obs_len:, 1].std(axis=0).mean()
    print(f"  q1 diversity (avg std): {diversity_q1:.6f}")
    print(f"  q2 diversity (avg std): {diversity_q2:.6f}")
    
    # Visualize predictions
    print("\n" + "="*60)
    print("Creating visualization...")
    print("="*60)
    
    visualize_predictions_advanced(
        observation=observation,
        predictions=predictions,
        ground_truth=ground_truth,
        num_to_show=args.num_to_show,
        save_path=args.save_path,
        show_confidence=True
    )
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
