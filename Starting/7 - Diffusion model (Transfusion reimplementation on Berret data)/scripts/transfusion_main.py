"""
Main training script for TransFusion on joint angle prediction.

This script trains the TransFusion model on your joint angle dataset
and evaluates its performance on the test set.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import os
import json
from datetime import datetime

# Import custom modules
# Make sure these files are in the same directory or adjust the imports
from transfusion_dataloader import create_dataloaders
from transfusion_model import TransFusion
from transfusion_inference import train_epoch, evaluate, visualize_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TransFusion for joint angle prediction')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing S01, S02, ... folders')
    parser.add_argument('--obs_frames', type=int, default=25,
                        help='Number of observation frames')
    parser.add_argument('--pred_frames', type=int, default=100,
                        help='Number of prediction frames')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512,
                        help='Dimension of model')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=9,
                        help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--num_dct_coeffs', type=int, default=20,
                        help='Number of DCT coefficients to keep')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                        help='Number of diffusion steps during training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.8,
                        help='Learning rate decay factor')
    parser.add_argument('--lr_decay_step', type=int, default=100,
                        help='Decay learning rate every N epochs')
    parser.add_argument('--classifier_free_prob', type=float, default=0.2,
                        help='Probability of dropping condition during training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Evaluation parameters
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='Evaluate every N epochs')
    parser.add_argument('--num_eval_samples', type=int, default=50,
                        help='Number of samples to generate during evaluation')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define train and test subjects
    # Adjust these based on your data structure
    all_subjects = [f'S{i:02d}' for i in range(1, 16)]  # S01 to S15
    train_subjects = all_subjects[:12]  # S01 to S12 for training
    test_subjects = all_subjects[12:]   # S13 to S15 for testing
    
    print(f"Train subjects: {train_subjects}")
    print(f"Test subjects: {test_subjects}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        obs_frames=args.obs_frames,
        pred_frames=args.pred_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = TransFusion(
        input_dim=2,  # q1 and q2
        obs_frames=args.obs_frames,
        pred_frames=args.pred_frames,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_dct_coeffs=args.num_dct_coeffs,
        T=args.diffusion_steps
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metrics = {}
    if args.resume:
        start_epoch, best_metrics = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
        start_epoch += 1
    
    # Training loop
    print("\nStarting training...")
    best_ade = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            classifier_free_prob=args.classifier_free_prob
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.6f}, LR: {current_lr:.6f}")
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_interval == 0:
            print("\nEvaluating on test set...")
            metrics, (predictions, targets, observations) = evaluate(
                model, test_loader, device, num_samples=args.num_eval_samples
            )
            
            print("\nTest Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
            
            # Save checkpoint if best model
            if metrics['ADE_best'] < best_ade:
                best_ade = metrics['ADE_best']
                best_metrics = metrics
                save_path = os.path.join(args.save_dir, 'best_model.pt')
                save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path)
                
                # Visualize predictions
                print("\nGenerating visualization...")
                visualize_predictions(observations, predictions, targets, idx=0)
        
        # Save periodic checkpoint
        if (epoch + 1) % 100 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, {}, save_path)
    
    # Final evaluation
    print("\n" + "="*50)
    print("Training completed! Running final evaluation...")
    print("="*50)
    
    metrics, (predictions, targets, observations) = evaluate(
        model, test_loader, device, num_samples=args.num_eval_samples
    )
    
    print("\nFinal Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save final model
    save_path = os.path.join(args.save_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, metrics, save_path)
    
    # Save best metrics
    with open(os.path.join(args.save_dir, 'best_metrics.json'), 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    print("\nTraining finished!")
    print(f"Best ADE: {best_ade:.6f}")
    print(f"Checkpoints saved in: {args.save_dir}")


if __name__ == "__main__":
    main()
