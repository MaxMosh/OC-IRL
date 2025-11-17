import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

class TestJointAngleDataset(Dataset):
    """Dataset de test pour les angles articulaires q1 et q2 avec structure CSV transposée"""
    def __init__(self, data_root, sequence_length=50, normalization_stats=None):
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalization_stats = normalization_stats
        self.sequences = []
        self.labels = []
        self.file_names = []
        
        # Collecter tous les CSV dans le dossier S18
        self.csv_files = glob.glob(os.path.join(data_root, "S18", "*.csv"))
        
        if not self.csv_files:
            # Essayer aussi dans data/S18 si la structure est différente
            self.csv_files = glob.glob(os.path.join("data", "S18", "*.csv"))
        
        print(f"Found {len(self.csv_files)} CSV files in S18")
        
        # Chargement des séquences de test
        self._load_test_sequences()
    
    def _load_test_sequences(self):
        """Charge toutes les séquences de test depuis les CSV de S18"""
        print("Loading test sequences from S18...")
        for csv_file in tqdm(self.csv_files, desc="Loading test CSVs"):
            try:
                # Charger sans header et transposer
                df = pd.read_csv(csv_file, header=None)
                # Transposer pour avoir (time_steps, 2)
                angles = df.values.T  # Shape: (time_steps, 2)
                
                if angles.shape[1] < 2:
                    print(f"Warning: CSV {csv_file} doesn't have enough columns")
                    continue
                
                # Normalisation avec les stats d'entraînement
                if self.normalization_stats:
                    angle_min = self.normalization_stats['min']
                    angle_max = self.normalization_stats['max']
                    angles = 2 * (angles - angle_min) / (angle_max - angle_min) - 1
                
                # Créer des séquences pour le test
                # On prend plusieurs séquences par fichier pour plus de données de test
                num_sequences = min(10, len(angles) - self.sequence_length)  # Max 10 séquences par fichier
                step = max(1, (len(angles) - self.sequence_length) // num_sequences)
                
                for i in range(0, len(angles) - self.sequence_length, step):
                    if len(self.sequences) >= num_sequences * len(self.csv_files):
                        break
                        
                    seq = angles[i:i + self.sequence_length]
                    target = angles[i + 1:i + self.sequence_length + 1]  # Prédiction un pas ahead
                    
                    self.sequences.append(seq)
                    self.labels.append(target)
                    self.file_names.append(os.path.basename(csv_file))
                    
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        print(f"Created {len(self.sequences)} test sequences of length {self.sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.labels[idx])
        file_name = self.file_names[idx]
        return sequence, target, file_name
    
    def denormalize(self, angles):
        """Dénormalise les angles si la normalisation était activée"""
        if self.normalization_stats:
            angle_min = self.normalization_stats['min']
            angle_max = self.normalization_stats['max']
            angles = (angles + 1) * (angle_max - angle_min) / 2 + angle_min
        return angles

class JointAnglePredictor(nn.Module):
    """Prédicteur d'angles articulaires basé sur l'architecture DAN"""
    def __init__(self, x_dim=2, h_dim=64, a_deep=3, b_deep=3):
        super(JointAnglePredictor, self).__init__()
        
        # Dimensions
        self.x_dim = x_dim  # q1, q2
        self.h_dim = h_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, h_dim)
        )
        
        # Module A (assimilation)
        a_layers = []
        a_layers.append(nn.Linear(2 * h_dim, h_dim))
        a_layers.append(nn.LeakyReLU())
        for _ in range(a_deep - 1):
            a_layers.append(nn.Linear(h_dim, h_dim))
            a_layers.append(nn.LeakyReLU())
        self.a = nn.Sequential(*a_layers)
        
        # Module B (propagation) 
        b_layers = []
        for _ in range(b_deep):
            b_layers.append(nn.Linear(h_dim, h_dim))
            b_layers.append(nn.LeakyReLU())
        self.b = nn.Sequential(*b_layers)
        
        # Module C (décodage -> prédiction)
        self.c = nn.Sequential(
            nn.Linear(h_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, x_dim)
        )
    
    def forward(self, x, hidden=None):
        # Gestion flexible des dimensions d'entrée
        if x.dim() == 2:
            # Si input est [seq_len, features], ajouter une dimension batch
            x = x.unsqueeze(1)
        
        seq_len, batch_size, _ = x.shape
        
        if hidden is None:
            hidden = torch.zeros(batch_size, self.h_dim, device=x.device)
        elif hidden.dim() == 3:
            # Si hidden a 3 dimensions [num_layers, batch_size, hidden_dim], prendre le dernier layer
            hidden = hidden[-1]
        
        predictions = []
        hidden_states = []
        
        for t in range(seq_len):
            x_encoded = self.encoder(x[t])  # [batch_size, h_dim]
            
            if t == 0:
                hidden = self.a(torch.cat([hidden, x_encoded], dim=1))
            else:
                hidden = self.b(hidden)
                hidden = self.a(torch.cat([hidden, x_encoded], dim=1))
            
            hidden_states.append(hidden)
            pred = self.c(hidden)  # [batch_size, x_dim]
            predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)  # [seq_len, batch_size, x_dim]
        hidden_states = torch.stack(hidden_states, dim=0)  # [seq_len, batch_size, h_dim]
        
        return predictions, hidden_states

def load_model(model_path, device):
    """Charge un modèle entraîné avec gestion de weights_only"""
    print(f"Loading model from {model_path}...")
    
    try:
        # Essayer d'abord avec weights_only=True (sécurisé)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except:
        print("weights_only=True failed, trying with weights_only=False...")
        # Si ça échoue, utiliser weights_only=False (moins sécurisé mais compatible)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Récupérer les stats de normalisation
    normalization_stats = checkpoint.get('normalization_stats', None)
    
    # Créer le modèle
    model = JointAnglePredictor(x_dim=2, h_dim=64)
    
    # Vérifier la structure du checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Si le checkpoint est directement le state_dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    if normalization_stats:
        print("Normalization stats found and loaded")
    else:
        print("No normalization stats found in checkpoint")
    
    return model, normalization_stats

def test_model(model, test_loader, device, dataset):
    """Test complet du modèle sur le dataset de test"""
    model.eval()
    criterion = nn.MSELoss()
    
    test_losses = []
    all_predictions = []
    all_targets = []
    file_names = []
    
    print("\nStarting model testing...")
    with torch.no_grad():
        for sequences, targets, files in tqdm(test_loader, desc="Testing", ncols=100):
            sequences = sequences.transpose(0, 1).to(device)
            targets = targets.transpose(0, 1).to(device)
            
            predictions, _ = model(sequences)
            loss = criterion(predictions, targets)
            
            test_losses.append(loss.item())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            file_names.extend(files)
    
    avg_test_loss = np.mean(test_losses)
    
    # Dénormaliser les résultats pour l'analyse
    all_predictions_denorm = dataset.denormalize(np.concatenate(all_predictions, axis=1))
    all_targets_denorm = dataset.denormalize(np.concatenate(all_targets, axis=1))
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    print(f"Average Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Number of test sequences: {len(test_loader.dataset)}")
    print(f"Number of test files: {len(set(file_names))}")
    
    # Calculer RMSE dénormalisé
    rmse = np.sqrt(np.mean((all_predictions_denorm - all_targets_denorm)**2))
    print(f"RMSE (denormalized): {rmse:.4f} radians")
    
    # Calculer MAE dénormalisé
    mae = np.mean(np.abs(all_predictions_denorm - all_targets_denorm))
    print(f"MAE (denormalized): {mae:.4f} radians")
    
    return {
        'test_loss': avg_test_loss,
        'rmse': rmse,
        'mae': mae,
        'predictions': all_predictions_denorm,
        'targets': all_targets_denorm,
        'file_names': file_names
    }

def predict_future_trajectory_fixed(model, initial_sequence, prediction_steps, dataset, device):
    """Version corrigée pour la prédiction future"""
    model.eval()
    
    # S'assurer que la séquence initiale a la bonne forme [seq_len, batch_size, features]
    if initial_sequence.dim() == 2:
        initial_sequence = initial_sequence.unsqueeze(1)  # [seq_len, 1, features]
    
    print(f"Initial sequence shape: {initial_sequence.shape}")
    
    current_sequence = initial_sequence.clone().to(device)
    future_predictions = []
    
    print(f"\nPredicting {prediction_steps} future steps...")
    with torch.no_grad():
        # État initial basé sur la séquence d'entrée complète
        _, hidden_states = model(current_sequence)
        current_hidden = hidden_states[-1]  # Dernier état caché [batch_size, hidden_dim]
        print(f"Initial hidden state shape: {current_hidden.shape}")
        
        # Dernier point de la séquence comme point de départ [1, features]
        current_point = current_sequence[-1, 0, :]  # [features]
        print(f"Initial current point shape: {current_point.shape}")
        
        for step in tqdm(range(prediction_steps), desc="Future prediction"):
            # Préparer l'input pour le modèle [1, 1, features]
            current_input = current_point.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
            print(f"Step {step}: current_input shape: {current_input.shape}")
            
            # Prédire le prochain point
            pred, new_hidden_states = model(current_input, hidden=current_hidden.unsqueeze(0))
            print(f"Step {step}: pred shape: {pred.shape}")
            print(f"Step {step}: new_hidden_states shape: {new_hidden_states.shape}")
            
            # Le prédiction est [1, 1, features], on prend le dernier élément
            next_point = pred[-1, 0, :]  # [features]
            print(f"Step {step}: next_point shape: {next_point.shape}")
            
            future_predictions.append(next_point.cpu().numpy())
            
            # Mettre à jour pour l'itération suivante
            current_point = next_point
            current_hidden = new_hidden_states[-1]  # [batch_size, hidden_dim]
            print(f"Step {step}: updated current_hidden shape: {current_hidden.shape}")
            
            # Arrêter après quelques steps pour debug
            if step >= 2:  # Juste pour debug
                print("Stopping early for debugging...")
                break
    
    future_predictions = np.array(future_predictions)
    print(f"Final future_predictions shape: {future_predictions.shape}")
    
    # Dénormaliser
    future_predictions_denorm = dataset.denormalize(future_predictions)
    
    return future_predictions_denorm

def plot_results(test_results, future_predictions=None, save_dir="test_results"):
    """Visualise les résultats du test"""
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = test_results['predictions']
    targets = test_results['targets']
    
    # Plot 1: Comparaison des prédictions vs targets pour quelques séquences
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Prendre 4 séquences aléatoires à visualiser
    num_sequences = min(4, predictions.shape[1])
    if num_sequences > 0:
        indices = np.random.choice(predictions.shape[1], num_sequences, replace=False)
        
        for i, idx in enumerate(indices):
            seq_len = min(50, predictions.shape[0])  # Afficher les 50 premiers pas de temps
            time_steps = range(seq_len)
            
            # q1
            axes[i*2].plot(time_steps, targets[:seq_len, idx, 0], 'b-', label='True q1', linewidth=2)
            axes[i*2].plot(time_steps, predictions[:seq_len, idx, 0], 'r--', label='Pred q1', linewidth=2)
            axes[i*2].set_title(f'Sequence {idx+1} - q1')
            axes[i*2].set_xlabel('Time step')
            axes[i*2].set_ylabel('q1 (rad)')
            axes[i*2].legend()
            axes[i*2].grid(True)
            
            # q2
            axes[i*2+1].plot(time_steps, targets[:seq_len, idx, 1], 'g-', label='True q2', linewidth=2)
            axes[i*2+1].plot(time_steps, predictions[:seq_len, idx, 1], 'orange', linestyle='--', label='Pred q2', linewidth=2)
            axes[i*2+1].set_title(f'Sequence {idx+1} - q2')
            axes[i*2+1].set_xlabel('Time step')
            axes[i*2+1].set_ylabel('q2 (rad)')
            axes[i*2+1].legend()
            axes[i*2+1].grid(True)
    else:
        # Si pas de séquences, créer des plots vides
        for i in range(4):
            axes[i].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Sequence {i+1}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_predictions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution des erreurs
    errors = np.abs(predictions - targets)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogramme des erreurs pour q1
    axes[0].hist(errors[:, :, 0].flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_xlabel('Absolute Error (rad)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution - q1')
    axes[0].grid(True, alpha=0.3)
    
    # Histogramme des erreurs pour q2
    axes[1].hist(errors[:, :, 1].flatten(), bins=50, alpha=0.7, color='green')
    axes[1].set_xlabel('Absolute Error (rad)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution - q2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Prédiction future si disponible
    if future_predictions is not None and len(future_predictions) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(future_predictions[:, 0], 'r-', label='Predicted q1', linewidth=2)
        plt.plot(future_predictions[:, 1], 'b-', label='Predicted q2', linewidth=2)
        plt.xlabel('Future Time Steps')
        plt.ylabel('Joint Angle (rad)')
        plt.title('Future Trajectory Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'future_trajectory.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved in {save_dir}/")

def save_test_results(test_results, save_path="test_results.json"):
    """Sauvegarde les résultats du test dans un fichier JSON"""
    results_dict = {
        'test_loss': float(test_results['test_loss']),
        'rmse': float(test_results['rmse']),
        'mae': float(test_results['mae']),
        'num_sequences': len(test_results['file_names']),
        'num_files': len(set(test_results['file_names']))
    }
    
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Test results saved to {save_path}")

def main():
    """Fonction principale de test"""
    # Paramètres
    model_path = "best_joint_angle_predictor.pth"  # ou "final_joint_angle_predictor.pth"
    data_root = "data"  # Dossier racine contenant S18
    sequence_length = 50
    batch_size = 32
    prediction_steps = 5  # Réduit à 5 steps pour debug
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Available model files:")
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return
    
    # Charger le modèle
    model, normalization_stats = load_model(model_path, device)
    
    # Créer le dataset de test avec S18
    test_dataset = TestJointAngleDataset(
        data_root=data_root,
        sequence_length=sequence_length,
        normalization_stats=normalization_stats
    )
    
    if len(test_dataset) == 0:
        print("No test data found in S18!")
        print("Please check if S18 folder exists in your data directory.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test du modèle
    test_results = test_model(model, test_loader, device, test_dataset)
    
    # Exemple de prédiction future avec une séquence de test
    if len(test_dataset) > 0:
        # Prendre la première séquence du dataset de test
        sample_sequence, _, file_name = test_dataset[0]
        print(f"Sample sequence shape: {sample_sequence.shape}")
        
        # Utiliser la version corrigée avec seulement 5 steps pour debug
        future_predictions = predict_future_trajectory_fixed(
            model, sample_sequence, prediction_steps, test_dataset, device
        )
        
        print(f"\nFuture prediction example from {file_name}:")
        print(f"Predicted trajectory shape: {future_predictions.shape}")
        if len(future_predictions) > 0:
            print(f"All predicted points:")
            print(f"q1: {future_predictions[:, 0]}")
            print(f"q2: {future_predictions[:, 1]}")
    else:
        future_predictions = None
    
    # Visualisation des résultats
    plot_results(test_results, future_predictions, save_dir="test_results_S18")
    
    # Sauvegarde des résultats
    save_test_results(test_results, "test_results_S18.json")
    
    print(f"\n{'='*50}")
    print("TEST COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Performance summary:")
    print(f"  - RMSE: {test_results['rmse']:.4f} radians")
    print(f"  - MAE: {test_results['mae']:.4f} radians")
    print(f"  - Test Loss (MSE): {test_results['test_loss']:.6f}")

if __name__ == "__main__":
    # Vérifier si tqdm est installé
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm...")
        import subprocess
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm
    
    # Vérifier si matplotlib est installé
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Installing matplotlib...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    
    main()