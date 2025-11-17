import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import time

class JointAngleDataset(Dataset):
    """Dataset pour les angles articulaires q1 et q2 avec structure CSV transposée"""
    def __init__(self, data_root, sequence_length=50, normalize=True):
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.sequences = []
        self.labels = []
        self.angle_min = None
        self.angle_max = None
        
        # Collecter tous les CSV dans les dossiers S01 à S15
        self.csv_files = []
        for subject in range(1, 16):
            subject_folder = os.path.join(data_root, f"S{subject:02d}")
            if os.path.exists(subject_folder):
                csv_files = glob.glob(os.path.join(subject_folder, "*.csv"))
                self.csv_files.extend(csv_files)
        
        print(f"Found {len(self.csv_files)} CSV files")
        
        # Première passe pour calculer les stats de normalisation
        if normalize:
            self._compute_normalization_stats()
        
        # Chargement des séquences
        self._load_sequences()
    
    def _compute_normalization_stats(self):
        """Calcule les statistiques de normalisation sur tous les CSV"""
        all_angles = []
        
        print("Computing normalization statistics...")
        for csv_file in tqdm(self.csv_files, desc="Processing CSV files"):
            try:
                # Charger sans header et transposer
                df = pd.read_csv(csv_file, header=None)
                # Transposer pour avoir (time_steps, 2)
                angles = df.values.T  # Shape: (2, time_steps) -> après transposition: (time_steps, 2)
                
                if angles.shape[1] >= 2:  # Vérifier qu'on a bien q1 et q2
                    all_angles.append(angles)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        if all_angles:
            all_angles = np.vstack(all_angles)
            self.angle_min = all_angles.min(axis=0)
            self.angle_max = all_angles.max(axis=0)
            print(f"Normalization stats - Min: {self.angle_min}, Max: {self.angle_max}")
        else:
            self.angle_min = np.array([0.0, 0.0])
            self.angle_max = np.array([1.0, 1.0])
    
    def _load_sequences(self):
        """Charge toutes les séquences depuis les CSV"""
        print("Loading sequences...")
        for csv_file in tqdm(self.csv_files, desc="Creating sequences"):
            try:
                # Charger sans header et transposer
                df = pd.read_csv(csv_file, header=None)
                # Transposer pour avoir (time_steps, 2)
                angles = df.values.T  # Shape: (time_steps, 2)
                
                if angles.shape[1] < 2:
                    continue
                
                # Normalisation
                if self.normalize and self.angle_min is not None and self.angle_max is not None:
                    angles = 2 * (angles - self.angle_min) / (self.angle_max - self.angle_min) - 1
                
                # Créer des séquences superposées
                for i in range(len(angles) - self.sequence_length):
                    seq = angles[i:i + self.sequence_length]
                    target = angles[i + 1:i + self.sequence_length + 1]  # Prédiction un pas ahead
                    
                    self.sequences.append(seq)
                    self.labels.append(target)
                    
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        print(f"Created {len(self.sequences)} sequences of length {self.sequence_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor(self.labels[idx])
        return sequence, target
    
    def denormalize(self, angles):
        """Dénormalise les angles si la normalisation était activée"""
        if self.normalize and self.angle_min is not None and self.angle_max is not None:
            angles = (angles + 1) * (self.angle_max - self.angle_min) / 2 + self.angle_min
        return angles

class JointAnglePredictor(nn.Module):
    """Prédicteur d'angles articulaires basé sur l'architecture DAN"""
    def __init__(self, x_dim=2, h_dim=64, a_deep=3, b_deep=3):
        super(JointAnglePredictor, self).__init__()
        
        # Dimensions
        self.x_dim = x_dim  # q1, q2
        self.h_dim = h_dim
        
        # Encoder - réduit la dimension pour éviter les problèmes
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 16),  # Réduit de 2 à 16
            nn.LeakyReLU(),
            nn.Linear(16, h_dim)
        )
        
        # Module A (assimilation) - CORRECTION ICI
        # Input: hidden (h_dim) + encoded (h_dim) = 2*h_dim
        a_layers = []
        a_layers.append(nn.Linear(2 * h_dim, h_dim))  # CORRIGÉ: 2*h_dim au lieu de h_dim + x_dim
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
            nn.Linear(32, x_dim)  # Prédit q1_next, q2_next
        )
        
        self.scores = {
            "train_loss": [],
            "val_loss": []
        }
    
    def forward(self, x, hidden=None):
        """
        x: séquence d'entrée [seq_len, batch, x_dim]
        hidden: état caché initial
        """
        seq_len, batch_size, _ = x.shape
        
        # Initialiser l'état caché si non fourni
        if hidden is None:
            hidden = torch.zeros(batch_size, self.h_dim, device=x.device)
        
        predictions = []
        hidden_states = []
        
        # Traiter la séquence pas à pas
        for t in range(seq_len):
            # Encoder l'entrée courante
            x_encoded = self.encoder(x[t])
            
            # Mettre à jour l'état caché avec le module A
            if t == 0:
                # Pour le premier pas, concaténer avec l'état caché initial
                hidden = self.a(torch.cat([hidden, x_encoded], dim=1))
            else:
                # Propagation avec le module B
                hidden = self.b(hidden)
                # Assimilation avec le module A
                hidden = self.a(torch.cat([hidden, x_encoded], dim=1))
            
            hidden_states.append(hidden)
            
            # Prédiction avec le module C
            pred = self.c(hidden)
            predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        hidden_states = torch.stack(hidden_states, dim=0)
        
        return predictions, hidden_states

def train_joint_angle_predictor():
    """Fonction d'entraînement pour le prédicteur d'angles"""
    
    # Paramètres
    data_root = "data"  # Dossier racine contenant S01, S02, etc.
    sequence_length = 50
    batch_size = 32
    learning_rate = 1e-3
    epochs = 3                                                                                      # INITIALLY 100
    
    # Chargement des données
    print("Loading dataset...")
    dataset = JointAngleDataset(data_root, sequence_length)
    
    if len(dataset) == 0:
        print("No data found! Check your data directory structure.")
        return None
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    # Modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = JointAnglePredictor(x_dim=2, h_dim=64).to(device)
    
    # Afficher les dimensions pour debug
    print(f"Model architecture:")
    print(f"  - Input dim: {model.x_dim}")
    print(f"  - Hidden dim: {model.h_dim}")
    print(f"  - Module A input: {2 * model.h_dim}")  # hidden + encoded
    print(f"  - Module A output: {model.h_dim}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Test d'une passe forward pour vérifier les dimensions
    print("Testing forward pass with sample data...")
    with torch.no_grad():
        sample_sequence, sample_target = next(iter(train_loader))
        sample_sequence = sample_sequence.transpose(0, 1).to(device)
        predictions, _ = model(sample_sequence)
        print(f"Sample input shape: {sample_sequence.shape}")
        print(f"Sample predictions shape: {predictions.shape}")
        print(f"Sample target shape: {sample_target.transpose(0, 1).shape}")
    
    # Entraînement
    print(f"\nStarting training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Phase d'entraînement
        model.train()
        train_loss = 0
        
        # Barre de progression pour l'entraînement
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                         leave=False, ncols=100)
        
        for batch_idx, (sequences, targets) in enumerate(train_pbar):
            sequences = sequences.transpose(0, 1).to(device)  # [seq_len, batch, features]
            targets = targets.transpose(0, 1).to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(sequences)
            
            # Calcul de la loss (prédiction un pas ahead)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Mettre à jour la barre de progression
            current_loss = train_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'loss': f'{current_loss:.6f}',
                'batch': f'{batch_idx+1}/{len(train_loader)}'
            })
        
        # Phase de validation
        model.eval()
        val_loss = 0
        
        # Barre de progression pour la validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, (sequences, targets) in enumerate(val_pbar):
                sequences = sequences.transpose(0, 1).to(device)
                targets = targets.transpose(0, 1).to(device)
                
                predictions, _ = model(sequences)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                
                # Mettre à jour la barre de progression
                current_val_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'val_loss': f'{current_val_loss:.6f}',
                    'batch': f'{batch_idx+1}/{len(val_loader)}'
                })
        
        # Calcul des losses moyennes
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        model.scores["train_loss"].append(avg_train_loss)
        model.scores["val_loss"].append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        # Sauvegarder le meilleur modèle
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'normalization_stats': {
                    'min': dataset.angle_min,
                    'max': dataset.angle_max
                } if dataset.normalize else None
            }, 'best_joint_angle_predictor.pth')
        
        # Affichage détaillé de la progression
        print(f'Epoch {epoch+1:03d}/{epochs} | '
              f'Time: {epoch_time:.2f}s | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'Best Val: {best_val_loss:.6f}')
        
        # Affichage d'une barre de progression globale
        progress = (epoch + 1) / epochs * 100
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) // epochs)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'[{bar}] {progress:.1f}% - Epoch {epoch+1}/{epochs}\n')
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    return model, dataset

def predict_trajectory(model, initial_sequence, prediction_steps, dataset=None):
    """Prédire une trajectoire future"""
    model.eval()
    device = next(model.parameters()).device
    
    current_sequence = initial_sequence.clone().to(device)  # [seq_len, 1, x_dim]
    predictions = []
    
    print(f"Predicting {prediction_steps} future steps...")
    with torch.no_grad():
        # État initial basé sur la séquence d'entrée
        _, hidden = model(current_sequence)
        last_hidden = hidden[-1:]  # Dernier état caché
        
        # Dernier point de la séquence comme point de départ
        current_point = current_sequence[-1:]
        
        # Barre de progression pour la prédiction
        pred_pbar = tqdm(range(prediction_steps), desc="Predicting trajectory", ncols=80)
        
        for step in pred_pbar:
            # Prédire le prochain point
            pred, last_hidden = model(current_point.unsqueeze(1), hidden=last_hidden)
            next_point = pred[-1:]
            
            predictions.append(next_point.squeeze().cpu().numpy())
            current_point = next_point
            
            pred_pbar.set_postfix({'step': step + 1})
    
    predictions = np.array(predictions)
    
    # Dénormaliser si nécessaire
    if dataset is not None:
        predictions = dataset.denormalize(predictions)
    
    return predictions

def load_single_csv_for_prediction(csv_path, sequence_length=50, dataset=None):
    """Charge un CSV individuel pour faire des prédictions"""
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    angles = df.values.T  # Transposer pour avoir (time_steps, 2)
    
    print(f"Original data shape: {angles.shape}")
    
    if dataset and dataset.normalize:
        angles = 2 * (angles - dataset.angle_min) / (dataset.angle_max - dataset.angle_min) - 1
    
    # Prendre la dernière séquence de la longueur requise
    if len(angles) >= sequence_length:
        sequence = angles[-sequence_length:]
    else:
        # Padding si la séquence est trop courte
        padding = np.zeros((sequence_length - len(angles), 2))
        sequence = np.vstack([padding, angles])
        print(f"Padded sequence from {len(angles)} to {sequence_length}")
    
    print(f"Final sequence shape: {sequence.shape}")
    return torch.FloatTensor(sequence).unsqueeze(1)  # [seq_len, 1, 2]

# Utilisation
if __name__ == "__main__":
    # Vérifier si tqdm est installé
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess
        subprocess.check_call(["pip", "install", "tqdm"])
        from tqdm import tqdm
    
    # Entraînement du modèle
    print("Starting joint angle prediction training...")
    trained_model, dataset = train_joint_angle_predictor()
    
    if trained_model is not None:
        # Sauvegarder le modèle final
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'normalization_stats': {
                'min': dataset.angle_min,
                'max': dataset.angle_max
            } if dataset.normalize else None,
            'scores': trained_model.scores
        }, 'final_joint_angle_predictor.pth')
        
        print("\nModel saved as 'final_joint_angle_predictor.pth'")
        print("Best model saved as 'best_joint_angle_predictor.pth'")
        
        # Exemple de prédiction sur un nouveau CSV
        test_csv_path = "data/S01/some_trajectory.csv"  # À adapter
        if os.path.exists(test_csv_path):
            print(f"\nMaking prediction on: {test_csv_path}")
            initial_sequence = load_single_csv_for_prediction(test_csv_path, dataset=dataset)
            future_trajectory = predict_trajectory(trained_model, initial_sequence, 30, dataset)
            
            print(f"\nTrajectoire prédite shape: {future_trajectory.shape}")
            print(f"Premières prédictions:\n{future_trajectory[:5]}")