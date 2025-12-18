import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import pinocchio as pin

# suffix_number = 10000
suffix = "simplex_21_lim_joint_velocities_800"
n_components = 10  # Number of PC to analyze

def main():
    # Paths
    base_dir = os.path.dirname(__file__)
    # traj_path = os.path.join(base_dir, f'../data/array_results_angles_{suffix_number}.npy')
    traj_path = os.path.join(base_dir, f'../data/array_results_angles_{suffix}.npy')
    # weights_path = os.path.join(base_dir, f'../data/array_w_{suffix_number}.npy')
    weights_path = os.path.join(base_dir, f'../data/array_w_{suffix}.npy')
    urdf_path = os.path.join(base_dir, '../assets/mon_robot.urdf')
    
    # Load data
    if not os.path.exists(traj_path) or not os.path.exists(weights_path):
        print(f"Error: Files not found. Checked {traj_path} and {weights_path}")
        return

    print(f"Loading trajectories from {traj_path}...")
    trajectories = np.load(traj_path) # (nb_traj, 50, 2)
    print(f"Loading weights from {weights_path}...")
    cost_weights = np.load(weights_path) # (nb_traj, 5)
    
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Cost Weights shape: {cost_weights.shape}")

    # Reshape trajectories for PCA: (n_samples, n_frames * n_features)
    n_samples, n_frames, n_features = trajectories.shape
    data_reshaped = trajectories.reshape(n_samples, -1)

    # Apply PCA
    print(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_reshaped)
    
    # Explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

    # --- Plot 1: Explained Variance ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, align='center', label='Individual explained variance')
    plt.step(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative explained variance')
    
    # Add text labels above bars
    for i, v in enumerate(pca.explained_variance_ratio_):
        plt.text(i + 1, v + 0.01, f'{v*100:.1f}%', ha='center', va='bottom')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title('Scree Plot')
    plt.tight_layout()
    plot1_path = os.path.join(base_dir, f'../plots/pca_scree_{suffix}_{n_components}_components.png')
    os.makedirs(os.path.dirname(plot1_path), exist_ok=True)
    plt.savefig(plot1_path)
    print(f"Saved scree plot to {plot1_path}")

    # --- Plot 2: Eigen-Trajectories (The "weights" of the PCA) ---
    # The components have shape (n_components, n_features_total)
    # We reshape them back to (n_components, n_frames, n_features) to visualize them as trajectories
    eigen_trajectories = pca.components_.reshape(n_components, n_frames, n_features)
    mean_trajectory = pca.mean_.reshape(n_frames, n_features)

    fig2, axes2 = plt.subplots(n_components, n_features, figsize=(10, 3 * n_components), sharex=True)
    if n_components == 1: axes2 = axes2.reshape(1, -1)
    
    for i in range(n_components):
        for j in range(n_features):
            ax = axes2[i, j]
            # Plot mean trajectory
            ax.plot(mean_trajectory[:, j], 'k--', label='Mean', alpha=0.5)
            # Plot mean + component
            ax.plot(mean_trajectory[:, j] + eigen_trajectories[i, :, j], 'r', label=f'Mean + PC{i+1}')
            # Plot mean - component
            ax.plot(mean_trajectory[:, j] - eigen_trajectories[i, :, j], 'b', label=f'Mean - PC{i+1}')
            
            ax.set_title(f'PC {i+1} - Angle {j+1}')
            if i == n_components - 1:
                ax.set_xlabel('Time steps')
            if j == 0:
                ax.set_ylabel('Rad')
            ax.grid(True)
            if i == 0 and j == 0:
                ax.legend()

    fig2.suptitle('Principal Components (Eigen-Trajectories)')
    fig2.tight_layout()
    plot2_path = os.path.join(base_dir, f'../plots/pca_components_{suffix}_{n_components}_components.png')
    fig2.savefig(plot2_path)
    print(f"Saved components plot to {plot2_path}")

    # --- Plot 3: Projection on PC1 vs PC2 colored by Cost Weights ---
    # We want to see if the position in PC space correlates with the cost weights
    n_cost_weights = cost_weights.shape[1]
    fig3, axes3 = plt.subplots(1, n_cost_weights, figsize=(4 * n_cost_weights, 4))
    if n_cost_weights == 1: axes3 = [axes3]

    for i in range(n_cost_weights):
        ax = axes3[i]
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=cost_weights[:, i], cmap='viridis', alpha=0.5, s=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Colored by Weight {i+1}')
        plt.colorbar(sc, ax=ax)
        ax.grid(True)

    fig3.suptitle('Projection on PC1-PC2 colored by Cost Weights')
    fig3.tight_layout()
    plot3_path = os.path.join(base_dir, f'../plots/pca_projection_weights_{suffix}_{n_components}_components.png')
    fig3.savefig(plot3_path)
    print(f"Saved projection plot to {plot3_path}")
    
    # --- Plot 4: Weights (Loadings) Visualization separated by Joint ---
    # Reshape components to separate q1 and q2
    # eigen_trajectories shape is (n_components, n_frames, n_features)
    
    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Heatmap for q1
    im1 = axes4[0].imshow(eigen_trajectories[:, :, 0], aspect='auto', cmap='coolwarm', vmin=-0.2, vmax=0.2)
    axes4[0].set_title('Loadings for Joint 1 (q1)')
    axes4[0].set_ylabel('Principal Component')
    axes4[0].set_yticks(range(n_components))
    axes4[0].set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    plt.colorbar(im1, ax=axes4[0], label='Weight')

    # Heatmap for q2
    im2 = axes4[1].imshow(eigen_trajectories[:, :, 1], aspect='auto', cmap='coolwarm', vmin=-0.2, vmax=0.2)
    axes4[1].set_title('Loadings for Joint 2 (q2)')
    axes4[1].set_xlabel('Time (frames)')
    axes4[1].set_ylabel('Principal Component')
    axes4[1].set_yticks(range(n_components))
    axes4[1].set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    plt.colorbar(im2, ax=axes4[1], label='Weight')

    fig4.suptitle('Heatmap of Principal Component Weights (Loadings) per Joint')
    fig4.tight_layout()
    plot4_path = os.path.join(base_dir, f'../plots/pca_loadings_heatmap_split_{suffix}_{n_components}_components.png')
    fig4.savefig(plot4_path)
    print(f"Saved split loadings heatmap to {plot4_path}")

if __name__ == "__main__":
    main()
