import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import pinocchio as pin

# suffix_number = 10000
suffix = "simplex_21_lim_joint_velocities_800"

nb_clusters = 20
bin_width = 0.1
bins = np.arange(0, 1, bin_width)

def main():
    # Paths
    base_dir = os.path.dirname(__file__)
    traj_path = os.path.join(base_dir, f'../data/array_results_angles_{suffix}.npy')
    weights_path = os.path.join(base_dir, f'../data/array_w_{suffix}.npy')
    urdf_path = os.path.join(base_dir, '../assets/mon_robot.urdf')
    
    # Load data
    if not os.path.exists(traj_path) or not os.path.exists(weights_path):
        print(f"Error: Files not found. Checked {traj_path} and {weights_path}")
        return

    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        return

    print(f"Loading trajectories from {traj_path}...")
    trajectories = np.load(traj_path) # (suffix_number, 50, 2)
    print(f"Loading weights from {weights_path}...")
    weights = np.load(weights_path) # (suffix_number, 5)
    
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Weights shape: {weights.shape}")

    # Load Robot Model
    print(f"Loading robot model from {urdf_path}...")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    ee_frame_id = model.getFrameId("ee_link")

    # Reshape trajectories for clustering: (n_samples, n_frames * n_features)
    n_samples, n_frames, n_features = trajectories.shape
    data_reshaped = trajectories.reshape(n_samples, -1)

    # Apply K-means
    # n_clusters = 5
    n_clusters = nb_clusters
    print(f"Applying K-means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_reshaped)

    # Compute EE trajectories for all samples
    print("Computing End-Effector trajectories...")
    ee_trajectories = np.zeros((n_samples, n_frames, 3)) # (x, y, z)
    
    for i in range(n_samples):
        for t in range(n_frames):
            q = trajectories[i, t, :]
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            ee_trajectories[i, t, :] = data.oMf[ee_frame_id].translation

    # Compute EE trajectories for cluster centers
    centers = kmeans.cluster_centers_.reshape(n_clusters, n_frames, n_features)
    ee_centers = np.zeros((n_clusters, n_frames, 3))
    for k in range(n_clusters):
        for t in range(n_frames):
            q = centers[k, t, :]
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            ee_centers[k, t, :] = data.oMf[ee_frame_id].translation


    # --- Plot 1: All trajectories colored by cluster ---
    print("Generating global cluster plot...")
    fig1, axes1 = plt.subplots(n_features, 1, figsize=(10, 8), sharex=True)
    if n_features == 1: axes1 = [axes1]
    
    colors = plt.colormaps['tab10']
    
    for i in range(n_samples):
        cluster_idx = labels[i]
        color = colors(cluster_idx)
        for j in range(n_features):
            axes1[j].plot(trajectories[i, :, j], color=color, alpha=0.3)
            
    # Plot centers
    for k in range(n_clusters):
        color = colors(k)
        for j in range(n_features):
            axes1[j].plot(centers[k, :, j], color=color, linewidth=2, linestyle='--', label=f'Cluster {k}')

    for j in range(n_features):
        axes1[j].set_ylabel(f'Angle {j+1} (rad)')
        axes1[j].grid(True)
    axes1[-1].set_xlabel('Time (frames)')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors(k), lw=2) for k in range(n_clusters)]
    fig1.legend(custom_lines, [f'Cluster {k}' for k in range(n_clusters)], loc='upper right')
    fig1.suptitle(f'K-means Clustering of Trajectories (n_clusters={n_clusters})')
    fig1.tight_layout()
    
    plot1_path = os.path.join(base_dir, f'../plots/kmeans_clusters_{suffix}_{nb_clusters}_clusters.png')
    os.makedirs(os.path.dirname(plot1_path), exist_ok=True)
    fig1.savefig(plot1_path)
    print(f"Saved global plot to {plot1_path}")

    # --- Plot 2: Detailed view per cluster with weights ---
    print("Generating detailed cluster analysis plot...")
    
    n_weights = weights.shape[1]
    # Grid: n_weights rows (Histograms) + 2 rows (Angle 1, Angle 2) + 1 row (EE Traj), n_clusters columns
    n_rows = n_weights + 3
    fig2, axes2 = plt.subplots(n_rows, n_clusters, figsize=(4 * n_clusters, 5 * n_rows))
    
    # If n_clusters is 1, axes2 is 1D array, make it 2D
    if n_clusters == 1:
        axes2 = axes2.reshape(n_rows, 1)

    for k in range(n_clusters):
        cluster_indices = np.where(labels == k)[0]
        cluster_trajs = trajectories[cluster_indices]
        cluster_weights = weights[cluster_indices]
        cluster_ee = ee_trajectories[cluster_indices]
        
        # 1. Histograms for each weight (Rows 0 to n_weights-1)
        for w_idx in range(n_weights):
            ax_hist = axes2[w_idx, k]
            if len(cluster_weights) > 0:
                ax_hist.hist(cluster_weights[:, w_idx], bins=bins, color=colors(k), alpha=0.7, edgecolor='black')
            if w_idx == 0:
                ax_hist.set_title(f'Cluster {k}\nWeight {w_idx+1} Dist.')
            else:
                ax_hist.set_title(f'Weight {w_idx+1} Dist.')
                
            if k == 0:
                ax_hist.set_ylabel('Count')
            ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
            ax_hist.set_xlim(0, 1)
            # ax_hist.set_ylim(0, int(suffix_number/8))
            ax_hist.set_ylim(0, 1000)

        # 2. Trajectories Angle 1 (Row n_weights)
        ax_traj1 = axes2[n_weights, k]
        for i in range(len(cluster_trajs)):
            ax_traj1.plot((360/(2*np.pi))*cluster_trajs[i, :, 0], color=colors(k), alpha=0.3)
        
        # Plot cluster center Angle 1
        ax_traj1.plot((360/(2*np.pi))*centers[k, :, 0], color='black', linewidth=2, linestyle='--', label='Center')
        
        ax_traj1.set_title(f'Cluster {k} - Angle 1')
        ax_traj1.grid(True)
        if k == 0:
            ax_traj1.set_ylabel('Angle 1 (°)')
        
        # 3. Trajectories Angle 2 (Row n_weights + 1)
        ax_traj2 = axes2[n_weights + 1, k]
        for i in range(len(cluster_trajs)):
            ax_traj2.plot((360/(2*np.pi))*cluster_trajs[i, :, 1], color=colors(k), alpha=0.3)
            
        # Plot cluster center Angle 2
        ax_traj2.plot((360/(2*np.pi))*centers[k, :, 1], color='black', linewidth=2, linestyle='--', label='Center')
        
        ax_traj2.set_title(f'Cluster {k} - Angle 2')
        ax_traj2.grid(True)
        if k == 0:
            ax_traj2.set_ylabel('Angle 2 (°)')
        ax_traj2.set_xlabel('Time')

        # 4. End-Effector Trajectories (Row n_weights + 2)
        ax_ee = axes2[n_weights + 2, k]
        for i in range(len(cluster_ee)):
            # Assuming Planar robot in X-Z plane (based on typical 2D arm setups or check data)
            # If it's X-Y, change indices. Usually 2D arms are X-Y or X-Z.
            # Let's check OCP_solving_cpin.py... it uses L_1*cos(q1) + L_2*cos(q1+q2) = x_fin.
            # This looks like X coordinate.
            # Let's plot X vs Z (indices 0 and 2) or X vs Y (0 and 1).
            # Pinocchio usually aligns Z up.
            # Let's try X (0) and Z (2) as it's common for vertical planar arms.
            # Or X (0) and Y (1) if it's horizontal.
            # Given "gravity" context usually implies vertical -> X-Z.
            # But let's look at the data ranges if possible.
            # For now, I will plot X (0) vs Y (1) as default, but if it looks flat I'll switch.
            # Actually, standard planar URDFs often use X-Y or X-Z.
            # Let's plot X vs Y (0 vs 1) first.
            ax_ee.plot(cluster_ee[i, :, 0], cluster_ee[i, :, 1], color=colors(k), alpha=0.3)
        
        # Plot cluster center EE
        ax_ee.plot(ee_centers[k, :, 0], ee_centers[k, :, 1], color='black', linewidth=2, linestyle='--', label='Center')
        
        ax_ee.set_title(f'Cluster {k} - EE Traj (X-Y)')
        ax_ee.grid(True)
        if k == 0:
            ax_ee.set_ylabel('Y (m)')
        ax_ee.set_xlabel('X (m)')
        ax_ee.axis('equal') # Important for spatial trajectories
        ax_ee.set_xlim(-3, 3)
        ax_ee.set_ylim(-3, 3)

    fig2.suptitle('Cluster Analysis: Weight Histograms and Trajectories')
    fig2.tight_layout()
    
    plot2_path = os.path.join(base_dir, f'../plots/kmeans_clusters_detailed_{suffix}_{nb_clusters}_clusters.png')
    fig2.savefig(plot2_path)
    print(f"Saved detailed plot to {plot2_path}")

if __name__ == "__main__":
    main()
