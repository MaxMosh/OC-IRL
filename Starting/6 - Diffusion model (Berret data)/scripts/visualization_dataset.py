import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SUBJECT = "S05"
BASE_DIR = "data"
# Path to the model file containing arm lengths
MODEL_FILE = os.path.join(BASE_DIR, f"{SUBJECT}_model.csv")
# Directory containing the trajectory CSV files
SUBJECT_DIR = os.path.join(BASE_DIR, SUBJECT)

# --- 1. Load Arm Lengths ---
# The file contains 1 column and 2 rows (Upper Arm, Forearm)
try:
    lengths = pd.read_csv(MODEL_FILE, header=None)
    l1 = lengths.iloc[0, 0]  # Upper Arm Length
    l2 = lengths.iloc[1, 0]  # Forearm Length
    print(f"Loaded lengths: L1={l1}, L2={l2}")
except Exception as e:
    print(f"Error loading model file: {e}")
    # Default values in case of error
    l1, l2 = 0.3, 0.25

# --- 2. Load Trajectories ---
trajectories = []
# Select the first 3 CSV files found in the directory
files = sorted([f for f in os.listdir(SUBJECT_DIR) if f.endswith('.csv')])[:3]

for f in files:
    path = os.path.join(SUBJECT_DIR, f)
    # Load CSV following the logic in train5.py: (2 rows, T columns) -> Transpose -> (T rows, 2 columns)
    df = pd.read_csv(path, header=None)
    traj = df.T.values 
    trajectories.append(traj)

# --- 3. Plot 1: Joint Angles (3 trials x 2 angles) ---
fig1, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

for i, traj in enumerate(trajectories):
    # Plot q1 (Shoulder)
    axes[i, 0].plot(traj[:, 0], color='tab:blue')
    axes[i, 0].set_title(f'Trial {i+1}: Shoulder ($q_1$)')
    axes[i, 0].set_ylabel('Angle (rad)')
    axes[i, 0].set_xlabel('Time step')
    axes[i, 0].grid(True)
    
    # Plot q2 (Elbow)
    axes[i, 1].plot(traj[:, 1], color='tab:orange')
    axes[i, 1].set_title(f'Trial {i+1}: Elbow ($q_2$)')
    axes[i, 1].set_ylabel('Angle (rad)')
    axes[i, 1].set_xlabel('Time step')
    axes[i, 1].grid(True)

fig1.suptitle(f'Joint Trajectories - Subject {SUBJECT}')
# Save the figure
fig1.savefig(f'joint_angles_{SUBJECT}.png')
print(f"Saved joint_angles_{SUBJECT}.png")
plt.show()

# --- 4. Plot 2: Cartesian Space (End-Effector) ---
plt.figure(figsize=(8, 8))

for i, traj in enumerate(trajectories):
    q1 = traj[:, 0]
    q2 = traj[:, 1]
    
    # Forward Kinematics
    # Assumption: q1 is absolute angle, q2 is relative to the upper arm
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    
    plt.plot(x, y, label=f'Trial {i+1}', linewidth=2)
    plt.scatter(x[0], y[0], marker='o') # Start point
    plt.scatter(x[-1], y[-1], marker='x') # End point

# Plot vertical dashed line at 0.95 * (L1 + L2)
limit_x = 0.9 * (l1 + l2)
plt.axvline(x=limit_x, color='gray', linestyle='--', alpha=0.7, label=f'Limit (0.95 * Reach)')

plt.axis('equal')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title(f'End-Effector Trajectories (XY Plane) - {SUBJECT}')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(f'end_effector_{SUBJECT}.png')
print(f"Saved end_effector_{SUBJECT}.png")
plt.show()
