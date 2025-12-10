import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import sys
import os

sys.path.append(os.getcwd())
from tools.dan_model import DAN_WeightEstimator
from tools.OCP_solving_cpin import solve_DOC

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_DIM = 128
SEQ_LEN = 50

# --- 1. Load Resources ---
print(f"Loading DAN model on {DEVICE}...")

try:
    with open('scaler_dan_w.pkl', 'rb') as f:
        scaler_w = pickle.load(f)
except FileNotFoundError:
    print("Error: Scaler not found. Run train_dan.py first.")
    exit()

model = DAN_WeightEstimator(input_dim=2, hidden_dim=HIDDEN_DIM, w_dim=3).to(DEVICE)

try:
    # Load the trained weights
    model.load_state_dict(torch.load("trained_models_dan/dan_model_final.pth", map_location=DEVICE))
    model.eval()
except FileNotFoundError:
    print("Error: Model weights not found.")
    exit()

# --- 2. Generate Test Data ---
print("Generating new test trajectory...")

def generate_random_test_case():
    # Reuse your logic for generating test cases
    while True:
        try:
            w_base = 0.01
            log_w_max = np.random.uniform(np.log(1), np.log(20))
            w_max = np.exp(log_w_max)
            t_transition = np.random.randint(25, 45)
            k_intensity = np.random.uniform(1, 5)
            
            t = np.linspace(0, 49, 50)
            sigmoid = 1 / (1 + np.exp(-k_intensity * (t - t_transition)))
            w_3 = w_base + (w_max - w_base) * sigmoid
            
            w_1 = np.full(50, 0.01)
            w_2 = np.full(50, 0.01)
            w_true = np.column_stack((w_1, w_2, w_3)) 
            
            q_res, dq_res = solve_DOC(w_true, x_fin=-1.0, q_init=[0, np.pi/4])
            return q_res, w_true
        except:
            continue

q_test, w_test_true = generate_random_test_case()

# --- 3. Inference ---
# Prepare input: (1, 50, 2)
traj_tensor = torch.FloatTensor(q_test).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    # The DAN processes the whole sequence at once, respecting causality internally
    w_pred_norm = model(traj_tensor) # Output: (1, 50, 3)

# Inverse Transform
w_pred_norm_np = w_pred_norm.cpu().numpy().reshape(-1, 3)
w_pred = scaler_w.inverse_transform(w_pred_norm_np) # (50, 3)

# --- 4. Animation ---
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

# Plot 1: Trajectory
ax_traj = fig.add_subplot(gs[0])
ax_traj.set_title("Robot Trajectory Observation")
ax_traj.set_xlim(0, 50)
ax_traj.set_ylim(-1.0, 3.5)
ax_traj.grid(True, alpha=0.3)
ax_traj.plot(q_test[:, 0], 'b--', alpha=0.3, label="q1 (True)")
ax_traj.plot(q_test[:, 1], 'g--', alpha=0.3, label="q2 (True)")

line_q1, = ax_traj.plot([], [], 'b-', lw=2, label="q1 (Seen)")
line_q2, = ax_traj.plot([], [], 'g-', lw=2, label="q2 (Seen)")
ax_traj.legend()

# Plot 2: Weight Estimation
ax_weight = fig.add_subplot(gs[1])
ax_weight.set_title("Recursive Weight Estimation (DAN)")
ax_weight.set_xlim(0, 50)
ymax = np.max(w_test_true[:, 2]) * 1.5
ax_weight.set_ylim(-0.5, max(10, ymax))
ax_weight.grid(True, alpha=0.3)

# Ground Truth
ax_weight.plot(w_test_true[:, 2], 'k--', lw=2, label="w3 (True)", zorder=10)

# Prediction line
line_pred, = ax_weight.plot([], [], 'r-', lw=2, label="w3 (Estimated)")
ax_weight.legend()

def update(frame):
    # Update Trajectory lines (what the robot has done so far)
    line_q1.set_data(np.arange(frame), q_test[:frame, 0])
    line_q2.set_data(np.arange(frame), q_test[:frame, 1])
    
    # Update Weight Prediction lines
    # The DAN has predicted the whole sequence, but at time 'frame',
    # we only show what it has estimated up to that point.
    line_pred.set_data(np.arange(frame), w_pred[:frame, 2])
    
    return line_q1, line_q2, line_pred

print("Generating animation...")
ani = animation.FuncAnimation(fig, update, frames=np.arange(1, SEQ_LEN+1), interval=100, blit=True)
ani.save("dan_inference.gif", writer='pillow', fps=10)
print("Saved dan_inference.gif")
plt.show()