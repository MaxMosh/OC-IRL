import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import specific variables and functions from your original script
# Ensure 'steps_two_arms_robot_cpin.py' is in the same directory
try:
    from tools.steps_two_arms_robot_cpin import solve_DOC, L_1, L_2, t_f, N
except ImportError:
    print("Error: Could not import 'steps_two_arms_robot_cpin.py'.")
    print("Please make sure the file exists in the current directory.")
    exit()

def compute_forward_kinematics(q1_traj, q2_traj, l1, l2):
    """
    Computes the (x, y) positions of the elbow and the end-effector 
    based on joint angles and link lengths.
    """
    # Elbow position (end of link 1)
    x_elbow = l1 * np.cos(q1_traj)
    y_elbow = l1 * np.sin(q1_traj)
    
    # End-effector position (end of link 2)
    x_ee = x_elbow + l2 * np.cos(q1_traj + q2_traj)
    y_ee = y_elbow + l2 * np.sin(q1_traj + q2_traj)
    
    return x_elbow, y_elbow, x_ee, y_ee

def run_visualization():
    # --- 1. SETUP SIMULATION PARAMETERS ---
    
    # Define a list of different weight configurations (w)
    weights_scenarios = [
        np.array([0.8, 0.01, 0.19]),
        np.array([0.7, 0.05, 0.25]),
        np.array([0.6, 0.01, 0.39]),
        np.array([0.5, 0.1, 0.4])
    ]
    
    scenario_names = [  
        "$w = [0.8, 0.01, 0.19]^T$",
        "$w = [0.7, 0.05, 0.25]^T$",
        "$w = [0.6, 0.01, 0.39]^T$",
        "$w = [0.5, 0.1, 0.4]^T$"
    ]
    
    # Index of the scenario to visualize in the 2D Workspace plot
    SCENARIO_TO_PLOT_2D = 0 
    
    colors = cm.viridis(np.linspace(0, 0.9, len(weights_scenarios)))
    time_vector = np.linspace(0, t_f, N)

    # --- 2. PREPARE FIGURES ---
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # Figure 1: Joint Profiles (Comparisons)
    fig_joints, ax_joints = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig_joints.suptitle('Joint Trajectories Profiles (Comparison)', fontsize=16, fontweight='bold')

    # Figure 2: 2D Workspace Motion (Single Focused Trajectory)
    fig_work, ax_work = plt.subplots(figsize=(10, 10))
    title_2d = f'Robot Arm Motion in 2D Plane\n(Scenario: {scenario_names[SCENARIO_TO_PLOT_2D]})'
    ax_work.set_title(title_2d, fontsize=16, fontweight='bold')
    
    # --- 3. COMPUTATION AND PLOTTING LOOP ---
    
    print("Solving optimal control problems...")
    
    for i, w in enumerate(weights_scenarios):
        print(f"  > Simulating Scenario {i+1}: {scenario_names[i]}...")
        
        # Call the solver
        q_res, dq_res = solve_DOC(w)
        
        q1 = q_res[:, 0]
        q2 = q_res[:, 1]
        
        # --- PLOT JOINTS (Keep comparison for all) ---
        ax_joints[0].plot(time_vector, q1, linewidth=2.5, color=colors[i], label=f'{scenario_names[i]}')
        ax_joints[1].plot(time_vector, q2, linewidth=2.5, color=colors[i], label=f'{scenario_names[i]}')

        # --- PLOT WORKSPACE (Only for the selected scenario) ---
        if i == SCENARIO_TO_PLOT_2D:
            # Compute Cartesian positions
            x_elb, y_elb, x_ee, y_ee = compute_forward_kinematics(q1, q2, L_1, L_2)
            
            # 1. Plot the main trajectory line
            ax_work.plot(x_ee, y_ee, linewidth=3, color=colors[i], label='End-Effector Trajectory')
            
            # 2. Plot Ghost Trace with Gradient Opacity
            # We skip some frames to avoid drawing too many lines (e.g., every 8th step)
            step_size = 8
            ghost_indices = range(0, N, step_size)
            
            for idx in ghost_indices:
                # Calculate opacity factor based on time (0 to 1)
                # t=0 -> factor=0, t=N -> factor=1
                progress = idx / N 
                
                # Define alpha range: starts at 0.1 (transparent), ends at 0.9 (opaque)
                alpha_val = 0.1 + (0.8 * progress)
                
                # Robot structure: Origin -> Elbow -> End-Effector
                arm_x = [0, x_elb[idx], x_ee[idx]]
                arm_y = [0, y_elb[idx], y_ee[idx]]
                
                # Draw arm structure
                # We use 'gray' or the scenario color. Gray usually looks cleaner for ghosts.
                ax_work.plot(arm_x, arm_y, color='gray', alpha=alpha_val, linewidth=2)
                
                # Draw joints (dots)
                ax_work.scatter(arm_x, arm_y, color='black', s=25, alpha=alpha_val, zorder=3)

    # --- 4. FINALIZE FORMATTING ---

    # Formatting Figure 1 (Joints)
    ax_joints[0].set_ylabel('Angle $q_1$ [rad]', fontsize=12)
    ax_joints[0].set_xlabel('Time [s]', fontsize=12)
    ax_joints[0].grid(True, linestyle=':', alpha=0.6)
    # Legend only on the second subplot to avoid clutter, or both if preferred
    
    ax_joints[1].set_ylabel('Angle $q_2$ [rad]', fontsize=12)
    ax_joints[1].set_xlabel('Time [s]', fontsize=12)
    ax_joints[1].legend(loc='lower right', frameon=True, fontsize=10)
    ax_joints[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()

    # Formatting Figure 2 (Workspace)
    ax_work.set_xlabel('X Position [m]', fontsize=12)
    ax_work.set_ylabel('Y Position [m]', fontsize=12)
    ax_work.legend(loc='upper left', frameon=True)
    ax_work.axis('equal') # Crucial for robot geometry
    ax_work.grid(True, linestyle=':', alpha=0.6)
    
    # Add base marker explicitly
    ax_work.scatter([0], [0], color='black', s=150, marker='^', label='Base', zorder=5)
    
    # Save the figures
    print("\nSaving figures...")
    fig_joints.savefig('robot_joint_profiles.png', dpi=300)
    fig_work.savefig('robot_workspace_motion.png', dpi=300)
    print("Done! Images saved.")

    plt.show()

if __name__ == "__main__":
    run_visualization()