import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Create animation showing MPC predictions
class MPCAnimator:
    def __init__(self, model, time_vector, x_trajectory, pred_trajectory, x_ref_trajectory, u_trajectory, mode_sequences, dwell_times_history):
        """Initialize the MPC animation class with simulation data
        
        Args:
            model: System model with 'A' and 'B' matrices for each mode
            time_vector: Time points of the simulation
            x_trajectory: State trajectory history (numpy array)
            pred_trajectory: Predicted state trajectories from the MPC (list of numpy arrays)
            x_ref_trajectory: Reference state trajectory history (numpy array)
            u_trajectory: Control input history
            mode_sequences: Mode sequences for each time step
            dwell_times_history: Dwell times history for each time step
        """
        self.model = model
        self.time_vector = time_vector
        self.x_trajectory = x_trajectory
        self.pred_trajectory = pred_trajectory
        self.x_ref_trajectory = x_ref_trajectory
        self.u_trajectory = u_trajectory
        self.mode_sequences = mode_sequences
        self.dwell_times_history = dwell_times_history
        
        # Import matplotlib only when needed
        self.plt = plt
        self.FuncAnimation = FuncAnimation
        
        # Create figure for animation
        self.fig_anim, self.ax_anim = plt.subplots(figsize=(10, 8))
        
    def init_plot(self):
        """Initialize the animation plot."""
        self.ax_anim.clear()
        self.ax_anim.grid(True)
        self.ax_anim.set_xlim(-5, 5)  
        self.ax_anim.set_ylim(-3, 3)
        self.ax_anim.set_xlabel('x1')
        self.ax_anim.set_ylabel('x2')
        self.ax_anim.set_title('MPC Prediction Animation')
        self.ax_anim.set_aspect('equal', 'box')
        return []
    
    def update_plot(self, frame):
        """Update function for animation at each frame."""
        self.ax_anim.clear()
        
        # Plot past trajectory
        if frame > 0:
            past_x = self.x_trajectory[:frame]
            self.ax_anim.plot(past_x[:, 0], past_x[:, 1], 'b-', 
                    linewidth=2, label='Actual Trajectory')
        
        # Get current state and mode sequence
        x_current_frame = self.x_trajectory[frame]
        current_mode_seq = self.mode_sequences[frame]
        current_dwell_times = self.dwell_times_history[frame]
        
        # Plot current state
        self.ax_anim.plot(x_current_frame[0], x_current_frame[1], 'bo', 
                markersize=10, label='Current State')
        
        # Get the predicted trajectory for current frame
        x_pred = self.pred_trajectory[frame]
        
        x_pred = np.array(x_pred)
        if x_pred.ndim == 1:
            # Reshape from [x1, x2, x1, x2, ...] to [[x1, x2], [x1, x2], ...]
            x_pred = x_pred.reshape(-1, 2)
        
        # Plot prediction
        self.ax_anim.plot(x_pred[:, 0], x_pred[:, 1], 'r--', linewidth=2, 
                marker='o', markersize=4, alpha=0.7, label='MPC Prediction')
        
        # Mark origin/target
        self.ax_anim.plot(self.x_ref_trajectory[frame, 0], self.x_ref_trajectory[frame, 1], 'k*', markersize=15, label='Target')
        
        # Set labels and legend
        self.ax_anim.set_xlabel('x1')
        self.ax_anim.set_ylabel('x2')
        self.ax_anim.set_title(f'MPC Prediction (Step {frame}, Mode {current_mode_seq[0]})')
        self.ax_anim.legend(loc='upper right')
        self.ax_anim.grid(True)
        
        # Set reasonable axis limits based on data range
        padding = 0.5
        max_x = max(np.max(self.x_trajectory[:, 0]), np.max(x_pred[:, 0])) + padding
        min_x = min(np.min(self.x_trajectory[:, 0]), np.min(x_pred[:, 0])) - padding
        max_y = max(np.max(self.x_trajectory[:, 1]), np.max(x_pred[:, 1])) + padding
        min_y = min(np.min(self.x_trajectory[:, 1]), np.min(x_pred[:, 1])) - padding
        
        self.ax_anim.set_xlim(min_x, max_x)
        self.ax_anim.set_ylim(min_y, max_y)
        
        # Add current time and active mode info
        self.ax_anim.text(0.02, 0.98, f'Time: {self.time_vector[frame]:.2f}s', 
                transform=self.ax_anim.transAxes, fontsize=10, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.ax_anim.text(0.02, 0.92, f'Mode: {current_mode_seq[0]}', 
                transform=self.ax_anim.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        self.ax_anim.text(0.02, 0.86, f'Control: {self.u_trajectory[frame]:.3f}', 
                transform=self.ax_anim.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        return []
    
    def create_animation(self, interval=200, save_path=None):
        """Create and optionally save the animation
        
        Args:
            interval: Time between frames in milliseconds
            save_path: If provided, path to save the animation file
            
        Returns:
            The animation object
        """
        ani = self.FuncAnimation(
            self.fig_anim, 
            self.update_plot, 
            frames=len(self.time_vector),
            init_func=self.init_plot, 
            blit=True, 
            interval=interval
        )
        
        # Save animation if requested
        if save_path:
            try:
                ani.save(save_path, writer='ffmpeg', fps=5)
                print(f"Animation saved to '{save_path}'")
            except Exception as e:
                print(f"Couldn't save animation: {e}")
            
        return ani
