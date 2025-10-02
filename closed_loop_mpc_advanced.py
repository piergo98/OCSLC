#!/usr/bin/env python3
"""
Advanced Closed-Loop MPC Example with Optimizable Dwell Times

This example demonstrates advanced closed-loop MPC features including:
- Optimizable dwell times with proper aging and horizon management
- Mode switching when phases run out
- Warm starting for faster convergence
- State disturbances and noise
- Reference tracking (not just regulation)
- Performance monitoring
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
from ocslc.switched_linear_mpc import SwitchedLinearMPC

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

class RecedingHorizonMPC:
    """
    A clean implementation of receding horizon MPC with optimizable dwell times.
    
    Handles:
    (i) Aging the first phase by Δt
    (ii) Keeping total horizon length T constant
    (iii) Triggering mode switches when phases run out
    """
    
    def __init__(self, model, n_phases, prediction_horizon, dt_sampling, 
                 Q, R, E=None, bounds=None, multiple_shooting=True, hybrid=False):
        """
        Initialize the receding horizon MPC controller.
        
        Args:
            model: Dictionary with 'A' and 'B' matrices for each mode
            n_phases: Number of phases in the prediction horizon
            prediction_horizon: Total prediction horizon time T
            dt_sampling: Sampling time Δt
            Q, R, E: Cost matrices
            bounds: Dictionary with 'states_lb', 'states_ub', 'inputs_lb', 'inputs_ub'
            multiple_shooting: Whether to use multiple shooting
        """
        self.model = model
        self.n_phases = n_phases
        self.prediction_horizon = prediction_horizon
        self.dt_sampling = dt_sampling
        self.Q = Q
        self.R = R
        self.E = E if E is not None else 100.0 * Q
        self.bounds = bounds
        self.multiple_shooting = multiple_shooting
        self.hybrid = hybrid
        
        # System dimensions
        self.n_states = model['A'][0].shape[0]
        self.n_inputs = model['B'][0].shape[1]
        self.n_modes = len(model['A'])
        
        # Current mode sequence and dwell times
        self.current_mode_sequence = list(range(min(n_phases, self.n_modes))) * (n_phases // self.n_modes + 1)
        self.current_mode_sequence = self.current_mode_sequence[:n_phases]
        self.current_dwell_times = np.full(n_phases, prediction_horizon / n_phases)
        
        # Optimization variables from previous solve (for warm starting)
        self.prev_inputs = None
        self.prev_deltas = None
        self.prev_states = None
        
        # Time tracking
        self.time_remaining_in_current_phase = self.current_dwell_times[0]
        
        print(f"Receding Horizon MPC initialized:")
        print(f"  - {self.n_states} states, {self.n_inputs} inputs, {self.n_modes} modes")
        print(f"  - {n_phases} phases over {prediction_horizon}s horizon")
        print(f"  - Sampling time: {dt_sampling}s")
        print(f"  - Initial mode sequence: {self.current_mode_sequence}")
    
    def update_dwell_times_and_modes(self):
        """
        Core method that implements the three key operations:
        (i) Age the first phase by Δt
        (ii) Keep total horizon T constant 
        (iii) Trigger mode switch if first phase runs out
        """
        print(f"\n--- Updating Dwell Times ---")
        print(f"Before: remaining_time={self.time_remaining_in_current_phase:.4f}, deltas={self.current_dwell_times}")
        
        # (i) Age the first phase by Δt
        self.time_remaining_in_current_phase -= self.dt_sampling
        
        # (iii) Check if we need to switch modes (first phase runs out)
        mode_switched = False
        if self.time_remaining_in_current_phase <= 1e-6:  # Phase exhausted
            print(f"Phase exhausted! Switching modes...")
            
            # Remove the exhausted phase
            self.current_mode_sequence.pop(0)
            self.current_dwell_times = self.current_dwell_times[1:]
            
            # Add a new phase at the end (cycle through modes)
            if len(self.current_mode_sequence) > 0:
                # Continue the pattern or use the last mode
                next_mode = (self.current_mode_sequence[-1] + 1) % self.n_modes
            else:
                next_mode = 0
            
            self.current_mode_sequence.append(next_mode)
            
            # (ii) Keep total horizon T constant - add time to the last phase
            overtime = -self.time_remaining_in_current_phase  # This is positive
            new_phase_duration = self.prediction_horizon / self.n_phases + overtime
            self.current_dwell_times = np.append(self.current_dwell_times, new_phase_duration)
            
            # Reset time remaining for new first phase
            self.time_remaining_in_current_phase = self.current_dwell_times[0]
            mode_switched = True
            
        else:
            # (ii) Keep total horizon T constant - subtract time from first phase, add to last
            self.current_dwell_times[0] = self.time_remaining_in_current_phase
            self.current_dwell_times[-1] += self.dt_sampling
        
        print(f"After: remaining_time={self.time_remaining_in_current_phase:.4f}, deltas={self.current_dwell_times}")
        print(f"Mode sequence: {self.current_mode_sequence}")
        print(f"Total horizon: {np.sum(self.current_dwell_times):.6f} (should be {self.prediction_horizon})")
        
        return mode_switched
    
    def create_switched_model(self):
        """
        Create the switched model based on current mode sequence.
        """
        A_seq = [self.model['A'][mode] for mode in self.current_mode_sequence]
        B_seq = [self.model['B'][mode] for mode in self.current_mode_sequence]
        
        return {'A': A_seq, 'B': B_seq}
    
    def setup_mpc(self, x0):
        """
        Setup the MPC controller with the initial switched model.
        
        Args:
            x0: Initial state
        """
        switched_model = self.create_switched_model()
        
        self.mpc = SwitchedLinearMPC(
            model=switched_model,
            n_phases=self.n_phases,
            time_horizon=self.prediction_horizon,
            auto=False,
            multiple_shooting=self.multiple_shooting,
            hybrid=self.hybrid,
            propagation='exp'
        )
        
        # Precompute matrices
        self.mpc.precompute_matrices(x0, self.Q, self.R, self.E)
        
        # Set bounds
        if self.bounds:
            self.mpc.set_bounds(
                self.bounds['inputs_lb'], self.bounds['inputs_ub'],
                self.bounds.get('states_lb'), self.bounds.get('states_ub')
            )
        
        # Set up constraints
        if self.multiple_shooting:
            self.mpc.multiple_shooting_constraints(x0)
            
    def solve_mpc(self, x_current, x_ref):
        """
        Solve the MPC optimization problem for the current state.
        
        Args:
            x_current: Current state
            
        Returns:
            u_applied: Control input to apply
            solve_info: Dictionary with solve information
        """
        
        # Set up constraints
        if self.multiple_shooting:
            self.mpc.multiple_shooting_constraints(x_current, update=True)
        
        # Set cost function
        self.mpc.set_cost_function(self.Q, self.R, x_current, self.E, reference=x_ref)
        
        # Warm start with adapted previous solution
        if self.prev_inputs is not None and self.prev_deltas is not None:
            try:
                # Use current dwell times as initial guess
                self.mpc.set_initial_guess(
                    x_current,
                    initial_control_inputs=self.prev_inputs,
                    initial_phases_duration=self.current_dwell_times
                )
            except Exception as e:
                print(f"Warm start failed: {e}, using default initial guess")
                self.mpc.set_initial_guess(self.prediction_horizon, x_current)
        else:
            # Use current dwell times even for first solve
            self.mpc.set_initial_guess(
                x_current,
                initial_phases_duration=self.current_dwell_times
            )
        
        # Create solver
        self.mpc.create_solver(
            solver='ipopt',
            print_level=0,
            verbose=False,
            tol=1e-6,
            max_iter=100
        )
        
        # Solve
        solve_start = time.time()
        inputs_opt, deltas_opt, states_opt = self.mpc.solve()
        solve_time = time.time() - solve_start
        
        # Store solution for next warm start
        self.prev_inputs = np.array(inputs_opt)
        self.prev_deltas = deltas_opt.copy()
        if self.multiple_shooting:
            self.prev_states = np.array(states_opt)
        
        # Update dwell times with optimized values
        self.current_dwell_times = deltas_opt.copy()
        self.time_remaining_in_current_phase = self.current_dwell_times[0]
        
        # Extract first control input
        u_applied = inputs_opt[0] if isinstance(inputs_opt[0], (int, float)) else inputs_opt[0]
        
        solve_info = {
            'solve_time': solve_time,
            'cost': self.mpc.opt_cost[0],
            'inputs_opt': inputs_opt,
            'deltas_opt': deltas_opt,
            'states_opt': states_opt,
            'mode_sequence': self.current_mode_sequence.copy(),
            'dwell_times': self.current_dwell_times.copy()
        }
        
        return u_applied, solve_info

def demonstrate_n_modes_generalization():
    """
    Demonstrate that the implementation works with arbitrary number of modes.
    This example uses 5 different modes to show full generalizability.
    """
    print("=== Demonstrating n-Modes Generalization (5 modes) ===")
    
    # Define a 5-mode switched system 
    model = {
        'A': [
            # Mode 0: Stable oscillator
            np.array([[0.95, 0.15], [-0.15, 0.95]]),
            # Mode 1: Unstable growth
            np.array([[1.05, 0.1], [0.0, 1.02]]),
            # Mode 2: Damped system
            np.array([[0.8, 0.2], [-0.1, 0.9]]),
            # Mode 3: Pure integrator chain
            np.array([[1.0, 0.1], [0.0, 1.0]]),
            # Mode 4: Decoupled stable
            np.array([[0.92, 0.0], [0.0, 0.88]])
        ],
        'B': [
            # Mode 0
            np.array([[0.3], [0.8]]),
            # Mode 1  
            np.array([[1.0], [0.2]]),
            # Mode 2
            np.array([[0.5], [1.2]]),
            # Mode 3
            np.array([[0.1], [0.9]]),
            # Mode 4
            np.array([[0.7], [0.4]])
        ]
    }
    
    # MPC parameters
    n_phases = 8  # More phases to see mode cycling
    prediction_horizon = 2.5
    dt_sampling = 0.12
    n_sim_steps = 30
    
    # Cost weights
    Q = np.diag([15.0, 2.0])
    R = np.array([[1.5]])
    E = np.diag([150.0, 15.0])
    
    # Bounds
    bounds = {
        'states_lb': np.array([-6.0, -4.0]),
        'states_ub': np.array([6.0, 4.0]),
        'inputs_lb': -2.5,
        'inputs_ub': 2.5
    }
    
    # Initial state
    x_current = np.array([4.0, 2.5])
    
    # Create receding horizon MPC controller
    mpc_controller = RecedingHorizonMPC(
        model=model,
        n_phases=n_phases,
        prediction_horizon=prediction_horizon,
        dt_sampling=dt_sampling,
        Q=Q, R=R, E=E,
        bounds=bounds,
        multiple_shooting=True
    )
    
    print(f"\nSystem Configuration:")
    print(f"- Number of modes: {mpc_controller.n_modes}")
    print(f"- Initial mode sequence: {mpc_controller.current_mode_sequence}")
    print(f"- Phases per complete cycle: {mpc_controller.n_modes}")
    print(f"- Initial state: {x_current}")
    
    # Verify mode sequence covers all modes
    unique_modes_in_sequence = set(mpc_controller.current_mode_sequence)
    print(f"- Unique modes in sequence: {sorted(unique_modes_in_sequence)}")
    print(f"- All modes represented: {len(unique_modes_in_sequence) == mpc_controller.n_modes}")
    
    # Storage for results
    time_vector = []
    x_trajectory = []
    u_trajectory = []
    mode_sequences = []
    mode_switches = []
    active_modes = []
    
    # Simulation loop
    for step in range(n_sim_steps):
        current_time = step * dt_sampling
        
        if step % 5 == 0:  # Print every 5 steps to reduce output
            print(f"\nStep {step + 1}: t={current_time:.3f}s, state=[{x_current[0]:.3f}, {x_current[1]:.3f}], active_mode={mpc_controller.current_mode_sequence[0]}")
        
        # Solve MPC
        u_applied, solve_info = mpc_controller.solve_mpc(x_current)
        
        # Store results
        time_vector.append(current_time)
        x_trajectory.append(x_current.copy())
        u_trajectory.append(u_applied)
        mode_sequences.append(solve_info['mode_sequence'].copy())
        active_modes.append(mpc_controller.current_mode_sequence[0])
        
        # Apply control (use current active mode)
        current_mode = mpc_controller.current_mode_sequence[0]
        A_current = model['A'][current_mode]
        B_current = model['B'][current_mode]
        
        # System evolution
        x_next = A_current @ x_current + B_current.flatten() * u_applied * dt_sampling
        
        # Small process noise
        x_next += np.random.normal(0, 0.005, x_current.shape)
        
        # Update dwell times and check for mode switches
        mode_switched = mpc_controller.update_dwell_times_and_modes()
        mode_switches.append(mode_switched)
        
        x_current = x_next
        
        # Early termination if converged
        if np.linalg.norm(x_current) < 0.05:
            print(f"Converged at step {step + 1}")
            break
    
    # Analysis
    time_vector = np.array(time_vector)
    x_trajectory = np.array(x_trajectory)
    u_trajectory = np.array(u_trajectory)
    active_modes = np.array(active_modes)
    
    # Mode usage statistics
    unique_active_modes = np.unique(active_modes)
    mode_usage = {mode: np.sum(active_modes == mode) for mode in unique_active_modes}
    total_switches = sum(mode_switches)
    
    print(f"\n=== n-Modes Analysis ===")
    print(f"Total modes available: {mpc_controller.n_modes}")
    print(f"Modes actually used: {sorted(unique_active_modes)}")
    print(f"Mode usage distribution: {mode_usage}")
    print(f"Total mode switches: {total_switches}")
    print(f"Average time per mode: {len(time_vector) / len(unique_active_modes):.2f} steps")
    print(f"Mode cycling demonstrated: {len(unique_active_modes) == mpc_controller.n_modes}")
    
    # Visualize mode transitions
    mode_transitions = {}
    for i in range(len(active_modes) - 1):
        if mode_switches[i + 1]:  # If there was a switch
            from_mode = active_modes[i]
            to_mode = active_modes[i + 1]
            transition = f"{from_mode}→{to_mode}"
            mode_transitions[transition] = mode_transitions.get(transition, 0) + 1
    
    print(f"Mode transitions observed: {mode_transitions}")
    
    # Plot results specific to n-modes demonstration
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # State trajectory
        axes[0, 0].plot(time_vector, x_trajectory[:, 0], 'b-', linewidth=2, label='x1')
        axes[0, 0].plot(time_vector, x_trajectory[:, 1], 'r-', linewidth=2, label='x2')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('States')
        axes[0, 0].set_title(f'State Evolution with {mpc_controller.n_modes} Modes')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Mode sequence over time
        axes[0, 1].step(time_vector, active_modes, 'ko-', where='post', linewidth=2, markersize=3)
        # Color different modes
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
        for mode in unique_active_modes:
            mode_times = time_vector[active_modes == mode]
            mode_values = active_modes[active_modes == mode]
            if len(mode_times) > 0:
                axes[0, 1].scatter(mode_times, mode_values, 
                                 color=colors[mode % len(colors)], 
                                 label=f'Mode {mode}', s=20, alpha=0.7)
        
        axes[0, 1].set_xlabel(r'Time [s]')
        axes[0, 1].set_ylabel(r'Active Mode')
        axes[0, 1].set_title(r'Mode Switching Pattern')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim(-0.5, mpc_controller.n_modes - 0.5)
        
        # Mode usage histogram
        modes_list = list(mode_usage.keys())
        usage_list = list(mode_usage.values())
        bars = axes[1, 0].bar(modes_list, usage_list, alpha=0.7)
        # Color bars differently
        for i, bar in enumerate(bars):
            bar.set_color(colors[i % len(colors)])
        axes[1, 0].set_xlabel('Mode')
        axes[1, 0].set_ylabel('Usage Count (steps)')
        axes[1, 0].set_title('Mode Usage Distribution')
        axes[1, 0].grid(True, axis='y')
        
        # Phase portrait with mode coloring
        for mode in unique_active_modes:
            mask = active_modes == mode
            if np.any(mask):
                axes[1, 1].scatter(x_trajectory[mask, 0], x_trajectory[mask, 1], 
                                 color=colors[mode % len(colors)], 
                                 label=f'Mode {mode}', s=15, alpha=0.7)
        
        axes[1, 1].plot(x_trajectory[0, 0], x_trajectory[0, 1], 'go', markersize=10, label='Start')
        axes[1, 1].plot(x_trajectory[-1, 0], x_trajectory[-1, 1], 'ro', markersize=10, label='End')
        axes[1, 1].plot(0, 0, 'k*', markersize=12, label='Target')
        axes[1, 1].set_xlabel('x1')
        axes[1, 1].set_ylabel('x2')
        axes[1, 1].set_title('Phase Portrait (colored by mode)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return {
        'n_modes': mpc_controller.n_modes,
        'mode_usage': mode_usage,
        'mode_transitions': mode_transitions,
        'total_switches': total_switches,
        'modes_used': sorted(unique_active_modes),
        'time': time_vector,
        'states': x_trajectory,
        'controls': u_trajectory,
        'active_modes': active_modes
    }

def closed_loop_mpc_with_dwell_times():
    """
    Closed-loop MPC example with proper dwell time management.
    """
    print("=== Closed-Loop MPC with Optimizable Dwell Times ===")
    
    # Define a switched system model (2 modes)
    model = {
        'A': [
            np.array([[0.0, 1.0], [0.0, 0.0]]),  # Mode 0: stable
            # np.array([[0.0, 0.0], [0.1, 0.9]])   # Mode 1: slightly unstable
        ],
        'B': [
            np.array([[0.0], [1.0]]),  # Mode 0
            # np.array([[1.0], [0.5]])   # Mode 1  
        ]
    }
    
    # MPC parameters
    n_phases = 6
    prediction_horizon = 2.0
    dt_sampling = 0.1  # Sampling time
    n_sim_steps = 50
    
    # Cost weights
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    E = np.diag([100.0, 10.0])
    
    # Bounds
    bounds = {
        'states_lb': np.array([-8.0, -5.0]),
        'states_ub': np.array([8.0, 5.0]),
        'inputs_lb': -3.0,
        'inputs_ub': 3.0
    }
    
    # Initial state
    x_current = np.array([3.0, 2.0])
    
    # Create receding horizon MPC controller
    mpc_controller = RecedingHorizonMPC(
        model=model,
        n_phases=n_phases,
        prediction_horizon=prediction_horizon,
        dt_sampling=dt_sampling,
        Q=Q, R=R, E=E,
        bounds=bounds,
        multiple_shooting=True
    )

    
    # Storage for results
    time_vector = []
    x_trajectory = []
    u_trajectory = []
    costs = []
    solve_times = []
    mode_sequences = []
    dwell_times_history = []
    mode_switches = []
    
    print(f"\nStarting closed-loop simulation...")
    print(f"Initial state: {x_current}")
    # Initial setup
    mpc_controller.setup_mpc(x_current)
    # Simulation loop
    for step in range(n_sim_steps):
        current_time = step * dt_sampling
        
        print(f"\n=== Step {step + 1}/{n_sim_steps} (t={current_time:.3f}s) ===")
        print(f"Current state: [{x_current[0]:.4f}, {x_current[1]:.4f}]")
        print(f"Active mode: {mpc_controller.current_mode_sequence[0]}")
        print(f"Time left in phase: {mpc_controller.time_remaining_in_current_phase:.4f}s")
        
        # Solve MPC
        u_applied, solve_info = mpc_controller.solve_mpc(x_current)
        
        print(f"Applied control: u = {u_applied:.6f}")
        print(f"Solve time: {solve_info['solve_time']:.4f}s")
        print(f"Cost: {solve_info['cost']:.6f}")
        
        # Store results before updating
        time_vector.append(current_time)
        x_trajectory.append(x_current.copy())
        u_trajectory.append(u_applied)
        costs.append(solve_info['cost'])
        solve_times.append(solve_info['solve_time'])
        mode_sequences.append(solve_info['mode_sequence'].copy())
        dwell_times_history.append(solve_info['dwell_times'].copy())
        
        # Apply control to system (use current active mode)
        current_mode = mpc_controller.current_mode_sequence[0]
        A_current = model['A'][current_mode]
        B_current = model['B'][current_mode]
        
        # Simple forward Euler integration
        x_next = A_current @ x_current + B_current.flatten() * u_applied * dt_sampling
        
        # Add some process noise
        process_noise = np.random.normal(0, 0.01, x_current.shape)
        x_next += process_noise
        
        # Update dwell times and check for mode switches
        mode_switched = mpc_controller.update_dwell_times_and_modes()
        mode_switches.append(mode_switched)
        
        if mode_switched:
            print(f">>> MODE SWITCH occurred! New active mode: {mpc_controller.current_mode_sequence[0]}")
        
        # Update state
        x_current = x_next
        
        # Check convergence
        if np.linalg.norm(x_current) < 0.1:
            print(f"Converged to origin at step {step + 1}")
            break
    
    # Convert to numpy arrays
    time_vector = np.array(time_vector)
    x_trajectory = np.array(x_trajectory)
    u_trajectory = np.array(u_trajectory)
    
    # Analysis
    total_mode_switches = sum(mode_switches)
    print(f"\n=== Simulation Summary ===")
    print(f"Total simulation time: {time_vector[-1]:.2f}s")
    print(f"Final state: [{x_trajectory[-1, 0]:.6f}, {x_trajectory[-1, 1]:.6f}]")
    print(f"Final error: {np.linalg.norm(x_trajectory[-1]):.6f}")
    print(f"Total mode switches: {total_mode_switches}")
    print(f"Average solve time: {np.mean(solve_times):.4f}s")
    print(f"Max solve time: {np.max(solve_times):.4f}s")
    print(f"Total control effort: {np.sum(np.abs(u_trajectory)):.6f}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # State trajectory
        axes[0, 0].plot(time_vector, x_trajectory[:, 0], 'b-', linewidth=2, label='x1')
        axes[0, 0].plot(time_vector, x_trajectory[:, 1], 'r-', linewidth=2, label='x2')
        axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('States')
        axes[0, 0].set_title('State Trajectory')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Control input with mode switches
        axes[0, 1].step(time_vector, u_trajectory, 'g-', where='post', linewidth=2)
        # Mark mode switches
        switch_times = [time_vector[i] for i, switched in enumerate(mode_switches) if switched]
        for t_switch in switch_times:
            axes[0, 1].axvline(x=t_switch, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Control Input')
        axes[0, 1].set_title('Control Input (red lines = mode switches)')
        axes[0, 1].grid(True)
        
        # Active modes over time
        active_modes = [seq[0] for seq in mode_sequences]
        axes[0, 2].step(time_vector, active_modes, 'mo-', where='post', linewidth=2, markersize=4)
        axes[0, 2].set_xlabel('Time [s]')
        axes[0, 2].set_ylabel('Active Mode')
        axes[0, 2].set_title('Active Mode Evolution')
        axes[0, 2].set_ylim(-0.5, max(active_modes) + 0.5)
        axes[0, 2].grid(True)
        
        # Phase portrait
        axes[1, 0].plot(x_trajectory[:, 0], x_trajectory[:, 1], 'b-', linewidth=2, marker='o', markersize=3)
        axes[1, 0].plot(x_trajectory[0, 0], x_trajectory[0, 1], 'go', markersize=10, label='Start')
        axes[1, 0].plot(x_trajectory[-1, 0], x_trajectory[-1, 1], 'ro', markersize=10, label='End')
        axes[1, 0].plot(0, 0, 'k*', markersize=15, label='Target')
        axes[1, 0].set_xlabel('x1')
        axes[1, 0].set_ylabel('x2')
        axes[1, 0].set_title('Phase Portrait')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Solve times
        axes[1, 1].plot(time_vector, solve_times, 'co-', markersize=4)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Solve Time [s]')
        axes[1, 1].set_title('Computational Performance')
        axes[1, 1].grid(True)
        
        # Dwell times evolution (first phase)
        first_phase_dwell_times = [dwell[0] for dwell in dwell_times_history]
        axes[1, 2].plot(time_vector, first_phase_dwell_times, 'ko-', markersize=4)
        axes[1, 2].axhline(y=dt_sampling, color='r', linestyle='--', alpha=0.7, label=f'Δt = {dt_sampling}s')
        axes[1, 2].set_xlabel('Time [s]')
        axes[1, 2].set_ylabel('First Phase Dwell Time [s]')
        axes[1, 2].set_title('Dwell Time Evolution')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
            
        # Use the class to create the animation
        plt.tight_layout()
            
        animator = MPCAnimator(
        model=model,
        time_vector=time_vector,
        x_trajectory=x_trajectory,
        u_trajectory=u_trajectory,
        mode_sequences=mode_sequences,
        dwell_times_history=dwell_times_history
        )
        
        # Create animation and optionally save it
        animation = animator.create_animation(
        interval=500, 
        )
        
        # Show the animation
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return {
        'time': time_vector,
        'states': x_trajectory,
        'controls': u_trajectory,
        'costs': costs,
        'solve_times': solve_times,
        'mode_sequences': mode_sequences,
        'dwell_times': dwell_times_history,
        'mode_switches': mode_switches,
        'controller': mpc_controller
    }

def closed_loop_tracking_mpc():
    """
    Original closed-loop MPC with reference tracking and disturbances.
    """
    print("=== Advanced Closed-Loop MPC with Tracking ===")
    
    # Define system model (double integrator-like system)
    model = {
        'A': [np.array([[0.0, 1.0], 
                       [0.0, 0.0]]), 
              np.array([[0.0, 1.0], 
                         [1.0, 1.0]]),
        ],
        'B': [np.array([[0.0], 
                       [1.0]]),
              np.array([[0.005], 
                         [0.5]]),
        ]
    }
    
    # System dimensions
    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]
    
    # MPC parameters
    n_phases = 6
    prediction_horizon = 1.5
    dt_control = 0.1
    n_sim_steps = 50
    multiple_shooting = True
    hybrid = True
    
    # Cost weights
    Q = np.diag([10.0, 1.0])      # Position more important than velocity
    R = np.array([[0.1]])         # Control cost
    E = np.diag([100.0, 100.0])    # Terminal cost
    
    # Initial conditions
    x_current = np.array([0.0, 0.0])  # Start at origin
    
    # Time-varying reference trajectory (sinusoidal)
    def reference_trajectory(t):
        return np.array([2.0 * np.sin(0.5 * t), 
                        1.0 * np.cos(0.5 * t)])
    
    # Bounds
    bounds = {
        'states_lb': np.array([-5.0, -3.0]),
        'states_ub': np.array([5.0, 3.0]),
        'inputs_lb': -2.0,
        'inputs_ub': 2.0,
    }
    
    # Noise and disturbance parameters
    process_noise_std = 0.01  # Standard deviation of process noise
    measurement_noise_std = 0.005  # Standard deviation of measurement noise
    
    print(f"System: {n_states} states, {n_inputs} inputs")
    print(f"Prediction horizon: {prediction_horizon}s")
    print(f"Control sampling: {dt_control}s")
    print(f"Process noise std: {process_noise_std}")
    
    # Create receding horizon MPC controller
    mpc_controller = RecedingHorizonMPC(
        model=model,
        n_phases=n_phases,
        prediction_horizon=prediction_horizon,
        dt_sampling=dt_control,
        Q=Q, R=R, E=E,
        bounds=bounds,
        multiple_shooting=multiple_shooting,
        hybrid=hybrid
    )
    
    # Storage for results
    time_vector = []
    x_trajectory = []
    x_measured = []
    u_trajectory = []
    ref_trajectory = []
    costs = []
    solve_times = []
    tracking_errors = []
    mode_sequences = []
    dwell_times_history = []
    mode_switches = []
    pred_trajectory = []
    
    # Initial setup
    mpc_controller.setup_mpc(x_current)
    
    # Initialize previous solution for warm starting
    prev_inputs = None
    prev_deltas = None
    
    # Simulation loop
    for step in range(n_sim_steps):
        current_time = step * dt_control
        
        # Get current reference
        x_ref = reference_trajectory(current_time)
        
        # Add measurement noise to current state
        x_measured_current = x_current + np.random.normal(0, measurement_noise_std, n_states)
        
        print(f"\nStep {step + 1}/{n_sim_steps} (t={current_time:.2f}s)")
        print(f"True state: [{x_current[0]:.4f}, {x_current[1]:.4f}]")
        print(f"Measured:   [{x_measured_current[0]:.4f}, {x_measured_current[1]:.4f}]")
        print(f"Reference:  [{x_ref[0]:.4f}, {x_ref[1]:.4f}]")
        
        # Create & solve the MPC controller
        solve_start = time.time()
        
        # Solve MPC
        u_applied, solve_info = mpc_controller.solve_mpc(x_measured_current, x_ref)
        
        solve_time = time.time() - solve_start
        
        print(f"Applied control: u = {u_applied:.6f}")
        print(f"Solve time: {solve_info['solve_time']:.4f}s")
        print(f"Cost: {solve_info['cost']:.6f}")
        
        # Apply control to true system with process noise
        A_c = model['A'][0]
        B_c = model['B'][0]
        
        # Process noise
        w = np.random.normal(0, process_noise_std, n_states)
        
        # For continuous system, use matrix exponential for state evolution
        from scipy.linalg import expm
        
        # Discretize system matrices using matrix exponential method
        M = np.zeros((n_states + n_inputs, n_states + n_inputs))
        M[:n_states, :n_states] = A_c
        M[:n_states, n_states:] = B_c
        
        expM = expm(M * dt_control)
        A_d = expM[:n_states, :n_states]
        B_d = expM[:n_states, n_states:]
        
        # System evolution using discretized matrices
        x_next = A_d @ x_current + B_d @ np.array([u_applied]) + w
        
        # Compute tracking error
        tracking_error = np.linalg.norm(x_current - x_ref)
        
        # Store results
        time_vector.append(current_time)
        x_trajectory.append(x_current.copy())
        x_measured.append(x_measured_current.copy())
        u_trajectory.append(u_applied)
        ref_trajectory.append(x_ref.copy())
        costs.append(solve_info['cost'])
        solve_times.append(solve_time)
        tracking_errors.append(tracking_error)
        mode_sequences.append(solve_info['mode_sequence'].copy())
        dwell_times_history.append(solve_info['dwell_times'].copy())
        pred_trajectory.append(solve_info['states_opt'].copy())
        
        # Update dwell times and check for mode switches
        mode_switched = mpc_controller.update_dwell_times_and_modes()
        mode_switches.append(mode_switched)
        
        # Update state
        x_current = x_next
        
        print(f"Tracking error: {tracking_error:.6f}")
    
    # Convert to numpy arrays
    time_vector = np.array(time_vector)
    x_trajectory = np.array(x_trajectory)
    x_measured = np.array(x_measured)
    u_trajectory = np.array(u_trajectory)
    ref_trajectory = np.array(ref_trajectory)
    tracking_errors = np.array(tracking_errors)
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    print(f"Average solve time: {np.mean(solve_times):.4f}s ± {np.std(solve_times):.4f}s")
    print(f"Max solve time: {np.max(solve_times):.4f}s")
    print(f"Average tracking error: {np.mean(tracking_errors):.6f}")
    print(f"Max tracking error: {np.max(tracking_errors):.6f}")
    print(f"RMS tracking error: {np.sqrt(np.mean(tracking_errors**2)):.6f}")
    print(f"Total control effort: {np.sum(np.abs(u_trajectory)):.6f}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # State tracking
        axes[0, 0].plot(time_vector, x_trajectory[:, 0], 'b-', linewidth=2, label='True x1')
        axes[0, 0].plot(time_vector, ref_trajectory[:, 0], 'r--', linewidth=2, label='Ref x1')
        axes[0, 0].plot(time_vector, x_measured[:, 0], 'g:', alpha=0.7, label='Measured x1')
        axes[0, 0].set_xlabel(r'Time $[s]$')
        axes[0, 0].set_ylabel(r'Position $[m]$')
        axes[0, 0].set_title(r'Position Tracking')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(time_vector, x_trajectory[:, 1], 'b-', linewidth=2, label='True x2')
        axes[0, 1].plot(time_vector, ref_trajectory[:, 1], 'r--', linewidth=2, label='Ref x2')
        axes[0, 1].plot(time_vector, x_measured[:, 1], 'g:', alpha=0.7, label='Measured x2')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Velocity')
        axes[0, 1].set_title('Velocity Tracking')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Control input
        axes[0, 2].step(time_vector, u_trajectory, 'g-', where='post', linewidth=2)
        axes[0, 2].axhline(y=bounds['inputs_lb'], color='r', linestyle='--', alpha=0.7, label='Bounds')
        axes[0, 2].axhline(y=bounds['inputs_ub'], color='r', linestyle='--', alpha=0.7)
        axes[0, 2].set_xlabel('Time [s]')
        axes[0, 2].set_ylabel('Control Input')
        axes[0, 2].set_title('Control Signal')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Tracking error
        axes[1, 0].plot(time_vector, tracking_errors, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Tracking Error')
        axes[1, 0].set_title('Tracking Error Evolution')
        axes[1, 0].grid(True)
        
        # Solve times
        axes[1, 1].plot(time_vector, solve_times, 'mo-', markersize=4)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Solve Time [s]')
        axes[1, 1].set_title('Computational Performance')
        axes[1, 1].grid(True)
        
        # Phase portrait
        axes[1, 2].plot(x_trajectory[:, 0], x_trajectory[:, 1], 'b-', linewidth=2, label='Actual')
        axes[1, 2].plot(ref_trajectory[:, 0], ref_trajectory[:, 1], 'r--', linewidth=2, label='Reference')
        axes[1, 2].plot(x_trajectory[0, 0], x_trajectory[0, 1], 'go', markersize=8, label='Start')
        axes[1, 2].plot(x_trajectory[-1, 0], x_trajectory[-1, 1], 'ro', markersize=8, label='End')
        axes[1, 2].set_xlabel('Position')
        axes[1, 2].set_ylabel('Velocity')
        axes[1, 2].set_title('Phase Portrait')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        animator = MPCAnimator(
        model=model,
        time_vector=time_vector,
        x_trajectory=x_trajectory,
        pred_trajectory=pred_trajectory,
        x_ref_trajectory=ref_trajectory,
        u_trajectory=u_trajectory,
        mode_sequences=mode_sequences,
        dwell_times_history=dwell_times_history
        )
        
        # Create animation and optionally save it
        animation = animator.create_animation(
        interval=500, 
        )
        
        # Active modes over time
        active_modes = [seq[0] for seq in mode_sequences]
        
        # Create a new figure for active modes over time
        fig_modes, ax_modes = plt.subplots(figsize=(10, 4))
        ax_modes.step(time_vector, active_modes, 'mo-', where='post', linewidth=2, markersize=4)
        
        # Mark mode switches with vertical lines
        for i, switched in enumerate(mode_switches):
            if switched and i < len(time_vector):
                ax_modes.axvline(x=time_vector[i], color='r', linestyle='--', alpha=0.5)
            
        ax_modes.set_xlabel('Time [s]')
        ax_modes.set_ylabel('Active Mode')
        ax_modes.set_title('Active Mode Evolution')
        ax_modes.set_ylim(-0.5, max(active_modes) + 0.5)
        ax_modes.grid(True)
        
        # Add annotations for dwell times
        for i in range(min(5, len(time_vector))):  # Annotate first few time steps
            if i < len(dwell_times_history):
                dt = dwell_times_history[i][0]  # First phase dwell time
            ax_modes.annotate(f"{dt:.2f}s", 
                     (time_vector[i], active_modes[i]),
                     textcoords="offset points",
                     xytext=(0,10), 
                     ha='center')
                     
        plt.tight_layout()
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return (time_vector, x_trajectory, x_measured, u_trajectory, 
            ref_trajectory, tracking_errors, solve_times)

if __name__ == '__main__':
    try:
        print("Choose which example to run:")
        print("1. Closed-Loop MPC with Optimizable Dwell Times (2 modes)")
        print("2. Original Tracking MPC")
        print("3. n-Modes Generalization Demo (5 modes)")
        
        # For demo purposes, run the n-modes example to show generalization
        choice = "2"  # You can change this or add input()
        
        if choice == "1":
            print("\nRunning Closed-Loop MPC with Optimizable Dwell Times...")
            results = closed_loop_mpc_with_dwell_times()
            print("\nDwell times MPC completed successfully!")
            
            # Print some key insights
            print(f"\nKey Insights:")
            print(f"- Total mode switches: {sum(results['mode_switches'])}")
            print(f"- Modes used: {set([seq[0] for seq in results['mode_sequences']])}")
            print(f"- Average solve time: {np.mean(results['solve_times']):.4f}s")
            
        elif choice == "2":
            print("\nRunning Original Tracking MPC...")
            results = closed_loop_tracking_mpc()
            print("\nTracking MPC completed successfully!")
            
        elif choice == "3":
            print("\nRunning n-Modes Generalization Demo...")
            results = demonstrate_n_modes_generalization()
            print("\nn-Modes demonstration completed successfully!")
            
            print(f"\nGeneralization Results:")
            print(f"- System had {results['n_modes']} modes available")
            print(f"- Actually used modes: {results['modes_used']}")
            print(f"- Total mode switches: {results['total_switches']}")
            print(f"- Mode cycling successful: {len(results['modes_used']) == results['n_modes']}")
        
        else:
            print("Invalid choice")
        
    except Exception as e:
        print(f"Error occurred: {e}")