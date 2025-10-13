import numpy as np
import time

from ocslc.switched_linear_mpc import SwitchedLinearMPC

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
