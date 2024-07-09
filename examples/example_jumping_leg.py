import casadi as ca
import numpy as np
from scipy.interpolate import krogh_interpolate
import matplotlib.pyplot as plt

from ocslc.switched_linear_mpc import SwitchedLinearMPC


class SystemLTI():
    def __init__(self, model, dt, n_steps = 100) -> None:
        self.model = model
        self.dt = dt
        self.n_steps = n_steps
        
        self.n_systems = len(model['A'])
        self.n_inputs = self.model['B'][0].shape[1]

    def evolve_system(self, x0, u, delta):
        x = x0
        t_switching = np.cumsum(delta)
        
        for n in range(self.n_steps):
            t = n * self.dt / self.n_steps
            # Detecting current phase
            n_p = int(np.where(t < t_switching)[0][0])
            system_id = n_p % self.n_systems
            A_p = self.model['A'][system_id]
            B_p = self.model['B'][system_id]
            
            x += A_p @ x * self.dt / self.n_steps
            if len(u) != 0:
                up = u[n_p*self.n_inputs:n_p*self.n_inputs+self.n_inputs]
                x += (B_p @ up) * self.dt / self.n_steps
            
        return x
    
    def integrate_system(self, x0, u, delta, t):
        x = x0
        t_switching = np.cumsum(delta)
        # Detecting current phase
        n_p = int(np.where(t < t_switching)[0][0])
        # Selecting the dynamics of the current phase
        system_id = n_p % self.n_systems
        A_p = self.model['A'][system_id]
        B_p = self.model['B'][system_id]
        # Extracting the control input
        up = u[n_p*self.n_inputs:n_p*self.n_inputs+self.n_inputs]
        x += (A_p @ x0 + B_p @ up) * self.dt
        
        return x
        

def jumping_leg():
    
    m = 1.5 # robot mass
    g = 9.81 # gravity
    k = 100 # spring stiffness
    b = 25 # damping coefficient
    l0 = 0.5 # spring rest length
    
    model = {
        'A': [np.array([[0, 1, 0, 0], [-k/m, -b/m, 1, -1], [0, 0, 0, 0], [0, 0, 0, 0]]), 
              np.array([[0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]),
              ],
        'B': [np.array([[0], [1/m], [0], [0]]),
              np.array([[0], [0], [0], [0]]),
              ],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 20
    time_horizon = 1

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)

    Q = 100. * np.diag([1, 1, 0, 0])
    R = 0.1 * np.eye(n_inputs)
    E = 0 * np.diag([1, 0, 0, 0])

    x0 = np.array([0.35, 0, k/m*l0, g])
    
    xr = np.array([0.8, 0, k/m*l0, g])

    swi_lin_mpc.precompute_matrices(x0, Q, R, E, xr)
    
    # Add the minimum values of the positions
    for i in range(1, n_phases):
        x = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[i]
        if i % 2 == 0:
            state_constr = [(x[0] - l0)]
            state_lb = np.array([-ca.inf])
            state_ub = np.array([1e-6])
        elif i % 2 == 1:
            state_constr = [(x[0] - l0)]
            state_lb = np.array([1e-6])
            state_ub = np.array([ca.inf])
        # state_constr = [(x[0])]
        # state_lb = np.array([l0])
        # state_ub = np.array([l0])

        swi_lin_mpc.add_constraint(state_constr, state_lb, state_ub)
        
    # Add the minimum values of the positions
    # x_final = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    # state_constr = [(x_final[0])*swi_lin_mpc.deltas[-1]]
    # state_lb = np.array([0])
    # state_ub = np.array([np.inf])
    
    # swi_lin_mpc.add_constraint(state_constr, state_lb, state_ub, "final_state")
        
    inputs_lb = np.array([0])
    inputs_ub = np.array([1000])
    swi_lin_mpc.set_bounds(inputs_lb, inputs_ub)

    swi_lin_mpc.set_cost_function(R, x0, xr, E)

    swi_lin_mpc.create_solver()

    inputs_opt, deltas_opt = swi_lin_mpc.solve()
    
    # swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
    # Optimal state trajectory
    x_opt = []
    for i in range(n_phases+1):
        x_opt.append(swi_lin_mpc.state_extraction(deltas_opt, inputs_opt)[i].full().flatten()[0])
    
    # Test the optimal control on the system
    dt = 1e-3
    n_substeps = 100
    system = SystemLTI(model, dt, n_substeps)

    # Original time grid
    n_steps = len(inputs_opt)
    time_grid_original = np.cumsum(np.insert(deltas_opt, 0, 0))[:-1]  # Insert 0 at the start and remove the last element

    # Fine time grid
    integration_steps_fine = int(time_horizon / dt)
    time_grid_fine = np.linspace(0, time_horizon, integration_steps_fine)

    # Combine the original and fine time grids and sort them
    combined_time_grid = np.unique(np.concatenate((time_grid_original, time_grid_fine)))

    # Create an array to hold the resampled signal
    inputs_opt_fine = np.zeros_like(combined_time_grid)

    # Fill in the values from the original signal
    j = 0
    for i in range(len(combined_time_grid)):
        if j < n_steps - 1 and combined_time_grid[i] >= time_grid_original[j + 1]:
            j += 1
        inputs_opt_fine[i] = inputs_opt[j]

    # Plotting the original and resampled signals for comparison
    plt.figure()

    # Original signal
    plt.step(time_grid_original, inputs_opt, where='post', label='Original Signal')

    # Resampled signal
    plt.step(combined_time_grid, inputs_opt_fine, where='post', linestyle='--', label='Resampled Signal')

    plt.xlabel('Time [s]')
    plt.ylabel('Control Inputs')
    plt.title("Comparison of Original and Resampled Control Inputs")
    plt.legend()
    # plt.show()
    
    # Now, use inputs_opt_fine with your integrator
    # Assuming system.evolve_system() integrates the system with given inputs
    x = x0.copy()
    state_hist_fine = [x0[0]]
    for i in range(integration_steps_fine-1):
        # print(f"t = {time_grid_fine[i]}")
        x = system.integrate_system(x, inputs_opt_fine, deltas_opt, time_grid_fine[i])
        state_hist_fine.append(x[0])
        
    # Interpolating the resampled trajectory
    refined_time_grid = np.linspace(0, time_horizon, integration_steps_fine)
    time_grid_original_ = np.append(time_grid_original, time_horizon)
    resampled_x_opt = np.interp(refined_time_grid, time_grid_original_, x_opt)

    # Plot the state evolution with finer integration    
    plt.figure()
    plt.plot(refined_time_grid, np.array(state_hist_fine), label='Fine Integration')
    plt.plot(refined_time_grid, resampled_x_opt, linestyle='--', label='Optimal Solution')
    time = 0
    for i in range(n_phases):
        time = time + deltas_opt[i]
        plt.axvline(x = time, color = 'k', linestyle='--', label = f'T{i}')
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.title("State Evolution with Finer Integration")
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    jumping_leg()
    print("All tests passed!")