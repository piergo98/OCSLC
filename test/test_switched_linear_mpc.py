import pytest
import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

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
            n_p = int(np.where(t < t_switching)[0][0])  # phase index
            system_id = n_p % self.n_systems
            A_p = self.model['A'][system_id]
            B_p = self.model['B'][system_id]
            
            x += A_p @ x * self.dt / self.n_steps
            if len(u) != 0:
                up = u[n_p*self.n_inputs:n_p*self.n_inputs+self.n_inputs]
                x += (B_p @ up) * self.dt / self.n_steps
            
        return x
    
    
@pytest.mark.skip()
def test_linear_mpc():
    model = {
        'A': [np.array([[-1, 0], [1, -2]])],
        'B': [np.eye(2)]
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 20
    time_horizon = 5

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)
    
    dt = 0.1
    n_substeps = 100
    sys = SystemLTI(model, dt, n_steps=n_substeps)

    Q =  0. * np.eye(n_states)
    R =  1. * np.eye(n_inputs)
    E = 10. * np.eye(n_states)

    x0 = np.array([1., 3])
    
    xr1 = np.array([1.,  3])
    xr2 = np.array([2., -5])
    
    swi_lin_mpc.set_bounds(-10, 10)
    
    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    # final_state_constr = [xf]
    # final_state_lb = xr1
    # final_state_ub = xr1
    # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub, "final_state")

    swi_lin_mpc.set_cost_function(R, x0)

    swi_lin_mpc.create_solver()
    
    n_steps = 100
    x = x0.copy()
    state_hist = [x0]
    for i in range(n_steps):
        x_meas = x

        xr = xr1 + (xr2 - xr1) * i/n_steps        
        # swi_lin_mpc.update_constraint("final_state", lbg=xr, ubg=xr)
        inputs_opt, deltas_opt = swi_lin_mpc.step(R, x_meas)
        
        x = sys.evolve_system(x, inputs_opt, deltas_opt)
        state_hist.append(x.flatten())
        
    print(state_hist)
    
    # Plot the state evolution
    time_grid = np.linspace(0, n_steps*dt, n_steps+1)
    plt.plot(time_grid, np.array(state_hist))
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('States')
    plt.title("State Evolution")
    
    plt.show()

if __name__ == "__main__":
    test_linear_mpc()
