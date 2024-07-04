import casadi as ca
import numpy as np
from math import sqrt

from ocslc.switched_linear_mpc import SwitchedLinearMPC

def jumping_leg():
    
    m = 1.5 # robot mass
    g = 9.81 # gravity
    dmin = 0.05 # minimum distance between the feet
    l_box = 0.002 # box size
    leg_length = 0.35 # leg length
    h_box = (leg_length**2 - l_box**2)**0.5 - dmin # box height
    
    model = {
        'A': [np.array([[0, 1, 0, 0], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]), 
              np.array([[0, 1, 0, 0], [0, 0, 0, -1], [0, 1, 0, 0], [0, 0, 0, 0]]),
              ],
        'B': [np.array([[0, 0], [1/m, 0], [0, 0], [0, 0]]),
              np.array([[0, 0], [0, 0], [0, 1], [0, 0]]),
              ],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 20
    time_horizon = 1

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)

    Q = 100. * np.diag([1, 0, 0, 0])
    R = 0. * np.eye(n_inputs)
    E = 1. * np.diag([1, 0, 0, 0])

    x0 = np.array([0.1, 0, 0, g])
    
    xr = np.array([1.5, 0, 1.0, g])

    swi_lin_mpc.precompute_matrices(x0, Q, R, E, xr)
    
    # Add the minimum values of the positions
    for i in range(n_phases):
        x = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs, i)[i]
        state_constr = [x]
        state_lb = np.array([0, -np.inf, 0, g])
        state_ub = np.array([np.inf, np.inf, np.inf, g])

        swi_lin_mpc.add_constraint(state_constr, state_lb, state_ub)
        
        box_const = [x[0] - x[2]]
        box_lb = dmin
        box_ub = h_box
        swi_lin_mpc.add_constraint(box_const, box_lb, box_ub)
    
    inputs_lb = np.array([-100, -10])
    inputs_ub = np.array([100, 10])
    swi_lin_mpc.set_bounds(inputs_lb, inputs_ub)

    swi_lin_mpc.set_cost_function(R, x0)

    swi_lin_mpc.create_solver()

    inputs_opt, deltas_opt = swi_lin_mpc.solve()
    
    swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
if __name__ == '__main__':
    jumping_leg()
    print("All tests passed!")