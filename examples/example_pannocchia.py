import time

import casadi as ca
import numpy as np

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_non_autonomous_switched_linear_pannocchia():
    model = {
        'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
        'B': [np.array([[0.25], [2], [0]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 80
    time_horizon = 10
    
    x0 = np.array([1.3440, -4.5850, 5.6470])
    
    xr = np.array([1, 2])

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False, multiple_shooting=True, x0=x0)    

    Q = 1. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    states_lb = np.array([-100, -100, -100])
    states_ub = np.array([100, 100, 100]) 
    
    swi_lin_mpc.set_bounds(-1, 1, states_lb, states_ub)
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints(x0)
    
    # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    # final_state_constr = [xf]
    # final_state_lb = xr
    # final_state_ub = xr

    # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

    swi_lin_mpc.set_cost_function(Q, R, x0)
    
    # Set the initial guess  
    swi_lin_mpc.set_initial_guess(time_horizon, x0)

    swi_lin_mpc.create_solver('ipopt')
    
    inputs_opt, deltas_opt, _ = swi_lin_mpc.solve()
    
    swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
if __name__ == '__main__':
    start = time.time()
    test_non_autonomous_switched_linear_pannocchia()
    print(f"Execution time: {time.time() - start}")
    print("All tests passed!")