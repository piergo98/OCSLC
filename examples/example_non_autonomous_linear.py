import time

import casadi as ca
import numpy as np

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_non_autonomous_switched_linear_3():
    model = {
        'A': [np.array([[-1, 0], [1, 2]])],
        'B': [np.array([[1], [1]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 40
    time_horizon = 1

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)

    Q = 100. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([3, 5])
    
    xr = np.array([1, 2])

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    swi_lin_mpc.set_bounds(-100, 100)
    
    # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    # final_state_constr = [xf]
    # final_state_lb = xr
    # final_state_ub = xr

    # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

    swi_lin_mpc.set_cost_function(R, x0)

    swi_lin_mpc.create_solver()

    inputs_opt, deltas_opt = swi_lin_mpc.solve()
    
    swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
if __name__ == '__main__':
    test_non_autonomous_switched_linear_3()
    print("All tests passed!")