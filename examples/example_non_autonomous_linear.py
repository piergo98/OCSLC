import time

import casadi as ca
from matplotlib import pyplot as plt
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
    
    Q = 1000. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([3, 5])
    
    xr = np.array([1, 2])

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False, multiple_shooting=False, x0=x0)

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    states_lb = np.array([-100, -100])
    states_ub = np.array([100, 100]) 
    
    swi_lin_mpc.set_bounds(-100, 100, states_lb, states_ub)
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints(x0)
    
    # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    # final_state_constr = [xf]
    # final_state_lb = xr
    # final_state_ub = xr

    # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

    swi_lin_mpc.set_cost_function(Q, R, x0)

    optimal_cost = []
    for iter_num in range(1, 227):
        # Set the initial guess  
        swi_lin_mpc.set_initial_guess(time_horizon, x0)
        swi_lin_mpc.create_solver(max_iter=iter_num)

        # DSS: 226 iterations
        # DMS: 1960 iterations
        inputs_opt, deltas_opt, _ = swi_lin_mpc.solve()
        optimal_cost.append(swi_lin_mpc.opt_cost)
    
    x_axes = np.arange(1, 227, 1)
    plt.plot(x_axes, optimal_cost)
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel('Optimal Cost')
    
    plt.show()
    # swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
if __name__ == '__main__':
    start = time.time()
    test_non_autonomous_switched_linear_3()
    print(f"Execution time: {time.time() - start}")
    print("All tests passed!")