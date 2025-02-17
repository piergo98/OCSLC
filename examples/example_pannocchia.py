import time

import casadi as ca
import numpy as np
import scipy.io
from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_non_autonomous_switched_linear_pannocchia(args):
    
    integrator = args.integrator
    if args.shooting == 'ss':
        multiple_shooting = False
    elif args.shooting == 'ms':
        multiple_shooting = True
    else:
        return ValueError("Invalid shooting method.")
    hybrid = args.hybrid
    n_steps = args.n_steps
    plot = args.plot
    
    # ======================================================================= #
    
    start = time.time()
    
    model = {
        'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
        'B': [np.array([[0.25], [2], [0]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 10
    
    x0 = np.array([1.3440, -4.5850, 5.6470])

    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_steps, 
        time_horizon, 
        auto=False,
        x0=x0,
        multiple_shooting=multiple_shooting,
        propagation=integrator,
        inspect = False,
        hybrid=hybrid,
        plot=plot,
    )

    Q = 1. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    precompute_time = time.time() - start
    print(f"Precomputation time: {precompute_time}")
    start = time.time()
    
    loaded_data = scipy.io.loadmat('optimal_results_hybrid.mat')
    fixed_states = loaded_data['trajectory'][0]
    fixed_inputs = loaded_data['controls'][0]
    
    states_lb = np.array([-100, -100, -100])
    states_ub = np.array([100, 100, 100]) 
    
    swi_lin_mpc.set_bounds(
        -1, 
        1, 
        states_lb, 
        states_ub, 
        # inspect_inputs=fixed_inputs,
        # inspect_states=fixed_states,
    )
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints(x0)

    swi_lin_mpc.set_cost_function(Q, R, x0, P)
    
    # Set the initial guess  
    exp_dist = 0.99**np.arange(80)
    phase_durations = exp_dist * time_horizon / np.sum(exp_dist)
    
    swi_lin_mpc.set_initial_guess(
        time_horizon, 
        x0, 
        # initial_phases_duration=phase_durations
    )

    swi_lin_mpc.create_solver('ipopt')
    
    setup_time = time.time() - start
    print(f"Setup time: {setup_time}")
    start = time.time()
    
    inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve()
    solving_time = time.time() - start
    print(f"Solving time: {solving_time}")
    print("--------------------------------")
    print(f"Total time: {precompute_time + setup_time + solving_time}")
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt, states_opt)
    else:
        swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--integrator',
        type=str, metavar="{int, exp}", default='int', required=False,
        help='Integration method to use. Default is int.'
    )
    parser.add_argument('--shooting',
        type=str, metavar="{ss, ms}", default='ss', required=False,
        help='Shooting method.'
    )
    parser.add_argument('--hybrid',
        type=str, default=False, required=False,
        help='Hybrid method.'
    )
    parser.add_argument('--n_steps',
        type=int, metavar="int", default=80, required=False,
        help='Number of steps.'
    )
    parser.add_argument('--plot',
        type=str, metavar="{display, save, none}", default="display", required=False,
        help='How to plot the results.'
    )
    args = parser.parse_args()
    if args.hybrid in ('False', 'false', '0'):
        args.hybrid = False
    
    start = time.time()
    test_non_autonomous_switched_linear_pannocchia(args)
    # print(f"Execution time: {time.time() - start}")
    print("All tests passed!")
