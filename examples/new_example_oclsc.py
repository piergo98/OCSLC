import time

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwitchedLinearMPC

def LQR_controller(x0, A, B, P, R, time_horizon, n_states, phase_durations):
    # Compute the optimal control
    K = np.linalg.inv(R) @ np.transpose(B) @ P

    # Plot the state trajectory
    state = ca.MX.sym('state', n_states)
    t = ca.MX.sym('t')

    xdot = (A - B @ K) @ state
    dae = {'x': state, 'ode': xdot}

    traj = [x0]
    M = 1
    for phase in phase_durations:
        dt = phase / M
        for i in range(M):
            dF = ca.integrator('dF', 'cvodes', dae, 0, dt)
            x0 = dF(x0=x0)['xf']
            traj.append(x0.full().flatten())

    traj = np.array(traj)
    control_input = -K @ traj[:, :3].T
    # control_input = np.clip(control_input, -10, 10)
    
    # Plot the control input and the state trajectory
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(traj[:, 0], label='x1')
    # plt.plot(traj[:, 1], label='x2')
    # plt.plot(traj[:, 2], label='x3')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(control_input[0, :], label='u')
    # plt.legend()
    # plt.show()
    
    

    return traj, control_input.flatten()

def test_example_linear_oclsc(args):
    
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
        'A': [np.array([[0.5, 0, 0.1], [-4, -2.58, 1.57], [1.12, -1.94, -2.76]])],
        'B': [np.array([[1], [0], [0]])],
        # 'A': [np.array([[-2.1, -25.2, -2.5], [1, 0, 0], [0, 1, 0]])],
        # 'B': [np.array([[1], [0], [0]])],
        # 'A': [np.array([[1.63, 0.41], [1, 0]])],
        # 'B': [np.array([[1], [0]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 1.0
    
    x0 = np.array([0.22170992, 0.64490946, 0.13338063])
    # x0 = np.array([0.45858989, 0.54141011])

    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_steps, 
        time_horizon, 
        auto=False,
        x0=x0,
        multiple_shooting=multiple_shooting,
        propagation=integrator,
        inspect=False,
        hybrid=hybrid,
        plot=plot,
    )

    Q = 10. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    precompute_time = time.time() - start
    print(f"Precomputation time: {precompute_time}")
    start = time.time()
    
    states_lb = np.array([-100, -100, -100])
    states_ub = np.array([100, 100, 100]) 
    # states_lb = np.array([-100, -100])
    # states_ub = np.array([100, 100]) 
    
    
    swi_lin_mpc.set_bounds(-10, 10, states_lb, states_ub)
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints(x0)

    swi_lin_mpc.set_cost_function(Q, R, x0, P)
    
    ## Set the initial guess
    # Set the initial guess for the phase durations
    exp_dist = 1.**np.arange(n_steps)
    phase_durations = exp_dist * time_horizon / np.sum(exp_dist)
    
    # Set the unconstrained LQR as initial guess
    init_traj, init_controls = LQR_controller(
        x0, 
        model['A'][0], 
        model['B'][0], 
        P, 
        R, 
        time_horizon, 
        n_states,
        phase_durations
    )
    
    swi_lin_mpc.set_initial_guess(
        time_horizon, x0, 
        # initial_state_trajectory=init_traj, 
        # initial_control_inputs=init_controls, 
        # initial_phases_duration=phase_durations
    )
    # swi_lin_mpc.set_initial_guess(time_horizon, x0)

    swi_lin_mpc.create_solver('sqpmethod', print_level=0)
    
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
    test_example_linear_oclsc(args)
    # print(f"Execution time: {time.time() - start}")
    print("All tests passed!")
