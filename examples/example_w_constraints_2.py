import time

import casadi as ca
import numpy as np
import scipy.io
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_non_uniform_non_autonomous_linear_constrained(args):
    
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
    epsilon = 1e-1
    model = {
        'A': [np.array([[-40.0, 10.0, 0.0], [0.0, -2.0, 1.0], [0.0, 0.0, -0.05]])],
        'B': [np.array([[1.0], [0.5], [1.0]])],
        # 'A': [np.array([[-1/epsilon, 1/epsilon], [0.0, -1.0]])],
        # 'B': [np.array([[1/epsilon], [0.0]])],
    }
    # print("--------------------------------")
    # print(np.linalg.eigvals(model['A'][0]))

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 20.0
    
    x0 = np.array([2.0, 1.0, 1.0])

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

    Q = np.diag([10.0, 1.0, 1.0])
    R = 0.05 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))
    print(np.linalg.eigvals(P))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    precompute_time = time.time() - start
    print(f"Precomputation time: {precompute_time}")
    start = time.time()
    
    # loaded_data = scipy.io.loadmat('optimal_results_hybrid.mat')
    # fixed_states = loaded_data['trajectory'][0]
    # fixed_inputs = loaded_data['controls'][0]
    
    states_lb = np.array([-2.5, -1.5, -100.0])
    states_ub = np.array([2.5, 1.5, 100.0]) 
    # states_ub = np.array([0.2, 0.2, 0.2]) 
    control_lb = np.array([-0.4])
    control_ub = np.array([0.4])
    
    swi_lin_mpc.set_bounds(
        control_lb, 
        control_ub, 
        states_lb, 
        states_ub, 
        # inspect_inputs=fixed_inputs,
        # inspect_states=fixed_states,
    )
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints()

    swi_lin_mpc.set_cost_function(Q, R, x0, P)
    
    # Set the initial guess  
    exp_dist = 1.0**np.arange(n_steps)
    phase_durations = exp_dist * time_horizon / np.sum(exp_dist)
    
    # Starting from x0, compute the unconstrained LQR optimal inputs
    # and states as initial guess
    K_lqr = np.linalg.inv(R + model['B'][0].T @ P @ model['B'][0]) @ (model['B'][0].T @ P @ model['A'][0])
    xk = x0.copy()
    x_init = [xk.flatten()]
    u_init = []
    for i in range(n_steps):
        uk = ca.DM(-K_lqr @ xk)
        xk = swi_lin_mpc.autonomous_evol[i](phase_durations[i]) @ xk + swi_lin_mpc.forced_evol[i](uk, phase_durations[i])
        x_init.append(xk.full().flatten())
        u_init.append(uk)
    x_init = np.array(x_init).reshape((n_steps+1, n_states))
    u_init = np.array(u_init).reshape((n_steps, n_inputs))
    
    
    swi_lin_mpc.set_initial_guess(
        x0, 
        # initial_state_trajectory=x_init, 
        # initial_control_inputs=u_init,
        # initial_phases_duration=phase_durations
    )

    swi_lin_mpc.create_solver('ipopt', print_level=5)
    
    setup_time = time.time() - start
    print(f"Setup time: {setup_time}")
    start = time.time()
    
    inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve(x0)
    # print(deltas_opt)
    # raise NotImplementedError("Stop here for now, as the results are not good. Need to investigate.")
    solving_time = time.time() - start
    print(f"Solving time: {solving_time}")
    print("--------------------------------")
    print(f"Total time: {precompute_time + setup_time + solving_time}")
    
    # if swi_lin_mpc.multiple_shooting:
    #     swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt, states_opt)
    # else:
    #     swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    return swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt
        
def test_uniform_non_autonomous_linear_constrained(args):
    
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
    epsilon = 1e-1
    model = {
        'A': [np.array([[-40.0, 10.0, 0.0], [0.0, -2.0, 1.0], [0.0, 0.0, -0.05]])],
        'B': [np.array([[1.0], [0.5], [1.0]])],
        # 'A': [np.array([[-1/epsilon, 1/epsilon], [0.0, -1.0]])],
        # 'B': [np.array([[1/epsilon], [0.0]])],
    }
    # print("--------------------------------")
    # print(np.linalg.eigvals(model['A'][0]))

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 20.0
    
    x0 = np.array([2.0, 1.0, 1.0])

    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_steps, 
        time_horizon, 
        auto=False,
        x0=x0,
        multiple_shooting=multiple_shooting,
        propagation=integrator,
        inspect = True,
        hybrid=hybrid,
        plot=plot,
    )

    Q = np.diag([10.0, 1.0, 1.0])
    R = 0.05 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    precompute_time = time.time() - start
    print(f"Precomputation time: {precompute_time}")
    start = time.time()
    
    # loaded_data = scipy.io.loadmat('optimal_results_hybrid.mat')
    # fixed_states = loaded_data['trajectory'][0]
    # fixed_inputs = loaded_data['controls'][0]
    
    states_lb = np.array([-2.5, -1.5, -100.0])
    states_ub = np.array([2.5, 1.5, 100.0])
    # states_ub = np.array([0.2, 0.2, 0.2])
    control_lb = np.array([-0.4])
    control_ub = np.array([0.4])
    
    swi_lin_mpc.set_bounds(
        control_lb, 
        control_ub, 
        states_lb, 
        states_ub, 
        # inspect_inputs=fixed_inputs,
        # inspect_states=fixed_states,
    )
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints()

    swi_lin_mpc.set_cost_function(Q, R, x0, P)
    
    # Set the initial guess  
    exp_dist = 1.0**np.arange(n_steps)
    phase_durations = exp_dist * time_horizon / np.sum(exp_dist)
    
    # Starting from x0, compute the unconstrained LQR optimal inputs
    # and states as initial guess
    K_lqr = np.linalg.inv(R + model['B'][0].T @ P @ model['B'][0]) @ (model['B'][0].T @ P @ model['A'][0])
    xk = x0.copy()
    x_init = [xk.flatten()]
    u_init = []
    for i in range(n_steps):
        uk = ca.DM(-K_lqr @ xk)
        xk = swi_lin_mpc.autonomous_evol[i](phase_durations[i]) @ xk + swi_lin_mpc.forced_evol[i](uk, phase_durations[i])
        x_init.append(xk.full().flatten())
        u_init.append(uk)
    x_init = np.array(x_init).reshape((n_steps+1, n_states))
    u_init = np.array(u_init).reshape((n_steps, n_inputs))
    
    
    swi_lin_mpc.set_initial_guess(
        x0, 
        # initial_state_trajectory=x_init, 
        # initial_control_inputs=u_init,
        # initial_phases_duration=phase_durations
    )

    swi_lin_mpc.create_solver('ipopt')
    
    setup_time = time.time() - start
    print(f"Setup time: {setup_time}")
    start = time.time()
    
    inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve(x0)
    solving_time = time.time() - start
    print(f"Solving time: {solving_time}")
    print("--------------------------------")
    print(f"Total time: {precompute_time + setup_time + solving_time}")
    
    return swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt

def plot_comparison(non_uniform_solution, uniform_solutions, n_steps_list):
    """
    Plot comparison between non-uniform time sampling solution and uniform QP solutions.
    
    Args:
        non_uniform_solution: Tuple (cost, states, inputs, deltas) for non-uniform solution
        uniform_solutions: List of tuples [(cost, states, inputs, deltas), ...] for each uniform solution
        n_steps_list: List of n_steps values corresponding to uniform_solutions
    """
    nu_cost, nu_states, nu_inputs, nu_deltas = non_uniform_solution

    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_states_array = np.array(nu_states)
    if nu_states_array.ndim == 1:
        n_points = len(nu_times)
        if nu_states_array.size % n_points != 0:
            raise ValueError(
                f"State vector length {nu_states_array.size} is not divisible by number of time points {n_points}."
            )
        n_states = nu_states_array.size // n_points
        nu_states_array = nu_states_array.reshape(n_points, n_states)
    else:
        n_states = nu_states_array.shape[1]
        nu_states_array = nu_states_array.reshape(-1, n_states)

    nu_inputs_array = np.array(nu_inputs)
    if nu_inputs_array.ndim == 1:
        nu_inputs_array = nu_inputs_array.reshape(-1, 1)
    n_inputs = nu_inputs_array.shape[1]

    total_plots = 1 + n_states + 1 + 1
    if n_states >= 2:
        total_plots += 1

    ncols = 2
    nrows = int(np.ceil(total_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    axis_idx = 0

    # Subplot: Cost comparison
    ax_cost = axes[axis_idx]
    axis_idx += 1
    if len(uniform_solutions) > 0:
        costs = [sol[0].item() for sol in uniform_solutions]
        ax_cost.plot(n_steps_list, costs, 'o-', label='Uniform time sampling', linewidth=2, markersize=6)
    ax_cost.axhline(y=nu_cost.item(), color='r', linestyle='--', linewidth=2, label='Non-uniform time sampling')
    ax_cost.set_ylim(-nu_cost * 10, nu_cost * 10)
    ax_cost.set_xlabel('Number of steps', fontsize=12)
    ax_cost.set_ylabel('Optimal cost', fontsize=12)
    ax_cost.set_title('Cost Comparison', fontsize=14, fontweight='bold')
    ax_cost.legend(fontsize=10)
    ax_cost.grid(True, alpha=0.3)

    # Select a few uniform solutions for comparison
    selected_indices = []
    if len(uniform_solutions) > 0:
        selected_indices = [0, len(uniform_solutions) // 3, 2 * len(uniform_solutions) // 3, -1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, len(selected_indices))))

    # Subplots: State trajectories
    for state_idx in range(n_states):
        ax_state = axes[axis_idx]
        axis_idx += 1

        ax_state.plot(nu_times, nu_states_array[:, state_idx], 'r-', linewidth=2.5, label='Non-uniform', zorder=10)
        for idx, color in zip(selected_indices, colors):
            if idx < len(uniform_solutions):
                _, states, _, deltas = uniform_solutions[idx]
                times = np.concatenate([[0], np.cumsum(deltas)])
                states_array = np.array(states).reshape(-1, n_states)
                ax_state.plot(times, states_array[:, state_idx], '--', color=color, linewidth=1.5,
                              label=f'Uniform (N={n_steps_list[idx]})', alpha=0.8)

        min_val = nu_states_array[:, state_idx].min()
        max_val = nu_states_array[:, state_idx].max()
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0
        ax_state.set_ylim(min_val * 1.5, max_val * 1.5)
        ax_state.set_xlabel('Time [s]', fontsize=12)
        ax_state.set_ylabel(f'State x{state_idx + 1}', fontsize=12)
        ax_state.set_title(f'State Trajectory x{state_idx + 1}', fontsize=14, fontweight='bold')
        ax_state.legend(fontsize=9)
        ax_state.grid(True, alpha=0.3)

    # Subplot: Control input comparison
    ax_u = axes[axis_idx]
    axis_idx += 1

    for input_idx in range(n_inputs):
        ax_u.step(nu_times[:-1], nu_inputs_array[:, input_idx], where='post', linewidth=2.5,
                  label=f'Non-uniform u{input_idx + 1}' if n_inputs > 1 else 'Non-uniform')

    for idx, color in zip(selected_indices, colors):
        if idx < len(uniform_solutions):
            _, _, inputs, deltas = uniform_solutions[idx]
            times = np.concatenate([[0], np.cumsum(deltas)])
            inputs_array = np.array(inputs)
            if inputs_array.ndim == 1:
                inputs_array = inputs_array.reshape(-1, 1)
            for input_idx in range(inputs_array.shape[1]):
                ax_u.step(times[:-1], inputs_array[:, input_idx], where='post', linestyle='--', color=color,
                          linewidth=1.5, alpha=0.8,
                          label=f'Uniform (N={n_steps_list[idx]})' if input_idx == 0 else '')

    ax_u.set_xlabel('Time [s]', fontsize=12)
    ax_u.set_ylabel('Control input', fontsize=12)
    ax_u.set_title('Control Input', fontsize=14, fontweight='bold')
    ax_u.legend(fontsize=9)
    ax_u.grid(True, alpha=0.3)

    # Subplot: Phase portrait (only if at least 2 states)
    if n_states >= 2:
        ax_phase = axes[axis_idx]
        axis_idx += 1

        ax_phase.plot(nu_states_array[:, 0], nu_states_array[:, 1], 'r-', linewidth=2.5,
                      marker='o', markersize=4, label='Non-uniform', zorder=10)
        ax_phase.plot(nu_states_array[0, 0], nu_states_array[0, 1], 'go', markersize=10,
                      label='Initial state', zorder=11)
        ax_phase.plot(nu_states_array[-1, 0], nu_states_array[-1, 1], 'r*', markersize=15,
                      label='Final state', zorder=11)

        for idx, color in zip(selected_indices, colors):
            if idx < len(uniform_solutions):
                _, states, _, _ = uniform_solutions[idx]
                states_array = np.array(states).reshape(-1, n_states)
                ax_phase.plot(states_array[:, 0], states_array[:, 1], '--', color=color, linewidth=1.5,
                              alpha=0.8, label=f'Uniform (N={n_steps_list[idx]})')

        ax_phase.set_xlim(nu_states_array[:, 0].min() * 1.5, nu_states_array[:, 0].max() * 1.5 + 0.5)
        ax_phase.set_ylim(nu_states_array[:, 1].min() * 1.5, nu_states_array[:, 1].max() * 1.5 + 0.5)
        ax_phase.set_xlabel('State x1', fontsize=12)
        ax_phase.set_ylabel('State x2', fontsize=12)
        ax_phase.set_title('Phase Portrait', fontsize=14, fontweight='bold')
        ax_phase.legend(fontsize=9)
        ax_phase.grid(True, alpha=0.3)

    # Subplot: Time distribution
    ax_time = axes[axis_idx]
    axis_idx += 1
    ax_time.bar(range(len(nu_deltas)), nu_deltas, color='red', alpha=0.7, label='Non-uniform')

    # Show uniform time distribution for last solution
    if len(uniform_solutions) > 0:
        for idx, color in zip(selected_indices, colors):
            _, _, _, deltas = uniform_solutions[idx]
            ax_time.axhline(y=deltas[0], color=color, linestyle='--', linewidth=2, label=f'Uniform (N={n_steps_list[idx]})')

    ax_time.set_xlabel('Phase index', fontsize=12)
    ax_time.set_ylabel('Phase duration [s]', fontsize=12)
    ax_time.set_title('Time Distribution', fontsize=14, fontweight='bold')
    ax_time.legend(fontsize=10)
    ax_time.grid(True, alpha=0.3, axis='y')

    # Hide any unused axes
    for idx in range(axis_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('comparison_uniform_vs_nonuniform.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--integrator',
        type=str, metavar="{int, exp}", default='exp', required=False,
        help='Integration method to use. Default is exp.'
    )
    parser.add_argument('--shooting',
        type=str, metavar="{ss, ms}", default='ms', required=False,
        help='Shooting method. Default is ms.'
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
    
    non_uniform_solution = test_non_uniform_non_autonomous_linear_constrained(args)
    non_uniform_N = args.n_steps

    uniform_solutions = []
    n_steps_list = []
    for n_steps in range(20, 100, 10):
        args.n_steps = n_steps
        solution = test_uniform_non_autonomous_linear_constrained(args)
        uniform_solutions.append(solution)
        n_steps_list.append(n_steps)
    
    # Extract costs for summary
    non_uniform_opt_cost = non_uniform_solution[0]
    uniform_opt_costs = [sol[0] for sol in uniform_solutions]
    
    print("\n" + "="*40)
    print("Results Summary")
    print("="*40)
    print(f"Non-uniform optimal cost (N={non_uniform_N}): {non_uniform_opt_cost.item()}")
    print("="*40)
    print("Uniform time sampling results:")
    print("-"*40)
    print(f"{'N Steps':<15} {'Optimal Cost':<15} {'Difference %':<15}")
    print("-"*40)
    for n_steps, cost in zip(n_steps_list, uniform_opt_costs):
        diff_percent = 100 * (cost.item() - non_uniform_opt_cost.item()) / non_uniform_opt_cost.item()
        print(f"{n_steps:<15} {cost.item():<15.6f} {diff_percent:+.3f}%")
    print("="*40 + "\n")
    
    # Plot comparison
    print("Generating comparison plots...")
    plot_comparison(non_uniform_solution, uniform_solutions, n_steps_list)
    
    print("All tests passed!")
