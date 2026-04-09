import time

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwitchedLinearMPC


# Cart-pole parameters
M = 1.0    # cart mass [kg]
m = 0.1    # pendulum mass [kg]
l = 0.5    # pendulum length to CoM [m]
g = 9.81   # gravity [m/s^2]

# Linearized cart-pole model around upright equilibrium (theta=0)
# State: [x, x_dot, theta, theta_dot]
A = np.array([
    [0,   1,            0,       0],
    [0,   0,     -m*g/M,         0],
    [0,   0,            0,       1],
    [0,   0, (M+m)*g/(M*l),      0],
])

B = np.array([
    [0],
    [1/M],
    [0],
    [-1/(M*l)],
])

MODEL = {'A': [A], 'B': [B]}
N_STATES = A.shape[0]
N_INPUTS = B.shape[1]
TIME_HORIZON = 5.0

# Reference tracking via coordinate shift:
# x_target = [5, 0, 0, 0], so x0_shifted = [0,0,0,0] - x_target = [-5,0,0,0]
# Driving x_shifted -> 0 means x_actual -> x_target.
X0 = np.array([-5.0, 0.0, 0.0, 0.0])

Q = np.diag([1.0, 0.1, 100.0, 0.1])
R = 0.01 * np.eye(N_INPUTS)
P = np.array(solve_continuous_are(A, B, Q, R))

# Bounds (in shifted coordinates)
STATES_LB = np.array([-15.0, -10.0, -1.0, -2.0])
STATES_UB = np.array([ 10.0,  10.0,  1.0,  2.0])
CONTROL_LB = np.array([-10.0])
CONTROL_UB = np.array([10.0])


def _build_and_solve(args, inspect):
    integrator = args.integrator
    multiple_shooting = args.shooting == 'ms'
    hybrid = args.hybrid
    n_steps = args.n_steps
    plot = args.plot

    start = time.time()

    swi_lin_mpc = SwitchedLinearMPC(
        MODEL,
        n_steps,
        TIME_HORIZON,
        auto=False,
        x0=X0,
        multiple_shooting=multiple_shooting,
        propagation=integrator,
        inspect=inspect,
        hybrid=hybrid,
        plot=plot,
    )

    swi_lin_mpc.precompute_matrices(X0, Q, R, P)

    precompute_time = time.time() - start
    start = time.time()

    swi_lin_mpc.set_bounds(CONTROL_LB, CONTROL_UB, STATES_LB, STATES_UB)

    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints()

    swi_lin_mpc.set_cost_function(Q, R, X0, P)

    # LQR initial guess
    exp_dist = 1.0**np.arange(n_steps)
    phase_durations = exp_dist * TIME_HORIZON / np.sum(exp_dist)

    K_lqr = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    xk = X0.copy()
    x_init = [xk.flatten()]
    u_init = []
    for i in range(n_steps):
        uk = ca.DM(-K_lqr @ xk)
        xk = swi_lin_mpc.autonomous_evol[i](phase_durations[i]) @ xk + swi_lin_mpc.forced_evol[i](uk, phase_durations[i])
        x_init.append(xk.full().flatten())
        u_init.append(uk)
    x_init = np.array(x_init).reshape((n_steps+1, N_STATES))
    u_init = np.array(u_init).reshape((n_steps, N_INPUTS))

    swi_lin_mpc.set_initial_guess(
        X0,
        initial_state_trajectory=x_init,
        initial_control_inputs=u_init,
        initial_phases_duration=phase_durations,
    )

    swi_lin_mpc.create_solver('ipopt')

    setup_time = time.time() - start
    start = time.time()

    if multiple_shooting:
        inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve(x0=X0)
    else:
        inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve()

    solving_time = time.time() - start
    label = "Uniform" if inspect else "Non-uniform"
    print(f"[{label}, N={n_steps}] precompute={precompute_time:.3f}s  setup={setup_time:.3f}s  solve={solving_time:.3f}s")

    return swi_lin_mpc, swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt


def test_cartpole_non_uniform(args):
    return _build_and_solve(args, inspect=False)


def test_cartpole_uniform(args):
    return _build_and_solve(args, inspect=True)


def plot_comparison(non_uniform_solution, uniform_solutions, n_steps_list, plot_mode='display'):
    _, nu_cost, nu_states, nu_inputs, nu_deltas = non_uniform_solution

    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_states_array = np.array(nu_states).reshape(-1, N_STATES)
    nu_inputs_array = np.array(nu_inputs).reshape(-1, N_INPUTS)

    state_labels = ['x (cart pos)', r'$\dot{x}$ (cart vel)', r'$\theta$ (angle)', r'$\dot{\theta}$ (ang vel)']

    total_plots = 1 + N_STATES + 1 + 1  # cost + states + input + time distribution
    ncols = 2
    nrows = int(np.ceil(total_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    axis_idx = 0

    # Select a few uniform solutions for trajectory comparison
    selected_indices = []
    if len(uniform_solutions) > 0:
        selected_indices = [0, len(uniform_solutions) // 3, 2 * len(uniform_solutions) // 3, -1]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, len(selected_indices))))

    # Cost comparison
    ax = axes[axis_idx]; axis_idx += 1
    if len(uniform_solutions) > 0:
        costs = [sol[1].item() for sol in uniform_solutions]
        ax.plot(n_steps_list, costs, 'o-', label='Uniform discretization', linewidth=2, markersize=6)
    ax.axhline(y=nu_cost.item(), color='r', linestyle='--', linewidth=2, label='Non-uniform (optimized)')
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Optimal cost')
    ax.set_title('Cost Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # State trajectories
    for si in range(N_STATES):
        ax = axes[axis_idx]; axis_idx += 1
        ax.plot(nu_times, nu_states_array[:, si], 'r-', linewidth=2.5, label='Non-uniform', zorder=10)
        for idx, color in zip(selected_indices, colors):
            if idx < len(uniform_solutions):
                _, _, states, _, deltas = uniform_solutions[idx]
                times = np.concatenate([[0], np.cumsum(deltas)])
                states_array = np.array(states).reshape(-1, N_STATES)
                ax.plot(times, states_array[:, si], '--', color=color, linewidth=1.5,
                        label=f'Uniform (N={n_steps_list[idx]})', alpha=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(state_labels[si])
        ax.set_title(f'State: {state_labels[si]}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Control input
    ax = axes[axis_idx]; axis_idx += 1
    ax.step(nu_times[:-1], nu_inputs_array[:, 0], where='post', linewidth=2.5, color='r', label='Non-uniform')
    for idx, color in zip(selected_indices, colors):
        if idx < len(uniform_solutions):
            _, _, _, inputs, deltas = uniform_solutions[idx]
            times = np.concatenate([[0], np.cumsum(deltas)])
            inputs_array = np.array(inputs).reshape(-1, N_INPUTS)
            ax.step(times[:-1], inputs_array[:, 0], where='post', linestyle='--', color=color,
                    linewidth=1.5, alpha=0.8, label=f'Uniform (N={n_steps_list[idx]})')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.set_title('Control Input')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Time distribution
    ax = axes[axis_idx]; axis_idx += 1
    ax.bar(range(len(nu_deltas)), nu_deltas, color='red', alpha=0.7, label='Non-uniform')
    if len(uniform_solutions) > 0:
        ax.axhline(y=TIME_HORIZON / n_steps_list[-1], color='steelblue', linestyle='--',
                    linewidth=2, label=f'Uniform (T/N)')
    ax.set_xlabel('Phase index')
    ax.set_ylabel('Phase duration [s]')
    ax.set_title('Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Hide unused axes
    for idx in range(axis_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if plot_mode == 'save':
        plt.savefig('cartpole_comparison.png', dpi=300, bbox_inches='tight')
    elif plot_mode == 'display':
        plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cart-pole optimal control example')
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
        type=int, metavar="int", default=40, required=False,
        help='Number of steps.'
    )
    parser.add_argument('--plot',
        type=str, metavar="{display, save, none}", default="display", required=False,
        help='How to plot the results.'
    )
    args = parser.parse_args()
    if args.hybrid in ('False', 'false', '0'):
        args.hybrid = False

    # Non-uniform (optimized phase durations)
    non_uniform_N = args.n_steps
    non_uniform_solution = test_cartpole_non_uniform(args)

    # Plot the non-uniform solution (states, input, phase markers)
    mpc, _, states_opt, inputs_opt, deltas_opt = non_uniform_solution
    if mpc.multiple_shooting:
        mpc.plot_optimal_solution(deltas_opt, inputs_opt, states_opt)
    else:
        mpc.plot_optimal_solution(deltas_opt, inputs_opt)

    # Uniform (fixed phase durations) for a range of N
    uniform_solutions = []
    n_steps_list = []
    for n_steps in range(20, 100, 10):
        args.n_steps = n_steps
        solution = test_cartpole_uniform(args)
        uniform_solutions.append(solution)
        n_steps_list.append(n_steps)

    # Summary
    non_uniform_opt_cost = non_uniform_solution[1]
    print("\n" + "="*50)
    print("Results Summary")
    print("="*50)
    print(f"Non-uniform optimal cost (N={non_uniform_N}): {non_uniform_opt_cost.item():.6f}")
    print("="*50)
    print(f"{'N Steps':<15} {'Optimal Cost':<15} {'Difference %':<15}")
    print("-"*50)
    for n_steps, sol in zip(n_steps_list, uniform_solutions):
        cost = sol[1].item()
        diff_percent = 100 * (cost - non_uniform_opt_cost.item()) / (non_uniform_opt_cost.item() + 1e-8)
        print(f"{n_steps:<15} {cost:<15.6f} {diff_percent:+.3f}%")
    print("="*50 + "\n")

    # Plot comparison
    plot_comparison(non_uniform_solution, uniform_solutions, n_steps_list, plot_mode=args.plot)
    print("All tests passed!")
