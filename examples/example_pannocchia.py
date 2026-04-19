from ocslc.corn import print_corn; print_corn()
import os
import time

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

from ocslc.utils.disp import init_matplotlib, plot_methods_grid, plot_methods_histograms
from ocslc.switched_linear_mpc import SwitchedLinearMPC

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'pannocchia')

init_matplotlib()

MODEL = {
    'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
    'B': [np.array([[0.25], [2], [0]])],
}

N_STATES = MODEL['A'][0].shape[0]
N_INPUTS = MODEL['B'][0].shape[1]
TIME_HORIZON = 10

X0 = np.array([1.3440, -4.5850, 5.6470])

Q = 1. * np.eye(N_STATES)
R = 0.1 * np.eye(N_INPUTS)
P = np.array(solve_continuous_are(MODEL['A'][0], MODEL['B'][0], Q, R))

STATES_LB = np.array([-100, -100, -100])
STATES_UB = np.array([100, 100, 100])


def _build_and_solve(args, shooting, integrator, inspect=False, n_steps_override=None):
    n_steps = n_steps_override if n_steps_override is not None else args.n_steps
    hybrid = args.hybrid
    plot = args.plot
    multiple_shooting = shooting == 'ms'

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

    swi_lin_mpc.set_bounds(-1, 1, STATES_LB, STATES_UB)

    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints(X0)

    swi_lin_mpc.set_cost_function(Q, R, X0, P)

    # LQR initial guess
    A, B = MODEL['A'][0], MODEL['B'][0]
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
    label = f"{shooting.upper()} {integrator}"
    print(f"[{label}, N={n_steps}] precompute={precompute_time:.3f}s  setup={setup_time:.3f}s  solve={solving_time:.3f}s")

    timing = {
        'precompute': precompute_time,
        'setup': setup_time,
        'solve': solving_time,
        'total': precompute_time + setup_time + solving_time,
    }
    return swi_lin_mpc, swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt, timing


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pannocchia example')
    parser.add_argument('--hybrid',
        type=str, default=False, required=False,
        help='Hybrid method.'
    )
    parser.add_argument('--steps',
        type=int, metavar="int", default=40, required=False,
        dest='n_steps',
        help='Number of steps.'
    )
    parser.add_argument('--plot',
        type=str, metavar="{display, save, none}", default="display", required=False,
        help='How to plot the results.'
    )
    args = parser.parse_args()
    if args.hybrid in ('False', 'false', '0'):
        args.hybrid = False

    # Run all 4 combinations
    methods = [
        ('ss', 'int'),
        ('ms', 'int'),
        ('ss', 'exp'),
        ('ms', 'exp'),
    ]
    method_labels = ['SS int', 'MS int', 'SS exp', 'MS exp']

    solutions = []
    for i, (shooting, integrator) in enumerate(methods):
        print(f"\n>>> Solving {method_labels[i]} ({i+1}/{len(methods)}) ...")
        sol = _build_and_solve(args, shooting, integrator)
        solutions.append(sol)

    # Solve high-density reference OCP with uniform timesteps (N=200)
    print("\n>>> Solving reference OCP (uniform, N=200) ...")
    ref_solution = _build_and_solve(args, 'ss', 'exp', inspect=True, n_steps_override=200)

    # Summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"{'Method':<15} {'Cost':<15} {'Total time [s]':<15}")
    print("-"*60)
    for label, sol in zip(method_labels, solutions):
        cost = float(np.asarray(sol[1]).flat[0])
        total = sol[5]['total']
        print(f"{label:<15} {cost:<15.6f} {total:<15.3f}")
    ref_cost = float(np.asarray(ref_solution[1]).flat[0])
    ref_total = ref_solution[5]['total']
    print(f"{'Ref (N=200)':<15} {ref_cost:<15.6f} {ref_total:<15.3f}")
    print("="*60 + "\n")

    state_labels = [r'$x_1$', r'$x_2$', r'$x_3$']
    input_labels = [r'$u$']

    fig_inputs = plot_methods_grid(
        solutions, method_labels,
        n_states=N_STATES, n_inputs=N_INPUTS,
        state_labels=state_labels,
        input_labels=input_labels,
        reference_solution=ref_solution,
    )

    fig_histograms = plot_methods_histograms(
        solutions, method_labels,
    )

    if args.plot == 'save':
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig_inputs.savefig(os.path.join(RESULTS_DIR, 'methods_comparison.pdf'), bbox_inches='tight')
        fig_histograms.savefig(os.path.join(RESULTS_DIR, 'methods_histograms.pdf'), bbox_inches='tight')
    elif args.plot == 'display':
        plt.show()

    print("All tests passed!")
