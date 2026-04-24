from ocslc.corn import print_corn

print_corn()
import os
import time

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

from ocslc.utils.disp import (
    init_matplotlib,
    plot_comparison_dashboard,
    plot_optimal_cost,
    plot_computational_cost,
    plot_input_standalone,
    plot_states_standalone,
    plot_pareto_front,
)
from ocslc.switched_linear_mpc import SwitchedLinearMPC

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "cartpole"
)

init_matplotlib()

# Cart-pole parameters
M = 1.0  # cart mass [kg]
m = 0.1  # pendulum mass [kg]
l = 0.5  # pendulum length to CoM [m]
g = 9.81  # gravity [m/s^2]

# Linearized cart-pole model around upright equilibrium (theta=0)
# State: [x, x_dot, theta, theta_dot]
A = np.array(
    [
        [0, 1, 0, 0],
        [0, 0, -m * g / M, 0],
        [0, 0, 0, 1],
        [0, 0, (M + m) * g / (M * l), 0],
    ]
)

B = np.array(
    [
        [0],
        [1 / M],
        [0],
        [-1 / (M * l)],
    ]
)

MODEL = {"A": [A], "B": [B]}
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
STATES_LB = np.array([-15.0, -2.0, -0.3, -2.0])
STATES_UB = np.array([10.0, 2.0, 0.3, 2.0])
CONTROL_LB = np.array([-10.0])
CONTROL_UB = np.array([10.0])


def _build_and_solve(args, inspect):
    integrator = args.integrator
    multiple_shooting = args.shooting == "ms"
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
    exp_dist = 1.0 ** np.arange(n_steps)
    phase_durations = exp_dist * TIME_HORIZON / np.sum(exp_dist)

    K_lqr = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    xk = X0.copy()
    x_init = [xk.flatten()]
    u_init = []
    for i in range(n_steps):
        uk = ca.DM(-K_lqr @ xk)
        xk = swi_lin_mpc.autonomous_evol[i](
            phase_durations[i]
        ) @ xk + swi_lin_mpc.forced_evol[i](uk, phase_durations[i])
        x_init.append(xk.full().flatten())
        u_init.append(uk)
    x_init = np.array(x_init).reshape((n_steps + 1, N_STATES))
    u_init = np.array(u_init).reshape((n_steps, N_INPUTS))

    swi_lin_mpc.set_initial_guess(
        X0,
        initial_state_trajectory=x_init,
        initial_control_inputs=u_init,
        initial_phases_duration=phase_durations,
    )

    swi_lin_mpc.create_solver("ipopt")

    setup_time = time.time() - start
    start = time.time()

    if multiple_shooting:
        inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve(x0=X0)
    else:
        inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve()

    solving_time = time.time() - start
    label = "Uniform" if inspect else "Non-uniform"
    print(
        f"[{label}, N={n_steps}] precompute={precompute_time:.3f}s  setup={setup_time:.3f}s  solve={solving_time:.3f}s"
    )

    timing = {
        "precompute": precompute_time,
        "setup": setup_time,
        "solve": solving_time,
        "total": precompute_time + setup_time + solving_time,
    }
    return swi_lin_mpc, swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt, timing


def test_cartpole_non_uniform(args):
    return _build_and_solve(args, inspect=False)


def test_cartpole_uniform(args):
    return _build_and_solve(args, inspect=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cart-pole optimal control example")
    parser.add_argument(
        "--integrator",
        type=str,
        metavar="{int, exp}",
        default="exp",
        required=False,
        help="Integration method to use. Default is exp.",
    )
    parser.add_argument(
        "--shooting",
        type=str,
        metavar="{ss, ms}",
        default="ms",
        required=False,
        help="Shooting method. Default is ms.",
    )
    parser.add_argument(
        "--hybrid", type=str, default=False, required=False, help="Hybrid method."
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        metavar="int",
        default=40,
        required=False,
        help="Number of steps.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        metavar="{display, save, none}",
        default="display",
        required=False,
        help="How to plot the results.",
    )
    args = parser.parse_args()
    if args.hybrid in ("False", "false", "0"):
        args.hybrid = False

    # Reference non-uniform solution (N=200) treated as the absolute optimum
    args.n_steps = 200
    ref_200_solution = test_cartpole_non_uniform(args)
    reference_cost = float(np.asarray(ref_200_solution[1]).flat[0])

    # Non-uniform (optimized phase durations) for a range of N
    non_uniform_solutions = []
    non_uniform_n_steps_list = [11, 20, 30, 40, 50]
    for n_steps in non_uniform_n_steps_list:
        args.n_steps = n_steps
        solution = test_cartpole_non_uniform(args)
        non_uniform_solutions.append(solution)

    # Reference non-uniform solution (N=40) for trajectory plots
    ref_idx = non_uniform_n_steps_list.index(40)
    non_uniform_ref = non_uniform_solutions[ref_idx]

    # Uniform (fixed phase durations) for a range of N
    uniform_solutions = []
    uniform_n_steps_list = list(range(20, 100, 10))
    for n_steps in uniform_n_steps_list:
        args.n_steps = n_steps
        solution = test_cartpole_uniform(args)
        uniform_solutions.append(solution)

    # Summary
    nu_ref_cost = non_uniform_ref[1].item()
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for n_steps, sol in zip(non_uniform_n_steps_list, non_uniform_solutions):
        print(f"Non-uniform (N={n_steps}): cost={sol[1].item():.6f}")
    print("=" * 60)
    print(f"{'N Steps':<15} {'Uniform Cost':<15} {'vs Non-uniform N=40':<20}")
    print("-" * 60)
    for n_steps, sol in zip(uniform_n_steps_list, uniform_solutions):
        cost = sol[1].item()
        diff_percent = 100 * (cost - nu_ref_cost) / (nu_ref_cost + 1e-8)
        print(f"{n_steps:<15} {cost:<15.6f} {diff_percent:+.3f}%")
    print("=" * 60 + "\n")

    state_labels = [r"$x$", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
    input_labels = [r"$F$ [N]"]

    fig = plot_comparison_dashboard(
        non_uniform_ref,
        uniform_solutions,
        uniform_n_steps_list,
        n_states=N_STATES,
        n_inputs=N_INPUTS,
        state_labels=state_labels,
        input_labels=input_labels,
        states_lb=STATES_LB,
        states_ub=STATES_UB,
        non_uniform_solutions=non_uniform_solutions,
        non_uniform_n_steps_list=non_uniform_n_steps_list,
    )
    if args.plot == "save":
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULTS_DIR, "dashboard.pdf"), bbox_inches="tight")

        fig_cost = plot_optimal_cost(
            non_uniform_solutions,
            non_uniform_n_steps_list,
            uniform_solutions,
            uniform_n_steps_list,
        )
        fig_cost.savefig(
            os.path.join(RESULTS_DIR, "optimal_cost.pdf"), bbox_inches="tight"
        )

        fig_comp = plot_computational_cost(
            non_uniform_solutions,
            non_uniform_n_steps_list,
            uniform_solutions,
            uniform_n_steps_list,
        )
        fig_comp.savefig(
            os.path.join(RESULTS_DIR, "computational_cost.pdf"), bbox_inches="tight"
        )

        fig_pareto = plot_pareto_front(
            non_uniform_solutions,
            non_uniform_n_steps_list,
            uniform_solutions,
            uniform_n_steps_list,
            reference_cost=reference_cost,
        )
        fig_pareto.savefig(
            os.path.join(RESULTS_DIR, "pareto_front.pdf"), bbox_inches="tight"
        )

        fig_input = plot_input_standalone(
            non_uniform_ref,
            uniform_solutions,
            uniform_n_steps_list,
            n_inputs=N_INPUTS,
            input_labels=input_labels,
            zoom_xlim=(0.0, 0.5),
            zoom_loc=[0.24, 0.25, 0.44, 0.55],
        )
        fig_input.savefig(os.path.join(RESULTS_DIR, "input.pdf"), bbox_inches="tight")

        fig_states = plot_states_standalone(
            non_uniform_ref,
            n_states=N_STATES,
            state_labels=state_labels,
            states_lb=STATES_LB,
            states_ub=STATES_UB,
            state_pairs=[(0, 1), (2, 3)],
        )
        fig_states.savefig(os.path.join(RESULTS_DIR, "states.pdf"), bbox_inches="tight")
    elif args.plot == "display":
        plt.show()
    print("All tests passed!")
