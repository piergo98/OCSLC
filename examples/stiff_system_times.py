"""Stiff system: solve time vs (n_states, n_steps) sweep (non-uniform only)."""
import argparse
import contextlib
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are
from tqdm import tqdm

from ocslc.corn import print_corn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stiff_system  # noqa: E402
from stiff_system import _build_and_solve  # noqa: E402

from ocslc.switched_linear_mpc import SwitchedLinearMPC  # noqa: E402
from ocslc.utils.disp import (  # noqa: E402
    init_matplotlib,
    _ieee_rc_context,
    get_colors,
)

# Force a constant 1500 iterations per solve, regardless of precision.
_FIXED_SOLVER_OPTS = {
    "max_iter": 1500,
    "tol": 1e-30,
    "acceptable_tol": 1e-30,
    "print_level": 0,
    "print_time": False,
}
_orig_create_solver = SwitchedLinearMPC.create_solver


def _create_solver_fixed_iters(self, solver="ipopt", **kwargs):
    kwargs.update(_FIXED_SOLVER_OPTS)
    with contextlib.redirect_stdout(io.StringIO()):
        return _orig_create_solver(self, solver, **kwargs)


SwitchedLinearMPC.create_solver = _create_solver_fixed_iters

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "w_constraints"
)
CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
CSV_FILENAME = "stiff_system_2_solve_times_sweep.csv"
PLOT_FILENAME = "total_time_vs_n_steps_sweep.pdf"

N_STATES_LIST = [5, 10, 15, 20, 25, 30]
N_STEPS_LIST = list(range(20, 201, 20))

# Stiff-system parameters: keep |lambda_max| / |lambda_min| constant across n_states.
# Eigenvalues are placed geometrically, so each is k = STIFFNESS_RATIO**(1/(n-1))
# times smaller than the previous one.
STIFFNESS_RATIO = 1000.0
LAMBDA_MAX = -10.0

init_matplotlib()


def configure_stiff_system(n_states):
    """Override stiff_system module globals with an n_states-dimensional stiff system."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2 for a stiff system")

    magnitudes = np.geomspace(abs(LAMBDA_MAX), abs(LAMBDA_MAX) / STIFFNESS_RATIO, n_states)
    eigenvalues = -magnitudes
    a_mat = np.diag(eigenvalues)
    b_mat = np.ones((n_states, 1))
    n_inputs = b_mat.shape[1]

    x0 = -np.ones(n_states)
    q_mat = 1.0 * np.eye(n_states)
    r_mat = 0.01 * np.eye(n_inputs)
    p_mat = np.array(solve_continuous_are(a_mat, b_mat, q_mat, r_mat))

    states_lb = -100.0 * np.ones(n_states)
    states_ub = 0.2 * np.ones(n_states)

    stiff_system.A = a_mat
    stiff_system.B = b_mat
    stiff_system.MODEL = {"A": [a_mat], "B": [b_mat]}
    stiff_system.N_STATES = n_states
    stiff_system.N_INPUTS = n_inputs
    stiff_system.X0 = x0
    stiff_system.Q = q_mat
    stiff_system.R = r_mat
    stiff_system.P = p_mat
    stiff_system.STATES_LB = states_lb
    stiff_system.STATES_UB = states_ub

    k = STIFFNESS_RATIO ** (1.0 / (n_states - 1))
    print(
        f"Stiff system: n_states={n_states}, stiffness ratio={STIFFNESS_RATIO:g}, "
        f"k={k:.4f}, eigenvalues={eigenvalues}"
    )


def _plot_sweep(df, *, figsize=(3.8, 2.8)):
    with _ieee_rc_context():
        figure, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        colors = get_colors()
        for i, n_states in enumerate(sorted(df["n_states"].unique())):
            sub = df[df["n_states"] == n_states].sort_values("n_steps")
            ax.plot(
                sub["n_steps"],
                sub["total"],
                "s-",
                color=colors[i % len(colors)],
                linewidth=1.5,
                markersize=4,
                label=f"$n_s={int(n_states)}$",
            )
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Total time [s]")
        ax.legend(fontsize=8, ncol=2)
    return figure


def _run_sweep(args):
    rows = []
    pbar = tqdm(
        total=len(N_STATES_LIST) * len(N_STEPS_LIST),
        desc="Sweeping (n_s, n_c)",
        unit="run",
    )
    for n_states in N_STATES_LIST:
        configure_stiff_system(n_states)
        for n_steps in N_STEPS_LIST:
            args.n_steps = n_steps
            solution = _build_and_solve(args, inspect=False, warm_start=False)
            timing = solution[5]
            rows.append({
                "n_states": n_states,
                "n_steps": n_steps,
                "precompute": timing["precompute"],
                "setup": timing["setup"],
                "solve": timing["solve"],
                "total": timing["total"],
            })
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stiff system: solve time vs (n_states, n_steps) sweep (non-uniform only)"
    )
    parser.add_argument("--integrator", type=str, default="exp")
    parser.add_argument("--shooting", type=str, default="ms")
    parser.add_argument("--hybrid", type=str, default=False)
    parser.add_argument(
        "--plot",
        type=str,
        metavar="{display, save, none}",
        default="display",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip solving and re-plot from the saved CSV.",
    )
    args = parser.parse_args()
    if args.hybrid in ("False", "false", "0"):
        args.hybrid = False
    args.n_steps = 0  # overwritten in the sweep below

    os.makedirs(CSV_DIR, exist_ok=True)
    csv_path = os.path.join(CSV_DIR, CSV_FILENAME)

    if args.plot_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"--plot-only requires an existing CSV at {csv_path}; "
                "run without --plot-only first."
            )
        df = pd.read_csv(csv_path)
    else:
        print_corn()
        df = _run_sweep(args)
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 78)
        print("Total time vs (n_states, n_steps) (non-uniform, cold start)")
        print("=" * 78)
        print(
            f"{'n_states':<10} {'n_steps':<10} {'Precompute':<12} "
            f"{'Setup':<12} {'Solve':<12} {'Total':<12}"
        )
        print("-" * 78)
        for _, row in df.iterrows():
            print(
                f"{int(row['n_states']):<10} {int(row['n_steps']):<10} "
                f"{row['precompute']:<12.4f} {row['setup']:<12.4f} "
                f"{row['solve']:<12.4f} {row['total']:<12.4f}"
            )
        print("=" * 78 + "\n")

    fig = _plot_sweep(df)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, PLOT_FILENAME), bbox_inches="tight")
    if args.plot == "display":
        plt.show()
