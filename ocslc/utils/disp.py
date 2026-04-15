import math
from contextlib import contextmanager

from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

# From https://github.com/ddebenedittis/differentiable_motion_planning

def init_matplotlib(palette: str='matlab'):
    colors = get_colors(palette)

    lines = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']
    lines = lines[:len(colors)]
    
    default_cycler = (
        cycler(color=colors) +
        cycler('linestyle', lines)
    )

    colors = list(default_cycler.by_key()['color'])

    textsize = 12
    labelsize = 12

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)

    plt.rc("axes", grid=True, xmargin=0)
    plt.rc("grid", linestyle='dotted', linewidth=0.25)

    plt.rcParams['figure.constrained_layout.use'] = True
    
def get_colors(palette: str='matlab'):
    """Get a list of colors based on the specified palette."""
    if palette == 'okabe-ito':
        return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
    if palette == 'matlab':
        return ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    if palette == 'colorbrewer_1':
        return ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']

    raise ValueError(f"Unknown palette: {palette}")


def plot_zoh_input(ax, times, inputs, phase_lines=True, phase_line_kwargs=None, **step_kwargs):
    """Plot a piecewise-constant (zero-order hold) control input.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    times : array-like, shape (N+1,)
        Phase boundaries, including the initial time and the final horizon time.
    inputs : array-like, shape (N,) or (N, n_inputs)
        Control values held constant over ``[times[i], times[i+1]]``.
    phase_lines : bool, default True
        If True, draw vertical dashed lines at every phase boundary.
    phase_line_kwargs : dict, optional
        Override default style for phase boundary lines.
    **step_kwargs
        Forwarded to ``ax.step``. ``label`` is only applied to the first channel
        when ``inputs`` has multiple columns.

    Returns
    -------
    list of matplotlib.lines.Line2D
        One line per input channel.
    """
    times = np.asarray(times).flatten()
    inputs = np.asarray(inputs)
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)

    if times.shape[0] != inputs.shape[0] + 1:
        raise ValueError(
            f"Expected len(times) == len(inputs) + 1, got {times.shape[0]} and {inputs.shape[0]}."
        )

    # Repeat the last sample so the final ZOH segment is drawn up to times[-1].
    inputs_ext = np.vstack([inputs, inputs[-1:]])

    label = step_kwargs.pop('label', None)
    lines = []
    for k in range(inputs_ext.shape[1]):
        kwargs = dict(step_kwargs)
        if label is not None and k == 0:
            kwargs['label'] = label
        line, = ax.step(times, inputs_ext[:, k], where='post', **kwargs)
        lines.append(line)

    if phase_lines:
        _pl_kw = {'linestyle': '--', 'linewidth': 0.5, 'color': 'gray'}
        if phase_line_kwargs is not None:
            _pl_kw.update(phase_line_kwargs)
        for t in times:
            ax.axvline(x=t, **_pl_kw)

    return lines


@contextmanager
def _ieee_rc_context():
    """Temporarily override matplotlib rcParams for IEEE double-column figures."""
    overrides = {
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'legend.fontsize': 12,
    }
    old = {k: plt.rcParams[k] for k in overrides}
    plt.rcParams.update(overrides)
    try:
        yield
    finally:
        plt.rcParams.update(old)


def plot_comparison_dashboard(
    non_uniform_solution,
    uniform_solutions,
    n_steps_list,
    *,
    n_states,
    n_inputs,
    state_labels=None,
    input_labels=None,
    states_lb=None,
    states_ub=None,
    figsize=None,
    non_uniform_solutions=None,
    non_uniform_n_steps_list=None,
):
    """Plot a comparison dashboard of non-uniform vs uniform OC solutions.

    Parameters
    ----------
    non_uniform_solution : tuple
        ``(_, cost, states, inputs, deltas, timing)`` — first element is ignored.
        Used as the reference for trajectory plots.
    uniform_solutions : list of tuple
        Each element has the same shape as *non_uniform_solution*.
    n_steps_list : list of int
        Number of steps corresponding to each uniform solution.
    n_states, n_inputs : int
        Dimensions of the system.
    state_labels, input_labels : list of str, optional
        Subplot labels. Default to ``$x_i$`` / ``$u_i$``.
    states_lb, states_ub : array-like, shape (n_states,), optional
        Constraint bounds drawn on state subplots (only finite entries).
    figsize : tuple, optional
        Figure size. Defaults to ``(14, 3.5 * nrows)``.
    non_uniform_solutions : list of tuple, optional
        All non-uniform solutions for cost/timing comparison lines.
    non_uniform_n_steps_list : list of int, optional
        N values corresponding to *non_uniform_solutions*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, nu_cost, nu_states, nu_inputs, nu_deltas, _ = non_uniform_solution
    nu_deltas = np.asarray(nu_deltas).flatten()
    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_states_arr = np.array(nu_states).reshape(-1, n_states)
    nu_inputs_arr = np.array(nu_inputs).reshape(-1, n_inputs)
    time_horizon = nu_times[-1]
    non_uniform_n = len(nu_deltas)

    if state_labels is None:
        state_labels = [rf'$x_{{{i+1}}}$' for i in range(n_states)]
    if input_labels is None:
        input_labels = [rf'$u_{{{i+1}}}$' for i in range(n_inputs)]

    # Select a subset of uniform solutions for trajectory overlays
    selected = []
    if len(uniform_solutions) > 0:
        n = len(uniform_solutions)
        candidates = [0, n // 3, 2 * n // 3, n - 1]
        selected = list(dict.fromkeys(candidates))  # deduplicate, keep order

    colors = get_colors()

    total_panels = 3 + n_inputs + n_states
    ncols = 2
    nrows = math.ceil(total_panels / ncols)
    if figsize is None:
        figsize = (14, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    panel = 0

    # ── Optimal cost comparison ──────────────────────────────────────────
    ax = axes[panel]
    panel += 1
    if len(uniform_solutions) > 0:
        costs = [float(np.asarray(sol[1]).flat[0]) for sol in uniform_solutions]
        ax.plot(n_steps_list, costs, 'o-', color=colors[1], linewidth=2, markersize=6,
                label='Uniform')
    if non_uniform_solutions is not None and non_uniform_n_steps_list is not None:
        costs_nu = [float(np.asarray(sol[1]).flat[0]) for sol in non_uniform_solutions]
        ax.plot(non_uniform_n_steps_list, costs_nu, 's-', color=colors[0], linewidth=2,
                markersize=6, label='Non-uniform')
    else:
        ax.axhline(y=float(np.asarray(nu_cost).flat[0]), color=colors[0], linestyle='--', linewidth=2,
                   label=f'Non-uniform (N={non_uniform_n})')
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Optimal cost')
    ax.set_title('Optimal cost comparison')
    ax.legend()

    # ── Computational cost comparison ────────────────────────────────────
    ax = axes[panel]
    panel += 1
    if len(uniform_solutions) > 0:
        solve_times_u = [sol[5]['solve'] for sol in uniform_solutions]
        total_times_u = [sol[5]['total'] for sol in uniform_solutions]
        ax.plot(n_steps_list, total_times_u, 'o-', color=colors[1], linewidth=2,
                markersize=6, label='Uniform (total)')
        ax.plot(n_steps_list, solve_times_u, 'o--', color=colors[1], linewidth=1.5,
                markersize=5, alpha=0.7, label='Uniform (solve)')
    if non_uniform_solutions is not None and non_uniform_n_steps_list is not None:
        solve_times_nu = [sol[5]['solve'] for sol in non_uniform_solutions]
        total_times_nu = [sol[5]['total'] for sol in non_uniform_solutions]
        ax.plot(non_uniform_n_steps_list, total_times_nu, 's-', color=colors[0], linewidth=2,
                markersize=6, label='Non-uniform (total)')
        ax.plot(non_uniform_n_steps_list, solve_times_nu, 's--', color=colors[0], linewidth=1.5,
                markersize=5, alpha=0.7, label='Non-uniform (solve)')
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Time [s]')
    ax.set_title('Computational cost comparison')
    ax.legend(fontsize=9)

    # ── Time distribution ────────────────────────────────────────────────
    ax = axes[panel]
    panel += 1
    ax.bar(range(len(nu_deltas)), nu_deltas, color=colors[0], alpha=0.7,
           label='Non-uniform')
    if len(uniform_solutions) > 0:
        ax.axhline(y=time_horizon / n_steps_list[-1], color=colors[1],
                    linestyle='--', linewidth=2, label='Uniform (T/N)')
    ax.set_xlabel('Phase index')
    ax.set_ylabel('Phase duration [s]')
    ax.set_title('Time distribution')
    ax.legend()

    # ── Input evolution ──────────────────────────────────────────────────
    for k in range(n_inputs):
        ax = axes[panel]
        panel += 1
        plot_zoh_input(ax, nu_times, nu_inputs_arr[:, k:k+1],
                       phase_lines=False, color=colors[0], linewidth=2.5,
                       label='Non-uniform', zorder=10)
        for j, idx in enumerate(selected):
            _, _, _, u_inp, u_del, _ = uniform_solutions[idx]
            u_del = np.asarray(u_del).flatten()
            u_times = np.concatenate([[0], np.cumsum(u_del)])
            u_inp_arr = np.array(u_inp).reshape(-1, n_inputs)
            plot_zoh_input(ax, u_times, u_inp_arr[:, k:k+1],
                           phase_lines=False,
                           color=colors[(j + 1) % len(colors)],
                           alpha=0.7, linestyle='--', linewidth=1.5,
                           label=f'Uniform (N={n_steps_list[idx]})')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(input_labels[k])
        ax.set_title(f'Input: {input_labels[k]}' if n_inputs > 1 else 'Control input')
        ax.legend(fontsize=9)

    # ── State evolution ──────────────────────────────────────────────────
    for si in range(n_states):
        ax = axes[panel]
        panel += 1
        ax.plot(nu_times, nu_states_arr[:, si], color=colors[0], linewidth=2.5,
                label='Non-uniform', zorder=10)
        for j, idx in enumerate(selected):
            _, _, u_st, _, u_del, _ = uniform_solutions[idx]
            u_del = np.asarray(u_del).flatten()
            u_times = np.concatenate([[0], np.cumsum(u_del)])
            u_st_arr = np.array(u_st).reshape(-1, n_states)
            ax.plot(u_times, u_st_arr[:, si], '--',
                    color=colors[(j + 1) % len(colors)], linewidth=1.5,
                    alpha=0.7, label=f'Uniform (N={n_steps_list[idx]})')
        ylim = ax.get_ylim()
        if states_lb is not None and np.isfinite(states_lb[si]):
            ax.axhspan(ylim[0] - 1e3, states_lb[si], color='r', alpha=0.1)
        if states_ub is not None and np.isfinite(states_ub[si]):
            ax.axhspan(states_ub[si], ylim[1] + 1e3, color='r', alpha=0.1)
        ax.set_ylim(ylim)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(state_labels[si])
        ax.set_title(f'State: {state_labels[si]}')
        ax.legend(fontsize=9)

    # Hide unused axes
    for idx in range(panel, len(axes)):
        axes[idx].axis('off')

    return fig


def plot_optimal_cost(
    non_uniform_solutions,
    non_uniform_n_steps_list,
    uniform_solutions,
    uniform_n_steps_list,
    *,
    figsize=(3.5, 2.8),
):
    """Standalone optimal cost comparison for IEEE publication (no title)."""
    with _ieee_rc_context():
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        colors = get_colors()
        costs_nu = [float(np.asarray(sol[1]).flat[0]) for sol in non_uniform_solutions]
        ax.plot(non_uniform_n_steps_list, costs_nu, 's-', color=colors[0],
                linewidth=2, markersize=5, label='Non-uniform')
        costs_u = [float(np.asarray(sol[1]).flat[0]) for sol in uniform_solutions]
        ax.plot(uniform_n_steps_list, costs_u, 'o-', color=colors[1],
                linewidth=2, markersize=5, label='Uniform')
        ax.set_xlabel('Number of steps')
        ax.set_ylabel('Optimal cost')
        ax.legend()
    return fig


def plot_computational_cost(
    non_uniform_solutions,
    non_uniform_n_steps_list,
    uniform_solutions,
    uniform_n_steps_list,
    *,
    figsize=(3.5, 2.8),
):
    """Standalone computational cost comparison for IEEE publication (no title)."""
    with _ieee_rc_context():
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        colors = get_colors()
        solve_times_nu = [sol[5]['solve'] for sol in non_uniform_solutions]
        total_times_nu = [sol[5]['total'] for sol in non_uniform_solutions]
        ax.plot(non_uniform_n_steps_list, total_times_nu, 's-', color=colors[0],
                linewidth=2, markersize=5, label='Non-uniform (total)')
        ax.plot(non_uniform_n_steps_list, solve_times_nu, 's--', color=colors[0],
                linewidth=1.5, markersize=4, alpha=0.7, label='Non-uniform (solve)')
        solve_times_u = [sol[5]['solve'] for sol in uniform_solutions]
        total_times_u = [sol[5]['total'] for sol in uniform_solutions]
        ax.plot(uniform_n_steps_list, total_times_u, 'o-', color=colors[1],
                linewidth=2, markersize=5, label='Uniform (total)')
        ax.plot(uniform_n_steps_list, solve_times_u, 'o--', color=colors[1],
                linewidth=1.5, markersize=4, alpha=0.7, label='Uniform (solve)')
        ax.set_xlabel('Number of steps')
        ax.set_ylabel('Time [s]')
        ax.legend(fontsize=10)
    return fig


def plot_input_standalone(
    non_uniform_solution,
    uniform_solutions,
    uniform_n_steps_list,
    *,
    n_inputs,
    input_labels=None,
    uniform_overlay_ns=None,
    figsize=None,
    xlim=None,
):
    """Standalone ZOH input plot for IEEE publication (no title).

    Draws the non-uniform input with vertical dashed lines at timesteps
    (alpha=0.25) and overlays selected uniform solutions. When phase
    lines are drawn the default grid is disabled to avoid clutter.

    Parameters
    ----------
    xlim : tuple of (float, float), optional
        X-axis limits. Use to produce a zoomed-in view of the first
        portion of the horizon.
    """
    if uniform_overlay_ns is None:
        uniform_overlay_ns = [40, 60, 80]
    if input_labels is None:
        input_labels = [rf'$u_{{{i+1}}}$' for i in range(n_inputs)]
    if figsize is None:
        figsize = (3.5, 2.8 * n_inputs)

    _, _, _, nu_inputs, nu_deltas, _ = non_uniform_solution
    nu_deltas = np.asarray(nu_deltas).flatten()
    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_inputs_arr = np.array(nu_inputs).reshape(-1, n_inputs)

    with _ieee_rc_context():
        fig, axes = plt.subplots(n_inputs, 1, figsize=figsize,
                                  constrained_layout=True, squeeze=False)
        colors = get_colors()
        for k in range(n_inputs):
            ax = axes[k, 0]
            plot_zoh_input(ax, nu_times, nu_inputs_arr[:, k:k+1],
                           phase_lines=True,
                           phase_line_kwargs={'alpha': 0.25},
                           color=colors[0], linewidth=2,
                           label='Non-uniform', zorder=10)
            # Phase lines already provide vertical guides; suppress the
            # default grid so the figure is not visually cluttered.
            ax.grid(False)
            color_idx = 1
            for target_n in uniform_overlay_ns:
                if target_n in uniform_n_steps_list:
                    idx = uniform_n_steps_list.index(target_n)
                    _, _, _, u_inp, u_del, _ = uniform_solutions[idx]
                    u_del = np.asarray(u_del).flatten()
                    u_times = np.concatenate([[0], np.cumsum(u_del)])
                    u_inp_arr = np.array(u_inp).reshape(-1, n_inputs)
                    plot_zoh_input(ax, u_times, u_inp_arr[:, k:k+1],
                                   phase_lines=False,
                                   color=colors[color_idx % len(colors)],
                                   alpha=0.7, linestyle='--', linewidth=1.5,
                                   label=f'Uniform (N={target_n})')
                    color_idx += 1
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(input_labels[k])
            ax.legend()
    return fig


def plot_states_standalone(
    non_uniform_solution,
    *,
    n_states,
    state_labels=None,
    states_lb=None,
    states_ub=None,
    figsize=None,
):
    """Standalone state trajectories for IEEE publication (no title)."""
    if state_labels is None:
        state_labels = [rf'$x_{{{i+1}}}$' for i in range(n_states)]
    if figsize is None:
        figsize = (3.5, 2.0 * n_states)

    _, _, nu_states, _, nu_deltas, _ = non_uniform_solution
    nu_deltas = np.asarray(nu_deltas).flatten()
    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_states_arr = np.array(nu_states).reshape(-1, n_states)

    with _ieee_rc_context():
        colors = get_colors()
        fig, axes = plt.subplots(n_states, 1, figsize=figsize,
                                  constrained_layout=True, squeeze=False)
        for si in range(n_states):
            ax = axes[si, 0]
            ax.plot(nu_times, nu_states_arr[:, si], color=colors[0], linewidth=2)
            ylim = ax.get_ylim()
            if states_lb is not None and np.isfinite(states_lb[si]):
                ax.axhspan(ylim[0] - 1e3, states_lb[si], color='r', alpha=0.1)
            if states_ub is not None and np.isfinite(states_ub[si]):
                ax.axhspan(states_ub[si], ylim[1] + 1e3, color='r', alpha=0.1)
            ax.set_ylim(ylim)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(state_labels[si])
    return fig
