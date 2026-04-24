import math
from contextlib import contextmanager

from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

# From https://github.com/ddebenedittis/differentiable_motion_planning


def init_matplotlib(palette: str = "matlab"):
    colors = get_colors(palette)

    lines = ["-", "--", "-", "--", "-", "--", "-", "--", "-", "--"]
    lines = lines[: len(colors)]

    default_cycler = cycler(color=colors) + cycler("linestyle", lines)

    colors = list(default_cycler.by_key()["color"])

    textsize = 12
    labelsize = 12

    plt.rc("font", family="serif", serif="Times")
    plt.rc("text", usetex=True)
    plt.rc("xtick", labelsize=textsize)
    plt.rc("ytick", labelsize=textsize)
    plt.rc("axes", labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc("legend", fontsize=textsize)

    plt.rc("axes", grid=True, xmargin=0)
    plt.rc("grid", linestyle="dotted", linewidth=0.25)

    plt.rcParams["figure.constrained_layout.use"] = True


def get_colors(palette: str = "matlab"):
    """Get a list of colors based on the specified palette."""
    if palette == "okabe-ito":
        return [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#000000",
        ]
    if palette == "matlab":
        return [
            "#0072BD",
            "#D95319",
            "#EDB120",
            "#7E2F8E",
            "#77AC30",
            "#4DBEEE",
            "#A2142F",
        ]
    if palette == "colorbrewer_1":
        return [
            "#E41A1C",
            "#377EB8",
            "#4DAF4A",
            "#984EA3",
            "#FF7F00",
            "#FFFF33",
            "#A65628",
            "#F781BF",
            "#999999",
        ]

    raise ValueError(f"Unknown palette: {palette}")


def plot_zoh_input(
    ax, times, inputs, phase_lines=True, phase_line_kwargs=None, **step_kwargs
):
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

    label = step_kwargs.pop("label", None)
    lines = []
    for k in range(inputs_ext.shape[1]):
        kwargs = dict(step_kwargs)
        if label is not None and k == 0:
            kwargs["label"] = label
        (line,) = ax.step(times, inputs_ext[:, k], where="post", **kwargs)
        lines.append(line)

    if phase_lines:
        _pl_kw = {"linestyle": "--", "linewidth": 0.5, "color": "gray"}
        if phase_line_kwargs is not None:
            _pl_kw.update(phase_line_kwargs)
        for t in times:
            ax.axvline(x=t, **_pl_kw)

    return lines


@contextmanager
def _ieee_rc_context():
    """Temporarily override matplotlib rcParams for IEEE double-column figures."""
    overrides = {
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
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
        state_labels = [rf"$x_{{{i + 1}}}$" for i in range(n_states)]
    if input_labels is None:
        input_labels = [rf"$u_{{{i + 1}}}$" for i in range(n_inputs)]

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
        ax.plot(
            n_steps_list,
            costs,
            "o-",
            color=colors[1],
            linewidth=2,
            markersize=6,
            label="Uniform",
        )
    if non_uniform_solutions is not None and non_uniform_n_steps_list is not None:
        costs_nu = [float(np.asarray(sol[1]).flat[0]) for sol in non_uniform_solutions]
        ax.plot(
            non_uniform_n_steps_list,
            costs_nu,
            "s-",
            color=colors[0],
            linewidth=2,
            markersize=6,
            label="Non-uniform (N=40)",
        )
    else:
        ax.axhline(
            y=float(np.asarray(nu_cost).flat[0]),
            color=colors[0],
            linestyle="--",
            linewidth=2,
            label=f"Non-uniform (N={non_uniform_n})",
        )
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Optimal cost")
    ax.set_title("Optimal cost comparison")
    ax.legend()

    # ── Computational cost comparison ────────────────────────────────────
    ax = axes[panel]
    panel += 1
    if len(uniform_solutions) > 0:
        solve_times_u = [sol[5]["solve"] for sol in uniform_solutions]
        total_times_u = [sol[5]["total"] for sol in uniform_solutions]
        ax.plot(
            n_steps_list,
            total_times_u,
            "o-",
            color=colors[1],
            linewidth=2,
            markersize=6,
            label="Uniform (total)",
        )
        ax.plot(
            n_steps_list,
            solve_times_u,
            "o--",
            color=colors[1],
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
            label="Uniform (solve)",
        )
    if non_uniform_solutions is not None and non_uniform_n_steps_list is not None:
        solve_times_nu = [sol[5]["solve"] for sol in non_uniform_solutions]
        total_times_nu = [sol[5]["total"] for sol in non_uniform_solutions]
        ax.plot(
            non_uniform_n_steps_list,
            total_times_nu,
            "s-",
            color=colors[0],
            linewidth=2,
            markersize=6,
            label="Non-uniform (total)",
        )
        ax.plot(
            non_uniform_n_steps_list,
            solve_times_nu,
            "s--",
            color=colors[0],
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
            label="Non-uniform (solve)",
        )
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Time [s]")
    ax.set_title("Computational cost comparison")
    ax.legend(fontsize=9)

    # ── Time distribution ────────────────────────────────────────────────
    ax = axes[panel]
    panel += 1
    ax.bar(
        range(len(nu_deltas)),
        nu_deltas,
        color=colors[0],
        alpha=0.7,
        label="Non-uniform",
    )
    if len(uniform_solutions) > 0:
        ax.axhline(
            y=time_horizon / n_steps_list[-1],
            color=colors[1],
            linestyle="--",
            linewidth=2,
            label="Uniform (T/N)",
        )
    ax.set_xlabel("Phase index")
    ax.set_ylabel("Phase duration [s]")
    ax.set_title("Time distribution")
    ax.legend()

    # ── Input evolution ──────────────────────────────────────────────────
    for k in range(n_inputs):
        ax = axes[panel]
        panel += 1
        plot_zoh_input(
            ax,
            nu_times,
            nu_inputs_arr[:, k : k + 1],
            phase_lines=False,
            color=colors[0],
            linewidth=2.5,
            label=f"Non-uniform (N={non_uniform_n})",
            zorder=10,
        )
        for j, idx in enumerate(selected):
            _, _, _, u_inp, u_del, _ = uniform_solutions[idx]
            u_del = np.asarray(u_del).flatten()
            u_times = np.concatenate([[0], np.cumsum(u_del)])
            u_inp_arr = np.array(u_inp).reshape(-1, n_inputs)
            plot_zoh_input(
                ax,
                u_times,
                u_inp_arr[:, k : k + 1],
                phase_lines=False,
                color=colors[(j + 1) % len(colors)],
                alpha=0.7,
                linestyle="--",
                linewidth=1.5,
                label=f"Uniform (N={n_steps_list[idx]})",
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(input_labels[k])
        ax.set_title(f"Input: {input_labels[k]}" if n_inputs > 1 else "Control input")
        ax.legend(fontsize=9)

    # ── State evolution ──────────────────────────────────────────────────
    for si in range(n_states):
        ax = axes[panel]
        panel += 1
        ax.plot(
            nu_times,
            nu_states_arr[:, si],
            color=colors[0],
            linewidth=2.5,
            label=f"Non-uniform (N={non_uniform_n})",
            zorder=10,
        )
        for j, idx in enumerate(selected):
            _, _, u_st, _, u_del, _ = uniform_solutions[idx]
            u_del = np.asarray(u_del).flatten()
            u_times = np.concatenate([[0], np.cumsum(u_del)])
            u_st_arr = np.array(u_st).reshape(-1, n_states)
            ax.plot(
                u_times,
                u_st_arr[:, si],
                "--",
                color=colors[(j + 1) % len(colors)],
                linewidth=1.5,
                alpha=0.7,
                label=f"Uniform (N={n_steps_list[idx]})",
            )
        ylim = ax.get_ylim()
        if states_lb is not None and np.isfinite(states_lb[si]):
            ax.axhspan(ylim[0] - 1e3, states_lb[si], color="r", alpha=0.1)
        if states_ub is not None and np.isfinite(states_ub[si]):
            ax.axhspan(states_ub[si], ylim[1] + 1e3, color="r", alpha=0.1)
        ax.set_ylim(ylim)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(state_labels[si])
        ax.set_title(f"State: {state_labels[si]}")
        ax.legend(fontsize=9)

    # Hide unused axes
    for idx in range(panel, len(axes)):
        axes[idx].axis("off")

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
        ax.plot(
            non_uniform_n_steps_list,
            costs_nu,
            "s-",
            color=colors[0],
            linewidth=2,
            markersize=5,
            label="Non-uniform",
        )
        costs_u = [float(np.asarray(sol[1]).flat[0]) for sol in uniform_solutions]
        ax.plot(
            uniform_n_steps_list,
            costs_u,
            "o-",
            color=colors[1],
            linewidth=2,
            markersize=5,
            label="Uniform",
        )
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Optimal cost")
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
        solve_times_nu = [sol[5]["solve"] for sol in non_uniform_solutions]
        total_times_nu = [sol[5]["total"] for sol in non_uniform_solutions]
        ax.plot(
            non_uniform_n_steps_list,
            total_times_nu,
            "s-",
            color=colors[0],
            linewidth=2,
            markersize=5,
            label="Non-uniform (total)",
        )
        ax.plot(
            non_uniform_n_steps_list,
            solve_times_nu,
            "s--",
            color=colors[0],
            linewidth=1.5,
            markersize=4,
            alpha=0.7,
            label="Non-uniform (solve)",
        )
        solve_times_u = [sol[5]["solve"] for sol in uniform_solutions]
        total_times_u = [sol[5]["total"] for sol in uniform_solutions]
        ax.plot(
            uniform_n_steps_list,
            total_times_u,
            "o-",
            color=colors[1],
            linewidth=2,
            markersize=5,
            label="Uniform (total)",
        )
        ax.plot(
            uniform_n_steps_list,
            solve_times_u,
            "o--",
            color=colors[1],
            linewidth=1.5,
            markersize=4,
            alpha=0.7,
            label="Uniform (solve)",
        )
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Time [s]")
        ax.legend(fontsize=10)
    return fig


def _zoh_values_in_range(times, values, xlim):
    """Return ZOH signal values active within ``[xlim[0], xlim[1]]``."""
    t = np.asarray(times).flatten()
    v = np.asarray(values).flatten()
    result = []
    for i in range(len(v)):
        t_end = t[i + 1] if i + 1 < len(t) else t[-1]
        if t[i] < xlim[1] and t_end > xlim[0]:
            result.append(float(v[i]))
    return result


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
    zoom_xlim=None,
    zoom_loc=None,
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
    zoom_xlim : tuple of (float, float), optional
        When given, an inset axes is drawn on each input subplot
        showing the zoomed view of ``[zoom_xlim[0], zoom_xlim[1]]``
        with connector lines to the corresponding region.
    zoom_loc : list of float, optional
        Inset position as ``[x, y, width, height]`` in axes-relative
        coordinates.  Defaults to ``[0.45, 0.3, 0.5, 0.6]``.
    """
    if uniform_overlay_ns is None:
        uniform_overlay_ns = [40, 60, 80]
    if input_labels is None:
        input_labels = [rf"$u_{{{i + 1}}}$" for i in range(n_inputs)]
    if figsize is None:
        if zoom_xlim is not None:
            figsize = (7.5, 2.8 * n_inputs)
        else:
            figsize = (3.5, 2.8 * n_inputs)

    _, _, _, nu_inputs, nu_deltas, _ = non_uniform_solution
    nu_deltas = np.asarray(nu_deltas).flatten()
    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_inputs_arr = np.array(nu_inputs).reshape(-1, n_inputs)

    with _ieee_rc_context():
        fig, axes = plt.subplots(
            n_inputs, 1, figsize=figsize, constrained_layout=True, squeeze=False
        )
        colors = get_colors()
        for k in range(n_inputs):
            ax = axes[k, 0]
            plot_zoh_input(
                ax,
                nu_times,
                nu_inputs_arr[:, k : k + 1],
                phase_lines=True,
                phase_line_kwargs={"alpha": 0.25},
                color=colors[0],
                linewidth=2,
                label=f"Non-uniform (N={len(nu_deltas)})",
                zorder=10,
            )
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
                    plot_zoh_input(
                        ax,
                        u_times,
                        u_inp_arr[:, k : k + 1],
                        phase_lines=False,
                        color=colors[color_idx % len(colors)],
                        alpha=0.7,
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Uniform (N={target_n})",
                    )
                    color_idx += 1
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(input_labels[k])
            if zoom_xlim is not None:
                ax.legend(loc="upper right")
            else:
                ax.legend()

            if zoom_xlim is not None:
                _loc = zoom_loc if zoom_loc is not None else [0.38, 0.25, 0.55, 0.55]

                # Mask the parent plot explicitly before drawing the inset.
                # This is more reliable than relying on the inset axes patch
                # alone, especially in vector exports where draw ordering can
                # make the background appear not to cover the underlying lines.
                inset_bg = Rectangle(
                    (_loc[0], _loc[1]),
                    _loc[2],
                    _loc[3],
                    transform=ax.transAxes,
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    zorder=15,
                )
                ax.add_patch(inset_bg)

                axins = ax.inset_axes(_loc)
                axins.set_zorder(20)

                plot_zoh_input(
                    axins,
                    nu_times,
                    nu_inputs_arr[:, k : k + 1],
                    phase_lines=True,
                    phase_line_kwargs={"alpha": 0.25},
                    color=colors[0],
                    linewidth=2,
                    zorder=10,
                )
                axins.grid(False)

                all_y = _zoh_values_in_range(nu_times, nu_inputs_arr[:, k], zoom_xlim)

                zi = 1
                for target_n in uniform_overlay_ns:
                    if target_n in uniform_n_steps_list:
                        idx = uniform_n_steps_list.index(target_n)
                        _, _, _, u_inp, u_del, _ = uniform_solutions[idx]
                        u_del = np.asarray(u_del).flatten()
                        u_times = np.concatenate([[0], np.cumsum(u_del)])
                        u_inp_arr = np.array(u_inp).reshape(-1, n_inputs)
                        plot_zoh_input(
                            axins,
                            u_times,
                            u_inp_arr[:, k : k + 1],
                            phase_lines=False,
                            color=colors[zi % len(colors)],
                            alpha=0.7,
                            linestyle="--",
                            linewidth=1.5,
                        )
                        all_y.extend(
                            _zoh_values_in_range(u_times, u_inp_arr[:, k], zoom_xlim)
                        )
                        zi += 1

                axins.set_xlim(zoom_xlim)
                if all_y:
                    y_lo, y_hi = min(all_y), max(all_y)
                    pad = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.5
                    axins.set_ylim(y_lo - pad, y_hi + pad)
                else:
                    y_lo, y_hi = ax.get_ylim()

                # Keep the inset itself transparent and let the explicit mask
                # on the parent axes provide the 50% white background.
                axins.set_facecolor("none")
                for spine in axins.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.0)

                # Dashed rectangle on main axes around the zoomed region.
                # Use the main axes y-limits so the box frames the x-region
                # cleanly without overshooting the plot area.
                main_ylim = ax.get_ylim()
                source_rect = Rectangle(
                    (zoom_xlim[0], main_ylim[0]),
                    zoom_xlim[1] - zoom_xlim[0],
                    main_ylim[1] - main_ylim[0],
                    edgecolor="black",
                    facecolor="none",
                    linewidth=1.5,
                    linestyle="--",
                    zorder=5,
                    clip_on=False,
                )
                ax.add_patch(source_rect)

                # Connector lines: right side of source → left side of inset
                con_top = ConnectionPatch(
                    xyA=(zoom_xlim[1], main_ylim[1]),
                    xyB=(0, 1),
                    coordsA="data",
                    coordsB="axes fraction",
                    axesA=ax,
                    axesB=axins,
                    color="black",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=21,
                )
                con_bot = ConnectionPatch(
                    xyA=(zoom_xlim[1], main_ylim[0]),
                    xyB=(0, 0),
                    coordsA="data",
                    coordsB="axes fraction",
                    axesA=ax,
                    axesB=axins,
                    color="black",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=21,
                )
                fig.add_artist(con_top)
                fig.add_artist(con_bot)

    return fig


def _color_axis(ax, color, side="left"):
    """Color an axis spine, ticks, tick labels, and y-label."""
    ax.spines[side].set_color(color)
    ax.tick_params(axis="y", colors=color)
    ax.yaxis.label.set_color(color)


def _plot_bounds(ax, state_idx, states_lb, states_ub, color):
    """Draw dashed horizontal lines for bounds that fall within the y-limits."""
    ylim = ax.get_ylim()
    if states_lb is not None and np.isfinite(states_lb[state_idx]):
        if ylim[0] <= states_lb[state_idx] <= ylim[1]:
            ax.axhline(
                y=states_lb[state_idx],
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )
    if states_ub is not None and np.isfinite(states_ub[state_idx]):
        if ylim[0] <= states_ub[state_idx] <= ylim[1]:
            ax.axhline(
                y=states_ub[state_idx],
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )


def plot_states_standalone(
    non_uniform_solution,
    *,
    n_states,
    state_labels=None,
    states_lb=None,
    states_ub=None,
    figsize=None,
    state_pairs=None,
):
    """Standalone state trajectories for IEEE publication (no title).

    Parameters
    ----------
    state_pairs : list of tuple, optional
        Group states into subfigures with dual y-axes.  Each element is a
        tuple of one or two state indices: ``(left_idx,)`` or
        ``(left_idx, right_idx)``.  When provided, creates one column per
        pair with coloured left/right axes matching the plotted lines.
    """
    if state_labels is None:
        state_labels = [rf"$x_{{{i + 1}}}$" for i in range(n_states)]

    _, _, nu_states, _, nu_deltas, _ = non_uniform_solution
    nu_deltas = np.asarray(nu_deltas).flatten()
    nu_times = np.concatenate([[0], np.cumsum(nu_deltas)])
    nu_states_arr = np.array(nu_states).reshape(-1, n_states)

    with _ieee_rc_context():
        colors = get_colors()

        if state_pairs is not None:
            ncols = len(state_pairs)
            if figsize is None:
                figsize = (7.5, 2.4)
            fig, axes = plt.subplots(
                1, ncols, figsize=figsize, constrained_layout=True
            )
            if ncols == 1:
                axes = np.array([axes])

            color_left = colors[0]
            color_right = colors[1]

            for col, pair in enumerate(state_pairs):
                ax = axes[col]
                left_idx = pair[0]

                ax.plot(
                    nu_times,
                    nu_states_arr[:, left_idx],
                    color=color_left,
                    linewidth=2,
                )
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(state_labels[left_idx], labelpad=2)
                _color_axis(ax, color_left, "left")

                if len(pair) > 1:
                    right_idx = pair[1]

                    ax.spines["right"].set_visible(False)
                    ax2 = ax.twinx()
                    ax2.plot(
                        nu_times,
                        nu_states_arr[:, right_idx],
                        color=color_right,
                        linewidth=2,
                    )
                    ax2.set_ylabel(state_labels[right_idx], labelpad=2)
                    _color_axis(ax2, color_right, "right")
                    ax2.spines["left"].set_color(color_left)
                    ax2.grid(False)

                    _plot_bounds(ax, left_idx, states_lb, states_ub, color_left)
                    _plot_bounds(ax2, right_idx, states_lb, states_ub, color_right)
                else:
                    _plot_bounds(ax, left_idx, states_lb, states_ub, color_left)
        else:
            if figsize is None:
                figsize = (3.5, 2.0 * n_states)
            fig, axes = plt.subplots(
                n_states, 1, figsize=figsize, constrained_layout=True, squeeze=False
            )
            for si in range(n_states):
                ax = axes[si, 0]
                ax.plot(
                    nu_times, nu_states_arr[:, si], color=colors[0], linewidth=2
                )
                ylim = ax.get_ylim()
                if states_lb is not None and np.isfinite(states_lb[si]):
                    ax.axhspan(ylim[0] - 1e3, states_lb[si], color="r", alpha=0.1)
                if states_ub is not None and np.isfinite(states_ub[si]):
                    ax.axhspan(states_ub[si], ylim[1] + 1e3, color="r", alpha=0.1)
                ax.set_ylim(ylim)
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(state_labels[si])
    return fig


def plot_pareto_front(
    non_uniform_solutions,
    non_uniform_n_steps_list,
    uniform_solutions,
    uniform_n_steps_list,
    *,
    reference_cost,
    figsize=(3.5, 2.8),
    xlim=None,
    ylim=None,
):
    """Standalone Pareto front: cost sub-optimality vs total solving time.

    The y-axis shows ``cost - reference_cost`` on a log scale, where
    *reference_cost* is the optimal cost of a high-resolution
    non-uniform solve (e.g. N=200) treated as the absolute optimum.

    Parameters
    ----------
    reference_cost : float
        Optimal cost of the reference (N=200 non-uniform) solution.
    xlim, ylim : tuple of (float, float), optional
        Axis limits.
    """
    colors = get_colors()

    def _build(ns, sols):
        pts = [
            (n, sol[5]["total"], float(np.asarray(sol[1]).flat[0]) - reference_cost)
            for n, sol in zip(ns, sols)
        ]
        pts.sort(key=lambda x: x[1])
        return pts

    def _draw_on_ax(ax):
        for pts, color, marker, label in [
            (
                _build(non_uniform_n_steps_list, non_uniform_solutions),
                colors[0],
                "s",
                "Non-uniform",
            ),
            (
                _build(uniform_n_steps_list, uniform_solutions),
                colors[1],
                "o",
                "Uniform",
            ),
        ]:
            times = [p[1] for p in pts]
            costs = [p[2] for p in pts]
            ax.plot(
                times,
                costs,
                marker + "-",
                color=color,
                markersize=5,
                linewidth=2,
                label=label,
                zorder=5,
            )
            for n, t, c in pts:
                ax.annotate(
                    str(n),
                    (t, c),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=10,
                    color=color,
                )

    # Collect all data points to compute automatic limits
    all_pts = _build(non_uniform_n_steps_list, non_uniform_solutions) + _build(
        uniform_n_steps_list, uniform_solutions
    )
    all_times = [p[1] for p in all_pts]
    t_min, t_max = min(all_times), max(all_times)
    t_pad = 0.05 * (t_max - t_min)
    auto_xlim = (max(0, t_min - t_pad), t_max + t_pad)

    eff_xlim = xlim if xlim is not None else auto_xlim

    with _ieee_rc_context():
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        _draw_on_ax(ax)
        ax.set_yscale("log")
        ax.set_xlim(eff_xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Total time [s]")
        ax.set_ylabel(r"$J^* - J^*_{\mathrm{ref}}$")
        ax.legend()

    return fig


def plot_methods_grid(
    solutions,
    method_labels,
    *,
    n_states,
    n_inputs,
    state_labels=None,
    input_labels=None,
    figsize=None,
    reference_solution=None,
    reference_label="Reference ($N=200$)",
):
    """Plot a 2x2 grid of control inputs, one subplot per method.

    Parameters
    ----------
    solutions : list of tuple
        Four solutions ``(_, cost, states, inputs, deltas, timing)``.
    method_labels : list of str
        Labels for each of the four methods (e.g. ``['SS int', 'MS int', ...]``).
    n_states, n_inputs : int
        Dimensions of the system.
    state_labels : list of str, optional
        Per-state labels.  Defaults to ``$x_i$``.
    input_labels : list of str, optional
        Per-input labels.  Defaults to ``$u_i$``.
    figsize : tuple, optional
        Figure size.  Defaults to ``(10, 7)``.
    reference_solution : tuple, optional
        A high-density reference solution ``(_, cost, states, inputs, deltas, timing)``
        to overlay on each subplot.
    reference_label : str, optional
        Legend label for the reference solution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if state_labels is None:
        state_labels = [rf"$x_{{{i + 1}}}$" for i in range(n_states)]
    if input_labels is None:
        input_labels = [rf"$u_{{{i + 1}}}$" for i in range(n_inputs)]
    if figsize is None:
        figsize = (10, 7)

    colors = get_colors()

    # Extract reference solution data
    ref_times = None
    ref_inputs_arr = None
    if reference_solution is not None:
        _, _, _, ref_inputs, ref_deltas, _ = reference_solution
        ref_deltas = np.asarray(ref_deltas).flatten()
        ref_times = np.concatenate([[0], np.cumsum(ref_deltas)])
        ref_inputs_arr = np.array(ref_inputs).reshape(-1, n_inputs)

    overrides = {
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.labelsize": 22,
        "legend.fontsize": 14,
        "axes.titlesize": 22,
    }
    old = {k: plt.rcParams[k] for k in overrides}
    plt.rcParams.update(overrides)
    try:
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        for idx, (sol, label) in enumerate(zip(solutions, method_labels)):
            row, col = divmod(idx, 2)
            ax = axes[row, col]
            _, cost, _, inputs, deltas, _ = sol
            deltas = np.asarray(deltas).flatten()
            times = np.concatenate([[0], np.cumsum(deltas)])
            inputs_arr = np.array(inputs).reshape(-1, n_inputs)

            # Plot reference (behind)
            if ref_times is not None:
                for ki in range(n_inputs):
                    plot_zoh_input(
                        ax,
                        ref_times,
                        ref_inputs_arr[:, ki : ki + 1],
                        phase_lines=False,
                        color="gray",
                        alpha=0.5,
                        linewidth=1.0,
                        label=reference_label if ki == 0 else None,
                    )

            # Plot method's input
            for ki in range(n_inputs):
                plot_zoh_input(
                    ax,
                    times,
                    inputs_arr[:, ki : ki + 1],
                    phase_lines=True,
                    phase_line_kwargs={"alpha": 0.25},
                    color=colors[idx % len(colors)],
                    linewidth=1.5,
                    label=input_labels[ki],
                )
            ax.grid(False)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Input")
            cost_val = float(np.asarray(cost).flat[0])
            ax.set_title(rf"{label} ($J^*={cost_val:.4f}$)")
            ax.legend()
    finally:
        plt.rcParams.update(old)

    return fig


def plot_methods_histograms(
    solutions,
    method_labels,
    *,
    figsize=None,
    n_bins=20,
):
    """Plot a 2x2 grid of phase-duration histograms, one per method.

    Each subplot shows the frequency distribution of phase durations:
    x-axis is the duration, y-axis is the number of phases with that duration.

    Parameters
    ----------
    solutions : list of tuple
        Four solutions ``(_, cost, states, inputs, deltas, timing)``.
    method_labels : list of str
        Labels for each method.
    figsize : tuple, optional
        Figure size.  Defaults to ``(10, 7)``.
    n_bins : int, optional
        Number of histogram bins.  Defaults to 20.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if figsize is None:
        figsize = (10, 7)

    colors = get_colors()

    overrides = {
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.labelsize": 22,
        "legend.fontsize": 14,
        "axes.titlesize": 22,
    }
    old = {k: plt.rcParams[k] for k in overrides}
    plt.rcParams.update(overrides)
    try:
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        for idx, (sol, label) in enumerate(zip(solutions, method_labels)):
            row, col = divmod(idx, 2)
            ax = axes[row, col]
            _, _, _, _, deltas, _ = sol
            deltas = np.asarray(deltas).flatten()

            ax.hist(
                deltas,
                bins=n_bins,
                color=colors[idx % len(colors)],
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xlabel("Phase duration [s]")
            ax.set_ylabel("Number of samples")
            ax.set_title(label)
    finally:
        plt.rcParams.update(old)

    return fig
