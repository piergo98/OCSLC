import time

import casadi as ca
import numpy as np
import scipy.io
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_non_autonomous_switched_linear_constrained(args):
    
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
        'A': [np.array([[-0.5, -4.0], [4.0, -0.5]])],
        'B': [np.array([[0.0], [2.0]])],
    }
    # print("--------------------------------")
    # print(np.linalg.eigvals(model['A'][0]))

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 5.0
    
    x0 = np.array([-1.0, -1.0])

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

    Q = 10. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    precompute_time = time.time() - start
    print(f"Precomputation time: {precompute_time}")
    start = time.time()
    
    # loaded_data = scipy.io.loadmat('optimal_results_hybrid.mat')
    # fixed_states = loaded_data['trajectory'][0]
    # fixed_inputs = loaded_data['controls'][0]
    
    states_lb = np.array([-100.0, -100.0])
    states_ub = np.array([100.0, 100.0]) 
    control_lb = np.array([-2.0])
    control_ub = np.array([2.0])
    
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
        
    # Add a time-varying state constraint
    for k in range(n_steps):
        if k < 2*n_steps//5 and k > n_steps//5:
            # break
            swi_lin_mpc.add_constraint(
                [swi_lin_mpc.states[k]],
                np.array([-ca.inf, -ca.inf]), 
                np.array([-0.2, -0.2]), 
                name=f"state_constraint_{k}"
            )
        elif k > 3*n_steps//5:
            break
            swi_lin_mpc.add_constraint(
                [swi_lin_mpc.states[k]],  
                np.array([-0.2, -0.2]),
                np.array([ca.inf, ca.inf]),
                name=f"state_constraint_{k}"
            )

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
    
    # Define constraint regions for plotting (based on the constraint code)
    constraint_regions = {
        'upper_bound': {'start': 20, 'end': 40, 'value': -0.2},
        'lower_bound': {'start': 60, 'value': -0.2}
    }
    
    return swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt, n_steps, time_horizon, constraint_regions


def test_uniform_time_sampling(args):
    """
    Run the same optimization with uniform time sampling (inspect mode).
    """
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
        'A': [np.array([[-0.5, -4.0], [4.0, -0.5]])],
        'B': [np.array([[0.0], [2.0]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 5.0
    
    x0 = np.array([-1.0, -1.0])

    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_steps, 
        time_horizon, 
        auto=False,
        x0=x0,
        multiple_shooting=multiple_shooting,
        propagation=integrator,
        inspect=True,  # Enable inspect mode for uniform time sampling
        hybrid=hybrid,
        plot=plot,
    )

    Q = 10. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    precompute_time = time.time() - start
    print(f"Precomputation time (uniform): {precompute_time}")
    start = time.time()
    
    states_lb = np.array([-100.0, -100.0])
    states_ub = np.array([100.0, 100.0]) 
    control_lb = np.array([-2.0])
    control_ub = np.array([2.0])
    
    swi_lin_mpc.set_bounds(
        control_lb, 
        control_ub, 
        states_lb, 
        states_ub, 
    )
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints()
        
    # Add the same time-varying state constraints
    for k in range(n_steps):
        if k < 2*n_steps//5 and k > n_steps//5:
            swi_lin_mpc.add_constraint(
                [swi_lin_mpc.states[k]],
                np.array([-ca.inf, -ca.inf]), 
                np.array([-0.2, -0.2]), 
                name=f"state_constraint_{k}"
            )
        elif k > 3*n_steps//5:
            break
            swi_lin_mpc.add_constraint(
                [swi_lin_mpc.states[k]],  
                np.array([-0.2, -0.2]),
                np.array([ca.inf, ca.inf]),
                name=f"state_constraint_{k}"
            )

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
    
    swi_lin_mpc.set_initial_guess(x0)

    swi_lin_mpc.create_solver('ipopt')
    
    setup_time = time.time() - start
    print(f"Setup time (uniform): {setup_time}")
    start = time.time()
    
    inputs_opt, deltas_opt, states_opt = swi_lin_mpc.solve(x0)
    solving_time = time.time() - start
    print(f"Solving time (uniform): {solving_time}")
    print("--------------------------------")
    print(f"Total time (uniform): {precompute_time + setup_time + solving_time}")
    
    # Define constraint regions for plotting (based on the constraint code)
    constraint_regions = {
        'upper_bound': {'start': 20, 'end': 40, 'value': -0.2},
        'lower_bound': {'start': 60, 'value': -0.2}
    }
    
    return swi_lin_mpc.opt_cost, states_opt, inputs_opt, deltas_opt, n_steps, time_horizon, constraint_regions


def plot_optimization_results(cost, states, inputs, deltas, n_steps, time_horizon, constraint_regions, save_fig=False):
    """
    Plot the optimization results including states, controls, and time distribution.
    Shows time-varying constraints that are active at different phases.
    
    Args:
        cost: Optimal cost value
        states: Optimal state trajectory
        inputs: Optimal control inputs
        deltas: Optimal phase durations
        n_steps: Number of optimization steps
        time_horizon: Total time horizon
        constraint_regions: Dictionary defining constraint regions with 'upper_bound' and 'lower_bound' keys
        save_fig: Whether to save the figure
    """
    # Convert to numpy arrays
    states_array = np.array(states).reshape(-1, 2)
    inputs_array = np.array(inputs).flatten()
    deltas_array = np.array(deltas).flatten()
    
    # Compute time vector
    time_points = np.concatenate([[0], np.cumsum(deltas_array)])
    
    # Extract constraint boundaries
    upper_constraint = constraint_regions.get('upper_bound')
    lower_constraint = constraint_regions.get('lower_bound')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Subplot 1: State x1 vs time with time-varying constraints
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(time_points, states_array[:, 0], 'b-', linewidth=2.5, marker='o', markersize=4, zorder=5)
    
    # Add time-varying constraint regions
    if upper_constraint is not None:
        k_start = upper_constraint['start']
        k_end = upper_constraint['end']
        val = upper_constraint['value']
        if k_end < len(time_points):
            ax1.fill_between([time_points[k_start], time_points[k_end]], -2, val, 
                            color='red', alpha=0.15, 
                            label=f'Constrained ({k_start}≤k<{k_end}): x ≤ {val}')
            ax1.axhline(y=val, xmin=time_points[k_start]/time_horizon, 
                       xmax=time_points[k_end]/time_horizon, 
                       color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    if lower_constraint is not None:
        k_start = lower_constraint['start']
        k_end = lower_constraint.get('end', n_steps)
        val = lower_constraint['value']
        if k_start < len(time_points):
            ax1.fill_between([time_points[k_start], time_points[min(k_end, len(time_points)-1)]], 
                            val, 2, color='orange', alpha=0.15,
                            label=f'Constrained (k>{k_start}): x ≥ {val}')
            ax1.axhline(y=val, xmin=time_points[k_start]/time_horizon, xmax=1, 
                       color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Time [s]', fontsize=12)
    ax1.set_ylabel('State x₁', fontsize=12)
    ax1.set_title('State Trajectory x₁ (Time-Varying Constraints)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    
    # Subplot 2: State x2 vs time with time-varying constraints
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(time_points, states_array[:, 1], 'g-', linewidth=2.5, marker='o', markersize=4, zorder=5)
    
    # Add time-varying constraint regions
    if upper_constraint is not None:
        k_start = upper_constraint['start']
        k_end = upper_constraint['end']
        val = upper_constraint['value']
        if k_end < len(time_points):
            ax2.fill_between([time_points[k_start], time_points[k_end]], -2, val,
                            color='red', alpha=0.15,
                            label=f'Constrained ({k_start}≤k<{k_end}): x ≤ {val}')
            ax2.axhline(y=val, xmin=time_points[k_start]/time_horizon,
                       xmax=time_points[k_end]/time_horizon,
                       color='r', linestyle='--', linewidth=2, alpha=0.7)
    
    if lower_constraint is not None:
        k_start = lower_constraint['start']
        k_end = lower_constraint.get('end', n_steps)
        val = lower_constraint['value']
        if k_start < len(time_points):
            ax2.fill_between([time_points[k_start], time_points[min(k_end, len(time_points)-1)]], 
                            val, 2, color='orange', alpha=0.15,
                            label=f'Constrained (k>{k_start}): x ≥ {val}')
            ax2.axhline(y=val, xmin=time_points[k_start]/time_horizon, xmax=1,
                       color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time [s]', fontsize=12)
    ax2.set_ylabel('State x₂', fontsize=12)
    ax2.set_title('State Trajectory x₂ (Time-Varying Constraints)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # Subplot 3: Phase portrait with constraint visualization
    ax3 = plt.subplot(2, 3, 3)
    
    # Color code the trajectory by constraint phase
    for i in range(len(states_array) - 1):
        # Determine color based on active constraints
        if upper_constraint is not None and upper_constraint['start'] <= i < upper_constraint['end']:
            color = 'red'
            alpha = 0.6
        elif lower_constraint is not None and i >= lower_constraint['start']:
            color = 'orange'
            alpha = 0.6
        else:
            color = 'blue'
            alpha = 0.8
        
        ax3.plot(states_array[i:i+2, 0], states_array[i:i+2, 1], 
                color=color, linewidth=2.5, alpha=alpha)
        ax3.plot(states_array[i, 0], states_array[i, 1], 'o', 
                color=color, markersize=4, alpha=alpha)
    
    ax3.plot(states_array[0, 0], states_array[0, 1], 'go', markersize=12, label='Initial', zorder=10)
    ax3.plot(states_array[-1, 0], states_array[-1, 1], 'r*', markersize=16, label='Final', zorder=10)
    
    # Add constraint boundaries
    if upper_constraint is not None:
        val = upper_constraint['value']
        ax3.axhline(y=val, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axvline(x=val, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    if lower_constraint is not None:
        val = lower_constraint['value']
        ax3.axhline(y=val, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axvline(x=val, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add legend for phases
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Initial'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=12, label='Final')
    ]
    
    if upper_constraint is not None:
        k_start = upper_constraint['start']
        k_end = upper_constraint['end']
        val = upper_constraint['value']
        legend_elements.insert(0, Line2D([0], [0], color='red', lw=2, 
                                        label=f'Phase {k_start}≤k<{k_end}: x≤{val}'))
    if lower_constraint is not None:
        k_start = lower_constraint['start']
        val = lower_constraint['value']
        legend_elements.insert(-2 if upper_constraint else 0, 
                              Line2D([0], [0], color='orange', lw=2, 
                                    label=f'Phase k>{k_start}: x≥{val}'))
    
    # Add unconstrained phase
    legend_elements.insert(-2, Line2D([0], [0], color='blue', lw=2, label='Unconstrained phase'))
    
    ax3.legend(handles=legend_elements, fontsize=8, loc='best')
    
    ax3.set_xlabel('State x₁', fontsize=12)
    ax3.set_ylabel('State x₂', fontsize=12)
    ax3.set_title('Phase Portrait (Color-coded by Constraint Phase)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([min(-1.2, states_array[:, 0].min() - 0.1), max(0.2, states_array[:, 0].max() + 0.1)])
    ax3.set_ylim([min(-1.2, states_array[:, 1].min() - 0.1), max(0.2, states_array[:, 1].max() + 0.1)])
    
    # Subplot 4: Control input vs time (piecewise constant)
    ax4 = plt.subplot(2, 3, 4)
    for i in range(len(inputs_array)):
        # Color code by constraint phase
        if upper_constraint is not None and upper_constraint['start'] <= i < upper_constraint['end']:
            color = 'red'
        elif lower_constraint is not None and i >= lower_constraint['start']:
            color = 'orange'
        else:
            color = 'blue'
            
        ax4.plot([time_points[i], time_points[i+1]], [inputs_array[i], inputs_array[i]], 
                color=color, linewidth=2.5, alpha=0.7)
        if i < len(inputs_array) - 1:
            ax4.plot([time_points[i+1], time_points[i+1]], [inputs_array[i], inputs_array[i+1]], 
                    color=color, linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax4.axhline(y=-2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Bounds')
    ax4.axhline(y=2.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add vertical lines to show constraint phase transitions
    if upper_constraint is not None:
        ax4.axvline(x=time_points[upper_constraint['start']], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        ax4.axvline(x=time_points[upper_constraint['end']], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    if lower_constraint is not None and lower_constraint['start'] < len(time_points):
        ax4.axvline(x=time_points[lower_constraint['start']], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax4.set_xlabel('Time [s]', fontsize=12)
    ax4.set_ylabel('Control input u', fontsize=12)
    ax4.set_title('Control Input (Color-coded by Phase)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim([-2.5, 2.5])
    
    # Subplot 5: Phase durations
    ax5 = plt.subplot(2, 3, 5)
    colors_bar = []
    for i in range(len(deltas_array)):
        if upper_constraint is not None and upper_constraint['start'] <= i < upper_constraint['end']:
            colors_bar.append('red')
        elif lower_constraint is not None and i >= lower_constraint['start']:
            colors_bar.append('orange')
        else:
            colors_bar.append('blue')
    
    ax5.bar(range(len(deltas_array)), deltas_array, color=colors_bar, alpha=0.7, edgecolor='navy')
    ax5.axhline(y=time_horizon/n_steps, color='k', linestyle='--', linewidth=2, 
               label=f'Uniform ({time_horizon/n_steps:.4f}s)', alpha=0.5)
    
    # Add vertical lines for phase transitions
    if upper_constraint is not None:
        ax5.axvline(x=upper_constraint['start']-0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7, 
                   label='Constraint transitions')
        ax5.axvline(x=upper_constraint['end']-0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    if lower_constraint is not None:
        ax5.axvline(x=lower_constraint['start']-0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7,
                   label='Constraint transitions' if upper_constraint is None else '')
    
    ax5.set_xlabel('Phase index', fontsize=12)
    ax5.set_ylabel('Phase duration [s]', fontsize=12)
    ax5.set_title('Time Distribution by Constraint Phase', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.legend(fontsize=9)
    
    # Subplot 6: Cumulative time
    ax6 = plt.subplot(2, 3, 6)
    cumulative_time = np.cumsum(deltas_array)
    uniform_cumulative = np.linspace(time_horizon/n_steps, time_horizon, n_steps)
    
    ax6.plot(range(1, n_steps+1), cumulative_time, 'b-', linewidth=2.5, marker='o', 
            markersize=4, label='Non-uniform')
    ax6.plot(range(1, n_steps+1), uniform_cumulative, 'k--', linewidth=2, 
            alpha=0.5, label='Uniform')
    
    # Highlight constraint phase transitions
    if upper_constraint is not None:
        ax6.axvline(x=upper_constraint['start'], color='red', linestyle=':', linewidth=2, alpha=0.7, 
                   label='Phase transitions')
        ax6.axvline(x=upper_constraint['end'], color='red', linestyle=':', linewidth=2, alpha=0.7)
    if lower_constraint is not None:
        ax6.axvline(x=lower_constraint['start'], color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label='Phase transitions' if upper_constraint is None else '')
    
    ax6.set_xlabel('Phase index', fontsize=12)
    ax6.set_ylabel('Cumulative time [s]', fontsize=12)
    ax6.set_title('Cumulative Time Distribution', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)
    
    # Add overall title with cost
    fig.suptitle(f'Optimization Results with Time-Varying Constraints (N={n_steps}, Cost={cost.item():.6f})', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_fig:
        plt.savefig('time_varying_constraints_results.png', dpi=300, bbox_inches='tight')
        print("Figure saved as 'time_varying_constraints_results.png'")
    
    plt.show()
    
    return fig


def plot_comparison(non_uniform_sol, uniform_sol, constraint_regions, save_fig=False):
    """
    Compare non-uniform and uniform time sampling solutions side-by-side.
    
    Args:
        non_uniform_sol: Tuple (cost, states, inputs, deltas, n_steps, time_horizon)
        uniform_sol: Tuple (cost, states, inputs, deltas, n_steps, time_horizon)
        constraint_regions: Dictionary defining constraint regions
        save_fig: Whether to save the figure
    """
    nu_cost, nu_states, nu_inputs, nu_deltas, nu_n_steps, nu_time_horizon, _ = non_uniform_sol
    u_cost, u_states, u_inputs, u_deltas, u_n_steps, u_time_horizon, _ = uniform_sol
    
    # Convert to numpy arrays
    nu_states_array = np.array(nu_states).reshape(-1, 2)
    nu_inputs_array = np.array(nu_inputs).flatten()
    nu_deltas_array = np.array(nu_deltas).flatten()
    nu_time_points = np.concatenate([[0], np.cumsum(nu_deltas_array)])
    
    u_states_array = np.array(u_states).reshape(-1, 2)
    u_inputs_array = np.array(u_inputs).flatten()
    u_deltas_array = np.array(u_deltas).flatten()
    u_time_points = np.concatenate([[0], np.cumsum(u_deltas_array)])
    
    # Extract constraint boundaries
    upper_constraint = constraint_regions.get('upper_bound')
    lower_constraint = constraint_regions.get('lower_bound')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: State x1 comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(nu_time_points, nu_states_array[:, 0], 'b-', linewidth=2.5, marker='o', markersize=3, 
            label='Non-uniform', zorder=5)
    ax1.plot(u_time_points, u_states_array[:, 0], 'r--', linewidth=2, marker='s', markersize=3, 
            label='Uniform', alpha=0.7, zorder=4)
    
    # Add constraint regions
    if upper_constraint is not None:
        k_start, k_end = upper_constraint['start'], upper_constraint['end']
        val = upper_constraint['value']
        ax1.fill_between([nu_time_points[k_start], nu_time_points[k_end]], -2, val, 
                        color='red', alpha=0.1)
        ax1.axhline(y=val, xmin=nu_time_points[k_start]/nu_time_horizon, 
                   xmax=nu_time_points[k_end]/nu_time_horizon, 
                   color='r', linestyle=':', linewidth=1.5, alpha=0.5)
    
    if lower_constraint is not None:
        k_start = lower_constraint['start']
        val = lower_constraint['value']
        ax1.fill_between([nu_time_points[k_start], nu_time_horizon], val, 2, 
                        color='orange', alpha=0.1)
        ax1.axhline(y=val, xmin=nu_time_points[k_start]/nu_time_horizon, xmax=1, 
                   color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('State x₁', fontsize=11)
    ax1.set_title('State x₁ Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: State x2 comparison
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(nu_time_points, nu_states_array[:, 1], 'b-', linewidth=2.5, marker='o', markersize=3, 
            label='Non-uniform', zorder=5)
    ax2.plot(u_time_points, u_states_array[:, 1], 'r--', linewidth=2, marker='s', markersize=3, 
            label='Uniform', alpha=0.7, zorder=4)
    
    # Add constraint regions
    if upper_constraint is not None:
        k_start, k_end = upper_constraint['start'], upper_constraint['end']
        val = upper_constraint['value']
        ax2.fill_between([nu_time_points[k_start], nu_time_points[k_end]], -2, val, 
                        color='red', alpha=0.1)
        ax2.axhline(y=val, xmin=nu_time_points[k_start]/nu_time_horizon, 
                   xmax=nu_time_points[k_end]/nu_time_horizon, 
                   color='r', linestyle=':', linewidth=1.5, alpha=0.5)
    
    if lower_constraint is not None:
        k_start = lower_constraint['start']
        val = lower_constraint['value']
        ax2.fill_between([nu_time_points[k_start], nu_time_horizon], val, 2, 
                        color='orange', alpha=0.1)
        ax2.axhline(y=val, xmin=nu_time_points[k_start]/nu_time_horizon, xmax=1, 
                   color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('State x₂', fontsize=11)
    ax2.set_title('State x₂ Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Phase portrait comparison
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(nu_states_array[:, 0], nu_states_array[:, 1], 'b-', linewidth=2.5, marker='o', 
            markersize=3, label='Non-uniform', zorder=5)
    ax3.plot(u_states_array[:, 0], u_states_array[:, 1], 'r--', linewidth=2, marker='s', 
            markersize=3, label='Uniform', alpha=0.7, zorder=4)
    ax3.plot(nu_states_array[0, 0], nu_states_array[0, 1], 'go', markersize=10, 
            label='Initial', zorder=10)
    ax3.plot(nu_states_array[-1, 0], nu_states_array[-1, 1], 'b*', markersize=12, 
            label='Final (NU)', zorder=10)
    ax3.plot(u_states_array[-1, 0], u_states_array[-1, 1], 'r*', markersize=12, 
            label='Final (U)', zorder=10)
    
    # Add constraint boundaries
    if upper_constraint is not None:
        val = upper_constraint['value']
        ax3.axhline(y=val, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
        ax3.axvline(x=val, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    if lower_constraint is not None:
        val = lower_constraint['value']
        ax3.axhline(y=val, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
        ax3.axvline(x=val, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax3.set_xlabel('State x₁', fontsize=11)
    ax3.set_ylabel('State x₂', fontsize=11)
    ax3.set_title('Phase Portrait Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Control input comparison (non-uniform)
    ax4 = plt.subplot(3, 3, 4)
    for i in range(len(nu_inputs_array)):
        ax4.plot([nu_time_points[i], nu_time_points[i+1]], 
                [nu_inputs_array[i], nu_inputs_array[i]], 
                'b-', linewidth=2.5, alpha=0.7)
        if i < len(nu_inputs_array) - 1:
            ax4.plot([nu_time_points[i+1], nu_time_points[i+1]], 
                    [nu_inputs_array[i], nu_inputs_array[i+1]], 
                    'b:', linewidth=1.5, alpha=0.5)
    
    ax4.axhline(y=-2.0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax4.axhline(y=2.0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('Control u', fontsize=11)
    ax4.set_title('Control Input (Non-uniform)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-2.5, 2.5])
    
    # Subplot 5: Control input comparison (uniform)
    ax5 = plt.subplot(3, 3, 5)
    for i in range(len(u_inputs_array)):
        ax5.plot([u_time_points[i], u_time_points[i+1]], 
                [u_inputs_array[i], u_inputs_array[i]], 
                'r-', linewidth=2.5, alpha=0.7)
        if i < len(u_inputs_array) - 1:
            ax5.plot([u_time_points[i+1], u_time_points[i+1]], 
                    [u_inputs_array[i], u_inputs_array[i+1]], 
                    'r:', linewidth=1.5, alpha=0.5)
    
    ax5.axhline(y=-2.0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax5.axhline(y=2.0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('Control u', fontsize=11)
    ax5.set_title('Control Input (Uniform)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([-2.5, 2.5])
    
    # Subplot 6: Cost comparison bar chart
    ax6 = plt.subplot(3, 3, 6)
    costs = [nu_cost.item(), u_cost.item()]
    colors_bar = ['blue', 'red']
    bars = ax6.bar(['Non-uniform', 'Uniform'], costs, color=colors_bar, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Optimal Cost', fontsize=11)
    ax6.set_title('Cost Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    # Add percentage difference
    diff_percent = 100 * (u_cost.item() - nu_cost.item()) / nu_cost.item()
    ax6.text(0.5, max(costs) * 0.5, f'Difference: {diff_percent:+.2f}%',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Subplot 7: Phase durations comparison
    ax7 = plt.subplot(3, 3, 7)
    ax7.bar(range(len(nu_deltas_array)), nu_deltas_array, color='blue', alpha=0.5, 
           label='Non-uniform', edgecolor='navy')
    ax7.axhline(y=nu_time_horizon/nu_n_steps, color='r', linestyle='--', linewidth=2, 
               label='Uniform', alpha=0.7)
    
    # Add constraint transition lines
    if upper_constraint is not None:
        ax7.axvline(x=upper_constraint['start']-0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        ax7.axvline(x=upper_constraint['end']-0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    if lower_constraint is not None:
        ax7.axvline(x=lower_constraint['start']-0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax7.set_xlabel('Phase index', fontsize=11)
    ax7.set_ylabel('Phase duration [s]', fontsize=11)
    ax7.set_title('Time Distribution Comparison', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Subplot 8: Cumulative time comparison
    ax8 = plt.subplot(3, 3, 8)
    nu_cumulative = np.cumsum(nu_deltas_array)
    u_cumulative = np.cumsum(u_deltas_array)
    
    ax8.plot(range(1, nu_n_steps+1), nu_cumulative, 'b-', linewidth=2.5, marker='o', 
            markersize=3, label='Non-uniform')
    ax8.plot(range(1, u_n_steps+1), u_cumulative, 'r--', linewidth=2, marker='s', 
            markersize=3, label='Uniform', alpha=0.7)
    
    # Add constraint transition lines
    if upper_constraint is not None:
        ax8.axvline(x=upper_constraint['start'], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        ax8.axvline(x=upper_constraint['end'], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    if lower_constraint is not None:
        ax8.axvline(x=lower_constraint['start'], color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    ax8.set_xlabel('Phase index', fontsize=11)
    ax8.set_ylabel('Cumulative time [s]', fontsize=11)
    ax8.set_title('Cumulative Time Distribution', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Subplot 9: Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_data = [
        ['Metric', 'Non-uniform', 'Uniform', 'Difference'],
        ['Cost', f'{nu_cost.item():.6f}', f'{u_cost.item():.6f}', 
         f'{diff_percent:+.2f}%'],
        ['Avg Δt [s]', f'{np.mean(nu_deltas_array):.6f}', f'{np.mean(u_deltas_array):.6f}', 
         f'{100*(np.mean(u_deltas_array)-np.mean(nu_deltas_array))/np.mean(nu_deltas_array):+.2f}%'],
        ['Min Δt [s]', f'{np.min(nu_deltas_array):.6f}', f'{np.min(u_deltas_array):.6f}', '—'],
        ['Max Δt [s]', f'{np.max(nu_deltas_array):.6f}', f'{np.max(u_deltas_array):.6f}', '—'],
        ['Std Δt [s]', f'{np.std(nu_deltas_array):.6f}', f'{np.std(u_deltas_array):.6f}', '—'],
    ]
    
    table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                     bbox=[0, 0.2, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax9.set_title('Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Add overall title
    fig.suptitle(f'Non-uniform vs Uniform Time Sampling (N={nu_n_steps})', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_fig:
        plt.savefig('time_varying_constraints_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison figure saved as 'time_varying_constraints_comparison.png'")
    
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
        type=int, metavar="int", default=100, required=False,
        help='Number of steps.'
    )
    parser.add_argument('--plot',
        type=str, metavar="{display, save, none}", default="display", required=False,
        help='How to plot the results.'
    )
    args = parser.parse_args()
    if args.hybrid in ('False', 'false', '0'):
        args.hybrid = False
    
    # Run non-uniform optimization
    print("="*60)
    print("RUNNING NON-UNIFORM TIME SAMPLING OPTIMIZATION")
    print("="*60)
    non_uniform_sol = test_non_autonomous_switched_linear_constrained(args)
    
    # Run uniform optimization
    print("\n" + "="*60)
    print("RUNNING UNIFORM TIME SAMPLING OPTIMIZATION")
    print("="*60)
    args.n_steps = 2000
    uniform_sol = test_uniform_time_sampling(args)
    
    # Extract results
    nu_cost, nu_states, nu_inputs, nu_deltas, n_steps, time_horizon, constraint_regions = non_uniform_sol
    u_cost, u_states, u_inputs, u_deltas, _, _, _ = uniform_sol
    
    # Print summary comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'Non-uniform':<20} {'Uniform':<20}")
    print("-"*60)
    print(f"{'Optimal cost':<25} {nu_cost.item():<20.6f} {u_cost.item():<20.6f}")
    cost_diff = 100 * (u_cost.item() - nu_cost.item()) / nu_cost.item()
    print(f"{'Cost difference':<25} {'':<20} {cost_diff:+.3f}%")
    print("-"*60)
    print(f"{'Average phase dur. [s]':<25} {np.mean(nu_deltas):<20.6f} {np.mean(u_deltas):<20.6f}")
    print(f"{'Min phase dur. [s]':<25} {np.min(nu_deltas):<20.6f} {np.min(u_deltas):<20.6f}")
    print(f"{'Max phase dur. [s]':<25} {np.max(nu_deltas):<20.6f} {np.max(u_deltas):<20.6f}")
    print(f"{'Std phase dur. [s]':<25} {np.std(nu_deltas):<20.6f} {np.std(u_deltas):<20.6f}")
    print("-"*60)
    nu_states_array = np.array(nu_states).reshape(-1, 2)
    u_states_array = np.array(u_states).reshape(-1, 2)
    print(f"{'Initial state':<25} [{nu_states[0]:.4f}, {nu_states[1]:.4f}]")
    print(f"{'Final state (NU)':<25} [{nu_states_array[-1, 0]:.4f}, {nu_states_array[-1, 1]:.4f}]")
    print(f"{'Final state (U)':<25} [{u_states_array[-1, 0]:.4f}, {u_states_array[-1, 1]:.4f}]")
    print("="*60 + "\n")
    
    # Generate comparison plot
    save_figure = args.plot == 'save'
    plot_comparison(non_uniform_sol, uniform_sol, constraint_regions, save_fig=save_figure)
    
    print("All tests passed!")