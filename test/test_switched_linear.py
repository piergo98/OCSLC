import time

import casadi as ca
import numpy as np

from ocslc.switched_linear_mpc import SwitchedLinearMPC




def test_autonomous_switched_linear():
    model = {
        'A': [np.array([[-1, 0], [1, 2]]), np.array([[1, 1], [1, -2]])],
        'B': [np.array([[1], [1]]), np.array([[2], [-1]])]
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 6
    time_horizon = 1

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=True)

    Q =  1.0 * np.eye(n_states)
    R = 10.  * np.eye(n_inputs)
    E =  0.  * np.eye(n_states)

    x0 = np.array([1, 1])

    # start = time.time()
    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    # print(f"Execution time: {time.time() - start}")

    swi_lin_mpc.set_cost_function(R, x0)

    swi_lin_mpc.create_solver()

    _, deltas_opt = swi_lin_mpc.solve()
    
    assert np.allclose(
        np.cumsum(deltas_opt),
        np.array([0.100, 0.297, 0.433, 0.642, 0.767, 1.0]),
        atol=1e-3,
    ), "The optimal phase durations are not correct."
    
def test_non_autonomous_switched_linear_1():
    model = {
        'A': [np.zeros((2, 2))],
        'B': [np.eye(2)],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 1
    time_horizon = 10

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)

    Q = 0. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([0, 0])
    
    xr = np.array([1, -3])

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    swi_lin_mpc.set_bounds(-100, 100)
    
    xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    final_state_constr = [xf]
    final_state_lb = xr
    final_state_ub = xr

    swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

    swi_lin_mpc.set_cost_function(R, x0)

    swi_lin_mpc.create_solver()

    inputs_opt, deltas_opt = swi_lin_mpc.solve()
    
    assert np.allclose(
        inputs_opt,
        np.array([0.1, -0.3]),
        atol=1e-3,
    ), "The optimal inputs durations are not correct."
    
    assert np.allclose(
        np.cumsum(deltas_opt),
        np.array([10.0]),
        atol=1e-3,
    ), "The optimal phase durations are not correct."
    
def test_non_autonomous_switched_linear_2():
    model = {
        'A': [np.zeros((1, 1)), np.eye(1)],
        'B': [10*np.eye(1), np.zeros((1, 1))],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 2
    time_horizon = 1

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)

    Q = 0. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([0])
    
    xr = np.array([1])

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    swi_lin_mpc.set_bounds(-100, 100)
    
    xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    final_state_constr = [xf]
    final_state_lb = xr
    final_state_ub = xr

    swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

    swi_lin_mpc.set_cost_function(R, x0)

    swi_lin_mpc.create_solver()

    inputs_opt, deltas_opt = swi_lin_mpc.solve()
    
    assert np.allclose(
        inputs_opt,
        np.array([0.1213, 0.0]),
        atol=1e-3,
    ), "The optimal inputs durations are not correct."
    
    assert np.allclose(
        np.cumsum(deltas_opt),
        np.array([0.5, 1.0]),
        atol=1e-3,
    ), "The optimal phase durations are not correct."
