import time

import casadi as ca
import numpy as np

from ocslc.switched_linear_mpc import SwitchedLinearMPC



model = {
    'A': [np.array([[-1, 0], [1, 2]]), np.array([[1, 1], [1, -2]])],
    # 'A': [np.array([[0.6, 1.2], [-0.8, 3.4]]), np.array([[4, 3], [-1, 0]])],
    'B': [np.array([[1], [1]]), np.array([[2], [-1]])],
}

n_states = model['A'][0].shape[0]
n_inputs = model['B'][0].shape[1]

n_phases = 6
time_horizon = 1

swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False)

Q =  1.0 * np.eye(n_states)
R = 10   * np.eye(n_inputs)
E =  0   * np.eye(n_states)

x0 = np.array([1, 1])

xr = np.array([4, 2])

start = time.time()
swi_lin_mpc.precompute_matrices(x0, Q, R, E)
print(f"Execution time: {time.time() - start}")

swi_lin_mpc.set_cost_function(R, x0)

swi_lin_mpc.set_bounds(-1, 1)

xf = swi_lin_mpc.state_extraction(
    swi_lin_mpc.deltas, swi_lin_mpc.inputs,
)[-1]
final_state_constr = [xr - xf]
final_state_lb = [0, 0]
final_state_ub = [0, 0]

swi_lin_mpc.add_constraints(final_state_constr, final_state_lb, final_state_ub)

swi_lin_mpc.create_solver()

swi_lin_mpc.solve()
