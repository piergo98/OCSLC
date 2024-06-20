from utils.Switched_Linear import SwiLin
import numpy as np
import casadi as ca

# Define the optimization settings
T = 2
Np = 20
Nx = 2
Nu = 1

#  Define the model
model = {
    'A': [np.array([[-1, 0], [1, 2]]), np.array([[1, 1], [1, -2]])],
    # 'A': [np.array([[0.6, 1.2], [-0.8, 3.4]]), np.array([[4, 3], [-1, 0]])],
    'B': [np.array([[1], [1]]), np.array([[2], [-1]])]
}
 
prob = SwiLin(np=Np, nx=Nx, nu=Nu, auto=False)
prob.load_model(model)

Q = 0.5*np.eye(Nx)
R = 10*np.eye(Nu)
E = 0 * np.eye(Nx)

# Initial state
x0 = [1, 1]

# Final state
xr = [4, 2]

import time
start = time.time()

prob.precompute_matrices(x0, Q, R, E)

end = time.time()
execution_time = end - start
print(f"Execution time: {execution_time}")
# input("Press Enter to continue...")




##########################################################################################
# SET UP THE OPTIMIZATION PROBLEM
##########################################################################################

w = []
w0 = [] 
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# Define the decision variables, their bounds, and the initial guess
for i in range(Np):
    if Nu > 0:
        w += [ca.MX.sym(f"U_{i}", Nu)]
        U0 = [0] * Nu
        w0 += U0
        lbw += [-1] * Nu
        ubw += [1] * Nu

DELTA = ca.MX.sym("DELTA", Np)
w += [DELTA]
DELTA0 = [T/Np] * Np
w0 += DELTA0
lbw += [0] * Np
ubw += [T] * Np

# Initial augmented state
x0_aug = x0 + [1]

# Define the cost function
cost = prob.cost_function(R, x0_aug)

if prob.Nu == 0:
    J = cost(DELTA)
else:
    J = cost(*w)


# Define the constraints
g = []
lbg = []
ubg = []

# Constraint on the phases duration
g += [ca.sum1(DELTA)]
lbg += [T]
ubg += [T]

# Final state constraint
xf = prob.state_extraction(DELTA, w[:Np*Nu])[-1]
g += [xr - xf]
lbg += [0, 0]
ubg += [0, 0]

# Solve the optimization problem
# Create an NLP solver
problem = {
    'f': J, 
    'x': ca.vertcat(*w), 
    'g': ca.vertcat(*g)
    }

# NLP solver options
opts = {
    'ipopt.max_iter': 5e3,
    # 'ipopt.gradient_approximation': 'finite-difference-values',
    # 'ipopt.hessian_approximation': 'limited-memory', 
    # 'ipopt.hsllib': "/usr/local/libhsl.so",
    # 'ipopt.linear_solver': 'mumps',
    # 'ipopt.mu_strategy': 'adaptive',
    # 'ipopt.adaptive_mu_globalization': 'kkt-error',
    # 'ipopt.tol': 1e-6,
    # 'ipopt.acceptable_tol': 1e-4,
    # 'ipopt.print_level': 3
    } 

solver = ca.nlpsol('solver', 'ipopt', problem, opts)

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

opt_sol = sol['x'].full().flatten()

# Extract the optimal control input
if prob.Nu > 0:
    u_opt = opt_sol[:Np*Nu]
    print(f"Optimal control input: {u_opt}")

# Extract the optimal phase durations
delta_opt = opt_sol[Np*Nu:]
print(f"Optimal phase durations: {delta_opt}")
print(f"Optimal switching instants: {np.cumsum(delta_opt)}")

# delta_opt = [0.1002, 0.1972, 0.1356, 0.2088, 0.1249, 0.2334]

for i in range(len(prob.S_num)):
    if prob.Nu > 0:
        print(f"S matrix: {prob.S_int[i](delta_opt, *u_opt)}")
    else:
        print(f"S matrix: {prob.S_int[i](delta_opt)}")

# Extract the optimal state trajectory
if prob.Nu == 0:
    x_opt = prob.state_extraction(delta_opt)
else:
    x_opt = prob.state_extraction(delta_opt, u_opt)

# Create a list with all the optimal state trajectories
x_opt_num = [x_opt[i].elements() for i in range(len(x_opt))]
print(f"Optimal state trajectory: {x_opt_num}")