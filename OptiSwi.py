from utils.Switched_Linear import SwiLin
import numpy as np
import casadi as ca

# Define the optimization settings
T = 1
Np = 12
Nx = 2
Nu = 1

#  Define the model
model = {
    'num_modes': 2,
    'A': [np.array([[-1, 0], [1, 2]]), np.array([[1, 1], [1, -2]])],
    'B': [np.array([[1], [1]]), np.array([[1], [1]])]
}
 
prob = SwiLin(np=Np, nx=Nx, nu=Nu, auto=False)
prob.load_model(model)

Q = np.eye(Nx)
R = 10*np.eye(Nu)
E = 0 * np.eye(Nx)

# Initial state
x0 = [1, 1]

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

# print(f"Decision variables: {w[0]}")
# print(f"Decision variables length: {len(w)}")
# print(f"Problem input: {len(prob.u)}")


# print(f"Delta: {DELTA}")
# print(f"Delta_type: {type(DELTA)}")
# input("Press Enter to continue...")


# Initial augmented state
x0_aug = x0 + [1]
# print(f"Initial augmented state: {x0_aug}")

# Define the cost function
cost = prob.cost_function(R, x0_aug)

if prob.Nu == 0:
    J = cost(DELTA)
else:
    J = cost(*w)

# print(f"Cost function: {J(x0_aug, DELTA0)}")
# input("Press Enter to continue...")


# Define the constraints
g = []
lbg = []
ubg = []

# Constraint on the phases duration
g += [ca.sum1(DELTA)]
lbg += [T]
ubg += [T]

# Solve the optimization problem
# Create an NLP solver
problem = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
# NLP solver options
opts = {'ipopt.max_iter': 5e3, 'ipopt.hessian_approximation': 'limited-memory', 'ipopt.hsllib': "libhsl.so"} 
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

# delta_opt = [0.1002, 0.1972, 0.1356, 0.2088, 0.1249, 0.2334]

for i in range(len(prob.S_num)):
    if prob.Nu > 0:
        print(f"S matrix: {prob.S_num[i](delta_opt, *u_opt)}")
    else:
        print(f"S matrix: {prob.S_num[i](delta_opt)}")

# Extract the optimal state trajectory
prob.state_extraction(delta_opt, u_opt)

print(f"Optimal state trajectory: {prob.x_opt}")