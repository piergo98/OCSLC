from utils.Switched_Linear import SwiLin
import numpy as np
import casadi as ca

# Define the optimization settings
T = 1
Np = 3
Nx = 2
Nu = 0

#  Define the model
model = {
    'num_modes': 2,
    'A': [np.array([[-1, 0], [1, 2]]), np.array([[1, 1], [1, -2]])],
    'B': [np.array([[0], [0]]), np.array([[0], [0]])]
}
 
prob = SwiLin(np=Np, nx=Nx, nu=Nu, auto=True)
prob.load_model(model)

Q = np.eye(Nx)
R = np.eye(Nu)
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


# print(prob.delta)
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

# Define the decision variables
U = ca.symvar(prob.u)
DELTA = prob.delta
print(f"Delta: {DELTA}")
input("Press Enter to continue...")
w = ca.vertcat(U, DELTA)

# Initial guess
U0 = [0] * (Np*Nu)
DELTA0 = [0] * Np
w0 = U0 + DELTA0

# Initial augmented state
x0_aug = x0 + [1]
print(f"Initial augmented state: {x0_aug}")

# Define the cost function
J = prob.cost_function(R, x0)
print(f"Cost function: {J(x0_aug, DELTA0)}")

input("Press Enter to continue...")

# Bounds on the decision variables
lbw = [-1] * (Np*Nu) + [0] * Np
ubw = [1] * (Np*Nu) + [T] * Np

# Constraints
g = []
lbg = []
ubg = []
# Constraint on the phases duration
g += [ca.sum1(DELTA)]
lbg += [T]
ubg += [T]

# Solve the optimization problem
# Create an NLP solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
# NLP solver options
opts = {'ipopt.max_iter': 5e3, 'ipopt.hsllib': "libhsl.so"} 
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

opt_sol = sol['x'].full().flatten()

# Extract the optimal control input
u_opt = opt_sol[:Np*Nu]

# Extract the optimal phase durations
delta_opt = opt_sol[Np*Nu:]

# print(f"Optimal phase durations: {delta_opt}")