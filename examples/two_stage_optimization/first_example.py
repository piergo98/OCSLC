import time

import casadi as ca
import numpy as np
import scipy.io
from scipy.linalg import solve_continuous_are

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_grad_and_hessian(args):
    
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
        'A': [np.array([[-0.1, 0, 0], [0, -2, -6.25], [0, 4, 0]])],
        'B': [np.array([[0.25], [2], [0]])],
    }
    # print("--------------------------------")
    # print(np.linalg.eigvals(model['A'][0]))

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    time_horizon = 10
    
    x0 = np.array([1.3440, -4.5850, 5.6470])

    swi_lin_mpc = SwitchedLinearMPC(
        model, 
        n_steps, 
        time_horizon, 
        auto=True,
        x0=x0,
        multiple_shooting=multiple_shooting,
        propagation=integrator,
        inspect = False,
        hybrid=hybrid,
        plot=plot,
    )

    Q = 1. * np.eye(n_states)
    R = 0.1 * np.eye(n_inputs)
    # Solve the Algebraic Riccati Equation
    P = np.array(solve_continuous_are(model['A'][0], model['B'][0], Q, R))

    swi_lin_mpc.precompute_matrices(x0, Q, R, P)
    
    exp_dist = 1.**np.arange(n_steps)
    delta = exp_dist * time_horizon / np.sum(exp_dist)
    # print(f"Delta: {delta}")
    # input("Press Enter to compute gradient and Hessian...")
    grad = swi_lin_mpc.grad(*delta.tolist())
    print(f"Gradient: {grad}")
    
    hessian = swi_lin_mpc.hessian(*delta.tolist())
    print(f"Hessian_shape: {hessian.shape}")
    print(f"Hessian eigenvalues: {np.linalg.eigvals(hessian)}")
    if np.linalg.eigvals(hessian).min() < 0:
        print("Warning: Hessian is not positive definite.")
        print("The smallest eigenvalue is:", np.linalg.eigvals(hessian).min())
    elif np.linalg.eigvals(hessian).min() == 0:
        print("Warning: Hessian is positive semi-definite.")
        print("The smallest eigenvalue is:", np.linalg.eigvals(hessian).min())
    else:
        print("Hessian is positive definite.")
        print("The smallest eigenvalue is:", np.linalg.eigvals(hessian).min())
    
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
        type=int, metavar="int", default=80, required=False,
        help='Number of steps.'
    )
    parser.add_argument('--plot',
        type=str, metavar="{display, save, none}", default="display", required=False,
        help='How to plot the results.'
    )
    args = parser.parse_args()
    if args.hybrid in ('False', 'false', '0'):
        args.hybrid = False
    
    start = time.time()
    test_grad_and_hessian(args)
    # print(f"Execution time: {time.time() - start}")
    print("All tests passed!")
