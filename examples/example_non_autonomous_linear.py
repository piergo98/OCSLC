import time

import casadi as ca
from matplotlib import pyplot as plt
import numpy as np
import scipy.io

from ocslc.switched_linear_mpc import SwitchedLinearMPC


def test_non_autonomous_switched_linear_comparison():
    model = {
        'A': [np.array([[-1, 0], [1, 2]])],
        'B': [np.array([[1], [1]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 40
    time_horizon = 1
    
    Q = 1000. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([3, 5])
    
    xr = np.array([1, 2])

    optimal_costs = []
    optimal_constraints = []
    iter_array = [221, 850]
    multiple_shooting = False
    for iterations in iter_array:
        start = time.time()
        
        swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False, multiple_shooting=multiple_shooting, x0=x0, inspect=False)

        swi_lin_mpc.precompute_matrices(x0, Q, R, E)
        
        states_lb = np.array([-100, -100])
        states_ub = np.array([100, 100])
        
        inspect_inputs = np.array([-99.99999820907408, -87.82592334399766, -76.17826491519043, -65.0819167662988, 
                                -54.56564084417181, -44.66343498170417, -35.41621229879333, -26.874475074987547, 
                                -19.102770019807544, -12.18778706026806, -6.2551802821135, -1.513197500476302, 
                                1.5674890735341362, 1.454997982983552, 1.386492949930222, 1.3228183450249815, 
                                1.2620728601586688, 1.2033846381408486, 1.146401284663227, 1.0909916680275953, 
                                1.0371110053880483, 0.9847464755390286, 0.9338965328977575, 0.8845633029603772, 
                                0.836749839267076, 0.7904591271410186, 0.7456936923558545, 0.7024554037846498, 
                                0.660745308094442, 0.6205633521768089, 0.5819075142862474, 0.5447700171769275, 
                                0.5091184996615892, 0.4747972812057939, 0.4410239961986244, 0.4044701184712488, 
                                0.35650157406931204, 0.28713326367198294, 0.19213512182518305, 0.07133598523975132])
        
        swi_lin_mpc.set_bounds(-100, 100, states_lb, states_ub)
        
        if swi_lin_mpc.multiple_shooting:
            swi_lin_mpc.multiple_shooting_constraints(x0)
        
        # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
        # final_state_constr = [xf]
        # final_state_lb = xr
        # final_state_ub = xr

        # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

        swi_lin_mpc.set_cost_function(Q, R, x0)
    
    
        optimal_cost = []
        constraints_value = []
        for iter_num in range(1, iterations):
            # Set the initial guess  
            swi_lin_mpc.set_initial_guess(time_horizon, x0)
            # swi_lin_mpc.create_solver()
            swi_lin_mpc.create_solver(max_iter=iter_num)

            # DSS: 226 iterations
            # DMS: 1960 iterations
            inputs_opt, deltas_opt, _ = swi_lin_mpc.solve()
            optimal_cost.append(swi_lin_mpc.opt_cost)
            constraints_value.append(np.transpose(swi_lin_mpc.optimal_lambda_g) @ swi_lin_mpc.constraints_value)
            
        optimal_costs.append(optimal_cost)
        optimal_constraints.append(constraints_value)
        multiple_shooting = True
        print(f"Execution time: {time.time() - start}")
        
    # Save data to a .mat file
    data_to_save = {
        'optimal_costs': optimal_costs,
        'optimal_constraints': optimal_constraints,
        'iter_array': iter_array
    }

    scipy.io.savemat('comparison_results.mat', data_to_save)
    
    # Plot the optimal costs function value
    plt.figure()
    plt.semilogx(np.log(optimal_costs[0]), label='DSS')
    plt.semilogx(np.log(optimal_costs[1]), label='DMS')
    plt.xlabel('Iterations')
    plt.ylabel('Optimal Cost')
    plt.grid()
    plt.legend()
    
    constraints_value_ss = constraints_value[0]
    constraints_value_ms = constraints_value[1]
    
    plt.figure()
    plt.semilogx(constraints_value_ss, label='DSS', linestyle="-")
    plt.semilogx(constraints_value_ms, label='DMS', linestyle="--")
    plt.xlabel('Iterations')
    plt.ylabel('Constraints value')
    plt.grid()
    plt.legend()
    
    plt.show()
    
    # print("----------------------------------------")
    # print(f"Optimal cost function value DSS: {1.1549873979296872e+03}")
    # print(f"Optimal cost function value DMS: {swi_lin_mpc.opt_cost[0]}")
    # print("\n")
    # print(f"Difference: {1.1549873979296872e+03 - swi_lin_mpc.opt_cost[0]}")
    # print("----------------------------------------")
    
    swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
def test_non_autonomous_switched_linear_inspection():
    model = {
        'A': [np.array([[-1, 0], [1, 2]])],
        'B': [np.array([[1], [1]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 40
    time_horizon = 1
    
    Q = 1000. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([3, 5])
    
    xr = np.array([1, 2])

    optimal_costs = []
    shooting = [False, True]
    for case in shooting:
        start = time.time()
        
        swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False, multiple_shooting=case, x0=x0, propagation='int',inspect=True)

        swi_lin_mpc.precompute_matrices(x0, Q, R, E)
        
        states_lb = np.array([-100, -100])
        states_ub = np.array([100, 100])
        
        inspect_inputs = np.array([-99.99999820907408, -87.82592334399766, -76.17826491519043, -65.0819167662988, 
                                -54.56564084417181, -44.66343498170417, -35.41621229879333, -26.874475074987547, 
                                -19.102770019807544, -12.18778706026806, -6.2551802821135, -1.513197500476302, 
                                1.5674890735341362, 1.454997982983552, 1.386492949930222, 1.3228183450249815, 
                                1.2620728601586688, 1.2033846381408486, 1.146401284663227, 1.0909916680275953, 
                                1.0371110053880483, 0.9847464755390286, 0.9338965328977575, 0.8845633029603772, 
                                0.836749839267076, 0.7904591271410186, 0.7456936923558545, 0.7024554037846498, 
                                0.660745308094442, 0.6205633521768089, 0.5819075142862474, 0.5447700171769275, 
                                0.5091184996615892, 0.4747972812057939, 0.4410239961986244, 0.4044701184712488, 
                                0.35650157406931204, 0.28713326367198294, 0.19213512182518305, 0.07133598523975132])
        
        swi_lin_mpc.set_bounds(-100, 100, states_lb, states_ub)
        
        if swi_lin_mpc.multiple_shooting:
            swi_lin_mpc.multiple_shooting_constraints(x0)
        
        # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
        # final_state_constr = [xf]
        # final_state_lb = xr
        # final_state_ub = xr

        # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

        swi_lin_mpc.set_cost_function(Q, R, x0)
    
        # Set the initial guess  
        swi_lin_mpc.set_initial_guess(time_horizon, x0)
        swi_lin_mpc.create_solver()

        inputs_opt, deltas_opt, _ = swi_lin_mpc.solve()
        optimal_costs.append(swi_lin_mpc.opt_cost)
            
        print(f"Execution time: {time.time() - start}")
        
    # Save data to a .mat file
    data_to_save = {
        'optimal_costs': optimal_costs,
        # 'optimal_constraints': optimal_constraints,
        # 'iter_array': iter_array
    }

    scipy.io.savemat('inspection_results.mat', data_to_save)
    
    print("----------------------------------------")
    print(f"Optimal cost function value DSS: {optimal_costs[0]}")
    print(f"Optimal cost function value DMS: {optimal_costs[1]}")
    print("\n")
    print(f"Difference: {optimal_costs[0] - optimal_costs[1]}")
    print("----------------------------------------")
    
    # swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
def test_non_autonomous_switched_linear_toy():
    start = time.time()
    
    model = {
        'A': [np.array([[-1, 0], [1, 2]])],
        'B': [np.array([[1], [1]])],
    }

    n_states = model['A'][0].shape[0]
    n_inputs = model['B'][0].shape[1]

    n_phases = 40
    time_horizon = 1
    
    Q = 1000. * np.eye(n_states)
    R = 1. * np.eye(n_inputs)
    E = 0. * np.eye(n_states)

    x0 = np.array([3, 5])
    
    xr = np.array([1, 2])

    multiple_shooting = True

    swi_lin_mpc = SwitchedLinearMPC(model, n_phases, time_horizon, auto=False, multiple_shooting=multiple_shooting, x0=x0)

    swi_lin_mpc.precompute_matrices(x0, Q, R, E)
    
    states_lb = np.array([-100, -100])
    states_ub = np.array([100, 100])
    
    swi_lin_mpc.set_bounds(-100, 100, states_lb, states_ub)
    
    if swi_lin_mpc.multiple_shooting:
        swi_lin_mpc.multiple_shooting_constraints(x0)
    
    # xf = swi_lin_mpc.state_extraction(swi_lin_mpc.deltas, swi_lin_mpc.inputs)[-1]
    # final_state_constr = [xf]
    # final_state_lb = xr
    # final_state_ub = xr

    # swi_lin_mpc.add_constraint(final_state_constr, final_state_lb, final_state_ub)

    swi_lin_mpc.set_cost_function(Q, R, x0)

    swi_lin_mpc.set_initial_guess(time_horizon, x0)
    swi_lin_mpc.create_solver()
    inputs_opt, deltas_opt, _ = swi_lin_mpc.solve()
    
    print(f"Execution time: {time.time() - start}")
        
    # Save data to a .mat file
    # data_to_save = {
    #     'optimal_costs': optimal_costs,
    #     'iter_array': iter_array
    # }

    # scipy.io.savemat('results.mat', data_to_save)
    
    # print("----------------------------------------")
    # print(f"Optimal cost function value DSS: {1.1549873979296872e+03}")
    # print(f"Optimal cost function value DMS: {swi_lin_mpc.opt_cost[0]}")
    # print("\n")
    # print(f"Difference: {1.1549873979296872e+03 - swi_lin_mpc.opt_cost[0]}")
    # print("----------------------------------------")
    
    swi_lin_mpc.plot_optimal_solution(deltas_opt, inputs_opt)
    
if __name__ == '__main__':
    start = time.time()
    test_non_autonomous_switched_linear_inspection()
    # print(f"Execution time: {time.time() - start}")
    print("All tests passed!")