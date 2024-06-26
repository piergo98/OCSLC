# Here the class SwiLin is defined
# It provides all the tools to instanciate a Switched Linear Optimization problem presented in the TAC paper.
# Starting from the switched linear system, it provides the cost function and its gradient w.r.t. the control input and the phases duration

from math import factorial
from numbers import Number

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import block_diag


class SwiLin:
    def __init__(self, n_phases, n_states, n_inputs, time_horizon, auto=False) -> None:
        """
        Set up the SwiLin class
        
        Args:
            np          (int): Number of phases.
            nx          (int): Number of states
            nu          (int): NUmber of controls
            auto        (bool): Flag to set the optimization for autonomous systems
        """
        # Check if number of phases is greater than 1
        if n_phases < 1:
            raise ValueError("The number of phases must be greater than 0.")
        self.n_phases = n_phases
        # Check if the number of states is greater than 0
        if n_states < 1:
            raise ValueError("The number of states must be greater than 0.")
        self.n_states = n_states
        # Check if the number of controls is greater than 0 or if the system is autonomous
        if n_inputs < 0:
            raise ValueError("The number of controls must be greater than 0.")
        self.n_inputs = n_inputs
        
        if time_horizon < 0:
            raise ValueError("The time horizon must be greater than 0.")
        self.time_horizon = time_horizon
        
        # Define the system's variables
        self.x = []
        # Control input defined as a list of symbolic variables
        self.u = []
        for i in range(self.n_phases):
            if auto:
                self.n_inputs = 0
                self.u.append(ca.MX.zeros(1))
            else:
                self.u.append(ca.MX.sym(f'u_{i}', self.n_inputs))
                
        self.n_opti = self.n_phases*self.n_inputs + self.n_phases
        
        # Phase duration as a symbolic variable 
        self.delta = ca.MX.sym('delta', self.n_phases)
        
        # Initialize the matrices
        self.E = []
        self.phi_f = []
        self.H = []
        self.S = []
        self.C = []
        self.N = []
        self.D = []
        self.G = []
        self.x_opt = []
        self.S_num = []
        self.S_int = []
    
    def load_model(self, model) -> None:
        """
        Load the switched linear model 
        
        Args:
            model   (struct): Structure that stores all the informations about the switched linear model
        """
        A = []
        B = []
        for i in range(self.n_phases):
            id = i % len(model['A'])
            A.append(model['A'][id])
            # Check if the input matrix is empty
            if model['B']:
                B.append(model['B'][id])
            else:
                B.append(np.zeros((model['A'][id].shape[0], 1)))
              
        self.A = A
        self.B = B
        
    def integrator(self, func: ca.Function, t0, tf: ca.MX, *args):
        """
        Integrates f(t) between t0 and tf using the given function func using the composite Simpson's 1/3 rule.
        
        Args:
            func    (ca.Function): The function that describes the system dynamics
            t0      (ca.MX): The initial time
            tf      (ca.MX): The final time
            *args   (ca.MX): Additional arguments to pass to the function
            
        Returns:
            integral    (ca.MX): The result of the integration
        """
        # Number of steps for the integration
        steps = 10
        
        # Check if args is not empty and set the input accordingly
        input = args[0] if args else None

        # Integration using the composite Simpson's 1/3 rule
        h = (tf - t0) / steps
        t = t0

        # Determine if the system is autonomous or not
        is_autonomous = input == 'auto'

        # Integration for autonomous systems
        if is_autonomous:
            S = func(t) + func(tf)
        else:
            # Determine if the input is symbolic or not
            is_symbolic = input is not None and input.is_symbolic()
            # Integration for non-autonomous systems or general integrator
            S = func(t, input) + func(tf, input) if is_symbolic else func(tf, t) + func(tf, tf)
    
        S_ = 0
        for k in range(1, steps):
            coefficient = 2 if k % 2 == 0 else 4
            t += h
            if is_autonomous:
                S += func(t) * coefficient
            else:
                # Determine if the input is symbolic or not
                is_symbolic = input is not None and input.is_symbolic()
                # Integration for non-autonomous systems or general integrator
                S += func(t, input) * coefficient if is_symbolic else func(tf, t)*coefficient

        integral = S * (h / 3)
        
        return integral
    
    def compute_integral(self, A, B, tmin, tmax):
        """
        Computes the forced evolution of the system's state using CasADi for symbolic operations.
        
        Args:
        A (numpy.ndarray): The system matrix.
        B (numpy.ndarray): The input matrix.
        tmin (float): The start time for the integration.
        tmax (float): The end time for the integration.
        
        Returns:
        numpy.ndarray: The result of the integral computation.
        """
        # Define the symbolic variable
        s = ca.MX.sym('s') 
        
        # Define the function to be integrate
        f = self.expm(A, (tmax-s)) @ B
        
        int_function = ca.Function('int', [*ca.symvar(tmax), s], [f])
    
        integral_result = self.integrator(int_function, tmin, tmax)
        
        # print(f"Integral result: {ca.symvar(integral_result)}")
        
        # phi_f = ca.Function('phi_f', [*ca.symvar(integral_result)], [integral_result])
        
        
        return integral_result
    
    def expm(self, A, delta):
        """
        Computes the matrix exponential of A[index] * delta_i using CasADi.
        
        Args:
            A (np.array): "A" matrix the mode for which to compute the matrix exponential.
            delta (ca.MX): time variable
            
        Returns:
            exp_max (ca.MX): The computed matrix exponential.
        """        
                
        n = A.shape[0]  # Size of matrix A
        result = ca.MX.eye(n)   # Initialize result to identity matrix
        
        # Number of terms for the Taylor series expansion
        num_terms = self._get_n_terms_expm_approximation()
        
        for k in range(1, num_terms+1):
            term = np.linalg.matrix_power(A, k) * ca.power(delta, k) / factorial(k)
            result = result + term
    
        return result
    
    def _get_n_terms_expm_approximation(self):
        threshold = 1e-3
        
        n_terms_min = 6
        n_terms_max = 100
        
        err = self.time_horizon**n_terms_min / factorial(n_terms_min)
        
        for n_terms in range(n_terms_min+1, n_terms_max):
            if err < threshold:
                return n_terms_min
            
            err *= self.time_horizon / n_terms
            
        return n_terms_max
    
    def mat_exp_prop(self, index):
        """
        Compute matrix exponential properties.

        Args:
        index   (int): The index of the mode.

        Returns:
        Ei      (ca.MX): The matrix exponential of Ai*delta_i.
        phi_f_i (ca.MX): The integral part multiplied by the control input ui.
        Hi      (ca.MX): A list of matrices constructed in the loop, based on phi_f_i_ and Ai.
        """        
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
                
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Compute matrix exponential
        Ei = self.expm(A, delta_i)
        
        # Distinct case for autonomous systems
        # Extract the control input
        if self.n_inputs > 0:
            ui = self.u[index]
        
            # Compute the integral of the system matrix and input matrix over the time interval
            phi_f_i_ = self.compute_integral(A, B, 0, delta_i)
            
            phi_f_i = phi_f_i_ @ ui
        
        # Create the H matrix related to the i-th mode (only for the non-autonomous case)
        if self.n_inputs > 0:
            Hi = []
            
            # Fill the Hk matrix with the k-th column of phi_f_i_ (integral term)
            Hk = ca.MX.sym('Hi', self.n_states + 1, self.n_states + 1)
            Hk = 0*Hk
            for k in range(ui.shape[0]):
                Hk[:self.n_states, self.n_states] =  phi_f_i_[:, k]
                Hi.append(Hk)
        
            return Ei, phi_f_i, Hi
        else:
            return Ei, 0, 0
        
    def _propagate_state(self, x0):
        self.x = [x0]
        
        for i in range(self.n_phases):
            if self.n_inputs > 0:
                self.x.append(self.E[i] @ self.x[i] + self.phi_f[i])
            else:
                self.x.append(self.E[i] @ self.x[i])
        
    def transition_matrix(self, phi_a, phi_f):
        """
        Computes the transition matrix for the given index.
        
        Args:
        phi_a (ca.MX): The matrix exponential of the system matrix.
        phi_f (ca.MX): The integral term.
        
        Returns:
        phi (ca.MX): The transition matrix.
        
        """
        phi = ca.MX.sym('phi', self.n_states+1, self.n_states+1)
        phi = 0*phi
        
        # Distinct case for autonomous and non-autonomous systems
        if self.n_inputs > 0:
            phi[:self.n_states, :self.n_states] = phi_a
            phi[:self.n_states, self.n_states] = phi_f
            phi[-1, -1] = 1
        else:
            phi[:self.n_states, :self.n_states] = phi_a
            phi[-1, -1] = 1
        
        return phi
           
    def D_matrix(self, index, Q, tau_i):
        """
        Computes the D matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        Q (np.array): The weight matrix.
        tau_i (ca.MX): The sum of the phases elapsed.
        
        Returns:
        D (ca.MX): The D matrix.
        
        """
        eta = ca.MX.sym('eta')
        
        # Define the system matrices for the given index
        B = self.B[index]
        A = self.A[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Compute the integral term
        phi_a_t = self.expm(A, eta)
        phi_f_t = self.compute_integral(A, B, 0, eta)
        phi_t = self.transition_matrix(phi_a_t, phi_f_t@ui)
        
        # Create the D matrix related to the i-th mode
        D = []
        
        # Fill the D matrix with the Dij terms
        Hij_t = ca.MX.sym('Hij_t', self.n_states + 1, self.n_states + 1)
        Hij_t = 0*Hij_t
        for k in range(ui.shape[0]):
            Hij_t[:self.n_states, self.n_states] =  phi_f_t[:, k]
        
            arg = ca.mtimes([ca.transpose(Hij_t), Q, phi_t]) + ca.mtimes([ca.transpose(phi_t), Q, Hij_t])
        
            # Compute D matrix
            f = ca.Function('f', [eta, ui], [arg])
            Dij = self.integrator(f, 0, delta_i, ui)
            D.append(Dij)
        
        return D
    
    def S_matrix(self, index, Q, tau_i):
        """
        Computes the S matrix for the given index.
        
        Args:
        index   (int):      The index of the mode.
        Q       (np.array): The weight matrix.
        tau_i   (ca.MX):    The sum of the phases elapsed.
        
        Returns:
        S       (ca.MX):    The S matrix.
        
        """
        eta = ca.MX.sym('eta')
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Extract the autonomous and non-autonomous parts of the state
        E = self.E[index]
        phi_f = self.phi_f[index]
        
        # print(f"S_matrix: {self.S[0]}")
        # Extract the S matrix of the previous iteration
        S_prev = self.S[0]
        
        # Compute the integral term
        phi_a_t = self.expm(A, eta)
        phi_f_t = self.compute_integral(A, B, 0, eta)
        # print(f"phi_a_t: {phi_f_t}")
        
        if self.n_inputs == 0:
            phi_t = self.transition_matrix(phi_a_t, phi_f_t)
        elif self.n_inputs > 0:
            phi_t = self.transition_matrix(phi_a_t, phi_f_t@ui)
        else:
            raise ValueError("The number of controls must be greater than 0.")
        
        # Check the transition matrix (DEBUG)
        # trans = ca.Function('trans', [eta], [phi_t])
        # delta_opt = [0.1002, 0.1972, 0.1356, 0.2088, 0.1249, 0.2334]
        # print(f"Transition: {trans(delta_opt[index])}")
        
        f = ca.transpose(phi_t) @ Q @ phi_t
        
        # Debug the matrix f (DEBUG)
        # f_num = ca.Function('f_num', [eta], [f])
        # delta_opt = [0.1002, 0.1972, 0.1356, 0.2088, 0.1249, 0.2334]
        # print(f"f_num: {f_num(delta_opt[index])}")
        
        if self.n_inputs == 0:
            f_int = ca.Function('f_int', [eta], [f])
        elif self.n_inputs > 0:
            f_int = ca.Function('f_int', [eta, ui], [f])
        else:
            raise ValueError("The number of controls must be greater than 0.")
        
        # Compute the integral of the S matrix
        if self.n_inputs == 0:
            S_int = self.integrator(f_int, 0, delta_i, 'auto')
            S_int_num = ca.Function('S_int_num', [self.delta], [S_int])
            self.S_int.append(S_int_num)
        else:
            S_int = self.integrator(f_int, 0, delta_i, ui)
            S_int_num = ca.Function('S_int_num', [self.delta, *self.u], [S_int])
            self.S_int.append(S_int_num)
            
        # Debug the matrix S_int (DEBUG)
        # print(f"deltai: {delta_i}")
        # delta_opt = [0.1002, 0.1972, 0.1356, 0.2088, 0.1249, 0.2334]
        # delta_opt = [0.2649, 0.7351]
        # print(f"S_int_num: {S_int_num(delta_opt)}")
        
        phi_i = self.transition_matrix(E, phi_f)
        
        # Compute S matrix
        S = S_int + ca.mtimes([ca.transpose(phi_i), S_prev, phi_i])
        
        return S
    
        
    def C_matrix(self, index, Q):
        """
        Computes the C matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        Q (np.array): The weight matrix.
        
        Returns:
        C (ca.MX): The C matrix.
        
        """
        # Define the system matrices for the given index
        A = self.A[index]
        B = self.B[index]
        
        # Extract the control input
        ui = self.u[index]
        
        # Define the M matrix
        M = ca.MX.sym('M', self.n_states + 1, self.n_states + 1)
        M = 0*M
        
        M[:self.n_states, :self.n_states] = A
        M[:self.n_states, self.n_states] = B @ ui
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[index+1]
        
        C = Q + ca.transpose(M) @ S_prev + S_prev @ M
        
        return C
           
    def N_matrix(self, index):
        """
        Computes the N matrix for the given index.
        
        Args:
        index (int): The index of the mode.
        
        Returns:
        N (ca.MX): The N matrix.
        
        """
        # Initialize the N matrix of the current iteration
        N = []
        
        # Extract the S matrix of the previous iteration
        S_prev = self.S[index+1]
        
        # Extract the H matrix of the current iteration
        H = self.H[index]
        
        for j in range(self.n_inputs):
            Hij = H[j]
            # Compute N matrix
            Nij = ca.transpose(Hij) @ S_prev + S_prev @ Hij
            N.append(Nij)
        
        return N
        
    def G_matrix(self, R):
        """
        Computes the G matrix.
        
        Args:
        R (np.array): The weight matrix.
        
        Returns:
        G (ca.MX): The G matrix.
        
        """
        
        G = 0
        for i in range(self.n_phases):
            pippo = (ca.transpose(self.u[i]) @ R @ self.u[i]) * self.delta[i]
            G += pippo
            
        return G
        
    def cost_function(self, R, x0, xf=None, E=None):
        """
        Computes the cost function.
        
        Args:
        R (np.array): The weight matrix.
        x0 (np.array): The initial state.
        xf (np.array): The final state.
        E (np.array): The weight matrix for the terminal state.
        
        Returns:
        J (ca.MX): The cost function.
        
        """
        # Compute the cost function
        # J = sum(x0[i] * self.S[0][i, j] * x0[j] for i in range(self.Nx+1) for j in range(self.Nx+1))
        J = 0
        for i in range(self.n_states+1):
            for j in range(self.n_states+1):
                J += x0[i] * self.S[0][i, j] * x0[j]
        
        if self.n_inputs > 0:
            J += self.G_matrix(R)
            
        if xf is not None:
            J += -2 * xf.reshape(1, -1) @ E @ self.x[-1]
        
        # print(f"Control input: {self.u}")
        # print(f"Phase duration: {type(self.delta)}")
        # input("Press Enter to continue...")
        
        if self.n_inputs == 0:
            cost = ca.Function('cost', [self.delta], [J])
        else:
            cost = ca.Function('cost', [*self.u, self.delta], [J])
            
        # print(f"Cost function: {ca.evalf(J)}")
        
        return cost
        
    def grad_cost_function(self, index, R):
        """
        Computes the gradient of the cost function.
        
        Args:
        index (int): The index of the mode.
        R (np.array): The weight matrix.
        
        Returns:
        du (ca.MX): The gradient of the cost function with respect to the control input.
        d_delta (ca.MX): The gradient of the cost function with respect to the phase duration.
        
        """
        
        # Create the augmented state vectors
        x_aug = ca.MX.sym('x_aug', self.n_states + 1)
        x_next_aug = ca.MX.sym('x_next_aug', self.n_states + 1)
        x_aug[:self.n_states] = self.x[:, index]
        x_next_aug[:self.n_states] = self.x[:, index+1]
        
        # Extract the control input
        ui = self.u[:, index]
        
        # Extract the phase duration
        delta_i = self.delta[index]
        
        # Extract the C matrix of the current iteration
        C = self.C[index]
        
        # Extract the N matrix of the current iteration
        N = self.N[index]
        
        # Extract the D matrix of the current iteration
        D = self.D[index]
        
        # Compute the gradient of the cost function with respect to the control input      
        du = []
        for j in range(self.n_inputs):
            du_j = 2 * ui[j] * R[j, j] * delta_i + ca.mtimes([ca.transpose(x_aug), D, x_aug]) + ca.mtimes([ca.transpose(x_next_aug), N, x_next_aug])
            du.append(du_j)
        
        # Compute the gradient of the cost function with respect to the phase duration
        d_delta = ca.mtimes([ca.transpose(ui), R, ui]) + ca.mtimes([ca.transpose(x_next_aug), C, x_next_aug])
        
        return du, d_delta
    
    def precompute_matrices(self, x0, Q, R, E) -> None:
        """
        Precomputes the matrices that are necessary to write the cost function and its gradient.
        
        Args:
        x0  (np.array): The initial state.
        Q   (np.array): The weight matrix for the state.
        R   (np.array): The weight matrix for the control.
        E   (np.array): The weight matrix for the terminal state.
        """  
        # Augment the weight matrices
        Q_ = block_diag(Q, 0)
        E_ = block_diag(E, 0)
        
        # Compute the cumulative duration vector for the phases
        times = ca.vertcat(0, ca.cumsum(self.delta))
        
        for i in range(self.n_phases):
            # Compute the matrix exponential properties
            Ei, phi_f_i, Hi = self.mat_exp_prop(i)
            self.E.append(Ei)
            self.phi_f.append(phi_f_i)
            self.H.append(Hi)
        
            if self.n_inputs > 0:
                # Compute the D matrix
                D = self.D_matrix(i, Q_, times[i])
                self.D.append(D)
            
                # Compute the G matrix
                G = self.G_matrix(R)
                self.G.append(G)
        
        # Initialize the S matrix with the terminal cost
        self.S.append(E_)
        for i in range(self.n_phases-1, -1, -1):
            # Compute the S matrix
            S = self.S_matrix(i, Q_, times[i])
            
            # Create the S_num function for debugging
            if self.n_inputs == 0:
                S_num = ca.Function('S_num', [self.delta], [S])
            else:
                S_num = ca.Function('S_num', [self.delta, *self.u], [S])
                
            self.S.insert(0, S)
            self.S_num.insert(0, S_num)
        
        for i in range(self.n_phases):
            # Compute the C matrix
            C = self.C_matrix(i, Q_)
            self.C.append(C)
            
            if self.n_inputs > 0:
                # Compute the N matrix
                N = self.N_matrix(i)
                self.N.append(N)
                
        # Propagate the state using the computed matrices.
        self._propagate_state(x0)
    
    
    def state_extraction(self, delta_opt, *args):
        """
        Extract the optimal values of the state trajectory based on the optimized values of u and delta
        """    
        
        # Check if args is not empty and set the input accordingly
        u_opt = args[0] if args else None   
        
        x_opt = []
        for i in range(self.n_phases+1):
            # Autonomous systems case
            if self.n_inputs == 0:
                state = ca.Function('state', [self.delta], [self.x[i]])
                x_opt.append(state(delta_opt))
                # print(f"State: {self.x_opt}")
            
            # Non-autonomous systems case
            else:
                state = ca.Function('state', [*self.u, self.delta], [self.x[i]])
                if isinstance(u_opt[0], Number):
                    u_opt_list = np.reshape(u_opt, (-1, self.n_inputs)).tolist()
                    u_opt = u_opt_list
                x_opt.append(state(*u_opt, delta_opt))
                
        return x_opt
    
    def plot_optimal_solution(self, delta_opt, *args):
        """
        Plot the optimal state trajectory based on the optimized values of u and delta
        Plot the optimal control input if the system is non-autonomous
        """    
        
        # Check if args is not empty and set the input accordingly
        u_opt = args[0] if args else None   
        
        # Extract the optimal state trajectory
        if self.n_inputs == 0:
            x_opt = self.state_extraction(delta_opt)
        else:
            x_opt = self.state_extraction(delta_opt, u_opt)
        
        x_opt_num = [x_opt[i].elements() for i in range(len(x_opt))]
        
        # Create the time grid
        tgrid = []
        points = 1
        time = 0
        next_time = 0
        for i in range(self.n_phases):
            next_time = next_time + delta_opt[i]
            tgrid = np.concatenate((tgrid, np.linspace(time, next_time, points, endpoint=False)))
            time = time + delta_opt[i]
        tgrid = np.concatenate((tgrid, [self.time_horizon]))
        
        # Plot the state trajectory
        fig, ax= plt.subplots()
        ax.plot(tgrid, np.array(x_opt_num))
        # Add vertical lines to identify phase changes instants
        time = 0
        for i in range(self.n_phases):
            time = time + delta_opt[i]
            plt.axvline(x=time, color='k', linestyle='--')
        ax.set(xlabel='Time', ylabel='State', title='State trajectory')
        ax.grid()
        
        # Plot the control input if the system is non-autonomous
        if self.n_inputs > 0:
            fig, ax = plt.subplots(self.n_inputs, 1)
            u_opt_list = np.reshape(u_opt, (-1, self.n_inputs)).tolist()
            u_opt_list += u_opt_list[-1:]
            if self.n_inputs >= 2:
                for i in range(self.n_inputs):
                    # Extract the optimal control input at different time instants
                    input = [sublist[i] for sublist in u_opt_list]
                    ax[i].step(tgrid, np.array(input), where='post')
                    ax[i].set(xlabel='Time', ylabel='U_'+str(i))
                    ax[i].grid()
                    # Add vertical lines to identify phase changes instants
                    time = 0
                    for j in range(self.n_phases):
                        time = time + delta_opt[j]
                        plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)
            else:
                ax.step(tgrid, np.array(u_opt_list), where='post')
                ax.set(xlabel='Time', ylabel='U')
                ax.grid()
                # Add vertical lines to identify phase changes instants
                time = 0
                for i in range(self.n_phases):
                    time = time + delta_opt[i]
                    plt.axvline(x=time, color='k', linestyle='--', linewidth=0.5)   
            
        plt.show()

