import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from .switched_linear import SwiLin



class SwitchedLinearMPC(SwiLin):
    class Constraint():
        def __init__(self, g, lbg, ubg, name=None):
            self.g = g
            self.lbg = lbg
            self.ubg = ubg
            self.name = name
    
    def __init__(self, model, n_phases, time_horizon, auto=False, multiple_shooting=False, x0=None) -> None:
        self._check_model_structure(model)
        n_states = model['A'][0].shape[0]
        n_inputs = model['B'][0].shape[1]
        
        super().__init__(n_phases, n_states, n_inputs, time_horizon, auto)
        self.load_model(model)
        
        # Store flags
        self.multiple_shooting = multiple_shooting
                
        if not auto:
            self.inputs = [ca.SX.sym(f"U_{i}", self.n_inputs) for i in range(self.n_phases)]
        else:
            self.inputs = []
        
        self.deltas = [ca.SX.sym(f"Delta_{i}") for i in range(self.n_phases)]
        
        self.n_opti = (self.n_inputs + 1) * self.n_phases
        
        self.shift = self.n_inputs + 1
        
        # Add state variables if multiple shooting is enabled
        if self.multiple_shooting:
            self.states = [ca.SX.sym(f"X_{i}", self.n_states) for i in range(self.n_phases+1)]
            self.n_opti += (self.n_phases + 1) * self.n_states
            self.shift += self.n_states
        
        # Build the optimization vector
        if self.n_inputs == 0:
            self.opt_var = self.deltas
            if self.multiple_shooting:
                self.opt_var = [val for pair in zip(self.states, self.deltas) for val in pair]     # To be fixed if N > n_phases
                self.opt_var.append(self.states[-1])
        else:
            self.opt_var = [val for pair in zip(self.inputs, self.deltas) for val in pair]
            if self.multiple_shooting:
                self.opt_var = [val for pair in zip(self.states, self.inputs, self.deltas) for val in pair]     # To be fixed if N > n_phases
                self.opt_var.append(self.states[-1])
        
        # Set general bounds
        self.lb_opt_var = - np.ones(self.n_opti) * np.inf
        self.ub_opt_var =   np.ones(self.n_opti) * np.inf
        
        # Set bounds for the phase durations
        self.lb_opt_var[self.shift-1::self.shift] = 0
        self.ub_opt_var[self.shift-1::self.shift] = time_horizon
        
        # Initialize cost and constraints
        self.cost = 0
        self.constraints = []
        
        # Set the total time constraint
        self._set_constraints_deltas()
        
        # if self.multiple_shooting:
        #     self.multiple_shooting_constraints(x0)
        
        # # Set the initial guess  
        # self.set_initial_guess(time_horizon, x0) # TO DO
        
    def set_initial_guess(self, time_horizon, x0=None):
        '''
        This method sets the initial guess for the optimization variables.
        '''
        # Check that x0 is provided if multiple shooting is enabled
        if self.multiple_shooting and x0 is None:
            raise ValueError("x0 must be provided when multiple shooting is enabled.")
        
        u0 = np.zeros(self.n_inputs)
        delta0 = time_horizon / self.n_phases
        
        temp = []
        if self.multiple_shooting:
            temp += x0.tolist()
            # Propagate dynamics from initial state
            for i in range(self.n_phases):
                if self.n_inputs > 0:
                    x_next = self.autonomous_evol[i](delta0) @ x0 + self.forced_evol[i](u0, delta0)
                else:
                    x_next = self.autonomous_evol[i](delta0) @ x0
                temp += [0] * self.n_inputs + [time_horizon / self.n_phases]
                temp += x_next.full().flatten().tolist()
                
                x0 = x_next
        else:
            for _ in range(self.n_phases):
                temp += [0] * self.n_inputs + [time_horizon / self.n_phases]
        
        self.opt_var_0 = np.array(temp)
          
    @staticmethod
    def _check_model_structure(model):
        if set(model.keys()) != {'A', 'B'}:
            raise ValueError("The model must have and only have the keys 'A' and 'B'.")
        
        if len(model['A']) != len(model['B']):
            raise ValueError("'A' and 'B' do not have the same number of matrices.")
        
        A_shape = model['A'][0].shape
        for i, A in enumerate(model['A']):
            if A.shape != A_shape:
                raise ValueError("All 'A' matrices are not the same size.")
            
            if A.shape[0] != A.shape[1]:
                raise ValueError(f"A[{i}] is not a square matrix.")
        
        B_shape = model['B'][0].shape
        if B_shape[0] != A_shape[0]:
            raise ValueError("The number of rows in 'B' matrices does not match the number of rows in 'A' matrices.")
        for B in model['B']:
            if B.shape != B_shape:
                raise ValueError("All 'B' matrices are not the same size.")
        
    def set_bounds(self, inputs_lb, inputs_ub, states_lb=None, states_ub=None):
        """
        This method sets the lower and upper bounds for the control inputs and states.
        
        Args:
            inputs_lb (np.array): The lower bounds for the control inputs.
            inputs_ub (np.array): The upper bounds for the control inputs.
            states_lb (np.array): The lower bounds for the states. Only required if multiple_shooting is enabled.
            states_ub (np.array): The upper bounds for the states. Only required if multiple_shooting is enabled.
        """
        # Check if the states bounds are required
        if self.multiple_shooting and (states_lb is None or states_ub is None):
            raise ValueError("States bounds must be provided when multiple shooting is enabled.")    
        
        if self.multiple_shooting:
            # Set inputs bounds
            for i in range(self.n_states, self.n_opti, self.shift):
                self.lb_opt_var[i:i+self.n_inputs] = inputs_lb
                self.ub_opt_var[i:i+self.n_inputs] = inputs_ub
            
            # Set states bounds
            for i in range(0, self.n_opti, self.shift):
                self.lb_opt_var[i:i+self.n_states] = states_lb
                self.ub_opt_var[i:i+self.n_states] = states_ub
            
        else:
            if self.n_inputs > 0:
                for i in range(0, self.n_opti, self.shift):
                    self.lb_opt_var[i:i+self.n_inputs] = inputs_lb
                    self.ub_opt_var[i:i+self.n_inputs] = inputs_ub
                
    def _set_constraints_deltas(self):
        self.constraints.append(self.Constraint(
            g=[ca.sum1(ca.vertcat(*self.deltas))],
            lbg=np.array([self.time_horizon]),
            ubg=np.array([self.time_horizon]),
            name="Total time",
        ))
        
    def add_constraint(self, g, lbg, ubg, name=None):
        if name is not None:
            for constraint in self.constraints:
                if constraint.name == name:
                    raise ValueError(f"Constraint {name} already exists.")
        
        self.constraints += [self.Constraint(
            g=g,
            lbg=np.array([lbg]).flatten(),
            ubg=np.array([ubg]).flatten(),
            name=name,
        )]
        
    def multiple_shooting_constraints(self, x0, displacement=0, update=False):
        '''
        This method creates the constraints for the multiple shooting approach.
        '''
        # Check if x0 is provided when multiple shooting is enabled
        if x0 is None:
            raise ValueError("x0 must be provided when multiple shooting is enabled.")
        
        # Set bounds for the first state
        self.lb_opt_var[:self.n_states] = x0
        self.ub_opt_var[:self.n_states] = x0
        
        for i in range(self.n_phases):
            x = self.states[i]
            x_next = self.states[i+1]
            delta = self.deltas[i]
            if self.n_inputs > 0:
                u = self.inputs[i]
                # Check the index to avoid out of bounds
                
                x_next_pred = self.autonomous_evol[(i+displacement)%self.n_phases](delta) @ x + self.forced_evol[(i+displacement)%self.n_phases](u, delta)
            else:
                x_next_pred = self.autonomous_evol[(i+displacement)%self.n_phases](delta) @ x
            
            
            # Update or add the constraint
            if update:
                self.update_constraint(f"State_{i+1}", g=[x_next - x_next_pred])
            else:
                self.add_constraint([x_next - x_next_pred], np.zeros(self.n_states), np.zeros(self.n_states), f"State_{i+1}")
            
    def update_constraint(self, name, g=None, lbg=None, ubg=None):
        for constraint in self.constraints:
            if constraint.name == name:
                if g is not None:
                    constraint.g = g
                if lbg is not None:
                    constraint.lbg = np.array([lbg]).flatten()
                if ubg is not None:
                    constraint.ubg = np.array([ubg]).flatten()
                return
            
        raise ValueError(f"Constraint {name} not found.")
        
    def set_cost_function_single_shooting(self, R, x0, xr=None, E=None):
        '''
        This method sets the cost function for the optimization problem using the single shooting approach.
        '''
        x0_aug = np.append(x0, 1)
        
        if xr is not None and E is not None:
            cost = self.cost_function(R, x0_aug, xr, E)
        elif xr is not None and E is None:
            raise ValueError("xr must be provided with E.")
        elif xr is None and E is not None:
            raise ValueError("E must be provided with xr.")
        else:
            cost = self.cost_function(R, x0_aug)
                
        if self.n_inputs == 0:
            self.cost = cost(*self.deltas)
        else:
            self.cost = cost(*self.inputs, *self.deltas)
        
    def set_cost_function_multiple_shooting(self, Q, R, x_ref=None, E=None):
        '''
        This method sets the cost function for the optimization problem using the multiple shooting approach.
        '''
        if x_ref is None:
            x_ref = np.zeros(self.n_states)
        if E is None:
            E = np.zeros((self.n_states, self.n_states))
        
        L = 0
        for i in range(self.n_phases):
            # Get variables
            x_i = self.states[i]
            delta_i = self.deltas[i]
            if self.n_inputs == 0:
                # Compute ith integral of the objective function using the Euler method
                L += (ca.transpose(x_i-x_ref) @ Q @ (x_i-x_ref)) * delta_i
            else:
                u_i = self.inputs[i]
                # Compute ith integral of the objective function using the Euler method
                L += (ca.transpose(x_i-x_ref) @ Q @ (x_i-x_ref) + ca.transpose(u_i) @ R @ u_i) * delta_i
        
        L += ca.transpose(self.states[-1]-x_ref) @ E @ (self.states[-1]-x_ref)
        
        self.cost = L
        
    def set_cost_function(self, Q, R, x0, xf=None, E=None):
        '''
        This method sets the cost function for the optimization problem.
        '''
        if self.multiple_shooting: 
            self.set_cost_function_multiple_shooting(Q, R, xf, E)
        else:
            self.set_cost_function_single_shooting(R, x0, xf, E)
            
    def update_opt_vector(self, x0, inputs_opt, deltas_opt, dt, time_horizon):
        '''
        This method updates the optimization vector for the subsequent optimization problem.
        It updates the phases taking into account the time elapsed.
        It also updates the control inputs and phases sequence.
        '''
        u0_new = inputs_opt.copy()
        deltas0_new = deltas_opt.copy().tolist()
        
        temp = []
        # Check if the dt is greater than the first n phases duration
        # i is the index that indicates that referres to the first phase 
        # that is not completely covered by the time dt. It also indicates how many phases are completed.
        i = np.where(dt < np.cumsum(deltas_opt))[0][0]
        if i == 0:
            # Update the first phase and add the time to the last phase
            deltas0_new[0] = deltas_opt[0] - dt
            deltas0_new[-1] = deltas_opt[-1] + dt
            
            # Propagate the states
            if self.multiple_shooting:
                temp += x0.tolist()
                # Propagate dynamics from initial state
                for j in range(self.n_phases):
                    delta0 = deltas0_new[j]
                    
                    if self.n_inputs > 0:
                        u0 = u0_new[j*self.n_inputs:j*self.n_inputs+self.n_inputs]
                        if isinstance(u0, np.ndarray):
                            u0 = u0.tolist()
                        
                        x_next = self.autonomous_evol[j](delta0) @ x0 + self.forced_evol[j](u0, delta0)
                        temp += u0 + [delta0]
                    else:
                        x_next = self.autonomous_evol[j](delta0) @ x0
                        temp += [delta0]
                    temp += x_next.full().flatten().tolist()
                    
                    x0 = x_next
                    
        else:
            # Shift the phases and update
            old_times = deltas_opt[:i]
            deltas0_new = self._shift(deltas0_new, i)
            for j in range(i):
                deltas0_new[-j-1] = np.sum(old_times) / i
            
            # Shift the controls
            if self.n_inputs > 0:
                u0_new = self._shift(u0_new, i)
                for j in range(i):
                    u0_new[-j-1] = 0
            
            # Sort the multiple shooting constraint with the new order
            if self.multiple_shooting:
                self.multiple_shooting_constraints(x0, displacement=i, update=True)
            
            # Propagate the states
            if self.multiple_shooting:
                temp += x0.tolist()
                # Propagate dynamics from initial state
                for j in range(self.n_phases):
                    delta0 = deltas0_new[j]
                    u0 = u0_new[j*self.n_inputs:j*self.n_inputs+self.n_inputs]
                    # Convert the control input to a list if is a numpy array
                    if isinstance(u0, np.ndarray):
                        u0 = u0.tolist()
                    
                    j_displaced = (j+i)%self.n_phases
                    if self.n_inputs > 0:
                        x_next = self.autonomous_evol[j_displaced](delta0) @ x0 \
                            + self.forced_evol[j_displaced](u0, delta0)
                        temp += u0 + [delta0]
                    else:
                        x_next = self.autonomous_evol[j_displaced](delta0) @ x0
                        temp += [delta0]
                    temp += x_next.full().flatten().tolist()
                    
                    x0 = x_next
                    
        # print(f"Optimization vector updated: {temp}")
        
        # update the optimization vector
        self.opt_var_0 = np.array(temp)
        
    def _shift(self, list: list, n):
        '''
        Shifts the elements of a list by n positions to the left.
        '''
        for _ in range(n):
            list.append(list.pop(0))
            
        return list      
            
    def create_solver(self, solver='ipopt'):
        g = []
        for constraint in self.constraints:
            g += constraint.g
        
        problem = {
            'f': self.cost,
            'x': ca.vertcat(*self.opt_var),
            'g': ca.vertcat(*g)
        }
        
        if solver == 'ipopt':        
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
                'ipopt.print_level': 3,
                'print_time': True,
                # 'ipopt.warm_start_init_point': 'yes',
            }
                
        elif solver == 'sqpmethod':
            opts = {
                'qpsol': 'qrqp', 
                'qpsol_options': {'print_iter': False, 'error_on_fail': False}, 
                'print_time': False,
            }
            
        elif solver == 'fatrop':
            opts = {
                'structure_detection': 'auto',
                'debug': True,
                'equality': [True for _ in range(self.n_states * (self.n_phases)+1)],
            }
            
        else:
            raise ValueError(f"Solver {solver} is not supported.")
            
        self.solver = ca.nlpsol('solver', solver, problem, opts)
                         
    def solve(self):
        lbg = np.empty(0)
        ubg = np.empty(0)
        for constraint in self.constraints:
            lbg = np.concatenate((lbg, constraint.lbg))
            ubg = np.concatenate((ubg, constraint.ubg))
        
        r = self.solver(
            x0=self.opt_var_0,
            lbx=self.lb_opt_var.tolist(), ubx=self.ub_opt_var.tolist(),
            lbg=lbg, ubg=ubg,
        )
        
        sol = r['x'].full().flatten()
        
        self.opt_var_0 = sol
        
        states_opt = []
        inputs_opt = []
        
        for i in range(0, self.n_opti, self.shift):
            if self.multiple_shooting:
                states_opt.extend(sol[i:i+self.n_states])
                inputs_opt.extend(sol[i+self.n_states:i+self.n_states+self.n_inputs])
        
            else:
                inputs_opt.extend(sol[i:i+self.n_inputs])
            
        deltas_opt = sol[self.shift-1::self.shift]
        
        
        # inputs_opt = sol[:self.n_inputs*self.n_phases]
        # deltas_opt = sol[self.n_inputs*self.n_phases:self.n_inputs*self.n_phases + self.n_phases]
        
        # print(f"Optimal control input: {inputs_opt}")
        # print(f"Optimal phase durations: {deltas_opt}")
        # print(f"Optimal switching instants: {np.cumsum(deltas_opt)}")

        return inputs_opt, deltas_opt, states_opt
    
    def step(self, Q, R, x0, xf=None, E=None):
        self._propagate_state(x0)
        
        # Set bounds for the first state
        self.lb_opt_var[:self.n_states] = x0
        self.ub_opt_var[:self.n_states] = x0
        
        self.set_cost_function(Q, R, x0, xf, E)
        
        self.create_solver()
        
        return self.solve()
