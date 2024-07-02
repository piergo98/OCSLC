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
    
    def __init__(self, model, n_phases, time_horizon, auto=False) -> None:
        self._check_model_structure(model)
        n_states = model['A'][0].shape[0]
        n_inputs = model['B'][0].shape[1]
        
        super().__init__(n_phases, n_states, n_inputs, time_horizon, auto)
        self.load_model(model)
                
        if not auto:
            self.inputs = [ca.MX.sym(f"U_{i}", n_inputs) for i in range(n_phases)]
        else:
            self.inputs = []
        
        self.deltas = ca.MX.sym("delta", n_phases)
        
        self.opt_var = self.inputs + [self.deltas]
        self.opt_var_0 = np.array(
            [0] * self.n_inputs * self.n_phases \
            + [time_horizon / n_phases] * self.n_phases
        )
        
        self.lb_opt_var = - np.ones(self.n_opti) * np.inf
        self.ub_opt_var =   np.ones(self.n_opti) * np.inf
        
        self.lb_opt_var[self.n_inputs*self.n_phases:] = 0
        self.ub_opt_var[self.n_inputs*self.n_phases:] = time_horizon
        
        self.cost = 0
        self.constraints = []
        
        self._set_constraints_deltas()
        
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
        
    def set_bounds(self, inputs_lb, inputs_ub):
        self.lb_opt_var[0:self.n_inputs*self.n_phases] = inputs_lb
        self.ub_opt_var[0:self.n_inputs*self.n_phases] = inputs_ub
                
    def _set_constraints_deltas(self):
        self.constraints.append(self.Constraint(
            g=[ca.sum1(self.deltas)],
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
        
    def set_cost_function(self, R, x0, xr=None, E=None):
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
            self.cost = cost(self.deltas)
        else:
            self.cost = cost(*self.opt_var)
        
    def create_solver(self):
        g = []
        for constraint in self.constraints:
            g += constraint.g
        
        problem = {
            'f': self.cost,
            'x': ca.vertcat(*self.opt_var),
            'g': ca.vertcat(*g)
        }
                
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
            'ipopt.print_level': 3
        }
                        
        self.solver = ca.nlpsol('solver', 'ipopt', problem, opts)
        
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
        
        inputs_opt = sol[:self.n_inputs*self.n_phases]
        deltas_opt = sol[self.n_inputs*self.n_phases:self.n_inputs*self.n_phases + self.n_phases]
        
        print(f"Optimal control input: {inputs_opt}")
        print(f"Optimal phase durations: {deltas_opt}")
        print(f"Optimal switching instants: {np.cumsum(deltas_opt)}")

        return inputs_opt, deltas_opt
    
    def step(self, R, x0, xf=None, E=None):
        self._propagate_state(x0)
        
        self.set_cost_function(R, x0, xf, E)
        
        self.create_solver()
        
        return self.solve()
