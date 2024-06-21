import casadi as ca
import numpy as np

from .switched_linear import SwiLin



class SwitchedLinearMPC(SwiLin):
    def __init__(self, model, n_phases, time_horizon, auto=False) -> None:
        self._check_model_structure(model)
        n_states = model['A'][0].shape[0]
        n_inputs = model['B'][0].shape[1]
        
        super().__init__(n_phases, n_states, n_inputs, auto)
        self.load_model(model)
        
        self.time_horizon = time_horizon
        
        if not auto:
            self.inputs = [ca.MX.sym(f"U_{i}", n_inputs) for i in range(n_phases)]
            self.inputs_0 = [np.zeros(n_inputs) for _ in range(n_phases)]
        else:
            self.inputs = []
            self.inputs_0 = [np.zeros(0)]
        
        self.deltas = ca.MX.sym("delta", n_phases)
        self.deltas_0 = np.ones(n_phases) * time_horizon / n_phases
        
        self.opt_var = self.inputs + [self.deltas]
        self.opt_var_0 = self.inputs_0 + [self.deltas_0]
        
        self.lb_opt_var = - np.ones(self.n_opti) * np.inf
        self.ub_opt_var =   np.ones(self.n_opti) * np.inf
        
        self.lb_opt_var[self.n_inputs*self.n_phases:] = 0
        self.ub_opt_var[self.n_inputs*self.n_phases:] = time_horizon
        
        self.cost = 0
        self.g = []
        self.lbg = []
        self.ubg = []
        
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
        self.g += [ca.sum1(self.deltas)]
        self.lbg += [self.time_horizon]
        self.ubg += [self.time_horizon]
        
    def add_constraints(self, g, lbg, ubg):
        self.g += g
        self.lbg += lbg
        self.ubg += ubg
        
    def set_cost_function(self, R, x0):
        x0_aug = np.concatenate((x0, [1]))
        
        cost = self.cost_function(R, x0_aug)
                
        if self.n_inputs == 0:
            self.cost = cost(self.deltas)
        else:
            self.cost = cost(*self.opt_var)
        
    def create_solver(self):
        problem = {
            'f': self.cost,
            'x': ca.vertcat(*self.opt_var),
            'g': ca.vertcat(*self.g)
        }
                
        opts = {
            'ipopt.max_iter': 5e3,
        }
                        
        self.solver = ca.nlpsol('solver', 'ipopt', problem, opts)
        
    def solve(self):
        r = self.solver(
            x0=np.concatenate(self.opt_var_0),
            lbx=self.lb_opt_var.tolist(), ubx=self.ub_opt_var.tolist(),
            lbg=self.lbg, ubg=self.ubg
        )
        
        sol = r['x'].full().flatten()
        
        opt_inputs = sol[:self.n_inputs*self.n_phases]
        opt_deltas = sol[self.n_inputs*self.n_phases:self.n_inputs*self.n_phases + self.n_phases]
        
        print(f"Optimal control input: {opt_inputs}")
        print(f"Optimal phase durations: {opt_deltas}")
        print(f"Optimal switching instants: {np.cumsum(opt_deltas)}")

        return opt_inputs, opt_deltas
