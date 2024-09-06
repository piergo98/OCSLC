# OCSLC
This Python class addresses Switched Linear Systems optimization exploiting matrix exponential computation and optimizing over controls and switching instants at the same time.

## Installation

Run
```shell
pip install -e .
```
Alternatively, you can create your own virtualenv or conda env and run this package inside it. Remember to install all the required Python packages.

## Explanation
The class provides methods to instantiate and solve an optimization problem, in a single shot or in the MPC fashion, in the following form:

```math
\begin{aligned}
\min_{u, \delta} \quad & \int_{0}^{T} \frac{1}{2} \left(u(t)^\top R u(t) + x(t)^\top Q x(t) \right) \; \textnormal{d}t + x(T)^\top E x(T)\\
\text{s.t.} \quad  & \dot{x}(t) = A_ix(t) + B_iu_i, \quad t \in [\tau_i, \, \tau_{i+1}] \\
& x(0) = x_0 \\
& u \in U \\
& \delta \in \Delta
\end{aligned}
```

We formulate the optimization problem leveraging the matrix exponential computation of a linear switched system in order to optimize both the control input of each phase and the switching instants.
This class introduces the possibility to optimize simultaneously both controls and switching instants, which usually is addressed in two different optimization.

## Usage


### TO DO
- Complete the README
- Manage warm start for MPC
- Compute matrix exponential using Padè approximation (low priority)