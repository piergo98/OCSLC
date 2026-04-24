# OCSLC

A Python library for solving optimal control problems on linear systems using matrix exponential computations. It formulates the problem as an NLP (solved via CasADi/IPOPT), simultaneously optimizing control inputs and phase durations.

## Installation

```shell
pip install -e .
```

To run tests:
```shell
pip install -e ".[test]"
pytest
```

## Problem Formulation

The library solves optimal control problems of the form:

```math
\begin{aligned}
\min_{u, \delta} \quad & \int_{0}^{T} \frac{1}{2} \left(u(t)^\top R\, u(t) + x(t)^\top Q\, x(t) \right) \textnormal{d}t + x(T)^\top E\, x(T)\\
\text{s.t.} \quad  & \dot{x}(t) = A_ix(t) + B_iu_i, \quad t \in [\tau_i, \, \tau_{i+1}] \\
& x(0) = x_0 \\
& u \in \mathcal{U} \\
& x \in \mathcal{X} \\
& \delta \in \Delta
\end{aligned}
```

The dynamics are piecewise-linear with constant control inputs per phase. The matrix exponential (Van Loan method) provides exact propagation of the state, while both control inputs and phase durations are optimized simultaneously.

Two shooting methods are supported:
- **Single shooting** (`--shooting ss`): propagates the state forward, optimizing only controls and durations.
- **Multiple shooting** (`--shooting ms`): adds state variables at phase boundaries for better numerical conditioning.

Two propagation modes are available:
- **Matrix exponential** (`--integrator exp`): exact for piecewise-constant inputs (default).
- **Numerical integration** (`--integrator int`): uses Simpson's rule.

## Examples

Most example scripts accept common command-line arguments: `--integrator {int,exp}`, `--shooting {ss,ms}`, `--hybrid`, and `--n_steps N`.

### Stiff System

A simple constrained optimal control problem with state and input bounds. Visualizes optimal costs, trajectories, and computation time.

```shell
python examples/stiff_system.py --integrator exp --shooting ss
```

### Cart-Pole

MPC for the classic cart-pole balancing problem.

```shell
python examples/cartpole.py --integrator exp --shooting ms
```

### Stable System Benchmark

A reference benchmark problem comparing different MPC methods.

```shell
python examples/stable_sys_benchmark.py --hybrid --steps 10
```

## Citation

TODO