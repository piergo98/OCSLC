# CLAUDE.md

## Project Overview

OCSLC (Optimal Control of Switched Linear Systems with CasADi) is a Python library for solving optimal control problems on switched linear systems. It simultaneously optimizes control inputs and switching instants using matrix exponential computations, formulated as NLPs solved via CasADi/IPOPT.

## Architecture

### Core Classes (inheritance chain)

- **`SwiLin`** (`ocslc/switched_linear.py`): Base class. Handles model loading, matrix exponential computation (Van Loan method), state propagation (`'exp'` or `'int'` modes), cost function construction via symbolic CasADi expressions, and gradient/Hessian computation. Modes cycle over phases: `mode_index = i % len(model['A'])`.

- **`SwitchedLinearMPC`** (`ocslc/switched_linear_mpc.py`): Inherits `SwiLin`. Builds the NLP: defines CasADi symbolic optimization variables (controls `U_i`, phase durations `Delta_i`, and optionally states `X_i` for multiple shooting), assembles cost and constraints, creates IPOPT solver, and extracts solutions. Supports both single-shooting and multiple-shooting formulations.

- **`RecedingHorizonMPC`** (`ocslc/receding_horizon_mpc.py`): Wraps `SwitchedLinearMPC` for receding-horizon use. Manages dwell-time aging, mode switching when phases exhaust, and warm-starting across MPC steps.

### Optimization Vector Layout

The optimization vector packs variables per-phase with stride `shift`:
- **Single shooting**: `[U_0, Delta_0, U_1, Delta_1, ...]` (shift = n_inputs + 1)
- **Multiple shooting**: `[X_0, U_0, Delta_0, X_1, U_1, Delta_1, ..., X_N]` (shift = n_states + n_inputs + 1)

### Key Concepts

- **Model**: A dict `{'A': [A_0, ...], 'B': [B_0, ...]}` where each pair defines a linear mode `dx = A_i x + B_i u`.
- **Propagation**: `'exp'` uses matrix exponential (exact for piecewise-constant inputs); `'int'` uses numerical integration (Simpson's rule).
- **`auto=True`**: Autonomous system (no control inputs, optimize switching instants only).
- **`inspect=True`**: Debug mode that fixes phase durations to diagnose solver issues.
- **`hybrid=True`**: Uses simplified (Euler-like) cost terms instead of full Van Loan integral.
- **Constraints**: Managed via `SwitchedLinearMPC.Constraint` objects; total-time constraint `sum(deltas) = T` is always added automatically.

## Notes

- The IPOPT solver config in `create_solver` has a hardcoded HSL library path (`/home/pietro/...`). This needs updating per machine or making configurable.
- Some tests are marked `@pytest.mark.skip()` and require arguments to run.
- Examples in `examples/` demonstrate various problem setups (autonomous, constrained, time-varying constraints, two-stage optimization).
