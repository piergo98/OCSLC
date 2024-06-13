import sympy as sp
import numpy as np
import casadi as ca
from Switched_Linear import SwiLin

opt_points = 10
nx = 2
nu = 1
swi_lin = SwiLin(opt_points, nx, nu)

def matrix_exponential(A, t, n=50):
    """
    Compute the symbolic matrix exponential of the input matrix A.
    
    Parameters:
    A (sympy.Matrix): The input matrix.
    t (sympy.Symbol): Time variable.
    n (int): The number of terms to include in the Taylor series approximation (default is 50).
    
    Returns:
    sympy.Matrix: The symbolic matrix exponential of A.
    """
    # Check if the input matrix is square
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    # Initialize the result matrix
    A_t = sp.Matrix(A * t)
   
    # Compute the matrix exponential
    expA = A_t.exp()
    
    return expA

def integrate_matrix(F, x, a, b):
    """
    Integrates a SymPy matrix F(x) with respect to the variable x
    from a to b.
    
    Args:
        F (sympy.Matrix): The matrix to integrate.
        x (sympy.Symbol): The variable to integrate with respect to.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.
    
    Returns:
        sympy.Matrix: The result of the integration.
    """
    # Get the dimensions of the matrix
    rows, cols = F.shape
    
    # Initialize the result matrix
    result = sp.zeros(rows, cols)
    
    # Integrate each element of the matrix
    result = sp.integrate(F, (x, a, b))
    
    return result
    
# Example usage
# Define the symbolic variables
x, y, z = sp.symbols('x y z')

# # Define the function to integrate
# f = x**2 + y*x + z

# # Integrate with respect to x, treating y and z as constants
# result = integrate_symbolic(f, x, 0, 2)
# pippo = sp.lambdify([x, y, z], result, "numpy")

# -------------------------------------------------------------------------------------------

A = sp.MatrixSymbol('A', swi_lin.Nx, swi_lin.Nx)

expA = matrix_exponential(A, y-x)
expm = sp.lambdify([A, x, y], expA, "numpy")

int_exponential = integrate_matrix(expA, x, 0, 2)
# int_exp = sp.lambdify([A, x, y])



 # Load the model
model = {
    'num_modes': 2,
    'A': [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
    'B': [np.array([1, 2]), np.array([3, 4])]
}
swi_lin.load_model(model)


# expA = matrix_exponential(A, x)
# expm = sp.lambdify([A, x], expA)

t = ca.SX.sym('t')
A_ = swi_lin.A[0]
delta = swi_lin.delta[0]

# A_x = sp.Matrix(A_*x)
# expA_x = A_x.exp()

print(int_exponential)
# print(expA[0,0])
# print(int_exp(A_, t, delta))

# print(pippo(1, x_ca, 3))
