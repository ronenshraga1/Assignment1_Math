from typing import Callable

import function
from sympy import symbols, sympify




def str_to_func(function_str: str) -> Callable[[float], float]:
    x = symbols('x')
    function = sympify(function_str)
    return lambda val: float(function.subs(x, val))

    
def Secant_method(function: str, x0: float, x1: float) -> float:
    """
    Finds an approximation of a root using the Secant method.

    :param function: A string representing the function (e.g., "x**2 - 2")
    :param x0: First initial guess
    :param x1: Second initial guess
    :return: An approximate root or the best approximation after 30 iterations
    """
    graph = str_to_func(function)
    for item in range(30):
        y0 = graph(x0)
        y1=graph(x1)
        if x1-x0 == 0:
            return None
        x2 = x1 - y1 * (x1 - x0) / (y1 - y0)
        print(x2)
        x0,x1 = x1,x2
    return x2
def ge(matrix: list[list[float]]) -> list[list[float]]:
    """
    Transforms a matrix into upper triangular form using Gaussian elimination.

    :param matrix: A list of lists representing the input matrix.
    :return: A new matrix in upper triangular form (as a list of lists of floats).
    """
    new_matrix = [row[:] for row in matrix]
    for i in range(min(len(new_matrix),len(new_matrix[0]))):
        if new_matrix[i][i] == 0:
            for counter in range(i+1,len(new_matrix)):
                if new_matrix[counter][i] != 0:
                    new_matrix[i], new_matrix[counter] = new_matrix[counter], new_matrix[i]
                    break
        dividor = new_matrix[i][i]
        if dividor != 0:
            new_matrix[i] = [x / dividor for x in new_matrix[i]]
            for j in range(i+1,len(new_matrix)):
                scalar = new_matrix[j][i]
                new_matrix[j] = [x-scalar*y for x,y in zip(new_matrix[j],new_matrix[i])]
    return new_matrix
