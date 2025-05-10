#!/usr/bin/env python3
'''
<<<<<<< HEAD
    The function below calculates
    the integral of a polynomial
=======
Write a function that calculates the integral of a polynomial.
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def poly_integral(poly, C=0):
<<<<<<< HEAD
    """
    calculates the integral of the given polynomial
    """
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    for coefficient in poly:
        if type(coefficient) is not int and type(coefficient) is not float:
            return None
    if type(C) is float and C.is_integer():
        C = int(C)
    integral = [C]
    for power, coefficient in enumerate(poly):
        if (coefficient % (power + 1)) is 0:
            new_coefficient = coefficient // (power + 1)
        else:
            new_coefficient = coefficient / (power + 1)
        integral.append(new_coefficient)
    while integral[-1] is 0 and len(integral) > 1:
        integral = integral[:-1]
=======
    '''
    Function that calculates the integral of a polynomial.
    '''
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    # if not all(isinstance(coef, (int, float)) for coef in poly):
    #     return None

    if len(poly) == 0:
        return None

    # Calculate the integral
    integral = [C]  # Start with the integration constant
    for power, coef in enumerate(poly):
        new_coef = coef / (power + 1)
        # If the new coefficient is a whole number, convert it to an integer
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
    return integral
