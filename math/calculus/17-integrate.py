#!/usr/bin/env python3
<<<<<<< HEAD
'''
    The function below calculates
    the integral of a polynomial
'''


def poly_integral(poly, C=0):
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
    return integral
=======
"""_summary_
This file contains the implementation of poly_integral
"""


def poly_integral(poly, C=0):
    """_summary_
    Computes the coefficients of the terms in the
    integral of a function using Sum rule
    """
    if not isinstance(poly, list) or not isinstance(C, (int, float)) or not poly:  # noqa
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integrals = [C]
    for power, coefficient in enumerate(poly):
        if power == 0:
            integrals.append(coefficient)
        else:
            integral = coefficient / (power + 1)
            integrals.append(
                int(integral) if integral.is_integer() else integral
            )
    while integrals[-1] == 0 and len(integrals) > 1:
        integrals.pop()
    return integrals
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
