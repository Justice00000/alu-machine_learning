#!/usr/bin/env python3
<<<<<<< HEAD
'''
    this function
    calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
        calculates the derivative of a polynomial
    '''
    if not isinstance(poly, list) or not poly:
        return None
    for coefficient in poly:
        if not isinstance(coefficient, (int, float)):
            return None

    # Calculate derivative
    if len(poly) == 1:
        return [0]
    derivative = [
        coefficient * power
        for power, coefficient in enumerate(poly)
    ][1:]
    return derivative


# Example usage:
if __name__ == "__main__":
    poly = [5, 3, 0, 1]  # Represents the polynomial 5 + 3x + x^3
    print(poly_derivative(poly))
=======
"""_summary_
This file contains the implementation of poly_derivative
"""


def poly_derivative(poly):
    """_summary_
    Computes the coefficients of the terms in the
    derivative of a function using Sum rule
    """
    if not isinstance(poly, list) or not poly:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]

    derivatives = []
    for power, coefficient in enumerate(poly):
        if power == 0:
            continue
        derivatives.append(power * coefficient)

    return derivatives
>>>>>>> b97810dfb28d5b1f54da638ce77b8c8bcde0000c
