#!/usr/bin/env python3
'''
<<<<<<< HEAD
    this function
    calculates the derivative of a polynomial
=======
Write a function that takes a list of integers as input
and returns the derivative of the polynomial represented by the list.
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
'''


def poly_derivative(poly):
    '''
<<<<<<< HEAD
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
    Returns the derivative of the polynomial represented by the list poly.
    '''
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power, coeff in enumerate(poly[1:], start=1):
        derivative.append(coeff * power)

    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative if any(derivative) else [0]
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
