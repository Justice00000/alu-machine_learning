#!/usr/bin/env python3
'''
Write a function that takes a list of integers as input and returns the derivative of the polynomial represented by the list.
'''


def poly_derivative(poly):
    '''
    Returns the derivative of the polynomial represented by the list poly.
    '''
    if len(poly) < 2:
        return [0]
    
    return [poly[i] * i for i in range(1, len(poly))]
