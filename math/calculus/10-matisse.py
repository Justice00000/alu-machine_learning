'''
'''


def poly_derivative(poly):
    '''

    '''
    if len(poly) < 2:
        return [0]
    
    return [poly[i] * i for i in range(1, len(poly))]
