<<<<<<< HEAD
#!/usr/bin/env python3
"""Calculates the posterior probability that the probability of developing
severe side effects falls within a specific range given the data
"""


from scipy import special


def posterior(x, n, p1, p2):
    """ Returns: the posterior probability that p is within the range [p1, p2]
        given x and n
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) is not int or x < 0:
        error = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(error)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')
    return special.btdtr(x+1, n-x+1, p2) - special.btdtr(x+1, n-x+1, p1)
=======
#!/usr/bin/env python3
'''
Calculates the posterior probability that the probability
of developing severe side effects is within
a specific range given the data
'''
from scipy import special


def posterior(x, n, p1, p2):
    '''
    Calculates the posterior probability that the probability
    '''
    # Input validations
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Calculate the parameters of the Beta distribution
    alpha = x + 1
    beta = n - x + 1

    # Calculate the cumulative probability
    # using the regularized incomplete beta function
    cdf_p2 = special.betainc(alpha, beta, p2)
    cdf_p1 = special.betainc(alpha, beta, p1)

    # Calculate the posterior probability
    posterior_prob = cdf_p2 - cdf_p1

    return posterior_prob
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
