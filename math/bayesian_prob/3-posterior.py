<<<<<<< HEAD
#!/usr/bin/env python3
""""Function  that calculates the posterior
probability for the various hypothetical probabilities
of developing severe side effects given the data"""

import numpy as np


def posterior(x, n, P, Pr):
    """""Function  that calculates the posterior
    probability for the various hypothetical probabilities
    of developing severe side effects given the data"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(P, np.ndarray)) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if (not isinstance(Pr, np.ndarray)) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1.])[0]:
        raise ValueError("Pr must sum to 1")
    num = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)
    coeficient = num / den
    likehood = coeficient * (P ** x) * ((1 - P) ** (n - x))
    intersection = likehood * Pr
    return intersection / np.sum(intersection)
=======
#!/usr/bin/env python3
'''
Defines a function that calculates the posterior probability
'''
import numpy as np


def posterior(x, n, P, Pr):
    '''
    Defines a function that calculates the posterior probability
    '''
    # Input validations
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that"
                         " is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate binomial coefficient
    log_binom_coeff = (
        np.sum(np.log(np.arange(n - x + 1, n + 1))) -
        np.sum(np.log(np.arange(1, x + 1)))
    )

    # Calculate likelihood
    likelihood = np.zeros_like(P)
    for i, p in enumerate(P):
        if p == 0:
            likelihood[i] = 1.0 if x == 0 else 0.0
        elif p == 1:
            likelihood[i] = 1.0 if x == n else 0.0
        else:
            likelihood[i] = np.exp(
                log_binom_coeff +
                x * np.log(p) +
                (n - x) * np.log(1 - p)
            )

    # Calculate intersection
    intersection = likelihood * Pr

    # Calculate marginal probability
    marginal_prob = np.sum(intersection)

    # Calculate posterior probabilities
    posterior_probs = intersection / marginal_prob

    return posterior_probs
>>>>>>> eb1c0f93d156ce747d976a0c95dd86710b1286e6
