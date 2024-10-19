#!/usr/bin/env python3
import numpy as np


def likelihood(x, n, P):
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer \
            that is greater than or equal to 0")

    # Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in the range [0, 1]
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    log_coeff = np.sum(np.log(np.arange(n - x + 1, n + 1)))
    - np.sum(np.log(np.arange(1, x + 1)))
    coeff = np.exp(log_coeff)

    # Calculate the likelihood using the binomial probability mass function
    return coeff * (P ** x) * ((1 - P) ** (n - x))
