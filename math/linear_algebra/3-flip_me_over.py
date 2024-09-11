#!/usr/bin/env python3
def matrix_transpose(matrix):
    # Transpose the matrix using zip and unpacking
    return [list(row) for row in zip(*matrix)]
