'''
Title: Slice like a Ninja | Python
'''


def np_slice(matrix, axes={}):
    """
    Slices a matrix (nested list) along specific axes.
    The slicing is similar to numpy slicing.
    """
    def apply_slice(data, slices, dim=0):
        if not isinstance(data, list):
            return data
        if dim in slices:
            start, stop, step = slices[dim]
            data = data[start:stop:step]
        return [apply_slice(item, slices, dim + 1) for item in data]

    return apply_slice(matrix, axes)
