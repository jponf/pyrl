# -*- coding: utf-8 -*-

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import numpy as np


################################################################################

def random_3d_point(x_range, y_range, z_range, rand=None):
    """Generates a random 3D point within the given ranges.

    :param x_range: A tuple with the (lower, upper) bounds of the X coordinate.
    :param y_range: A tuple with the (lower, upper) bounds of the Y coordinate.
    :param z_range: A tuple with the (lower, upper) bounds of the Z coordinate.
    :param rand: A random generator object with a `uniform(low, high)` method.

    :return: A numpy array with 3 elements: (x, y, z) coordinates.
    """
    if rand is None:
        rand = np.random

    return np.array([rand.uniform(*x_range),
                     rand.uniform(*y_range),
                     rand.uniform(*z_range)])


def scale(x, min_x, max_x, min_out, max_out):
    """Scales `x` that is a value known to be in range `[min_x, max_x]` to
    the corresponding value in the range `[min_out, max_out]`.

    :param x: Value to scale from [min_x, max_x] to [min_out, max_out].
    :param min_x: Minimum possible value of `x`.
    :param max_x: Maximum possible value of `x`.
    :param min_out: Minimum possible value of the output value.
    :param max_out: Maximum possible value of the output value.

    :return: `x` scaled to be between [min_out, max_out].
    """
    scaling_factor = (max_out - min_out) / (max_x - min_x)
    translation = x - min_x
    scaled = translation * scaling_factor + min_out
    return np.clip(scaled, min_out, max_out)
