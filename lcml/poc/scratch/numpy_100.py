#!/usr/bin/env python3
"""Numpy exercises from https://github.com/rougier/numpy-100"""
import numpy as np


def t2():
    print(np.__version__)
    np.show_config()

def t3():
    return np.zeros(10)

def t4():
    return np.info(np.add)

def t5():
    z = np.zeros(10)
    z[4] = 1
    return z

def t6():
    return np.arange(10, 50)

def t7():
    """reverse"""
    z = np.arange(50)
    return z[::-1]

t8 = lambda: np.arange(9).reshape(3, 3)

t9 = lambda: np.nonzero([1, 2, 0, 0, 4, 0])

t10 = lambda: np.eye(3)

t11 = lambda: np.random.random((3, 3, 3))

def t12():
    z = np.random.random((10, 10))
    return z.min(), z.max()

t13 = lambda: np.random.random(30).mean()

def t14():
    """Ones on border; 0's elsewhere"""
    z = np.ones((10, 10))
    z[1: -1, 1: -1] = 0
    return z

def t16():
    """Create a 5x5 matrix with 1,2,3,4 just below diagonal"""
    return np.diag(np.arange(4) + 1, k=-1)

def t17():
    """Checkerboard"""
    z = np.zeros((8, 8), dtype=int)

    # format: [dim0-start-index::dim0-steps, dim1-start-index::dim1-steps]
    # 0 can be omitted like in standard python
    z[1::2, ::2] = 1
    z[::2, 1::2] = 1
    return z

t18 = lambda: np.unravel_index(100, (6, 7, 8))

#: create a large array by tiling the first argument array
t19 = lambda: np.tile(np.array([[0, 1], [1, 0]]), (4, 4))

#: normalize
def t20():
    z = np.random.random((5, 5))
    zMin = z.min()
    return (z - zMin) / (z.max() - zMin)

t21 = lambda: np.dtype([("r", np.ubyte, 1), ("g", np.ubyte, 1),
                        ("b", np.ubyte, 1), ("a", np.ubyte, 1)])

#: matrix mult
t22 = lambda: np.dot(np.ones((5, 3)), np.ones((3, 2)))

def t23():
    """negative all elements between 3 and 8 in place"""
    z = np.arange(11)
    # logical indexing
    z[(3 < z) & (z <= 8)] *= -1
    return z

def t27():
    # round a random float array away from zero
    z = np.random.uniform(-10, 10, 10)

    # copy sign takes signs of z and applies them to 0.5
    return np.trunc(z + np.copysign(0.5, z))

def t28():
    # extract integer part
    z = np.random.uniform(0, 10, 4)
    return (z - z % 1,
            np.floor(z),
            np.ceil(z) - 1,
            z.astype(int),
            np.trunc(z))

def t29():
    z = np.zeros((5, 6))
    z += np.arange(6)
    return z

def t30():
    def gen():
        for x in range(10):
            yield x

    return np.fromiter(gen(), dtype=float, count=-1)

t31 = lambda: np.linspace(0, 1, 12, endpoint=True)[1: -1]

def t32():
    Z = np.random.random(10)
    Z.sort()
    return Z

def t33():
    # sum small array
    z = np.arange(10)
    return np.add.reduce(z)

def t34():
    a = np.random.randint(0, 2, 5)
    b = np.random.randint(0, 2, 5)
    return np.allclose(a, b)

def t35():
    z = np.zeros(10)
    z.flags.writeable = False
    try:
        z[0] = 1
    except ValueError:
        return "caught value error"

    return "didn't catch"

def t36():
    """cartesian to polar coordinates"""
    z = np.random.random((10, 2))
    x, y = z[:, 0], z[:, 1]
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(y, x)
    return r, t

def t37():
    z = np.random.random(10)
    z[z.argmax()] = 0
    return z

def t38():
    """structured array"""
    z = np.zeros((10, 10), [('x', float), ('y', float)])
    z['x'], z['y'] = np.meshgrid(np.linspace(0, 9, 10), np.linspace(0, 9, 10))
    return z

def t39():
    x = np.arange(8)
    y = x + .5
    interm = np.subtract.outer(x, y)
    c = 1.0 / interm
    return np.linalg.det(c)

def t40():
    for dtype in [np.int8, np.int32, np.int64]:
        print(np.iinfo(dtype).min)
        print(np.iinfo(dtype).max)

    for dtype in [np.float32, np.float64]:
        print(np.finfo(dtype).min)
        print(np.finfo(dtype).max)
        print(np.finfo(dtype).eps)

def t41():
    # Print all array values
    np.set_printoptions(threshold=np.nan)
    return np.zeros((25, 25))

def t42():
    # find closest value in an array
    z = np.arange(100)
    v = np.random.uniform(0, 100)
    index = np.abs(z - v).argmin()
    return v, z[index]

def t43():
    """structured array representing a position (x,y) and a color (r,g,b)"""
    return np.zeros(10, [('position', [('x', float, 1),
                                       ('y', float, 1)]),
                         ('color', [('r', float, 1),
                                    ('g', float, 1),
                                    ('b', float, 1)])])

def t44():
    """find point-by-point distances"""
    from scipy import spatial
    Z = np.random.random((10, 2))
    return spatial.distance.cdist(Z, Z)


def t45():
    """in-place conversion from float to int"""
    z = np.arange(10, dtype=np.int32)
    return z.astype(np.float32, copy=False)


def main():
    f = t45
    print(f())


if __name__ == "__main__":
    main()
