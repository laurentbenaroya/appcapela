# -*- coding: utf-8 -*-
"""
Symmetric alpha-stable random variable generator

Popa, S., D. Stanescu, and S. S. Wulff.
"Numerical Generation of Symmetric alpha-stable random variables."
STDIA Univ. BABES-BOLYAI, Mathematica 53.3 (2008): 105-117.
"""
# Author : E.L. Benaroya - laurent.benaroya@gmail.com
# 03/2019
# License : GNU GPL v3

import numpy as np

# Popa's tables
# alpha
popa_a = np.array([0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3,
                  1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
# sigma_x1 and sigma_x2
popa_s1 = np.array([0.88, 0.93, 0.971, 1., 0.951, 0.855,
                   0.8, 0.729, 0.61, 0.396, 0.28, 0.11, 0.001])
popa_s2 = np.array(
    [2.000e-03, 1.000e-03, 0.000e+00, 2.700e-02, 2.150e-01, 4.100e-01, 5.230e-01, 6.450e-01, 8.000e-01, 1.008, 1.100,
     1.200, 1.231])


def alpha_rnd(alpha, row=1, col=1, plot_hist=False, verbose=False):
    """
    Popa's algorithm (fast generator)
    Parameters
    ----------
    alpha
    row
    col
    plot_hist
    verbose

    Returns
    -------
    v : numpy array (row, col)
        output alpha stable variable
    """
    alpha0 = alpha
    alpha = np.round(alpha * 10) / 10

    if alpha != alpha0:
        raise Exception("alpha must have a single decimal")

    if alpha < 0.7 or alpha > 1.9:
        raise Exception("Unsupported alpha value, 0.7 <= alpha <= 1.9")
    # index in tables
    inda = np.argwhere(popa_a == alpha)

    # tabulated values
    s1 = popa_s1[inda]
    s2 = popa_s2[inda]
    if verbose:
        print("alpha : %.2f, s1 : %.3f, s2 : %.3f" % (alpha, s1, s2))

    # generate four normal variables
    x1 = s1 * np.random.randn(row, col)
    y1 = np.random.randn(row, col)

    x2 = s2 * np.random.randn(row, col)
    y2 = np.random.randn(row, col)

    # compute v
    v1 = x1 / np.power(np.abs(y1), 1 / alpha)
    v2 = x2 / np.power(np.abs(y2), 1 / alpha / 2)
    v = v1 + v2

    if plot_hist:
        plt.hist(v.flatten(), bins=np.linspace(-5, 5, 100))
        plt.show()

    return v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w1 = alpha_rnd(0.8, 1025, 15000)
    w2 = alpha_rnd(1.5, 1025, 15000)
    w3 = np.random.randn(1025, 15000)

    h1, b = np.histogram(w1.flatten(), np.linspace(-5, 5, 50), density=True)
    h2, _ = np.histogram(w2.flatten(), np.linspace(-5, 5, 50), density=True)
    h3, _ = np.histogram(w3.flatten(), np.linspace(-5, 5, 50), density=True)

    plt.plot(b[:-1], np.log10(h1), label='0.8')
    plt.plot(b[:-1], np.log10(h2), label='1.5')
    plt.plot(b[:-1], np.log10(h3), label='2.0')
    plt.legend()
    plt.show()
