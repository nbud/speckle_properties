"""
Plot distribution of n iid Rayleigh distributed variables for various n
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def rayleigh_logpdf(x):
    return np.log(x) - (x * x) / 2


def rayleigh_logcdf(x):
    return np.log(1 - np.exp(-(x * x) / 2))


def maxrayleigh_logpdf(x, n, sigma):
    """
    pdf of max of n iid Rayleigh distribution
    """
    return (
        np.log(n)
        + rayleigh_logpdf(x / sigma)
        + (n - 1) * rayleigh_logcdf(x / sigma)
        - np.log(sigma)
    )


def test_maxrayleigh_logpdf():
    neff = 10
    sigma = 2.0
    m = 2000
    np.random.seed(123)
    samples = [np.max(stats.rayleigh.rvs(size=neff, scale=sigma)) for i in range(m)]
    t = np.linspace(1, 10, 100)
    plt.figure()
    plt.hist(samples, bins=30, density=True)
    plt.plot(t, np.exp(maxrayleigh_logpdf(t, neff, sigma)))
    plt.title("test_maxraleigh_logpdf")


test_maxrayleigh_logpdf()

#%%

sigma = 1.0
t = np.linspace(0, 8, 1000)
plt.figure()
for n in [10, 100, 1000, 10000, 100000, 1000000]:
    plt.plot(t, np.exp(maxrayleigh_logpdf(t, n, sigma)), label=f"n={n}")
plt.legend()
plt.title("distribution of n iid Rayleigh")
