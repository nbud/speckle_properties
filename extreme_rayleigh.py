"""
Plot distribution of n iid Rayleigh distributed variables for various n

Show the asymptotic is a Gumbel distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def rayleigh_logpdf(x):
    return np.log(x) - (x * x) / 2


def rayleigh_logcdf(x):
    return np.log(1 - np.exp(-(x * x) / 2))


def maxrayleigh_logcdf(x, n, sigma):
    """
    cdf of max of n iid Rayleigh distribution
    """
    return n * rayleigh_logcdf(x / sigma)


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

#%% The mode of MaxRayleigh increases in sqrt(log(n))

sigma = 1.0
t = np.linspace(0, 8, 2000)[1:]
nvect = np.array([100, 1000, 10000, 100000, 200000, 500000, 1000000])
nvect = 10 ** np.linspace(2, 6, 20)
modes = np.zeros_like(nvect)
for k, n in enumerate(nvect):
    modes[k] = t[np.argmax(np.exp(maxrayleigh_logpdf(t, n, sigma)))]

plt.figure()
plt.plot(modes, np.sqrt(np.log(nvect)), ".-")
plt.xlabel("sqrt(log(n))")
plt.ylabel("mode")
# plt.gca().set_xscale("log")


#%% Convergence of (M-b)/a towards standard Gumbel for one x

sigma = 1.0


def rayleigh_cdf(x):
    return 1 - np.exp(-x ** 2 / (sigma ** 2 * 2))


def asymptotic_rayleigh(x, n):
    a = sigma / np.sqrt(2 * np.log(n))
    b = sigma * np.sqrt(2 * np.log(n))
    return rayleigh_cdf(a * x + b) ** n


def asymptotic_exponential(x, n):
    # Exponential distribution
    return (1 - np.exp(-x) / n) ** n


def gumbel_cdf(x):
    return np.exp(-np.exp(-x))


def gumbel_pdf(x):
    return np.exp(-(x + np.exp(-x)))


x = 1.0
plt.figure()
nvect = 10 ** np.linspace(1, 8, 20)
plt.plot(nvect, asymptotic_rayleigh(x, nvect))
# plt.plot(nvect, asymptotic_exponential(x, nvect))
plt.axhline(gumbel_cdf(x), color="C1")
plt.gca().set_xscale("log")

#%% Convergence of (M-b)/a towards standard Gumbel for cdf
sigma = 1.0
t = np.linspace(0, 8, 1000)
plt.figure()
for n in [10, 100, 1000, 1e5, 1e6, 1e7, 1e8]:
    a = sigma / np.sqrt(2 * np.log(n))
    b = sigma * np.sqrt(2 * np.log(n))
    plt.plot(t, np.exp(maxrayleigh_logcdf(a * t + b, n, sigma)), label=f"n={n}")
plt.plot(t, gumbel_cdf(t), "k--", label="Gumbel")
plt.legend()

#%% Convergence of (M-b)/a towards standard Gumbel for pdf
sigma = 1.0
t = np.linspace(0, 8, 1000)
plt.figure()
for n in [10, 100, 1000, 10000]:
    a = sigma / np.sqrt(2 * np.log(n))
    b = sigma * np.sqrt(2 * np.log(n))
    plt.plot(t, a * np.exp(maxrayleigh_logpdf(a * t + b, n, sigma)), label=f"n={n}")
plt.plot(t, gumbel_pdf(t), "k--", label="Gumbel")
plt.legend()

#%% Convergence of M towards Gumbel for pdf

sigma = 1.0
t = np.linspace(0, 8, 1000)[1:]
plt.figure()
for n in [10, 100, 1000, 10000]:
    a = sigma / np.sqrt(2 * np.log(n))
    b = sigma * np.sqrt(2 * np.log(n))
    plt.plot(t, np.exp(maxrayleigh_logpdf(t, n, sigma)), label=f"n={n}")
    plt.plot(t, stats.gumbel_r(loc=b, scale=a).pdf(t), "k--", label=f"Gumbel n={n}")
plt.legend()
