# -*- coding: utf-8 -*-
"""
Generate speckle Z = X + i J

Results:
    Abs(Z) is Rayleigh(Sigma)
    X is normally distribution Norm(0, Sigma^2) (but the variance is wiggly)


Method 1: FFT and threshold at 1/e
The correlation length is around half the wavelength
Fundamental flaw: catches not only the main cluster by further one


Padding the FFT reduces the variance of the ACA

Final method: FFT padded + label identification
ACA ~ (wavelength / 2)^2
Number of unique points ~ 1/(wavelength / 2)^2


"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numba
import math
import tqdm
from scipy import ndimage
import functools
import pandas as pd

pd.set_option("display.max_columns", None)
#%%

x = np.linspace(0, 1, 200, endpoint=False)
y = np.linspace(0, 1, 200, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
xx, yy = np.meshgrid(x, y)
sigma = 1.0


@numba.njit(parallel=True)
def _make_field(xx, yy, xp, yp, complex_amps, wavelength):
    res = np.zeros(xx.shape, np.complex_)
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            for k in range(xp.shape[0]):
                r = math.sqrt((xx[i, j] - xp[k]) ** 2 + (yy[i, j] - yp[k]) ** 2)
                res[i, j] += complex_amps[k] * np.exp(2j * np.pi / wavelength * r)
    return res


def make_field(n=500, wavelength=0.1):
    rootn = np.sqrt(n)

    xp = stats.uniform(-1, 3.0).rvs(size=n)
    yp = stats.uniform(-1, 3.0).rvs(size=n)

    complex_amps = stats.norm(scale=sigma / rootn).rvs(2 * n).view(np.complex_)
    """
    plt.figure()
    plt.hist(complex_amps.real, bins=30, density=True)
    plt.hist(complex_amps.imag, bins=30, density=True)
    t = np.linspace(-3/rootn, 3/rootn)
    plt.plot(t, stats.norm(scale=sigma/rootn).pdf(t))
    """

    field = _make_field(xx, yy, xp, yp, complex_amps, wavelength)

    return field, xp, yp


@functools.lru_cache(maxsize=100, typed=False)
def make_fields(m, n, wavelength):
    np.random.seed(123)
    fields = np.zeros((m, len(x), len(y)), np.complex_)
    for k in tqdm.trange(m):
        field, _, _ = make_field(n, wavelength)
        fields[k] = field
    return fields


np.random.seed(123)
field, xp, yp = make_field()
field_ = np.ravel(field)

#%% Plot scatterers
plt.figure()
plt.plot(xp, yp, ".")
plt.axis("square")
rect = mpl.patches.Rectangle(
    (0, 0), 1, 1, linewidth=1, edgecolor="C1", facecolor="none"
)
plt.gca().add_patch(rect)
plt.title("Scatterers")

#%% Plot field
plt.figure()
plt.imshow(np.abs(field), extent=(0, 1, 0, 1))
plt.axis("square")

#%% == PART A: estimation of sigma

all_reports_sigma = pd.DataFrame()


def report_sigma(method_name, estimates):
    out = {}
    out["mean"] = np.mean(estimates)
    out["trueness_err"] = np.abs(1 - np.mean(estimates))
    out["std"] = np.std(estimates)
    out["q5"] = np.quantile(estimates, 0.05)
    out["q95"] = np.quantile(estimates, 0.95)
    out["range90"] = out["q95"] - out["q5"]
    s = pd.Series(out, name=method_name)
    all_reports_sigma[method_name] = s
    return s


def make_subfield(field):
    """
    the one place to downsample, use a smaller field, etc
    """
    return field


#%% From real part of pixels
n = 500
wavelength = 0.1

m = 200
estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.std(field.real)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("std_real", estimates))

#%% From real part of pixels - bis
n = 500
wavelength = 0.1

m = 200
estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(field.real ** 2))

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("std_real2", estimates))

#%% From RMS (Rayleigh max lihelihood)
n = 500
wavelength = 0.1

m = 200
estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(np.abs(field) ** 2)) / np.sqrt(2)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_ml", estimates))

#%% From RMS (Rayleigh max lihelihood) + bootstrap
n = 500
wavelength = 0.1

m = 200
estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
np.random.seed(123)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    estimate_tmp = []
    _field = field.ravel()
    for _ in range(10):
        field_bootstraped = np.random.choice(_field, size=len(_field), replace=True)
        estimate_tmp.append(
            np.sqrt(np.mean(np.abs(field_bootstraped) ** 2)) / np.sqrt(2)
        )
    estimates[k] = np.mean(estimate_tmp)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_ml_boot", estimates))

#%% From mean of Rayleigh
# theoretically suboptimal but why not
n = 500
wavelength = 0.1

m = 200
estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.mean(np.abs(field)) * np.sqrt(2 / np.pi)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_mean", estimates))

#%% From Rayleigh order statistic
# See Siddiqui 1964

k_opt = round(0.79681 * (xx.size + 1) - 0.39841 + 1.16312 / (xx.size + 1))
n = 500
wavelength = 0.1

m = 200
estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    _field = np.sort(np.ravel(np.abs(field) ** 2))
    estimates[k] = np.sqrt((0.6275 * _field[k_opt - 1]) / 2)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_order", estimates))


#%% Rayleigh ML if we were independant
m = 200
estimates = np.zeros(m)
for k in range(m):
    field = stats.rayleigh(scale=1).rvs(xx.size)
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(np.abs(field) ** 2)) / np.sqrt(2)
print(report_sigma("rayleigh_ml_best_case", estimates))

#%% Conclusion part A
print(all_reports_sigma.T)

#%% Plot Normal dist
t = np.linspace(-3 * sigma, +3 * sigma, 100)
plt.figure()
plt.hist(field_.real, density=True, bins=30)
plt.plot(t, stats.norm(scale=sigma).pdf(t), label="Norm(0, Sigma^2)")
plt.legend()

#%% Plot Rayleigh dist
rms = np.sqrt(np.mean(np.abs(field_) ** 2))

rayleigh = stats.rayleigh(scale=sigma)
fitte_drayleigh = stats.rayleigh(scale=rms / np.sqrt(2))
t = np.linspace(0, rayleigh.ppf(0.9999), 100)
plt.figure()
plt.hist(np.abs(field_), bins=30, density=True)
plt.plot(t, rayleigh.pdf(t), label="Rayleigh(sigma)")
plt.plot(t, fitte_drayleigh.pdf(t), label="Fitted Rayleigh")
plt.legend()

#%% == PART B: estimation of effective sample size

all_reports_neff = pd.DataFrame()


def report_neff(method_name, estimates, save=True):
    out = {}
    out["mean"] = np.mean(estimates)
    out["std"] = np.std(estimates)
    out["q5"] = np.quantile(estimates, 0.05)
    out["q95"] = np.quantile(estimates, 0.95)
    out["range90"] = out["q95"] - out["q5"]
    out["aca"] = 1 / np.mean(estimates)
    out["acl"] = np.sqrt(out["aca"] / np.pi)
    s = pd.Series(out, name=method_name)
    if save:
        all_reports_neff[method_name] = s
    return s


#%% From fft
n = 500
wavelength = 0.1
m = 200


def aca_fft(field):
    Z = np.fft.fft2(field)
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    aca = np.sum(np.abs(autocorr_normed) > (1 / np.e)) * dx * dy
    return aca, autocorr_normed


estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed = aca_fft(field)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("fft", estimates))

#%% From fft with label
n = 500
wavelength = 0.1
m = 200


def aca_fft_label(field):
    Z = np.fft.fft2(field)
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    labels, _ = ndimage.measurements.label(np.abs(autocorr_normed) > 1 / np.e)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    aca = np.sum(labels == centre_label) * dx * dy
    return aca, autocorr_normed, labels, centre_label


estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = aca_fft_label(field)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("fft_label", estimates))


#%% From fft with label and padded
n = 500
wavelength = 0.1
m = 200

# padding factor
p = 2


def aca_fft_label_padded(field, p):
    Z = np.fft.fft2(field, (p * field.shape[0], p * field.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    labels, _ = ndimage.measurements.label(np.abs(autocorr_normed) > 1 / np.e)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    aca = np.sum(labels == centre_label) * dx * dy
    return aca, autocorr_normed, labels, centre_label


estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = aca_fft_label_padded(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("fft_label_padded", estimates))

#%% Plot fft autocorr
plt.figure()
plt.imshow(np.abs(autocorr_normed), extent=(0, 1, 0, 1))
plt.title("autocorrelation with fft")

plt.figure()
plt.imshow(np.abs(autocorr_normed) > 1 / np.e, extent=(0, 1, 0, 1))
plt.title("autocorrelation with fft")

plt.figure()
plt.imshow(labels, extent=(0, 1, 0, 1))
plt.title("autocorrelation with fft")

plt.figure()
plt.imshow(labels == centre_label, extent=(0, 1, 0, 1))
plt.title("autocorrelation with fft")


#%% From variance of Rayleigh sigma ML estimator
# Var(sigma ML estimator) = sigma^2 / (4 n)
# from http://ocw.mit.edu/ans7870/18/18.443/s15/projects/Rproject3_rmd_rayleigh_theory.html

# Numerical validation of Var(sigma ML estimator) = sigma^2 / (4 n)
m = 1000
neff = 50
sigma_estimates = np.zeros(m)
np.random.seed(123)
for k in range(m):
    field = stats.rayleigh(scale=sigma).rvs(size=neff)
    sigma_estimates[k] = np.sqrt(np.mean(np.abs(field) ** 2)) / np.sqrt(2)
print("---")
print(f"Experimental Var(sigma_ml) = {np.var(sigma_estimates)}")
print(f"Theoretical Var(sigma_ml) = {sigma ** 2 / (4 * neff)}")
print(f"Experimental neff = {sigma **2 / (4 * np.var(sigma_estimates))}")
print(f"Actual neff = {neff}")
print("---")

# Estimate neff from variance
neff = sigma ** 2 / (4 * all_reports_sigma.loc["std", "rayleigh_ml"] ** 2)
print(report_neff("var_sigma_ml", [neff]))

#%% From dist of maxima

# TODO

n = 500
wavelength = 0.2

m = 200
np.random.seed(123)
maximas = np.zeros(m)
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    maximas[k] = np.max(np.abs(field))

# around 400 for wavelength=0.2
# around 2000 for wavelength=0.1

t = np.linspace(0, 1.5 * maximas.max(), 200)


def maxrayleigh_pdf(t, n):
    dist = stats.rayleigh(scale=sigma)
    return n * dist.cdf(t) ** (n - 1) * dist.pdf(t)


plt.figure()
plt.hist(maximas, density=True, bins=20)
# plt.plot(t, maxrayleigh_pdf(t, 400))
# plt.plot(t, maxrayleigh_pdf(t, 2000))
plt.plot(t, maxrayleigh_pdf(t, 400))


#%% From Goodman // Wagner 1983 eq 31 (unpadded)
n = 500
wavelength = 0.1
m = 200


def aca_goodman(field, p=1):
    field = make_subfield(field)
    Z = np.fft.fft2(field, (p * field.shape[0], p * field.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    aca = np.trapz(np.trapz(np.abs(autocorr_normed), dx=dy), dx=dx)
    return aca


estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca = aca_goodman(field, p=1)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("goodman", estimates))

#%% From Goodman // Wagner 1983 eq 31 (padded)
n = 500
wavelength = 0.1
m = 200

p = 2


estimates = np.zeros(m)
fields = make_fields(m, n, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca = aca_goodman(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("goodman_padded", estimates))


#%% Conclusion part B
print(all_reports_neff.T)

#%% == PART C: pretty plots

#%% ACA estimates vs wavelength

n = 500
wavelengths = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
m = 200

# padding factor
p = 2

res = []
for wavelength in tqdm.tqdm(wavelengths):
    estimates_fft = np.zeros(m)
    estimates_fft_label = np.zeros(m)
    estimates_fft_label_padded = np.zeros(m)
    estimates_goodman = np.zeros(m)
    fields = make_fields(m, n, wavelength)
    for k, field in enumerate(fields):
        field = make_subfield(field)
        aca, _ = aca_fft(field)
        estimates_fft[k] = 1 / aca

        aca, _, _, _ = aca_fft_label(field)
        estimates_fft_label[k] = 1 / aca

        aca, _, _, _ = aca_fft_label_padded(field, p)
        estimates_fft_label_padded[k] = 1 / aca

        aca = aca_goodman(field, p)
        estimates_goodman[k] = 1 / aca

    report = report_neff("fft", estimates_fft, save=False)
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff("fft_label", estimates_fft_label, save=False)
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff("fft_label_padded", estimates_fft_label_padded, save=False)
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff("goodman_padded", estimates_goodman, save=False)
    report["wavelength"] = wavelength
    res.append(report)

#%% ACA FFT vs wavelength - Plot
df = pd.DataFrame(res)
df.index.name = "method"
df.reset_index().pivot(index="wavelength", columns="method", values="mean").plot(
    logy=True, logx=True
)
plt.plot(wavelengths, 4 / wavelengths ** 2, "o", label="theory")
