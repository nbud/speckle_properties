# -*- coding: utf-8 -*-
"""
Generate speckle by convolving the PSF to a random field

The convolution is done by a multiplication in the Fourier domain. Limits:
    - the random field coincides with the pixel grid
    - the pixel size must be larger than the wavelength

Results: Abs(Z) is Rayleigh(Sigma) X is normally distribution Norm(0, Sigma^2)
    (but the variance is wiggly)


Method 1: FFT and threshold at 1/e The correlation length is around half the
wavelength Fundamental flaw: catches not only the main cluster by further one


Padding the FFT reduces the variance of the ACA

Final method: FFT padded + label identification ACA ~ (wavelength / 2)^2
Number of unique points ~ 1/(wavelength / 2)^2

Sources of errors:
    - Wavelength too small: discretisation becomes too rough
    - Wavelength too large: measured area becomes too small to be
      statistically representative


"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
from scipy import ndimage, stats, optimize
import functools
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-s", "--save", action="store_true", default=False)
parser.add_argument("--paper", action="store_true", default=False)
args = parser.parse_args()
save = args.save
is_paper = args.paper

pd.set_option("display.max_columns", None)
plt.rcParams["image.origin"] = "lower"
if is_paper:
    plt.rcParams["savefig.format"] = "pdf"

if is_paper:
    FIGSIZE_HALF_SQUARE = (3.3, 3.3)
else:
    FIGSIZE_HALF_SQUARE = None
#%% Init
# Measurement area
x = np.linspace(0, 1, 256, endpoint=False)
y = np.linspace(0, 1, 256, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
xx, yy = np.meshgrid(x, y)

# Speckle generation area (larger to avoid border effect)
x_ext = np.linspace(-0.5, 1.5, 2 * len(x), endpoint=False)
y_ext = np.linspace(-0.5, 1.5, 2 * len(y), endpoint=False)
xx_ext, yy_ext = np.meshgrid(x_ext, y_ext)
r = np.sqrt((xx_ext - 0.5) ** 2 + (yy_ext - 0.5) ** 2)

extent = (0.0, 1.0, 0.0, 1.0)
extent_ext = (-0.5, 1.5, -0.5, 1.5)
extent_centred = (-0.5, 0.5, -0.5, 0.5)
extent_ext_centred = (-1.0, 1.0, -1.0, 1.0)

# Underlying and apparent sigma.
# The relationship betweent the two depends on PSF
sigma = 1.0
underlying_sigma = sigma / np.sqrt(xx_ext.size)


@functools.lru_cache(maxsize=100, typed=False)
def make_psf(wavelength):
    psf = np.exp(2j * np.pi / wavelength * r)
    # psf = np.exp(2j * np.pi / wavelength * r) * np.exp(-(r / (2 * wavelength))**2)
    psf_fft = np.fft.fft2(psf)
    return psf, psf_fft


def select_centre(arr):
    return arr[(len(x) // 2) : -(len(x) // 2), (len(y) // 2) : -(len(y) // 2)]


def make_field(wavelength=0.1):
    """
    Generate speckle in Fourier domain
    Valid as long as the pixel size is way smaller than the wavelength
    """
    psf, psf_fft = make_psf(wavelength)

    random_field = (
        stats.norm(scale=underlying_sigma)
        .rvs(2 * xx_ext.size)
        .view(np.complex_)
        .reshape(xx_ext.shape)
    )
    speckle_ext = np.fft.fftshift(
        np.fft.ifft2(psf_fft.conj() * np.fft.fft2(random_field))
    )
    speckle = select_centre(speckle_ext)
    return speckle, random_field


@functools.lru_cache(maxsize=100, typed=False)
def make_fields(m, wavelength, seed=123):
    np.random.seed(seed)
    fields = np.zeros((m, len(x), len(y)), np.complex_)
    for k in tqdm.trange(m, desc="Generate fields"):
        field, _ = make_field(wavelength)
        fields[k] = field
    return fields


#%% Plot random field
wavelength = 0.1
np.random.seed(123)
field, random_field = make_field(wavelength)

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.real(random_field), extent=extent_ext, cmap="Greys", interpolation="none")
rect = mpl.patches.Rectangle(
    (0, 0), 1, 1, linewidth=1, edgecolor="C1", facecolor="none"
)
plt.gca().add_patch(rect)
plt.xlabel("x")
plt.ylabel("y")
if save:
    plt.savefig("random_field")
#%% PSF
wavelength = 0.1
psf, psf_fft = make_psf(wavelength)

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.real(psf), extent=extent_ext_centred, interpolation="none")
plt.xlabel("x")
plt.ylabel("y")
if save:
    plt.savefig("psf")

psf_autocorr = np.fft.fftshift(np.fft.ifft2(psf_fft * psf_fft.conj()))
# psf_autocorr = psf_autocorr[(len(x) // 2): -(len(x) // 2), (len(y) // 2): -(len(y) // 2)]
psf_autocorr_normed = np.abs(psf_autocorr)
psf_autocorr_normed /= np.max(psf_autocorr_normed)
plt.figure()
plt.imshow(psf_autocorr_normed, extent=extent_ext_centred)
plt.clim(0, 1)
plt.xlabel("x")
plt.ylabel("y")
if save:
    plt.savefig("psf_autocorr")

#%% Plot field
for wavelength in [0.01, 0.05, 0.1, 0.2, 0.5]:
    np.random.seed(123)
    field, _ = make_field(wavelength)
    plt.figure(figsize=FIGSIZE_HALF_SQUARE)
    plt.imshow(np.abs(field), extent=extent)
    plt.axis("square")
    plt.xlabel("x")
    plt.ylabel("y")
    if not is_paper:
        plt.title(f"wavelength={wavelength}")
    wavelength_str = str(wavelength).replace(".", "_")
    if save:
        plt.savefig(f"field_wavelength_{wavelength_str}")

#%% == PART A: estimation of sigma

all_reports_sigma = pd.DataFrame()
wavelength = 0.1
m = 200


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
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.std(field.real)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("std_real", estimates))

#%% From real part of pixels - bis
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(field.real ** 2))

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("std_real2", estimates))

#%% From RMS (Rayleigh max lihelihood)
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(np.abs(field) ** 2)) / np.sqrt(2)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_ml", estimates))

#%% From RMS (Rayleigh max lihelihood) + bootstrap
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
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
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.mean(np.abs(field)) * np.sqrt(2 / np.pi)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_mean", estimates))

#%% From Rayleigh order statistic
# See Siddiqui 1964

k_opt = round(0.79681 * (xx.size + 1) - 0.39841 + 1.16312 / (xx.size + 1))
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    _field = np.sort(np.ravel(np.abs(field) ** 2))
    estimates[k] = np.sqrt((0.6275 * _field[k_opt - 1]) / 2)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_order", estimates))


#%% Rayleigh ML if we were independant
estimates = np.zeros(m)
for k in range(m):
    field = stats.rayleigh(scale=sigma).rvs(xx.size)
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(np.abs(field) ** 2)) / np.sqrt(2)
print(report_sigma("rayleigh_ml_if_iid", estimates))

#%% Conclusion part A
print(all_reports_sigma.T)
if save:
    all_reports_sigma.to_csv("sigma.csv")

all_reports_sigma_df = all_reports_sigma.T.copy()
all_reports_sigma_df.index.name = "method"
all_reports_sigma_df = all_reports_sigma_df.loc[
    ["std_real", "rayleigh_ml", "rayleigh_mean", "rayleigh_order", "rayleigh_ml_if_iid"]
]
vals = all_reports_sigma_df["mean"]
q5 = all_reports_sigma_df["q5"]
q95 = all_reports_sigma_df["q95"]
yerrs = np.stack((vals - q5, q95 - vals), axis=1).T
plt.figure(figsize=(6.4, 2.0))
vals.plot.bar(yerr=yerrs, capsize=5)
plt.gca().set_xticklabels(vals.index, rotation=45)
plt.ylabel("estimated sigma")
plt.ylim([0.8, 1.15])
if save:
    plt.savefig("sigma_estimate")

#%% Plot Normal dist
field, _ = make_field(wavelength)
t = np.linspace(-3 * sigma, +3 * sigma, 100)
plt.figure()
plt.hist(np.ravel(field).real, density=True, bins=30)
plt.plot(t, stats.norm(scale=sigma).pdf(t), label="Norm(0, Sigma^2)")
plt.legend()

#%% Plot Rayleigh dist
field, _ = make_field(wavelength)
rms = np.sqrt(np.mean(np.abs(field) ** 2))
rayleigh = stats.rayleigh(scale=sigma)
fitte_drayleigh = stats.rayleigh(scale=rms / np.sqrt(2))
t = np.linspace(0, rayleigh.ppf(0.9999), 100)
plt.figure()
plt.hist(np.abs(np.ravel(field)), bins=30, density=True)
plt.plot(t, rayleigh.pdf(t), label="Rayleigh(sigma)")
plt.plot(t, fitte_drayleigh.pdf(t), label="Fitted Rayleigh")
plt.legend()

#%% == PART B: estimation of effective sample size

all_reports_neff = pd.DataFrame()
wavelength = 0.1
m = 200


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


#%% From FWHM (cutoff 1/e) (all lobes)
def aca_fwhm(field):
    Z = np.fft.fft2(field)
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    aca = np.sum(np.abs(autocorr_normed) > (1 / np.e)) * dx * dy
    return aca, autocorr_normed


estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed = aca_fwhm(field)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("fwhm", estimates))

#%% From FWHM (cutoff 1/e) (main lobe)
def aca_fwhm_main_lobe(field, p=1):
    Z = np.fft.fft2(field, (p * field.shape[0], p * field.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    labels, _ = ndimage.measurements.label(np.abs(autocorr_normed) > 1 / np.e)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    aca = np.sum(labels == centre_label) * dx * dy
    return aca, autocorr_normed, labels, centre_label


estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = aca_fwhm_main_lobe(field)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("fwhm_main_lobe", estimates))


#%% From FWHM (cutoff 1/e) (main lobe and padded)
# padding factor
p = 2

estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = aca_fwhm_main_lobe(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("fwhm_main_lobe_padded", estimates))

#%% Plot FWHM
plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.abs(autocorr_normed), extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("autocorrelation")
if save:
    plt.savefig("fwhm_a")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.abs(autocorr_normed) > 1 / np.e, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("autocorrelation thresholded with 1/e")
if save:
    plt.savefig("fwhm_b")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(labels, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("clusters in thresholded autocorrelation")
if save:
    plt.savefig("fwhm_c")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(labels == centre_label, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("main lobe for FWHM")
if save:
    plt.savefig("fwhm_d")


#%% From variance of Rayleigh sigma ML estimator
# Var(sigma ML estimator) = sigma^2 / (4 n)
# from http://ocw.mit.edu/ans7870/18/18.443/s15/projects/Rproject3_rmd_rayleigh_theory.html

# Numerical validation of Var(sigma ML estimator) = sigma^2 / (4 n)
_m = 1000
neff = 50
sigma_estimates = np.zeros(_m)
np.random.seed(123)
for k in range(_m):
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

#%% From dist of maxima (preparation)
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

#%% From dist of maxima
def neff_maxima_method(fields):
    m = len(fields)
    maximas = np.zeros(m)
    for k, field in enumerate(fields):
        field = make_subfield(field)
        maximas[k] = np.max(np.abs(field))

    # Calculate max likelihood
    likelihood_func = lambda n: -maxrayleigh_logpdf(maximas, n, sigma).sum()
    res = optimize.minimize_scalar(
        likelihood_func, bounds=(1, xx.size), method="bounded"
    )
    assert res.success
    neff = res.x
    q5 = None  # TODO
    q95 = None  # TODO
    return neff, q5, q95, maximas


fields = make_fields(m, wavelength)
neff, q5, q95, maximas = neff_maxima_method(fields)
#%%
# Plot
nvect = np.arange(1, 10000, 10)
log_likelihood = maxrayleigh_logpdf(
    maximas[np.newaxis], nvect[..., np.newaxis], sigma
).sum(axis=-1)
plt.figure()
plt.plot(nvect, np.exp(log_likelihood - np.max(log_likelihood)))
plt.xlabel("effective sample size")
if not is_paper:
    plt.title(f"maximum likelihood={neff:.0f}")
if save:
    plt.savefig("neff_maximas")

# TODO: calculate HPD
print(report_neff("maxima", [neff]))


#%% From area under curve (Goodman // Wagner 1983 eq 31) (unpadded)
def aca_auc(field, p=1):
    """
    ACA based on total area under autocorrelation curve
    """
    field = make_subfield(field)
    Z = np.fft.fft2(field, (p * field.shape[0], p * field.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    aca = np.trapz(np.trapz(np.abs(autocorr_normed), dx=dy), dx=dx)
    return aca


p = 2
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca = aca_auc(field, p=1)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("auc", estimates))

#%% From area under main lobe
def aca_area_main_lobe(field, p=1):
    """
    ACA based on area under main lobe
    
    """
    field = make_subfield(field)
    Z = np.fft.fft2(field, (p * field.shape[0], p * field.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    grady, gradx = np.gradient(np.abs(autocorr_normed))
    fx = np.fft.fftshift(np.fft.fftfreq(Z.shape[0], dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Z.shape[1], dy))
    fxx, fyy = np.meshgrid(fx, fy)
    phi = np.arctan2(fyy, fxx)
    gradr = gradx * np.cos(phi) + grady * np.sin(phi)
    # the centre is sometime shaky: fix it
    gradr[gradr.shape[0] // 2, gradr.shape[1] // 2] = 0.0
    labels, _ = ndimage.measurements.label(gradr <= 0)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    main_lobe = labels == centre_label
    aca = np.sum(main_lobe) * dx * dy
    return aca, autocorr_normed, gradr, main_lobe


p = 2
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, gradr, main_lobe = aca_area_main_lobe(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("area_main_lobe", estimates))

#%% Plot area under main lobe
plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.abs(autocorr_normed), extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("autocorr")
if save:
    plt.savefig("auc_a")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(gradr, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("grad_r")
if save:
    plt.savefig("auc_b")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(gradr <= 0, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("grad_r <= 0")
if save:
    plt.savefig("auc_c")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(main_lobe, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("y")
if is_paper:
    plt.xlim([-0.25, 0.25])
    plt.ylim([-0.25, 0.25])
if not is_paper:
    plt.title("Main lobe")
if save:
    plt.savefig("auc_d")

#%% Conclusion part B
print(all_reports_neff.T)
report_neff("from_wavelength", [1 / wavelength ** 2])
if save:
    all_reports_neff.to_csv("neff.csv")

all_reports_neff_df = all_reports_neff.T.copy()
all_reports_neff_df.index.name = "method"
all_reports_neff_df = all_reports_neff_df.drop("auc")
vals = all_reports_neff_df["mean"]
q5 = all_reports_neff_df["q5"]
q95 = all_reports_neff_df["q95"]
yerrs = np.stack((vals - q5, q95 - vals), axis=1).T
yerrs[yerrs == 0] = np.nan
plt.figure()
vals.plot.bar(yerr=yerrs, capsize=5)
plt.gca().set_xticklabels(vals.index, rotation=80)
plt.ylabel("estimated effective sample size")
if save:
    plt.savefig("neff_estimate")


#%% == PART C: pretty plots

#%% Effective sample size vs wavelength

wavelengths = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
m = 200

# padding factor
p = 2

res = []
for wavelength in tqdm.tqdm(wavelengths):
    estimates_fwhm = np.zeros(m)
    estimates_fwhm_main_lobe = np.zeros(m)
    estimates_fwhm_main_lobe_padded = np.zeros(m)
    estimates_auc = np.zeros(m)
    estimates_area_main_lobe = np.zeros(m)
    fields = make_fields(m, wavelength)
    for k, field in enumerate(fields):
        field = make_subfield(field)
        aca, _ = aca_fwhm(field)
        estimates_fwhm[k] = 1 / aca

        aca, _, _, _ = aca_fwhm_main_lobe(field)
        estimates_fwhm_main_lobe[k] = 1 / aca

        aca, _, _, _ = aca_fwhm_main_lobe(field, p)
        estimates_fwhm_main_lobe_padded[k] = 1 / aca

        aca = aca_auc(field, p)
        estimates_auc[k] = 1 / aca

        aca, _, _, _ = aca_area_main_lobe(field, p)
        estimates_area_main_lobe[k] = 1 / aca

    report = report_neff("fwhm", estimates_fwhm, save=False)
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff("fwhm_main_lobe", estimates_fwhm_main_lobe, save=False)
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff(
        "fwhm_main_lobe_padded", estimates_fwhm_main_lobe_padded, save=False
    )
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff("auc", estimates_auc, save=False)
    report["wavelength"] = wavelength
    res.append(report)

    report = report_neff("area_main_lobe", estimates_area_main_lobe, save=False)
    report["wavelength"] = wavelength
    res.append(report)

#%% Effective sample size estimates which based on a series of fields
neff_maximas = np.zeros(len(wavelengths))
for k, wavelength in enumerate(wavelengths):
    fields = make_fields(m, wavelength)
    neff, _, _, _ = neff_maxima_method(fields)
    neff_maximas[k] = neff

#%% Plot effective sample size vs wavelength

df = pd.DataFrame(res)
df.index.name = "method"
df = df[df.index.isin(["area_main_lobe", "fwhm_main_lobe_padded"])]
df = df.rename(
    index={"area_main_lobe": "area of main lobe", "fwhm_main_lobe_padded": "FWHM"}
)
vals = df.reset_index().pivot(index="wavelength", columns="method", values="mean")
q5 = df.reset_index().pivot(index="wavelength", columns="method", values="q5")
q95 = df.reset_index().pivot(index="wavelength", columns="method", values="q95")
yerrs = np.stack((vals - q5, q95 - vals), axis=1).T
vals.plot(label=".-", logy=True, logx=True, yerr=yerrs, capsize=5)
plt.plot(wavelengths, 1 / wavelengths ** 2, "k--", label=r"$n=\lambda^{-2}$")
plt.plot(wavelengths, neff_maximas, "-o", label="maxima method")
plt.axis("auto")
plt.ylabel("effective sample size")
plt.legend()
if save:
    plt.savefig("wavelength_vs_neff")


#%% == PART D: determine extremal index

#%% For one (plot and debug)
m = 200
wavelength = 0.05
fields = make_fields(m, wavelength)
maximas = np.zeros(m)
for k, field in enumerate(fields):
    field = make_subfield(field)
    maximas[k] = np.max(np.abs(field))
    # maximas[k] = np.max(stats.rayleigh(scale=sigma).rvs(xx.size))  # debug

a = sigma / np.sqrt(2 * np.log(field.size))
b = sigma * np.sqrt(2 * np.log(field.size))

# debug
# maximas = stats.gumbel_r(loc=b, scale=a).rvs(m)

# maximum likelihood of a gumbel distribution
log_theta = -np.log(np.mean(np.exp(-(maximas - b) / a)))
theta = np.exp(log_theta)
print(f"theta={theta}")

## Equivalent. mu = a * log_theta + b
# mu = -a * np.log(np.mean(np.exp(-maximas / a)))
# log_theta_ = (mu - b) / a
# theta_ = np.exp(log_theta_)
# print(f"theta_={theta_}")

print(f"Experimental mean: {np.mean(maximas)}")
print(f"Th. mean inc. theta: {a * (log_theta + np.euler_gamma) + b}")
print(f"Th. mean if iid: {np.euler_gamma * a + b}")

#%%
t = np.linspace(3.4, 6, 200)
plt.figure(figsize=(6.4, 2))
plt.hist(maximas, bins=20, density=True, label="Experimental maxima")
plt.plot(
    t, stats.gumbel_r.pdf(t, loc=b + a * log_theta, scale=a), label="Gumbel (fitted)"
)
plt.plot(t, stats.gumbel_r.pdf(t, loc=b, scale=a), label="Gumbel (iid)")
if not is_paper:
    plt.title(f"wavelength={wavelength}")
plt.legend()
if save:
    plt.savefig("fitted_gumbel")

#%%
wavelengths = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
log_thetas = []
for wavelength in tqdm.tqdm(wavelengths):
    fields = make_fields(m, wavelength)
    maximas = np.zeros(m)
    for k, field in enumerate(fields):
        field = make_subfield(field)
        maximas[k] = np.max(np.abs(field))
    a = sigma / np.sqrt(2 * np.log(field.size))
    b = sigma * np.sqrt(2 * np.log(field.size))
    log_theta = -np.log(np.mean(np.exp(-(maximas - b) / a)))
    log_thetas.append(log_theta)
log_thetas = np.array(log_thetas)
thetas = np.exp(log_thetas)

#%%
plt.figure(figsize=(6.4, 2))
plt.loglog(wavelengths, thetas, ".-")
plt.ylabel("extremal index")
plt.xlabel("wavelength")
if save:
    plt.savefig("extremal_index")
