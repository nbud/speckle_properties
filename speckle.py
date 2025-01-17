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
      
If changing the PSF: adjust underlying_sigma, true_sigma, psf

Remark: "autocorrelation" is used for "autocovariance" by abuse of language.

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
from scipy import ndimage, stats
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

underlying_sigma = 1.0 / np.sqrt(xx_ext.size)


@functools.lru_cache(maxsize=100, typed=False)
def make_psf(wavelength):
    # PSF which does not vanish
    # psf = np.exp(2j * np.pi / wavelength * r)

    # PSF which vanish in lambda
    psf = np.exp(2j * np.pi / wavelength * r) * np.exp(-((r / wavelength) ** 2))

    # PSF which vanish in 1
    # a = 1/np.sqrt(-np.log(0.01))
    # psf = np.exp(2j * np.pi / wavelength * r) * np.exp(-(r / a) ** 2)

    # rescale so that sigma=1
    psf /= np.sqrt(np.mean(np.abs(psf) ** 2))
    psf_fft = np.fft.fft2(psf)
    return psf, psf_fft


def select_centre(arr):
    return arr[(len(x) // 2) : -(len(x) // 2), (len(y) // 2) : -(len(y) // 2)]


@functools.lru_cache(maxsize=100, typed=False)
def true_sigma(wavelength):
    """
    True sigma. Must be consistent with PSF!
    """
    return np.sqrt(np.mean(np.abs(make_psf(wavelength)[0]) ** 2))


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


def make_ecdf(samples):
    """ Empiral cdf"""
    x = np.sort(samples)
    n = x.size
    cdf = np.arange(1, n + 1) / n
    return x, cdf


def pp_plot(samples, cdf_func, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_HALF_SQUARE)
    ecdf_x, ecdf = make_ecdf(samples)
    ax.plot(cdf_func(ecdf_x), ecdf, ".", label=label)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("theoretical cumulative distribution")
    ax.set_ylabel("empirical cumulative distribution")
    if not is_paper:
        ax.set_title("P-P plot")
    return ax


def make_subfield(field):
    """
    the one place to downsample, use a smaller field, etc
    """
    return field


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
plt.ylabel("z")
if save:
    plt.savefig("random_field")
#%% PSF
wavelength = 0.1
psf, psf_fft = make_psf(wavelength)

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.real(psf), extent=extent_ext_centred, interpolation="none")
plt.xlabel("x")
plt.ylabel("z")
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
plt.ylabel("z")
if save:
    plt.savefig("psf_autocorr")

#%% Plot field
for wavelength in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5]:
    np.random.seed(123)
    field, _ = make_field(wavelength)
    plt.figure(figsize=FIGSIZE_HALF_SQUARE)
    plt.imshow(np.abs(field), extent=extent)
    plt.axis("square")
    plt.xlabel("x")
    plt.ylabel("z")
    if not is_paper:
        plt.title(f"wavelength={wavelength}")
    wavelength_str = str(wavelength).replace(".", "_")
    if save:
        plt.savefig(f"field_wavelength_{wavelength_str}")

#%% == PART A: estimation of sigma

all_reports_sigma = pd.DataFrame()
wavelength = 0.1
m = 2000


def report_sigma(method_name, estimates):
    out = {}
    out["mean"] = np.mean(estimates)
    out["trueness_err"] = np.abs(true_sigma(wavelength) - np.mean(estimates))
    out["std"] = np.std(estimates)
    out["q5"] = np.quantile(estimates, 0.05)
    out["q95"] = np.quantile(estimates, 0.95)
    out["range90"] = out["q95"] - out["q5"]
    s = pd.Series(out, name=method_name)
    all_reports_sigma[method_name] = s
    return s


#%% From real part of pixels - MLE of variance of univariate Gaussian for known mean
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(field.real ** 2))

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("std_real", estimates))

#%% From RMS (Rayleigh max lihelihood)
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(fields):
    field = make_subfield(field)
    estimates[k] = np.sqrt(np.mean(np.abs(field) ** 2)) / np.sqrt(2)

plt.figure()
plt.hist(estimates, density=True, bins=20)
print(report_sigma("rayleigh_ml", estimates))

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
np.random.seed(123)
estimates = np.zeros(m)
for k in range(m):
    field = stats.rayleigh(scale=true_sigma(wavelength)).rvs(xx.size)
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
plt.ylim([0.9, 1.1])
if save:
    plt.savefig("sigma_estimate")

#%% Plot Normal dist
sigma = true_sigma(wavelength)
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

#%% Close figures
plt.close("all")

#%% == PART B: estimation of effective sample size

all_reports_neff = pd.DataFrame()
wavelength = 0.1
m = 200
sigma = true_sigma(wavelength)


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


#%% From width (cutoff 1/e) (main lobe) [COMPLEX]
def width_main_lobe_complex(field, p=1):
    Z = np.fft.fft2(field, (p * field.shape[0], p * field.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    labels, _ = ndimage.measurements.label(np.abs(autocorr_normed) > 1 / np.e)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    aca = np.sum(labels == centre_label) * dx * dy
    return aca, autocorr_normed, labels, centre_label


# padding factor
p = 1

estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = width_main_lobe_complex(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("width_main_lobe_complex", estimates))

#%% Plot width
plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.abs(autocorr_normed), extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("autocorrelation")
if save:
    plt.savefig("width_a")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.abs(autocorr_normed) > 1 / np.e, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("autocorrelation thresholded with 1/e")
if save:
    plt.savefig("width_b")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(labels, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("clusters in thresholded autocorrelation")
if save:
    plt.savefig("width_c")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(labels == centre_label, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("main lobe for width")
if save:
    plt.savefig("width_d")


#%% From width (cutoff 1/e) (main lobe) [AMP]
def width_main_lobe_amp(field, p=1):
    field = np.abs(field)
    field -= np.mean(field)
    return width_main_lobe_complex(field, p)


# padding factor
p = 1

estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = width_main_lobe_amp(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("width_main_lobe_amp", estimates))

#%% From width (cutoff 1/e) (main lobe) [INT]
def width_main_lobe_int(field, p=1):
    field = np.abs(field ** 2)
    field -= np.mean(field)
    return width_main_lobe_complex(field, p)


# padding factor
p = 1

estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, labels, centre_label = width_main_lobe_int(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("width_main_lobe_int", estimates))

#%% From dist of maxima (preparation)
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
    t = np.linspace(0.01, 10, 100)
    plt.figure()
    plt.hist(samples, bins=30, density=True)
    plt.plot(t, np.exp(maxrayleigh_logpdf(t, neff, sigma)))
    plt.title("test_maxraleigh_logpdf")


test_maxrayleigh_logpdf()

#%% From MLE of Max Rayleigh
def neff_maxrayleigh_method(fields, sigma):
    m = len(fields)
    maxima = np.zeros(m)
    for k, field in enumerate(fields):
        field = make_subfield(field)
        maxima[k] = np.max(np.abs(field))
    neff = -m / np.sum(rayleigh_logcdf(maxima / sigma))
    q5 = None  # TODO
    q95 = None  # TODO
    return neff, q5, q95, maxima


def test_neff_maxrayleigh_method():
    # check max likelihood estimation with mock data
    m = 200
    true_neff = 50
    sigma = 2.0
    fields = stats.rayleigh.rvs(size=(m, true_neff), scale=sigma)
    neff, q5, q95, maxima = neff_maxrayleigh_method(fields, sigma)

    print(f"[TEST] true neff: {true_neff}")
    print(f"[TEST] estimated neff: {neff}")


test_neff_maxrayleigh_method()

fields = make_fields(m, wavelength)
neff, q5, q95, maxima = neff_maxrayleigh_method(fields, true_sigma(wavelength))

# Plot
nvect = np.arange(1, 10000, 10)
log_likelihood = maxrayleigh_logpdf(
    maxima[np.newaxis], nvect[..., np.newaxis], true_sigma(wavelength)
).sum(axis=-1)
plt.figure()
plt.plot(nvect, np.exp(log_likelihood - np.max(log_likelihood)))
plt.xlabel("effective sample size")
if not is_paper:
    plt.title(f"maximum likelihood={neff:.0f}")
if save:
    plt.savefig("neff_maxrayleigh")

# TODO: calculate HPD
print(report_neff("maxrayleigh", [neff]))


fitted_rayleigh_cdf = lambda t: np.exp(
    maxrayleigh_logcdf(t, neff, true_sigma(wavelength))
)
ecdf_x, ecdf = make_ecdf(maxima)
plt.figure()
plt.step(ecdf_x, ecdf)
t = np.linspace(3, 6, 200)
plt.plot(t, fitted_rayleigh_cdf(t))
plt.title("Fit results for MaxRayleigh - cdf")

ax = pp_plot(maxima, fitted_rayleigh_cdf)
ax.set_title("Fit results for MaxRayleigh - PP-plot")

fitted_rayleigh_pdf = lambda t: np.exp(
    maxrayleigh_logpdf(t, neff, true_sigma(wavelength))
)
plt.figure()
plt.hist(maxima, density=True)
plt.plot(t, fitted_rayleigh_pdf(t))


#%% From area under main lobe [COMPLEX]
# (Goodman // Wagner 1983 eq 31) (unpadded)
def area_main_lobe_complex(field, p=1):
    """
    ACA based on area under main lobe
    """
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


p = 1
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, gradr, main_lobe = area_main_lobe_complex(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("area_main_lobe_complex", estimates))

#%% Plot area under main lobe
plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(np.abs(autocorr_normed), extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("autocorr")
if save:
    plt.savefig("auc_a")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(gradr, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("grad_r")
if save:
    plt.savefig("auc_b")

plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(gradr <= 0, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("grad_r <= 0")
if save:
    plt.savefig("auc_c")

autocorr_normed_censored = np.abs(autocorr_normed).copy()
autocorr_normed_censored[~main_lobe] = 0.0
plt.figure(figsize=FIGSIZE_HALF_SQUARE)
plt.imshow(autocorr_normed_censored, extent=extent_centred)
plt.xlabel("x")
plt.ylabel("z")
if not is_paper:
    plt.title("Main lobe")
if save:
    plt.savefig("auc_d")

#%% From area under main lobe [AMP]
def area_main_lobe_amp(field, p=1):
    field = np.abs(field)
    field -= np.mean(field)
    return area_main_lobe_complex(field, p)


p = 1
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, gradr, main_lobe = area_main_lobe_amp(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("area_main_lobe_amp", estimates))

#%% From area under main lobe [INT]
def area_main_lobe_int(field, p=1):
    field = np.abs(field ** 2)
    field -= np.mean(field)
    return area_main_lobe_complex(field, p)


p = 1
estimates = np.zeros(m)
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    aca, autocorr_normed, gradr, main_lobe = area_main_lobe_int(field, p)
    estimates[k] = 1 / aca

plt.figure()
plt.hist(estimates, density=True, bins=30)

print(report_neff("area_main_lobe_int", estimates))

#%% Conclusion part B
print(all_reports_neff.T)
report_neff("from_wavelength", [1 / wavelength ** 2])
if save:
    all_reports_neff.to_csv("neff.csv")

all_reports_neff_df = all_reports_neff.T.copy()
all_reports_neff_df.index.name = "method"
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

speckle_cell_lengths = pd.Series(
    np.sqrt(1 / all_reports_neff_df["mean"]) / wavelength, name="speckle_cell_lenght"
)
print(speckle_cell_lengths)
#%% Close figures
plt.close("all")

#%% == PART C: determine extremal index

#%% For one (plot and debug)
m = 200
wavelength = 0.05
fields = make_fields(m, wavelength)
maxima = np.zeros(m)
for k, field in enumerate(fields):
    field = make_subfield(field)
    maxima[k] = np.max(np.abs(field))
    # maxima[k] = np.max(stats.rayleigh(scale=sigma).rvs(xx.size))  # debug

a = true_sigma(wavelength) / np.sqrt(2 * np.log(field.size))
b = true_sigma(wavelength) * np.sqrt(2 * np.log(field.size))

# debug
# maxima = stats.gumbel_r(loc=b, scale=a).rvs(m)

# maximum likelihood of a gumbel distribution
theta = 1 / (field.size ** 2 * np.mean(np.exp(-(maxima) / a)))
log_theta = np.log(theta)
print(f"theta={theta}")

print(f"Experimental mean: {np.mean(maxima)}")
print(f"Th. mean inc. theta: {a * (log_theta + np.euler_gamma) + b}")
print(f"Th. mean if iid: {np.euler_gamma * a + b}")

fitted_gumbel = stats.gumbel_r(loc=b + a * log_theta, scale=a)
t = np.linspace(3.7, 6, 200)
# t = np.linspace(maxima.min(), maxima.max(), 200)
plt.figure(figsize=(6.4, 2))
plt.hist(maxima, bins=20, density=True, label="Experimental maxima")
plt.plot(t, fitted_gumbel.pdf(t), label="Gumbel (fitted)")
plt.plot(t, stats.gumbel_r.pdf(t, loc=b, scale=a), label="Gumbel (iid)")
if not is_paper:
    plt.title(f"wavelength={wavelength}")
plt.legend()
if save:
    plt.savefig("fitted_gumbel")

# pp_plot(maxima, fitted_gumbel.cdf)

#%% Close figures
plt.close("all")

#%% == PART D plot neff vs wavelength
wavelengths = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
m = 200

#%% MaxRayleigh neff vs wavelength
neff_maxrayleigh = np.zeros(len(wavelengths))
maxrayleigh_cdfs = []
maxima_per_wavelength = []
for k, wavelength in enumerate(wavelengths):
    fields = make_fields(m, wavelength)
    neff, _, _, maxima = neff_maxrayleigh_method(fields, true_sigma(wavelength))
    neff_maxrayleigh[k] = neff

    fitted_rayleigh_cdf = lambda t: np.exp(
        maxrayleigh_logcdf(t, neff_maxrayleigh[k], true_sigma(wavelength))
    )
    maxrayleigh_cdfs.append(fitted_rayleigh_cdf)
    maxima_per_wavelength.append(maxima)
    # ax = pp_plot(maxima, fitted_rayleigh_cdf, label="Rayleigh")
    # ax.set_title(f"P-P plot Max Rayleigh wavelength={wavelength}")

#%% Gumbel neff vs wavelength
thetas = []
gumbels = []
for k, wavelength in enumerate(wavelengths):
    maxima = maxima_per_wavelength[k]
    a = true_sigma(wavelength) / np.sqrt(2 * np.log(field.size))
    b = true_sigma(wavelength) * np.sqrt(2 * np.log(field.size))
    theta = 1 / (field.size ** 2 * np.mean(np.exp(-(maxima) / a)))
    thetas.append(theta)

    fitted_gumbel = stats.gumbel_r(loc=b + a * np.log(theta), scale=a)
    gumbels.append(fitted_gumbel)
    # ax = pp_plot(maxima, fitted_gumbel.cdf)
    # ax.set_title(f"P-P plot Gumbel wavelength={wavelength}")
thetas = np.array(thetas)
neff_extremal_index = thetas * xx.size

# Plot extremal index vs wavelength
plt.figure(figsize=(6.4, 2))
plt.loglog(wavelengths, thetas, ".-")
plt.ylabel("extremal index")
plt.xlabel("wavelength")
if save:
    plt.savefig("extremal_index")

#%% Calculate neff for various techniques
# padding factor
p = 1

res = []
estimator_funcs = [
    area_main_lobe_complex,
    area_main_lobe_amp,
    area_main_lobe_int,
    width_main_lobe_complex,
    width_main_lobe_amp,
    width_main_lobe_int,
]
for wavelength in tqdm.tqdm(wavelengths):
    fields = make_fields(m, wavelength)
    for estimator in estimator_funcs:
        estimates = np.zeros(m)
        for k, field in enumerate(fields):
            field = make_subfield(field)
            aca, _, _, _ = estimator(field, p)
            estimates[k] = 1 / aca

        report = report_neff(estimator.__name__, estimates, save=False)
        report["wavelength"] = wavelength
        res.append(report)

#%% Some stats
df = pd.DataFrame(res)
df.index.name = "method"
print("Mean spread (range90/mean)")
print((df["range90"] / df["mean"]).groupby(df.index).mean())

#%% P-P plots
df = pd.DataFrame(res)
df.index.name = "method"
df = df.set_index("wavelength", append=True)

for k, wavelength in enumerate(wavelengths):
    maxima = maxima_per_wavelength[k]
    ax = pp_plot(maxima, maxrayleigh_cdfs[k], label="MaxRayleigh")
    pp_plot(maxima, gumbels[k].cdf, label="Gumbel", ax=ax)

    neff = df.loc[("width_main_lobe_amp", wavelength), "mean"]
    cdf = lambda t: np.exp(maxrayleigh_logcdf(t, neff, true_sigma(wavelength)))
    pp_plot(maxima, cdf, label="Width", ax=ax)

    neff = df.loc[("area_main_lobe_amp", wavelength), "mean"]
    cdf = lambda t: np.exp(maxrayleigh_logcdf(t, neff, true_sigma(wavelength)))
    pp_plot(maxima, cdf, label="Area", ax=ax)
    ax.legend()

    if not is_paper:
        plt.title(f"P-P plot, fitted MaxRayleigh and Gumbel, wavelength={wavelength}")
    if save:
        wavelength_str = str(wavelength).replace(".", "_")
        plt.savefig(f"pp_plot_fitted_maxima_{wavelength_str}")

#%% Histograms MaxRayleigh and Gumbel
df = pd.DataFrame(res)
df.index.name = "method"
df = df.set_index("wavelength", append=True)

_tmin = np.inf
_tmax = 0
for k, wavelength in enumerate(wavelengths):
    maxima = maxima_per_wavelength[k]
    _tmin = min(_tmin, np.min(maxima))
    _tmax = max(_tmax, np.max(maxima))


for k, wavelength in enumerate(wavelengths):
    maxima = maxima_per_wavelength[k]
    t = np.linspace(_tmin, _tmax, 500)

    fitted_rayleigh_pdf = lambda t: np.exp(
        maxrayleigh_logpdf(t, neff_maxrayleigh[k], true_sigma(wavelength))
    )
    plt.figure(figsize=FIGSIZE_HALF_SQUARE)
    plt.hist(maxima, density=True, bins=20, color="grey")
    plt.plot(t, fitted_rayleigh_pdf(t), label="MaxRayl.")
    plt.plot(t, gumbels[k].pdf(t), label="Gumbel")

    neff = df.loc[("width_main_lobe_amp", wavelength), "mean"]
    plt.plot(
        t, np.exp(maxrayleigh_logpdf(t, neff, true_sigma(wavelength))), label="Width"
    )
    neff = df.loc[("area_main_lobe_amp", wavelength), "mean"]
    plt.plot(
        t, np.exp(maxrayleigh_logpdf(t, neff, true_sigma(wavelength))), label="Area"
    )
    plt.legend()
    plt.xlabel("peak amplitudes")
    plt.ylabel("density")
    if not is_paper:
        plt.title(f"wavelength={wavelength}")
    if save:
        wavelength_str = str(wavelength).replace(".", "_")
        plt.savefig(f"hist_fitted_maxima_{wavelength_str}")


#%%
def rename_methods_short(df):
    return df.rename(
        index={
            "area_main_lobe_amp": "Area method from amplitudes",
            "area_main_lobe_complex": "Area method from complex amp.",
            "area_main_lobe_int": "Area method from intensities",
            "width_main_lobe_amp": "Width method from amplitudes",
            "width_main_lobe_complex": "Width method from complex amp.",
            "width_main_lobe_int": "Width method from intensities",
        }
    )


#%% Plot effective sample size vs wavelength [AREA]
df = pd.DataFrame(res)
df.index.name = "method"
df = df[
    df.index.isin(
        ["area_main_lobe_complex", "area_main_lobe_amp", "area_main_lobe_int"]
    )
]
df = rename_methods_short(df)
vals = df.reset_index().pivot(index="wavelength", columns="method", values="mean")
q5 = df.reset_index().pivot(index="wavelength", columns="method", values="q5")
q95 = df.reset_index().pivot(index="wavelength", columns="method", values="q95")
yerrs = np.stack((vals - q5, q95 - vals), axis=1).T
vals.plot(label=".-", logy=True, logx=True, yerr=yerrs, capsize=5)
plt.plot(wavelengths, 1 / wavelengths ** 2, "k--", label=r"$n=\lambda^{-2}$")
plt.axis("auto")
plt.ylabel("effective sample size")
plt.legend()
if save:
    plt.savefig("wavelength_vs_neff_area_main_lobe")

#%% Plot effective sample size vs wavelength [WIDTH]
df = pd.DataFrame(res)
df.index.name = "method"
df = df[
    df.index.isin(
        ["width_main_lobe_complex", "width_main_lobe_amp", "width_main_lobe_int"]
    )
]
df = rename_methods_short(df)
vals = df.reset_index().pivot(index="wavelength", columns="method", values="mean")
q5 = df.reset_index().pivot(index="wavelength", columns="method", values="q5")
q95 = df.reset_index().pivot(index="wavelength", columns="method", values="q95")
yerrs = np.stack((vals - q5, q95 - vals), axis=1).T
vals.plot(label=".-", logy=True, logx=True, yerr=yerrs, capsize=5)
plt.plot(wavelengths, 1 / wavelengths ** 2, "k--", label=r"$n=\lambda^{-2}$")
plt.axis("auto")
plt.ylabel("effective sample size")
plt.legend()
if save:
    plt.savefig("wavelength_vs_neff_width_main_lobe")

#%% Plot effective sample size vs wavelength [ALL]
df = pd.DataFrame(res)
df.index.name = "method"
df = df[df.index.isin(["area_main_lobe_amp", "width_main_lobe_amp"])]
df = rename_methods_short(df)
vals = df.reset_index().pivot(index="wavelength", columns="method", values="mean")
q5 = df.reset_index().pivot(index="wavelength", columns="method", values="q5")
q95 = df.reset_index().pivot(index="wavelength", columns="method", values="q95")
yerrs = np.stack((vals - q5, q95 - vals), axis=1).T
vals.plot(label=".-", logy=True, logx=True, yerr=yerrs, capsize=5)
plt.plot(wavelengths, 1 / wavelengths ** 2, "k--", label=r"$n=\lambda^{-2}$")
plt.plot(wavelengths, neff_maxrayleigh, "-o", label="MaxRayleigh method")
plt.plot(wavelengths, neff_extremal_index, "-o", label="Extremal index method")
plt.axis("auto")
plt.ylabel("effective sample size")
plt.legend()
if save:
    plt.savefig("wavelength_vs_neff")

#%% Close figures
plt.close("all")

#%% == PART E: further tests

#%% Check "RMS SNR ratio"
# Goodman 1975 eq 1.113
# Smith 1983 eq 18
# Works better with the "narrow" PSF
wavelengths = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5])
m = 200

snr_rmses = np.zeros(len(wavelengths))
for i, wavelength in enumerate(wavelengths):
    estimates = np.zeros(m)
    fields = make_fields(m, wavelength)
    for k, field in enumerate(fields):
        estimates[k] = np.mean(np.abs(field))
    snr_rmses[i] = np.mean(estimates) / np.std(estimates)

plt.figure()
plt.loglog(wavelengths, snr_rmses, label="exp")
plt.loglog(wavelengths, 1 / wavelengths, label="1/wavelength")
plt.xlabel("wavelength")
plt.title("RMS signal-to-noise ratio")
plt.legend()

#%% Block estimate of extremal index
# Smith and Weissman 1994
# Problem: depends a LOT on the threshold and the block size
wavelength = 0.02
estimates = np.zeros(m)
block_size = 32
threshold = stats.rayleigh(scale=true_sigma(wavelength)).ppf(0.95)
numblock_x = len(x) // block_size
numblock_y = len(y) // block_size
print(f"numblocks {numblock_x*numblock_y}")
fields = make_fields(m, wavelength)
for k, field in enumerate(tqdm.tqdm(fields)):
    field = make_subfield(field)
    num_pixels_above_threshold = np.sum(field >= threshold)
    num_blocks_above_threshold = 0
    for i in range(numblock_x):
        for j in range(numblock_y):
            block = field[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            num_blocks_above_threshold += bool(np.any(block >= threshold))
    estimates[k] = num_blocks_above_threshold / num_pixels_above_threshold

plt.figure()
plt.hist(estimates, density=True, bins=30)
plt.axvline(
    thetas[np.nonzero(wavelengths == wavelength)[0][0]], color="C1", label="MLE"
)
plt.xlim([0, 1])
plt.xlabel("extremal index")
plt.title("block estimate of extremal index")
