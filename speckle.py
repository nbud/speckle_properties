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

Goodman does not work

"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numba
import math
import tqdm
from scipy import ndimage

#%%
x = np.linspace(0, 1, 200, endpoint=False)
y = np.linspace(0, 1, 200, endpoint=False)
dx = x[1] - x[0]
dy = y[1] - y[0]
xx, yy = np.meshgrid(x, y)
sigma=1.

@numba.njit(parallel=True)
def _make_field(xx, yy, xp, yp, complex_amps, wavelength):
    res = np.zeros(xx.shape, np.complex_)
    for i in numba.prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            for k in range(xp.shape[0]):
                r = math.sqrt((xx[i, j]-xp[k])**2 + (yy[i,j]-yp[k])**2)
                res[i, j] += complex_amps[k] * np.exp(2j * np.pi / wavelength * r)
    return res


def make_field(n = 500, wavelength=0.1):    
    rootn = np.sqrt(n)

    xp = stats.uniform(-1, 3.).rvs(size=n)
    yp = stats.uniform(-1, 3.).rvs(size=n)
    
    complex_amps = stats.norm(scale=sigma / rootn).rvs(2 * n).view(np.complex_)
    '''
    plt.figure()
    plt.hist(complex_amps.real, bins=30, density=True)
    plt.hist(complex_amps.imag, bins=30, density=True)
    t = np.linspace(-3/rootn, 3/rootn)
    plt.plot(t, stats.norm(scale=sigma/rootn).pdf(t))
    '''
    
    field = _make_field(xx, yy, xp, yp, complex_amps, wavelength)
    
    return field, xp, yp

np.random.seed(123)
field, xp, yp = make_field()
field_ = np.ravel(field)

#%% Plot scatterers
plt.figure()
plt.plot(xp, yp, ".")
plt.axis("square")
rect = mpl.patches.Rectangle((0, 0), 1, 1,linewidth=1,edgecolor="C1", facecolor='none')
plt.gca().add_patch(rect)
plt.title("Scatterers")

#%% Plot field
plt.figure()
plt.imshow(np.abs(field), extent=(0,1,0,1))
plt.axis("square")


#%% Calculate stds of pixels
n = 1000
wavelength = 0.2

m = 200
np.random.seed(123)
stds = np.zeros(m)
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    stds[k] = np.std(field.real)

plt.figure()
plt.hist(stds, density=True, bins=20)

print(f"Mean std: {np.mean(stds)}")
print(f"Std std: {np.std(stds)}")

#%%
np.min(field)
np.max(field)
np.mean(field)

#%% Normal dist
t = np.linspace(-3 * sigma, +3 * sigma, 100)
plt.figure()
plt.hist(field_.real, density=True, bins=30)
plt.plot(t, stats.norm(scale=sigma).pdf(t), label="Norm(0, Sigma^2)")
plt.legend()


#%% Rayleigh dist
rms = np.sqrt(np.mean(np.abs(field_)**2))

rayleigh = stats.rayleigh(scale=sigma)
fitte_drayleigh = stats.rayleigh(scale=rms / np.sqrt(2))
t = np.linspace(0, rayleigh.ppf(0.9999), 100)
plt.figure()
plt.hist(np.abs(field_), bins=30, density=True)
plt.plot(t, rayleigh.pdf(t), label="Rayleigh(sigma)")
plt.plot(t, fitte_drayleigh.pdf(t), label="Fitted Rayleigh")
plt.legend()

#%% ACA fft
n = 500
wavelength = 0.2

m = 200
np.random.seed(123)
acas = np.zeros(m)
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    z = field
    Z = np.fft.fft2(z)
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real

    acas[k] = np.sum(np.abs(autocorr_normed) > (1/np.e)) * dx * dy

plt.figure()
plt.hist(acas, density=True, bins=30)
plt.xlabel("ACA")

aca = np.mean(acas)

print("Method FFT2 autocorr")
print(f"ACA: {aca}")
print(f"Std: {np.std(acas)}")
print(f"Std/Mean: {np.std(acas)/aca}")
print(f"ACL: {np.sqrt(aca/np.pi)}")
print(f"Unique points: {1/aca}")

#%% ACA FFT with fine cluster identification

n = 500
wavelength = 0.2

m = 200
np.random.seed(123)
acas = np.zeros(m)
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    z = field
    Z = np.fft.fft2(z)
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    labels, _ = ndimage.measurements.label(np.abs(autocorr_normed) > 1 / np.e)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    acas[k] = np.sum(labels == centre_label) * dx * dy

plt.figure()
plt.hist(acas, density=True, bins=30)
plt.xlabel("ACA")

aca = np.mean(acas)

print("Method FFT2 autocorr with label identification")
print(f"ACA: {aca}")
print(f"Std: {np.std(acas)}")
print(f"Std/Mean: {np.std(acas)/aca}")
print(f"ACL: {np.sqrt(aca/np.pi)}")
print(f"Unique points: {1/aca}")


#%% ACA fft padded
n = 500
wavelength = 0.2

m = 200
np.random.seed(123)
fields = []
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    fields.append(field)

#%%
p = 2
acas = np.zeros(len(fields))
for k, field in enumerate(fields):
    z = field[:200,:200]
    Z = np.fft.fft2(z, (p * z.shape[0], p * z.shape[1]))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    labels, _ = ndimage.measurements.label(np.abs(autocorr_normed) > 1 / np.e)
    centre_label = labels[labels.shape[0] // 2, labels.shape[1] // 2]
    acas[k] = np.sum(labels == centre_label) * dx * dy

plt.figure()
plt.hist(acas, density=True, bins=30)
plt.xlabel("ACA")

aca = np.mean(acas)

print("Method FFT2 autocorr padded with label identification")
print(f"ACA: {aca}")
print(f"Std: {np.std(acas)}")
print(f"Std/Mean: {np.std(acas)/aca}")
print(f"ACL: {np.sqrt(aca/np.pi)}")
print(f"Unique points: {1/aca}")


µ#%%
plt.figure()
plt.imshow(np.abs(autocorr_normed), extent=(0,1,0,1))
plt.title("autocorrelation with fft")

plt.figure()
plt.imshow(np.abs(autocorr_normed) > 1 / np.e, extent=(0,1,0,1))
plt.title("autocorrelation with fft")

plt.figure()
plt.imshow(labels == centre_label, extent=(0,1,0,1))
plt.title("autocorrelation with fft")

#%% 1D autocorr

z = field[0]
Z = np.fft.fft(z)
_autocorr = np.fft.ifft(Z * np.conj(Z))
autocorr = np.fft.fftshift(_autocorr)
autocorr_normed = autocorr / _autocorr[0].real


plt.figure()
plt.plot(y, np.abs(autocorr_normed))

print("FWHM")
print(np.sum(np.abs(autocorr_normed) > 1/np.e) * dy)

#%% Find number of unique points using the dist of maximas

n = 500
wavelength = 0.2

m = 200
np.random.seed(123)
maximas = np.zeros(m)
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    maximas[k] = np.max(np.abs(field))
    
#%%

# around 400 for wavelength=0.2
# around 2000 for wavelength=0.1
    
t = np.linspace(0, 1.5 * maximas.max(), 200)

def maxrayleigh_pdf(t, n):
    dist = stats.rayleigh(scale=sigma)
    return n * dist.cdf(t)**(n-1) * dist.pdf(t)

plt.figure()
plt.hist(maximas, density=True, bins=20)
#plt.plot(t, maxrayleigh_pdf(t, 400))
#plt.plot(t, maxrayleigh_pdf(t, 2000))
plt.plot(t, maxrayleigh_pdf(t, 400))

#%% ACA goodman

n = 500
wavelength = 0.2

p = 2

m = 200
np.random.seed(123)
acas = np.zeros(m)
for k in tqdm.trange(m):
    field, xp, yp = make_field(n, wavelength)
    z = field
    Z = np.fft.fft2(z, (z.shape[0] * p, z.shape[1] * p))
    _autocorr = np.fft.ifft2(Z * np.conj(Z))
    autocorr = np.fft.fftshift(_autocorr)
    autocorr_normed = autocorr / _autocorr[0, 0].real
    acas[k] = np.trapz(np.trapz(np.abs(autocorr_normed), dx=dy/p), dx=dx/p)

plt.figure()
plt.hist(acas, density=True, bins=30)
plt.xlabel("ACA")

aca = np.mean(acas)

print("Method Goodman")
print(f"ACA: {aca}")
print(f"Std: {np.std(acas)}")
print(f"Std/Mean: {np.std(acas)/aca}")
print(f"ACL: {np.sqrt(aca/np.pi)}")
print(f"Unique points: {1/aca}")

#%% Autocorr of cos

t = np.linspace(-1, 1, 200, endpoint=False)
z = np.fft.fft(np.sin(2 * np.pi * t))
autocorr = np.fft.fftshift(np.fft.ifft(z * np.conj(z)))
autocorr = autocorr / np.max(np.abs(autocorr))
plt.figure()
plt.plot(t, np.abs(autocorr))
plt.plot(t, np.abs(autocorr) > 1/np.e)

#%% Autocorr of exp(-x^2) cos(x)

t = np.linspace(-1, 1, 200, endpoint=False)
y = np.cos(2 * np.pi * t) * np.exp(-t**2)

plt.figure()
plt.plot(t, y)

z = np.fft.fft(y)
autocorr = np.fft.fftshift(np.fft.ifft(z * np.conj(z)))
autocorr = autocorr / np.max(np.abs(autocorr))
plt.figure()
plt.plot(t, np.abs(autocorr))
plt.plot(t, np.abs(autocorr) > 1/np.e)


