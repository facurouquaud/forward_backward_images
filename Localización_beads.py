# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:31:57 2026

@author: Luis1
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import tifffile as tiff
from PIL import Image
from scipy.signal import find_peaks
import scan_datafile as sd
from skimage.measure import profile_line
plt.style.use(r"C:\Users\Luis1\Downloads\gula_style.mplstyle")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import random


datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_02_scan.NPY")
ida_img = datos[15][0]
vuelta_img = datos[15][1]

# Inicializo acumuladores
ida_sum = None
vuelta_sum = None

for frame in datos:
    ida = frame[0]
    vuelta = frame[1]
    
    if ida_sum is None:
        ida_sum = np.array(ida, dtype=float)
        vuelta_sum = np.array(vuelta, dtype=float)
    else:
        ida_sum += ida
        vuelta_sum += vuelta
#%%
def graficar(imagen,tamano_um,n_pix):
    shape = (n_pix,n_pix)
    x = np.linspace(0, tamano_um, shape[1])  # horizontal (cols)
    y = np.linspace(0, tamano_um, shape[0])  # vertical (rows)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

   # ax.set_title("(ida) ", fontsize=12, fontweight='bold')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')  
graficar(ida_sum,10,200)

#%%
from scipy.optimize import curve_fit

def gaussian_2d(coords, A, x0, y0, sx, sy, B):
    x, y = coords
    g = A * np.exp(-((x - x0)**2)/(2*sx**2) - ((y - y0)**2)/(2*sy**2)) + B
    return g.ravel()

img = ida_sum.T

def fit_gaussian_2d(image):
    ny, nx = image.shape
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y)

    # Estimaciones iniciales
    A0 = image.max()
    y0_0, x0_0 = np.unravel_index(np.argmax(image), image.shape)
    sigma0 = 3
    offset0 = np.median(image)

    p0 = (A0, x0_0, y0_0, sigma0, sigma0, offset0)

    popt, _ = curve_fit(
        gaussian_2d,
        (x, y),
        image.ravel(),
        p0=p0
    )

    return popt  # parámetros ajustados
params_ida = fit_gaussian_2d(ida_sum.T)
params_vuelta = fit_gaussian_2d(vuelta_sum.T)

A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida
A_v, x0_v, y0_v, sx_v, sy_v, off_v = params_vuelta

# Tamaño de la imagen
ny, nx = ida_sum.shape

# Campo de visión en µm (ajustá estos valores a tu escaneo real)
FOV_x = 10  # µm
FOV_y = 10  # µm

# Tamaño de píxel en µm
px_size_x = FOV_x / nx
px_size_y = FOV_y / ny

# Convertir posiciones a µm
x_i_um = x0_i * px_size_x
y_i_um = y0_i * px_size_y

x_v_um = x0_v * px_size_x
y_v_um = y0_v * px_size_y

# Convertir sigmas a µm
sx_i_um = sx_i * px_size_x
sy_i_um = sy_i * px_size_y

sx_v_um = sx_v * px_size_x
sy_v_um = sy_v * px_size_y

print("IDA:")
print(f"x0 = {x_i_um:.3f} µm, y0 = {y_i_um:.3f} µm")
print(f"sigma_x = {sx_i_um:.3f} µm, sigma_y = {sy_i_um:.3f} µm")

print("\nVUELTA:")
print(f"x0 = {x_v_um:.3f} µm, y0 = {y_v_um:.3f} µm")
print(f"sigma_x = {sx_v_um:.3f} µm, sigma_y = {sy_v_um:.3f} µm")


graficar(ida_sum.T,10,200)
plt.scatter(x_i_um, y_i_um, c='red', s=50)

graficar(vuelta_sum.T,10,200)
plt.scatter(x_v_um, y_v_um, c='red', s=50)


#%%
ny, nx = ida_sum.shape

FOV_x = 10  # µm
FOV_y = 10  # µm

px_size_x = FOV_x / nx
px_size_y = FOV_y / ny


def cross_correlation_shift(img1, img2):
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)

    corr = fftconvolve(img1, img2[::-1, ::-1], mode='same')

    y0, x0 = np.unravel_index(np.argmax(corr), corr.shape)

    center_y = corr.shape[0] // 2
    center_x = corr.shape[1] // 2

    dy_pix = y0 - center_y
    dx_pix = x0 - center_x

    return dy_pix, dx_pix, corr

dy_pix, dx_pix, corr = cross_correlation_shift(ida_sum.T, vuelta_sum.T)

# Convertir a micrómetros
dx_um = dx_pix * px_size_x
dy_um = dy_pix * px_size_y

print("Shift por correlación cruzada:")
print(f"dx = {dx_pix:.3f} px  ({dx_um:.3f} µm)")
print(f"dy = {dy_pix:.3f} px  ({dy_um:.3f} µm)")

# Convertir a micrómetros
dx_um = dx_pix * px_size_x
dy_um = dy_pix * px_size_y

print("Shift por correlación cruzada:")
print(f"dx = {dx_pix:.3f} px  ({dx_um:.3f} µm)")
print(f"dy = {dy_pix:.3f} px  ({dy_um:.3f} µm)")
from scipy.ndimage import shift

vuelta_alineada = shift(
    vuelta_sum.T,
    shift=(dy_pix, dx_pix),
    mode='constant'
)
# Diferencia residual
residual = ida_sum.T - vuelta_alineada
rms_error = np.sqrt(np.mean(residual**2))

print(f"\nError RMS intensidad = {rms_error:.3f}")

# Error geométrico total
error_total_um = np.sqrt(dx_um**2 + dy_um**2)

print(f"Desplazamiento total = {error_total_um:.3f} µm")
graficar(ida_sum.T,10,200)
graficar(vuelta_alineada,10,200)

def crop_overlap(img1, img2, dy, dx):
    ny, nx = img1.shape

    y_min = max(0, dy)
    y_max = min(ny, ny + dy)

    x_min = max(0, dx)
    x_max = min(nx, nx + dx)

    img1_crop = img1[y_min:y_max, x_min:x_max]
    img2_crop = img2[y_min:y_max, x_min:x_max]

    return img1_crop, img2_crop
ida_crop, vuelta_crop = crop_overlap(ida_sum.T, vuelta_alineada, dy_pix, dx_pix)


graficar(ida_crop + vuelta_crop,10,200)

#%% Construccion de la curva de localización simulada


