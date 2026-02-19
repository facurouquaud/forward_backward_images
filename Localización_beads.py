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
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import random


datos = sd.ScanDataFile.open(r"C:\Users\Lenovo\Downloads\Calibracion_ida_vuelta\scan_07_scan.NPY")
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
#graficar(ida_sum,10,200)



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

    A0 = image.max()
    y0_0, x0_0 = np.unravel_index(np.argmax(image), image.shape)
    sigma0 = 3
    offset0 = np.median(image)

    p0 = (A0, x0_0, y0_0, sigma0, sigma0, offset0)

    popt, pcov = curve_fit(
        gaussian_2d,
        (x, y),
        image.ravel(),
        p0=p0
    )

    perr = np.sqrt(np.diag(pcov))  # error estándar de cada parámetro

    return popt, perr

graficar(ida_img,10,200)
#%%
params_ida, err_ida = fit_gaussian_2d(ida_sum.T)
params_vuelta, err_vuelta = fit_gaussian_2d(vuelta_sum.T)

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
print(f"y0 = {x_i_um:.3f} µm, x0 = {y_i_um:.3f} µm")
print(f"sigma_x = {sx_i_um:.3f} µm, sigma_y = {sy_i_um:.3f} µm")

print("\nVUELTA:")
print(f"y0 = {x_v_um:.3f} µm, x0 = {y_v_um:.3f} µm")
print(f"sigma_x = {sx_v_um:.3f} µm, sigma_y = {sy_v_um:.3f} µm")

error_x0_ida = err_ida[1]
error_y0_ida = err_ida[2]

error_x0_vuelta = err_vuelta[1]
error_y0_vuelta = err_vuelta[2]

error_x0_ida_um = error_x0_ida * px_size_x
error_y0_ida_um = error_y0_ida * px_size_y
#graficar(ida_sum.T,10,200)
#plt.scatter(x_i_um, y_i_um, c='red', s=50)

#graficar(vuelta_sum.T,10,200)
#plt.scatter(x_v_um, y_v_um, c='red', s=50)


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
#graficar(ida_sum.T,10,200)
#graficar(vuelta_alineada,10,200)

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

imagen_unida = ida_crop + vuelta_crop
graficar(imagen_unida,10,200)

params_union, err_union = fit_gaussian_2d(imagen_unida.T)

A_u, x0_u, y0_u, sx_u, sy_u, off_u = params_union
# Tamaño de píxel en µm
px_size_x = FOV_x / nx
px_size_y = FOV_y / ny

# Convertir posiciones a µm
x_u_um = x0_u * px_size_x
y_u_um = y0_u * px_size_y

# Convertir sigmas a µm
sx_u_um = sx_u * px_size_x
sy_u_um = sy_u * px_size_y


error_x0_union = err_union[1]
error_y0_union = err_union[2]


#%% Construccion de la curva de localización simulada
# Región de imagen
FOV = 10.0  # µm
n_pixels = 200
px_size = FOV / n_pixels

# Grilla en µm
x = np.linspace(0, FOV, n_pixels)
y = np.linspace(0, FOV, n_pixels)
X, Y = np.meshgrid(x, y)

# Parámetros verdaderos del emisor
x0_true = 5.0       # µm
y0_true = 5.0       # µm
sigma_true = 0.15  # µm (PSF)
background = 5   # cuentas promedio por pixel

# def simulate_gaussian_image(Nphotons):

#     # Gaussiana discreta normalizada
#     gauss = np.exp(-((X - x0_true)**2/(2*sigma_true**2) +
#                      (Y - y0_true)**2/(2*sigma_true**2)))

#     gauss /= gauss.sum()   # NORMALIZACIÓN DISCRETA

#     image = Nphotons * gauss + background

#     noisy = np.random.poisson(image)

#     y_max, x_max = np.unravel_index(np.argmax(noisy), noisy.shape)

#     p0 = (
#         noisy.max(),
#         x[x_max],
#         y[y_max],
#         0.2,
#         0.2,
#         background
#     )

#     try:
#         popt, _ = curve_fit(
#             gaussian_2d,
#             (X, Y),
#             noisy.ravel(),
#             p0=p0,
#             maxfev=5000
#         )
#         return popt[1], popt[2], noisy

#     except:
#         return np.nan, np.nan
def simulate_gaussian_image(Nphotons):

    # --------- Generar imagen ----------
    gauss = np.exp(-((X - x0_true)**2/(2*sigma_true**2) +
                      (Y - y0_true)**2/(2*sigma_true**2)))

    gauss /= gauss.sum()

    image = Nphotons * gauss + background
    noisy = np.random.poisson(image)

    # --------- Encontrar máximo ----------
    y_max, x_max = np.unravel_index(np.argmax(noisy), noisy.shape)

    # --------- Recortar ROI ----------
    roi_half = 10   # 21x21 píxeles aprox
    xmin = max(x_max - roi_half, 0)
    xmax = min(x_max + roi_half, n_pixels)
    ymin = max(y_max - roi_half, 0)
    ymax = min(y_max + roi_half, n_pixels)

    roi = noisy[ymin:ymax, xmin:xmax]
    X_roi = X[ymin:ymax, xmin:xmax]
    Y_roi = Y[ymin:ymax, xmin:xmax]

    # --------- Estimación inicial robusta (centroide) ----------
    total = roi.sum()
    if total <= 0:
        return np.nan, np.nan, noisy

    x0_guess = (X_roi * roi).sum() / total
    y0_guess = (Y_roi * roi).sum() / total

    A_guess = roi.max() - np.median(roi)
    sigma_guess = 0.2
    offset_guess = np.median(roi)

    p0 = (A_guess, x0_guess, y0_guess,
          sigma_guess, sigma_guess, offset_guess)

    # --------- Bounds físicos ----------
    bounds = (
        [0, x0_guess-1, y0_guess-1, 0.05, 0.05, 0],
        [np.inf, x0_guess+1, y0_guess+1, 1.0, 1.0, 10]
    )

    # --------- Ajuste ----------
    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (X_roi, Y_roi),
            roi.ravel(),
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

        return popt[1], popt[2], noisy

    except:
        return np.nan, np.nan, noisy



_,_, simu= simulate_gaussian_image(1000)
graficar(simu, 10,200)
params_simu = fit_gaussian_2d(simu)
A_s, x0_s, y0_s, sx_s, sy_s, off_s = params_simu
#Tamaño de píxel en µm
px_size_x = FOV_x / nx
px_size_y = FOV_y / ny

# Convertir posiciones a µm
x_s_um = x0_s * px_size_x
y_s_um = y0_s * px_size_y



# Convertir sigmas a µm
sx_s_um = sx_s * px_size_x
sy_s_um = sy_s * px_size_y


#%%

def localization_precision(Nphotons, n_repeats=200):

    x_positions = []

    for _ in range(n_repeats):

        x_fit, y_fit, _ = simulate_gaussian_image(Nphotons)

        if not np.isnan(x_fit):
            x_positions.append(x_fit)

    x_positions = np.array(x_positions)

    return np.std(x_positions)


N_values = np.linspace(250,1000,60)
precisions = []
for N in N_values:
    prec = localization_precision(N)
    precisions.append(prec)
precisions = np.array(precisions)
#%%
plt.figure()

plt.plot(N_values, precisions, 'o-', label="Simulación")
sigma = np.sqrt(0.13**2 + 0.13**2)
#sigma = 0.26
plt.scatter(595,sigma, label = "Experimental",color = "g")
plt.xlabel("Número de fotones N")
plt.ylabel("Precisión de localización (µm)")
plt.legend()
plt.grid()
plt.show()

