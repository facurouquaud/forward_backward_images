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
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import random
from scipy.optimize import curve_fit
from scipy.ndimage import shift


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
#%%
def graficar(imagen,tamano_um,n_pix):
    shape = (n_pix,n_pix)
    x = np.linspace(0, tamano_um, shape[1])  # horizontal (cols)
    y = np.linspace(0, tamano_um, shape[0])  # vertical (rows)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')  
def graficar_pos(imagen, tamano_um, n_pix, x_i_um, y_i_um):

    shape = (n_pix, n_pix)
    x = np.linspace(0, tamano_um, shape[1])
    y = np.linspace(0, tamano_um, shape[0])

    fig, ax = plt.subplots(constrained_layout=True)

    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')

    # marcador
    ax.scatter(x_i_um, y_i_um,
               c='aliceblue',
               s=60,
               edgecolor='black',
               label=f"({x_i_um:.2f}, {y_i_um:.2f}) µm")

    ax.set_xlabel("x [µm]", fontsize = 20)
    ax.set_ylabel("y [µm]", fontsize = 20)
    ax.legend(fontsize = 15)

    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')

    plt.show()
graficar(ida_sum.T,10,200)

#%%

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


graficar(ida_sum.T,10,200, x_i_um, y_i_um)
plt.legend()

graficar(vuelta_sum.T,10,200)
plt.scatter(x_v_um, y_v_um, c='red', s=50)


#%%

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


# Convertir a micrómetros
dx_um = dx_pix * px_size_x
dy_um = dy_pix * px_size_y



vuelta_alineada = shift(
    vuelta_sum.T,
    shift=(+dy_pix, +dx_pix),
    mode='nearest')
# Diferencia residual
residual = ida_sum.T - vuelta_alineada
rms_error = np.sqrt(np.mean(residual**2))


# Error geométrico total
error_total_um = np.sqrt(dx_um**2 + dy_um**2)


def crop_overlap(img1, img2, dx, dy):
    ny, nx = img1.shape

    y_min = max(0, dy)
    y_max = min(ny, ny + dy)

    x_min = max(0, dx)
    x_max = min(nx, nx + dx)

    img1_crop = img1[y_min:y_max, x_min:x_max]
    img2_crop = img2[y_min:y_max, x_min:x_max]

    return img1_crop, img2_crop
ida_crop, vuelta_crop = crop_overlap(ida_sum.T, vuelta_alineada, -dx_pix, - dy_pix)

ida_vuelta = ida_crop + vuelta_crop
params_ida = fit_gaussian_2d(ida_sum.T)
params_vuelta = fit_gaussian_2d(vuelta_sum.T)
params_ida_vuelta = fit_gaussian_2d(ida_vuelta)

A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida
A_v, x0_v, y0_v, sx_v, sy_v, off_v = params_vuelta
A_i, x0_iv, y0_iv, sx_i, sy_i, off_i = params_ida_vuelta


# Convertir posiciones a µm
x_i_um = x0_i * px_size_x
y_i_um = y0_i * px_size_y
x_v_um = x0_v * px_size_x
y_v_um = y0_v * px_size_y
x_iv_um = x0_iv * px_size_x
y_iv_um = y0_iv * px_size_y
print(x0_i,y0_i)
print(x0_v,y0_v)
print(x0_iv,y0_iv)

graficar_pos(ida_sum.T,10,200,x_i_um, y_i_um)
graficar_pos(vuelta_sum.T,10,200,x_v_um, y_v_um)
graficar_pos(ida_vuelta,10,200, x_iv_um, y_iv_um)
#%%%
#%%

x_ida = []
y_ida = []
x_vuelta = []
y_vuelta= []
x_union = []
y_union = []

n_frames = len(datos)

for i in range(0, n_frames-1, 2):

    px_size = 10/200

    # --- 1️⃣ ROI + suma ---
    ida_sum = datos[i][0][15:40,35:85].T + datos[i+1][0][15:40,35:85].T
    vuelta_sum = datos[i][1][15:40,35:85].T + datos[i+1][1][15:40,35:85].T
    

    # --- 2️⃣ Shift ---
    dy_pix, dx_pix, _ = cross_correlation_shift(ida_sum, vuelta_sum)

    # --- 3️⃣ Alinear vuelta sobre ida ---
    vuelta_alineada = shift(
        vuelta_sum,
        shift=(dy_pix, dx_pix),
        mode='nearest')

    # --- 4️⃣ Crop consistente ---
    ida_crop, vuelta_crop = crop_overlap(
        ida_sum,
        vuelta_alineada,
        -dy_pix,
        -dx_pix
    )

    imagen_unida = ida_crop + vuelta_crop

    # --- 5️⃣ Ajustes ---
    A_i, x0_i, y0_i, sx_i, sy_i, off_i = fit_gaussian_2d(ida_sum)
    A_v, x0_v, y0_v, sx_v, sy_v, off_v = fit_gaussian_2d(vuelta_sum)
    A_u, x0_u, y0_u, sx_u, sy_u, off_u = fit_gaussian_2d(imagen_unida)
    plt.scatter(x0_i, y0_i, c='red', s=50)
    plt.scatter(x0_v, y0_v, c= "blue", s = 50)
    plt.scatter(x0_u, y0_u, c = "green", s = 50)
    #plt.imshow(ida_sum)
    plt.imshow(vuelta_sum)
    #plt.imshow(imagen_unida)

    # --- 6️⃣ Convertir a µm ---
    x0_i *= px_size
    y0_i *= px_size

    x0_v *= px_size
    y0_v *= px_size

    x0_u *= px_size
    y0_u *= px_size

    # --- 7️⃣ Guardar ---
    x_ida.append(x0_i)
    y_ida.append(y0_i)

    x_vuelta.append(x0_v)
    y_vuelta.append(y0_v)

    x_union.append(x0_u)
    y_union.append(y0_u)

x_ida = np.array(x_ida) 
y_ida = np.array(y_ida)
x_vuelta = np.array(x_vuelta) 
y_vuelta = np.array(y_vuelta)
x_union = np.array(x_union) 
y_union = np.array(y_union)
sigma_x_exp = np.std(x_union)
sigma_y_exp = np.std(x_union)

print(f"Precisión experimental:")
print(f"sigma_x = {sigma_x_exp:.3f} µm")
print(f"sigma_y = {sigma_y_exp:.3f} µm")
#%%
# #%%
# graficar(ida_sum.T,10,200)
x_ida_r = x_ida + 0.7
x_vuelta_r = x_vuelta + 0.7
x_union_r = x_union + 0.7
y_ida_r = y_ida + 01.75
y_vuelta_r = y_vuelta + 1.75
y_union_r = y_union + 1.75
plt.scatter(x_ida_r, y_ida_r, c='darkred', s=35, label = "Ida")
plt.scatter(x_vuelta_r, y_vuelta_r, c='darkblue', s=35, label = "Vuelta")
plt.scatter(x_union_r, y_union_r, c="olivedrab", s=35, label = "Unión")
plt.grid()
plt.legend(fontsize = 13, loc = "upper left")
plt.xlim(1.10,1.42)
plt.ylim(2.82,3.1)
plt.xlabel("x [µm]", fontsize = 16)
plt.ylabel("y [µm]", fontsize = 16)




#%%
plt.hist(x_ida_r ,bins=5, alpha=0.7, label="Ida", color = "darkred")
plt.hist(x_vuelta_r, bins=5, alpha=0.7, label="Vuelta", color = "darkblue")
plt.hist(x_union_r, bins=5, alpha=0.7, label="Ida + Vuelta", color = "olivedrab")
plt.grid()
plt.legend()
plt.xlabel("Posición en x ($\mu m$)", fontsize = 16)
plt.ylabel("Frecuencia", fontsize = 16)

print("Bias ida-vuelta (nm):",
      np.mean(x_vuelta - x_ida))
# graficar(vuelta_alineada,10,200)
# graficar(ida_vuelta,10,200)
#%%
