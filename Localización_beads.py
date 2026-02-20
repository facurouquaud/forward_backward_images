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
from scipy.optimize import curve_fit
from scipy.ndimage import shift


datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")
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


graficar(ida_sum.T,10,200)
plt.scatter(x_i_um, y_i_um, c='red', s=50)

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
    mode='constant'
)
# Diferencia residual
residual = ida_sum.T - vuelta_alineada
rms_error = np.sqrt(np.mean(residual**2))


# Error geométrico total
error_total_um = np.sqrt(dx_um**2 + dy_um**2)

graficar(ida_sum.T,10,200)
graficar(vuelta_alineada,10,200)

def crop_overlap(img1, img2, dx, dy):
    ny, nx = img1.shape

    y_min = max(0, dy)
    y_max = min(ny, ny + dy)

    x_min = max(0, -dx)
    x_max = min(nx, nx - dx)

    img1_crop = img1[y_min:y_max, x_min:x_max]
    img2_crop = img2[y_min:y_max, x_min:x_max]

    return img1_crop, img2_crop
ida_crop, vuelta_crop = crop_overlap(ida_sum.T, vuelta_alineada, dy_pix, dx_pix)

ida_vuelta = ida_crop + vuelta_crop

graficar(ida_vuelta,10,200)
params_ida_vuelta = fit_gaussian_2d(ida_vuelta)

A_i, x0_iv, y0_iv, sx_i, sy_i, off_i = params_ida_vuelta


# Convertir posiciones a µm
x_i_um = x0_iv * px_size_x
y_i_um = y0_iv * px_size_y
print(x0_i,y0_i)
print(x0_v,y0_v)
print(x0_iv,y0_iv)
#%%

x_centers = []
y_centers = []

n_frames = len(datos)

for i in range(0, n_frames-1, 2):

    # --- 1️⃣ Sumar dos idas y dos vueltas ---
    ida_sum = datos[i][0][15:40,35:85].T + datos[i+1][0][15:40,35:85].T
    vuelta_sum = datos[i][1][15:40,35:85].T + datos[i+1][1][15:40,35:85].T

    # --- 2️⃣ Calcular shift ---
    dy_pix, dx_pix, _ = cross_correlation_shift(ida_sum, vuelta_sum)

    # --- 3️⃣ Alinear vuelta ---
    vuelta_alineada = shift(
        vuelta_sum,
        shift=(dy_pix, dx_pix),
        mode='constant'
    )

    # --- 4️⃣ Recortar región común ---
    ida_crop, vuelta_crop = crop_overlap(
        ida_sum,
        vuelta_alineada,
        dy_pix,
        dx_pix
    )

    # --- 5️⃣ Unir ---
    imagen_unida = ida_crop + vuelta_crop
    imagen_unida[10:60,10:60]
    # 5️⃣ Ajustar
    params = fit_gaussian_2d(imagen_unida)
    A_i, x0_i, y0_i, sx_i, sy_i, off_i = params
    
    x0 = x0_i
    y0 = y0_i

    x_centers.append(x0)
    y_centers.append(y0)
    plt.imshow(imagen_unida)
    plt.scatter(x0_i, y0_i, c='red', s=50)

x_centers = np.array(x_centers) * px_size_x
y_centers = np.array(y_centers) * px_size_y
sigma_x_exp = np.std(x_centers)
sigma_y_exp = np.std(y_centers)

print(f"Precisión experimental:")
print(f"sigma_x = {sigma_x_exp:.3f} µm")
print(f"sigma_y = {sigma_y_exp:.3f} µm")
#%%
img_ida = datos[0][0] + datos[0][1]
img_vuelta = datos[1][0] + datos[1][1]
dy_pix, dx_pix, _ = cross_correlation_shift(img_ida, img_vuelta)

# --- 3️⃣ Alinear vuelta ---
vuelta_alineada = shift(
    img_vuelta,
    shift=(dy_pix, dx_pix),
    mode='constant'
)
ida_crop, vuelta_crop = crop_overlap(
    img_ida,
    vuelta_alineada,
    dy_pix,
    dx_pix
)
imagen_unida = ida_crop + vuelta_crop
roi = imagen_unida[15:40,35:85]
background = np.median(roi)
roi_sin_fondo = roi - background
roi_sin_fondo[roi_sin_fondo < 0] = 0

N_total = np.sum(roi_sin_fondo)
plt.imshow(roi_sin_fondo)



#%%Estudio del sesgo 
datos_75 =  sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_05_scan.NPY")
datos_14 = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_00_scan.NPY")
datos_285 = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")
datos_17 = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_02_scan.NPY")
def localizar(image):

    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    roi = image[y_max-12:y_max+12, x_max-12:x_max+12]

    params = fit_gaussian_2d(roi)
    A, x0, y0, sx, sy, off = params

    return x0 , y0 

x_ida_list = []
x_vuelta_list = []
x_union_list = []

half = 8
n_frames = len(datos)
n_frames = n_frames - (n_frames % 2)

x_ida_list = []
x_vuelta_list = []
x_union_list = []

n_frames = len(datos)

def procesar_dataset(datos):

    x_ida_list = []
    x_vuelta_list = []
    x_union_list = []

    n_frames = len(datos)

    for k in range(n_frames):

        ida = datos[k][0]
        vuelta = datos[k][1]

        # ---- Localización ida
        x_i, y_i = localizar(ida)
        x_ida_list.append(x_i)

        # ---- Localización vuelta
        x_v, y_v = localizar(vuelta)
        x_vuelta_list.append(x_v)

        # ---- Alinear y unir
        dy, dx, _ = cross_correlation_shift(ida, vuelta)
        vuelta_alineada = shift(vuelta, shift=(dy, dx), mode='constant')

        ida_crop, vuelta_crop = crop_overlap(ida, vuelta_alineada, dy, dx)
        union = ida_crop + vuelta_crop

        x_u, y_u = localizar(union)
        x_union_list.append(x_u)

    return (
        np.array(x_ida_list),
        np.array(x_vuelta_list),
        np.array(x_union_list)
    )

#%%
x_ida_75, x_vuelta_75, x_union_75 = procesar_dataset(datos_75)

#%%
x_ida_14, x_vuelta_14, x_union_14 = procesar_dataset(datos_14)
#%%
x_ida_17, x_vuelta_17, x_union_17 = procesar_dataset(datos_17)
#%%
x_ida_285, x_vuelta_285, x_union_285 = procesar_dataset(datos_285)


#%%
def localizar(image):

    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)

    roi = image[15:40,35:85]

    params = fit_gaussian_2d(roi)
    A, x0, y0, sx, sy, off = params

    return x0 , y0 

x_ida_list = []
x_vuelta_list = []
x_union_list = []

half = 8
n_frames = len(datos)
n_frames = n_frames - (n_frames % 2)

x_ida_list = []
x_vuelta_list = []
x_union_list = []

n_frames = len(datos)

def procesar_dataset(datos):

    x_ida_list = []
    x_vuelta_list = []
    x_union_list = []

    n_frames = len(datos)

    for k in range(n_frames):

        ida = datos[k][0]
        vuelta = datos[k][1]

        # ---- Localización ida
        x_i, y_i = localizar(ida)
        x_ida_list.append(x_i)

        # ---- Localización vuelta
        x_v, y_v = localizar(vuelta)
        x_vuelta_list.append(x_v)

        # ---- Alinear y unir
        dy, dx, _ = cross_correlation_shift(ida, vuelta)
        vuelta_alineada = shift(vuelta, shift=(dy, dx), mode='constant')

        ida_crop, vuelta_crop = crop_overlap(ida, vuelta_alineada, dy, dx)
        union = ida_crop + vuelta_crop

        x_u, y_u = localizar(union)
        x_union_list.append(x_u)

    return (
        np.array(x_ida_list),
        np.array(x_vuelta_list),
        np.array(x_union_list)
    )

#%%
x_ida_75, x_vuelta_75, x_union_75 = procesar_dataset(datos_75)

#%%
x_ida_14, x_vuelta_14, x_union_14 = procesar_dataset(datos_14)
#%%
x_ida_17, x_vuelta_17, x_union_17 = procesar_dataset(datos_17)
#%%
x_ida_285, x_vuelta_285, x_union_285 = procesar_dataset(datos_285)


#%%

#%% sumando de a dos frames
# Procesar SOLO dataset 285
x_ida_285, x_vuelta_285, x_union_285 = procesar_dataset(datos_17)

# Convertir a nm
pixel_size_nm = 10/200  # ajustá si hace falta

x_ida_nm = x_ida_285 * pixel_size_nm
x_vuelta_nm = x_vuelta_285 * pixel_size_nm
x_union_nm = x_union_285 * pixel_size_nm
plt.figure()
plt.hist(x_ida_nm, bins=12, alpha=0.7, label="Ida")
plt.hist(x_vuelta_nm, bins=12, alpha=0.7, label="Vuelta")
plt.hist(x_union_nm, bins=8, alpha=0.7, label="Ida + Vuelta")

plt.legend()
plt.xlabel("Posición en x ($\mu m$)")
plt.ylabel("Frecuencia")

print("Bias ida-vuelta (nm):",
      np.mean(x_vuelta_nm - x_ida_nm))
