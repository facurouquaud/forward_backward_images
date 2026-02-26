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

    image_flat = image.ravel()

    # --- Estimaciones iniciales ---
    A0 = image.max() - np.median(image)
    y0_0, x0_0 = np.unravel_index(np.argmax(image), image.shape)
    sigma0 = 3
    offset0 = np.median(image)

    p0 = (A0, x0_0, y0_0, sigma0, sigma0, offset0)

    # --- Ruido Poisson (ponderación) ---
    sigma_noise = np.sqrt(np.maximum(image_flat, 1))

    # --- Límites físicos ---
    bounds = (
        (0, 0, 0, 0.5, 0.5, 0),           # lower
        (np.inf, nx, ny, 20, 20, np.inf)  # upper
    )

    popt, pcov = curve_fit(
        gaussian_2d,
        (x, y),
        image_flat,
        p0=p0,
        sigma=sigma_noise,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=10000
    )

    perr = np.sqrt(np.diag(pcov))

    return popt, perr
#datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")
datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\Calibracion_ida_vuelta\10x10\calibracion_10x10_00_scan.NPY")
ida_img = datos[0][0] 
vuelta_img = datos[0][1]
params_ida,errors_ida = fit_gaussian_2d(ida_img.T)
params_vuelta, erros_vuelta = fit_gaussian_2d(vuelta_img.T)

A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida
_, err_x, err_y,_,_,_ = errors_ida
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


graficar(ida_img.T,10,200)
plt.scatter(x_i_um, y_i_um, c='red', s=50)

graficar(vuelta_img.T,10,200)
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
params_ida,_ = fit_gaussian_2d(ida_sum.T)
params_vuelta,_ = fit_gaussian_2d(vuelta_sum.T)
params_ida_vuelta,_ = fit_gaussian_2d(ida_vuelta)

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
datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_05_scan.NPY")

ida = datos[0][0][10:50,30:70].T 
vuelta = datos[0][1][10:50,30:70].T 
vuelta_alineada = shift(
    vuelta,
    shift=(+dy_pix, +dx_pix),
    mode='nearest')
ida_crop, vuelta_crop = crop_overlap(ida, vuelta_alineada, -dx_pix, - dy_pix)
ida_vuelta = ida_crop + vuelta_crop


#graficar(imagen_fondo,1,20)
imagen_fondo =  ida_vuelta[0:20,0:20]
fondo = np.mean(imagen_fondo)
imagen_sin_fondo = ida_vuelta - fondo
imagen_sin_fondo[imagen_sin_fondo < 0] = 0
imagen_sin_fondo = imagen_sin_fondo[12:22,16:26]
graficar(imagen_sin_fondo,0.5,10)

N_fotones = np.sum(imagen_sin_fondo)
#graficar(imagen_sin_fondo,1,20)
print("Fotones detectados:", N_fotones)

#%%Encontremos los fotones del emisor con el ajuste Gaussiano:

params_ida, _ = fit_gaussian_2d(ida)
params_iv, _ = fit_gaussian_2d(ida_vuelta)

A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida
A_iv, x0_iv, y0_iv, sx_iv, sy_iv, off_iv = params_iv

# Fotones totales (sin fondo)
N_ida = A_i * 2 * np.pi * sx_i * sy_i
N_iv = A_iv * 2 * np.pi * sx_iv * sy_iv

print("Fotones Ida:", N_ida)
print("Fotones Ida-Vuelta:", N_iv)
#%%
ida_2 = datos[0][0][10:50,30:70].T + datos[1][0][10:50,30:70].T 
fondo = np.mean(imagen_fondo)
imagen_sin_fondo = ida_2 - fondo
imagen_sin_fondo[imagen_sin_fondo < 0] = 0
imagen_sin_fondo = imagen_sin_fondo[12:22,16:26]
graficar(ida_2[14:22,18:26],0.4,8)

N_fotones = np.sum(imagen_sin_fondo)
#graficar(imagen_sin_fondo,1,20)
print("Fotones detectados:", N_fotones)

params_ida, _ = fit_gaussian_2d(ida_2)
A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida

# Fotones totales (sin fondo)
N_ida = A_i * 2 * np.pi * sx_i * sy_i

print("Fotones Ida:", N_ida)

#%%

def calcular_localizaciones(datos):
    x_ida = []
    y_ida = []
    x_vuelta = []
    y_vuelta= []
    x_union = []
    y_union = []
    
    n_frames = len(datos)
    
    for i in range(0, n_frames-1, 1):
    
        px_size = 10/200
    
        # --- 1️⃣ ROI + suma ---
        ida_sum = datos[i][0][10:50,30:70].T + datos[i+1][0][10:50,30:70].T
        vuelta_sum = datos[i][1][10:50,30:70].T + datos[i+1][1][10:50,30:70].T
        
    
        # --- 2️⃣ Shift ---
        dy_pix, dx_pix, _ = cross_correlation_shift(ida_sum, vuelta_sum)
    
        # --- 3️⃣ Alinear vuelta sobre ida ---
        vuelta_alineada = shift(
            vuelta_sum,
            shift=(dy_pix, dx_pix),
            mode='nearest')
        imagen_fondo =  vuelta_alineada[0:20,0:20]
        fondo = np.mean(imagen_fondo)
        vuelta_sin_fondo = vuelta_alineada - fondo
        vuelta_sin_fondo[vuelta_sin_fondo < 0] = 0
    
        # --- 4️⃣ Crop consistente ---
        ida_crop, vuelta_crop = crop_overlap(
            ida_sum,
            vuelta_sin_fondo,
            -dy_pix,
            -dx_pix
        )
    
        imagen_unida = ida_crop + vuelta_crop
        
     
    
        # --- 5️⃣ Ajustes ---
        params_ida, _ = fit_gaussian_2d(ida_sum)
        params_vuelta,_ = fit_gaussian_2d(vuelta_sum)
        params_ida_vuelta, _ = fit_gaussian_2d(imagen_unida)
        A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida
        A_v, x0_v, y0_v, sx_v, sy_v, off_v = params_vuelta
        A_u, x0_u, y0_u, sx_u, sy_u, off_u = params_ida_vuelta
        #plt.scatter(x0_i, y0_i, c='red', s=50)
        #plt.scatter(x0_v, y0_v, c= "blue", s = 50)
        #plt.scatter(x0_u, y0_u, c = "green", s = 50)
        #plt.imshow(ida_sum)
       # plt.imshow(vuelta_crop)
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
    
    x_ida_r = x_ida + 0.7
    x_vuelta_r = x_vuelta + 0.7
    x_union_r = x_union + 0.7
    y_ida_r = y_ida + 1.75
    y_vuelta_r = y_vuelta + 1.75
    y_union_r = y_union + 1.75
    return x_ida_r, y_ida_r, x_vuelta_r, y_vuelta_r, x_union_r, y_union_r


#%%
datos_14 =  sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_00_scan.NPY")

datos_7 = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_05_scan.NPY")
datos_17 = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_02_scan.NPY")
#datos_28 = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")




#x_ida_r28, y_ida_r28, x_vuelta_r28, y_vuelta_r28, x_union_r28, y_union_r28 = calcular_localizaciones(datos_28)
x_ida_r7, y_ida_r7, x_vuelta_r7, y_vuelta_r7, x_union_r7, y_union_r7 = calcular_localizaciones(datos_7)
x_ida_r17, y_ida_r17, x_vuelta_r17, y_vuelta_r17, x_union_r17, y_union_r17 = calcular_localizaciones(datos_17)
x_ida_r14, y_ida_r14, x_vuelta_r14, y_vuelta_r14, x_union_r14, y_union_r14 = calcular_localizaciones(datos_14)

#%%
def estadistico(x,y):
    return x-y

# x_28_iv = estadistico(x_ida_r28,x_vuelta_r28)
# x_28_iu = estadistico(x_ida_r28, x_union_r28)
# y_28_iv = estadistico(y_ida_r28,y_vuelta_r28)
# y_28_iu = estadistico(y_ida_r28, y_union_r28)





# plt.scatter(x_28_iv, y_28_iv, c='darkred', s=35, label = "Ida - vuelta")
# plt.scatter(x_28_iu, y_28_iu, c='olivedrab', s=35, label = "ida - union")
# plt.grid()
# plt.legend(fontsize = 13, loc = "upper left")
# # plt.xlim(1.10,1.42)
# # plt.ylim(2.82,3.1)
# plt.xlabel("x [µm]", fontsize = 16)
# plt.ylabel("y [µm]", fontsize = 16)

#%%
def procesar_dataset(datos):
    
    x_ida, y_ida, x_vuelta, y_vuelta, x_union, y_union = calcular_localizaciones(datos)
    
    resultados = {
        "x_ida": x_ida,
        "y_ida": y_ida,
        "x_vuelta": x_vuelta,
        "y_vuelta": y_vuelta,
        "x_union": x_union,
        "y_union": y_union,
        "dx_iv": x_ida - x_vuelta,
        "dy_iv": y_ida - y_vuelta,
        "dx_iu": x_ida - x_union,
        "dy_iu": y_ida - y_union,
    }
    
    return resultados


# --- Diccionario con todas las potencias ---
datasets = {
    "r7":  datos_7,
    "r17": datos_17,
    "r14": datos_14,
}


# --- Procesar todo automáticamente ---
resultados = {}

for key, datos in datasets.items():
    resultados[key] = procesar_dataset(datos)



x_7_iv  = resultados["r7"]["dx_iv"]
y_7_iv  = resultados["r7"]["dy_iv"]

x_17_iv = resultados["r17"]["dx_iv"]
y_17_iv = resultados["r17"]["dy_iv"]

x_14_iv = resultados["r14"]["dx_iv"]
y_14_iv = resultados["r14"]["dy_iv"]


x_7_iu = resultados["r7"]["dx_iu"]
y_7_iu = resultados["r7"]["dy_iu"]

x_17_iu = resultados["r17"]["dx_iu"]
y_17_iu = resultados["r17"]["dy_iv"]

x_14_iu = resultados["r14"]["dx_iu"]
y_14_iu = resultados["r14"]["dy_iu"]

x_iv = np.concatenate([x_7_iv, x_14_iv, x_17_iv])
y_iv = np.concatenate([y_7_iv, y_14_iv, y_17_iv])
x_iu = np.concatenate([x_7_iu, x_14_iu, x_17_iu])
y_iu = np.concatenate([y_7_iu, y_14_iu, y_17_iu])


x_iv_prom = np.mean(x_iv)
y_iv_prom = np.mean(y_iv)
x_iu_prom = np.mean(x_iu)
y_iu_prom = np.mean(y_iu)
plt.figure(constrained_layout=True)

plt.scatter(x_iv, y_iv, c='darkred', s=35, label = "Ida - vuelta")
plt.scatter(x_iu, y_iu, c='olivedrab', s=35, label = "ida - union")
plt.grid()
plt.legend(fontsize=12, loc="upper left")
plt.xlim(-0.07, 0.25)
plt.ylim(-0.15, 0.15)
plt.xlabel("x [µm]", fontsize=16)
plt.ylabel("y [µm]", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=11)

plt.show()

#%%
plt.figure(constrained_layout=True)

plt.hist(x_iv ,bins=16, alpha=0.7, label="Ida - vuelta", color = "darkred")
plt.hist(x_iu, bins = 16, alpha=0.7, label="ida - union", color = "olivedrab")
plt.grid()
plt.legend(fontsize = 14, loc = "upper center")
plt.xlabel("Posición en x ($\mu m$)", fontsize = 16)
plt.tick_params(axis='both', labelsize=11)
plt.ylabel("Frecuencia", fontsize = 16)


#%%

def calcular_fwhm(datos, bins=18):
    # Histograma (densidad para que no dependa del número de puntos)
    counts, edges = np.histogram(datos, bins=bins, density=True)
    
    centers = 0.5 * (edges[1:] + edges[:-1])
    
    max_value = np.max(counts)
    half_max = max_value / 2

    # Índices donde la curva supera la mitad
    indices = np.where(counts >= half_max)[0]
    
    if len(indices) < 2:
        return None  # No se puede determinar

    fwhm = centers[indices[-1]] - centers[indices[0]]
    
    return fwhm
fwhm_iv = calcular_fwhm(x_iv, bins=18)
fwhm_iu = calcular_fwhm(x_iu, bins=18)

print("Promedio ida - vuelta:", np.mean(x_iv), "µm")
print("FWHM Ida-vuelta:", fwhm_iv, "µm")

print("Promedio ida - unión:", np.mean(x_iu), "µm")
print("FWHM Ida-union:", fwhm_iu, "µm")

