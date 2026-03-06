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


# datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_02_scan.NPY")

# # Inicializo acumuladores
# ida_sum = None
# vuelta_sum = None

# for frame in datos:
#     ida = frame[0][5:55,20:70]
#     vuelta = frame[1][5:55,20:70]
    
#     if ida_sum is None:
#         ida_sum = np.array(ida, dtype=float)
#         vuelta_sum = np.array(vuelta, dtype=float)
#     else:
#         ida_sum += ida
#         vuelta_sum += vuelta


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
    fig.colorbar(im, ax=ax, label="Número de fotones", fontsize = 16)
    ax.set_aspect('equal', adjustable='box')  
def graficar_pos(imagen, tamano_um, n_pix, x_i_um, y_i_um, conv_x, conv_y):

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
               label=f"({x_i_um + conv_x:.3f}, {y_i_um + conv_y:.3f}) µm")

    ax.set_xlabel("x [µm]", fontsize = 20)
    ax.set_ylabel("y [µm]", fontsize = 20)
    ax.legend(fontsize = 15)

    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')

    plt.show()
# graficar(ida_sum,2.5,50)

#%%

def gaussian_2d(coords, A, x0, y0, sx, sy, offset):
    x, y = coords
    g = A * np.exp(
        -((x - x0)**2 / (2*sx**2) +
          (y - y0)**2 / (2*sy**2))
    ) + offset
    return g.ravel()

def fit_gaussian_2d(image):

    ny, nx = image.shape
    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y)

    image_flat = image.ravel()

    # --- Estimaciones iniciales ---
    offset0 = np.median(image)
    A0 = image.max() - offset0
    y0_0, x0_0 = np.unravel_index(np.argmax(image), image.shape)
    sigma0 = 15

    p0 = (A0, x0_0, y0_0, sigma0, sigma0, offset0)

    bounds = (
        (0, 0, 0, 0.5, 0.5, 0),
        (np.inf, nx, ny, 15, 15, np.inf)
    )

  

    popt, pcov = curve_fit(
        gaussian_2d,
        (x, y),
        image_flat,
        p0=p0,
       
        bounds=bounds,
        maxfev=200000
    )
    perr = np.sqrt(np.diag(pcov))

    return popt, perr
#datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")
# ida_img = datos[0][0][5:55,20:70]
# vuelta_img = datos[0][1][5:55,20:70]
# params_ida,errors_ida = fit_gaussian_2d(ida_img.T)
# params_vuelta, erros_vuelta = fit_gaussian_2d(vuelta_img.T)

# A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida
# _, err_x, err_y,_,_,_ = errors_ida
# A_v, x0_v, y0_v, sx_v, sy_v, off_v = params_vuelta

# # Tamaño de la imagen
# ny, nx = ida_sum.shape

# # Campo de visión en µm (ajustá estos valores a tu escaneo real)
# FOV_x = 2.5  # µm
# FOV_y = 2.5  # µm

# # Tamaño de píxel en µm
# px_size_x = FOV_x / nx
# px_size_y = FOV_y / ny

# # Convertir posiciones a µm
# x_i_um = x0_i * px_size_x
# y_i_um = y0_i * px_size_y

# x_v_um = x0_v * px_size_x
# y_v_um = y0_v * px_size_y


# graficar(ida_img,2.5,50)
# plt.scatter(x_i_um, y_i_um, c='red', s=50)
# #%%
# graficar(vuelta_img.T,2.5,50)
# plt.scatter(x_v_um, y_v_um, c='red', s=50)


# #%%

# #%%%
# datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")

# ida = datos[0][0][5:50,30:75].T 
# vuelta = datos[0][1][5:50,30:75].T 
# vuelta_alineada = shift(
#     vuelta,
#     shift=(+dy_pix, +dx_pix),
#     mode='nearest')


# #graficar(imagen_fondo,1,20)
# imagen_fondo =  ida_vuelta[0:20,0:20]
# fondo = np.mean(imagen_fondo)
# imagen_sin_fondo = ida_vuelta - fondo
# imagen_sin_fondo[imagen_sin_fondo < 0] = 0
# imagen_sin_fondo = imagen_sin_fondo[12:22,16:26]
# graficar(imagen_sin_fondo,0.5,10)

# N_fotones = np.sum(imagen_sin_fondo)
# #graficar(imagen_sin_fondo,1,20)
# print("Fotones detectados:", N_fotones)



# #%%
# ida_2 = datos[0][0][5:50,30:75].T 
# fondo = np.mean(imagen_fondo)
# imagen_sin_fondo = ida_2 - fondo
# imagen_sin_fondo[imagen_sin_fondo < 0] = 0
# imagen_sin_fondo = imagen_sin_fondo
# graficar(ida_2,2.25,45)
# graficar(ida_2,2.25,45)
# N_fotones = np.sum(imagen_sin_fondo)
# #graficar(imagen_sin_fondo,1,20)

# params_ida, _ = fit_gaussian_2d(ida_2)
# A_i, x0_i, y0_i, sx_i, sy_i, off_i = params_ida

# # Fotones totales (sin fondo)
# N_ida = A_i * 2 * np.pi * sx_i * sy_i

# print("Fotones Ida:", N_ida)


#%%

#%% Analisis para varios frames
from skimage.registration import phase_cross_correlation

from scipy.ndimage import shift
import numpy as np
from itertools import combinations
datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_00_scan.NPY")
img_ida = datos[10][0][50:80,95:125] 
img_vuelta = datos[10][1][50:80,95:125] 
px_size_x = 10/200
px_size_y = px_size_x
# Ajustar primero
params_ida, err_i = fit_gaussian_2d(img_ida.T)
params_vuelta, err_v = fit_gaussian_2d(img_vuelta.T)

A_i, x0_i, y0_i, sx_i, sy_i, _ = params_ida
_, x0_v, y0_v, sx_v, sy_v, _ = params_vuelta
# Fotones totales (sin fondo)
N_ida = A_i * 2 * np.pi * sx_i * sy_i
print(N_ida)
# Shift basado en centros
# dx_pix = y0_i - y0_v
# dy_pix = x0_i - x0_v


shifts, error, diffphase = phase_cross_correlation(
    img_ida.T,
    img_vuelta.T,
    upsample_factor=200
)

dx_pix, dy_pix = shifts
# Alinear vuelta
vuelta_alineada = shift(
    img_vuelta.T,
    shift=(dx_pix, dy_pix),
    mode='constant',
    cval=0,
    
    order=3
)



# suma
ida_vuelta = img_ida.T + vuelta_alineada


params_ida_vuelta,err_u = fit_gaussian_2d(ida_vuelta)



A_i, x0_iv, y0_iv, sx_iv, sy_iv,_ = params_ida_vuelta



# Convertir posiciones a µm
x_i_um = x0_i * px_size_x 
y_i_um = y0_i * px_size_x 
x_v_um = x0_v * px_size_x
y_v_um = y0_v * px_size_y
x_iv_um = x0_iv * px_size_x 
y_iv_um = y0_iv * px_size_y 
print(f"{x_i_um,y_i_um} ida")
print(f"{x_v_um,y_v_um} vuelta")
print(f"{x_iv_um, y_iv_um} union")
conv_x = (10/200)*50
conv_y = (10/200) * 95
print(x_v_um + conv_x)
print(y_v_um + conv_y)
graficar_pos(img_ida.T,1.5,30,x_i_um , y_i_um, conv_x, conv_y )
graficar_pos(img_vuelta.T,1.5,30,x_v_um , y_v_um , conv_x, conv_y)
graficar_pos(vuelta_alineada,1.5,30,x_iv_um, y_iv_um, conv_x, conv_y)
graficar_pos(ida_vuelta,1.5,30, x_iv_um , y_iv_um , conv_x, conv_y)
#%%

#%%


K = 1

datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_00_scan.NPY")

def localizar_combinacion(datos, indices):

    px_size = 10/200  # µm por pixel
    ida_sum = None
    vuelta_sum = None
    for i in indices:

       img_ida = datos[i][0][50:80,95:125] #+ datos[0][0][40:90,85:135] + datos[4][0][40:90,85:135]
       img_vuelta = datos[i][1][50:80,95:125]#+ datos[0][1][40:90,85:135] + datos[4][1][40:90,85:135]

       if ida_sum is None:
           ida_sum = np.array(img_ida, dtype=float)
           vuelta_sum = np.array(img_vuelta, dtype=float)
       else:
           ida_sum += img_ida
           vuelta_sum += img_vuelta
    
    params_ida, _ = fit_gaussian_2d(ida_sum.T)
    params_vuelta, _ = fit_gaussian_2d(vuelta_sum.T)

    _, x0_i, y0_i, _, _, _ = params_ida
    _, x0_v, y0_v, _, _, _ = params_vuelta

    # # Shift basado en centros
    # dx_pix = y0_i - y0_v
    # dy_pix = x0_i - x0_v

    # # --- 
    
    shifts, error, diffphase = phase_cross_correlation(
         img_ida.T,
         img_vuelta.T,
         upsample_factor=2000
     )
     
    dx_pix, dy_pix = shifts
    # Alinear vuelta
    vuelta_alineada = shift(
         img_vuelta.T,
         shift=(dx_pix, dy_pix),
         mode='constant',
         cval=0,
         
         order=3
     )

    

    # ---  Sumar --
    ida_vuelta = ida_sum.T + vuelta_alineada


   
    params_ida_vuelta,_ = fit_gaussian_2d(ida_vuelta)

    _, x0_u, y0_u, _, _, _ = params_ida_vuelta
    
    # --- Convertir a µm ---
    x0_i *= px_size
    y0_i *= px_size
    x0_v *= px_size
    y0_v *= px_size
    x0_u *= px_size
    y0_u *= px_size
    
    return x0_i, y0_i, x0_v, y0_v, x0_u, y0_u
def calcular_localizaciones(datos, max_k=K):

    n_frames = len(datos)
    resultados = {}

    for k in range(1, max_k+1):

        dx_iv = []
        dy_iv = []
        dx_iu = []
        dy_iu = []

        x_i_list = []
        y_i_list = []
        x_v_list = []
        y_v_list = []
        x_u_list = []
        y_u_list = []

        for comb in combinations(range(n_frames), k):

            xi, yi, xv, yv, xu, yu = localizar_combinacion(datos, comb)

            # diferencias
            dx_iv.append(xi - xv)
            dy_iv.append(yi - yv)

            dx_iu.append(xi - xu)
            dy_iu.append(yi - yu)

            # guardar posiciones
            x_i_list.append(xi)
            y_i_list.append(yi)

            x_v_list.append(xv)
            y_v_list.append(yv)

            x_u_list.append(xu)
            y_u_list.append(yu)
            print(comb)
            

        resultados[k] = {
            "dx_iv": np.array(dx_iv),
            "dy_iv": np.array(dy_iv),
            "dx_iu": np.array(dx_iu),
            "dy_iu": np.array(dy_iu),
            "x_i": np.array(x_i_list),
            "y_i": np.array(y_i_list),
            "x_v": np.array(x_v_list),
            "y_v": np.array(y_v_list),
            "x_u": np.array(x_u_list),
            "y_u": np.array(y_u_list),
        }

        print(f"k = {k} | combinaciones = {len(dx_iv)}")

    return resultados


resultados = calcular_localizaciones(datos, max_k=K)


#%%
k = 1
x_i = resultados[k]["x_i"]
y_i = resultados[k]["y_i"]
sig_i = np.sqrt(np.std(x_i)**2 + np.std(y_i)**2)
x_v =  resultados[k]["x_v"]
y_v = resultados[k]["y_v"]
sig_v = np.sqrt(np.std(x_v)**2 + np.std(y_v)**2)
x_u = resultados[k]["x_u"]
y_u = resultados[k]["y_u"]
sig_u = np.sqrt(np.std(x_u)**2 + np.std(y_u)**2)


print("Sigma ida:", sig_i)
print("Sigma vuelta:", sig_v)
print("Sigma union:", sig_u)
print("Ratio:",sig_i / sig_u)

plt.figure(constrained_layout=True)

plt.scatter(x_i, y_i, c="darkred", alpha=0.5, label="Ida")
plt.scatter(x_u, y_u, c="olivedrab", alpha=0.5, label="Unión")
plt.scatter(x_v, y_v, c="darkblue", alpha=0.5, label="vuelta")
plt.xlabel("$ x [um ]$", fontsize=16)
plt.ylabel("$ y [um ]$", fontsize=16)
# plt.xlim(-0.05*1E3, 0.25*1E3)
# plt.ylim(-0.05*1E3, 0.05*1E3)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize = 14)
plt.grid(True)
plt.show()
#%%

#%%
k = 1
dx_iv = resultados[k]["dx_iv"]*1E3
dy_iv = resultados[k]["dy_iv"]*1E3

dx_iu = resultados[k]["dx_iu"]*1E3
dy_iu = resultados[k]["dy_iu"]*1E3

plt.figure(constrained_layout=True)

plt.scatter(dx_iv, dy_iv, c="darkred", alpha=0.6, label="Ida - Vuelta")
plt.scatter(dx_iu, dy_iu, c="olivedrab", alpha=0.6, label="Ida - Unión")



plt.xlabel("$\Delta x [nm ]$", fontsize=16)
plt.ylabel("$\Delta y [nm ]$", fontsize=16)
plt.xlim(-0.05*1E3, 0.25*1E3)
plt.ylim(-0.05*1E3, 0.05*1E3)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize = 14)
plt.grid(True)
plt.show()
#%%
plt.figure(constrained_layout=True)

plt.hist(dx_iv, bins=18, alpha=0.7, label="Ida - Vuelta", color="darkred")
plt.hist(dx_iu, bins=18, alpha=0.7, label="Ida - Unión", color="olivedrab")

plt.xlabel("Δx [nm]", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)

plt.legend(fontsize = 14, loc = "upper center")
plt.grid(True)
plt.show()


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
fwhm_iv = calcular_fwhm(dx_iv, bins=18)
fwhm_iu = calcular_fwhm(dx_iu, bins=18)

print("Promedio ida - vuelta:", np.mean(dx_iv), "nm")
print("FWHM Ida-vuelta:", fwhm_iv, "nm")

print("Promedio ida - unión:", np.mean(dx_iu), "nm")
print("FWHM Ida-union:", fwhm_iu, "nm")
#%%


from scipy.ndimage import shift
import numpy as np
n_ida = []
n_union = []
def analizar_frames_ida_vuelta(datos):

    px_size = 10/200  # µm por pixel
    n_frames = len(datos)

    x_i_list = []
    y_i_list = []
    x_v_list = []
    y_v_list = []
    x_u_list = []
    y_u_list = []

    roi = (slice(50, 80), slice(95, 125))
    for i in range(n_frames - 6):

        # ROI ---

        ida_sum = (
            datos[i][0][roi].astype(float) +
            datos[54][0][roi].astype(float)+
            datos[55][0][roi].astype(float) + 
            datos[53][0][roi].astype(float) + 
            datos[52][0][roi].astype(float))
        
    
        vuelta_sum = (datos[i][1][roi].astype(float) +
                      datos[54][1][roi].astype(float)+
                      datos[55][1][roi].astype(float)+
                      datos[53][1][roi].astype(float)+ 
                      datos[52][0][roi].astype(float))
        
        # --Ajuste gaussiano ---
        params_ida, _ = fit_gaussian_2d(ida_sum.T)
        params_vuelta, _ = fit_gaussian_2d(vuelta_sum.T)
   
        A_i, x0_i, y0_i, sx_i, sy_i, _ = params_ida
        _, x0_v, y0_v, _, _, _ = params_vuelta
        N_ida = A_i * 2 * np.pi * sx_i * sy_i
        n_ida.append(N_ida)
        print(N_ida)
   
     
        
        shifts, error, diffphase = phase_cross_correlation(
            img_ida.T,
            img_vuelta.T,
            upsample_factor=200
        )
        
        dx_pix, dy_pix = shifts
        # Alinear vuelta
        vuelta_alineada = shift(
            img_vuelta.T,
            shift=(dx_pix, dy_pix),
            mode='constant',
            cval=0,
            
            order=2
        )
   
        # --- Sumar ---
        ida_vuelta = ida_sum.T + vuelta_alineada
     
        params_ida_vuelta,_ = fit_gaussian_2d(ida_vuelta)
   
        A_u, x0_u, y0_u, sx_u, sy_u, off = params_ida_vuelta
        N_union = A_u * 2 * np.pi * sx_u * sy_u
        print(N_union)
        n_union.append(N_union)
        # ---Convertir a µm ---
        x0_i *= px_size
        y0_i *= px_size
        x0_v *= px_size
        y0_v *= px_size
        x0_u *= px_size
        y0_u *= px_size
        x_i_list.append(x0_i)
        y_i_list.append(y0_i)

        x_v_list.append(x0_v)
        y_v_list.append(y0_v)

        x_u_list.append(x0_u)
        y_u_list.append(y0_u)

    resultados = {
        "x_i": np.array(x_i_list),
        "y_i": np.array(y_i_list),
        "x_v": np.array(x_v_list),
        "y_v": np.array(y_v_list),
        "x_u": np.array(x_u_list),
        "y_u": np.array(y_u_list),
        "dx_iv": np.array(x_i_list) - np.array(x_v_list),
        "dy_iv": np.array(y_i_list) - np.array(y_v_list),
        "dx_iu": np.array(x_i_list) - np.array(x_u_list),
        "dy_iu": np.array(y_i_list) - np.array(y_u_list),
    }

    return resultados
# print("Desvío ida en x:", np.std(resultados["x_i"]))
# print("Desvío ida en y:", np.std(resultados["y_i"]))

# print("Desvío vuelta en x:", np.std(resultados["x_v"]))
# print("Desvío vuelta en y:", np.std(resultados["y_v"]))

# print("Desvío unión en x:", np.std(resultados["x_u"]))
# print("Desvío unión en y:", np.std(resultados["y_u"]))
resultados = analizar_frames_ida_vuelta(datos)

x_i = resultados["x_i"]*1E3
y_i = resultados["y_i"]*1E3
sig_i = np.sqrt(np.std(x_i)**2 + np.std(y_i)**2)
x_u = resultados["x_u"]*1E3
y_u = resultados["y_u"]*1E3
sig_u = np.sqrt(np.std(x_u)**2 + np.std(y_u)**2)
x_v =  resultados["x_v"]*1E3
y_v = resultados["y_v"]*1E3


print("Sigma ida:", sig_i)
print("Sigma union:", sig_u)
print("Ratio:",sig_i / sig_u)

plt.figure(constrained_layout=True)

plt.scatter(x_i, y_i, c="darkred", alpha=0.5, label="Ida")
plt.scatter(x_u, y_u, c="olivedrab", alpha=0.5, label="Unión")
plt.scatter(x_v, y_v, c="darkblue", alpha=0.5, label="vuelta")
plt.xlabel("$ x [nm ]$", fontsize=16)
plt.ylabel("$ y [nm ]$", fontsize=16)
plt.xlim(0.53*1E3, 0.81*1E3)
plt.ylim(0.5*1E3, 0.7*1E3)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.legend(fontsize = 14, loc = "upper center")
plt.grid(True)
plt.show()
#%%
plt.figure(constrained_layout=True)

plt.hist(x_i, bins=18, alpha=0.7, label="Ida", color="darkred")
plt.hist(x_u, bins=18, alpha=0.7, label="Unión", color="olivedrab")

plt.xlabel("Δx [nm]", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)

plt.legend(fontsize = 14, loc = "upper center")
plt.grid(True)
plt.show()

sep_iv = np.mean(x_i) - np.mean(x_v)
err = np.std(x_i)