# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:55:05 2025

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

path = r"C:\Users\Lenovo\Downloads\Calibracion_ida_vuelta\matplotvanda.py"

sys.path.append(path)
dwell_time_10 = [18,29,39,59,69,79,90,100,500]
dwell_time_5 = [19,29,39,49,59,70,79,90,100,501]
datos_10 = []
datos_5 = []
for i in range(9):
   datos_10.append(sd.ScanDataFile.open(rf"C:\Users\Lenovo\Downloads\Calibracion_ida_vuelta\10x10\calibracion_10x10_0{i}_scan.NPY"))
for j in range(9):   
    datos_5.append(sd.ScanDataFile.open(rf"C:\Users\Lenovo\Downloads\Calibracion_ida_vuelta\5x5\calibracion_5x5_0{j}_scan.NPY"))

datos_5.append(sd.ScanDataFile.open(rf"C:\Users\Lenovo\Downloads\Calibracion_ida_vuelta\5x5\calibracion_5x5_0C_scan.NPY"))
#%%
def graficar(imagen, pixel_size_um=1/20, titulo="(Vuelta (soft))"):
    """
    Muestra la imagen directamente, calculando los ejes físicos automáticamente.
    
    Parámetros
    ----------
    imagen : 2D array
        Matriz de intensidades (e.g., número de fotones por píxel).
    pixel_size_um : float, opcional
        Tamaño físico de cada píxel en µm (por defecto 1 µm/píxel).
    titulo : str, opcional
        Título que se muestra en la figura.
    """
    # Calcular ejes físicos
    imagen = imagen.T
    nx, ny = imagen.shape
    x_extent = nx * pixel_size_um
    y_extent = ny * pixel_size_um

    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='turbo',
                   extent=[0, x_extent, 0, y_extent],
                   origin='lower',
                   aspect='equal')

    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.show()


#generar el perfil de intensidad con la ida y la vuelta.
def guardar_imagen_tiff(file, imagen_ida, imagen_vuelta):
    tiff.imwrite( path + file + "_ida.tif", imagen_ida.astype(np.float32))
    tiff.imwrite( path + file + "_vuelta.tif", np.flip(imagen_vuelta, axis = 1).astype(np.float32))

    
    
p = sd.ScanDataFile.open(r"C:\Users\Lenovo\Downloads\Calibracion_ida_vuelta\10x10\calibracion_10x10_00_scan.NPY")

corr = 0.00
def compare_profiles(dist_ida, prof_ida, dist_vuelta, prof_vuelta, dwell_time=None):
    """Compara los perfiles ida/vuelta, calcula desplazamiento y FWHM, y lo grafica."""
    
    # Picos
    peak_ida = dist_ida[np.argmax(prof_ida)]
    peak_vuelta = dist_vuelta[np.argmax(prof_vuelta)]
    shift_um = peak_ida - peak_vuelta

    # FWHM e incertidumbre
    fwhm_ida = fwhm(dist_ida, prof_ida)
    fwhm_vuelta = fwhm(dist_vuelta, prof_vuelta)
    err_um = 0.5 * np.sqrt(fwhm_ida**2 + fwhm_vuelta**2)

    # Gráfico comparativo
    plt.figure()
    plt.plot(dist_ida , prof_ida, label= "Ida", lw=2, color="r")
    plt.plot(dist_vuelta , prof_vuelta, label="Vuelta", lw=2, color="b")

    # Marcar los picos
    plt.axvline(peak_ida + corr  , color='r', ls='--', alpha=0.3)
    plt.axvline(peak_vuelta   , color='b', ls='--', alpha=0.3)

    # Marcar FWHM visualmente
    plt.axhline(np.max(prof_ida)/2, color='r', ls=':', alpha=0.6)
    plt.axhline(np.max(prof_vuelta)/2, color='b', ls=':', alpha=0.6)

    plt.xlabel("Distancia [µm]")
    plt.ylabel("Intensidad (a.u.)")
    plt.title("Comparación de perfiles (ida vs vuelta)")
    plt.legend(loc = "center right")
    plt.grid(True)

    # Texto con desplazamiento y error
    plt.text(0.05, 0.95,
             f"Desplazamiento: {shift_um:.3f} ± {err_um:.3f} µm\n"
             f"FWHM ida: {fwhm_ida:.3f} µm, vuelta: {fwhm_vuelta:.3f} µm\n"
             f"Dwell time: {dwell_time} us",
             transform=plt.gca().transAxes,
             va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.show()

    print(f"Desplazamiento entre picos: {(shift_um + corr)} ± {err_um:.3f} µm")

    return shift_um, err_um, fwhm_ida, fwhm_vuelta

# graficar(p[0][0], 0.050)
# graficar(p[0][1], 0.050)
#%%
def plot_ida_vuelta(im_ida, im_vuelta, pixel_size_um ,dwell_time,
                    titulo_ida="Ida", titulo_vuelta="Vuelta",
                    flip_vuelta_axis=1, cmap='inferno', vmax=None ):
    """
    Muestra: (A) imagen ida, (B) imagen vuelta (volteada si es necesario),
    (C) overlay (ida en rojo, vuelta en azul) y (D) diferencia con contorno.
    Devuelve las imágenes usadas (ida, vuelta_flip, diff).
    """
    # Asegurarse de trabajar con floats
    ida = np.array(im_ida, dtype=float).T
    vuelta = np.array(im_vuelta, dtype=float).T

    # extent físico
    ny, nx = ida.shape
    x_extent = nx * pixel_size_um
    y_extent = ny * pixel_size_um
    extent = [0, x_extent, 0, y_extent]

    # normalizar color scale si pide
    if vmax is None:
        vmax = max(np.nanmax(ida), np.nanmax(vuelta))

    fig, axs = plt.subplots(1, 4, figsize=(18,4), constrained_layout=True)

    axs[0].imshow(ida, origin='lower', extent=extent, aspect='equal', cmap=cmap, vmax=vmax)
    axs[0].set_title(titulo_ida); axs[0].set_xlabel("x [µm]"); axs[0].set_ylabel("y [µm]")

    axs[1].imshow(vuelta, origin='lower', extent=extent, aspect='equal', cmap=cmap, vmax=vmax)
    axs[1].set_title(titulo_vuelta); axs[1].set_xlabel("x [µm]"); axs[1].set_ylabel("y [µm]")

    # overlay: roja = ida, azul = vuelta (convertir a RGB por canales simples)
    # escalamos a [0,1] para visual overlay
    def _norm(img):
        m = np.nanmax(img)
        if m == 0 or np.isnan(m):
            return np.zeros_like(img)
        return img / m

    ida_n = _norm(ida)
    vuelta_n = _norm(vuelta)
    overlay = np.stack([ida_n, np.zeros_like(ida_n), vuelta_n], axis=-1)  # R=ida, B=vuelta
    tiff.imwrite("ida_vuelta_10_18.tif", overlay.astype(np.float32))

    axs[2].imshow(overlay, origin='lower', extent=extent, aspect='equal')
    axs[2].set_title("R: ida, A: vuelta"); axs[2].set_xlabel("x [µm]")
    axs[2].plot([], [], color='none', label=f"{dwell_time} µs")  # handle vacío con texto
    axs[2].legend()

    # diferencia y contorno
    diff = ida - vuelta
    im = axs[3].imshow(diff, origin='lower', extent=extent, aspect='equal', cmap='bwr')
    axs[3].contour(diff, levels=1, linewidths=1, origin='lower', extent=extent)
    axs[3].set_title("Diferencia (ida - vuelta)"); axs[3].set_xlabel("x [µm]")
    fig.colorbar(im, ax=axs[3], label='Intensidad (arb.)')

    plt.show()
    return ida, vuelta, diff

def plot_soft_life(im_ida, im_vuelta, pixel_size_um ,dwell_time,
                    titulo_ida="Software", titulo_vuelta="Lifetime",
                    flip_vuelta_axis=1, cmap='inferno', vmax=None ):
    """
    Muestra: (A) imagen software, (B) imagen lifetime""",
   
    # Asegurarse de trabajar con floats
    ida = np.array(im_ida, dtype=float).T
    vuelta = np.array(im_vuelta, dtype=float).T

    # extent físico
    ny, nx = ida.shape
    x_extent = nx * pixel_size_um
    y_extent = ny * pixel_size_um
    extent = [0, x_extent, 0, y_extent]

    # normalizar color scale si pide
    if vmax is None:
        vmax = max(np.nanmax(ida), np.nanmax(vuelta))

    fig, axs = plt.subplots(1, 4, figsize=(18,4), constrained_layout=True)

    axs[0].imshow(ida, origin='lower', extent=extent, aspect='equal', cmap=cmap, vmax=vmax)
    axs[0].set_title(titulo_ida); axs[0].set_xlabel("x [µm]"); axs[0].set_ylabel("y [µm]")

    axs[1].imshow(vuelta, origin='lower', extent=extent, aspect='equal', cmap=cmap, vmax=vmax)
    axs[1].set_title(titulo_vuelta); axs[1].set_xlabel("x [µm]"); axs[1].set_ylabel("y [µm]")

    # overlay: roja = ida, azul = vuelta (convertir a RGB por canales simples)
    # escalamos a [0,1] para visual overlay
    def _norm(img):
        m = np.nanmax(img)
        if m == 0 or np.isnan(m):
            return np.zeros_like(img)
        return img / m

    ida_n = _norm(ida)
    vuelta_n = _norm(vuelta)
    overlay = np.stack([ida_n, np.zeros_like(ida_n), vuelta_n], axis=-1)  # R=ida, B=vuelta

    axs[2].imshow(overlay, origin='lower', extent=extent, aspect='equal')
    axs[2].set_title("R: Software, A: Lifetime"); axs[2].set_xlabel("x [µm]")
    axs[2].plot([], [], color='none', label=f"{dwell_time} µs")  # handle vacío con texto
    axs[2].legend()

     # diferencia y contorno
    diff = ida - vuelta
    im = axs[3].imshow(diff, origin='lower', extent=extent, aspect='equal', cmap='bwr')
    axs[3].contour(diff, levels=1, linewidths=1, origin='lower', extent=extent)
    axs[3].set_title("Diferencia (soft - life)"); axs[3].set_xlabel("x [µm]")
    fig.colorbar(im, ax=axs[3], label='Intensidad (arb.)')

    plt.show()
    return ida, vuelta, diff

def fwhm(x, y):
    """Calcula el ancho a mitad de altura (FWHM) de un pico."""
    half_max = np.max(y) / 2.0
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return np.nan
    return x[indices[-1]] - x[indices[0]]


def line_profile(img, x0, y0, x1, y1, pixel_size_um=0.05, npoints=None, cmap='inferno', title=''):
    """Devuelve el perfil de intensidad a lo largo de una línea."""
    ny, nx = img.shape
    extent = [0, nx * pixel_size_um, 0, ny * pixel_size_um]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(img, cmap='inferno', origin='lower', extent=extent, aspect='equal')
    ax.plot([x0 * pixel_size_um, x1 * pixel_size_um],
            [y0 * pixel_size_um, y1 * pixel_size_um], 'r--', lw=1.5)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_title("Línea de perfil (Ida)")
    plt.tight_layout()
    plt.show()


    prof = profile_line(img, (y0, x0), (y1, x1), mode='reflect', linewidth=1, order=1)
    if len(prof) != npoints:
        prof = np.interp(np.linspace(0, len(prof)-1, npoints),
                         np.arange(len(prof)), prof)
    length_pix = np.hypot(x1 - x0, y1 - y0)
    dist_um = np.linspace(0, length_pix * pixel_size_um, npoints)

    plt.figure()
    plt.plot(dist_um, prof)
    plt.xlabel("Distancia [µm]")
    plt.ylabel("Intensidad (a.u.)")
    plt.title(f"Perfil de línea {title}")
    plt.grid(True)
    plt.show()

    return dist_um, prof


def get_line_profiles(ida_img, vuelta_img, x0, y0, x1, y1, pixel_size_um=0.05, npoints=200):
    """Obtiene los perfiles de ida y vuelta."""
    dist_ida, prof_ida = line_profile(ida_img.T, x0, y0, x1, y1, pixel_size_um, npoints, title='(Ida)')
    dist_vuelta, prof_vuelta = line_profile(vuelta_img.T, x0, y0, x1, y1, pixel_size_um, npoints, title='(Vuelta)')
    return dist_ida, prof_ida, dist_vuelta, prof_vuelta


def compare_profiles(dist_ida, prof_ida, dist_vuelta, prof_vuelta, dwell_time=None):
    """Compara los perfiles ida/vuelta, calcula desplazamiento y FWHM, y lo grafica."""
    
    # Picos
    peak_ida = dist_ida[np.argmax(prof_ida)]
    peak_vuelta = dist_vuelta[np.argmax(prof_vuelta)]
    shift_um = peak_ida - peak_vuelta

    # FWHM e incertidumbre
    fwhm_ida = fwhm(dist_ida, prof_ida)
    fwhm_vuelta = fwhm(dist_vuelta, prof_vuelta)
    err_um = 0.5 * np.sqrt(fwhm_ida**2 + fwhm_vuelta**2)

    # Gráfico comparativo
    plt.figure()
    plt.plot(dist_ida , prof_ida, label= "Ida", lw=2, color="r")
    plt.plot(dist_vuelta , prof_vuelta, label="Vuelta", lw=2, color="b")

    # Marcar los picos
    plt.axvline(peak_ida - 0.07 , color='r', ls='--', alpha=0.3)
    plt.axvline(peak_vuelta  , color='b', ls='--', alpha=0.3)

    # Marcar FWHM visualmente
    plt.axhline(np.max(prof_ida)/2, color='r', ls=':', alpha=0.6)
    plt.axhline(np.max(prof_vuelta)/2, color='b', ls=':', alpha=0.6)

    plt.xlabel("Distancia [µm]")
    plt.ylabel("Intensidad (a.u.)")
    plt.title("Comparación de perfiles (ida vs vuelta)")
    plt.legend(loc = "center right")
    plt.grid(True)

    # Texto con desplazamiento y error
    plt.text(0.05, 0.95,
             f"Desplazamiento: {shift_um:.3f} ± {err_um:.3f} µm\n"
             f"FWHM ida: {fwhm_ida:.3f} µm, vuelta: {fwhm_vuelta:.3f} µm\n"
             f"Dwell time: {dwell_time} us",
             transform=plt.gca().transAxes,
             va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.show()

    print(f"Desplazamiento entre picos: {shift_um:.3f} ± {err_um:.3f} µm")

    return shift_um, err_um, fwhm_ida, fwhm_vuelta

pixel_size = 0.050

x0_10, y0_10 = 50,65
x1_10, y1_10 = 100, 65


for i in range(len(datos_10)):
    ida_img = datos_10[i][0][0]
    vuelta_img = datos_10[i][0][1]
    ida, vuelta, diff = plot_ida_vuelta(ida_img, vuelta_img,pixel_size, dwell_time_10[i])
    dist_ida, prof_ida, dist_vuelta, prof_vuelta = get_line_profiles(ida_img, vuelta_img, x0_10, y0_10, x1_10, y1_10)
    compare_profiles(dist_ida, prof_ida, dist_vuelta, prof_vuelta, dwell_time_10[i])
# #%%
# for j in range(len(datos_5)):
#     ida_img = datos_5[j][0][0]
#     vuelta_img = datos_5[j][0][1]
#     #ida, vuelta, diff = plot_ida_vuelta(ida_img, vuelta_img, pixel_size_um=pixel_size)
#     dist_ida, prof_ida, dist_vuelta, prof_vuelta = get_line_profiles(ida_img, vuelta_img, x0_5, y0_5, x1_5, y1_5)
#     compare_profiles(dist_ida, prof_ida, dist_vuelta, prof_vuelta, dwell_time_5[i])
c = 0
ida_img = datos_10[c][0][0]
vuelta_img = datos_10[c][0][1]
dist_ida, prof_ida, dist_vuelta, prof_vuelta = get_line_profiles(ida_img, vuelta_img, x0_10, y0_10, x1_10, y1_10)
compare_profiles(dist_ida , prof_ida, dist_vuelta, prof_vuelta, dwell_time_10[c])

# c = 9
# x0_5, y0_5 = 30, 42.5
# x1_5, y1_5 = 100, 42.5
# ida_img = datos_5[c][0][0]
# vuelta_img = datos_5[c][0][1]
# dist_ida, prof_ida, dist_vuelta, prof_vuelta = get_line_profiles(ida_img, vuelta_img, x0_5, y0_5, x1_5, y1_5)
#%%

datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")
ida_img = datos[15][0]
vuelta_img = datos[15][1]
plot_ida_vuelta(ida_img, vuelta_img, 2/50, 400)



























# compare_profiles(dist_ida , prof_ida, dist_vuelta, prof_vuelta, dwell_time_5[c])
#%% Hagamos un promedio con las tres nanoparticulas que aparecen en 10
distancia_10_1 = np.array([0.585,0.318,0.318,0.233, 0.231,0.234, 0.233, 0.160, 0.158])
distancia_10_2 = np.array([0.490,0.359,0.302,0.262, 0.239,0.214, 0.153, 0.181, 0.158])
distancia_10_3 = np.array([0.510,0.315,0.289,0.254,0.201,0.191, 0.219, 0.200, 0.150])


distancia_10 = (distancia_10_1 + distancia_10_2 + distancia_10_3 ) / 3
distancia_10*=1000




#%% 

px_size_v = np.ones_like(dwell_time_10)*pixel_size
v_10 = px_size_v /dwell_time_10

fig, ax = plt.subplots()
ax.scatter(v_10,distancia_10)
ax.set_ylabel("Distancia [nm]")
ax.set_xlabel("Velocidad[um/us]")
vd.gula_grid(ax)
ax.legend()
# err_10 = np.array([0.22, 0.22]) 



#%%
distancia_5 = np.array([0.501, 0.356, 0.334,0.226 ,0.230,0.185,0.146, 0.126,0.130, 0.120])
distancia_5*=1000
px_size_v = np.ones_like(dwell_time_5)*pixel_size
v_5 = px_size_v /dwell_time_5

fig, ax = plt.subplots()
ax.scatter(v_5,distancia_5)
ax.set_ylabel("Distancia [nm]")
ax.set_xlabel("Velocidad[um/us]")
vd.gula_grid(ax)
ax.legend()

#%%
# Ajuste cuadrático 10
coef = np.polyfit(v_10, distancia_10, deg=2)  # a, b, c
fit_func = np.poly1d(coef)

# Valores para graficar la curva ajustada
v_fit = np.linspace(np.min(v_10), np.max(v_10), 200)
dist_fit = fit_func(v_fit)

# Gráfico
fig, ax = plt.subplots(constrained_layout=True)
ax.scatter(v_10, distancia_10,color = "green", label = "Datos 10x10 µm")
ax.plot(v_fit, dist_fit, color = "grey", label=f"Ajuste: {coef[0]:.0f} v² + {coef[1]:.0f} v + {coef[2]:.0f}",linestyle = "--", linewidth = 2)
ax.set_xlabel("Velocidad [µm/µs]")
ax.set_ylabel("Distancia [nm]")
vd.gula_grid(ax)
ax.legend(loc = "upper left")
plt.show()

# Ajuste cuadrático 5
coef = np.polyfit(v_5, distancia_5, deg=2)  # a, b, c
fit_func = np.poly1d(coef)

# Valores para graficar la curva ajustada
v_fit_5 = np.linspace(np.min(v_5), np.max(v_5), 100)
dist_fit_5 = fit_func(v_fit_5)

# Gráfico
fig, ax = plt.subplots(constrained_layout=True)
ax.scatter(v_5, distancia_5,color = "green", label = "Datos 5x5 µm")
ax.plot(v_fit_5, dist_fit_5, color = "grey", label=f"Ajuste: {coef[0]:.0f} v² + {coef[1]:.0f} v + {coef[2]:.0f}",linestyle = "--", linewidth = 2)
ax.set_xlabel("Velocidad [µm/µs]")
ax.set_ylabel("Distancia [nm]")
vd.gula_grid(ax)
ax.legend(loc = "upper left")
plt.show()
#Todas las curvas juntas


#%%
fig, ax = plt.subplots(constrained_layout=True)

ax.plot(v_fit, dist_fit, color = "gold", label="Curva de calibración 10x10 (software)", linestyle = "--", linewidth = 2)
ax.plot(v_fit_5, dist_fit_5, color = "lightcoral",label="Curva de calibración 5x5 (software)", linestyle = "--", linewidth = 2)
ax.set_xlabel("Velocidad [µm/µs]")
ax.set_ylabel("Distancia [nm]")
vd.gula_grid(ax)
ax.legend(loc = "upper left")
plt.show()




#%% Analicemos las  imagenes por lifetime
# x0_10, y0_10 = 50,65
# x1_10, y1_10 = 100, 65
x0_10, y0_10 = 40,180
x1_10, y1_10 = 90, 180
imagen_ida_10 = []
imagen_vuelta_10 = []
number_of_pixels = 200
image_size_um = 10
pixeles_ida_al_cero = 2
for i in range(len(dwell_time_10)):
    file = f"Calibracion_ida_vuelta\\10x10\\10x10um{dwell_time_10[i]}us"
    a, b, imagen_ida,imagen_vuelta = al.imagen_ida_vuelta(file, number_of_pixels,
    image_size_um, pixeles_ida_al_cero)
    imagen_ida_10.append(imagen_ida)
    imagen_vuelta_10.append(imagen_vuelta)
#%%
x0_10, y0_10 = 140,100
x1_10, y1_10 = 180, 100
c = 8
dist_ida, prof_ida, dist_vuelta, prof_vuelta = get_line_profiles(imagen_ida_10[c].T, np.flip(imagen_vuelta_10[c], axis = 1).T, x0_10, y0_10, x1_10, y1_10)
compare_profiles(dist_ida , prof_ida, dist_vuelta, prof_vuelta, dwell_time_10[c])
distancia_10_1l = np.array([0.502,0.400,0.320,0.263,0.220, 0.261,0.261, 0.205,0.150])
distancia_10_2l = np.array([0.63,0.440,0.31,0.251,0.251,0.260,0.214,0.220,0.123])
distancia_10_3l = np.array([])


#%%

# x0_5, y0_5 = 30, 42.5
# x1_5, y1_5 = 100, 42.5

x, y, imagen_ida, imagen_vuelta = al.imagen_ida_vuelta(file, number_of_pixels,
image_size_um, pixeles_ida_al_cero)
al.graficar_ida(x,y,imagen_ida)
al.graficar_vuelta(x,y,imagen_vuelta)
# guardar_imagen_tiff(file, imagen_ida, imagen_vuelta)
dist_ida, prof_ida, dist_vuelta, prof_vuelta = get_line_profiles(imagen_ida.T, np.flip(imagen_vuelta, axis = 1).T, x0_10, y0_10, x1_10, y1_10)
compare_profiles(dist_ida , prof_ida, dist_vuelta, prof_vuelta, dwell_time_10[0])
distancia_10_l = np.array([0.502,])



#%%


    