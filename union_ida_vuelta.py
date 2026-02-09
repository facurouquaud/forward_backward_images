# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:36:11 2026

@author: Luis1
"""

from scipy.signal import correlate
import matplotlib.pyplot as plt
import numpy as np
import sys
import tifffile as tiff
from PIL import Image
from scipy.signal import find_peaks
import scan_datafile as sd
from skimage.measure import profile_line
import Analisis_lifetime as al
from scipy.signal import correlate

datos = sd.ScanDataFile.open(r"C:\Users\Luis1\Downloads\medicion_idavuelta\scan_07_scan.NPY")

img_fwd = datos[0][0].T
img_bwd = datos[0][1].T

profile_fwd = img_fwd.mean(axis=0)
profile_bwd = img_bwd.mean(axis=0)

corr = correlate(profile_fwd, profile_bwd, mode="full")
shift = corr.argmax() - (len(profile_fwd) - 1)
img_bwd_aligned = np.roll(img_bwd, shift, axis=1)
img_combined = 0.5 * (img_fwd + img_bwd_aligned)
img_combined = np.median(
    np.stack([img_fwd, img_bwd_aligned]),
    axis=0
)
fig, ax = plt.subplots(1,3, figsize=(12,4))

ax[0].imshow(img_fwd)
ax[0].set_title("Ida")

ax[1].imshow(img_bwd)
ax[1].set_title("Vuelta espejada")

ax[2].imshow(img_fwd - img_bwd_aligned)
ax[2].set_title("Diferencia")
#%%


def align_bidirectional(img_fwd, img_bwd):
    """
    Alinea imagen forward y backward de un escaneo bidireccional.

    Parámetros
    ----------
    img_fwd : ndarray (Ny, Nx)
        Imagen de ida
    img_bwd : ndarray (Ny, Nx)
        Imagen de vuelta (sin espejar)

    Retorna
    -------
    img_bwd_aligned : ndarray
        Imagen de vuelta espejada y alineada
    shift : int
        Corrimiento aplicado en píxeles
    """

    # 1) Espejar la vuelta en el eje rápido (X)
    img_bwd_flip = img_bwd

    # 2) Perfiles promedio en Y
    prof_fwd = img_fwd.mean(axis=0)
    prof_bwd = img_bwd.mean(axis=0)

    # 3) Correlación cruzada
    corr = correlate(prof_fwd, prof_bwd, mode="full")
    shift = corr.argmax() - (len(prof_fwd) - 1)

    # 4) Aplicar corrimiento
    img_bwd_aligned = np.roll(img_bwd_flip, shift, axis=1)

    return img_bwd_aligned, shift

# Ejemplo con tu estructura
img_fwd = datos[2][0].T
img_bwd = datos[2][1].T

img_bwd_aligned, shift = align_bidirectional(img_fwd, img_bwd)

print(f"Shift encontrado: {shift} píxeles")

# Imagen combinada
img_combined = 0.5 * (img_fwd + img_bwd_aligned)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

ax[0].imshow(img_fwd)
ax[0].set_title("Ida")

ax[1].imshow(img_bwd)
ax[1].set_title("Vuelta espejada")

ax[2].imshow(img_bwd_aligned)
ax[2].set_title("Vuelta alineada")

ax[3].imshow(img_fwd - img_bwd_aligned)
ax[3].set_title("Diferencia")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(5,4))
plt.imshow(img_fwd - img_bwd_aligned, cmap="gray")
plt.title("Diferencia (ida - vuelta alineada)")
plt.colorbar()
plt.show()
