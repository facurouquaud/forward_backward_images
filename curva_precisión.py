# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:01:27 2026

@author: Luis1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar datos ---
#data = np.load(r"C:\Users\Luis1\Downloads\Calibracion_ida_vuelta\cam_CRB_curve.npy")
data = np.load(r"C:\Users\Luis1\Downloads\Calibracion_ida_vuelta\cam_CRB_curve_camaraSBRreal.npy")
sig = pd.read_excel(r"C:\Users\Luis1\Downloads\sigma_exp.xlsx")

sigma = sig["sigma custom fit"]
N_exp = sig["ph tot"] 
def sigma_e(N, sigma):
    return sigma / np.sqrt(N)
N = np.linspace(300,3500,1000)
sigma = sigma_e(N, 150)
import numpy as np

def sigma_pixel_model(N, s, a, b):
    
    term1 = (s**2 + a**2/12) / N
    # term2 = (8*np.pi*s**4 * b**2) / (a**2 * N**2)
    
    sigma = np.sqrt(term1) #+ term2)
    
    return sigma

s = 150  # nm (sigma PSF)
a = 50 # nm pixel
b = 1# background photons/pixel

sigma = sigma_pixel_model(N, s, a, b)
sigma_i= 14.8
N_i = 542

sigma_u =  8
N_u = 1125

sigma_2i =  8
N_2i = 998

sigma_2u = 5.5
N_2u = 1522

sigma_3i = 5.3
N_3i = 1442

sigma_3u = 4.1
N_3u = 2098

sigma_4i = 3.98
N_4i = 1948

sigma_4u = 3.28
N_4u = 2517

sigma_5i = 3.15
N_5i = 2488

sigma_5u = 2.7
N_5u = 3057

N_us = np.array( [N_u, N_2u, N_3u, N_4u])
N_is = np.array( [N_i,N_2i, N_3i, N_4i])
sigma_is = np.array( [sigma_i, sigma_2i, sigma_3i, sigma_4i])
sigma_us = np.array( [sigma_u, sigma_2u, sigma_3u, sigma_4u])

N_exp = sig["ph tot"]

sigma_instr = 1 # nm

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

markers = ['o', 's', '^', 'D', 'P', 'X']

fig, ax = plt.subplots(figsize=(7,6))

# --- Curva CRB ---
ax.loglog(
    N,
    sigma,
    color="slategray",
    linewidth=2.5,
    label="CRB"
)

# --- Scatter ida / unión ---
for i in range(len(N_is)):

    ax.scatter(
        N_is[i], sigma_is[i],
        s=35,
        color="firebrick",
        edgecolor="black",
        marker=markers[i],
        zorder=5,
        label="Ida" if i == 0 else None
    )

    ax.scatter(
        N_us[i], sigma_us[i],
        s=35,
        color="olivedrab",
        edgecolor="black",
        marker=markers[i],
        zorder=5,
        label="Unión" if i == 0 else None
    )

# --- Escalas ---
ax.set_xscale("log")
ax.set_yscale("log")

# --- Ticks prolijos ---
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_locator(LogLocator(base=10))

ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,5]))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2,3,5]))

ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

# --- Labels ---
ax.set_xlabel("Número de fotones", fontsize=15)
ax.set_ylabel("Precisión de localización σ (nm)", fontsize=15)

# --- Estética ---
ax.tick_params(axis='both', which='major', labelsize=15, length=6)
ax.tick_params(axis='both', which='minor', length=3)

ax.grid(which="major", linestyle="--", alpha=0.4)

ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
#%%


