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
data = np.load(r"C:\Users\Luis1\Downloads\Calibracion_ida_vuelta\cam_CRB_curve.npy")
sig = pd.read_excel(r"C:\Users\Luis1\Downloads\sigma_exp.xlsx")

sigma = sig["sigma custom fit"]
N_exp = sig["ph tot"] 
def sigma_e(N, sigma):
    sigma / np.sqrt(N)

    
sigma_i= 14.71
N_i = 500

sigma_u =  7.96
N_u = 1060

sigma_2i =  7.4
N_2i = 1082

sigma_2u = 5.1
N_2u = 1680

sigma_3i = 4.96
N_3i = 1605

plt.figure(figsize=(7,6))

N_exp = sig["ph tot"]

sigma_instr = 1 # nm

# CRB degradada por ruido instrumental
sigma_sim_err_1 = np.sqrt(data[1]**2 + sigma_instr**2)
sigma_sim_err_2 = np.sqrt(data[1]**2 - sigma_instr**2)

plt.figure(figsize=(7,6))

# CRB ideal
plt.loglog(
    data[0],
    data[1],
    color="slategray",
    linewidth=2,
    label="CRB ideal"
)

# banda con ruido instrumental
plt.fill_between(
    data[0],
    data[1],
    sigma_sim_err_1,
    color="slategray",
    alpha=0.3
)
plt.fill_between(
    data[0],
    data[1],
    sigma_sim_err_2,
    color="slategray",
    alpha=0.3,
    label="CRB + ruido instrumental"
)


# --- Puntos ---
plt.scatter(N_i, sigma_i,
            s=50, color="firebrick", edgecolor="black",
            zorder=5, label="Ida")

plt.scatter(N_2i, sigma_2i,
            s=50, color="firebrick", edgecolor="black",
            zorder=5)

plt.scatter(N_3i, sigma_3i,
            s=50, color="firebrick", edgecolor="black",
            zorder=5)

plt.scatter(N_u, sigma_u,
            s=50, color="olivedrab", edgecolor="black",
            zorder=5, label="Unión ida-vuelta")

plt.scatter(N_2u, sigma_2u,
            s=50, color="olivedrab", edgecolor="black",
            zorder=5)



# --- Estética ---
plt.grid(True, which="both", alpha=0.3)

plt.xlabel("Número de fotones", fontsize=14)
plt.ylabel("Precisión de localización σ (nm)", fontsize=14)

plt.legend(fontsize=12, loc="upper right")

plt.tight_layout()
#%%


