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
data = np.load(r"C:\Users\Luis1\Downloads\cam_CRB_curve_SBRcte.npy")
sig = pd.read_excel(r"C:\Users\Luis1\Downloads\sigma_exp.xlsx")

sigma = sig["sigma custom fit"]
N_exp = sig["ph tot"] 


sigma_medido_14_i= 13
N_medido_14_i = 411

sigma_medido_14_u = 14
N_medido_14_u = 392
# --- Figura ---
plt.figure(figsize=(7,6))

# Curva teórica (CRB)
plt.loglog(data[0], data[1], 
           color="slategray", 
           linewidth=2,
           label="Curva teórica cámara (CRB)")


# Punto específico medido
plt.scatter(N_medido_14_u, sigma_medido_14_u,
            s=80,
            color="olivedrab",
            edgecolor="black",
            zorder=5,
            label=f"Precisión ida + vuelta\nN = {N_medido_14_u}\nσ = {sigma_medido_14_u} nm")
plt.scatter(N_medido_14_i, sigma_medido_14_i,
            s=80,
            color="red",
            edgecolor="black",
            zorder=5,
            label=f"Precisión ida + ida \nN = {N_medido_14_i}\nσ = {sigma_medido_14_i} nm")


plt.xlabel("Número de fotones", fontsize=16)
plt.ylabel("Precisión [nm]", fontsize=16)
plt.grid(which="both", alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

#%%
