# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:01:27 2026

@author: Luis1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.load(r"C:\Users\Luis1\Downloads\cam_CRB_curve.npy")
sigma_medido = 18
N_medido =  1753
print(type(data))
sig = pd.read_excel(r"C:\Users\Luis1\Downloads\sigma_exp.xlsx")
sigma  = sig["sigma custom fit"]
plt.plot(data[0],data[1], color = "slategray", label = "Curva teórica") 
plt.grid()
plt.scatter(N_medido,sigma_medido, label = "Experimento", color = "olivedrab")
plt.xlabel("Numero de fotones", fontsize = 16)
plt.ylabel("Precisión [nm]", fontsize = 16)
plt.tight_layout()
plt.legend()