import numpy as np
import csv
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt


PX_FOR_PSF = 5
PX_SIZE_NM = 112
FINAL_PSF_SIZE_NM = PX_FOR_PSF * PX_SIZE_NM
NUM_MOL_PER_SET = 3

# Background parameters
M_BCKG = 0.08110981
Q_BCKG = 36.94920377


def est_bckg(ph_tot):
    return 5


def est_sbr(ph_tot):
    return 77


def crb_map_optimized(p_base, sbr, N):
    """
    Computes isotropic CRB sigma map for given PSF probability array,
    SBR and total photon number N.
    """

    d = 2
    dx = 1
    dy = 1
    eps = 1e-15

    # Incorporate background analytically
    alpha = sbr / (sbr + 1)
    beta = 1 / (PX_FOR_PSF**2 * (sbr + 1))
    p = p_base * alpha + beta

    # Accumulators (2D only, no giant 4D arrays)
    A_sum = np.zeros((FINAL_PSF_SIZE_NM, FINAL_PSF_SIZE_NM))
    B_sum = np.zeros_like(A_sum)
    C_sum = np.zeros_like(A_sum)

    # Loop only over camera pixels
    for i in range(PX_FOR_PSF):
        for j in range(PX_FOR_PSF):

            p_ij = np.clip(p[i, j], eps, None)

            dpdy_ij, dpdx_ij = np.gradient(p_ij, -dy, dx)

            inv_p = 1.0 / p_ij

            A_sum += inv_p * dpdx_ij**2
            B_sum += inv_p * dpdy_ij**2
            C_sum += inv_p * (dpdx_ij * dpdy_ij)

    # Fisher determinant term
    F = A_sum * B_sum - C_sum**2
    F = np.clip(F, eps, None)

    # CRB (isotropic average)
    sigma_x2 = B_sum / (N * F)
    sigma_y2 = A_sum / (N * F)

    sigma_iso = np.sqrt((sigma_x2 + sigma_y2) / 2)

    return sigma_iso

if __name__=="__main__":



    prob_arr = np.load(r"C:\Users\Luis1\Downloads\prob_arr_forcam_MLE.npy")

    points_for_crb_curve = 1000
    n_tot_axis = np.linspace(100, 3000, points_for_crb_curve)

    crb_curve = np.zeros((2, points_for_crb_curve))
    crb_curve[0] = n_tot_axis

    center = FINAL_PSF_SIZE_NM // 2

    for ii in range(points_for_crb_curve):

        N = n_tot_axis[ii]
        sbr = est_sbr(N)

        sigma_map = crb_map_optimized(prob_arr, sbr, N)

        crb_curve[1, ii] = sigma_map[center, center]

    np.save('cam_CRB_curve_SBRcte.npy', crb_curve)

    plt.plot(crb_curve[0], crb_curve[1], color='black')
    plt.xlabel('N photons')
    plt.ylabel(r'$\sigma$')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

        
