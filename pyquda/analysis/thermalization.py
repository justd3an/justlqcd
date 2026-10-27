import gvar as gv
import matplotlib.pyplot as plt
import numpy as np


if True:
    plaq_l16t32= np.array(gv.load("/fs1/home/justd3an/data/pyquda/S16T32/thermalization/plaquette_thermalization.dump"))

    plt.figure(figsize=(8,6))
    plt.plot(np.arange(len(plaq_l16t32)), plaq_l16t32[:, 0], '--x', label = "Plaq", color='blue')
    plt.grid()
    plt.xlabel("Iterations", fontsize=16)
    plt.ylabel("Plaquette", fontsize=16)
    plt.title(r"Thermalization for $16^3 \times 32$", fontsize=20)
    plt.ylim(0.371676, 0.375)
    plt.tight_layout()
    plt.legend()
    plt.savefig("/fs1/home/justd3an/justlqcd/pyquda/figures/thermalization/unquenched_hmc_therm.png")