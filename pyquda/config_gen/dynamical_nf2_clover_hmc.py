# Adapted from PyQUDA examples/1_Quenched_HMC.py and tests/test_hmc_clover.py
# https://github.com/CLQCD/PyQUDA  --  MIT License, Copyright (c) 2022-2024 PyQUDA Developers
#
# Dynamical Nf=2 (degenerate light) Wilson-clover HMC.
# Same trajectory loop as the quenched driver; the differences from quenched are:
#   1. a CloverWilsonAction is added to the monomial list  (the sea quark)
#   2. hmc.samplePhi() is called each trajectory           (pseudofermion heatbath)
#   3. a fermion-aware integrator is used (O2Nf2Ng0V / O4Nf5Ng0V), optionally nested
# For Nf=2 the rational parameter is trivial (RationalParam() defaults) -- no Remez needed.

from math import exp
from time import perf_counter
from pyquda_utils import core, io
from pyquda.hmc import HMC, O2Nf2Ng0V  # , O4Nf5Ng0V for a 4th-order option
from pyquda.action import GaugeAction, CloverWilsonAction
from pyquda.enum_quda import QudaVerbosity
from pyquda_utils import core
from pyquda_utils.hmc_param import (
    symanzikTreeGaugeLoopParam as loopParam,   # or wilsonGaugeLoopParam for plain plaquette
    fermionRationalParam as rationalParam,
)

# ---------------------------------------------------------------------------
# Physics / run parameters  --  TUNE THESE
# ---------------------------------------------------------------------------
beta, u_0 = 6.20, 0.855         # gauge coupling / tadpole factor
clover_csw = 0.0                # 0.0 = unimproved Wilson (your stated setup)
                                # 1 / u_0**3  = tree-level tadpole-improved clover (QUDA has the force)
mass = 0.3                      # bare quark mass; kappa = 1/(2*(mass+4)). TUNE to your target m_pi
tol, maxiter = 1e-6, 1000       # sloppy CG for the MD force (loosen -> larger |dH|)

start, stop, warm, save = 0, 2000, 500, 5   # trajectories, thermalization, save cadence
t, n_steps = 1.0, 20                          # trajectory length (MD units) and integrator steps

save_path="/scratch/justd3an/lqcd/configurations/unquenched/l1632f00b5950m0300a"

# ---------------------------------------------------------------------------
# Lattice  --  ONE LINE controls the volume
# ---------------------------------------------------------------------------
core.init([1, 1, 1, 1], enable_force_monitor=True)
#          ^ MPI grid [x,y,z,t]; use e.g. [1,1,1,2] or [2,2,2,2] for multi-GPU 32^4

latt_info = core.LatticeInfo([16, 16, 16, 32], t_boundary=-1, anisotropy=1.0)   # 16^3x32 

# ---------------------------------------------------------------------------
# Action monomials: gauge + one Nf=2 clover fermion
# ---------------------------------------------------------------------------
monomials = [
    GaugeAction(latt_info, loopParam(u_0), beta),
    # rationalParam(num_flavor, md_degree, fa_degree, lower_bound, upper_bound, precision)
    # For num_flavor=2 the degrees/bounds are ignored (defaults returned), but must be passed.
    CloverWilsonAction(
        latt_info, rationalParam(2, 12, 15, 7e-4, 32, 50),
        mass, 2, tol, maxiter, clover_csw,
    ),
]

# Single-timescale integrator (simplest; start here):
hmc = HMC(latt_info, monomials, O2Nf2Ng0V(n_steps))

# Multi-timescale alternative (gauge force on a finer inner step than the fermion force):
#   hmc_inner = HMC(latt_info, monomials[:1], O2Nf2Ng0V(4))   # gauge only, 4 sub-steps
#   hmc       = HMC(latt_info, monomials[1:], O2Nf2Ng0V(n_steps), hmc_inner)  # fermion outer

hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
gauge = core.LatticeGauge(latt_info)     # cold (unit) start
hmc.initialize(10086, gauge)             # seed + load gauge/momenta

plaq = hmc.plaquette()
core.getLogger().info(f"Trajectory {start}:\nPlaquette = {plaq}\n")

# ---------------------------------------------------------------------------
# Reversibility check (run this ONCE before a long chain, then set to False)
# Integrate forward by +t, then backward by -t; a correct, reversible
# integrator returns to the starting Hamiltonian. |dH_reversibility| should be
# tiny (~1e-6 or smaller, limited by CG tol and float precision). A large value
# means a bug or too-loose solver, NOT a physics result -- do not trust the
# ensemble until this passes.
# ---------------------------------------------------------------------------
CHECK_REVERSIBILITY = True
if CHECK_REVERSIBILITY:
    hmc.gaussMom()
    hmc.samplePhi()
    h_start = hmc.momAction() + hmc.gaugeAction() + hmc.fermionAction()
    hmc.integrate(t, 2e-15)            # forward
    hmc.integrate(-t, 2e-15)           # backward (negative tau reverses the trajectory)
    h_back = hmc.momAction() + hmc.gaugeAction() + hmc.fermionAction()
    core.getLogger().info(
        f"[reversibility] dH = {h_back - h_start:.3e}  "
        f"(want |dH| << 1, ideally < 1e-6)\n"
    )
    # restore the cold start so the check doesn't perturb the production chain
    gauge = core.LatticeGauge(latt_info)
    hmc.initialize(10086, gauge)

# ---------------------------------------------------------------------------
# Trajectory loop  
# ---------------------------------------------------------------------------
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom()      # momentum heatbath
    hmc.samplePhi()     # <-- pseudofermion heatbath (NEW vs quenched)

    kinetic_old, potential_old = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, 2e-15)   # MD evolution; 2nd arg = SU(3) reprojection tol

    kinetic, potential = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy = kinetic + potential

    accept = hmc.accept(energy - energy_old)   # Metropolis on dH
    if accept or i < warm:                     # force-accept during thermalization
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaq = hmc.plaquette()
    core.getLogger().info(
        f"Trajectory {i + 1}:\n"
        f"Plaquette = {plaq}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"acceptance rate = {exp(min(energy_old - energy, 0)) * 100:.2f}%\n"
        f"accept? {accept}   warmup? {i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )
    if (i + 1) % save == 0:
        io.writeNERSCGauge(f"{save_path}/l1632f00b5950m0300a.{i + 1}", latt_info.global_size, gauge)
