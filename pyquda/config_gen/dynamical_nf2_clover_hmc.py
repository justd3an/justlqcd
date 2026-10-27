# Adapted from PyQUDA examples/1_Quenched_HMC.py and tests/test_hmc_clover.py
# https://github.com/CLQCD/PyQUDA  --  MIT License, Copyright (c) 2022-2024 PyQUDA Developers
#
# Dynamical Nf=2 (degenerate light) Wilson-clover HMC.
# Same trajectory loop as the quenched driver; the differences from quenched are:
#   1. a CloverWilsonAction is added to the monomial list  (the sea quark)
#   2. hmc.samplePhi() is called each trajectory           (pseudofermion heatbath)
#   3. a fermion-aware integrator is used (O2Nf2Ng0V / O4Nf5Ng0V), optionally nested
# For Nf=2 the rational parameter is trivial (RationalParam() defaults) -- no Remez needed.

import os
import re
import gvar as gv
from math import exp
from time import perf_counter
from pyquda_utils import core, io
from pyquda.hmc import HMC, O2Nf2Ng0V  # , O4Nf5Ng0V for a 4th-order option
from pyquda.action import GaugeAction, CloverWilsonAction
from pyquda.enum_quda import QudaVerbosity
from pyquda_utils.hmc_param import (
    symanzikTreeGaugeLoopParam as loopParam,   # or wilsonGaugeLoopParam for plain plaquette
    fermionRationalParam as rationalParam,
)

# ---------------------------------------------------------------------------
# Physics / run parameters  --  TUNE THESE
# ---------------------------------------------------------------------------
beta, u_0 = 6.20, 0.855         # gauge coupling / tadpole factor
# Nf=2 non-perturbative c_sw(g0^2), Jansen & Sommer 1998 (ALPHA Collaboration),
# https://arxiv.org/abs/hep-lat/9803017, eq. (2.25), valid for beta >= 5.4.
# At beta=6.20 (g0^2 = 6/beta = 0.9677) this evaluates to c_sw ~= 1.4746.
clover_csw = 1.4746
mass = 0.3                      # bare quark mass; kappa = 1/(2*(mass+4)). TUNE to your target m_pi
tol, maxiter = 1e-6, 1000       # sloppy CG for the MD force (loosen -> larger |dH|)

warm, save = 600, 5                          # thermalization trajectories, save cadence
n_confs = 256                                # how many post-warmup configs to generate
stop = warm + n_confs * save                 # total trajectories = warmup + production
t, n_steps = 1.0, 20                          # trajectory length (MD units) and integrator steps

save_path = "/scratch/justd3an/lqcd/configurations/unquenched/l1632f00b620m0300a"
ckpt_prefix = "l1632f00b620m0300a"           # basename used for saved gauge configs

therm_dump_dir = "/fs1/home/justd3an/data/pyquda/S16T32/thermalization"
therm_dump_path = f"{therm_dump_dir}/plaquette_thermalization.dump"

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

# ---------------------------------------------------------------------------
# Checkpoint / restart: resume from the latest saved config in save_path if
# one exists, so a job that hits the walltime limit can be resubmitted
# without losing prior thermalization/production trajectories.
#
# Production configs are numbered as a sequential index (0, 1, 2, ...) --
# one per saved config, not the trajectory count -- so the 256 generated
# configs are labeled .0 through .255. The thermalized starting gauge is
# saved separately as "{ckpt_prefix}.therm" so it never collides with that
# numbering.
# ---------------------------------------------------------------------------
therm_ckpt_path = f"{save_path}/{ckpt_prefix}.therm"

os.makedirs(save_path, exist_ok=True)
saved_prod_traj = [
    int(m.group(1))
    for name in os.listdir(save_path)
    if (m := re.fullmatch(rf"{re.escape(ckpt_prefix)}\.(\d+)", name))
]

if saved_prod_traj:
    conf_idx = max(saved_prod_traj)
    start = warm + (conf_idx + 1) * save
    gauge = io.readNERSCGauge(f"{save_path}/{ckpt_prefix}.{conf_idx}")
    core.getLogger().info(f"Resuming from checkpoint at trajectory {start} (production config {conf_idx})\n")
elif os.path.exists(therm_ckpt_path):
    start = warm
    gauge = io.readNERSCGauge(therm_ckpt_path)
    core.getLogger().info(f"Resuming from thermalized checkpoint at trajectory {start}\n")
else:
    start = 0
    gauge = core.LatticeGauge(latt_info)     # cold (unit) start

hmc.initialize(10086 + start, gauge)         # seed + load gauge/momenta

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
CHECK_REVERSIBILITY = False
if CHECK_REVERSIBILITY and start == 0:      # only on a fresh chain, not on a resume
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
# Thermalization (force-accept every sweep; no Metropolis test needed).
# Toggle off once you've confirmed `warm` sweeps is enough -- watch the
# plaquette below level off; if it's still drifting at the last sweep,
# `warm` is too small.
# ---------------------------------------------------------------------------
RUN_THERMALIZATION = False
if RUN_THERMALIZATION and start == 0:
    therm_plaq = []
    for i in range(warm):
        s = perf_counter()
        hmc.gaussMom()
        hmc.samplePhi()
        hmc.integrate(t, 2e-15)
        hmc.saveGauge(gauge)   # force-accept
        plaq = hmc.plaquette()
        therm_plaq.append(plaq)
        core.getLogger().info(
            f"[thermalization] sweep {i + 1}/{warm}:\n"
            f"Plaquette = {plaq}\n"
            f"HMC time = {perf_counter() - s:.3f} secs\n"
        )
    start = warm

    os.makedirs(therm_dump_dir, exist_ok=True)
    gv.dump(therm_plaq, therm_dump_path)
    core.getLogger().info(f"Wrote thermalization plaquette trace to {therm_dump_path}\n")

    # save the thermalized gauge as a checkpoint so a future run resumes from
    # here instead of re-thermalizing (picked up by the "therm" fallback in
    # the checkpoint scan above)
    os.makedirs(save_path, exist_ok=True)
    io.writeNERSCGauge(therm_ckpt_path, gauge)
    core.getLogger().info(f"Saved thermalized gauge to {therm_ckpt_path}\n")

# ---------------------------------------------------------------------------
# Trajectory loop  --  set False to only test thermalization above
# ---------------------------------------------------------------------------
RUN_TRAJECTORIES = True
if RUN_TRAJECTORIES:
    for i in range(start, stop):
        s = perf_counter()

        hmc.gaussMom()      # momentum heatbath
        hmc.samplePhi()     # <-- pseudofermion heatbath

        kinetic_old, potential_old = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
        energy_old = kinetic_old + potential_old

        hmc.integrate(t, 2e-15)   # MD evolution; 2nd arg = SU(3) reprojection tol

        kinetic, potential = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
        energy = kinetic + potential

        accept = hmc.accept(energy - energy_old)   # Metropolis on dH
        if accept:
            hmc.saveGauge(gauge)
        else:
            hmc.loadGauge(gauge)

        plaq = hmc.plaquette()
        core.getLogger().info(
            f"Trajectory {i + 1}:\n"
            f"Plaquette = {plaq}\n"
            f"Delta_E = {energy - energy_old}\n"
            f"acceptance rate = {exp(min(energy_old - energy, 0)) * 100:.2f}%\n"
            f"accept? {accept}\n"
            f"HMC time = {perf_counter() - s:.3f} secs\n"
        )
        if (i + 1) % save == 0:
            conf_idx = (i + 1 - warm) // save - 1
            io.writeNERSCGauge(f"{save_path}/{ckpt_prefix}.{conf_idx}", gauge)
