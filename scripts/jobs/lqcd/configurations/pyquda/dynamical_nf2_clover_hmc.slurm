#!/bin/bash
#SBATCH --job-name=dnf2clover1632
#SBATCH --output=/fs1/home/justd3an/outs/lqcd/configurations/pyquda/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=cascade-lake      # g8-11, the 32GB V100s
#
# Author: Justin Dean
# Date Modified: 09/11/2026
# Run dynamical Nf=2 Wilson-clover HMC (16^3 x 32) on a single V100.
# Submit:  sbatch submit_hmc_v100.slurm

set -euo pipefail

module purge
module load os/rhel_8
module load spack/2023a
module load gcc/12.2.0-2023a-gcc_8.5.0-e643dqu
module load cuda/12.1.0-2023a-gcc_12.2.0-eqyilhs
module load mpich/4.0.2-2023a-gcc_12.2.0-cuda-slurm-zeartfs


source "$HOME/.miniforge3/etc/profile.d/conda.sh"
conda activate lqcd

# point PyQUDA at the QUDA build 
BUILD_DIR="/fs1/home/justd3an/.quda-build"
export QUDA_PATH="$BUILD_DIR"
export LD_LIBRARY_PATH="$BUILD_DIR/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# reuse the tunecache the build-time ctest populated (no re-autotune)
export QUDA_RESOURCE_PATH="$BUILD_DIR/tunecache"
mkdir -p "$QUDA_RESOURCE_PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# confirm running on a 32GB V100 ---
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# --- run (single process; QUDA is MPI=OFF, core.init([1,1,1,1])) ---
cd "/fs1/home/justd3an/justlqcd/pyquda/config_gen"
srun -n1 python dynamical_nf2_clover_hmc.py