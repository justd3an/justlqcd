#!/bin/bash
#SBATCH --job-name=l16t32_term
#SBATCH --output=/fs1/home/justd3an/outs/lqcd/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00

set -euo pipefail

source "$HOME/.miniforge3/etc/profile.d/conda.sh"
conda activate lqcd

cd /fs1/home/justd3an/justlqcd/pyquda/analysis

python3 thermalization.py