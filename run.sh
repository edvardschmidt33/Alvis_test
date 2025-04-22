#! #! /usr/bin/bash
#SBATCH --time=01:00:00
#SBATCH -A NAISS2025-5-98
#SBATCH --gpus-per-node=A40:1


EXE="apptainer exec --nv /<absolute>/<dir>/example.sif python3"


${EXE} main.py