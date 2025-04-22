#!/usr/bin/bash
#SBATCH --time=00:10:00
#SBATCH -A NAISS2025-5-98
#SBATCH --gpus-per-node=T4:1
#SBATCH -J mnist_train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err 

EXE="apptainer exec --nv /cephyr/users/schmidte/Alvis/Alvis_test-1/example.sif python3"


${EXE} train.py
