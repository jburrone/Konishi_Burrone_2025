#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/%u/%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
singularity exec --nv \
    segreg_ubuntu20.sif \
    python3.8 scan_xyzTz_batch.py
