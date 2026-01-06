#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-gpu=1
#SBATCH --job-name=mvit-v2_disjoint_mixup_f=32
#SBATCH --time=06:00:00
#SBATCH --output=%x.out

echo "Running on host $(hostname)"
echo "Time is $(date)"
echo "Slurm job ID is $SLURM_JOB_ID"

cd ~/imslowfast

# Purge
module purge

# Load CUDA
module load cudatoolkit/23.9_12.2

singularity exec --nv --bind /lus/lfs1aip1/home/aiape/obrookes.aiape:/mnt ./singularity/slowfast.sif python -W ignore ./tools/run_net.py \
    --cfg '/imslowfast/configs/species_classification/train/mvitv2/MVIT_B_16x4_DISJOINT_MIXUP_F=32.yaml'
