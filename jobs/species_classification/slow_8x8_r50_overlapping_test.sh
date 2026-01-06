#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=slow_8x8_r50_overlapping_test
#SBATCH --time=00:30:00
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
    --cfg '/mnt/imslowfast/configs/species_classification/SLOW_8x8_R50_OVERLAPPING_TEST.yaml'
