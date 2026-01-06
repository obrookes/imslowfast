#!/bin/bash      
#SBATCH --nodes=1    
#SBATCH --gres=gpu:1    
#SBATCH --ntasks-per-gpu=1    
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB    
#SBATCH --partition=hopper
#SBATCH --job-name=mvitv2_offset_f=32_test
#SBATCH --time=12:00:00
#SBATCH --output=%x.out 
    
echo "Running on host $(hostname)"    
echo "Time is $(date)"    
echo "Slurm job ID is $SLURM_JOB_ID"    
    
cd ~/imslowfast  

# Purge
module purge
    
# Load CUDA    
module load cuda/12.3
    
singularity exec --nv --bind /lfs1i3/home/b35u/obrookes.b35u:/mnt ~/slowfast.sif python -W ignore ./tools/run_net.py \
	--cfg '/mnt/imslowfast/configs/species_classification/test/mvit/MVIT_B_16x4_OFFSET_F=32_TEST.yaml'
