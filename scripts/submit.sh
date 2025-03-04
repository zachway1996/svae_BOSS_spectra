#!/bin/bash
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=176
#SBATCH --nodes=1

echo Activating environment
source /home/way/MDwarf_Continuum/auto_encoder/ae_mdwarf/bin/activate
echo Setting up OpenMP
export OMP_NUM_THREADS=176

echo Running python
python -u -W ignore train_normed_BOSS.py
