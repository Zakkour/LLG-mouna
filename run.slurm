#!/bin/bash
#SBATCH --job-name=LLG-calcul
#SBATCH --mail-type=ALL --mail-user=mona.kassan96@gmail.com
#SBATCH --partition=bigmem01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=16
module load anaconda/3 
python solve_2.py
