#!/bin/bash
#SBATCH --job-name=savsrware-mappo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=savitha@comp.nus.edu.sg
#SBATCH --partition=long
#SBATCH --time=05:00:00
source ~/.bashrc
conda activate ft-gym-env
chmod +x get_faulty_count_lbf.sh
chmod +x run_and_count_stuck_*
srun get_faulty_count_lbf.sh
