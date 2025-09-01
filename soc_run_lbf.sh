#!/bin/bash
#SBATCH --job-name=savsrware-mappo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=savitha@comp.nus.edu.sg
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
source ~/.bashrc
conda activate ft-gym-env   # On Linux/macOS
srun python -u src/main.py --config=mappo_lbf --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-10x10-3p-3f-v3"

