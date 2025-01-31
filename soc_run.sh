#!/bin/sh
#SBATCH --job-name=savsrware-mappo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=savitha@comp.nus.edu.sg
#SBATCH --partition=long
#SBATCH --time=03:00:00
conda activate ft-gym-env   # On Linux/macOS
srun python src/main.py --config=mappo --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
