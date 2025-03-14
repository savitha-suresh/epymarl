#!/bin/bash
#SBATCH --job-name=savs-filter-obs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=savitha@comp.nus.edu.sg
#SBATCH --partition=long
#SBATCH --time=36:00:00
source ~/.bashrc
conda activate ft-gym-env   # On Linux/macOS
#srun python src/main.py --config=mappo --env-config=sc2 with env_args.map_name="2s_vs_1sc"
#srun python src/main.py --config=maa2c --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
#srun python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
#srun python src/main.py --config=coma  --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-adversary-v3" env_args.pretrained_wrapper="PretrainedAdversary"
#srun python src/main.py --config=maddpg --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
srun python filter_obs.py
srun python filter_agents.py
srun python count_stuck_agent.py
