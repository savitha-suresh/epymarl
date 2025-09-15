#!/bin/bash

# Step 1: Cleanup previous logs and outputs
rm observation_lbforaging:*.log
rm -rf filtered_json
rm -rf output_json

# Step 2: Run the simulation
python -u src/main.py --config=mappo_lbf --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-2s-10x10-4p-4f-v3" fault_idx=0

# Step 3: Find the most recent observation log file
latest_log=$(ls -t observation_lbforaging:*.log | head -n 1)
echo "Latest log file: $latest_log"

# Step 4: Run filtering and processing scripts with log file as argument
python filter_obs.py "$latest_log"
python filter_agents.py
python count_stuck_agent.py
