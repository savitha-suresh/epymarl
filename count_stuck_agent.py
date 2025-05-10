import json
import os
import glob
from collections import defaultdict

def analyze_agent_positions(input_dir):
    """
    Read all JSON files in the input directory and analyze when agents stay in
    the same position for multiple timesteps.
    
    Args:
        input_dir: Directory containing filtered JSON files
    
    Returns:
        Dictionary with analysis results
    """
    # Get all JSON files in the input directory
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return None
    
    print(f"Found {len(json_files)} JSON files to analyze")
    
    
    # Dictionary to track agent positions
    # Key: agent_id, Value: {position: count, last_step: step}
    agent_tracking = defaultdict(list)
    
    
    
    # Process each file
    for i, json_file in enumerate(json_files):
        print(f"Processing file {i+1}/{len(json_files)}: {os.path.basename(json_file)}")
        
      
            # Read the input file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Process each environment entry
        for env_data in data:
            # Get the step number
            step = env_data['step']
            for agent_id, obs in env_data['agents'].items():
                # store step and the position of the agent
                agent_tracking[agent_id].append((step, (obs[0], obs[1])))
                
    return agent_tracking

def display_stuck_agents(agent_tracking):
    """
    Display agents that stayed in the same position for more than 4 timesteps.
    
    Args:
        agent_tracking: Dictionary with agent tracking data
    """
    if not agent_tracking:
        return
    prev_step = 0
    results = defaultdict(list)
    position_tracking = defaultdict(lambda: defaultdict(int))
    for agent_id, positions in agent_tracking.items():
        prev_step = 0
        for step, position in positions:
            # end of episode
            if (prev_step!=0)and ((step < prev_step) or (step > prev_step+1)):
                for position, count in position_tracking[agent_id].items():
                    if count > 4:
                        results[agent_id].append(count)
                position_tracking[agent_id]=defaultdict(int)
            
            elif step == prev_step+1 or prev_step == 0:
                position_tracking[agent_id][position] += 1
            
            prev_step = step
    
    for agent_id, positions in position_tracking.items():
        for position, count in positions.items():
            if count > 4:
                results[agent_id].append(count)
    #print(results)
    print(results.keys())
    for agent_id in results:
        print(f"Agent id {agent_id} {len(results[agent_id])}")

def main():
    input_dir = 'filtered_json'  # Directory containing the filtered JSON files
    
    print(f"Input directory: {input_dir}")
    
    # Analyze agent positions
    agent_tracking = analyze_agent_positions(input_dir)
    
    display_stuck_agents(agent_tracking)

if __name__ == "__main__":
    main()
