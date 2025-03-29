import json
import os
import glob
from datetime import datetime

def filter_agents(data):
    """
    Filter agents according to the specified criteria:
    1. Agent ID is not 0 or 1
    2. The value at index 1 (2nd element) equals 1.0
    
    Args:
        data: List of environment data dictionaries
    
    Returns:
        List of filtered environment data
    """
    filtered_data = []
    
    for env_data in data:
        filtered_agents = {}
        
        for agent_id, values in env_data['agents'].items():
            # Convert agent_id to int if it's stored as string in JSON
            agent_id = int(agent_id) if isinstance(agent_id, str) else agent_id
            if len(values) != 71:
                raise Exception(f"agent {agent_id} and step {env_data['step']} {len(values)}")
            # Make sure we have at least 2 values
            if len(values) > 1:
                # Filter agents whose id is not 0 or 1 and 2nd index (index 1) is 1
                # if agent_id not in [failed_agents]
                #if agent_id not in [0, 1] and values[2] == 1.0:
                if values[2] == 1.0: 
                    filtered_agents[agent_id] = values
        
        if filtered_agents:
            # Create a copy of the original env_data to avoid modifying it
            filtered_env = {
                'env_id': env_data['env_id'],
                'step': env_data['step'],
                'reward': env_data['reward'],
                'agents': filtered_agents
            }
            filtered_data.append(filtered_env)
    
    return filtered_data

def process_json_files(input_dir, output_dir):
    """
    Process all JSON files in the input directory, filter them,
    and write the results to the output directory.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory to save filtered output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for i, json_file in enumerate(json_files):
        print(f"Processing file {i+1}/{len(json_files)}: {os.path.basename(json_file)}")
        
        try:
            # Read the input file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Apply filtering
            filtered_data = filter_agents(data)
            
            # Generate output filename
            base_filename = os.path.basename(json_file)
            output_filename = f"filtered_{base_filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Write filtered data
            with open(output_path, 'w') as f:
                json.dump(filtered_data, f)
            
            print(f"  - Original entries: {len(data)}")
            print(f"  - Filtered entries: {len(filtered_data)}")
            print(f"  - Wrote filtered data to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print("Filtering complete!")

def main():
    input_dir = 'output_json_2'  # Directory containing the JSON files
    output_dir = 'filtered_json_2'  # Directory to save filtered JSON files
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    process_json_files(input_dir, output_dir)

if __name__ == "__main__":
    main()
