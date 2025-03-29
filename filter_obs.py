import re
import json
import os
from datetime import datetime

def process_file(file_path, output_dir, chunk_size=100):
    """
    Process the data file and periodically write chunks to JSON files.
    
    Args:
        file_path: Path to the input data file
        output_dir: Directory to save JSON output files
        chunk_size: Number of Env: 0 entries to process before writing to a file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables
    result = []
    current_env_data = None
    current_env_id = None
    current_agent_id = None
    current_agent_values = []
    file_counter = 1
    
    with open(file_path, 'r') as file:
        line = file.readline()
        
        while line:
            # Check if a new environment data block starts
            env_match = re.match(r'Env: (\d+) Reward: ([\d\.-]+) Step: (\d+)', line)
            
            if env_match:
                env_id = int(env_match.group(1))
                reward = float(env_match.group(2))
                step = int(env_match.group(3))
                
                # Start collecting data if it's Env: 0
                if env_id == 0:
                    # Create a new entry for this Env: 0 instance
                    current_env_data = {
                        'env_id': env_id,
                        'reward': reward,
                        'step': step,
                        'agents': {}
                    }
                    current_agent_id = None
                    current_agent_values = []
                    result.append(current_env_data)
                    
                    # Write to file when we've collected enough entries
                    if len(result) >= chunk_size:
                        write_to_json(result, output_dir, file_counter)
                        file_counter += 1
                        result = []  # Reset after writing
                else:
                    current_env_data = None
                
                current_env_id = env_id
            # Check for agent data
            agent_match = re.search(r'Agent (\d+): \[(.*)', line)
            if current_env_id == 0 and agent_match:
                # Save any in-progress agent data
                
                
                agent_id = int(agent_match.group(1))
                values_str = agent_match.group(2)
                
                # Remove trailing bracket if present in this line
                
                
                # Start collecting values for this agent
                current_agent_values = []
                for val in values_str.split():
                    if val.strip():  # Skip empty strings
                        current_agent_values.append(float(val.rstrip('.')))
                
                current_agent_id = agent_id
                
            # Check for continuation lines of agent data (just numbers, no Agent prefix)
            elif current_env_id ==0 and current_agent_id is not None and re.match(r'\s*[\d\.\s]+', line):
                # This is a continuation of the previous agent's data
                # Clean up the line - remove trailing bracket if present
                line = line.rstrip()
                
                # Add values to the current agent's data
                for val in line.split():
                    if val.strip() and val.rstrip(']'):  # Skip empty strings
                        val = val.strip().rstrip(']')
                        current_agent_values.append(float(val.rstrip('.')))
                
                # If we found the closing bracket, add this agent to the current environment
                if ']' in line:
                    if current_env_data:
                        assert len(current_agent_values) == 71
                        current_env_data['agents'][current_agent_id] = current_agent_values
                    current_agent_id = None
                    current_agent_values = []
            
            line = file.readline()
    
    
    # Write any remaining data
    if result:
        write_to_json(result, output_dir, file_counter)

def write_to_json(data, output_dir, file_counter):
    """
    Write data to a JSON file.
    
    Args:
        data: List of environment data to write
        output_dir: Directory to save the file
        file_counter: Current file number
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"env_data_{file_counter}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"Wrote {len(data)} environment entries to {filepath}")

def main():
    file_path = 'observation_rware:rware-tiny-4ag-v2_2_1741973654.9803455.log'  # Replace with your actual file path
    output_dir = 'output_json_2'  # Directory to save JSON files
    chunk_size = 10000  # Number of Env: 0 entries to process before writing
    
    print(f"Processing file: {file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Writing new JSON file after every {chunk_size} Env: 0 entries")
    
    process_file(file_path, output_dir, chunk_size)
    print("Processing complete!")

if __name__ == "__main__":
    main()
