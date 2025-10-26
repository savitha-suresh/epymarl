from .rnn_agent import RNNAgent
from .transformer_agent import TransformerAgent
import random
import torch

class TransformerFaultyAgent(TransformerAgent):
    """
    An agent that becomes faulty for a specified percentage of time steps.
    Multiple agents can become faulty, specified by args.n_faulty_agents.
    When faulty, these agents will always select action 0.
    
    Data format: Interleaved agents per batch
    """
    
    def __init__(self, input_shape, args):
        super(TransformerFaultyAgent, self).__init__(input_shape, args)
        self.args = args
        if self.args.action_fault:
            raise ValueError("Cannot use this network fault with action_fault set to True")
        self._faulty = False
        self.init_random_fault()
        self.reset()
        self.faulty_row = self.args.faulty_row
        self.no_op_action = 0
        
    
    def init_random_fault(self):
        self.faulty_agent_indices = set(random.sample(range(self.args.n_agents), 
                                                      self.args.n_faulty_agents))

        print(f"Agents {self.faulty_agent_indices} have become network faulty!")
        self._faulty = False


    def generate_agent_labels(self, batch_size):
        agent_labels = torch.ones(batch_size, self.args.n_agents, device=self.args.device)
        if hasattr(self, 'faulty_agent_indices'):
            for idx in self.faulty_agent_indices:
                agent_labels[:, idx] = 0
        return agent_labels
    
    def build_cross_attn_mask(self):
        # Create a mask that allows agents to attend to each other
        # This is a square mask of size n_agents x n_agents
        mask = torch.ones(self.args.n_agents, self.args.n_agents, device=self.args.device)
        eye_mask = torch.eye(self.args.n_agents, device=self.args.device).unsqueeze(0)
        eye_mask = eye_mask.expand(self.args.batch_size, -1, -1)
        self_exclusion_mask = 1.0 - eye_mask
        if self.faulty_agent_indices:
            for idx in self.faulty_agent_indices:
                mask[:, idx] = 0
                mask[idx, :] = 0
        mask =  mask.unsqueeze(0).expand(self.args.batch_size, -1, -1)
        #mask = mask * self_exclusion_mask
        return mask

    def reset(self):
        self._faulty = False
        self.timestep = 0
        self.fault_schedule = self.generate_fault_schedule()
        #print(f"Faulty timesteps: {len(self.fault_schedule)}, {self.fault_schedule}")

    def generate_fault_schedule(self):
        """Generate blocks of faulty behavior based on fault percentage"""
        total_timesteps = self.args.max_seq_len - 1
        fault_percentage = getattr(self.args, 'fault_percentage', 20)  # Default 20%
        num_faulty_steps = int(total_timesteps * fault_percentage / 100)
        
        # Block parameters
        min_block_size = getattr(self.args, 'min_fault_block', 5)
        calculated_max = max(min_block_size, int(total_timesteps * fault_percentage / 200))
        max_block_size = calculated_max
        
        faulty_timesteps = set()
        current_pos = 0
        
        while len(faulty_timesteps) < num_faulty_steps and current_pos < total_timesteps:
            # Calculate remaining steps needed
            remaining_steps = num_faulty_steps - len(faulty_timesteps)
            
            # Pick random block size (between 0 and max_block_size)
            # But don't exceed remaining steps or remaining timesteps
            max_possible_block = min(max_block_size, remaining_steps, total_timesteps - current_pos)
            block_size = random.randint(0, max_possible_block)
            
            # Add the block to faulty_timesteps
            if block_size > 0:
                faulty_timesteps.update(range(current_pos, current_pos + block_size))
                current_pos += block_size
            
            # Move to next position (skip one step to create gap)
            current_pos += 1
        
        return faulty_timesteps

    
    # def generate_fault_schedule(self):
    #     """Generate blocks of faulty behavior based on fault percentage"""
    #     faulty_timesteps = set()
    #     #if random.random() < self.args.fault_prob:
    #     faulty_timesteps.update(range(10, 60))
        
    #     faulty_timesteps.update(range(80,120))
    #     faulty_timesteps.update(range(200,300))
    #     faulty_timesteps.update(range(350,420))
        
    #     return faulty_timesteps
    
    def forward(self, inputs, memory=None, attn_mask=None, actions=None, return_aux_losses=False, timestep=0):
        # Check if current timestep should be faulty
        self._faulty = timestep in self.fault_schedule

        # Get regular Q-values/logits from parent class
        if return_aux_losses:
            q, h, aux_losses = super().forward(inputs, memory=memory, attn_mask=attn_mask, actions=actions, return_aux_losses=return_aux_losses)
        else:
            q, h = super().forward(inputs, memory=memory, attn_mask=attn_mask, actions=actions, return_aux_losses=return_aux_losses)
        
        if self.faulty_agent_indices and self._faulty:
            # For interleaved data, faulty agents appear every n_agents rows
            for faulty_idx in self.faulty_agent_indices:
                # Modify Q-values for the specific faulty agents
                if not self.args.constrained_faults:
                    q[faulty_idx::self.args.n_agents, :, 0] = 1e10  # High logit for action 0
                    q[faulty_idx::self.args.n_agents, :, 1:] = -1e10  # Low logit for all other actions
                else:
                    # inputs is of the form (bs*n_agents, 75)
                    # row 0, 4, 8 are for agent 0, row 1, 5,9 are for agent 1 in the bs environments
                    # 75 is 70 obs, last action encoding, agent_id
                    # we will check the pos of the faulty agent
                    # [:, 0] is 0the column in all rows
                    positions_of_agent = inputs[faulty_idx::self.args.n_agents][:, 1]
                    index_to_halt = []
                    for env_idx in range(positions_of_agent.size(0)):
                        if positions_of_agent[env_idx] == self.faulty_row:
                            index_to_halt.append(env_idx)
                    for halt_idx in index_to_halt:
                        q_index = faulty_idx + (self.args.n_agents*halt_idx)
                        q[q_index, :, 0] = 1e10
                        q[q_index, :, 1:] = -1e10
        if return_aux_losses:
            return q, h, aux_losses
        return q, h
        
