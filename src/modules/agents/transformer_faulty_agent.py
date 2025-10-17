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
        mask = mask * self_exclusion_mask
        return mask

    def reset(self):
        self._faulty = False
        self.timestep = 0
        self.fault_schedule = self.generate_fault_schedule()
        T_max = self.args.max_seq_len
        self.time_to_next_transition = [0] * T_max
        
        # Calculate time remaining until the next mode change
        # Start from the last bin's start and go backward to 0
        current_time_to_transition = 0
        for t in reversed(range(T_max)):  # This goes from last_bin_start down to 0
            is_faulty = t in self.fault_schedule
            is_next_faulty = (t + 1) in self.fault_schedule
            if is_faulty != is_next_faulty:  # Transition point
                current_time_to_transition = 1
            else:
                current_time_to_transition += 1
            self.time_to_next_transition[t] = current_time_to_transition

    def generate_fault_schedule(self):
        """Generate blocks of faulty behavior based on fixed bin selection"""
        faulty_timesteps = set()
        
        total_timesteps = self.args.max_seq_len - 1
        bin_size = 25
        num_bins = total_timesteps//bin_size
        
        num_bins_to_select = self.args.num_faulty_bins
        num_bins_to_select = min(num_bins_to_select, num_bins)
        
        # Randomly select bins without replacement
        selected_bins = random.sample(range(num_bins), num_bins_to_select)
        
        # Add all timesteps from selected bins to faulty_timesteps
        for bin_idx in selected_bins:
            bin_start = bin_idx * bin_size
            bin_end = min(bin_start + bin_size, total_timesteps + 1)
            faulty_timesteps.update(set(range(bin_start, bin_end)))
        
        
        return faulty_timesteps
    
    def forward(self, inputs, memory=None, attn_mask=None, actions=None, return_aux_losses=False, timestep=0):
        # Check if current timestep should be faulty
        self._faulty = timestep in self.fault_schedule
        #print(f"timestep {timestep} faulty {self._faulty}")
        # Get regular Q-values/logits from parent class
        
        if return_aux_losses:
            q, h, aux_losses = super().forward(inputs, memory=memory, attn_mask=attn_mask, actions=actions, return_aux_losses=return_aux_losses)
        else:
            q, h = super().forward(inputs, memory=memory, attn_mask=attn_mask, actions=actions, return_aux_losses=return_aux_losses)
        
        agent_mask = torch.zeros(self.args.n_agents, device=q.device)
        for faulty_idx in self.faulty_agent_indices:
            agent_mask[faulty_idx] = 1.0 
        if self.faulty_agent_indices and self._faulty:
            # For interleaved data, faulty agents appear every n_agents rows
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

        T_max = self.args.max_seq_len# Assuming T_max is available
        delta_t = self.time_to_next_transition[timestep]/T_max
        
        # Feature 2: Current Mode
        current_mode = 1.0 if self._faulty else 0.0
        
       
        context_list = [
            torch.tensor([delta_t], device=q.device),
            torch.tensor([current_mode], device=q.device),
            agent_mask 
        ]
        F_t = torch.cat(context_list, dim=0) # Shape:
        
        # The output must be adapted to return the context (F_t) for storage in the batch
        if return_aux_losses:
            return q, h, aux_losses, F_t.unsqueeze(0) # unsqueeze to
        return q, h, F_t.unsqueeze(0)
        