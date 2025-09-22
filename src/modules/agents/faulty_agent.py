from .rnn_agent import RNNAgent
import random

class FaultyAgent(RNNAgent):
    """
    An agent that becomes faulty with a certain probability.
    Multiple agents can become faulty, specified by args.n_faulty_agents.
    When faulty, these agents will always select action 0.
    
    Data format: Interleaved agents per batch
    """
    
    def __init__(self, input_shape, args):
        super(FaultyAgent, self).__init__(input_shape, args)
        self.args = args
        if self.args.action_fault:
            raise ValueError("Cannot use this network fault with action_fault set to True")
        self._faulty = False
        self.init_random_fault()
        self.faulty_row = self.args.faulty_row
        self.no_op_action = 0
    
    
    def init_random_fault(self):
        self.faulty_agent_indices = set(random.sample(range(self.args.n_agents), 
                                                      self.args.n_faulty_agents))
        print(f"Agents {self.faulty_agent_indices} have become network faulty!")
        #self.faulty_agent_indices = {1}
        self._faulty = False
        self._faulty_timestep = 0

    def reset_fault(self):
        self._faulty = False
        T = self.args.max_seq_len - 1
        self._faulty_timestep = self.sample_fault_timestep(mean=3*(T/4))
        

    def sample_fault_timestep(self,  mean=None, std=None, spread=0.2):
        """
        Sample a timestep when the agent becomes faulty.
        
        T: total timesteps in the episode
        mean: desired mean timestep (default T/2)
        std: standard deviation. If None, it is derived from mean and spread.
            ~95% of samples fall in [mean*(1-spread), mean*(1+spread)]
        spread: fraction of mean that defines the 95% interval (default 0.5)
        
        Returns: integer timestep in [0, T-1]
        """
        T = self.args.max_seq_len - 1
        if mean is None:
            mean = T / 2

        if std is None:
            std = (spread * mean) / 2  # because 95% ≈ ±2σ
        
        k = int(random.gauss(mean, std))
       
        return max(0, k)
    

    def forward(self, inputs, hidden_state, t):
        # Check if we should make agents faulty
        # if self.faulty_agent_indices and not self._faulty and random.random() < self.args.fault_prob:
        #     self._faulty = True
        if self.faulty_agent_indices and not self._faulty and t >= self._faulty_timestep:
            self._faulty = True
            
        # Get regular Q-values/logits from parent class
        q, h = super().forward(inputs, hidden_state)
        
        if self.faulty_agent_indices and self._faulty:
            # For interleaved data, faulty agents appear every n_agents rows
            for faulty_idx in self.faulty_agent_indices:
                # Modify Q-values for the specific faulty agents
                if not self.args.constrained_faults:
                    q[faulty_idx::self.args.n_agents, 0] = 1e10  # High logit for action 0
                    q[faulty_idx::self.args.n_agents, 1:] = -1e10  # Low logit for all other actions
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
                        q[q_index, 0] = 1e10
                        q[q_index, 1:] = -1e10

        return q, h
