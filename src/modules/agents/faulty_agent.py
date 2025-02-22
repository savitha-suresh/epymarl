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
        self._faulty = False
        self.faulty_agent_indices = set(random.sample(range(self.args.n_agents), 
                                                      self.args.n_faulty_agents))
        
    def forward(self, inputs, hidden_state):
        # Check if we should make agents faulty
        if not self._faulty and random.random() < self.args.fault_prob:
            self._faulty = True
            print(f"Agents {self.faulty_agent_indices} have become faulty!")
            
        # Get regular Q-values/logits from parent class
        q, h = super().forward(inputs, hidden_state)
        
        if self._faulty:
            # For interleaved data, faulty agents appear every n_agents rows
            for faulty_idx in self.faulty_agent_indices:
                # Modify Q-values for the specific faulty agents
                q[faulty_idx::self.args.n_agents, 0] = 1e10  # High logit for action 0
                q[faulty_idx::self.args.n_agents, 1:] = -1e10  # Low logit for all other actions

        return q, h
