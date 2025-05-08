import torch as th

class StuckPenaltyRewardShaper:
    def __init__(self, max_lookback=50, base_penalty=0.05, penalty_growth_rate=1.5):
        """
        Initialize the stuck penalty reward shaper.
        Args:
            max_lookback: Maximum number of timesteps to look back (default: 10)
            base_penalty: Base penalty value for being stuck 2 timesteps (default: 0.05)
            penalty_growth_rate: How much to multiply the penalty for each additional step (default: 1.5)
        """
        self.max_lookback = max_lookback
        self.base_penalty = base_penalty
        self.penalty_growth_rate = penalty_growth_rate
        
        # Create penalty coefficients for different stuck durations
        self.penalty_coeffs = th.zeros(max_lookback + 1)
        for i in range(8, max_lookback + 1):
            # Start with base_penalty at 2 steps, then increase according to growth rate
            self.penalty_coeffs[i] = self.base_penalty * (self.penalty_growth_rate ** (i - 2))
    
    def compute_stuck_penalties(self, positions, mask=None):
        """
        Compute penalties for each agent at each timestep based on how long they've been stuck.
        Args:
            positions: Tensor of shape [batch_size, seq_length, n_agents, pos_dim] containing position info
            mask: Optional mask of shape [batch_size, seq_length, n_agents] for valid timesteps
        Returns:
            penalties: Tensor of shape [batch_size, seq_length, 1] with penalty values aggregated across agents
        """
        # Limit to first 500 timesteps if needed
        if positions.size(1) > 500:
            positions = positions[:, :500, :, :]
            
        batch_size, seq_length, n_agents, pos_dim = positions.shape
        device = positions.device
        self.penalty_coeffs = self.penalty_coeffs.to(device)
        
        # Initialize penalties for each agent
        per_agent_penalties = th.zeros(batch_size, seq_length, n_agents, device=device)
        stuck_count = th.ones(batch_size, n_agents, device=device)

        
        # For each timestep, look back and check how long agent has been stuck
        for t in range(1, seq_length):
            # We can only look back up to t or max_lookback, whichever is smaller
            is_same = th.all(positions[:, t] == positions[:, t - 1], dim=-1)
            stuck_count = th.where(is_same, stuck_count + 1, th.ones_like(stuck_count,  device=device))
            stuck_count = th.clamp(stuck_count, 0, self.max_lookback)
            per_agent_penalties[:, t] = self.penalty_coeffs[stuck_count.long()]
        # Apply mask if provided
        if mask is not None:
            per_agent_penalties = per_agent_penalties * mask
        
        # Aggregate penalties across agents (mean) and reshape to match rewards
        #penalties = per_agent_penalties.mean(dim=2, keepdim=True)  # [batch_size, seq_length, 1]
        
        return per_agent_penalties

    def shape_rewards(self, rewards, positions, mask=None):
        """
        Apply stuck penalties to rewards.
        Args:
            rewards: Tensor of shape [batch_size, seq_length, 1]
            positions: Tensor of shape [batch_size, seq_length, n_agents, pos_dim]
            mask: Optional mask for valid timesteps
        Returns:
            shaped_rewards: Original rewards minus penalties, same shape as rewards
        """
        penalties = self.compute_stuck_penalties(positions, mask)
        
        # Make sure penalties match rewards shape
        if rewards.shape != penalties.shape:
            raise ValueError(f"Rewards shape {rewards.shape} doesn't match penalties shape {penalties.shape}")
            
        # Subtract penalties from rewards
        shaped_rewards = rewards - penalties
        
        return shaped_rewards
    


class OscillationPenaltyRewardShaper:
    def __init__(self, lookback=20, osc_coeff=0.01, growth_rate=1.5):
        """
        Penalizes agents for revisiting the same position multiple times in the last `lookback` timesteps.
        Args:
            lookback: Number of timesteps to look back when computing oscillation penalty.
            osc_coeff: Coefficient for penalty scaling based on visit counts > 1.
        """
        self.lookback = lookback
        self.osc_coeff = osc_coeff
        self.growth_rate = growth_rate

    def compute_oscillation_penalties(self, positions, mask=None):
        batch_size, seq_length, n_agents, pos_dim = positions.shape
        device = positions.device
        per_agent_penalties = th.zeros(batch_size, seq_length, n_agents, device=device)

        for t in range(seq_length):
            # Determine sliding window
            start_t = max(0, t - self.lookback)
            history = positions[:, start_t:t+1]  # [B, T_lookback, A, pos_dim]
            history = history.transpose(1, 2)    # [B, A, T_lookback, pos_dim]

            counts = th.zeros(batch_size, n_agents, device=device)

            for b in range(batch_size):
                for a in range(n_agents):
                    pos_seq = []
                    prev_pos = None

                    # Build sequence without consecutive repeats
                    for i in range(history.size(2)):
                        curr_pos = tuple(history[b, a, i].tolist())
                        if curr_pos != prev_pos:
                            pos_seq.append(curr_pos)
                        prev_pos = curr_pos

                    # Count revisits
                    pos_counter = {}
                    for pos in pos_seq:
                        pos_counter[pos] = pos_counter.get(pos, 0) + 1

                    # Oscillation penalty only for positions visited >1 (excluding stuck)
                    for c in pos_counter.values():
                        if c > 1:
                            counts[b, a] += self.osc_coeff * (self.growth_rate ** (c - 2))

            per_agent_penalties[:, t] = counts

        if mask is not None:
            per_agent_penalties *= mask

        return per_agent_penalties


    def shape_rewards(self, rewards, positions, mask=None):
        """
        Apply oscillation penalties to rewards.
        Args:
            rewards: Tensor of shape [batch_size, seq_length, n_agents]
            positions: Tensor of shape [batch_size, seq_length, n_agents, pos_dim]
            mask: Optional mask of shape [batch_size, seq_length, n_agents]
        Returns:
            shaped_rewards: Tensor with penalties subtracted
        """
        penalties = self.compute_oscillation_penalties(positions, mask)

        if rewards.shape != penalties.shape:
            raise ValueError(f"Shape mismatch: rewards {rewards.shape} vs penalties {penalties.shape}")
        
        shaped_rewards = rewards - penalties
        return shaped_rewards
