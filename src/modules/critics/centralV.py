# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn
import torch.nn.functional as F
# Define number of quantiles for Distributional Critic
from torch.nn.utils.parametrizations import spectral_norm
N_QUANTILES = 5 # Tune this, typically 5 to 10

class CentralVCritic(nn.Module):
    def __init__(self, scheme, args, context_shape):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.n_quantiles = N_QUANTILES # Use defined number of quantiles
        hidden_dim = args.hidden_dim
        
        # NOTE: context_shape is the dimension of the features concatenated 
        # for *each agent* in the batch (delta_t, fault_mask, agent_mask).
        # Assuming delta_t is the first element (index 0) of the context tensor
        
        # Input shape calculation remains the same
        input_shape = self._get_input_shape(scheme) 
        self.output_type = "v"

        # --- SHARED BACKBONE ---
        self.fc1 = spectral_norm(nn.Linear(input_shape, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim) 
        
        # --- NEW: Temporal Context Embedding Layer (for Latent Cross) ---
        # The delta_t feature is a scalar (1 dimension)
        self.context_embed = nn.Linear(1, hidden_dim)
        
        # --- FAULT GATING MECHANISM (Input shape remains hidden_dim + 1) ---
        self.gate_fc = nn.Linear(hidden_dim + 1, hidden_dim)
        
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim) 

        # --- QUANTILE OUTPUT HEAD ---
        self.quantile_head = nn.Linear(hidden_dim, self.n_quantiles)
        
        # ... (tau and tau_hat registration remains the same) ...
        self.register_buffer('tau', th.arange(0, self.n_quantiles + 1, dtype=th.float32) / self.n_quantiles)
        self.register_buffer('tau_hat', ((self.tau[1:] + self.tau[:-1]) / 2.0).view(1, 1, 1, 1, -1))
        

    def forward(self, batch, t=None, faulty_indices={}):
        inputs, bs, max_t, context_features = self._build_inputs(batch, t=t, faulty_indices=faulty_indices)
        
        # Process inputs: Zero out features of faulty agents and flatten
        flat_inputs, flat_mask, bs, max_t, n_agents = self._process_inputs(inputs, context_features)

        # --- Extract Delta_T (Time to Next Transition) ---
        # Assuming delta_t is the first context feature (index 0) of the 
        # original batch["context"] tensor (before concatenation/flattening in _build_inputs)
        # Note: Your _process_inputs handles the complex flattening. 
        # We need the context feature corresponding to delta_t:
        
        # context_features shape: [bs, max_t, n_agents, context_dim=6]
        # Delta_t is at index 0 [2]
        delta_t_feature = context_features[..., 0:1] # Shape: [bs, max_t, n_agents, 1]
        
        # Flatten for input to FC layer: [bs*max_t*n_agents, 1]
        flat_delta_t = delta_t_feature.view(bs * max_t * n_agents, 1)


        # --- 1. Shared Backbone FC1 ---
        x = F.relu(self.ln1(self.fc1(flat_inputs))) # Output: [..., hidden_dim]
        
        # --- NEW: Latent Cross Modulation --- [1]
        # a. Embed the temporal context feature
        context_emb = F.relu(self.context_embed(flat_delta_t)) # Output: [..., hidden_dim]
        
        # b. Apply Element-wise Multiplication (The "Latent Cross") [1]
        # This allows the temporal feature to modulate the hidden state
        x = x * context_emb 
        
        # --- 2. Explicit Fault Gating (This uses the modulated features) ---
        gated_input = th.cat([x, flat_mask], dim=-1) # Output: [..., hidden_dim + 1]
        x = F.relu(self.gate_fc(gated_input))
        
        # --- 3. Shared Backbone FC2 ---
        x = F.relu(self.ln2(self.fc2(x))) # Output: [..., hidden_dim]
        
        # --- 4. Quantile Head output ---
        z_quantiles = self.quantile_head(x) 
        
        # Reshape and return (following the previous structure)
        z_quantiles = z_quantiles.view(bs, max_t, self.n_agents, self.n_quantiles) 
        mean_v = z_quantiles.mean(dim=-1, keepdim=True) 

        # Return the new output signature, including the mask
        flat_mask_reshaped = flat_mask.view(bs, max_t, self.n_agents, 1)
        return mean_v, z_quantiles, flat_mask_reshaped
    
    def _process_inputs(self, inputs, context_features):
        """
        Extracts the fault mask and applies it to the individual agent observations.
        
        Args:
            inputs (torch.Tensor): The concatenated input tensor [bs, max_t, n_agents, input_shape].
            context_features (torch.Tensor): The context features [bs, max_t, n_agents, 6].
            
        Returns:
            torch.Tensor: The processed inputs with masked features.
            torch.Tensor: The extracted fault mask [bs*max_t, 1].
        """
        bs, max_t, n_agents, total_input_shape = inputs.shape
        
        # --- 1. Extract Fault Mask ---
        # fault_mask: [bs, max_t, n_agents, 1]
        # Assuming fault mask is at index 1 of the last dimension of context_features (size 6)
        fault_mask = context_features[..., 1:2] 
        
        # Reshape fault_mask to match the total inputs for feature masking
        # fault_mask_broadcast: [bs, max_t, n_agents, total_input_shape]
        fault_mask_broadcast = fault_mask.repeat(1, 1, 1, total_input_shape)

        # --- 2. Apply Feature Zeroing (Critical Step) ---
        # The mask is 0 for normal, 1 for faulty. We want to keep normal features (1-0=1)
        # and zero out faulty features (1-1=0).
        # We assume the last features (context) contain the fault signal, but we want 
        # to mask ALL features of the faulty agent.
        
        # To mask ALL features, we need the initial concatenated input features (o_i, last_a, agent_id, context)
        # Masking factor: (1.0 - fault_mask_broadcast). If agent is faulty (1), factor is 0, features are zeroed.
        masked_inputs = inputs * (1.0 - fault_mask_broadcast)
        
        # --- 3. Prepare for Centralized Network ---
        # Flatten the batch and time dimensions: [bs*max_t*n_agents, input_shape]
        flat_inputs = masked_inputs.view(bs * max_t * n_agents, total_input_shape)
        
        # Flatten the mask for explicit use: [bs*max_t*n_agents, 1]
        flat_mask = fault_mask.view(bs * max_t * n_agents, 1)

        return flat_inputs, flat_mask, bs, max_t, n_agents
    
    def _build_inputs(self, batch, t=None, faulty_indices={}):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        context_features = batch["context"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1) 
        inputs.append(context_features)

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t, context_features

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        input_shape += 6
        return input_shape
