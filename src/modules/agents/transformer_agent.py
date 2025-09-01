import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import gc
import math

def get_pos_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    div_term = 1.0 / (10000.0 ** div_term)  # (d_model // 2,)

    # Repeat to interleave for even and odd dimensions
    div_term = div_term.repeat_interleave(2).unsqueeze(0)  # (1, d_model)

    encoding = position @ div_term  # (seq_len, d_model)

    encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
    encoding[:, 1::2] = torch.cos(encoding[:, 1::2])

    return encoding  # (seq_len, d_model)



class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sqrt_dk = math.sqrt(d_model)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k_e = nn.Linear(d_model, d_model, bias=False)
        self.w_k_r = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.final = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU()
        )
        self.pos_enc = get_pos_encoding(self.max_seq_len, self.d_model)
        self.u_param = nn.Parameter(torch.randn(1, 1, n_heads, self.d_head))
        self.v_param = nn.Parameter(torch.randn(1, 1, n_heads, self.d_head))

    def rel_enc_shift(self, arr):
        # arr: (batch_size, num_heads, l, m)
        batch_size, num_heads, l, m = arr.size()
        zeros = torch.zeros(batch_size, num_heads, l, 1, device=arr.device, dtype=arr.dtype)
        arr = torch.cat([arr, zeros], dim=-1)
        arr = arr.view(batch_size, num_heads, -1)
        arr = arr[:, :, l-1: -1]
        arr = arr.view(batch_size, num_heads, l, m)
        return arr

    def forward(self, query, key, value, attn_mask):
        
         
        batch_size, full_len, _ = value.size()
          # (batch_size, seq_len, d_model)
        _, seq_len, _ = query.size()
        rel_enc = self.pos_enc[:full_len, :]
        rel_enc = rel_enc.to(query.device)
        rel_enc = torch.flip(rel_enc, dims=[0]) 

        q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.w_k_e(key).view(batch_size, full_len, self.n_heads, self.d_head)
        v = self.w_v(value).view(batch_size, full_len, self.n_heads, self.d_head)

        A_C = torch.einsum('bsnd,bfnd->bnsf', q + self.u_param, k)

        Q = self.w_k_r(rel_enc)  # (full_len, d_model)
        #print("Q.shape", Q.shape, "full_len", full_len, "n_heads", self.n_heads, "d_head", self.d_head)
        Q = Q.view(full_len, self.n_heads, self.d_head)
        B_D_hat = torch.einsum('bsnd,fnd->bnsf', q + self.v_param, Q)
        B_D = self.rel_enc_shift(B_D_hat)

        attention_score = (A_C + B_D) / self.sqrt_dk
        #print("Attention score shape:", attention_score.shape, "attn_mask shape", attn_mask.shape)
        attention_score += attn_mask

        attention_weights = F.softmax(attention_score, dim=-1)
        max_weights = attention_weights.max(dim=-1).values.max(dim=-1).values
        attention_loss = max_weights.mean()

        attention_output = torch.einsum('bnsf,bfnd->bsnd', attention_weights, v)
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.d_model)

        output = self.final(attention_output)
        return output, attention_weights, attention_loss



class GRUGate(nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
        """
        super(GRUGate, self).__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.tensor} -- First input
            y {torch.tensor} -- Second input
        Returns:
            {torch.tensor} -- Output
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)
    


class BehaviorEmbedding(nn.Module):
    """
    Learns behavior embeddings for each agent to identify similar behaviors
    and filter out faulty (no-op) agents
    """
    def __init__(self, obs_dim, behavior_dim, n_agents, threshold=0.7):
        super(BehaviorEmbedding, self).__init__()
        self.behavior_dim = behavior_dim
        self.n_agents = n_agents
        self.threshold = threshold
        
        # Network to encode observations into behavior embeddings
        self.behavior_encoder = nn.Sequential(
            nn.Linear(obs_dim, behavior_dim * 2),
            nn.ReLU(),
            nn.Linear(behavior_dim * 2, behavior_dim),
            nn.LayerNorm(behavior_dim)
        )
        
        # Activity classifier to detect no-op agents
        self.activity_classifier = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [no-op, active]
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs):
        # obs: (batch_size * n_agents, seq_len, obs_dim)
        batch_size_agents, seq_len, obs_dim = obs.shape
        batch_size = batch_size_agents // self.n_agents
        
        # Get behavior embeddings
        behavior_emb = self.behavior_encoder(obs)  # (batch_size * n_agents, seq_len, behavior_dim)
        
        # Get activity predictions
        activity_scores = self.activity_classifier(obs)  # (batch_size * n_agents, seq_len, 2)
        activity_prob = activity_scores[:, :, 1]  # Probability of being active
        
        # Reshape for agent-wise operations
        behavior_emb = behavior_emb.view(batch_size, self.n_agents, seq_len, -1)
        activity_prob = activity_prob.view(batch_size, self.n_agents, seq_len)
        
        return behavior_emb, activity_prob
    
    def compute_behavior_losses(self, behavior_emb, activity_prob, agent_labels=None, actions=None):
        """
        Compute losses to encourage behavioral diversity and clear separation
        
        Args:
            behavior_emb: (batch_size, n_agents, seq_len, behavior_dim)
            activity_prob: (batch_size, n_agents, seq_len)
            agent_labels: Optional ground truth labels (0=faulty, 1=healthy) - for debugging only
            actions: Agent actions to infer activity from (batch_size, n_agents, seq_len)
        """
        batch_size, n_agents, seq_len, behavior_dim = behavior_emb.shape
        
        # Average behavior embedding over sequence
        avg_behavior = behavior_emb.mean(dim=2)  # (batch_size, n_agents, behavior_dim)
        avg_activity = activity_prob.mean(dim=2)  # (batch_size, n_agents)
        
        losses = {}
        
        # 1. Anti-Collapse Loss: Prevent all healthy agents from becoming identical
        # BUT allow role specialization by only penalizing EXACT similarity (> 0.95)
        norm_behavior = F.normalize(avg_behavior, p=2, dim=-1)
        similarity_matrix = torch.bmm(norm_behavior, norm_behavior.transpose(1, 2))
        
        # Create mask to exclude self-similarity and focus on active agents
        eye_mask = torch.eye(n_agents, device=similarity_matrix.device).unsqueeze(0)
        eye_mask = eye_mask.expand(batch_size, -1, -1)
        
        # Use predicted activity as proxy for healthy agents (self-supervised)
        healthy_mask = (avg_activity > self.threshold).float()
        healthy_pair_mask = healthy_mask.unsqueeze(1) * healthy_mask.unsqueeze(2)
        valid_mask = healthy_pair_mask * (1.0 - eye_mask)
        
        # Anti-collapse loss: only penalize VERY high similarity (near identical agents)
        collapse_threshold = 0.95  # Only penalize if similarity > 0.95
        collapse_mask = (similarity_matrix > collapse_threshold).float()
        anti_collapse_loss = (similarity_matrix * collapse_mask * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        losses['anti_collapse'] = anti_collapse_loss
        
        # 2. Self-Supervised Separation: Use action patterns to infer agent types
        if actions is not None:
            # Detect no-op agents based on action patterns
            # Assuming action 0 is no-op or you have a specific no-op action
            
            action_variance = torch.var(actions.float(), dim=2)  # Variance across time
            #print("action shape", actions.shape, "variance.shape", action_variance.shape)
            action_activity = action_variance.mean(dim=0)  # Average across action dims if multi-dim
            
            # Create pseudo-labels based on action patterns
            inferred_faulty = (action_activity < 0.1).float()  # Low variance = likely faulty
            inferred_healthy = 1.0 - inferred_faulty
            
            # Separation loss based on inferred labels
            healthy_behavior = avg_behavior * inferred_healthy.unsqueeze(-1)
            faulty_behavior = avg_behavior * inferred_faulty.unsqueeze(-1)
            
            healthy_count = inferred_healthy.sum(dim=0, keepdim=True)
            faulty_count = inferred_faulty.sum(dim=0, keepdim=True)
            
            # Only compute if we have both types
            mask = (healthy_count > 0) & (faulty_count > 0)
            
            if mask.any():
                healthy_centroid = healthy_behavior.sum(dim=0) / (healthy_count.unsqueeze(-1) + 1e-8)
                faulty_centroid = faulty_behavior.sum(dim=0) / (faulty_count.unsqueeze(-1) + 1e-8)
                
                centroid_similarity = F.cosine_similarity(healthy_centroid, faulty_centroid, dim=-1)
                separation_loss = (centroid_similarity * mask.squeeze(-1)).mean()
                losses['separation'] = separation_loss
            else:
                losses['separation'] = torch.tensor(0.0, device=behavior_emb.device)
        
        # 3. Activity Prediction Loss: Learn to predict activity from behavior
        if actions is not None:
            # Ground truth activity based on action variance
            action_variance = torch.var(actions.float(), dim=2)  # (batch_size, n_agents)
            gt_activity = (action_variance > 0.1).float()  # Binary active/inactive
            
            # Loss for activity classifier
            activity_pred_loss = F.binary_cross_entropy(avg_activity, gt_activity)
            losses['activity_prediction'] = activity_pred_loss
        
        # 4. Supervised losses (only if ground truth labels provided - for debugging)
        if agent_labels is not None:
            # Supervised separation loss
            healthy_behavior_gt = avg_behavior * agent_labels.unsqueeze(-1)
            faulty_behavior_gt = avg_behavior * (1 - agent_labels).unsqueeze(-1)
            
            healthy_count_gt = agent_labels.sum(dim=1, keepdim=True)
            faulty_count_gt = (1 - agent_labels).sum(dim=1, keepdim=True)
            
            if (healthy_count_gt > 0).all() and (faulty_count_gt > 0).all():
                healthy_centroid_gt = healthy_behavior_gt.sum(dim=1) / (healthy_count_gt.unsqueeze(-1) + 1e-8)
                faulty_centroid_gt = faulty_behavior_gt.sum(dim=1) / (faulty_count_gt.unsqueeze(-1) + 1e-8)
                
                supervised_separation = F.cosine_similarity(healthy_centroid_gt, faulty_centroid_gt, dim=-1).mean()
                losses['supervised_separation'] = supervised_separation
            
            # Supervised activity loss
            gt_activity_labels = agent_labels.float()
            supervised_activity_loss = F.binary_cross_entropy(avg_activity, gt_activity_labels)
            losses['supervised_activity'] = supervised_activity_loss
        
        return losses
    
    def compute_similarity_mask(self, behavior_emb, activity_prob):
        """
        Compute similarity mask between agents based on behavior embeddings
        and filter out inactive (no-op) agents
        """
        batch_size, n_agents, seq_len, behavior_dim = behavior_emb.shape
        #print("Computing similarity mask with behavior_emb shape:", behavior_emb.shape,
        #      "activity_prob shape:", activity_prob.shape)
        # Average behavior embedding over sequence for each agent
        avg_behavior = behavior_emb.mean(dim=2)  # (batch_size, n_agents, behavior_dim)
        avg_activity = activity_prob.mean(dim=2)  # (batch_size, n_agents)
        
        # Compute cosine similarity between all agent pairs
        norm_behavior = F.normalize(avg_behavior, p=2, dim=-1)
        similarity_matrix = torch.bmm(norm_behavior, norm_behavior.transpose(1, 2))
        # (batch_size, n_agents, n_agents)
        
        # Create self-exclusion mask (diagonal = 0, agent can't attend to itself)
        eye_mask = torch.eye(n_agents, device=similarity_matrix.device).unsqueeze(0)
        eye_mask = eye_mask.expand(batch_size, -1, -1)
        self_exclusion_mask = 1.0 - eye_mask  # 0 on diagonal, 1 elsewhere
        
        # Create activity mask - only attend to active agents
        # activity_mask = (avg_activity > self.threshold).float()  # (batch_size, n_agents)
        # # For cross-attention: row agent attends to column agents
        # # So we only care if the TARGET agent (column) is active
        # activity_cross_mask = activity_mask.unsqueeze(1).expand(-1, n_agents, -1)
        # (batch_size, n_agents, n_agents) where activity_cross_mask[b, i, j] = activity of agent j
        
        # Combine similarity, activity, and self-exclusion masks
        similarity_threshold = 0.5  # Adjust based on your needs
        similarity_mask = (similarity_matrix > similarity_threshold).float()
        
        # Final mask: can attend to agent j if:
        # 1. Agent j is active (activity_cross_mask)
        # 2. Agent i and j are similar (similarity_mask) 
        # 3. Agent i != j (self_exclusion_mask)
        
        #final_mask = similarity_mask * activity_cross_mask * self_exclusion_mask
        final_mask = similarity_mask * self_exclusion_mask
        
        return final_mask, similarity_matrix

class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block that allows agents to attend to other similar active agents
    """
    def __init__(self, d_model, n_heads, n_agents, max_seq_len):
        super(CrossAttentionBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_agents = n_agents
        self.norm_kv = nn.LayerNorm(d_model)
        
        self.cross_attn = RelativeMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len*self.n_agents
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = GRUGate(d_model, 0.0)
        
    def forward(self, query, key_value, cross_attn_mask=None):
        """
        query: current agent representations (batch_size * n_agents, seq_len, d_model)
        key_value: all agents representations (batch_size * n_agents, kv_seq_len, d_model)
        cross_attn_mask: [B, A, A] mask for which agents can attend to which
        """
        batch_size_nagents, Tq, d_model = query.shape
        batch_size = batch_size_nagents // self.n_agents
        _, Tk, _ = key_value.shape
        A = self.n_agents

        # -------------------------------
        # Interleave agent data (time-major ordering)
        # -------------------------------
        # query: [B*A, Tq, d] -> [B, A, Tq, d] -> [B, Tq, A, d] -> flatten [B, Tq*A, d]
        query = query.view(batch_size, A, Tq, d_model).permute(0, 2, 1, 3).reshape(batch_size, Tq * A, d_model)

        # key/value: [B*A, Tk, d] -> [B, A, Tk, d] -> [B, Tk, A, d] -> flatten [B, Tk*A, d]
        kv_flat = key_value.view(batch_size, A, Tk, d_model).permute(0, 2, 1, 3).reshape(batch_size, Tk * A, d_model)

        # Normalize
        query_norm = self.norm1(query)
        kv_norm = self.norm_kv(kv_flat)

        # -------------------------------
        # Build cross-attention mask (vectorized)
        # -------------------------------
        if cross_attn_mask is not None:
            # cross_attn_mask: [B, A, A] -> expand to token level
            # For each timestep, only allow agent i to attend to allowed agents j
            # Result: [B, Tq*A, Tk*A]
            # B_idx = torch.arange(batch_size, device=query.device)[:, None, None]
            # Tq_idx = torch.arange(Tq, device=query.device)[None, :, None]
            # Tk_idx = torch.arange(Tk, device=query.device)[None, None, :]

            # Vectorized expansion using kron
            # block_diag over timesteps: [Tq*A, Tk*A]
            mask_block = torch.kron(torch.ones(Tq, Tk, device=query.device), cross_attn_mask[0])
            attn_mask = mask_block.unsqueeze(0).expand(batch_size, -1, -1)

            # Add heads dimension
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.n_heads, -1, -1)

            # Convert to additive mask (1=allow, 0=block -> 0/-inf)
            attn_mask = (1.0 - attn_mask).to(query.dtype) * (-1e9)
        else:
            attn_mask = None

        # -------------------------------
        # Cross attention
        # -------------------------------
        cross_out = self.cross_attn(query=query_norm,
                                    key=kv_norm,
                                    value=kv_norm,
                                    attn_mask=attn_mask)[0]

        # -------------------------------
        # Residual connection with gating
        # -------------------------------
        #output = self.gate(query, cross_out)
        # reshape back to [B*A, Tq, d]
        output = cross_out.view(batch_size * A, Tq, d_model)
        return output
    

class EnhancedDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, norm_first, max_seq_len, n_agents):
        super(EnhancedDecoderBlock, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.gate1 = GRUGate(d_model, 0.0)
        self.gate2 = GRUGate(d_model, 0.0)
        self.gate3 = GRUGate(d_model, 0.0)
        
        self.norm_kv = nn.LayerNorm(d_model)
        
        # Self attention (within agent)
        self.self_attn = RelativeMultiHeadAttention(
            d_model=d_model,
            n_heads=nhead,
            max_seq_len=max_seq_len,
        )
        
        # Cross attention (between agents)
        self.cross_attn_block = CrossAttentionBlock(d_model, nhead, n_agents, max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    def forward(self, x, memory, cross_attn_mask=None, attn_mask=None):
        # Self attention with memory
        if memory is not None:
            x_cat = torch.cat([memory, x], dim=1)
        else:
            x_cat = x
            
        self_attn_op = self.self_attn(
            self.norm1(x), self.norm_kv(x_cat), self.norm_kv(x_cat),
            attn_mask=attn_mask
        )[0]
        h1 = self.gate1(x, self_attn_op)
        
        # Cross attention with other agents
        
        cross_attn_op = self.cross_attn_block(
            query=h1,
            key_value=x_cat,
            cross_attn_mask=cross_attn_mask
        )
        h2 = self.gate2(h1, cross_attn_op)
        
        
        # Feed forward
        h2_norm = self.norm3(h2)
        forward = self.ffn(h2_norm)
        out = self.gate3(h2, forward)
        
        return out

class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.input_norm = nn.LayerNorm(args.hidden_dim)
        self.n_layers = args.n_layers
        
        # Behavior embedding module
        self.behavior_embedding = BehaviorEmbedding(
            obs_dim=input_shape,
            behavior_dim=args.hidden_dim // 4,
            n_agents=args.n_agents,
            threshold=0.7
        )
        
        # Enhanced decoder blocks with cross-attention
        self.layers = nn.ModuleList([
            EnhancedDecoderBlock(
                d_model=args.hidden_dim,
                nhead=args.n_heads,
                norm_first=True,
                max_seq_len=self.max_seq_len,
                n_agents=args.n_agents
            ) for _ in range(args.n_layers)
        ])
        
        self.mem_len = 250
        self.output_norm = nn.LayerNorm(args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        self.memories = [None for _ in range(args.n_layers)]

    def init_hidden(self):
        self.memories = [None for _ in range(self.n_layers)]

    def init_memory(self, batch_size):
        return [torch.zeros(batch_size * self.args.n_agents, 0, self.args.hidden_dim,
                           device=next(self.parameters()).device) for _ in self.layers]

    def update_memory(self, memory, hidden_states):
        with torch.no_grad():
            new_memory = []
            for mem, h in zip(memory, hidden_states):
                h = h.detach()
                mem = mem.detach()
                combined = torch.cat([mem, h], dim=1)
                new_memory.append(combined[:, -self.mem_len:].detach())
            return new_memory
        
    def generate_agent_labels(self, batch_size):
        agent_labels = torch.ones(batch_size, self.args.n_agents)
        return agent_labels

    def build_cross_attn_mask(self):
        pass 

    def forward(self, inputs, memory=None, attn_mask=None, actions=None, return_aux_losses=False):
        # inputs: (batch_size * n_agents, seq_len, input_dim)
        batch_size_agents, seq_len, input_dim = inputs.shape
        batch_size = batch_size_agents // self.args.n_agents
        hidden_states = []
        aux_losses = {}
        # Get behavior embeddings and activity predictions
        # behavior_emb, activity_prob = self.behavior_embedding(inputs)
        # cross_attn_mask, similarity_matrix = self.behavior_embedding.compute_similarity_mask(
        #     behavior_emb, activity_prob
        # )
        # print("cross attention mask", cross_attn_mask[0], "similarity matrix", similarity_matrix[0])  
        # agent_labels = self.generate_agent_labels(self.args.batch_size)
        # # Compute auxiliary losses for behavior learning
        # aux_losses = {}
        # if return_aux_losses:
        #     # Reshape actions if provided
        #     actions_reshaped = None
        #     if actions is not None:
        #         actions_reshaped = actions.view(batch_size, self.args.n_agents, -1)
            
        #     aux_losses = self.behavior_embedding.compute_behavior_losses(
        #         behavior_emb, activity_prob, agent_labels, actions_reshaped
        #     )
        cross_attn_mask = self.build_cross_attn_mask()
        #print("Cross_attention mask shape:", cross_attn_mask)
        # Process inputs
        x = F.relu(self.fc1(inputs))
        x = self.input_norm(x)
        
        # Store all agent states for cross-attention
        all_agent_states = x.clone()
        
        for i, layer in enumerate(self.layers):
            mem = None if memory is None else memory[i]
            
            # Reshape cross_attn_mask for current agent
            # agent_idx = torch.arange(batch_size_agents) % self.args.n_agents
            # current_cross_mask = None
            # if cross_attn_mask is not None:
            #     batch_indices = torch.arange(batch_size_agents) // self.args.n_agents
            #     current_cross_mask = cross_attn_mask[batch_indices, agent_idx]
            
            x = layer(
                x=x,
                memory=mem,
                cross_attn_mask=cross_attn_mask,
                attn_mask=attn_mask
            )
            hidden_states.append(x.detach().clone())
            
            # Update all_agent_states for next layer
            all_agent_states = x.clone()

        x = self.output_norm(x)
        q = self.fc2(x)
        
        if return_aux_losses:
            return q, hidden_states, aux_losses
        return q, hidden_states
    
# class TransformerAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(TransformerAgent, self).__init__()
#         self.args = args
#         self.max_seq_len = args.max_seq_len
#         self.fc1 = nn.Linear(input_shape, args.hidden_dim)  # Projects input to model dim
#         self.input_norm = nn.LayerNorm(args.hidden_dim)  # Normalize inputs
#         self.n_layers = args.n_layers
#         # Create decoder blocks without cross-attention (more like GPT architecture)
#         self.layers = nn.ModuleList([DecoderOnlyBlock(
#             d_model=args.hidden_dim,
#             nhead=args.n_heads,
#             norm_first=True,
#             max_seq_len=self.max_seq_len
#         ) for _ in range(args.n_layers)])
#         self.mem_len = 250
        
#         self.output_norm = nn.LayerNorm(args.hidden_dim)
#         self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
#         self.memories = [None for _ in range(args.n_layers)]

#     def init_hidden(self):
#         self.memories = [None for _ in range(self.n_layers)]

#     def init_memory(self, batch_size):
       
#         return [torch.zeros(batch_size * self.args.n_agents, 0, self.args.hidden_dim, 
#                             device=next(self.parameters()).device) for _ in self.layers]

#     def update_memory(self, memory, hidden_states):
#         # Save only last `max_mem_len` timesteps
#         with torch.no_grad():
#             new_memory = []
#             for mem, h in zip(memory, hidden_states):
#                 h= h.detach()
#                 mem = mem.detach()
#                 combined = torch.cat([mem, h], dim=1)
#                 new_memory.append(combined[:, -self.mem_len:].detach())
                
#             return new_memory
        
        
    
#     def forward(self, inputs, memory=None, attn_mask=None):
#         # inputs: (batch_size, seq_len, input_dim)
#         hidden_states = []     
        
        
#         x = inputs
#         # Process inputs
#         x = F.relu(self.fc1(inputs))
#         x = self.input_norm(x)
        
#         for i, layer in enumerate(self.layers):
#             mem = None if memory is None else memory[i]
#             x = layer(x, memory=mem, attn_mask=attn_mask)
#             hidden_states.append(x.detach().clone())

#         x = self.output_norm(x)
#         q = self.fc2(x)
        
#         return q, hidden_states

# # Helper class for GPT-style implementation
# class DecoderOnlyBlock(nn.Module):
#     def __init__(self, d_model, nhead, norm_first, max_seq_len):
#         super(DecoderOnlyBlock, self).__init__()
#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.gate1 = GRUGate(d_model, 0.0)
#         self.gate2 = GRUGate(d_model, 0.0)
#         self.norm_kv = nn.LayerNorm(d_model)

#         self.self_attn = RelativeMultiHeadAttention(
#             d_model=d_model,
#             n_heads=nhead,
#             max_seq_len=max_seq_len,
#         )
        
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU()
#         )
        
        
#     def forward(self, x, memory, attn_mask=None):
        
#         # Concatenate memory (if any) and compute attention
#         if memory is not None:
#             x_cat = torch.cat([memory, x], dim=1)
#         else:
#             x_cat = x
#         attn_op  = self.self_attn(
#             self.norm1(x), self.norm_kv(x_cat), self.norm_kv(x_cat), 
#             attn_mask=attn_mask
#         )[0]
#         h = self.gate1(x, attn_op)
#         h_ = self.norm2(h)
#         forward = self.ffn(h_)
#         out = self.gate2(h, forward)
#         return out