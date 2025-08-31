# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F
import torch

class ClusterSimilarityNet(nn.Module):
    def __init__(self, obs_dim, embed_dim, n_agents, threshold=0.7):
        super().__init__()
        self.n_agents = n_agents
        self.threshold = threshold
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, obs, labels=None):
        batch_size_agents, obs_dim = obs.shape
        batch_size = batch_size_agents // self.n_agents

        # Encode → embeddings
        emb = self.encoder(obs)                           # (batch*n_agents, embed_dim)
        emb = F.normalize(emb, p=2, dim=-1)               # cosine normalize
        emb = emb.view(batch_size, self.n_agents, -1)

        # Similarity matrix
        sim_matrix = torch.matmul(emb, emb.transpose(1, 2))  # (batch, n_agents, n_agents)
        eye = torch.eye(self.n_agents, device=obs.device).unsqueeze(0)
        sim_matrix = sim_matrix * (1.0 - eye)

        # Cluster-based mask
        attn_mask = (sim_matrix > self.threshold).float()

        # Loss (if labels given)
        aux_loss = None
        if labels is not None:
            aux_loss = self.cluster_loss(sim_matrix, labels)

        return emb, attn_mask, aux_loss
    
    def cluster_loss(self, sim_matrix, labels, margin=0.5):
        batch_size, n_agents, _ = sim_matrix.shape
        
        # Pairwise ground truth: 1 if same label, 0 if different
        label_sim = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()
        eye = torch.eye(n_agents, device=labels.device).unsqueeze(0)
        label_sim = label_sim * (1.0 - eye)  # no self-pairs
        
        # Loss for similar pairs (same label) → want sim close to 1
        pos_loss = (1 - sim_matrix) * label_sim
        
        # Loss for dissimilar pairs (different label) → want sim below margin
        neg_loss = F.relu(sim_matrix - margin) * (1.0 - label_sim)
        
        # Average
        loss = (pos_loss.sum() + neg_loss.sum()) / (label_sim.sum() + (1.0 - label_sim).sum() + 1e-8)
        return loss

    
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.similarity_net = ClusterSimilarityNet(
            input_shape,
            args.hidden_dim // 4,
            n_agents=args.n_agents
        )

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    

    def generate_agent_labels(self, batch_size):
        agent_labels = torch.ones(batch_size, self.args.n_agents)
        return agent_labels

    def forward(self, inputs, hidden_state, actions=None):
        agent_labels = self.generate_agent_labels(self.args.batch_size)
        active_prob, attn_mask, aux_loss = self.similarity_net(inputs, agent_labels)
       
            # Reshape actions if provided
        
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h, aux_loss

