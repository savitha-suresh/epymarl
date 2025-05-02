import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return x


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.pos_enc = PositionalEncoding(args.hidden_dim, max_len=args.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim, 
            nhead=args.n_heads, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=args.n_layers)

        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_random_fault(self):
        pass
    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # inputs: (seq_len, batch_size, input_dim)
        x = F.relu(self.fc1(inputs))  # (seq_len, batch_size, hidden_dim)
        x = self.pos_enc(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # (seq_len, batch_size, hidden_dim)

        # Use the last token's output as summary representation
        summary = x[-1]  # (batch_size, hidden_dim)
        q = self.fc2(summary)  # (batch_size, n_actions)
        return q, None

