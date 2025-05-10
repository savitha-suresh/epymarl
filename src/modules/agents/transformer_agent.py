import torch
import torch.nn as nn
import torch.nn.functional as F


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=100):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
#         pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
#         pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x: (seq_len, batch_size, d_model)
#         x = x + self.pe[:x.size(0)]
#         return x


# class TransformerAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(TransformerAgent, self).__init__()
#         self.args = args

#         self.fc1 = nn.Linear(input_shape, args.hidden_dim)
#         self.pos_enc = PositionalEncoding(args.hidden_dim, max_len=args.max_seq_len)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=args.hidden_dim, 
#             nhead=args.n_heads, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=args.n_layers)

#         self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

#     def init_random_fault(self):
#         pass
    
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         seq_len = inputs.size(1)
#         causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(inputs.device)  # [T, T]
#         causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')) \
#                                 .masked_fill(causal_mask == 1, float(0.0))
#         x = F.relu(self.fc1(inputs))  
#         x = self.pos_enc(x)  
#         x = self.transformer_encoder(x, mask=causal_mask, is_causal=True)  # (batch_size, seq_len, d_model)

        
#         q = self.fc2(x)
#         return q, None





class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.pos_embedding(positions)  # (batch_size, seq_len, d_model)


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)    # Projects input to model dim
        self.input_norm = nn.LayerNorm(args.hidden_dim)       # Normalize inputs
        self.dropout = nn.Dropout(0.1)

        self.pos_enc = LearnedPositionalEncoding(args.hidden_dim, max_len=self.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True  # Apply LayerNorm before residuals for stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=args.n_layers
        )

        self.output_norm = nn.LayerNorm(args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_random_fault(self):
        pass

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_state=None):
        # inputs: (batch_size, seq_len, input_dim)
        seq_len = inputs.size(1)

        # Causal mask to prevent future attention
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=inputs.device), diagonal=1
        )

        x = F.relu(self.fc1(inputs))        # (B, T, H)
        x = self.input_norm(x)
        x = self.dropout(x)

        x = self.pos_enc(x)                 # Add positional encoding
        x = self.transformer_encoder(x, mask=causal_mask, is_causal=True)  # Residuals + LN handled internally
        x = self.output_norm(x)

        q = self.fc2(x)                     # Predict Q-values
        return q, None
