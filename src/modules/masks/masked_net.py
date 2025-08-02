from torch import nn
class MaskNet(nn.Module):
    def __init__(self, args, obs_dim, hidden_dim):
        super().__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs value in (0, 1)
        )

    def forward(self, obs):  # obs: [B*A, T, obs_dim]
        B, T, D = obs.shape
        out = self.net(obs)  
        return out.view(B, T, 1)