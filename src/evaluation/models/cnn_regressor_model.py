import torch.nn as nn


class CNNRegressor(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (N, T, 1) -> (N, 1, T)
        x = self.net(x)
        x = x.squeeze(-1)      # (N, 32)
        return self.fc(x).squeeze(1)
