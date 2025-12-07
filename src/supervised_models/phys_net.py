import torch
import yaml
from pathlib import Path
import torch.nn as nn

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


class PhysNet(nn.Module):
    def __init__(self, in_channels=9):
        super(PhysNet, self).__init__()

        self.frontend3D = nn.Sequential(
                nn.Conv3d(in_channels, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),

                nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),

                nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride= 1, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
        )

        self.temporal_attention = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.rPPG_head = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        print(f"Shape of x is: {x.shape}")
        feats = self.frontend3D(x)
        B, C, T, H2, W2 = feats.shape

        print(f"Dimension is: {B}, {C}, {T}, {H2}, {W2}")

        feats = feats.view(B, C, T, -1).mean(-1)

        att = self.temporal_attention(feats)
        feats = feats * att

        feats = self.temporal_conv(feats)

        rppg = self.rPPG_head(feats)

        return rppg
