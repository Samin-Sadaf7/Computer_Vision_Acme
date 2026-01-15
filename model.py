import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class TemporalJerseyNet(nn.Module):
    def __init__(self):
        super().__init__()       
        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)

        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = 576

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.digit1_head = nn.Linear(512, 10)
        self.digit2_head = nn.Linear(512, 10)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        feats = self.cnn(x)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)

        gru_out, _ = self.gru(feats)
        temporal_feat = gru_out.mean(dim=1)

        d1 = self.digit1_head(temporal_feat)
        d2 = self.digit2_head(temporal_feat)

        return d1, d2
