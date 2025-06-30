import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── Building blocks ────────────────────────────
class ResidualBlock(nn.Module):
    """Simple 2-layer residual 1-D Conv block."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, C, T]
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)                                 # residual sum


class AttnPool1D(nn.Module):
    """Additive attention pooling over the temporal axis."""
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(channels))

    def forward(
        self,
        x: torch.Tensor,          # [B, C, T]
        mask: torch.Tensor | None = None  # [B, T] True = keep
    ) -> torch.Tensor:           # → [B, C]
        # scores: [B, T]
        scores = (x.permute(0, 2, 1) @ self.query) / math.sqrt(x.size(1))
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)                   # [B, T]
        return (x * weights.unsqueeze(1)).sum(dim=2)          # weighted sum


# ───────────────────────────── Main network ─────────────────────────────
class DifficultyRegressor(nn.Module):
    """
    1-D CNN + Residual blocks + Attention pooling for dive-difficulty regression.

    Input
    -----
    tracks  : FloatTensor [B, T, N, 2]  – padded sequence of (x,y) points.
    lengths : Optional LongTensor [B]   – true sequence lengths to mask padding.

    Output
    ------
    pred    : FloatTensor [B] – predicted difficulty score.
    """
    def __init__(
        self,
        num_points: int = 1000,
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_ch = num_points * 2

        self.initial = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden, kernel_size=3, dilation=2 ** (i % 3))
              for i in range(n_blocks)]
        )

        self.pool     = AttnPool1D(hidden)
        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)
        )

        # He initialisation for Convs & Linears
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        tracks: torch.Tensor,        # [B, T, N, 2]
        lengths: torch.Tensor | None = None
    ) -> torch.Tensor:               # [B]
        B, T, N, _ = tracks.shape
        x = tracks.reshape(B, T, N * 2).permute(0, 2, 1)   # [B, in_ch, T]

        x = self.initial(x)          # [B, hidden, T]
        x = self.res_blocks(x)       # [B, hidden, T]

        mask = None
        if lengths is not None:
            arange = torch.arange(T, device=tracks.device).unsqueeze(0)  # [1, T]
            mask = arange < lengths.unsqueeze(1)                         # [B, T]

        x = self.pool(x, mask)          # [B, hidden]
        x = self.dropout(x)
        return self.head(x).squeeze(-1) # [B]


# ────────────────────────────── sanity test ─────────────────────────────
if __name__ == "__main__":
    dummy = torch.randn(4, 120, 1000, 2)     # [B, T, N, 2]
    lengths = torch.tensor([120, 110, 90, 75])
    model = DifficultyRegressor()
    out = model(dummy, lengths)
    print("output shape:", out.shape)        # expected: torch.Size([4])