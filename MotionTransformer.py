import torch
import torch.nn as nn
import torch.nn.functional as F

class PointSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        H = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # each: [B*N, N, dim]
        q, k, v = map(lambda t: t.view(B, N, H, C // H).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(out)


class MotionTransformer(nn.Module):
    def __init__(
        self,
        num_points=1000,
        input_dim=3,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,  # Deeper transformer
        num_classes=29,
        max_frames=512,
        dropout=0.3,
        pooling="cls"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.pooling = pooling

        self.embedding = nn.Linear(input_dim, hidden_dim)


        self.point_attn = nn.Sequential(
            PointSelfAttention(hidden_dim, heads=4),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


        self.positional_encoding = nn.Parameter(torch.randn(1, max_frames + 1, hidden_dim))


        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

  
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):  # x: [B, T, N, 3]
        B, T, N, D = x.shape
        x = x.view(B * T, N, D)
        x = self.embedding(x)  

        
        residual = x
        x = self.point_attn(x)
        x = x + residual  
        x = x.mean(dim=1)  

     
        x = x.view(B, T, self.hidden_dim)

  
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  

        
        pe = self.positional_encoding[:, :x.size(1), :]
        x = x + pe

        
        x = self.transformer(x)  

        
        x = x[:, 0]

        return self.classifier(x) 