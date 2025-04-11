import torch
import torch.nn as nn
import math


class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        x_ln = self.ln1(x)
        attn_output, _ = self.self_attention(x_ln, x_ln, x_ln)
        x = x + attn_output

        x = x + self.mlp(self.ln2(x))

        return x


class VisionEncoder(nn.Module):
    def __init__(
        self,
        d_model=768,
        img_size=224,
        patch_size=16,
        n_channels=3,
        n_heads=12,
        n_layers=12,
        emb_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            n_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [
                VisionTransformer(d_model, n_heads, mlp_ratio=4, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_final = nn.LayerNorm(d_model)

        self.projection = nn.Linear(d_model, emb_dim)

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln_final(x)

        x = x[:, 0]

        x = self.projection(x)

        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)

        return x
