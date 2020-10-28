import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 dropout=0.):
        super().__init__()
        if not hidden_features:
            hidden_features = in_features
        if not out_features:
            out_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.actn = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.actn(x)
        x = self.fc2(x)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1./dim**0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c//self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dot = (q @ k.transpose(-2, -1)) * self.scale
        attn = dot.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.out(x)
        return x


class ImgPatches(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        patches = self.patch_embed(img).flatten(2).transpose(1, 2)
        return patches


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, drop_rate, drop_rate)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*mlp_ratio, dropout=drop_rate)

    def forward(self, x):
        x1 = self.ln1(x)
        x = x + self.attn(x1)
        x2 = self.ln2(x)
        x = x + self.mlp(x2)
        return x


class Transformer(nn.Module):
    def __init__(self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop_rate)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 drop_rate=0.):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError('Image size must be divisible by patch size.')
        num_patches = (img_size//patch_size) ** 2
        self.patch_size = patch_size

        # Image patches and embedding layer
        self.patches = ImgPatches(in_ch, embed_dim, self.patch_size)

        # Embedding for patch position and class
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.2)
        nn.init.trunc_normal_(self.cls_emb, std=0.2)

        self.drop = nn.Dropout(p=drop_rate)
        self.transfomer = Transformer(depth, embed_dim, num_heads,
                                      mlp_ratio, drop_rate)
        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        b = x.shape[0]
        cls_token = self.cls_emb.expand(b, -1, -1)

        x = self.patches(x)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_emb
        x = self.drop(x)
        x = self.transfomer(x)
        x = self.norm(x)
        x = self.out(x[:, 0])
        return x
