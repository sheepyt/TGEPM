import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels=1):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def patchify(self, images, patch_size):
        b, c, h, w = images.shape
        assert h % patch_size == 0 and w % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (h // patch_size) * (w // patch_size)

        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(b, c, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(b, num_patches, -1)
        return patches

    def forward(self, x):
        patches = self.patchify(x, self.patch_size)
        return patches


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PMGM(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, patch_size=16):
        super(PMGM, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)

        self.transformer_block1 = TransformerBlock(embed_dim, num_heads)
        self.transformer_block2 = TransformerBlock(embed_dim, num_heads)

        self.smoothing_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, tensor):
        padded_tensor = F.pad(tensor, (1, 1, 1, 1), mode='reflect')

        grad_x = F.conv2d(padded_tensor, self.sobel_x, padding=0)
        grad_y = F.conv2d(padded_tensor, self.sobel_y, padding=0)

        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        B, C, H, W = grad_mag.shape
        x = self.patch_embed(grad_mag)

        x = self.transformer_block1(x)
        x = self.transformer_block2(x)

        x = x.view(B, -1, self.patch_size, self.patch_size, 1)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        x = x.view(B, 1, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, 1, H, W)
        x = self.smoothing_conv(x)

        pro_map = (x - x.min()) / (x.max() - x.min())

        return pro_map


class BN_init_zero(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BN_init_zero, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                 track_running_stats=track_running_stats)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)


class APGM(nn.Module):
    def __init__(self, in_channel, stride):
        super(APGM, self).__init__()
        self.conv_beta = nn.Sequential(
            nn.Conv2d(1, in_channel, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0, bias=False),
            BN_init_zero(in_channel, eps=1e-3, momentum=0.01),
        )

    def forward(self, img, beta):  
        beta = self.conv_beta(beta)
        output_w = img * beta
        output = self.fusion(output_w) + img
        return output
