__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from collections import OrderedDict
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from models.ct_clip_utils import exists
from functools import partial
from torch import einsum


def params(module):
    return sum(p.numel() for p in module.parameters())

def visualize(images, x, y):
    c, h, w = images.shape
    a = torch.zeros(int(h * x), int(w * y))
    k = 0
    m = 0
    for i in range(x):
        l = 0
        for j in range(y):
            a[k:k + h, l:l + w] = images[m]
            l += w
            m += 1

        k += h
    plt.figure()
    plt.imshow(a)



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    




class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True, 
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class double_conv_block(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, *args, **kwargs):
        super().__init__()
        self.conv1 = conv_block(in_features=in_features, out_features=out_features1, *args, **kwargs)
        self.conv2 = conv_block(in_features=out_features1, out_features=out_features2, *args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class double_conv_block_a(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, norm1, norm2, act1, act2, *args, **kwargs):
        super().__init__()
        self.conv1 = conv_block(in_features=in_features, out_features=out_features1, norm_type=norm1, activation=act1, *args, **kwargs)
        self.conv2 = conv_block(in_features=out_features1, out_features=out_features2, norm_type=norm2, activation=act2, *args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class depthwise_conv_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1), 
                padding=(1, 1), 
                dilation=(1, 1),
                groups=None, 
                norm_type='bn',
                activation=True, 
                use_bias=True,
                pointwise=False, 
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation, 
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features, 
                                        out_features, 
                                        kernel_size=(1, 1), 
                                        stride=(1, 1), 
                                        padding=(0, 0),
                                        dilation=(1, 1), 
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class double_depthwise_convblock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features1,
                 out_features2,
                 kernels_per_layer=1,
                 normalization=None,
                 activation=None):
        super().__init__()
        if normalization is None:
            normalization = [True, True]
        if activation is None:
            activation = [True, True]
        self.block1 = depthwise_conv_block(in_features,
                                      out_features1,
                                      kernels_per_layer=kernels_per_layer,
                                      normalization=normalization[0],
                                      activation=activation[0])
        self.block2 = depthwise_conv_block(out_features1,
                                      out_features2,
                                      kernels_per_layer=kernels_per_layer,
                                      normalization=normalization[1],
                                      activation=activation[1])

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class transpose_conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(2, 2),
                 padding=(0, 0),
                 out_padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True, 
                 ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_features,
                                        out_channels=out_features,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=out_padding,
                                        dilation=dilation,
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)

        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class Upconv(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                activation=True,
                norm_type='bn', 
                scale=(2, 2)) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, 
                              mode='bilinear', 
                              align_corners=True)
        self.conv = conv_block(in_features=in_features, 
                                out_features=out_features, 
                                norm_type=norm_type, 
                                activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class bn_relu(nn.Module):
    def __init__(self, features) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU()
    def forward(self ,x):
        return self.relu(self.bn(x))
    
class SqueezeExciteBlock(nn.Module):
    def __init__(self, in_features, reduction:int=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_features, int(in_features // reduction), bias=False),
                                nn.ReLU(),
                                nn.Linear(int(in_features // reduction), in_features, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim, norm_type='bn'):
        super().__init__()
        if norm_type == 'gn':
            self.norm1 = nn.GroupNorm(32 if (input_encoder >= 32 and input_encoder % 32 == 0) else input_encoder,
                                      input_encoder)
            self.norm2 = nn.GroupNorm(32 if (input_decoder >= 32 and input_decoder % 32 == 0) else input_decoder,
                                      input_decoder)
            self.norm3 = nn.GroupNorm(32 if (output_dim >= 32 and output_dim % 32 == 0) else output_dim,
                                      output_dim)

        if norm_type == 'bn':
            self.norm1 = nn.BatchNorm2d(input_encoder)
            self.norm2 = nn.BatchNorm2d(input_decoder)
            self.norm3 = nn.BatchNorm2d(output_dim)
        
        else:
            self.norm1, self.norm2, self.norm3 = nn.Identity(), nn.Identity(), nn.Identity()

        self.conv_encoder = nn.Sequential(
            self.norm1,
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            self.norm2,
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            self.norm3,
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2

class ResConv(nn.Module):
    def __init__(self, in_features, out_features, stride=(1, 1)):
        super().__init__()
        self.conv = nn.Sequential(bn_relu(in_features),
                                  nn.Conv2d(in_channels=in_features, 
                                            out_channels=out_features, 
                                            kernel_size=(3, 3), 
                                            padding=(1, 1), 
                                            stride=stride),
                                    bn_relu(out_features), 
                                  nn.Conv2d(in_channels=out_features, 
                                            out_channels=out_features, 
                                            kernel_size=(3, 3), 
                                            padding=(1, 1), 
                                            stride=(1, 1))                                     
                                  )
        self.skip = nn.Conv2d(in_channels=in_features, 
                              out_channels=out_features, 
                              kernel_size=(1, 1), 
                              padding=(0, 0), 
                              stride=stride)


    def forward(self, x):
        return self.conv(x) + self.skip(x)



class rec_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                norm_type='bn', 
                activation=True,
                t=2):
        super().__init__()
        self.t = t
        self.conv = conv_block(in_features=in_features, 
                               out_features=out_features, 
                               norm_type=norm_type, 
                               activation=activation)

    def forward(self, x):
        x1 = self.conv(x)
        for _ in range(self.t):     
            x1 = self.conv(x + x1)
        return x1


class rrcnn_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                norm_type='bn', 
                activation=True, 
                t=2):
        super().__init__()
        self.conv = conv_block(in_features=in_features, 
                              out_features=out_features, 
                              kernel_size=(1, 1), 
                              padding=(0, 0), 
                              norm_type=None, 
                              activation=False)
        self.block = nn.Sequential(
            rec_block(in_features=out_features,
                      out_features=out_features,
                      t=t, 
                      norm_type=norm_type, 
                      activation=activation),
            rec_block(in_features=out_features,
                      out_features=out_features,
                      t=t, 
                      norm_type=None, 
                      activation=False)
                              )
        self.norm = nn.BatchNorm2d(out_features)
        self.norm_c = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x1 = self.norm_c(x)
        x1 = self.relu(x1)
        x1 = self.block(x1)
        xs = x + x1
        x = self.norm(xs)
        x = self.relu(x)
        return x, xs

class ASPP(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation
            )
        self.block2 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation            
            )
        self.block3 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation            
            )
        self.block4 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation            
            )

        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x = x1 + x2 + x3 + x4
        x = self.out(x)
        return x

class DoubleASPP(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                norm_type='bn', 
                activation=True, 
                rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1), 
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation, 
            )

        self.block2 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation, 
            use_bias=False
            )
        self.block3 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation, 
            use_bias=False
            )
        self.block4 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation, 
            use_bias=False

            )
        self.block5 = conv_block(
            in_features=in_features,
            out_features=out_features,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation, 
            use_bias=False            
            )

        self.out = conv_block(
            in_features=out_features * 5,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            use_bias=False
            )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = self.block5(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.out(x)
        return x

# class GELU(nn.Module):
#     def forward(self, x):
#         return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor        

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b) 