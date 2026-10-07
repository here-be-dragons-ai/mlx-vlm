"""Apertus 1.5 vision tokenizer (encode-only EMU3.5/IBQ tokenizer).

Images become a grid of discrete codes, one per 16x16 patch, which the
language model reads as ordinary vocabulary ids shifted by
``image_token_offset``. Code assignment is an argmax over a 131k codebook, so
the tokenizer must run in float32: half precision flips a noticeable share of
codes.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import VisionTokenizerConfig


def _swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True, pytorch_compatible=True)


# Input elements per convolution tile. MLX convolutions need roughly four
# times their output in scratch memory, so a 3x3 convolution over a
# full-resolution image would need about 9 GB; tiles keep that to a few
# hundred MB.
_TILE_ELEMENTS = 1 << 25


def _conv(conv: nn.Conv2d, x: mx.array, pad=(0, 0, 0, 0), stride: int = 1):
    """conv(pad(x)) with pad = (top, bottom, left, right), computed in row
    tiles when the input is large. Each output row only depends on its own
    input rows, so tiling does not change the result."""
    B, H, W, C = x.shape
    k = conv.weight.shape[1]
    top, bottom, left, right = pad

    def run(tile, pad_top, pad_bottom):
        tile = mx.pad(tile, [(0, 0), (pad_top, pad_bottom), (left, right), (0, 0)])
        y = mx.conv2d(tile, conv.weight, stride=stride)
        return y + conv.bias if "bias" in conv else y

    if B * H * W * C <= _TILE_ELEMENTS:
        return run(x, top, bottom)

    out_rows = (H + top + bottom - k) // stride + 1
    rows = max(1, _TILE_ELEMENTS // (B * W * C * stride))
    tiles = []
    for o0 in range(0, out_rows, rows):
        o1 = min(out_rows, o0 + rows)
        i0 = o0 * stride - top
        i1 = (o1 - 1) * stride + k - top
        tile = x[:, max(0, i0) : min(H, i1)]
        y = run(tile, max(0, -i0), max(0, i1 - H))
        mx.eval(y)
        tiles.append(y)
    return mx.concatenate(tiles, axis=1)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = _swish(self.norm1(x))
        mx.eval(h)
        h = _conv(self.conv1, h, (1, 1, 1, 1))
        h = _swish(self.norm2(h))
        mx.eval(h)
        h = _conv(self.conv2, h, (1, 1, 1, 1))
        if self.in_channels != self.out_channels:
            x = _conv(self.nin_shortcut, x)
        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention over all spatial positions."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, H * W, C)
        k = self.k(h).reshape(B, H * W, C)
        v = self.v(h).reshape(B, H * W, C)
        weights = mx.softmax((q @ k.transpose(0, 2, 1)) * C**-0.5, axis=-1)
        h = (weights @ v).reshape(B, H, W, C)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # Asymmetric right/bottom padding, as in the original VQGAN encoder.
        return _conv(self.conv, x, (0, 1, 0, 1), stride=2)


class DownLevel(nn.Module):
    def __init__(self, blocks: List[ResnetBlock], attn: List[AttnBlock], downsample):
        super().__init__()
        self.block = blocks
        self.attn = attn
        if downsample is not None:
            self.downsample = downsample


class MidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)


class Encoder(nn.Module):
    def __init__(self, config: VisionTokenizerConfig):
        super().__init__()
        multipliers = list(config.channel_multiplier)
        self.num_resolutions = len(multipliers)
        self.num_res_blocks = config.num_res_blocks
        self.conv_in = nn.Conv2d(
            config.in_channels, config.base_channels, 3, stride=1, padding=1
        )

        # Attention placement follows the reference resolution, not the input.
        resolution = config.resolution
        in_multipliers = [1] + multipliers
        self.down = []
        for level in range(self.num_resolutions):
            block_in = config.base_channels * in_multipliers[level]
            block_out = config.base_channels * multipliers[level]
            blocks, attn = [], []
            for _ in range(self.num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if resolution in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            downsample = None
            if level != self.num_resolutions - 1:
                downsample = Downsample(block_in)
                resolution //= 2
            self.down.append(DownLevel(blocks, attn, downsample))

        self.mid = MidBlock(block_in)
        self.norm_out = _group_norm(block_in)
        self.conv_out = nn.Conv2d(
            block_in, config.latent_channels, 3, stride=1, padding=1
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Evaluating after every block keeps only one block's temporaries
        # alive; a lazy graph over a full-resolution image needs tens of GB.
        h = _conv(self.conv_in, x, (1, 1, 1, 1))
        mx.eval(h)
        for level, down in enumerate(self.down):
            for i, block in enumerate(down.block):
                h = block(h)
                if down.attn:
                    h = down.attn[i](h)
                mx.eval(h)
            if level != self.num_resolutions - 1:
                h = down.downsample(h)
                mx.eval(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return _conv(self.conv_out, _swish(self.norm_out(h)), (1, 1, 1, 1))


class Codebook(nn.Module):
    """Plain weight holder, deliberately not an nn.Embedding so that model
    quantization never touches it."""

    def __init__(self, codebook_size: int, embed_dim: int):
        super().__init__()
        self.weight = mx.zeros((codebook_size, embed_dim))


class VectorQuantizer(nn.Module):
    """IBQ inference path: dot-product argmax over the codebook."""

    def __init__(self, config: VisionTokenizerConfig):
        super().__init__()
        self.embedding = Codebook(config.codebook_size, config.embed_dim)

    def __call__(self, h: mx.array) -> mx.array:
        B, H, W, D = h.shape
        logits = h.reshape(B, H * W, D) @ self.embedding.weight.T
        return mx.argmax(logits, axis=-1).reshape(B, H, W)


class VisionTokenizer(nn.Module):
    def __init__(self, config: VisionTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.quantize = VectorQuantizer(config)

    def encode(self, pixel_values: mx.array) -> mx.array:
        """(B, H, W, 3) in [-1, 1], sides multiples of 16 -> (B, H/16, W/16) codes."""
        h = self.encoder(pixel_values.astype(mx.float32))
        return self.quantize(self.quant_conv(h))

    @staticmethod
    def sanitize(weights):
        # PyTorch convolutions are (out, in, kh, kw); MLX wants (out, kh, kw, in).
        return {
            k: (v.transpose(0, 2, 3, 1) if v.ndim == 4 else v)
            for k, v in weights.items()
        }
