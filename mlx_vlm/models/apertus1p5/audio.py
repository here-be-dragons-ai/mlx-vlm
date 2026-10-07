"""Apertus 1.5 audio tokenizer (encode-only WavTokenizer).

24 kHz mono audio becomes one discrete code per 600 samples (40 per second):
a SEANet convolutional encoder with an LSTM, followed by a nearest-neighbour
lookup in a 4096-entry codebook. The codes are shifted by
``audio_token_offset`` and read through the ordinary token embedding. Like
the vision tokenizer it must run in float32.
"""

from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn

from .config import AudioTokenizerConfig


def _reflect_pad(x: mx.array, left: int, right: int) -> mx.array:
    """Reflect padding along time for (B, L, C); like the reference, short
    inputs get extra zeros on the right first so the reflection fits."""
    length = x.shape[1]
    extra = max(0, max(left, right) - length + 1)
    if extra:
        x = mx.pad(x, [(0, 0), (0, extra), (0, 0)])
        length += extra
    parts = []
    if left:
        parts.append(x[:, 1 : left + 1][:, ::-1])
    parts.append(x)
    if right:
        parts.append(x[:, length - right - 1 : length - 1][:, ::-1])
    x = mx.concatenate(parts, axis=1)
    return x[:, : x.shape[1] - extra] if extra else x


class SConv1d(nn.Module):
    """Non-causal Conv1d with SEANet padding (weight norm folded at load)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.pad_mode = pad_mode

    def __call__(self, x: mx.array) -> mx.array:
        length = x.shape[1]
        padding_total = self.kernel - self.stride
        n_frames = -(-(length - self.kernel + padding_total) // self.stride)
        extra = n_frames * self.stride + self.kernel - padding_total - length
        right = padding_total // 2
        left = padding_total - right
        if self.pad_mode == "reflect":
            x = _reflect_pad(x, left, right + extra)
        else:
            x = mx.pad(x, [(0, 0), (left, right + extra), (0, 0)])
        return self.conv(x)


class Elu(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.elu(x)


class ResnetBlock(nn.Module):
    def __init__(self, config: AudioTokenizerConfig, dim: int, dilations: List[int]):
        super().__init__()
        hidden = dim // config.compress
        kernel_sizes = (config.residual_kernel_size, 1)
        self.block = []
        for i, (kernel, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_ch = dim if i == 0 else hidden
            out_ch = dim if i == len(kernel_sizes) - 1 else hidden
            self.block += [
                Elu(),
                SConv1d(
                    in_ch, out_ch, kernel, dilation=dilation, pad_mode=config.pad_mode
                ),
            ]
        self.shortcut = SConv1d(dim, dim, 1, pad_mode=config.pad_mode)

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.block:
            h = layer(h)
        return self.shortcut(x) + h


class Lstm(nn.Module):
    """Stacked LSTM with a residual connection, as in SEANet."""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.lstm = [nn.LSTM(dim, dim) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.lstm:
            h, _ = layer(h)
        return h + x


class Encoder(nn.Module):
    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        pad = config.pad_mode
        layers = [
            SConv1d(
                config.audio_channels,
                config.num_filters,
                config.kernel_size,
                pad_mode=pad,
            )
        ]
        scale = 1
        for ratio in reversed(config.upsampling_ratios):
            dim = scale * config.num_filters
            for j in range(config.num_residual_layers):
                layers.append(
                    ResnetBlock(config, dim, [config.dilation_growth_rate**j, 1])
                )
            layers += [
                Elu(),
                SConv1d(dim, dim * 2, ratio * 2, stride=ratio, pad_mode=pad),
            ]
            scale *= 2
        dim = scale * config.num_filters
        layers += [
            Lstm(dim, config.num_lstm_layers),
            Elu(),
            SConv1d(dim, config.hidden_size, config.last_kernel_size, pad_mode=pad),
        ]
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class Codebook(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.embed = mx.zeros((codebook_size, codebook_dim))


class VectorQuantizer(nn.Module):
    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        self.codebook = Codebook(config.codebook_size, config.codebook_dim)

    def __call__(self, h: mx.array) -> mx.array:
        # Nearest neighbour by Euclidean distance; |h|^2 is constant per row.
        embed = self.codebook.embed
        dist = (embed * embed).sum(-1) - 2 * (h @ embed.T)
        return mx.argmin(dist, axis=-1)


class AudioTokenizer(nn.Module):
    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.quantizer = VectorQuantizer(config)

    @property
    def hop_length(self) -> int:
        hop = 1
        for ratio in self.config.upsampling_ratios:
            hop *= ratio
        return hop

    def encode(self, audio: mx.array) -> mx.array:
        """(B, samples) at 24 kHz -> (B, ceil(samples / hop_length)) codes."""
        h = self.encoder(audio.astype(mx.float32)[..., None])
        return self.quantizer(h)

    @staticmethod
    def sanitize(weights: Dict[str, Any]) -> Dict[str, Any]:
        """Fold weight norm, transpose convolutions, merge LSTM biases and drop
        the decoder and codebook training buffers."""
        out = {}
        for k, v in weights.items():
            if not k.startswith(("encoder.", "quantizer.codebook.embed")):
                continue
            if k.startswith("quantizer.codebook.embed_avg"):
                continue
            if k.endswith("parametrizations.weight.original0"):
                prefix = k[: -len("parametrizations.weight.original0")]
                g = v
                w = weights[prefix + "parametrizations.weight.original1"]
                norm = mx.sqrt((w * w).sum(axis=(1, 2), keepdims=True))
                out[prefix + "weight"] = (g * w / norm).transpose(0, 2, 1)
            elif k.endswith("parametrizations.weight.original1"):
                continue
            elif ".lstm.weight_ih_l" in k:
                out[k.replace(".weight_ih_l", ".") + ".Wx"] = v
            elif ".lstm.weight_hh_l" in k:
                out[k.replace(".weight_hh_l", ".") + ".Wh"] = v
            elif ".lstm.bias_ih_l" in k:
                layer = k.split("bias_ih_l")[1]
                hh = weights[k.replace("bias_ih_l", "bias_hh_l")]
                out[k.replace(".bias_ih_l" + layer, "." + layer) + ".bias"] = v + hh
            elif ".lstm.bias_hh_l" in k:
                continue
            else:
                out[k] = v
        return out
