"""Processor for Apertus 1.5 (text, images and audio).

Released transformers versions do not ship ``Apertus1p5Processor`` yet, so
this mirrors the reference processor from swiss-ai/transformers: images are
resized to multiples of 16 within a pixel budget, normalized to [-1, 1], and
each ``<|image|>`` placeholder is expanded into the structured run

    <|img_start|>{H}*{W}<|img_token_start|>{W x <|image|>}<|img_end_of_row|>...<|img_end|>

with one ``<|image|>`` per 16x16 patch. The model swaps those placeholders for
the image codes. Each ``<|audio|>`` placeholder becomes
``<|audio_start|>`` + one ``<|audio|>`` per 600 samples + ``<|audio_end|>``.
"""

from typing import List, Optional

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch


def smart_resize(
    height: int,
    width: int,
    factor: int = 16,
    min_pixels: int = 256 * 256,
    max_pixels: int = 1400 * 1400,
):
    """Clamp the pixel area to [min_pixels, max_pixels] keeping the aspect
    ratio, then round both sides half-up to multiples of ``factor``. Matches
    the reference pipeline, including its int() truncations."""
    target_area = max(min(max_pixels, height * width), min_pixels)
    aspect_ratio = width / height
    new_height = int((target_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + factor // 2) // factor) * factor
    new_width = ((new_width + factor // 2) // factor) * factor
    return max(new_height, factor), max(new_width, factor)


def _flatten_text(message):
    """Join the text parts of a non-user message into one string; the Apertus
    template rejects list content outside user turns."""
    if not isinstance(message, dict) or message.get("role") == "user":
        return message
    content = message.get("content")
    if not isinstance(content, list):
        return message
    text = "".join(
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    )
    return {**message, "content": text}


class Apertus1p5ImageProcessor:
    def __init__(
        self,
        min_pixels: int = 256 * 256,
        max_pixels: int = 1400 * 1400,
        spatial_factor: int = 16,
        **kwargs,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.spatial_factor = spatial_factor

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL image -> float32 (3, H, W) in [-1, 1]."""
        image = image.convert("RGB")
        height, width = smart_resize(
            image.height,
            image.width,
            self.spatial_factor,
            self.min_pixels,
            self.max_pixels,
        )
        if (height, width) != (image.height, image.width):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        # (x - 127.5) / 127.5 is bit-identical to the reference's fused
        # rescale + normalize; other float32 orderings differ in the last bit.
        pixels = np.asarray(image, dtype=np.float32)
        pixels = (pixels - np.float32(127.5)) / np.float32(127.5)
        return pixels.transpose(2, 0, 1)


class Apertus1p5FeatureExtractor:
    """24 kHz mono waveforms, peak-normalized to -3 dBFS as in the reference
    pipeline; one audio code per hop_length samples."""

    def __init__(self, sampling_rate: int = 24000, hop_length: int = 600, **kwargs):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def num_codes(self, num_samples: int) -> int:
        return -(-num_samples // self.hop_length)

    def preprocess(self, audio) -> np.ndarray:
        clip = np.asarray(audio, dtype=np.float32)
        if clip.ndim != 1 or clip.size == 0:
            raise ValueError(f"Expected non-empty mono audio, got shape {clip.shape}.")
        target_peak = 10.0 ** (-3.0 / 20.0)
        return clip * (target_peak / max(float(np.abs(clip).max()), 1e-10))


class Apertus1p5Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    supports_multiple_audio = True

    # transformers infers sub-processors from __init__ parameter names, so the
    # image and audio processors are passed as settings and attached afterwards.
    def __init__(
        self,
        tokenizer,
        chat_template=None,
        image_settings=None,
        audio_settings=None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.image_token = getattr(tokenizer, "image_token", None) or "<|image|>"
        self.boi_token = getattr(tokenizer, "boi_token", None) or "<|img_start|>"
        self.eoi_token = getattr(tokenizer, "eoi_token", None) or "<|img_end|>"
        self.image_wrapper_token = (
            getattr(tokenizer, "image_wrapper_token", None) or "<|img_token_start|>"
        )
        self.eol_token = getattr(tokenizer, "eol_token", None) or "<|img_end_of_row|>"
        self.audio_token = getattr(tokenizer, "audio_token", None) or "<|audio|>"
        self.boa_token = getattr(tokenizer, "boa_token", None) or "<|audio_start|>"
        self.eoa_token = getattr(tokenizer, "eoa_token", None) or "<|audio_end|>"
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        super().__init__(tokenizer, chat_template=self.tokenizer.chat_template)
        self.image_processor = Apertus1p5ImageProcessor(**(image_settings or {}))
        self.feature_extractor = Apertus1p5FeatureExtractor(**(audio_settings or {}))

    @property
    def chat_template(self):
        return getattr(self.tokenizer, "chat_template", None)

    @chat_template.setter
    def chat_template(self, value):
        self.tokenizer.chat_template = value

    def _expand_image(self, grid_height: int, grid_width: int) -> str:
        rows = self.eol_token.join([self.image_token * grid_width] * grid_height)
        return (
            f"{self.boi_token}{grid_height}*{grid_width}"
            f"{self.image_wrapper_token}{rows}{self.eoi_token}"
        )

    @staticmethod
    def _expand(text: List[str], token: str, replacements: List[str]) -> List[str]:
        """Replace each placeholder, in batch order and left to right."""
        count = sum(t.count(token) for t in text)
        if count != len(replacements):
            raise ValueError(
                f"The prompt contains {count} '{token}' placeholders but "
                f"{len(replacements)} inputs were passed."
            )
        it = iter(replacements)
        out = []
        for sample in text:
            parts = sample.split(token)
            out.append(parts[0] + "".join(next(it) + part for part in parts[1:]))
        return out

    def __call__(
        self,
        text=None,
        images: Optional[List[Image.Image]] = None,
        audio=None,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
        text = list(text)
        data = {}

        if images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            factor = self.image_processor.spatial_factor
            arrays = [self.image_processor.preprocess(img) for img in images]
            sizes = [a.shape[1:] for a in arrays]
            text = self._expand(
                text,
                self.image_token,
                [self._expand_image(h // factor, w // factor) for h, w in sizes],
            )
            # Pad to a common size; the model crops each image to image_sizes.
            max_h = max(h for h, _ in sizes)
            max_w = max(w for _, w in sizes)
            pixel_values = np.zeros((len(arrays), 3, max_h, max_w), dtype=np.float32)
            for i, a in enumerate(arrays):
                pixel_values[i, :, : a.shape[1], : a.shape[2]] = a
            data["pixel_values"] = pixel_values
            data["image_sizes"] = np.array(sizes, dtype=np.int64)

        if audio is not None and len(audio) > 0:
            if isinstance(audio, np.ndarray) and audio.ndim == 1:
                audio = [audio]
            clips = [self.feature_extractor.preprocess(a) for a in audio]
            # The <|audio|> runs are inserted after image expansion, so an
            # audio placeholder can never be mistaken for an image one.
            runs = [
                self.boa_token
                + self.audio_token * self.feature_extractor.num_codes(len(c))
                + self.eoa_token
                for c in clips
            ]
            text = self._expand(text, self.audio_token, runs)
            max_len = max(len(c) for c in clips)
            features = np.zeros((len(clips), max_len), dtype=np.float32)
            mask = np.zeros((len(clips), max_len), dtype=np.int32)
            for i, c in enumerate(clips):
                features[i, : len(c)] = c
                mask[i, : len(c)] = 1
            data["input_features"] = features
            data["feature_attention_mask"] = mask

        encoded = self.tokenizer(
            text,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        data["input_ids"] = encoded["input_ids"]
        data["attention_mask"] = encoded["attention_mask"]
        return BatchFeature(data=data, tensor_type=return_tensors)

    def apply_chat_template(self, conversation, *args, **kwargs):
        # mlx-vlm passes every message as a list of parts; only user turns
        # may stay that way.
        if isinstance(conversation, list):
            conversation = [_flatten_text(m) for m in conversation]
        return self.tokenizer.apply_chat_template(conversation, *args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        import json
        from pathlib import Path

        kwargs.pop("trust_remote_code", None)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        image_kwargs, audio_kwargs = {}, {}
        config_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            image_kwargs = {
                k: v
                for k, v in config.get("image_processor", {}).items()
                if k in ("min_pixels", "max_pixels", "spatial_factor")
            }
            audio_kwargs = {
                k: v
                for k, v in config.get("feature_extractor", {}).items()
                if k in ("sampling_rate", "hop_length")
            }
        return cls(tokenizer, image_settings=image_kwargs, audio_settings=audio_kwargs)


install_auto_processor_patch("apertus1p5", Apertus1p5Processor)

__all__ = [
    "Apertus1p5Processor",
    "Apertus1p5ImageProcessor",
    "Apertus1p5FeatureExtractor",
]
