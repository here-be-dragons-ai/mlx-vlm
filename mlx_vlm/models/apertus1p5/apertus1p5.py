from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..apertus.language import LanguageModel
from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .audio import AudioTokenizer
from .config import ModelConfig
from .vision import VisionTokenizer


class Model(nn.Module):
    """Apertus 1.5 with text, image and audio input.

    Images and audio enter Apertus 1.5 as discrete codes. The vision tokenizer
    turns each 16x16 patch into a codebook index and the WavTokenizer codec
    turns each 600 samples of 24 kHz audio into one; the codes are shifted by
    image_token_offset / audio_token_offset and read through the ordinary
    token embedding in place of the ``<|image|>`` / ``<|audio|>`` placeholders.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        if config.vision_tokenizer_config is not None:
            self.vision_tokenizer = VisionTokenizer(config.vision_tokenizer_config)
        if config.audio_tokenizer_config is not None:
            self.audio_tokenizer = AudioTokenizer(config.audio_tokenizer_config)

    def _image_ids(self, pixel_values: mx.array, image_sizes) -> mx.array:
        """Vocabulary ids of all image codes, image by image."""
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None]
        sizes = np.array(image_sizes).reshape(-1, 2)
        # NCHW from the processor -> NHWC for MLX convolutions.
        pixel_values = pixel_values.transpose(0, 2, 3, 1)
        ids = []
        for image, (height, width) in zip(pixel_values, sizes):
            # Encode each image at its true size: the encoder has global
            # attention, so batch padding would change the codes.
            image = image[None, : int(height), : int(width)]
            codes = self.vision_tokenizer.encode(image)
            ids.append(codes.reshape(-1) + self.config.image_token_offset)
        return mx.concatenate(ids)

    def _audio_ids(self, input_features: mx.array, feature_attention_mask) -> mx.array:
        """Vocabulary ids of all audio codes, clip by clip."""
        if input_features.ndim == 3:  # (clips, 1, samples)
            input_features = input_features[:, 0]
        lengths = np.array(feature_attention_mask).reshape(len(input_features), -1)
        lengths = lengths.sum(axis=-1)
        ids = []
        # The codec is small; on the CPU its codes match the reference exactly,
        # on the GPU about 1% of them flip.
        with mx.stream(mx.cpu):
            for clip, length in zip(input_features, lengths):
                codes = self.audio_tokenizer.encode(clip[None, : int(length)])
                ids.append(codes.reshape(-1) + self.config.audio_token_offset)
            ids = mx.concatenate(ids)
            mx.eval(ids)
        return ids

    @staticmethod
    def _fill(ids: np.ndarray, token_id: int, values: mx.array, kind: str):
        values = np.array(values)
        placeholders = ids == token_id
        if int(placeholders.sum()) != values.size:
            raise ValueError(
                f"{int(placeholders.sum())} {kind} placeholders in the prompt "
                f"but {values.size} {kind} codes."
            )
        ids[placeholders] = values

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        input_features = kwargs.get("input_features")
        if pixel_values is not None or input_features is not None:
            ids = np.array(input_ids)
            if pixel_values is not None:
                if getattr(self, "vision_tokenizer", None) is None:
                    raise ValueError(
                        "This checkpoint has no vision tokenizer weights; "
                        "convert it again from the original release."
                    )
                image_sizes = kwargs.get("image_sizes")
                if image_sizes is None:
                    raise ValueError("image_sizes is required with pixel_values.")
                image_ids = self._image_ids(pixel_values, image_sizes)
                self._fill(ids, self.config.image_token_id, image_ids, "image")
            if input_features is not None:
                if getattr(self, "audio_tokenizer", None) is None:
                    raise ValueError(
                        "This checkpoint has no audio tokenizer weights; "
                        "convert it again from the original release."
                    )
                mask = kwargs.get("feature_attention_mask")
                if mask is None:
                    raise ValueError(
                        "feature_attention_mask is required with input_features."
                    )
                audio_ids = self._audio_ids(input_features, mask)
                self._fill(ids, self.config.audio_token_id, audio_ids, "audio")
            input_ids = mx.array(ids)
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if pixel_values is not None or kwargs.get("input_features") is not None:
            embeds = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
            for key in ("image_sizes", "input_features", "feature_attention_mask"):
                kwargs.pop(key, None)
            return self.language_model(
                input_ids, cache=cache, inputs_embeds=embeds.inputs_embeds, **kwargs
            )
        return self.language_model(input_ids, cache=cache, **kwargs)

    @property
    def cast_predicate(self):
        # Code assignment is an argmax over a codebook; keep the tokenizers in
        # float32 when the rest of the model is cast down.
        return lambda k: not k.startswith(("vision_tokenizer", "audio_tokenizer"))

    def sanitize(self, weights):
        if any(k.startswith("language_model.") for k in weights):
            return weights
        text, vision, audio = {}, {}, {}
        for k, v in weights.items():
            if k.startswith("model.language_model."):
                text["model." + k[len("model.language_model.") :]] = v
            elif k == "lm_head.weight":
                text[k] = v
            elif k.startswith("model.vision_tokenizer."):
                vision[k[len("model.vision_tokenizer.") :]] = v
            elif k.startswith("model.audio_tokenizer."):
                audio[k[len("model.audio_tokenizer.") :]] = v
        text = self.language_model.sanitize(text)
        out = {f"language_model.{k}": v for k, v in text.items()}
        if getattr(self, "vision_tokenizer", None) is not None:
            vision = VisionTokenizer.sanitize(vision)
            out.update({f"vision_tokenizer.{k}": v for k, v in vision.items()})
        if getattr(self, "audio_tokenizer", None) is not None:
            audio = AudioTokenizer.sanitize(audio)
            out.update({f"audio_tokenizer.{k}": v for k, v in audio.items()})
        return out

    @property
    def layers(self):
        return self.language_model.layers
