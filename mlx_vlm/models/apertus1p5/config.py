from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..apertus.config import ModelConfig as ApertusConfig
from ..base import BaseModelConfig


@dataclass
class TextConfig(ApertusConfig):
    @classmethod
    def from_dict(cls, params):
        # transformers 5 nests rope_theta and the llama3 scaling under
        # rope_parameters; the Apertus text model still reads the old fields.
        params = dict(params)
        rope_parameters = params.get("rope_parameters") or {}
        if "rope_theta" in rope_parameters:
            params.setdefault("rope_theta", rope_parameters["rope_theta"])
        if rope_parameters.get("rope_type", "default") != "default":
            params.setdefault(
                "rope_scaling",
                {k: v for k, v in rope_parameters.items() if k != "rope_theta"},
            )
        return super().from_dict(params)


@dataclass
class VisionTokenizerConfig(BaseModelConfig):
    model_type: str = "apertus1p5_vision_tokenizer"
    in_channels: int = 3
    base_channels: int = 256
    channel_multiplier: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    num_res_blocks: int = 4
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    resolution: int = 256
    latent_channels: int = 256
    embed_dim: int = 256
    codebook_size: int = 131072
    spatial_scale_factor: int = 16


@dataclass
class AudioTokenizerConfig(BaseModelConfig):
    model_type: str = "wavtokenizer"
    audio_channels: int = 1
    num_filters: int = 32
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    num_residual_layers: int = 1
    dilation_growth_rate: int = 2
    compress: int = 2
    upsampling_ratios: List[int] = field(default_factory=lambda: [6, 5, 5, 4])
    num_lstm_layers: int = 2
    hidden_size: int = 512
    codebook_size: int = 4096
    codebook_dim: int = 512
    pad_mode: str = "reflect"
    sampling_rate: int = 24000


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    model_type: str = "apertus1p5"
    image_token_offset: int = 131272
    audio_token_offset: int = 262344
    image_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    vision_tokenizer_config: Optional[VisionTokenizerConfig] = None
    audio_tokenizer_config: Optional[AudioTokenizerConfig] = None

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.audio_tokenizer_config, dict):
            self.audio_tokenizer_config = AudioTokenizerConfig.from_dict(
                self.audio_tokenizer_config
            )
        if isinstance(self.vision_tokenizer_config, dict):
            self.vision_tokenizer_config = VisionTokenizerConfig.from_dict(
                self.vision_tokenizer_config
            )
