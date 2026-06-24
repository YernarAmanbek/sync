"""latent_text_gen — conditional text generation in sentence-latent space.

See README.md for the architecture spec, shape glossary, build order, and the
agent task index. Public API:"""

from .config import Config
from .codec import CodecInterface, LatentCodec, SonarCodecAdapter, build_codec
from .predictor import (
    CountHead,
    FlowMatchingPredictor,
    FlowSampler,
    flow_matching_target,
    ood_score,
)
from .data import (
    Chunker,
    Tokenizer,
    CodecChunkDataset,
    PairLatentDataset,
    compute_latent_scale,
)
from .training import train_codec, train_predictor, finetune_joint, freeze_and_scale
from .generate import TextGenerator

__all__ = [
    "Config",
    "CodecInterface",
    "LatentCodec",
    "SonarCodecAdapter",
    "build_codec",
    "FlowMatchingPredictor",
    "CountHead",
    "FlowSampler",
    "flow_matching_target",
    "ood_score",
    "Chunker",
    "Tokenizer",
    "CodecChunkDataset",
    "PairLatentDataset",
    "compute_latent_scale",
    "train_codec",
    "train_predictor",
    "finetune_joint",
    "freeze_and_scale",
    "TextGenerator",
]