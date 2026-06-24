"""Central configuration. Every shape symbol from README §2 lives here as a
dataclass field so agents never hardcode dimensions. Import `Config` and pass
sub-configs down; do not redefine these constants elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Tokenizer / chunking
# --------------------------------------------------------------------------- #
@dataclass
class TokenizerConfig:
    name: str = "facebook/nllb-200-distilled-600M"  # any HF subword tokenizer; pick one and FREEZE the choice
    vocab_size: int = -1            # filled at load time from the tokenizer (== V)
    pad_id: int = -1                # filled at load time
    bos_id: int = -1
    eos_id: int = -1
    mask_id: int = -1               # a dedicated [MASK] id for CMLM; add as special token if absent


@dataclass
class ChunkConfig:
    segmenter: str = "syntok"       # {"syntok", "spacy", "nltk"} — sentence segmentation backend
    max_tokens: int = 64            # == L. Hard upper bound per chunk.
    min_tokens: int = 4             # merge runt fragments below this
    band_overlong: str = "split"    # {"split", "drop"} policy for sentences > L tokens
    lang: str = "en"                # segmenter language; for translation, run per-side


# --------------------------------------------------------------------------- #
# Codec (Stage A)
# --------------------------------------------------------------------------- #
@dataclass
class CodecConfig:
    # --- build vs buy (README §6) ---
    use_pretrained_codec: bool = False     # False = train LatentCodec; True = SonarCodecAdapter
    sonar_encoder: str = "text_sonar_basic_encoder"
    sonar_decoder: str = "text_sonar_basic_decoder"

    # --- latent geometry ---
    latent_dim: int = 1024          # == d
    latents_per_chunk: int = 16     # == q (set 1 for single-vector / SONAR-style)

    # --- backbone ---
    d_model: int = 768
    n_heads: int = 12
    enc_layers: int = 6
    dec_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    max_tokens: int = 64            # == L (kept in sync with ChunkConfig.max_tokens)

    # --- VAE objective ---
    beta_max: float = 1.0           # final KL weight after annealing
    beta_warmup_steps: int = 20_000 # linear β: 0 → beta_max
    free_bits: float = 0.5          # per-dimension nats below which KL is not penalized
    cmlm_mask_low: float = 0.0      # CMLM mask ratio sampled ~ U(low, high] per batch
    cmlm_mask_high: float = 1.0

    # --- iterative decode (Mask-Predict) ---
    decode_iters: int = 4           # refinement passes at inference


# --------------------------------------------------------------------------- #
# Predictor (Stage B)
# --------------------------------------------------------------------------- #
@dataclass
class PredictorConfig:
    latent_dim: int = 1024          # == d (must match CodecConfig.latent_dim)
    latents_per_chunk: int = 16     # == q (must match CodecConfig.latents_per_chunk)
    n_ctx_chunks: int = 32          # == N_ctx (context canvas)
    n_tgt_chunks: int = 32          # == M (target canvas)

    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    ffn_mult: int = 4
    dropout: float = 0.0            # diffusion/flow models usually train without dropout

    time_embed_dim: int = 256       # sinusoidal timestep embedding width

    # --- flow matching / sampling ---
    cfg_dropout: float = 0.10       # prob. of dropping context → null embedding (enables CFG)
    sample_steps: int = 16          # ODE integration steps at inference (K)
    guidance_scale: float = 2.0     # classifier-free guidance weight (1.0 = off)
    ode_solver: str = "euler"       # {"euler", "midpoint", "rk4"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    codec_corpus_paths: list[str] = field(default_factory=list)   # Phase-1 raw text shards
    pair_paths: list[str] = field(default_factory=list)           # Phase-2 (prompt, response) sources
    pair_format: str = "jsonl"      # {"jsonl", "tsv", "hf"}; jsonl rows: {"prompt": str, "response": str}
    latent_cache_dir: str = "./cache/latents"                     # memmap of precomputed Phase-2 latents
    num_workers: int = 8
    scale_sample_size: int = 50_000 # chunks used by compute_latent_scale


# --------------------------------------------------------------------------- #
# Training (shared + per-phase)
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 4_000
    max_steps: int = 500_000
    grad_clip: float = 1.0
    batch_size: int = 256
    amp_dtype: str = "bfloat16"     # {"bfloat16", "float16", "float32"}
    ema_decay: float = 0.9999       # EMA of weights (critical for the predictor)
    ckpt_every: int = 5_000
    val_every: int = 2_000


@dataclass
class TrainConfig:
    phase1: OptimConfig = field(default_factory=OptimConfig)        # codec
    phase2: OptimConfig = field(default_factory=lambda: OptimConfig(lr=1e-4))   # predictor
    phase3: OptimConfig = field(default_factory=lambda: OptimConfig(lr=1e-5, max_steps=20_000))  # finetune
    out_dir: str = "./runs"
    seed: int = 0


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    max_response_chunks: int = 32   # cap, ≤ M
    ood_gate: bool = False          # if True, refuse/fallback when ood_score exceeds threshold
    ood_threshold: float = 0.0      # set from training-latent statistics
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Top-level aggregate
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)

    # computed after Phase 1 and the codec is frozen (README §7). Shape [d] or scalar.
    latent_scale: Optional[list[float]] = None

    def validate(self) -> None:
        """AGENT TASK: assert cross-config consistency (latent_dim, q, L match
        across codec/predictor/chunk; M ≤ n_tgt_chunks; mask_id is set; etc.)."""
        raise NotImplementedError("AGENT: implement cross-field consistency checks")