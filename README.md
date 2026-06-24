# latent_text_gen — architecture blueprint
 
Conditional text generation that runs **entirely in sentence-latent space**:
the model never predicts tokens autoregressively. A frozen VAE codec defines a
smooth, decodable latent space; a flow-matching predictor maps prompt-latents to
response-latents *generatively* (so it models the distribution of valid
responses instead of their blurry mean); a non-autoregressive (NAT) decoder
renders each predicted latent back into a sentence in parallel.
 
This document is the contract. Each `.py` file is a self-contained unit of work
with fully specified interfaces (exact tensor shapes, loss formulas). Bodies are
stubs (`raise NotImplementedError("AGENT: ...")`); agents implement against the
signatures and docstrings without changing them.
 
---
 
## 1. The three trained phases
 
```
                          ┌─────────────── STAGE A: frozen codec ───────────────┐
  prompt str ──chunk──►  tokens ──► [VAEEncoder] ──► context latents  C          │
                                                          │                       │
                          STAGE B: generative predictor   ▼                       │
  noise ──────────────────────────────► [FlowMatchingPredictor] ──► target Z      │
                                              (cross-attends to C)    │           │
                          ┌──────────────── STAGE A again ────────────▼───────────┘
  response str ◄──join──  chunks ◄── [NATDecoder] (parallel, CMLM) ◄── Z
```
 
- **Phase 1** (`training.train_codec`): self-supervised VAE on raw sentences →
  freeze. Defines the latent space. *No task labels.*
- **Phase 2** (`training.train_predictor`): flow matching on (prompt, response)
  latent pairs produced by the **frozen** codec. *The only task-specific model.*
- **Phase 3** (`training.finetune_joint`, optional/last/light): unfreeze decoder,
  backprop token CE through sampled latents to tighten predictor↔decoder
  coupling. Only if the predictor samples slightly off-manifold latents.
---
 
## 2. Shape glossary (used verbatim in every docstring)
 
| symbol | meaning | default |
|---|---|---|
| `B` | batch size | — |
| `V` | tokenizer vocab size | from tokenizer |
| `L` | max tokens per chunk (sentence) | 64 |
| `d` | latent vector width | 1024 |
| `q` | latent vectors per chunk (Perceiver bottleneck; 1 = single-vector) | 16 |
| `d_model` | transformer hidden width (codec & predictor backbones) | 768 |
| `n` | actual prompt chunks (variable) | — |
| `m` | actual response chunks (variable) | — |
| `N_ctx` | max prompt chunks (context canvas) | 32 |
| `M` | max response chunks (target canvas) | 32 |
 
Key latent tensors:
- per chunk: `z` → `[B, q, d]`
- context (prompt) sequence: `C` → `[B, N_ctx, q, d]`, flattened to `[B, N_ctx*q, d]` for the predictor
- target (response) sequence: `Z` → `[B, M, q, d]`, flattened to `[B, M*q, d]`
**Flattening convention (predictor):** `[B, M, q, d] → [B, M*q, d]`, row-major
(chunk-major). Position of latent-token `i` is `(chunk=i//q, within=i%q)`,
encoded by `ChunkAwarePositionalEmbedding`. Self-attention over target
latent-tokens is masked by `target_mask` (only the first `m` chunks are valid);
cross-attention sees all valid context latent-tokens (`context_mask`, first `n`).
 
---
 
## 3. File map / ownership
 
| file | owns | depends on |
|---|---|---|
| `config.py` | all dataclass configs, defaults, the shape symbols | — |
| `components.py` | shared nn blocks: `TransformerStack`, `PerceiverResampler`, `TimestepEmbedding`, positional embeds, masking utils | config |
| `codec.py` | `VAEEncoder`, `NATDecoder`, `LengthHead`, `LatentCodec` (losses + reparam + iterative decode), `SonarCodecAdapter`, `CodecInterface` | components, config |
| `predictor.py` | `FlowMatchingPredictor`, `CountHead`, `flow_matching_target`, `FlowSampler`, `ood_score` | components, config |
| `data.py` | `Chunker`, `Tokenizer`, `CodecChunkDataset`, `PairLatentDataset` (+ precompute/cache), `compute_latent_scale`, collates | codec (interface only), config |
| `training.py` | `train_codec`, `train_predictor`, `finetune_joint`, optim/EMA/ckpt utils | all |
| `generate.py` | `TextGenerator` — end-to-end `str → str` | codec, predictor, data, config |
 
Everything downstream of the codec depends only on `CodecInterface`
(`encode_chunk`, `decode_latent`, `latent_dim`) — never on the concrete class —
so the build-vs-buy switch below is a one-line change.
 
---
 
## 4. Build order (suggested agent assignment)
 
1. **`config.py`** + **`components.py`** — unblock everyone. (agent 1)
2. **`codec.py`** — the foundation; nothing trains until this round-trips. (agent 2)
3. **`data.py`** — chunking + Phase-1 dataset can proceed in parallel with (2);
   `PairLatentDataset` needs a working frozen codec. (agent 3)
4. **`training.py::train_codec`** — Phase 1. Gate: held-out sentences round-trip
   (encode→decode) faithfully and latent interpolations decode plausibly. (agent 2/4)
5. **`predictor.py`** + **`training.py::train_predictor`** — Phase 2. Gate:
   sampled latents decode to coherent, *diverse* responses. (agent 5)
6. **`generate.py`** — wire the frozen codec + EMA predictor into one call. (agent 6)
7. **`training.py::finetune_joint`** — Phase 3, only if needed. (agent 5)
---
 
## 5. Data
 
**Phase-1 codec corpus (raw sentences, self-supervised).** Breadth of sentence
structure matters more than frontier quality. Use FineWeb / FineWeb-Edu, C4, or
OpenWebText + Wikipedia. The codec must cover every language/register that gets
*encoded or decoded* (for translation: both sides). Prep: clean+dedupe →
segment to sentences → length-band into `[4, L]` tokens (merge runts, split
overlong) → tokenize with an existing subword tokenizer → store padded id arrays.
Each chunk is simultaneously input and target.
 
**Phase-2 task pairs (prompt → response).** Start constrained — the architecture
is strongest on low-entropy, near-deterministic mappings. MVP picks: Gigaword or
XSum (single-sentence outputs → `m=1`, the whole sequence problem collapses to
one-latent-in/one-latent-out). Then WMT / OPUS (translation, the canonical NAT
domain), CNN/DailyMail, Quora/PAWS/ParaNMT (paraphrase). Open-ended instruction/
dialogue (FLAN, OpenAssistant) is the hardest — it stresses the high-entropy
regime NAT and latent-averaging handle worst. Prep: sentence-split **both** sides
with the *same* chunker → encode with the **frozen** codec → cache latents to
memmap → pad to `M`, record true `m`.
 
---
 
## 6. Build vs buy (the one decision to make first)
 
`CodecConfig.use_pretrained_codec`:
 
- **`False` (default, full custom vision):** train `LatentCodec` from scratch
  (Phase 1). Gives token-level NAT decode (true parallel rendering) but you pay
  for Phase 1 and its posterior-collapse tuning. Requires the Phase-1 corpus.
- **`True` (fastest credible path):** `SonarCodecAdapter` wraps frozen Meta
  SONAR (pretrained multilingual sentence encoder + decoder). Skips Phase 1 and
  the codec corpus entirely — you only need task pairs. Catch: SONAR's decoder is
  autoregressive, so you lose token-level NAT decode but keep cross-chunk
  parallelism via the predictor. Recommended for the first end-to-end signal.
Both expose the identical `CodecInterface`, so Phases 2/3, data, and inference
are unchanged either way.
 
---
 
## 7. Invariants & gotchas (violating these fails silently)
 
- **Tokenizer + chunker must be byte-for-byte identical across Phase 1 and Phase
  2**, or the frozen encoder sees out-of-distribution chunks.
- **Latent scaling factor.** After freezing the codec, `compute_latent_scale`
  encodes a sample, takes per-dim std, stores `1/std`. Multiply all latents by it
  so the predictor's target is ≈ unit variance (matches the flow model's
  unit-Gaussian source — same trick as latent diffusion). **Inverse-scale before
  decoding.** Store in `Config.latent_scale`.
- **Posterior collapse** (Phase 1): if KL → 0 the decoder ignores `z` and the
  latent is useless. Mitigate with β-annealing + free bits (see `codec.py`).
- **No abstention.** An out-of-distribution prompt never makes the model refuse;
  the ODE always integrates to *some* latent and the decoder renders *something*.
  Gate explicitly with `predictor.ood_score` if the input distribution is bounded.
- **Predictor conditioning uses the encoder mean**, not a sample (deterministic
  context). Sampling happens only in the flow ODE on the target side.
- **Loss is masked to real slots.** Flow MSE is computed over the first `m`
  response chunks only; the count head decides `m` at inference and the rest of
  the canvas is discarded.
 