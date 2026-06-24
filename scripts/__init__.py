"""Runnable entry points for the SONAR latent-flow MVP.

Run from the PARENT directory of this package, e.g.:

    python -m Sync.scripts.gate_ceiling --task gigaword --smoke
    python -m Sync.scripts.precompute_latents --task gigaword
    python -m Sync.scripts.gate_uncond_flow --task gigaword
    python -m Sync.scripts.train_predictor --task gigaword
    python -m Sync.scripts.eval_curves --task gigaword --ckpt runs/predictor_final.pt

Every script accepts --smoke to run on a tiny subset end-to-end.
"""
