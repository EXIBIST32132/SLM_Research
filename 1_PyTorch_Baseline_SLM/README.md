Baseline PyTorch character-level SLM
====================================

Structure
- `config.py`: hyperparameters and paths.
- `dataset.py`: character tokenizer + dataset + dataloaders.
- `model.py`: Transformer encoder used as an autoregressive LM.
- `train.py`: training loop with eval, sampling, and checkpointing.
- `generate.py`: load a checkpoint and generate text.
- `../0_Datasets/medium_corpus.txt`: training corpus (provide your own).
- `checkpoints/`: saved models.

What changed for speed & length
- Context length bumped to `block_size=512` and default generation to 800 chars so long prompts/outputs are supported.
- Model is slimmer (384-dim, 6 layers, flash-style attention via `scaled_dot_product_attention`) and uses `torch.compile`, mixed precision, fused AdamW, and TF32 for a large speed-up on GPU.
- Eval is capped to `max_eval_batches` to keep training moving; dataloaders use workers + pinned memory by default.
- `generate.py` now accepts prompt files, longer generation lengths, sampling overrides, checkpoint override, and device override; it also pulls config from the checkpoint so settings stay in sync.

Quickstart
1. Create venv and install deps:
   - `python -m venv .venv`
   - `source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
   - `pip install torch`
2. Add training text to `0_Datasets/medium_corpus.txt` (any plain text works).
3. Train: `python train.py`
   - Uses mixed precision + compile automatically when available.
   - Loss logged every 50 steps; eval uses up to `max_eval_batches` batches; checkpoints saved to `checkpoints/`.
4. Generate after training:
   - `python generate.py --prompt "Your long prompt here"` or `--prompt-file path/to/prompt.txt`
   - Customize length/sampling: `--max-new-tokens 1500 --temperature 0.7 --top-k 50`
   - Force a checkpoint: `--ckpt checkpoints/baseline_step_2000.pt`
   - Force device if needed: `--device cpu`

Notes
- The model context window is set by `block_size` in `config.py` and stored in checkpoints. To use very long prompts, bump `block_size` before training (compute cost scales with block size).
- If `torch.compile` or fused AdamW is unsupported on your setup, the code falls back automatically.
