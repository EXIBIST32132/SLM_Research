# From Scratch SLM

This folder contains a tiny character-level language model built from scratch in pure Python using only the standard library. It is intentionally small and slow, aimed at correctness and clarity.

## Usage

1) Place training data in `0_Datasets/tiny_corpus.txt` (repo root).
2) Train:

```bash
python3 train.py
```

3) Generate text:

```bash
python3 generate.py --prompt "Hello"
```

## Notes

- The model is a minimal RNN trained with teacher forcing and next-character prediction.
- It is designed for small datasets and short runs.
