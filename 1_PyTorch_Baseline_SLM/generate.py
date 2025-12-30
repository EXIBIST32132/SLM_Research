import argparse
import os
from typing import Optional

import torch

from config import Config
from dataset import CharTokenizer, load_text
from model import TransformerLanguageModel


def get_device(config: Config) -> torch.device:
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def resolve_checkpoint_path(config: Config, override: Optional[str]) -> str:
    ckpt_path = (
        override
        if override
        else os.path.join(config.ckpt_dir, config.final_ckpt_name)
    )
    if os.path.exists(ckpt_path):
        return ckpt_path

    pt_files = [
        os.path.join(config.ckpt_dir, f)
        for f in os.listdir(config.ckpt_dir)
        if f.endswith(".pt")
    ]
    if not pt_files:
        raise FileNotFoundError(
            f"No checkpoints found in {config.ckpt_dir}. Train the model first."
        )
    latest = max(pt_files, key=os.path.getmtime)
    print(f"Using latest checkpoint found: {latest}")
    return latest


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional prompt to start generation (must use chars from training data).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Optional path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k sampling.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint file to load (defaults to final_ckpt_name or latest).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force device override; otherwise uses config preference with availability check.",
    )
    args = parser.parse_args()

    config = Config()
    ckpt_path = resolve_checkpoint_path(config, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "config" in checkpoint:
        config = Config(**checkpoint["config"])
    if args.device:
        config.device = args.device

    device = get_device(config)
    print(f"Using device: {device}")

    text = load_text(config.data_path)
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    max_new_tokens = args.max_new_tokens or config.sample_tokens
    temperature = (
        args.temperature if args.temperature is not None else config.temperature
    )
    top_k = args.top_k if args.top_k is not None else config.top_k

    prompt = args.prompt or ""
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    missing = [c for c in prompt if c not in tokenizer.stoi]
    if missing:
        uniq_missing = "".join(sorted(set(missing)))
        raise ValueError(
            f"Prompt contains chars not in the training vocab: {uniq_missing!r}. "
            "Use only characters seen in the training data."
        )

    encoded_prompt = tokenizer.encode(prompt) if prompt else [0]
    if len(encoded_prompt) > config.block_size:
        print(
            f"Prompt length {len(encoded_prompt)} exceeds block_size {config.block_size}; "
            "truncating to the most recent tokens."
        )
        encoded_prompt = encoded_prompt[-config.block_size :]

    start_tokens = (
        torch.tensor(encoded_prompt, dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    amp_enabled = config.use_amp and device.type in ("cuda", "mps")
    amp_dtype = get_amp_dtype(device)

    with torch.no_grad():
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            out = model.generate(
                start_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

    gen_text = tokenizer.decode(out[0].tolist())
    print("=== Generated Text ===")
    print(gen_text)
    print("=======================")


if __name__ == "__main__":
    main()
