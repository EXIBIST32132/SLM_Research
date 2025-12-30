import os
from typing import Dict

import torch
from torch.optim import AdamW

from config import Config
from dataset import create_dataloaders
from model import TransformerLanguageModel


def estimate_loss(
    model,
    data_loader,
    device,
    max_batches: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    losses = []
    max_batches = max(1, max_batches)
    with torch.no_grad():
        for idx, (xb, yb) in enumerate(data_loader):
            if idx >= max_batches:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


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


def save_checkpoint(
    path: str,
    model: TransformerLanguageModel,
    optimizer: AdamW,
    step: int,
    config: Config,
    extra: Dict = None,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config.__dict__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[Checkpoint] Saved to {path}")


def main():
    config = Config()
    device = get_device(config)
    print(f"Using device: {device}")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(config.ckpt_dir, exist_ok=True)

    tokenizer, train_loader, val_loader = create_dataloaders(config)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
    ).to(device)

    if config.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
            print("Compiled model with torch.compile for faster training.")
        except Exception as e:  # pragma: no cover - best effort speed-up
            print(f"torch.compile failed ({e}); continuing without compilation.")

    optimizer_kwargs = {"lr": config.learning_rate}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        # fused not available (likely older PyTorch); fall back to standard
        optimizer_kwargs.pop("fused", None)
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    amp_dtype = get_amp_dtype(device)
    amp_enabled = config.use_amp and device.type in ("cuda", "mps")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")

    step = 0
    max_iters = config.max_iters

    while step < max_iters:
        for xb, yb in train_loader:
            if step >= max_iters:
                break

            step += 1
            xb = xb.to(device)
            yb = yb.to(device)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                _, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if step % 50 == 0:
                print(f"Step {step} | train loss: {loss.item():.4f}")

            if step % config.eval_interval == 0:
                train_loss = estimate_loss(
                    model,
                    train_loader,
                    device,
                    max_batches=config.max_eval_batches,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                val_loss = estimate_loss(
                    model,
                    val_loader,
                    device,
                    max_batches=config.max_eval_batches,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                print(
                    f"[Eval] Step {step} | "
                    f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
                )

            if step % config.sample_every == 0:
                print("=== Sample ===")
                model.eval()
                with torch.no_grad():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        start = torch.zeros((1, 1), dtype=torch.long, device=device)
                        out = model.generate(
                            start,
                            max_new_tokens=config.sample_tokens,
                            temperature=config.temperature,
                            top_k=config.top_k,
                        )
                model.train()
                text = tokenizer.decode(out[0].tolist())
                print(text)
                print("=== End sample ===")

            if step % config.save_interval == 0:
                ckpt_path = os.path.join(
                    config.ckpt_dir, f"baseline_step_{step}.pt"
                )
                save_checkpoint(ckpt_path, model, optimizer, step, config)

    final_path = os.path.join(config.ckpt_dir, config.final_ckpt_name)
    save_checkpoint(final_path, model, optimizer, step, config)
    print("Training complete.")


if __name__ == "__main__":
    main()
