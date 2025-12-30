import argparse
import json
import os

from model import load_checkpoint
from tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_checkpoint = os.path.join(base_dir, "checkpoints", "model.pkl")
    weights_path = args.checkpoint or default_checkpoint
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(base_dir, weights_path)
    meta_path = weights_path + ".json"

    if not os.path.exists(weights_path) or not os.path.exists(meta_path):
        print("Error: checkpoint not found. Run train.py first.")
        raise SystemExit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab_path = meta.get("vocab_path")
    if not vocab_path:
        print("Error: vocab_path missing from checkpoint metadata.")
        raise SystemExit(1)
    if not os.path.isabs(vocab_path):
        vocab_path = os.path.join(base_dir, vocab_path)
    if not os.path.exists(vocab_path):
        print("Error: vocab file not found at %s" % vocab_path)
        raise SystemExit(1)

    tokenizer = CharTokenizer().load_vocab(vocab_path)
    model, _, _ = load_checkpoint(weights_path, tokenizer.vocab_size)

    prompt = args.prompt
    if prompt:
        try:
            seed_ids = tokenizer.encode(prompt)
        except ValueError:
            filtered = [ch for ch in prompt if ch in tokenizer.stoi]
            if not filtered:
                seed_ids = [0]
            else:
                seed_ids = [tokenizer.stoi[ch] for ch in filtered]
            print("Warning: prompt contained out-of-vocab characters; filtering them.")
    else:
        seed_ids = [0]

    ids = model.generate(
        seed_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
