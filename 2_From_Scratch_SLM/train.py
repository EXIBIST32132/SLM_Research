import math
import os
import random
import time

from config import Config
from data import TextData
from math_ops import clip_grad_norm
from model import LanguageModel, save_checkpoint


def _is_matrix(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)


def _zeros_like(x):
    if _is_matrix(x):
        return [[0.0 for _ in row] for row in x]
    return [0.0 for _ in x]


class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [_zeros_like(p.grad) for p in params]
        self.v = [_zeros_like(p.grad) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self._update_param(p.value, p.grad, self.m[i], self.v[i])

    def _update_param(self, param, grad, m, v):
        if _is_matrix(param):
            for r in range(len(param)):
                row = param[r]
                grad_row = grad[r]
                m_row = m[r]
                v_row = v[r]
                for c in range(len(row)):
                    g = grad_row[c]
                    m_row[c] = self.beta1 * m_row[c] + (1 - self.beta1) * g
                    v_row[c] = self.beta2 * v_row[c] + (1 - self.beta2) * g * g
                    m_hat = m_row[c] / (1 - self.beta1 ** self.t)
                    v_hat = v_row[c] / (1 - self.beta2 ** self.t)
                    row[c] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        else:
            for i in range(len(param)):
                g = grad[i]
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * g
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * g * g
                m_hat = m[i] / (1 - self.beta1 ** self.t)
                v_hat = v[i] / (1 - self.beta2 ** self.t)
                param[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


def estimate_loss(model, data, config, eval_iters=5):
    total = 0.0
    for _ in range(eval_iters):
        x_batch, y_batch = data.get_batch("val", config.batch_size)
        logits = model.forward(x_batch)
        loss = model.compute_loss(logits, y_batch)
        total += loss
    return total / eval_iters


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base_dir, "..", "0_Datasets"))
    input_path = os.path.join(data_dir, "tiny_corpus.txt")

    config = Config()
    random.seed(config.seed)

    data = TextData(input_path, config.block_size)
    tokenizer = data.tokenizer

    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    tokenizer.save_vocab(vocab_path)

    model = LanguageModel(config, tokenizer.vocab_size)
    params = model.parameters()
    optimizer = Adam(params, lr=config.learning_rate)

    seed_ids = data.train_ids[: min(10, len(data.train_ids))]

    start = time.time()
    for step in range(1, config.max_steps + 1):
        x_batch, y_batch = data.get_batch("train", config.batch_size)
        logits = model.forward(x_batch)
        loss = model.compute_loss(logits, y_batch)
        model.backward()
        clip_grad_norm(params, 1.0)
        optimizer.step()
        model.zero_grad()

        if step % config.eval_interval == 0:
            val_loss = estimate_loss(model, data, config)
            elapsed = time.time() - start
            print(
                "step %d train_loss %.4f val_loss %.4f time %.1fs"
                % (step, loss, val_loss, elapsed)
            )

        if step % config.sample_every == 0:
            sample_ids = model.generate(
                seed_ids,
                config.sample_tokens,
                temperature=1.0,
                top_k=20,
            )
            print("---- sample ----")
            print(tokenizer.decode(sample_ids))
            print("---------------")

    weights_path = config.checkpoint_path
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(base_dir, weights_path)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    save_checkpoint(weights_path, model, config, os.path.abspath(vocab_path))
    print("Saved checkpoint to %s" % weights_path)


if __name__ == "__main__":
    main()
