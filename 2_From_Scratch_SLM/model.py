import json
import math
import os
import pickle
import random

from config import Config
from layers import Embedding, Linear, SimpleRNN
from math_ops import softmax


def _copy_into(dest, src):
    if isinstance(dest, list) and dest and isinstance(dest[0], list):
        for i in range(len(dest)):
            for j in range(len(dest[i])):
                dest[i][j] = src[i][j]
    elif isinstance(dest, list):
        for i in range(len(dest)):
            dest[i] = src[i]


def _sample_from_probs(probs):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def save_checkpoint(weights_path, model, config, vocab_path):
    state = model.state_dict()
    with open(weights_path, "wb") as f:
        pickle.dump(state, f)
    meta = {
        "config": config.to_dict(),
        "vocab_path": vocab_path,
    }
    with open(weights_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_checkpoint(weights_path, vocab_size):
    meta_path = weights_path + ".json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    config = Config.from_dict(meta["config"])
    model = LanguageModel(config, vocab_size)
    with open(weights_path, "rb") as f:
        state = pickle.load(f)
    model.load_state_dict(state)
    return model, config, meta


class LanguageModel:
    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, config.embed_dim)
        self.rnn = SimpleRNN(config.embed_dim, config.hidden_dim)
        self.output = Linear(config.hidden_dim, vocab_size)
        self._rnn_caches = None
        self._last_input_ids = None
        self._last_logits = None
        self.dlogits = None

    def forward(self, input_ids):
        self._last_input_ids = input_ids
        embeddings = self.embedding.forward(input_ids)
        h_seqs = []
        caches = []
        for emb_seq in embeddings:
            h_seq = self.rnn.forward(emb_seq)
            caches.append(self.rnn.cache)
            h_seqs.append(h_seq)
        self._rnn_caches = caches

        flat_h = []
        for h_seq in h_seqs:
            for h_t in h_seq:
                flat_h.append(h_t)
        flat_logits = self.output.forward(flat_h)

        logits = []
        idx = 0
        for h_seq in h_seqs:
            seq_logits = []
            for _ in h_seq:
                seq_logits.append(flat_logits[idx])
                idx += 1
            logits.append(seq_logits)
        self._last_logits = logits
        return logits

    def compute_loss(self, logits, target_ids):
        batch_size = len(logits)
        seq_len = len(logits[0])
        dlogits = []
        total_loss = 0.0
        for b in range(batch_size):
            row_grads = []
            for t in range(seq_len):
                logit_vec = logits[b][t]
                target = target_ids[b][t]
                probs = softmax(logit_vec)
                p = max(probs[target], 1e-12)
                total_loss += -math.log(p)
                grad = probs[:]
                grad[target] -= 1.0
                row_grads.append(grad)
            dlogits.append(row_grads)
        scale = 1.0 / (batch_size * seq_len)
        for b in range(batch_size):
            for t in range(seq_len):
                for i in range(len(dlogits[b][t])):
                    dlogits[b][t][i] *= scale
        self.dlogits = dlogits
        return total_loss * scale

    def backward(self):
        if self.dlogits is None:
            raise ValueError("compute_loss must be called before backward.")
        batch_size = len(self.dlogits)
        seq_len = len(self.dlogits[0])

        flat_dlogits = []
        for b in range(batch_size):
            for t in range(seq_len):
                flat_dlogits.append(self.dlogits[b][t])
        flat_dh = self.output.backward(flat_dlogits)

        dh_seqs = []
        idx = 0
        for _ in range(batch_size):
            seq = []
            for _ in range(seq_len):
                seq.append(flat_dh[idx])
                idx += 1
            dh_seqs.append(seq)

        d_embeddings = []
        for b in range(batch_size):
            self.rnn.cache = self._rnn_caches[b]
            dx_seq, _ = self.rnn.backward(dh_seqs[b])
            d_embeddings.append(dx_seq)

        self.embedding.backward(d_embeddings, self._last_input_ids)

    def zero_grad(self):
        self.embedding.zero_grad()
        self.rnn.zero_grad()
        self.output.zero_grad()

    def parameters(self):
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.rnn.parameters())
        params.extend(self.output.parameters())
        return params

    def state_dict(self):
        return {
            "embedding_W": self.embedding.W,
            "rnn_Wxh": self.rnn.Wxh,
            "rnn_Whh": self.rnn.Whh,
            "rnn_b": self.rnn.b,
            "output_W": self.output.W,
            "output_b": self.output.b,
        }

    def load_state_dict(self, state):
        _copy_into(self.embedding.W, state["embedding_W"])
        _copy_into(self.rnn.Wxh, state["rnn_Wxh"])
        _copy_into(self.rnn.Whh, state["rnn_Whh"])
        _copy_into(self.rnn.b, state["rnn_b"])
        _copy_into(self.output.W, state["output_W"])
        _copy_into(self.output.b, state["output_b"])

    def generate(self, seed_ids, max_new_tokens, temperature=1.0, top_k=None):
        ids = list(seed_ids)
        for _ in range(max_new_tokens):
            start = max(0, len(ids) - self.config.block_size)
            context = ids[start:]
            logits = self.forward([context])
            next_logits = logits[0][-1]

            temp = max(temperature, 1e-6)
            scaled = [v / temp for v in next_logits]
            if top_k is not None and top_k < len(scaled):
                order = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
                cutoff = set(order[top_k:])
                for i in cutoff:
                    scaled[i] = -1e9
            probs = softmax(scaled)
            next_id = _sample_from_probs(probs)
            ids.append(next_id)
        return ids
