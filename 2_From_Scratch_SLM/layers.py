import math

from math_ops import (
    random_matrix,
    random_vector,
    zeros_matrix,
    zeros_vector,
    matvec,
    matvec_transpose,
    vector_add,
    add_outer,
    zero_inplace,
)


class Parameter:
    def __init__(self, value, grad, name=""):
        self.value = value
        self.grad = grad
        self.name = name


def _is_vector(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (int, float)))


def _is_matrix(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)


def _add_inplace(a, b):
    for i in range(len(a)):
        a[i] += b[i]


class Linear:
    def __init__(self, in_dim, out_dim, weight_scale=0.1):
        self.W = random_matrix(out_dim, in_dim, weight_scale)
        self.b = zeros_vector(out_dim)
        self.dW = zeros_matrix(out_dim, in_dim)
        self.db = zeros_vector(out_dim)
        self.last_x = None
        self.last_xs = None

    def forward(self, x):
        if _is_vector(x):
            self.last_x = x
            self.last_xs = None
            y = matvec(self.W, x)
            return vector_add(y, self.b)
        if _is_matrix(x):
            self.last_xs = x
            self.last_x = None
            out = []
            for x_i in x:
                y = matvec(self.W, x_i)
                out.append(vector_add(y, self.b))
            return out
        raise ValueError("Linear.forward expects a vector or a list of vectors.")

    def backward(self, dout):
        if _is_vector(dout):
            add_outer(self.dW, dout, self.last_x)
            _add_inplace(self.db, dout)
            return matvec_transpose(self.W, dout)
        if _is_matrix(dout):
            dxs = []
            for x_i, dout_i in zip(self.last_xs, dout):
                add_outer(self.dW, dout_i, x_i)
                _add_inplace(self.db, dout_i)
                dxs.append(matvec_transpose(self.W, dout_i))
            return dxs
        raise ValueError("Linear.backward expects a vector or a list of vectors.")

    def zero_grad(self):
        zero_inplace(self.dW)
        zero_inplace(self.db)

    def parameters(self):
        return [
            Parameter(self.W, self.dW, "W"),
            Parameter(self.b, self.db, "b"),
        ]


class Embedding:
    def __init__(self, vocab_size, embed_dim, weight_scale=0.1):
        self.W = random_matrix(vocab_size, embed_dim, weight_scale)
        self.dW = zeros_matrix(vocab_size, embed_dim)
        self.last_ids = None

    def forward(self, ids):
        self.last_ids = ids
        if isinstance(ids, int):
            return self.W[ids][:]
        if _is_vector(ids):
            return [self.W[i][:] for i in ids]
        if _is_matrix(ids):
            return [[self.W[i][:] for i in row] for row in ids]
        raise ValueError("Embedding.forward expects an int, list, or list of lists.")

    def backward(self, dout, ids=None):
        if ids is None:
            ids = self.last_ids
        if isinstance(ids, int):
            for j in range(len(dout)):
                self.dW[ids][j] += dout[j]
            return
        if _is_vector(ids):
            for idx, grad_vec in zip(ids, dout):
                for j in range(len(grad_vec)):
                    self.dW[idx][j] += grad_vec[j]
            return
        if _is_matrix(ids):
            for row_ids, row_grads in zip(ids, dout):
                for idx, grad_vec in zip(row_ids, row_grads):
                    for j in range(len(grad_vec)):
                        self.dW[idx][j] += grad_vec[j]
            return
        raise ValueError("Embedding.backward expects ids aligned with dout.")

    def zero_grad(self):
        zero_inplace(self.dW)

    def parameters(self):
        return [Parameter(self.W, self.dW, "embeddings")]


class Tanh:
    def __init__(self):
        self.last_y = None

    def forward(self, x):
        if not _is_vector(x):
            raise ValueError("Tanh.forward expects a vector.")
        y = [math.tanh(v) for v in x]
        self.last_y = y
        return y

    def backward(self, dout):
        if not _is_vector(dout):
            raise ValueError("Tanh.backward expects a vector.")
        return [dout[i] * (1.0 - self.last_y[i] * self.last_y[i]) for i in range(len(dout))]


class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, weight_scale=0.1):
        self.Wxh = random_matrix(hidden_dim, input_dim, weight_scale)
        self.Whh = random_matrix(hidden_dim, hidden_dim, weight_scale)
        self.b = zeros_vector(hidden_dim)
        self.dWxh = zeros_matrix(hidden_dim, input_dim)
        self.dWhh = zeros_matrix(hidden_dim, hidden_dim)
        self.db = zeros_vector(hidden_dim)
        self.cache = None

    def forward(self, x_seq, h0=None):
        if h0 is None:
            h_prev = zeros_vector(len(self.b))
        else:
            h_prev = h0
        self.cache = []
        h_seq = []
        for x_t in x_seq:
            a = vector_add(matvec(self.Wxh, x_t), matvec(self.Whh, h_prev))
            a = vector_add(a, self.b)
            h_t = [math.tanh(v) for v in a]
            self.cache.append((x_t, h_prev, h_t))
            h_seq.append(h_t)
            h_prev = h_t
        return h_seq

    def backward(self, dh_seq):
        T = len(dh_seq)
        hidden_dim = len(self.b)
        dx_seq = [None for _ in range(T)]
        dh_next = zeros_vector(hidden_dim)
        for t in reversed(range(T)):
            x_t, h_prev, h_t = self.cache[t]
            dh = vector_add(dh_seq[t], dh_next)
            dtanh = [dh[i] * (1.0 - h_t[i] * h_t[i]) for i in range(hidden_dim)]
            add_outer(self.dWxh, dtanh, x_t)
            add_outer(self.dWhh, dtanh, h_prev)
            _add_inplace(self.db, dtanh)
            dx_seq[t] = matvec_transpose(self.Wxh, dtanh)
            dh_next = matvec_transpose(self.Whh, dtanh)
        return dx_seq, dh_next

    def zero_grad(self):
        zero_inplace(self.dWxh)
        zero_inplace(self.dWhh)
        zero_inplace(self.db)

    def parameters(self):
        return [
            Parameter(self.Wxh, self.dWxh, "Wxh"),
            Parameter(self.Whh, self.dWhh, "Whh"),
            Parameter(self.b, self.db, "b"),
        ]
