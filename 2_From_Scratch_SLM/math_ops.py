import math
import random


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def matmul(a, b):
    if not a or not b:
        return []
    m = len(a)
    n = len(a[0])
    p = len(b[0])
    out = zeros_matrix(m, p)
    for i in range(m):
        for k in range(n):
            aik = a[i][k]
            bk = b[k]
            row = out[i]
            for j in range(p):
                row[j] += aik * bk[j]
    return out


def matvec(matrix, vec):
    out = []
    for row in matrix:
        s = 0.0
        for j in range(len(vec)):
            s += row[j] * vec[j]
        out.append(s)
    return out


def matvec_transpose(matrix, vec):
    if not matrix:
        return []
    cols = len(matrix[0])
    out = [0.0 for _ in range(cols)]
    for i in range(len(matrix)):
        for j in range(cols):
            out[j] += matrix[i][j] * vec[i]
    return out


def vector_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def scalar_multiply(x, s):
    if not x:
        return []
    if isinstance(x[0], list):
        return [[v * s for v in row] for row in x]
    return [v * s for v in x]


def softmax(vec):
    if not vec:
        return []
    max_val = max(vec)
    exps = [math.exp(v - max_val) for v in vec]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(vec) for _ in vec]
    return [v / total for v in exps]


def cross_entropy(logits, target_index):
    probs = softmax(logits)
    p = max(probs[target_index], 1e-12)
    return -math.log(p)


def random_matrix(rows, cols, scale=0.1):
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def random_vector(size, scale=0.1):
    if scale == 0.0:
        return [0.0 for _ in range(size)]
    return [random.uniform(-scale, scale) for _ in range(size)]


def zeros_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def zeros_vector(size):
    return [0.0 for _ in range(size)]


def zeros_like(x):
    if not x:
        return []
    if isinstance(x[0], list):
        return [zeros_like(row) for row in x]
    return [0.0 for _ in x]


def zero_inplace(x):
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                zero_inplace(x[i])
            else:
                x[i] = 0.0


def add_outer(target, a, b):
    for i in range(len(a)):
        ai = a[i]
        row = target[i]
        for j in range(len(b)):
            row[j] += ai * b[j]


def sum_squares(x):
    if isinstance(x, list):
        total = 0.0
        for item in x:
            total += sum_squares(item)
        return total
    return x * x


def scale_inplace(x, scale):
    if isinstance(x, list):
        for i in range(len(x)):
            if isinstance(x[i], list):
                scale_inplace(x[i], scale)
            else:
                x[i] *= scale


def clip_grad_norm(params, max_norm):
    total = 0.0
    for p in params:
        total += sum_squares(p.grad)
    norm = math.sqrt(total)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for p in params:
            scale_inplace(p.grad, scale)
    return norm
