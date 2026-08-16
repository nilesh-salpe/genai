"""
Small CNN, no autograd -- builds on part-2-build.html SS6 (the delta-blame
backprop formulas) and answers SS8, "The dense layer never sees a picture":
a conv layer keeps the 2D grid, looks at small local patches, and reuses
the same weights everywhere. This file is that idea, trained end to end.

  conv(4 filters, 3x3, valid) -> ReLU -> 2x2 maxpool -> flatten(36) -> dense(36,10) -> softmax

Data: sklearn.datasets.load_digits() -- 8x8 grayscale digits, bundled with
scikit-learn (no download). sklearn is used ONLY to fetch that array; every
layer, gradient, and update below is plain numpy, same as SS6/SS7.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits  # dataset only -- no sklearn model/training code

HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "02_weights.npz"


# ---------- data ----------

def load_data(test_frac=0.15, seed=0):
    digits = load_digits()
    X = digits.images.astype(np.float64) / 16.0   # pixel range here is 0-16, not 0-255
    y = digits.target.astype(np.int64)

    idx = np.random.default_rng(seed).permutation(X.shape[0])
    X, y = X[idx], y[idx]

    n_test = int(round(X.shape[0] * test_frac))
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]
    return X_train, y_train, X_test, y_test


# ---------- init ----------

def init_params(seed=0):
    r = np.random.default_rng(seed)
    F, kh, kw = 4, 3, 3

    # He init: std = sqrt(2/fan_in). Each conv output pixel looks at kh*kw
    # inputs (1 input channel here); each dense output looks at 36 inputs.
    filters = r.normal(0, np.sqrt(2.0 / (kh * kw)), (F, kh, kw))
    bias = np.zeros(F)

    flat_dim = F * 3 * 3   # 4 filters x 3x3 pooled grid = 36
    W = r.normal(0, np.sqrt(2.0 / flat_dim), (flat_dim, 10))
    b = np.zeros(10)

    return dict(filters=filters, bias=bias, W=W, b=b)


# ---------- conv layer, via im2col ----------
# Instead of four nested Python loops per output pixel, unroll every 3x3
# patch into a row of a matrix ("im2col") and do the whole conv as one
# matmul. The loop below only ever runs kh*kw = 9 times, regardless of
# image size or batch size -- that's what makes it vectorized.

def im2col(X, kh, kw, stride=1):
    # X: (B, H, W) -> cols: (B, out_h*out_w, kh*kw)
    B, H, W = X.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    patches = np.empty((B, out_h, out_w, kh, kw), dtype=X.dtype)
    for di in range(kh):
        for dj in range(kw):
            patches[:, :, :, di, dj] = X[:, di:di + out_h * stride:stride,
                                             dj:dj + out_w * stride:stride]
    cols = patches.reshape(B, out_h * out_w, kh * kw)
    return cols, out_h, out_w


def col2im(dcols, x_shape, kh, kw, out_h, out_w, stride=1):
    # Inverse of im2col. Patches overlap (stride < kernel size here), so a
    # pixel touched by several patches must accumulate (+=), not overwrite.
    B, H, W = x_shape
    dX = np.zeros((B, H, W), dtype=dcols.dtype)
    dpatches = dcols.reshape(B, out_h, out_w, kh, kw)
    for di in range(kh):
        for dj in range(kw):
            dX[:, di:di + out_h * stride:stride, dj:dj + out_w * stride:stride] += dpatches[:, :, :, di, dj]
    return dX


def conv_forward(X, filters, bias, stride=1):
    # X: (B, H, W), filters: (F, kh, kw), bias: (F,) -> out: (B, F, out_h, out_w)
    B = X.shape[0]
    F, kh, kw = filters.shape
    cols, out_h, out_w = im2col(X, kh, kw, stride)         # (B, N, K), N=out_h*out_w, K=kh*kw
    W_col = filters.reshape(F, kh * kw)                    # (F, K)

    out = cols @ W_col.T + bias                            # (B,N,K)(K,F) + (F,) -> (B,N,F)
    out = out.transpose(0, 2, 1).reshape(B, F, out_h, out_w)

    cache = (X, cols, filters, out_h, out_w, kh, kw, stride)
    return out, cache


def conv_backward(dOut, cache):
    # dOut: (B, F, out_h, out_w), gradient of the loss w.r.t. the conv output.
    X, cols, filters, out_h, out_w, kh, kw, stride = cache
    B, F, OH, OW = dOut.shape
    W_col = filters.reshape(F, kh * kw)

    dOut_col = dOut.reshape(B, F, OH * OW).transpose(0, 2, 1)   # (B, N, F)

    # out[b,n,f] = sum_k cols[b,n,k] * W_col[f,k] + bias[f]  -- same shape
    # of algebra as the dense layer's Z = A W^T + b, just per-patch.
    dfilters = np.einsum('bnf,bnk->fk', dOut_col, cols).reshape(F, kh, kw)   # dL/dW = delta . activity, summed over batch+patches
    dbias = dOut_col.sum(axis=(0, 1))                                       # dL/db = delta, summed over batch+patches
    dcols = dOut_col @ W_col                                                # (B,N,F)(F,K) -> (B,N,K), the patch-side "W^T delta"

    dX = col2im(dcols, X.shape, kh, kw, out_h, out_w, stride)
    return dfilters, dbias, dX


# ---------- 2x2 max-pool, stride 2 ----------

def maxpool_forward(X, pool=2, stride=2):
    # X: (B, F, H, W) -> out: (B, F, H//stride, W//stride)
    B, F, H, W = X.shape
    out_h, out_w = H // stride, W // stride

    # Stack the pool*pool candidate values per window, then argmax picks
    # which one wins -- that index is exactly what backward needs to route
    # the gradient to the single position that was the max.
    windows = np.stack([
        X[:, :, i:i + out_h * stride:stride, j:j + out_w * stride:stride]
        for i in range(pool) for j in range(pool)
    ], axis=0)                                          # (pool*pool, B, F, out_h, out_w)

    argmax_idx = np.argmax(windows, axis=0)              # (B, F, out_h, out_w)
    out = np.take_along_axis(windows, argmax_idx[None, ...], axis=0)[0]

    cache = (X.shape, argmax_idx, pool, stride)
    return out, cache


def maxpool_backward(dOut, cache):
    x_shape, argmax_idx, pool, stride = cache
    B, F, H, W = x_shape
    out_h, out_w = H // stride, W // stride
    dX = np.zeros(x_shape, dtype=dOut.dtype)

    for k in range(pool * pool):
        i, j = divmod(k, pool)
        mask = (argmax_idx == k)                          # 1 where this offset was the argmax, else 0
        dX[:, :, i:i + out_h * stride:stride, j:j + out_w * stride:stride] += dOut * mask
    return dX


# ---------- activations / loss (same conventions as the dense-only example) ----------

def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)   # log-sum-exp trick: stops overflow, changes nothing mathematically
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    B = y.shape[0]
    return -np.mean(np.log(probs[np.arange(B), y] + 1e-12))   # +1e-12: never log(0)


# ---------- full pipeline: conv -> relu -> pool -> flatten -> dense -> softmax ----------

def forward(X, params):
    # X: (B, 8, 8). Save every intermediate -- backward needs all of it.
    conv_out, conv_cache = conv_forward(X, params["filters"], params["bias"])   # (B,4,6,6)
    relu_mask = conv_out > 0
    relu_out = relu(conv_out)
    pool_out, pool_cache = maxpool_forward(relu_out)                            # (B,4,3,3)

    B = X.shape[0]
    flat = pool_out.reshape(B, -1)                                              # (B,36)
    logits = flat @ params["W"] + params["b"]                                   # (B,36)(36,10) + (10,)
    probs = softmax(logits)

    cache = dict(conv_cache=conv_cache, relu_mask=relu_mask, pool_cache=pool_cache,
                 flat=flat, pool_shape=pool_out.shape, probs=probs)
    return probs, cache


def backward(params, cache, y):
    probs = cache["probs"]
    B = y.shape[0]
    Y = np.zeros_like(probs)
    Y[np.arange(B), y] = 1.0

    # softmax + cross-entropy collapse to exactly (predicted - actual), same identity as SS6/SS7
    dlogits = (probs - Y) / B

    dW = cache["flat"].T @ dlogits            # dL/dW = delta . activity, summed over the batch
    db = dlogits.sum(axis=0)
    dflat = dlogits @ params["W"].T           # W^T delta, wired backwards through the dense layer
    dpool_out = dflat.reshape(cache["pool_shape"])

    drelu_out = maxpool_backward(dpool_out, cache["pool_cache"])
    dconv_out = drelu_out * cache["relu_mask"]   # ReLU blocks blame wherever the pre-activation was <= 0
    dfilters, dbias, _dX = conv_backward(dconv_out, cache["conv_cache"])

    return dict(filters=dfilters, bias=dbias, W=dW, b=db)


# ---------- gradient check (correctness proof, not part of normal training) ----------
# Compares the analytic grads above against central-difference numerical
# gradients on a few entries of the conv filters and the dense weights.
# Run with `python3 02_cnn_digit_classifier.py --gradcheck`.

def _gradient_check(eps=1e-5, n_samples=3, seed=0):
    X_train, y_train, _, _ = load_data(seed=seed)
    X, y = X_train[:n_samples], y_train[:n_samples]
    params = init_params(seed=seed)

    probs, cache = forward(X, params)
    grads = backward(params, cache, y)

    def loss_of(p):
        pr, _ = forward(X, p)
        return cross_entropy_loss(pr, y)

    def check(name, coords):
        arr, g = params[name], grads[name]
        worst = 0.0
        for idx in coords:
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = loss_of(params)
            arr[idx] = orig - eps
            lm = loss_of(params)
            arr[idx] = orig

            numerical = (lp - lm) / (2 * eps)
            analytic = g[idx]
            rel_err = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-8)
            worst = max(worst, rel_err)
            status = "OK" if rel_err < 1e-4 else "MISMATCH"
            print(f"  {name}{idx}: analytic={analytic: .6e}  numeric={numerical: .6e}"
                  f"  rel_err={rel_err:.2e}  [{status}]")
        return worst

    print("gradient check (eps=1e-5, central difference, n_samples=%d)" % n_samples)
    w1 = check("filters", [(0, 0, 0), (1, 1, 1), (2, 0, 2), (3, 2, 2)])
    w2 = check("bias", [(0,), (2,), (3,)])
    w3 = check("W", [(0, 0), (17, 5), (35, 9), (9, 3)])
    w4 = check("b", [(0,), (5,), (9,)])
    worst = max(w1, w2, w3, w4)
    print(f"\nworst relative error across all checked entries: {worst:.2e}")
    print("PASS" if worst < 1e-4 else "FAIL -- backprop has a bug")


# ---------- training ----------

def train(X_train, y_train, epochs=15, lr=0.05, batch_size=32, seed=0, verbose=True):
    params = init_params(seed=seed)
    r = np.random.default_rng(seed)
    n = X_train.shape[0]

    t0 = time.time()
    for ep in range(epochs):
        order = r.permutation(n)     # reshuffle each epoch
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            Xb, yb = X_train[idx], y_train[idx]

            _, cache = forward(Xb, params)
            grads = backward(params, cache, yb)
            for key in params:
                params[key] -= lr * grads[key]

        if verbose:
            train_probs, _ = forward(X_train, params)
            loss = cross_entropy_loss(train_probs, y_train)
            acc = (train_probs.argmax(axis=1) == y_train).mean()
            print(f"epoch {ep + 1:2d}/{epochs}   loss {loss:.4f}   train acc {acc:.4f}")

    if verbose:
        print(f"training took {time.time() - t0:.1f}s")

    np.savez(WEIGHTS_PATH, **params)
    return params


# ---------- inference ----------

def predict(X, params_or_path):
    if isinstance(params_or_path, (str, Path)):
        data = np.load(params_or_path)
        params = {k: data[k] for k in data.files}
    else:
        params = params_or_path

    probs, _ = forward(X, params)
    return probs.argmax(axis=1), probs


if __name__ == "__main__":
    if "--gradcheck" in sys.argv:
        _gradient_check()
        sys.exit(0)

    X_train, y_train, X_test, y_test = load_data()
    print(f"train: {X_train.shape[0]} images   test: {X_test.shape[0]} images")

    params = train(X_train, y_train, epochs=15, lr=0.05, batch_size=32)

    preds, _ = predict(X_test, params)
    test_acc = (preds == y_test).mean()
    print(f"\ntest accuracy: {test_acc:.4f}")
    print(f"weights saved to {WEIGHTS_PATH}")

    print("\nsample predictions (loaded back from 02_weights.npz, to prove the save/load round-trip):")
    sample_preds, _ = predict(X_test[:8], WEIGHTS_PATH)
    for i in range(8):
        mark = "" if sample_preds[i] == y_test[i] else "  <-- wrong"
        print(f"  predicted {sample_preds[i]}   actual {y_test[i]}{mark}")
