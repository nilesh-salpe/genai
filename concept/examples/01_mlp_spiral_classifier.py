"""2-32-32-3 MLP on a 3-arm spiral. Every gradient below is the hand-derived
formula from part-2-build.html §6-7 (backprop, shapes) and part-1-maths.html
§5-7 (why depth needs a kink, softmax, cross-entropy): delta = blame,
dL/dW = delta x^T, dL/db = delta, dL/dx = W^T delta. No autograd.
"""

import os

import numpy as np

SIZES = [2, 32, 32, 3]


# ---------- data ----------

def make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=0):
    # Classic CS231n spiral: radius grows linearly along each arm while the
    # angle sweeps ~2 turns, offset per class so the arms interleave. A
    # straight line cannot separate this -- that is the whole point of the
    # kink (part-1 §5): only a nonlinearity between layers can bend a
    # decision boundary around a spiral arm.
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=int)
    for c in range(n_classes):
        ix = slice(n_per_class * c, n_per_class * (c + 1))
        # A single parameter t drives both r and theta so the curve's shape
        # is independent of n_per_class (only the sampling density changes).
        # t = (i + 0.5) / n_per_class keeps t in (0, 1], i.e. r > 0 always --
        # r = 0 exactly would put every class's first point at the same
        # origin (0,0) with different labels, a contradiction, and would
        # also sit right on the ReLU's non-differentiable kink at init.
        t = (np.arange(n_per_class) + 0.5) / n_per_class
        r = t                                                  # 0 < r <= 1 along the arm
        theta = (t * 4 * np.pi                                  # ~2 turns
                 + c * (2 * np.pi / n_classes)                  # spread the arms out
                 + rng.normal(0, noise, n_per_class))           # small angular jitter
        X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[ix] = c
    return X, y


# ---------- init ----------

def init_params(seed=0):
    # He init: std = sqrt(2/n_in), as in part-2 §9. Zero init is symmetric --
    # every hidden unit would compute the same gradient forever -- so the
    # randomness is load-bearing, the scale just keeps activations sane.
    rng = np.random.default_rng(seed)

    def he(n_in, n_out):
        return rng.normal(0, np.sqrt(2.0 / n_in), (n_out, n_in))  # (out, in)

    params = {}
    for idx, (n_in, n_out) in enumerate(zip(SIZES[:-1], SIZES[1:]), start=1):
        params[f"W{idx}"] = he(n_in, n_out)
        params[f"b{idx}"] = np.zeros(n_out)
    return params


# ---------- forward ----------

def relu(z):
    return np.maximum(0, z)


def softmax(logits):
    # Subtract the row max before exponentiating (part-1 §6): shifts every
    # logit by a constant, which cancels in the ratio, and stops exp()
    # overflowing on large inputs.
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, params):
    # Save every intermediate -- backward needs Z (for the ReLU mask) and
    # A (as the "x" in dL/dW = delta x^T) at each layer (part-2 §6).
    Z1 = X @ params["W1"].T + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"].T + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"].T + params["b3"]
    probs = softmax(Z3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "probs": probs}
    return Z3, cache


def cross_entropy_loss(probs, y):
    B = y.shape[0]
    # probs[arange(B), y]: the predicted probability of the true class,
    # read out for all B examples at once (part-1 §7).
    return -np.mean(np.log(probs[np.arange(B), y] + 1e-12))  # +eps: never log(0)


# ---------- backward ----------

def backward(params, cache, y):
    # Hand-derived chain rule only -- no autograd, no numerical gradients
    # here (those are for the standalone check, see check_gradients below).
    B = y.shape[0]
    Y = np.zeros_like(cache["probs"])
    Y[np.arange(B), y] = 1  # one-hot true class per row

    # softmax + cross-entropy collapse to exactly (predicted - actual),
    # the delta ("blame") for the output layer (part-1 §7, part-2 §6).
    delta3 = (cache["probs"] - Y) / B  # /B averages the loss over the batch

    dW3 = delta3.T @ cache["A2"]       # dL/dW = delta x^T
    db3 = delta3.sum(axis=0)           # dL/db = delta

    # dL/dx = W^T delta, then the ReLU mask blocks blame at units that were
    # off during the forward pass (they contributed nothing, so they get
    # no blame). delta is a row per example here, so it's `@ W` rather than
    # `W.T @ delta` -- same formula, transposed layout.
    delta2 = (delta3 @ params["W3"]) * (cache["Z2"] > 0)
    dW2 = delta2.T @ cache["A1"]
    db2 = delta2.sum(axis=0)

    delta1 = (delta2 @ params["W2"]) * (cache["Z1"] > 0)
    dW1 = delta1.T @ cache["X"]
    db1 = delta1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


# ---------- train ----------

def train(epochs=4000, lr=0.5, n_per_class=100, n_classes=3, noise=0.2,
          seed=0, print_every=250, save_path=None):
    X, y = make_spiral(n_per_class, n_classes, noise, seed)
    params = init_params(seed)

    for epoch in range(epochs):
        logits, cache = forward(X, params)
        grads = backward(params, cache, y)
        for k in params:
            params[k] -= lr * grads[k]  # plain full-batch gradient descent

        if epoch % print_every == 0 or epoch == epochs - 1:
            loss = cross_entropy_loss(cache["probs"], y)
            acc = (cache["probs"].argmax(axis=1) == y).mean()
            print(f"epoch {epoch:4d}   loss {loss:.4f}   train acc {acc:.4f}")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_weights.npz")
    np.savez(save_path, **params)
    print(f"saved weights to {save_path}")

    return params


# ---------- predict ----------

def predict(X, params_or_path):
    if isinstance(params_or_path, (str, os.PathLike)):
        with np.load(params_or_path) as data:
            params = {k: data[k] for k in data.files}
    else:
        params = params_or_path

    logits, cache = forward(X, params)
    probs = cache["probs"]
    return probs.argmax(axis=1), probs


# ---------- numerical gradient check (optional, off by default) ----------

def check_gradients(seed=1, n_checks=3, eps=1e-5):
    # Central-difference check of backward() against forward() alone --
    # the same idea as part-2's "verify with a tiny numerical check"
    # sidebar. This is *not* how the network trains; it only proves the
    # hand-derived formulas above are correct.
    rng = np.random.default_rng(seed)
    X, y = make_spiral(n_per_class=5, n_classes=3, noise=0.2, seed=seed)
    params = init_params(seed)
    _, cache = forward(X, params)
    grads = backward(params, cache, y)

    def loss_at(params):
        _, cache = forward(X, params)
        return cross_entropy_loss(cache["probs"], y)

    worst_rel_err = 0.0
    for name in params:
        shape = params[name].shape
        idxs = [tuple(rng.integers(0, d) for d in shape) for _ in range(n_checks)]
        for idx in idxs:
            orig = params[name][idx]

            params[name][idx] = orig + eps
            loss_plus = loss_at(params)
            params[name][idx] = orig - eps
            loss_minus = loss_at(params)
            params[name][idx] = orig  # restore

            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = grads[name][idx]
            rel_err = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-8)
            worst_rel_err = max(worst_rel_err, rel_err)
            print(f"{name}{idx}: analytic {analytic:+.6e}  numeric {numeric:+.6e}  rel_err {rel_err:.2e}")

    print(f"worst relative error: {worst_rel_err:.2e}")
    return worst_rel_err


if __name__ == "__main__":
    import sys

    if "--check-grad" in sys.argv:
        err = check_gradients()
        print("PASS" if err < 1e-6 else "FAIL", "(threshold 1e-6)")
        sys.exit(0)

    params = train(epochs=4000, lr=0.5, print_every=250)

    X, y = make_spiral(n_per_class=100, n_classes=3, noise=0.2, seed=0)
    preds, _ = predict(X, params)
    acc = (preds == y).mean()
    print(f"\nfinal train accuracy: {acc:.4f}")

    # A handful of new points (seed=99), never seen during training.
    X_new, y_new = make_spiral(n_per_class=5, n_classes=3, noise=0.2, seed=99)
    preds_new, probs_new = predict(X_new, params)
    print("\nsample predictions on new points:")
    for i in range(len(X_new)):
        confidence = probs_new[i, preds_new[i]]
        mark = "ok" if preds_new[i] == y_new[i] else "MISS"
        print(f"  x={X_new[i]}  true={y_new[i]}  pred={preds_new[i]}  "
              f"confidence={confidence:.3f}  [{mark}]")
