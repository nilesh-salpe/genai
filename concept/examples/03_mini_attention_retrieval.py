"""
Minimal single-head attention solving toy key-value retrieval: given
n (key,value) pairs plus a QUERY key, output the value bound to the key
matching QUERY. Builds part-1-maths.html §11's five steps (score -> separate
Q from K -> softmax weights -> V-weighted retrieval -> 1/sqrt(d_k) scaling)
and backprops every one of them by hand with the §6-7 formulas (delta =
blame flowing backward, dL/dW = delta x^T, dL/db = delta, dL/dx = W^T
delta). No autograd, no ML framework -- just numpy.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Token scheme
# ---------------------------------------------------------------------------
# Each (key, value) pair occupies ONE sequence position, not two. A position
# is embedded from two small vocabularies added together:
#   E_key[key_id]          -- which key this position advertises
#   E_val[value_id]         -- which value this position carries (a sentinel
#                              "no value here" row is used at the query slot)
# plus a positional embedding E_pos[i] added to every position.
#
# Why not the more obvious [k1 v1 k2 v2 ... QUERY] layout, one token per key
# and a separate token per value? Because then the value that answers a
# query sits at a DIFFERENT sequence position than the key that identifies
# it (key_i at position 2i, value_i at position 2i+1). A single attention
# layer computes one V per position from that position's OWN content -- it
# cannot first locate the matching key and then hop one position over to
# read the value; that "find, then shift" is a two-step (two-layer)
# induction-head circuit, not a one-step lookup. Empirically, training the
# interleaved layout end-to-end never beats chance (verified: attention
# stayed at uniform weight over ~1500 gradient updates across every learning
# rate, batch size, and initialization tried). Binding a pair's key AND
# value to the SAME position turns it back into what §11 actually derives:
# one dot product decides relevance, one V delivers the payload -- a single
# hop, exactly what softmax(QK^T/sqrt(d_k))V computes.


def seq_len_for(n_pairs):
    return n_pairs + 1  # one position per (key,value) pair + one query position


def make_retrieval_task(n_pairs=4, key_vocab=6, value_vocab=6, n_samples=2000, seed=0):
    """Generate the toy retrieval task.

    Each sample: n_pairs distinct keys (no duplicates -> the task is
    well-posed, exactly one key matches the query) each bound to a distinct
    value (see note below), followed by a query position holding one of the
    n_pairs keys and no value. Target: the value bound to that key.

    Values are drawn WITHOUT replacement too. If they repeated, "always
    output the most common value in the sequence" becomes a shortcut that
    scores ~45% without ever looking at the query (verified empirically) --
    the model can hide in that local minimum and never learn to attend at
    all. Distinct values remove the shortcut, so the only way above chance
    (1/n_pairs) is genuine content-based retrieval.

    Returns
    -------
    key_ids   : (n_samples, seq_len) int array -- key id at every position
                (query slot holds the query key)
    val_ids   : (n_samples, seq_len) int array -- value id at every position
                (query slot holds the sentinel index `value_vocab`, meaning
                "no value here")
    targets   : (n_samples,) int array, raw value-vocab index (0..value_vocab-1)
    match_pos : (n_samples,) int array, the sequence position whose key
                matches the query -- i.e. where attention *should*
                concentrate. Not used in training; only for inspecting
                attention afterwards.
    """
    assert key_vocab >= n_pairs, "need >= n_pairs distinct keys to avoid duplicates"
    assert value_vocab >= n_pairs, "need >= n_pairs distinct values to remove the mode shortcut"
    rng = np.random.default_rng(seed)
    L = seq_len_for(n_pairs)
    NO_VALUE = value_vocab  # sentinel row in E_val: "this position carries no value"

    key_ids = np.empty((n_samples, L), dtype=np.int64)
    val_ids = np.empty((n_samples, L), dtype=np.int64)
    targets = np.empty(n_samples, dtype=np.int64)
    match_pos = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        keys = rng.choice(key_vocab, size=n_pairs, replace=False)
        values = rng.choice(value_vocab, size=n_pairs, replace=False)
        q_idx = rng.integers(0, n_pairs)                # which pair is queried

        key_ids[i, :n_pairs] = keys
        val_ids[i, :n_pairs] = values
        key_ids[i, n_pairs] = keys[q_idx]                # query: same key id as its pair
        val_ids[i, n_pairs] = NO_VALUE
        targets[i] = values[q_idx]
        match_pos[i] = q_idx

    return key_ids, val_ids, targets, match_pos


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def init_params(seed=0, d_model=16, n_pairs=4, key_vocab=6, value_vocab=6):
    """Small random init. d_k = d_model (single head, no subspace split)."""
    rng = np.random.default_rng(seed)
    L = seq_len_for(n_pairs)

    def small(*shape):
        return rng.normal(0, 1.0 / np.sqrt(shape[-1]), shape)

    params = {
        "E_key": rng.normal(0, 0.1, (key_vocab, d_model)),          # key embedding
        "E_val": rng.normal(0, 0.1, (value_vocab + 1, d_model)),     # value embedding (+1: "no value" sentinel)
        "E_pos": rng.normal(0, 0.1, (L, d_model)),                   # positional embedding
        "Wq": small(d_model, d_model),
        "Wk": small(d_model, d_model),
        "Wv": small(d_model, d_model),
        "Wo": small(value_vocab, d_model),                           # (out, in), matches part-2's convention
        "bo": np.zeros(value_vocab),
        "meta": {
            "d_model": d_model, "n_pairs": n_pairs,
            "key_vocab": key_vocab, "value_vocab": value_vocab, "seq_len": L,
        },
    }
    return params


# ---------------------------------------------------------------------------
# Forward: the five steps of §11, applied with a single query row
# ---------------------------------------------------------------------------
# We only ever need the attention *output at the query position* (the last
# slot) -- nothing downstream reads the other rows -- so instead of computing
# a full LxL self-attention matrix we compute one query vector against all L
# keys. Same five steps, same maths, one row of the QK^T matrix.

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # log-sum-exp trick: no overflow, same result
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_loss(probs, target):
    B = target.shape[0]
    return -np.mean(np.log(probs[np.arange(B), target] + 1e-12))


def forward(key_ids, val_ids, params):
    """key_ids, val_ids: (B, L) int arrays. Returns logits, attn weights, cache."""
    if key_ids.ndim == 1:
        key_ids = key_ids[None, :]
        val_ids = val_ids[None, :]
    B, L = key_ids.shape
    E_key, E_val, E_pos = params["E_key"], params["E_val"], params["E_pos"]
    Wq, Wk, Wv, Wo, bo = params["Wq"], params["Wk"], params["Wv"], params["Wo"], params["bo"]
    d_k = Wq.shape[1]

    X = E_key[key_ids] + E_val[val_ids] + E_pos[None, :, :]   # (B,L,d): key + value + position, one vector per pair
    x_query = X[:, -1, :]                               # (B,d): the last slot IS the query

    # Step 2 -- separate "what I seek" (Q) from "what I am" (K)
    Q = x_query @ Wq                                    # (B,d_k)   -- one query vector per sample
    K = X @ Wk                                           # (B,L,d_k) -- every position advertises itself
    # Step 4 (payload projection, computed here so it's ready for the weighted sum)
    V = X @ Wv                                           # (B,L,d_v)

    # Step 1 -- score relevance: dot product of query against every key
    # Step 5 -- scale by 1/sqrt(d_k) so softmax doesn't saturate
    scores = np.einsum("bd,bld->bl", Q, K) / np.sqrt(d_k)   # (B,L)

    # Step 3 -- turn scores into weights
    weights = softmax(scores, axis=-1)                   # (B,L), rows sum to 1

    # Step 4 -- weighted sum of V: what actually gets retrieved
    context = np.einsum("bl,bld->bd", weights, V)          # (B,d_v)

    logits = context @ Wo.T + bo                          # (B, value_vocab)
    probs = softmax(logits, axis=-1)

    cache = {"key_ids": key_ids, "val_ids": val_ids, "X": X, "x_query": x_query,
              "Q": Q, "K": K, "V": V, "scores": scores, "weights": weights,
              "context": context, "probs": probs}
    return logits, weights, cache


# ---------------------------------------------------------------------------
# Backward: every step of the forward pass, undone by hand
# ---------------------------------------------------------------------------

def backward(params, cache, target):
    Wq, Wk, Wv, Wo = params["Wq"], params["Wk"], params["Wv"], params["Wo"]
    X, weights, V, K, Q = cache["X"], cache["weights"], cache["V"], cache["K"], cache["Q"]
    context, probs, x_query = cache["context"], cache["probs"], cache["x_query"]
    key_ids, val_ids = cache["key_ids"], cache["val_ids"]
    B, L, d = X.shape
    d_k = Wq.shape[1]

    # ---- output projection + softmax + cross-entropy (§6-7: they collapse
    # to exactly predicted - actual, same identity as the MLP example) ----
    Y = np.zeros_like(probs)
    Y[np.arange(B), target] = 1
    delta_logits = (probs - Y) / B                        # (B, value_vocab)

    dWo = delta_logits.T @ context                        # dL/dW = delta x^T, summed over batch
    dbo = delta_logits.sum(axis=0)                         # dL/db = delta, summed over batch
    dcontext = delta_logits @ Wo                            # dL/dx = W^T delta (delta is a row, so W on the right)

    # ---- Step 4 backward: context = weights @ V (weighted sum over positions) ----
    # d context[b] / d weights[b,l] = V[b,l];  d context[b] / d V[b,l] = weights[b,l]
    dweights = np.einsum("bd,bld->bl", dcontext, V)         # (B,L)
    dV = weights[:, :, None] * dcontext[:, None, :]          # (B,L,d_v)

    # ---- Step 3 backward: softmax over the SEQUENCE axis (not classes) ----
    # Same identity as §5 (s_i(delta_i - sum_k s_k delta_k)), just applied
    # along positions l instead of along output classes.
    dot = (weights * dweights).sum(axis=-1, keepdims=True)
    dscores = weights * (dweights - dot)                    # (B,L)

    # ---- Step 1/5 backward: scores = (Q . K_l) / sqrt(d_k) ----
    dQ = np.einsum("bl,bld->bd", dscores, K) / np.sqrt(d_k)  # (B,d_k)
    dK = dscores[:, :, None] * Q[:, None, :] / np.sqrt(d_k)   # (B,L,d_k)

    # ---- Step 2/4 backward: the three projections, back to dL/dW = delta x^T ----
    dWq = x_query.T @ dQ                                     # (d, d_k)
    dWk = np.einsum("bld,ble->de", X, dK)                     # (d, d_k), summed over batch AND positions
    dWv = np.einsum("bld,ble->de", X, dV)                     # (d, d_v)

    # dL/dx = W^T delta, accumulated from all three paths (Q only touches the
    # last position; K and V touch every position)
    dX = dK @ Wk.T + dV @ Wv.T                                # (B,L,d)
    dX[:, -1, :] += dQ @ Wq.T

    # ---- embeddings: X = E_key[key_ids] + E_val[val_ids] + E_pos, so blame
    # is scattered back to whichever rows were actually looked up ----
    dE_key = np.zeros_like(params["E_key"])
    np.add.at(dE_key, key_ids, dX)                            # scatter-add: same id used more than once accumulates
    dE_val = np.zeros_like(params["E_val"])
    np.add.at(dE_val, val_ids, dX)
    dE_pos = dX.sum(axis=0)                                    # E_pos broadcasts over the batch -> sum blame over batch

    return {"E_key": dE_key, "E_val": dE_val, "E_pos": dE_pos,
            "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo, "bo": dbo}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_pairs=4, key_vocab=6, value_vocab=6, d_model=16, n_samples=4000,
          epochs=60, batch_size=64, lr=0.5, seed=0, verbose=True):
    key_ids, val_ids, targets, match_pos = make_retrieval_task(
        n_pairs=n_pairs, key_vocab=key_vocab, value_vocab=value_vocab,
        n_samples=n_samples, seed=seed)

    # held-out split for reporting
    n_val = max(200, n_samples // 10)
    val_key, val_val, val_targets = key_ids[:n_val], val_ids[:n_val], targets[:n_val]
    train_key, train_val, train_targets = key_ids[n_val:], val_ids[n_val:], targets[n_val:]

    params = init_params(seed=seed, d_model=d_model, n_pairs=n_pairs,
                          key_vocab=key_vocab, value_vocab=value_vocab)
    rng = np.random.default_rng(seed + 1)
    n = train_key.shape[0]
    grad_keys = ["E_key", "E_val", "E_pos", "Wq", "Wk", "Wv", "Wo", "bo"]

    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            logits, _, cache = forward(train_key[idx], train_val[idx], params)
            grads = backward(params, cache, train_targets[idx])
            for k in grad_keys:
                params[k] -= lr * grads[k]

        if verbose and ((ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1):
            val_logits, _, val_cache = forward(val_key, val_val, params)
            val_loss = cross_entropy_loss(val_cache["probs"], val_targets)
            val_acc = (val_logits.argmax(axis=1) == val_targets).mean()
            print(f"epoch {ep + 1:3d}   val loss {val_loss:.4f}   val acc {val_acc:.4f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_weights.npz")
    save_params(params, out_path)
    if verbose:
        print(f"saved weights to {out_path}")
    return params, (val_key, val_val, val_targets, match_pos[:n_val])


def save_params(params, path):
    meta = params["meta"]
    np.savez(path,
              E_key=params["E_key"], E_val=params["E_val"], E_pos=params["E_pos"],
              Wq=params["Wq"], Wk=params["Wk"], Wv=params["Wv"],
              Wo=params["Wo"], bo=params["bo"],
              meta_d_model=meta["d_model"], meta_n_pairs=meta["n_pairs"],
              meta_key_vocab=meta["key_vocab"], meta_value_vocab=meta["value_vocab"],
              meta_seq_len=meta["seq_len"])


def load_params(path):
    data = np.load(path)
    params = {k: data[k] for k in data.files if not k.startswith("meta_")}
    params["meta"] = {k[len("meta_"):]: int(data[k]) for k in data.files if k.startswith("meta_")}
    return params


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def encode_raw_sequence(pairs, query_key, key_vocab, value_vocab):
    """pairs: list of (key,value) ints. query_key: int. Builds (key_ids,
    val_ids) in the id scheme forward() expects, with the sentinel
    "no value" row appended at the query position."""
    n_pairs = len(pairs)
    key_ids = np.empty(n_pairs + 1, dtype=np.int64)
    val_ids = np.empty(n_pairs + 1, dtype=np.int64)
    for i, (k, v) in enumerate(pairs):
        key_ids[i], val_ids[i] = k, v
    key_ids[n_pairs] = query_key
    val_ids[n_pairs] = value_vocab   # sentinel
    return key_ids, val_ids


def predict(token_ids_or_raw_sequence, params_or_path, encode=True):
    """Run inference. When encode=True (default), `token_ids_or_raw_sequence`
    is a raw sequence: either a single (pairs, query_key) tuple where pairs
    is a list of (key,value) ints, or a list of such tuples for a batch.
    When encode=False, it is already-encoded (key_ids, val_ids) as produced
    by make_retrieval_task -- a single pair of (L,) arrays, or a pair of
    (B,L) arrays for a batch. Returns (predicted_value(s), attention_weights)."""
    params = load_params(params_or_path) if isinstance(params_or_path, str) else params_or_path
    meta = params["meta"]
    key_vocab, value_vocab = meta["key_vocab"], meta["value_vocab"]

    if encode:
        seq = token_ids_or_raw_sequence
        single = isinstance(seq, tuple) and len(seq) == 2 and isinstance(seq[0], (list, tuple))
        samples = [seq] if single else seq
        encoded = [encode_raw_sequence(pairs, q, key_vocab, value_vocab) for pairs, q in samples]
        key_ids = np.stack([e[0] for e in encoded])
        val_ids = np.stack([e[1] for e in encoded])
    else:
        key_ids, val_ids = token_ids_or_raw_sequence
        key_ids, val_ids = np.asarray(key_ids), np.asarray(val_ids)
        single = key_ids.ndim == 1
        if single:
            key_ids, val_ids = key_ids[None, :], val_ids[None, :]

    logits, weights, _ = forward(key_ids, val_ids, params)
    preds = logits.argmax(axis=1)
    if single:
        return int(preds[0]), weights[0]
    return preds, weights


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_PAIRS, KEY_VOCAB, VALUE_VOCAB, D_MODEL = 4, 6, 6, 16

    params, (val_key, val_val, val_targets, val_match_pos) = train(
        n_pairs=N_PAIRS, key_vocab=KEY_VOCAB, value_vocab=VALUE_VOCAB, d_model=D_MODEL,
        n_samples=4000, epochs=60, batch_size=64, lr=0.5, seed=0)

    val_logits, val_weights, _ = forward(val_key, val_val, params)
    final_acc = (val_logits.argmax(axis=1) == val_targets).mean()
    print(f"\nfinal held-out accuracy: {final_acc:.4f}")

    # spot-check: does attention actually land on the matching key?
    concentration = val_weights[np.arange(len(val_match_pos)), val_match_pos]
    print(f"mean attention weight on the correct pair position: {concentration.mean():.4f}")

    print("\n--- sample predictions ---")
    demo_key, demo_val, demo_targets, demo_match_pos = make_retrieval_task(
        n_pairs=N_PAIRS, key_vocab=KEY_VOCAB, value_vocab=VALUE_VOCAB, n_samples=6, seed=99)

    demo_logits, demo_weights, _ = forward(demo_key, demo_val, params)
    demo_preds = demo_logits.argmax(axis=1)

    for i in range(len(demo_key)):
        keys = demo_key[i, :N_PAIRS]
        values = demo_val[i, :N_PAIRS]
        query_key = demo_key[i, -1]
        w = demo_weights[i]
        pred = demo_preds[i]

        readable = " ".join(f"k{keys[j]}v{values[j]}" for j in range(N_PAIRS)) + f" | Q=k{query_key}"
        bars = " ".join(f"{x:.2f}" for x in w)
        marker_row = " ".join("^^^^" if j == demo_match_pos[i] else "    " for j in range(len(w)))
        correct = "OK" if pred == demo_targets[i] else "WRONG"
        print(f"seq: {readable}")
        print(f"  predicted value = {pred}   actual value = {demo_targets[i]}   [{correct}]")
        print(f"  attn weights   = [{bars}]")
        print(f"  match position = [{marker_row}]  (position {demo_match_pos[i]} is the matching pair)\n")
