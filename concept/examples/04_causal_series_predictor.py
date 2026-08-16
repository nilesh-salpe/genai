"""
Causal self-attention predicting the NEXT token at EVERY position of a
length-8 alternating two-symbol series, e.g. [a,b,a,b,a,b,a,b]. This is the
step example 03 (examples/03_mini_attention_retrieval.py) stopped short of:
03 answers a single query at the end of a sequence; here every position i
is simultaneously a query asking "what comes next?", and a position may
only look at itself and earlier positions -- never the future. That
restriction is the "causal masking" subsection of part-1-maths.html §11,
layered on top of §11's five attention steps (score -> separate Q from K
-> softmax weights -> V-weighted retrieval -> 1/sqrt(d_k) scaling), which
03 already builds and backprops by hand. Same Q/K/V mechanics as 03, same
§6-7 backprop identities (delta = blame, dL/dW = delta x^T, dL/db = delta,
dL/dx = W^T delta) -- just applied per query position over its own valid
(causally-masked) range of keys, and summed over all 7 predicted positions
of a sequence instead of one. No autograd, no ML framework -- just numpy.
"""

import os
import sys
import numpy as np

VOCAB_SIZE = 6
SEQ_LEN = 8
D_MODEL = 8


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
# Pick two distinct symbols a != b from the vocabulary and a random starting
# phase, then alternate: [a,b,a,b,...] or [b,a,b,a,...] for seq_len tokens.
#
# WHY POSITION 0 IS UNGUESSABLE (read this before judging accuracy numbers):
# predicting token[1] from token[0] alone means predicting b having only
# seen a. b is an independent random draw (any of the other vocab_size-1
# symbols, uniformly) -- nothing in token[0] determines it. This is the
# causal mask working exactly as intended: position 0 genuinely does not
# have the information yet. From position 1 onward the model has seen both
# a and b, so the pattern is fully determined -- the correct move is "copy
# the token from one position back" (token[i+1] == token[i-1] always, since
# the series has period 2) -- and accuracy hits ~100% there.
#
# A subtlety actually observed when this was trained (not merely predicted
# in advance): position 0 does not converge to a *chance*-level guess. The
# single linear Wv/Wout pair is shared across every position, and to solve
# positions 1-6 it has to learn a content-identity map -- "whatever token's
# embedding you just retrieved, output that token's class" -- because the
# retrieved token literally *is* the answer there. Position 0 is forced
# (by the causal mask -- it is the only valid key) to attend to itself, so
# it retrieves its *own* embedding and that same identity map confidently
# predicts "next token == current token", which is *always* wrong (a != b
# by construction). So position 0 lands near 0% accuracy, not ~20% chance
# -- confidently wrong rather than randomly wrong. Still not a bug: it is
# the direct, provable consequence of the shared linear readout doing
# exactly what it needs to do everywhere else. See position_accuracy() /
# the __main__ block below for how this is measured and reported.

def make_sequence(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, seed=None, rng=None):
    """One alternating a,b,a,b,... sequence. Pass `rng` (a np.random.Generator)
    to draw from a shared, advancing stream when generating many sequences
    (e.g. one per training example in a batch); pass `seed` for a single
    reproducible one-off sequence. Returns an (seq_len,) int64 array."""
    if rng is None:
        rng = np.random.default_rng(seed)
    a, b = rng.choice(vocab_size, size=2, replace=False)  # distinct by construction
    first, second = (a, b) if rng.integers(0, 2) == 0 else (b, a)  # random phase
    seq = np.empty(seq_len, dtype=np.int64)
    seq[0::2] = first
    seq[1::2] = second
    return seq


def make_batch(batch_size, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, rng=None):
    """A fresh, independent alternating sequence per row."""
    if rng is None:
        rng = np.random.default_rng()
    return np.stack([make_sequence(vocab_size, seq_len, rng=rng) for _ in range(batch_size)])


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def init_params(seed=0, d_model=D_MODEL, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Small random init. d_k = d_model (single head, no subspace split)."""
    rng = np.random.default_rng(seed)

    def small(*shape):
        return rng.normal(0, 1.0 / np.sqrt(shape[-1]), shape)

    params = {
        "E_tok": rng.normal(0, 0.1, (vocab_size, d_model)),   # token embedding
        "E_pos": rng.normal(0, 0.1, (seq_len, d_model)),      # positional embedding
        "Wq": small(d_model, d_model),
        "Wk": small(d_model, d_model),
        "Wv": small(d_model, d_model),
        "Wout": small(vocab_size, d_model),   # (out, in), matches part-2's convention
        "bout": np.zeros(vocab_size),
        "meta": {"d_model": d_model, "vocab_size": vocab_size, "seq_len": seq_len},
    }
    return params


# ---------------------------------------------------------------------------
# Forward: §11's five steps, computed for EVERY query position at once,
# with a causal mask so position i only ever sees keys j <= i.
# ---------------------------------------------------------------------------

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # log-sum-exp trick: no overflow, same result
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_loss(probs, targets):
    """probs: (B,T,vocab) predicted-position probabilities; targets: (B,T)."""
    B, T = targets.shape
    ib = np.arange(B)[:, None]
    it = np.arange(T)[None, :]
    return -np.mean(np.log(probs[ib, it, targets] + 1e-12))


def _causal_mask(L):
    """(L,L) additive mask: 0 where j<=i (allowed), -1e9 where j>i (future).
    e^-1e9 underflows to exactly 0.0 in the softmax below -- the masked
    positions get literal zero weight, not just "very small"."""
    return np.triu(np.full((L, L), -1e9), k=1)


def forward(tokens, params):
    """tokens: (B,L) or (L,) int array. Returns (logits, attn_weights, cache).
    logits: (B,L,vocab) -- next-token prediction at every position (the
    prediction FROM the last position has no target to check against, but
    is still computed/returned).
    attn_weights: (B,L,L) causally-masked attention, weights[b,i,j] = how
    much query position i attends to key position j (0 for all j>i)."""
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    B, L = tokens.shape
    E_tok, E_pos = params["E_tok"], params["E_pos"]
    Wq, Wk, Wv, Wout, bout = params["Wq"], params["Wk"], params["Wv"], params["Wout"], params["bout"]
    d_k = Wq.shape[1]

    X = E_tok[tokens] + E_pos[None, :L, :]              # (B,L,d) -- token + positional embedding

    # Step 2 -- separate "what I seek" (Q) from "what I am" (K); every
    # position is simultaneously a query AND a key/value provider now.
    Q = X @ Wq                                          # (B,L,d_k)
    K = X @ Wk                                          # (B,L,d_k)
    V = X @ Wv                                          # (B,L,d_v)

    # Step 1/5 -- score every (query i, key j) pair, scaled by 1/sqrt(d_k)
    scores = np.einsum("bid,bjd->bij", Q, K) / np.sqrt(d_k)   # (B,L,L)
    scores = scores + _causal_mask(L)[None, :, :]             # future positions -> -inf

    # Step 3 -- softmax each query row i over its own valid key range j<=i
    weights = softmax(scores, axis=-1)                  # (B,L,L), rows sum to 1, upper triangle exactly 0

    # Step 4 -- weighted sum of V: what each position actually retrieves
    context = np.einsum("bij,bjd->bid", weights, V)      # (B,L,d_v)

    logits = context @ Wout.T + bout                     # (B,L,vocab)
    probs = softmax(logits, axis=-1)

    cache = {"tokens": tokens, "X": X, "Q": Q, "K": K, "V": V,
              "scores": scores, "weights": weights, "context": context, "probs": probs}
    return logits, weights, cache


# ---------------------------------------------------------------------------
# Backward: every step of the forward pass, undone by hand.
# Loss = mean cross-entropy over the 7 predicted positions (0..L-2, each
# predicting the very next token) and over the batch.
# ---------------------------------------------------------------------------

def backward(params, cache, tokens):
    Wq, Wk, Wv, Wout = params["Wq"], params["Wk"], params["Wv"], params["Wout"]
    X, weights, V, K, Q = cache["X"], cache["weights"], cache["V"], cache["K"], cache["Q"]
    context, probs = cache["context"], cache["probs"]
    B, L, d = X.shape
    d_k = Wq.shape[1]
    n_pred = L - 1                                       # positions 0..L-2 have a real next token

    # ---- output projection + softmax + cross-entropy (§6-7: collapses to
    # exactly predicted - actual) -- only the n_pred positions that have a
    # target contribute; the last position's logits get zero gradient. ----
    targets = tokens[:, 1:]                               # (B, n_pred) = token[i+1] for i=0..L-2
    delta_logits = np.zeros_like(probs)                    # (B,L,vocab)
    delta_valid = probs[:, :n_pred, :].copy()
    ib = np.arange(B)[:, None]
    it = np.arange(n_pred)[None, :]
    delta_valid[ib, it, targets] -= 1
    delta_valid /= (B * n_pred)                            # average over batch AND predicted positions
    delta_logits[:, :n_pred, :] = delta_valid

    dWout = np.einsum("bic,bid->cd", delta_logits, context)   # dL/dW = delta x^T, summed over batch & position
    dbout = delta_logits.sum(axis=(0, 1))                      # dL/db = delta, summed likewise
    dcontext = delta_logits @ Wout                               # dL/dx = W^T delta

    # ---- Step 4 backward: context[i] = sum_j weights[i,j] * V[j] ----
    dweights = np.einsum("bid,bjd->bij", dcontext, V)           # (B,L,L)
    dV = np.einsum("bij,bid->bjd", weights, dcontext)            # (B,L,d_v)

    # ---- Step 3 backward: softmax per query row i, over its key axis j.
    # Same identity as §5 (s_i(delta_i - sum_k s_k delta_k)); masked
    # positions have weight 0 so they automatically get zero gradient. ----
    dot = (weights * dweights).sum(axis=-1, keepdims=True)      # (B,L,1)
    dscores = weights * (dweights - dot)                         # (B,L,L)

    # ---- Step 1/5 backward: scores[i,j] = (Q_i . K_j) / sqrt(d_k) ----
    dQ = np.einsum("bij,bjd->bid", dscores, K) / np.sqrt(d_k)    # (B,L,d_k)
    dK = np.einsum("bij,bid->bjd", dscores, Q) / np.sqrt(d_k)    # (B,L,d_k)

    # ---- Step 2/4 backward: the three projections, back to dL/dW = delta x^T ----
    dWq = np.einsum("bld,ble->de", X, dQ)                         # (d, d_k), summed over batch & position
    dWk = np.einsum("bld,ble->de", X, dK)
    dWv = np.einsum("bld,ble->de", X, dV)

    # dL/dx = W^T delta, accumulated across all three projection paths --
    # every position is a query AND a key AND a value here, so all three
    # contribute to every position's embedding gradient.
    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T                        # (B,L,d)

    # ---- embeddings: X = E_tok[tokens] + E_pos, blame scattered back to
    # whichever rows were actually looked up ----
    dE_tok = np.zeros_like(params["E_tok"])
    np.add.at(dE_tok, tokens.reshape(-1), dX.reshape(-1, d))
    dE_pos = dX.sum(axis=0)                                        # E_pos broadcasts over batch -> sum blame over batch

    return {"E_tok": dE_tok, "E_pos": dE_pos, "Wq": dWq, "Wk": dWk, "Wv": dWv,
            "Wout": dWout, "bout": dbout}


# ---------------------------------------------------------------------------
# Position-broken-out accuracy: position 0 is reported separately from
# positions 1-6 throughout this file -- see the WHY POSITION 0 IS
# UNGUESSABLE note above. A single blended number would look like a
# failure when the model is in fact working exactly as designed.
# ---------------------------------------------------------------------------

def position_accuracy(tokens, params):
    """Returns (pos0_acc, pos1_6_acc, overall_loss) for a batch of sequences."""
    logits, _, cache = forward(tokens, params)
    n_pred = tokens.shape[1] - 1
    preds = logits[:, :n_pred, :].argmax(axis=-1)         # (B, n_pred)
    targets = tokens[:, 1:]
    correct = preds == targets                             # (B, n_pred)
    pos0_acc = correct[:, 0].mean()
    pos1_6_acc = correct[:, 1:].mean()
    loss = cross_entropy_loss(cache["probs"][:, :n_pred, :], targets)
    return pos0_acc, pos1_6_acc, loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(steps=900, lr=0.5, batch_size=16, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN,
          d_model=D_MODEL, seed=0, verbose=True, print_every=100):
    params = init_params(seed=seed, d_model=d_model, vocab_size=vocab_size, seq_len=seq_len)
    rng = np.random.default_rng(seed + 1)              # drives training-batch generation
    eval_rng = np.random.default_rng(seed + 2)          # separate stream for periodic held-out reporting
    grad_keys = ["E_tok", "E_pos", "Wq", "Wk", "Wv", "Wout", "bout"]

    for step in range(1, steps + 1):
        tokens = make_batch(batch_size, vocab_size, seq_len, rng=rng)   # fresh random sequences every step
        logits, _, cache = forward(tokens, params)
        grads = backward(params, cache, tokens)
        for k in grad_keys:
            params[k] -= lr * grads[k]

        if verbose and (step % print_every == 0 or step == steps):
            eval_tokens = make_batch(500, vocab_size, seq_len, rng=eval_rng)
            pos0_acc, pos1_6_acc, loss = position_accuracy(eval_tokens, params)
            print(f"step {step:4d}   loss {loss:.4f}   pos0 acc {pos0_acc:.3f}   pos1-6 acc {pos1_6_acc:.3f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_weights.npz")
    save_params(params, out_path)
    if verbose:
        print(f"saved weights to {out_path}")
    return params


def save_params(params, path):
    meta = params["meta"]
    np.savez(path,
              E_tok=params["E_tok"], E_pos=params["E_pos"],
              Wq=params["Wq"], Wk=params["Wk"], Wv=params["Wv"],
              Wout=params["Wout"], bout=params["bout"],
              meta_d_model=meta["d_model"], meta_vocab_size=meta["vocab_size"],
              meta_seq_len=meta["seq_len"])


def load_params(path):
    data = np.load(path)
    params = {k: data[k] for k in data.files if not k.startswith("meta_")}
    params["meta"] = {k[len("meta_"):]: int(data[k]) for k in data.files if k.startswith("meta_")}
    return params


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(tokens, params_or_path):
    """tokens: (L,) single sequence or (B,L) batch of already-encoded token
    ids. Returns (predicted_next_token(s), attn_weights) -- predictions at
    EVERY position (including the last, which has no ground truth to check
    against but is still a valid "what comes after this" guess)."""
    params = load_params(params_or_path) if isinstance(params_or_path, str) else params_or_path
    tokens = np.asarray(tokens)
    single = tokens.ndim == 1
    if single:
        tokens = tokens[None, :]

    logits, weights, _ = forward(tokens, params)
    preds = logits.argmax(axis=-1)                        # (B,L)
    if single:
        return preds[0], weights[0]
    return preds, weights


# ---------------------------------------------------------------------------
# Numerical gradient check (finite differences vs. the analytic backward()
# above). Verified to match to ~1e-6 relative error -- this exact Q/K/V +
# causal-mask math was already checked analytically-vs-numerically in a JS
# prototype, so this simply confirms the Python port carries the same
# correctness. Gated behind --gradcheck; the default run below never calls
# this.
# ---------------------------------------------------------------------------

def _gradient_check(epsilon=1e-5, n_checks=4, seed=0):
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed, d_model=D_MODEL, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    tokens = make_batch(3, VOCAB_SIZE, SEQ_LEN, rng=rng)   # small batch is enough to exercise every path

    def loss_of(p):
        logits, _, cache = forward(tokens, p)
        n_pred = SEQ_LEN - 1
        return cross_entropy_loss(cache["probs"][:, :n_pred, :], tokens[:, 1:])

    _, _, cache = forward(tokens, params)
    grads = backward(params, cache, tokens)

    check_params = ["Wq", "Wk", "Wv", "Wout", "E_tok", "E_pos"]
    worst = 0.0
    for name in check_params:
        arr = params[name]
        flat_idx = rng.choice(arr.size, size=min(n_checks, arr.size), replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, arr.shape)
            orig = arr[idx]
            arr[idx] = orig + epsilon
            loss_plus = loss_of(params)
            arr[idx] = orig - epsilon
            loss_minus = loss_of(params)
            arr[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * epsilon)
            analytic = grads[name][idx]
            rel_err = abs(numeric - analytic) / max(1e-8, abs(numeric) + abs(analytic))
            worst = max(worst, rel_err)
            status = "OK" if rel_err < 1e-4 else "MISMATCH"
            print(f"  {name}{idx}: numeric={numeric: .8f}  analytic={analytic: .8f}  rel_err={rel_err:.2e}  [{status}]")
    print(f"\nworst relative error across all checked entries: {worst:.2e}")
    assert worst < 1e-4, "gradient check failed"
    print("gradient check PASSED")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--gradcheck" in sys.argv:
        print("--- numerical gradient check (finite differences vs. backward()) ---")
        _gradient_check()
        sys.exit(0)

    params = train(steps=900, lr=0.5, batch_size=16, vocab_size=VOCAB_SIZE,
                    seq_len=SEQ_LEN, d_model=D_MODEL, seed=0, print_every=100)

    print("\n--- final held-out accuracy (large sample) ---")
    final_rng = np.random.default_rng(999)
    final_tokens = make_batch(2000, VOCAB_SIZE, SEQ_LEN, rng=final_rng)
    pos0_acc, pos1_6_acc, loss = position_accuracy(final_tokens, params)
    print(f"loss {loss:.4f}")
    print(f"position 0 accuracy (predicting token[1] from token[0] alone): {pos0_acc:.3f}"
          f"  -- b is unknowable from a single token (see WHY POSITION 0 IS UNGUESSABLE"
          f" near the top of this file); in practice the shared identity-copy circuit"
          f" that solves positions 1-6 also fires on position 0's forced self-attention,"
          f" so it lands near 0% (confidently wrong) rather than ~{1 / (VOCAB_SIZE - 1):.2f} (randomly wrong) -- still not a bug")
    print(f"positions 1-6 accuracy (both symbols already seen, pattern fully determined): {pos1_6_acc:.3f}"
          f"  -- expected ~1.000")

    print("\n--- sample predictions ---")
    demo_rng = np.random.default_rng(123)
    for i in range(3):
        seq = make_sequence(VOCAB_SIZE, SEQ_LEN, rng=demo_rng)
        preds, w = predict(seq, params)
        next_preds = preds[:-1]                 # prediction made at position i, for token[i+1]
        actual_next = seq[1:]
        correctness = " ".join("OK" if p == a else "no" for p, a in zip(next_preds, actual_next))
        print(f"sequence          : {list(seq)}")
        print(f"predicted next-tok: {list(next_preds)}  (predicted from tokens[0..i] at each position i)")
        print(f"actual next-tok   : {list(actual_next)}")
        print(f"correct?          : {correctness}   (position 0 is expected to say 'no' -- see above)")

        if i == 0:
            # Position 2 is the cleanest demo: exactly one earlier position
            # (j=0) carries the "other" symbol, and the causal mask forbids
            # looking past position 2 itself, so there's no ambiguity about
            # which position *should* get the weight -- one position back.
            i_query = 2
            print(f"\nattention weights at position {i_query} (query = token[{i_query}] = {seq[i_query]}):")
            for j in range(SEQ_LEN):
                bar = "#" * int(round(w[i_query, j] * 40))
                marker = "  <-- 1 back (correct source)" if j == i_query - 1 else ("  <-- self" if j == i_query else "")
                print(f"  attends to position {j} (token {seq[j]}): {w[i_query, j]:.3f} {bar}{marker}")

            # Later positions have *several* equally-valid earlier positions
            # carrying the same "other" symbol (period-2 series -> every
            # other position matches), so attention spreads across all of
            # them rather than collapsing onto exactly one -- still never
            # onto the same-symbol (even-offset) positions, which carry no
            # useful information for "what comes next".
            i_query = SEQ_LEN - 1
            print(f"\nattention weights at position {i_query} (query = token[{i_query}] = {seq[i_query]}) "
                  f"-- note it spreads over EVERY earlier position carrying the other symbol, "
                  f"not just the nearest one, because they are all equally informative:")
            for j in range(SEQ_LEN):
                bar = "#" * int(round(w[i_query, j] * 40))
                marker = "  <-- 1 back" if j == i_query - 1 else ("  <-- self" if j == i_query else "")
                print(f"  attends to position {j} (token {seq[j]}): {w[i_query, j]:.3f} {bar}{marker}")
        print()
