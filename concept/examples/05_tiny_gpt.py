"""
A small, real, decoder-only Transformer -- character-level, trained on a
handful of sentences (the same "The animal didn't cross the street because
it was too tired" example part-5-transformer-architecture.html §01 uses),
generating new text one sampled character at a time.

Unlike 01-04 in this folder, this one is intentionally NOT hand-derived
NumPy -- it is PyTorch, on purpose. part-3-pytorch.html's whole argument is
that autograd exists so you stop re-deriving backward passes by hand once
you already understand what one is; this file is that argument, applied.
Every *architectural* piece is still exactly what part-5-transformer-
architecture.html §02-06 derive -- sinusoidal positional encoding, scaled
dot-product causal self-attention, the pre-norm residual block, and (§08)
temperature / top-k / top-p sampling -- just assembled with nn.Module and
trained with five calls (part-3-pytorch.html §05-06) instead of a
hand-written backward() function.

Run directly: python3 05_tiny_gpt.py
Needs only PyTorch (pip install torch).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Corpus and tokenizer -- character-level, on purpose
# ---------------------------------------------------------------------------
# A real model uses subword tokens (BPE) so it doesn't need one embedding
# row per possible word. At this toy scale that machinery buys nothing --
# a vocabulary of ~30 characters trains in seconds on a CPU and every step
# stays inspectable. See part-6-llm.html §02 for the actual trade-off.

CORPUS = """
the animal didn't cross the street because it was too tired.
the animal was too tired to cross the street.
the cat didn't cross the street because it was too scared.
the dog didn't stop because it was too excited.
the animal was too fast to catch.
""".strip().lower()

chars = sorted(set(CORPUS))
VOCAB_SIZE = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


DATA = encode(CORPUS)

# ---------------------------------------------------------------------------
# Model hyperparameters -- deliberately tiny; this is a demo, not a model
# ---------------------------------------------------------------------------
D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS       # part-5-transformer-architecture.html §05: d_k = d_model / h
N_BLOCKS = 3
FFN_HIDDEN = 4 * D_MODEL          # the paper's 4x ratio (part-5-transformer-architecture.html §06), scaled down
BLOCK_SIZE = 48                   # max context length this model was trained on
DROPOUT = 0.0


def sinusoidal_positional_encoding(max_len, d_model):
    """Exactly part-5-transformer-architecture.html §02's PE(pos,2i)=sin(...),
    PE(pos,2i+1)=cos(...) -- a fixed buffer, no parameters, computed once."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CausalSelfAttention(nn.Module):
    """part-5-transformer-architecture.html §03: softmax(QK^T/sqrt(d_k))V,
    masked so position i never sees j > i (§04's causal mask). Multi-head: h
    independent (Q,K,V) triples run in parallel and get concatenated
    (§05's multi-head attention)."""

    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        # torch.tril: part-5-transformer-architecture.html §04's causal mask,
        # built once and reused -- registered as a buffer so .to(device) moves it too.
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, D = x.shape
        H, DH = self.n_heads, self.d_head

        def split_heads(t):
            return t.view(B, T, H, DH).transpose(1, 2)  # (B, H, T, DH)

        q, k, v = split_heads(self.wq(x)), split_heads(self.wk(x)), split_heads(self.wv(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(DH)          # (B, H, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v                                          # (B, H, T, DH)
        out = out.transpose(1, 2).contiguous().view(B, T, D)        # concat heads
        return self.wo(out)


class DecoderBlock(nn.Module):
    """Pre-norm residual block, part-5-transformer-architecture.html §06's
    exact equation: x' = x + MHA(LN(x));  x'' = x' + FFN(LN(x'))."""

    def __init__(self, d_model, n_heads, ffn_hidden, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_blocks, ffn_hidden, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos_emb", sinusoidal_positional_encoding(block_size, d_model))
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, ffn_hidden, block_size) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)  # logits -- one score per vocab char

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:T]     # embedding + positional encoding, added
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                           # (B, T, vocab_size) logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """part-5-transformer-architecture.html §08: logits -> temperature ->
        (optional) top-k / top-p filtering -> softmax -> sample -> feed the
        sampled token back in as the newest piece of context. Autoregressive,
        one character at a time -- generation cannot skip ahead any more than
        a causal decoder can attend ahead (§04)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)  # last position only

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cum - sorted_probs) > top_p  # keep the first crossing
                sorted_probs[cutoff] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

            next_id = torch.multinomial(probs, num_samples=1)  # the actual sampling step
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Training -- part-3-pytorch.html §05's five calls, nothing new
# ---------------------------------------------------------------------------

def get_batch(batch_size, block_size):
    starts = torch.randint(0, len(DATA) - block_size - 1, (batch_size,))
    x = torch.stack([DATA[s:s + block_size] for s in starts])
    y = torch.stack([DATA[s + 1:s + block_size + 1] for s in starts])  # next-char targets
    return x, y


def train(model, steps=800, batch_size=16, lr=3e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        x, y = get_batch(batch_size, BLOCK_SIZE)
        logits = model(x)                                          # forward
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))  # loss
        loss.backward()                                             # backward
        optimizer.step()                                            # update
        optimizer.zero_grad()                                       # reset
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step:4d}   loss {loss.item():.3f}")


if __name__ == "__main__":
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_BLOCKS, FFN_HIDDEN, BLOCK_SIZE)
    print(f"vocab size {VOCAB_SIZE}, {sum(p.numel() for p in model.parameters()):,} parameters")

    train(model)

    prompt = "the animal"
    idx = encode(prompt).unsqueeze(0)
    for temp, k, p in [(0.3, None, None), (1.0, None, None), (1.0, 3, None), (1.0, None, 0.8)]:
        out = model.generate(idx.clone(), max_new_tokens=40, temperature=temp, top_k=k, top_p=p)
        label = f"T={temp}" + (f", top_k={k}" if k else "") + (f", top_p={p}" if p else "")
        print(f"{label:24s} -> {decode(out[0])!r}")

    torch.save(model.state_dict(), "05_tiny_gpt_weights.pth")
    print("Saved weights to 05_tiny_gpt_weights.pth")
