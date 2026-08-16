# Neural Networks and Transformers — an interactive course

Two self-contained HTML pages. No build step, no dependencies, no framework. Open either file in a browser.

| File | What it covers | Widgets |
|---|---|---|
| [`index.html`](index.html) | Landing page | — |
| [`part-1-maths.html`](part-1-maths.html) | The maths, derived from scratch | 13 |
| [`part-2-build.html`](part-2-build.html) | Building a working digit classifier | 10 |

## Part one — the maths

Every formula is derived from the problem it solves rather than stated and explained afterwards. Covers vectors and the dot product, matrices and the shape rule, why depth needs a nonlinearity, softmax (derived after two deliberate failed attempts), cross-entropy (derived from what "surprise" must satisfy), derivatives and the chain rule, backpropagation and where the transpose comes from, vanishing gradients, and then transformers: embeddings, positional encoding, attention built in five steps, multi-head, and the O(n²d) cost.

Assumes school algebra and a first calculus course. Linear algebra and the two required probability facts are built from scratch inside.

## Part two — actually build one

`784 → 128 → 64 → 10`, with every shape tabulated, all 109,386 parameters counted layer by layer, and a complete NumPy implementation with a hand-written backward pass. Also covers why zero initialization fails, what flattening an image actually costs, overfitting and the three data splits, and a debugging table keyed by symptom.

## The interactive figures

Each exists to make an argument checkable, not to decorate:

- **Softmax shift-invariance** — add 100 to every logit and watch nothing move
- **The surprise function** — three candidate losses tested against `S(pq) = S(p) + S(q)`; only `−log p` survives
- **√d_k** — turn the scaling off and watch attention's gradient collapse
- **The drunkard's walk** — 1,500 random walks showing spread grows like √n, the one probability fact used twice
- **Causal masking** — the upper triangle goes to zero while every row still sums to 1
- **Zero initialization** — a real network that never moves, because every gradient is exactly zero
- **The permutation test** — shuffle an image's pixels and see why a dense layer does not notice

## Notes

- Reduced-motion preferences are respected; every animation degrades to its final state.
- The only external request is a Google Fonts stylesheet. Offline, the pages fall back to system fonts and everything still works.
- Serve locally with `python3 -m http.server` if you prefer a real origin to `file://`.
