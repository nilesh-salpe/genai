# Part 8 — References (draft)

> **Status:** markdown draft, pre-HTML. New page — v2 currently has no References page or nav slot. Source lineage: v1 `references.html`, ported near-verbatim (this is a curated list, not derived teaching material, so it doesn't need the depth pass the other parts got) with every cross-reference updated to v2's numbering. Placed last, after Part 7.

---

Papers and explainers this course draws on, for going past what any one part covers. Nothing here is required — everything in the eight parts stands on its own — but each of these is where a specific idea in the course originally came from, or is explained a second way.

## Papers

**[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)**
Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin · 2017 · arXiv:1706.03762
The paper that introduced the Transformer. **Part 5 §03** derives the same attention mechanism from first principles; **Part 5 §07** covers the full encoder-decoder architecture this paper actually proposed, which Part 5's core derivation deliberately simplifies to the decoder-only half every current LLM uses.

**[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)**
He, Zhang, Ren, Sun · 2015 · arXiv:1512.03385
Introduced residual ("skip") connections, the `x + F(x)` pattern **Part 3 §10** uses to explain why gradients survive deep stacks, and that every block in **Part 5** and **Part 6** is built from.

**[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215)**
Sutskever, Vinyals, Le · 2014 · arXiv:1409.3215
The encoder-decoder architecture attention was originally invented to fix. Useful background for **Part 5 §07**'s framing of why translation needs two stacks talking to each other, not one.

**[Layer Normalization](https://arxiv.org/pdf/1607.06450)**
Ba, Kiros, Hinton · 2016 · arXiv:1607.06450
The normalization scheme **Part 3 §10** contrasts against BatchNorm, and that every block in **Part 5** and **Part 6** uses to keep activations stable across layers.

**[Understanding Transformers and Attention Mechanisms: An Introduction for Applied Mathematicians](https://arxiv.org/pdf/2604.00965)**
Serret · 2026 · arXiv:2604.00965
A second from-scratch, maths-first treatment of attention and Transformers, aimed at a similar reader to this course's **Part 0** and **Part 1** — worth reading alongside them for a differently-organized derivation of the same mechanism.

## Explainers and implementations

**[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
Jay Alammar
The best-known visual walkthrough of the Transformer, and a good companion to this course's own **Part 7 — Illustrated** step-by-step sandbox — the same mechanism, drawn by two different people.

**[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)**
Harvard NLP
"Attention Is All You Need," reproduced paragraph by paragraph next to a full working PyTorch implementation. The natural next step after **Part 6**'s tiny GPT, scaled up to the paper's actual architecture.

**[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)**
PyTorch official tutorials
PyTorch's own beginner path through tensors, datasets, autograd, and training loops. A good next stop after **Part 2**, which covers the same ground but only as much as this course needs.

**[CS230: Deep Learning](https://cs230.stanford.edu/lecture/)**
Stanford · Andrew Ng, Kian Katanforoosh
A full university course, for going deeper into territory this course only touches — CNN architectures beyond **Part 4**'s single filter, RNNs, and the broader deep learning toolkit outside Transformers.

---

Suggestions for additions are welcome — this list favors sources that connect directly back to something a specific part of this course builds, not a general reading list.
