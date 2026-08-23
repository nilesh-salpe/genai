# v2 course — content drafts

Markdown drafts for the course-content expansion, written before the HTML conversion pass. Each file carries enough depth that converting it to an interactive HTML page (following the conventions in the repo's `CLAUDE.md` — inline `<style>`/`<script>`, canvas/SVG widgets, `§`-numbered cross-references) is mostly a formatting and widget-building exercise, not a re-derivation of content.

**Written for a genuine beginner throughout** — every mechanism gets a plain-language mental model before its formula (attention as "a lookup that matches by content," backprop as "blame," positional encoding as "clock hands," a neuron as "a tiny scoring machine"), and no prerequisite beyond Part 1's from-scratch arithmetic is assumed.

## Final part order

| # | Part | File | Status |
|---|---|---|---|
| 0 | Big Picture | *(existing v2 page, untouched)* | Live |
| 1 | Foundations | *(existing v2 page, untouched)* | Live |
| 2 | **PyTorch** | [`part-2-pytorch.md`](part-2-pytorch.md) | **New draft.** Moved earlier than v1's position — comes before Neural Network, so it's reframed throughout to *forward*-reference Part 3 ("here's the tool; Part 3 shows you what it's automating") instead of assuming a from-scratch build already happened |
| 3 | Neural Network | [`part-3-neural-network.md`](part-3-neural-network.md) | Full draft (renumbered from an earlier "Part 2" draft) |
| 4 | Build Neural Networks | [`part-4-build-neural-networks.md`](part-4-build-neural-networks.md) | Full draft (renumbered from an earlier "Part 3" draft; renamed from "Examples") |
| 5 | **Transformer Architecture** | [`part-5-transformer-architecture.md`](part-5-transformer-architecture.md) | Full draft (renumbered from an earlier "Part 4" draft; renamed from "Attention"; **merged** — now absorbs what would have been a separate "block assembly / encoder-decoder / cross-attention / sampling" part, per plan) |
| 6 | LLM / Build a GPT | [`part-6-llm.md`](part-6-llm.md) | Full draft. Number unchanged. Replaces the live page's simplified "lookup table" training stand-in's role as the site's only full-pipeline visual with a new flagship animation (§01): a real, small, multi-block Transformer visualized end to end for one generation step — the explicit "how the transformer works" illustration requested |
| 7 | Illustrated | *(existing v2 page, untouched)* | Live |
| 8 | **References** | [`part-8-references.md`](part-8-references.md) | **New draft**, ported from v1 `references.html` with all cross-references updated to the numbering above |

Renumbering worked out cleanly: removing the standalone Transformer part (merged into 5) exactly offsets inserting PyTorch, so LLM and Illustrated keep the part numbers they already have live today.

## A gap caught during renumbering

The live `part-2-neural-network.html` has a §06 "Vanishing gradients, and the fix" section that the first pass at `part-3-neural-network.md` dropped. It's been restored as **§10** (right after the backward pass, before the training loop) — which is where several other drafts (Part 5's residual-connection section, Part 8's ResNet/LayerNorm references) already assumed it would be, so restoring it turned out to require no changes to those files. Everything from the old §10 onward in Part 3 shifted down by one (training loop is now §11, initialization §12, overfitting §13, etc.) — every cross-reference to those sections, in every draft file, has been updated to match.

## What changed content-wise, not just numbers

- **Part 2 (PyTorch)** — every "look back at Part 2's 60 lines" reference from v1 was rewritten as a forward pointer, and the autograd worked example (which in v1 literally reused numbers from the from-scratch build) was replaced with a self-contained equivalent, since that build hasn't happened yet at this point in the course.
- **Part 3 (Neural Network)** — curve-fitting framing opens with a hand-worked `y = ax + b` gradient-descent example (real numbers, two full steps) before any neuron vocabulary, then neuron → layer → network, softmax/cross-entropy with worked numeric examples, an explicit forward-pass/backward-pass split with diagrams, the training loop, initialization, over/underfitting, a "which architecture when" table, the full from-scratch NumPy implementation with a line-by-line table, a bugs/debugging section, and a glossary.
- **Part 4 (Build Neural Networks)** — all four v1 examples (spiral / conv filter / attention retrieval / causal prediction) restructured into: problem → outcome → example → why a NN is needed → solution → notebook link.
- **Part 5 (Transformer Architecture)** — opens with the paper (problem, core idea, the RNN comparison, the BLEU results) and "why attention" motivated *before* the mechanics, then the five-step derivation, causal mask, multi-head, block assembly (residuals/LayerNorm), the encoder/decoder split and cross-attention, sampling (temperature/top-k/top-p), cost, a traps/cheatsheet, exercises, and a glossary — everything that would previously have been split across two separate parts.
- **Part 8 (References)** — near-verbatim port; a curated list doesn't need the depth pass the teaching parts got.

## Diagrams

Every mermaid code block is a stand-in for an interactive canvas/SVG widget in the eventual HTML page, tagged with `> **Widget claim to check:** ...` — the one thing a reader should be able to verify by interacting with it, per the course's "every animation makes a checkable claim" convention. Where a v1 widget already exists and does the job, the tag says so explicitly ("already built as `id` — port/keep") rather than proposing a new one.

## Not yet started

- Actually converting these five drafts into styled, interactive HTML pages
- Renaming/renumbering the *live* v2 HTML files to match (`part-2-neural-network.html` → `part-3-neural-network.html`, etc.) and updating the `<nav class="progress">` block across every page — the drafts above assume this happens, but no live file has been touched yet
- Updating `v2/index.html`'s cards to match the new part list, titles, and numbers
