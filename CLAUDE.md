# genai

This repo hosts **"Neural networks and transformers, built from scratch"** — a free, self-contained interactive course at `concept/`, deployed as a static site to GitHub Pages (`.github/workflows/pages.yml` publishes the `concept/` folder on every push to `main`, no build step).

## Structure

- `concept/index.html` — landing page for the current course, one card per part.
- `concept/part-0-big-picture.html` … `part-8-references.html` — the nine numbered parts, in order (0 Big Picture, 1 Foundations, 2 PyTorch, 3 Neural Network, 4 Build Neural Networks, 5 Transformer Architecture, 6 LLM, 7 Illustrated, 8 References).
- `concept/code/`, `concept/examples/` — the Python/notebook implementations several parts link out to.
- `concept/drafts/` — markdown drafts of course content, written and reviewed before conversion to HTML. When adding a new part or a substantial rewrite, draft it here first.
- `concept/v1/` — the original seven-part single-track version of the course, archived but still deployed live at `/v1/` (not deleted — existing links to it must keep working). Cross-linked with the current course from both landing pages ("See v1" / "try v2"). Treat `v1/` as frozen: fix only broken links or build-breaking issues there, don't extend its content — new work goes into the numbered parts at `concept/` root.
- `notebooks/` — unrelated reference notebooks (annotated Transformer, attention visualization), not part of the course flow.

Every course page is a single self-contained HTML file: inline `<style>`, inline `<script>`, no dependencies beyond Google Fonts. Open any file directly in a browser or serve `concept/` with `python3 -m http.server`. New pages should follow the same pattern — copy the `<style>` block and canvas/SVG widget conventions from an existing part rather than introducing a new design system or a build step.

## Course-authoring conventions

These are load-bearing for anyone (human or Claude) adding or editing course content:

**Write only for the student.** Every sentence on a course page is either teaching content or student-facing navigation ("sections 1–3 cover X, 4–6 cover Y"; "recall from Part 1 §11"). Never write sentences that talk *about* the course as an authored artifact — no explaining why a page was structured a certain way, no referencing the course's own marketing/tagline language, no notes that read like instructions left for whoever generates or edits the course next. If a sentence would only make sense to someone designing the course rather than someone taking it, cut it or rewrite it as direct instruction to the student.

- Bad (authoring narration): *"Every later part in this course says 'derived from scratch,' but scratch has to start somewhere... This page builds those four... so that when Part 1 uses them, you are watching a tool you already own do a new job — not meeting the tool for the first time."*
- Good (direct teaching): *"Four tools carry the rest of this course: rearranging a formula, what a derivative is, the chain rule, and two facts about randomness. This page builds each one from nothing but arithmetic, with no neural network in sight."*

**Progressive disclosure, checked against a true beginner.** The stated audience is anyone comfortable with arithmetic — no calculus, linear algebra, or ML background assumed. Before introducing a symbol or operation (Σ, a derivative, `E[·]`, a matrix), either build it from scratch or point to the earlier section/part that did. Part 0 exists specifically to hold the pre-ML math (functions, derivatives, chain rule, expected value) that Part 1 needs but doesn't build itself — new foundational material belongs there, not bolted onto a later part as an aside.

**Every animation must make a specific claim checkable.** The course's own stated philosophy (see `index.html`'s "What the widgets are for") is that a widget exists to let the reader verify an argument by changing an input and watching a stated consequence happen — not to decorate the page. Before adding an interactive figure, name the one claim it lets the reader check. Before keeping one during a review pass, ask whether removing it would lose an argument or just a nice picture; recap/summary animations that restate already-taught material without adding a new checkable claim should be static instead (see the "checkpoint" boxes in Part 1 for the pattern: keep the content, drop the click-through mechanic).

**Cross-reference by section number, not vague gesture.** "Part 1 §11" beats "the maths guide." Section numbers (`<span class="num">`) are stable within a part — if you renumber a part's `§` sections, grep the whole `concept/` tree for `"Part N §"` references to that part and fix them, since parts 3, 5, and 6 in particular lean heavily on precise callbacks to Part 1 and Part 2.

**Nav and index stay in sync.** The `<nav class="progress">` block is duplicated verbatim (modulo the `.here` class and dark-vs-light styling on the bonus sandboxes) across every course page. Adding, removing, or renaming a page means updating that block everywhere, plus the matching card in `index.html` (title, description, figure/step count in `.meta`).

## Verifying changes

There's no test suite — verification is running the page. After editing a page's inline `<script>`:
1. Syntax-check it: extract the largest `<script>...</script>` block and run `node --check` on it.
2. Serve `concept/` locally (`python3 -m http.server`) and load the page in a browser; check the console for errors and exercise any changed widget (drag a slider, click Run/Step/Play) to confirm it still computes and renders.
3. Grep the whole tree for stale references (old figure/step counts in `index.html`, dangling `id`s, a nav link that doesn't resolve to a file) — the individual pages have no automated check for this.
