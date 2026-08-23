# Neural networks and transformers, built from scratch

A free, self-contained interactive course. No calculus, linear algebra, or ML background assumed — just arithmetic.

**Live site:** https://nilesh-salpe.github.io/genai/

## What's here

- **`concept/`** — the course itself, deployed to GitHub Pages on every push to `main` (`.github/workflows/pages.yml`), no build step. Every page is a single self-contained HTML file — inline styles, inline scripts, no dependencies beyond Google Fonts.
  - `index.html` — landing page for the current course (nine parts, Part 0 through Part 8)
  - `part-0-big-picture.html` … `part-8-references.html` — the nine numbered parts, in order
  - `code/`, `examples/` — the Python/notebook implementations several parts link out to
  - `drafts/` — markdown drafts for course content, written before conversion to HTML
  - `bonus-curve-fitting.html`, `bonus-transformer-illustrated.html` — standalone interactive sandboxes, linked from Part 7
- **`notebooks/`** — unrelated reference notebooks (annotated Transformer, attention visualization), not part of the course flow

## Running locally

```
python3 -m http.server -d concept
```
then open http://localhost:8000. Or just open any `concept/*.html` file directly in a browser.
