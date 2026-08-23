# Neural networks and transformers, built from scratch

A free, self-contained interactive course. No calculus, linear algebra, or ML background assumed — just arithmetic.

**Live site:** https://nilesh-salpe.github.io/genai/

## What's here

- **`concept/`** — the course itself, deployed to GitHub Pages on every push to `main` (`.github/workflows/pages.yml`), no build step. Every page is a single self-contained HTML file — inline styles, inline scripts, no dependencies beyond Google Fonts.
  - `index.html` — landing page
  - `part-0-foundations.html` … `part-6-build-gpt.html` — the seven numbered parts, in order
  - `bonus-curve-fitting.html`, `bonus-transformer-lab.html`, `bonus-transformer-illustrated.html` — standalone interactive sandboxes
  - `references.html` — further reading
  - `code/`, `examples/` — the Python/notebook implementations Part 3 and Part 6 link out to
- **`notebooks/`** — unrelated reference notebooks (annotated Transformer, attention visualization), not part of the course flow

## Running locally

```
python3 -m http.server -d concept
```
then open http://localhost:8000. Or just open any `concept/*.html` file directly in a browser.
