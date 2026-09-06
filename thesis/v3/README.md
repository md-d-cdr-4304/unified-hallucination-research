# thesis/v3 — the scripts that generate every figure and table

Nothing in the thesis is typed by hand. Every number in a table or a figure is
read from a result file in `harness/results/` or from the numbers manifest
`thesis/v3/data/manifest_v2.json`, which is itself derived from those result
files.

| Script | Reads | Writes |
|---|---|---|
| `build_core_tables.py <summary.json>` | `harness/results/core_prompt/summary.json` (from `analyze.py`) | `chapters/_core_tables.html` (Tables 5.1–5.3), `figures/f_core_deltas.png`, `figures/f_core_rates.png` |
| `make_figures.py` | `data/manifest_v2.json`, `harness/results/e6_sim/propagation.json` | every remaining `figures/*.png` |
| `build.py [--out THESIS.pdf] [--no-pdf]` | `chapters/*.html`, `bib/lit_*.json`, `figures/` | the HTML/PDF thesis, `build_report.json` |
| `to_latex.py` | `chapters/*.html`, `bib/lit_*.json`, `figures/` | `latex/thesis.tex`, `latex/refs.bib`, `latex/figures/` |

Build order: results → `build_core_tables.py` → `make_figures.py` → `build.py`
or `to_latex.py`.

## Folders

- `chapters/` — the chapter sources as HTML fragments. `00_front.html` carries
  the thesis data (degree, date, faculty, title, authors, supervisor) used by
  both builders. `_core_tables.html` is generated, not written by hand.
- `bib/` — the verified bibliography, `lit_*.json`, merged in file-name order
  (first file wins on a duplicate key). Citations in the chapters are `[@key]`
  and are numbered IEEE-style in order of first use.
- `data/manifest_v2.json` — the numbers manifest (see `data/README.md`).
- `figures/` — generated PNGs. Build product: not tracked in git, rebuilt by
  the two figure scripts.
- `latex/`, `html_build/` — build products, not tracked.

`build.py` writes `build_report.json` listing unknown citation keys, missing
figures and any remaining `DATA-NEEDED` / `CITE-NEEDED` markers. That report
must be clean before submission.

The LaTeX route follows the BTH degree-project template (`DP_thesis_tmpl/`:
`bth.cls`, `changepage.sty`, `bthnotext.pdf`, bibliography style `IEEEtranS`).
The template is not redistributed here; place it next to `latex/` and build
with `pdflatex → bibtex → pdflatex → pdflatex`.
