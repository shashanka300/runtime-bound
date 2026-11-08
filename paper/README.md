# Paper

- `main.tex` + `refs.bib` + `figures/`: the paper (Kaggle Paper Track, 3,000-word limit).
- `main.pdf`: compiled with [Tectonic](https://tectonic-typesetting.github.io) 0.17:
  `cd paper && tectonic -X compile main.tex`.
- `make_figures.py`: regenerates `figures/` (run from the repository root:
  `python paper/make_figures.py`); figure 2 reads the lb04 screen traces under `data/`.
- `writeup.md`: the same text in Markdown, for the Kaggle Writeup body.
