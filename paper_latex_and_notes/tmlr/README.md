# TMLR draft (10-18-2026)

Draft only. Not a submission. The PDF uses the official TMLR style in anonymous mode, so the author block in `main.tex` does not appear on the page.

Style files `tmlr.sty`, `tmlr.bst`, and `fancyhdr.sty` are vendored from <https://github.com/JmlrOrg/tmlr-style-file> at commit `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992` (Apache-2.0; see `TMLR_STYLE_LICENSE.txt`).

Numbers come from the archived results files under `experiments/archive/`. A `% Provenance:` comment marks the source next to each block of results. Intervals are the studies' eight-seed block bootstraps. Where a file says p-val=n/a, the draft keeps that.

Compile from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=_build main.tex
cp _build/main.pdf 00_main_tmlr.pdf
```

`_build/` is gitignored. `00_main_tmlr.pdf` is the tracked PDF.
