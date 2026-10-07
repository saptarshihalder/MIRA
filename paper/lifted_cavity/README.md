# Lifted Cavity Networks: paper source

NeurIPS 2026 format (`neurips_2026.sty`, anonymous submission mode). `main.pdf` is the compiled paper. Build:

```bash
cd paper/lifted_cavity
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where the numbers come from

Every number, table and figure in `generated/` and `figures/main.pdf` is written by
`experiments/lifted_cavity/make_paper.py` from saved per-task scores. No model runs during paper generation. The
inputs are committed under `artifacts/reports/lifted_cavity_paper/`:

| Folder | Contents |
|---|---|
| `runs/` | Per-task scores, metadata and checkpoints of every synthetic-trained model, plus the endpoint files `confirm_v2.json` and `confirm_v3.json` and `compute_cost.json` |
| `runs_real/<dataset>/` | Fine-tuned models on the real targets |
| `panels/` | Closed-form reference scores, cluster keys and metadata of every evaluation panel, exported from the panel caches by `export_panels.py` |
| `posthoc/`, `explore/`, `init/` | Post-hoc Gaussian process, exploratory fine-tuning variants, untrained models |

To regenerate:

```bash
cd experiments/lifted_cavity
A=../../artifacts/reports/lifted_cavity_paper
python make_paper.py --runs $A/runs --real $A/runs_real --cache /nonexistent --panels $A/panels \
  --posthoc $A/posthoc --explore $A/explore --init $A/init \
  --v1 ../../artifacts/reports/lifted_cavity_v1/confirmation/conf_results.json --out ../../paper/lifted_cavity
```

The panel caches themselves are rebuilt deterministically by `evaluate2.py panel` and `eval_real.py build` with the
seeds in the protocol documents. With `--cache` pointing at them, `make_paper.py` reads them directly and gives
identical output.

## Protocols

- `docs/LIFTED_CAVITY_PROTOCOL_V2.md`: committed before any v2 panel or test split existed.
- `docs/LIFTED_CAVITY_PROTOCOL_V3.md`: committed before any v3 panel or target existed. Exact commands:
  `experiments/lifted_cavity/run_protocol_v3.sh`.

## Hand-written text

`main.tex`, `appendix.tex`, `audit.tex`, `experiments*.tex`, `exp_*.tex`, `limitations.tex`, `conclusion.tex`,
`abstract*.tex` and `intro*.tex`. Their numbers come from macros defined in `generated/numbers.tex`. The method
figure is `figures/method.tex` (TikZ).
