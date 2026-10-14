# MIRA Neurocomputing manuscript

This expanded manuscript uses the Elsevier CAS double column template supplied in the conversation. The class files and bibliography style are unchanged.

Open main.pdf to read the paper. Compile main.tex to rebuild it. The original sample document is retained as evidence/original_template.txt so that it cannot be selected accidentally as the article's main source.

## Build

Use a TeX distribution with the CAS dependencies and run:

~~~text
pdflatex main
bibtex main
pdflatex main
pdflatex main
~~~

The manuscript was compiled with Tectonic 0.17.0, which runs the required bibliography and repeat passes automatically:

~~~text
tectonic main.tex --keep-logs --keep-intermediates
~~~

All content files, class files, bibliography data and included figures are in this package. The paper uses the supplied cas-dc layout with the a4paper and fleqn options, numeric natbib citations, and cas-model2-names bibliography style. No custom geometry or substitute conference style is used. The template's highlights page is included before the numbered article.

## Figures

The figures are vector drawings rather than images of LaTeX diagrams. Each new figure is supplied as PDF, SVG and a PNG with a width of 5184 pixels.

1. figures/architecture_overview shows incomplete calibration, the observed query subset, the support fitted anchor, both learned site routes and the Gaussian readout.
2. figures/cavity_refinement shows the joint cavity, prediction residuals, shared corrections, site reconstruction and likelihood objective.
3. figures/shared_noise_example shows an exact analytical sign reversal example and its joint Gaussian geometry.

The support cells and example query are explicitly illustrative. These figures do not introduce new experimental observations. The exact model checks and analytical coefficients are recorded in figures/figure_provenance.json. tools/draw_figures.py regenerates the three drawings with Python, NumPy and Matplotlib.

The historical training curve is retained from the original paper export.

## Evidence

The paper develops the repository's current lifted Gaussian site study. It preserves the historical v1, v2 and v3 results, including the declared outcomes that did not pass, the real transformer comparisons and the factor orientation replay issue. The five v4 efficacy endpoints are withheld after the recorded checkpoint provenance gate failed. No new fitting, scoring, paid computation or endpoint arithmetic was performed for this writing task.

evidence/claims.md connects claims to the relevant code and result sources. evidence/bibliography_audit.md records reference checks. evidence/manuscript_validation.json checks source links, numerical preservation, template hashes and compilation. tools/verify_manuscript.py performs the checks when run in the original repository workspace with pypdf installed.

The authors are anonymous, as in the source manuscript. The authors should supply their names, affiliations and declarations before submission.
