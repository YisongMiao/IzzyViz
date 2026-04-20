# N_abs

Ideation notes for a head-clustering extension to IzzyViz: collapse the M × N attention grid into a small set of archetypes plus a list of outliers, so the researcher renders ~10 heatmaps instead of 144.

Contents:

- `head_clustering_design.md`: full design document with worked numeric example on BERT-base, feature definitions, clustering method comparison, pipeline, and validation plan.
- `walkthrough/`: a concrete medium-complexity pseudo run showing what the pipeline outputs end to end.
  - `generate_figures.py`: reproducible script (seed 42) that builds 144 synthetic attention matrices from archetype templates, extracts 6 features, runs Ward clustering, and produces five PDFs.
  - `fig1_head_grid.pdf` through `fig5_outliers.pdf`: the generated figures.
  - `WALKTHROUGH.md`: plain-language step-by-step narration of the pipeline using those figures.

Status: design + illustrative run only, no integration with IzzyViz yet. Open questions are listed at the bottom of the design doc.
