# Known Limitations

## No head/layer selection guidance

IzzyViz renders **one `(layer, head)` attention matrix per call**. For `bert-base-uncased` that is 12 × 12 = 144 possible matrices. The library offers no mechanism to rank or recommend which pair to visualize.

The README's examples (`layer=-1, head=8`; `layer=6, head=5`; `layer=7/8, head=9`; `layer=6, head=9`; `layer=11, head=9`) are hand-picked by the author, not the output of any ranking procedure.

The auto-detection feature (`find_attention_regions_with_merging`, `auto_detect_regions=True`) operates *within* a single chosen matrix to box high-attention regions. It does **not** rank heads across the M × N grid.

### Assumed workflow

1. **Triage elsewhere** — use a head-browser such as BertViz `head_view` / `model_view`, or a 12 × 12 `plt.imshow` thumbnail grid, to identify heads worth keeping.
2. **Render in IzzyViz** — once the 1–3 heads of interest are known, use IzzyViz to produce the publication-quality PDF with top-cell annotations and region boxes.

### External heuristics for an automated shortlist

Not built in, but reasonable starting points:

- **Attention entropy per head** — low entropy means the head is focused on a few tokens.
- **Off-diagonal mass** — high values suggest non-trivial structure beyond identity / adjacency patterns.
- **Cross-run disagreement** — heads that change a lot across fine-tuning runs are interesting candidates for the stability heatmap.

Compute these outside the library, then feed the winners into IzzyViz.

### Possible future extension

A `rank_heads(attentions, metric=...)` utility that returns a sorted list of `(layer, head)` pairs by the chosen heuristic would close this gap and fit naturally alongside the existing region-detection code.
