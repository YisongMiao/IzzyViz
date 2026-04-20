# Head-Clustering Walkthrough (Pseudo Run)

This document walks through one medium-complexity run of the head-clustering pipeline, end to end, using a synthetic dataset designed to look like BERT-base on a single sentence. All five figures (`fig1_head_grid.pdf` through `fig5_outliers.pdf`) in this directory are the outputs of `generate_figures.py`.

The goal: turn 144 raw attention matrices into a small overview plot plus a handful of targeted PDFs, so the researcher can focus on the heads that actually carry signal.

Plain-English numbers throughout. No real transformer is run here; matrices are drawn from archetype templates plus noise, so the ground truth is known and we can check the clustering honestly.

## A shared color theme across Figures 1, 2, and 3

All three overview figures use the **same six colors for the same six clusters**:

| Color  | Cluster |
| ------ | ------- |
| Blue   | Diagonal |
| Orange | SEP-parking |
| Green  | CLS-broadcaster |
| Red    | Previous-token |
| Purple | Local window |
| Brown  | Broad mixer |
| Gray   | Outlier |

So when you look at an orange border in Figure 1, an orange branch in Figure 2, and an orange dot in Figure 3, you are looking at the same set of heads. That is how the three overviews tie together.

## Setup at a glance

- **Input sentence (pretend)**: `The quick brown fox jumps over the lazy dog.`
- **Tokens (T = 12)**: `[CLS] the quick brown fox jumps over the lazy dog . [SEP]`
- **Model (pretend)**: BERT-base, 12 layers × 12 heads = **144 attention matrices**, each of shape `[12, 12]`.
- **Feature set (6 numbers per head)**: `diag_mass`, `prev_mass`, `sep_mass`, `cls_mass`, `entropy`, `local_k3`.
- **Clustering method**: **Ward hierarchical** (medium complexity: more principled than k-means, simpler than HDBSCAN).
- **Outlier rule**: head is flagged if its distance to its own cluster medoid sits in the top 8 % of all within-cluster distances.

## Step 1: Look at what we're up against (Figure 1)

![Figure 1: head grid](preview_fig1_head_grid.png)

Figure 1 lays out all 144 heads as a 12 × 12 wall of tiny heatmaps. Rows are layers `L0` to `L11`, columns are heads `H0` to `H11`. The head at row *l* and column *h* is referenced as **`L{l}H{h}`** throughout this document, so e.g. `L0H2` means "row 0, column 2, the third thumbnail in the top row".

Reading the visual cues:

- **Thumbnail border color** = cluster assignment (same palette as Figures 2 and 3).
- **Thick black frame with colored outline** = cluster medoid, the one head picked as the archetypal example of its cluster.
- **Gray border** = outlier, a head that does not fit its nearest cluster cleanly.

Six thick-framed thumbnails are the medoids. You will see these six heads again in Figure 4:

| Archetype | Medoid location in Figure 1 |
| --------- | --------------------------- |
| Diagonal  | row `L0`, column `H2`  (top-left area) |
| SEP-parking | row `L6`, column `H8`  (center-right) |
| CLS-broadcaster | row `L7`, column `H8`  (below the SEP medoid) |
| Previous-token | row `L1`, column `H5`  (upper-middle) |
| Local window | row `L10`, column `H0`  (bottom-left) |
| Broad mixer | row `L9`, column `H7`  (lower-right) |

Even before reading any statistics, the colored borders make the archetype neighborhoods visible at a glance: most blue-bordered thumbnails clearly show a dark diagonal; most orange-bordered thumbnails clearly show a dark rightmost column; gray-bordered thumbnails look speckled and patternless.

## Step 2: Turn each matrix into six numbers

Every `12 × 12` matrix gets compressed to a 6-dimensional feature vector:

| Feature | Meaning in plain English |
| ------- | ------------------------ |
| `diag_mass` | Fraction of attention a token places on itself |
| `prev_mass` | Fraction placed on the immediately preceding token |
| `sep_mass` | Fraction placed on `[SEP]` |
| `cls_mass` | Fraction placed on `[CLS]` |
| `entropy`  | Row-averaged entropy, normalized to `[0, 1]`: 0 = sharply peaked, 1 = uniform |
| `local_k3` | Fraction falling in a window of ±3 tokens around the current position (excluding self) |

After this step we have a table of shape `144 × 6`. Each row is one head.

To make the numbers concrete, here are the actual raw feature values for each archetype's **medoid** (pulled straight from the synthetic run):

| Archetype | Medoid | `diag_mass` | `prev_mass` | `sep_mass` | `cls_mass` | `entropy` | `local_k3` |
| --------- | ------ | -----------:| -----------:| ----------:| ----------:| ---------:| ----------:|
| Diagonal | `L0H2` | **0.80** | 0.02 | 0.08 | 0.09 | 0.38 | 0.10 |
| SEP-parking | `L6H8` | 0.08 | 0.02 | **0.78** | 0.02 | 0.40 | 0.29 |
| CLS-broadcaster | `L7H8` | 0.08 | 0.09 | 0.02 | **0.77** | 0.42 | 0.29 |
| Previous-token | `L1H5` | 0.09 | **0.77** | 0.02 | 0.15 | 0.41 | 0.79 |
| Local window | `L10H0` | 0.34 | 0.17 | 0.07 | 0.07 | 0.73 | **0.62** |
| Broad mixer | `L9H7` | 0.09 | 0.08 | 0.08 | 0.08 | **0.94** | 0.40 |

Bolded cells show which feature dominates each archetype. The dominance is not subtle: the winning feature sits around `0.7`–`0.9` while the others stay near `0.1`. That separation is why clustering works.

We then **z-score** the columns using the median and median absolute deviation (the robust version, so one weird head can't skew the scale). After z-scoring, a value of `+2` on `sep_mass` means "this head sits 2 robust-deviations above the typical head in how much it parks on `[SEP]`".

## Step 3: Build a Ward dendrogram (Figure 2)

![Figure 2: dendrogram](preview_fig2_dendrogram.png)

Ward's algorithm is a bottom-up tree builder. It starts with 144 singletons and, at each step, merges the two groups whose combination produces the smallest increase in within-group variance.

Figure 2 shows the resulting tree. Every leaf at the bottom is one head; every horizontal bar is a merge. The higher the bar, the more different the two groups being merged are.

**The branch colors match Figure 1's thumbnail borders and Figure 3's scatter dots.** So the blue subtree at the far left is the same set of heads that carry blue borders in Figure 1 and blue dots in Figure 3, namely the Diagonal archetype. The orange, green, red, purple, and brown subtrees follow the same rule. This shared color scheme is what lets you trace a cluster visually across all three overview figures.

The red dashed line at `d ≈ 123.5` is our **cut**. Every vertical line the red line crosses becomes one cluster. Six vertical lines cross here, so we get **K = 6 clusters**. We picked 6 because that is where the tree has a clean gap: cutting lower gives many small clusters, cutting higher collapses meaningful structure.

Notice the subtree shape: the five colored blocks on the left and center are the tight archetypal clusters; the rightmost brown block is a larger, looser group that will turn out to be the Broad-mixer bucket where many of the 12 outliers live.

## Step 4: Project to 2D and look at the map (Figure 3)

![Figure 3: head space](preview_fig3_head_space.png)

Six dimensions is hard to look at, so we project to 2D. The projection here is **PCA** (principal component analysis), not t-SNE or UMAP. PCA is linear and deterministic: it rotates the feature space so that the first two axes capture as much variance as possible, then projects every point onto them. The distances in the scatter are a faithful linear shadow of the original 6D distances, so close-in-2D means close-in-feature-space. (t-SNE and UMAP would give prettier separation but non-linear, seed-dependent, and distances would not carry the same meaning.)

Here **PC1 captures 60.8 %** and **PC2 captures 28.9 %**: together, about **90 %** of the structure is preserved in 2D. That is why the scatter looks decisive rather than blurry.

**All 144 heads are on this plot**; there is one dot per head. Some regions look less populated than their Figure-1 counts would suggest because the Diagonal cluster has 42 near-identical heads stacked almost on top of each other at `(−58, 11)`. The scatter uses semi-transparent dots so the pile-up is visible as a denser color blob.

Reading the scatter:

- **Colored dots**: cluster members, sharing the Figure 1 / Figure 2 palette.
- **Black-edged stars**: each cluster's **medoid**. Labels show the layer/head, so the star at `(45, 45)` labeled `L6H8` is the head you saw in the upper-right region of Figure 1.
- **Red crosses**: the 12 outliers. Most of them cluster tightly near `L9H7`, indicating the Broad-mixer region is where the pipeline's uncertainty lives.

Tracing back to Figure 1 by color:

- Top-right orange star (`L6H8`) — find the orange-bordered thumbnail at row `L6`, column `H8` in Figure 1. Both are the same head.
- Left-side blue pile (`L0H2`) — that is the Diagonal neighborhood: every thumbnail with a blue border in Figure 1 sits inside that pile.
- Bottom green star (`L7H8`) — CLS-broadcaster medoid, one row below the SEP medoid in Figure 1.

This is the **"map"** of the head space. From 144 heads the reader now has one legible overview.

## Step 5: Find the archetypes (Figure 4)

![Figure 4: archetype gallery](preview_fig4_archetypes.png)

For each of the six clusters we pick the **medoid**: the one real head whose feature vector sits closest (by Euclidean distance) to the cluster's center. The medoid is the archetypal example of the cluster.

The top row of Figure 4 shows the six medoid heatmaps at full resolution. The bottom row shows each cluster's **mean feature signature**, a bar chart that tells you why the cluster earned its name. Bar colors match the cluster colors in Figures 1, 2, and 3.

Reading across the columns, and cross-referencing back to Figure 1's grid coordinates:

| Cluster | Medoid | Figure 1 position | Size | What its bars say |
| ------- | ------ | ----------------- | ---- | ----------------- |
| Diagonal | `L0H2` | row 0, col 2  (upper-left area) | 42 | `diag_mass ≈ 0.80`; everything else small |
| SEP-parking | `L6H8` | row 6, col 8  (center-right) | 28 | `sep_mass ≈ 0.78` dominates |
| CLS-broadcaster | `L7H8` | row 7, col 8  (below SEP medoid) | 12 | `cls_mass ≈ 0.77` tallest bar |
| Previous-token | `L1H5` | row 1, col 5  (upper-middle) | 16 | `prev_mass ≈ 0.77`; `local_k3` elevated |
| Local window | `L10H0` | row 10, col 0  (bottom-left) | 10 | `diag_mass` ≈ 0.34 and `local_k3` ≈ 0.62 (spread, not peaked) |
| Broad mixer | `L9H7` | row 9, col 7  (lower-right) | 36 | `entropy ≈ 0.94` (rows nearly uniform) |

You can physically locate each archetype in Figure 1 by jumping to its grid coordinate. The thick-framed thumbnail at `L0H2` is the same matrix shown enlarged under "Diagonal" in Figure 4; the thick-framed thumbnail at `L6H8` is the matrix under "SEP-parking"; and so on. This makes the compression visible: six grid positions represent 134 heads (144 minus 12 outliers minus 6 medoids already counted once), because each of the non-medoid cluster members behaves like its medoid in feature space.

The cluster labels were **assigned automatically** by a simple rule: for every (cluster, feature) pair, sort by z-score and greedily claim the strongest unique pairings. No hand-labeling.

The heatmap row (top) is the shortlist that goes into IzzyViz for publication-quality rendering. Six figures instead of 144.

## Step 6: Find the outliers (Figure 5)

![Figure 5: outliers](preview_fig5_outliers.png)

Twelve heads were flagged as outliers, meaning their feature vectors are far from the medoid of their assigned cluster. Figure 5 shows the six outliers with the largest distances.

These heads are interesting for a reason opposite to the archetypes: they are the heads that **refuse to compress**. If an archetype is "the typical behavior", an outlier is "the thing that does not fit the typical behavior".

In real BERT, outliers in this category tend to be:

- **Syntactic heads** (e.g. `(7, 6)` as a coreference head, `(8, 10)` as a direct-object head per Clark et al.).
- **Punctuation-attending heads** that none of our six archetypes captured.
- **Heads that combine two behaviors** (e.g. half `[SEP]`-parking, half diagonal).

In our synthetic run, the 12 flagged outliers are random-template heads that happen to live inside or near the Broad-mixer bucket but sit far from its medoid. The heatmaps look speckled because that is what `tpl_random` produces. In Figure 1, these correspond to the gray-bordered thumbnails.

These 12 heads also go into IzzyViz for individual rendering, so the researcher sees every head that does not fit the archetype taxonomy.

## Step 7: Hand off to IzzyViz

After clustering, the pipeline produces:

- **1 overview** (`head_space.pdf`), like Figure 3.
- **6 archetype PDFs** (one per cluster medoid), rendered by existing `visualize_attention_self_attention`.
- **12 outlier PDFs** (one per flagged head), same renderer.
- **1 summary JSON** mapping every `(layer, head)` to its cluster label.

Total: about **19 PDFs** produced from a grid of **144 heads**. The researcher reads 19 figures instead of 144, and the 19 were chosen for a reason they can state (medoid of cluster X, or far outlier).

## Final numbers from this run

```
Retained clusters (K = 6):
  cluster 1  Diagonal          medoid L0H2   size 42   effective 40
  cluster 2  SEP-parking       medoid L6H8   size 28   effective 28
  cluster 3  CLS-broadcaster   medoid L7H8   size 12   effective 12
  cluster 4  Previous-token    medoid L1H5   size 16   effective 16
  cluster 5  Local window      medoid L10H0  size 10   effective 10
  cluster 6  Broad mixer       medoid L9H7   size 36   effective 26

Outliers flagged: 12 / 144
Ward cut distance: 123.50
PC1 / PC2 variance: 60.8% / 28.9%   (about 90% captured in 2D)
```

The clustering recovered the six ground-truth archetypes we synthesized, with sizes matching the plant (42 / 28 / 12 / 16 / 10 / 36) and with all 12 outliers drawn from the genuinely hard-to-classify random-template heads. The pipeline behaves as designed.

## Caveats worth naming

1. **Synthetic data is easy.** Real BERT heads are messier; expect more outliers and blurrier cluster boundaries, especially in early layers where heads carry multiple behaviors at once.
2. **Feature choice shapes the answer.** Adding or removing features changes what the clusters mean. The six chosen here cover the Clark et al. taxonomy adequately; for other models, revisit the list.
3. **Sentence dependence.** This entire pipeline runs on one sentence. Corpus-level clustering (average features over K sentences) is the way to make claims about the model itself rather than the model on one input. That is the next document.
4. **Ward is deterministic, PCA is deterministic, but the dendrogram cut is a judgment call.** K = 6 was chosen here by visual inspection of the gap at `d ≈ 123`. In a real run, report cluster stability across a few choices of K.
5. **PNG vs PDF.** All embedded images in this document are PNG previews rendered at 200 DPI; the `fig*.pdf` files are vector originals and are the preferred format for any downstream use (paper figures, slides, deep zoom).

## How to reproduce

```bash
pip install numpy scipy matplotlib
cd N_abs/walkthrough
python generate_figures.py
```

Seed is fixed at 42, so figures are reproducible bit-for-bit.
