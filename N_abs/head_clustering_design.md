# Head Clustering Design

Design document for collapsing an M × N transformer attention grid (e.g. 12 × 12 = 144 heads in BERT-base) into a handful of **archetypes** and a list of **outliers**. The goal is to reduce the researcher's visualization burden from 144 figures to roughly 10, while surfacing both the common patterns and the heads that resist compression.

## 1. Why clustering is the right frame

Heads are not independent. Voita et al. (2019) show that about 5/6 of BERT's heads can be pruned with minor performance loss; Clark et al. (2019) document a recurring taxonomy of attention patterns (diagonal, previous-token, `[SEP]`-parking, broadcaster, syntactic). The grid has intrinsic low-dimensional structure. Clustering exploits this structure to:

- produce a **compressed gallery** of archetypal heads (cluster medoids), and
- flag **anomalous heads** (cluster noise) that deserve individual attention.

This is strictly an analysis layer on top of existing IzzyViz rendering, not a replacement for it. The output of clustering feeds directly into the existing `visualize_attention_self_attention` function.

## 2. Three design axes

### 2.1 Scope: single example, batch, or corpus

**Single-example** (one sentence S, forward pass once): all 144 matrices share shape `[T, T]`. Cell-wise diffs are defined. Compute cost is negligible. Findings are specific to S ("head (7, 3) attends to `[SEP]` *on this sentence*").

**Corpus-level** (K sentences, e.g. K=100 from SST-2): each head produces K matrices with variable T. Features must be length-invariant. The payoff is population-level claims ("head (7, 3) is an `[SEP]`-head in general") plus per-head variance as a new signal.

|                | 1 sentence                                | 100 sentences                                  |
| -------------- | ----------------------------------------- | ---------------------------------------------- |
| Forward passes | 1                                         | 100                                            |
| Length handling| Trivial (all share T)                     | Must be T-invariant                            |
| Claim strength | Anecdotal                                 | Statistical                                    |
| New signal     | None                                      | Per-head cross-sentence variance               |

**Recommendation**: start single-example for a clean prototype; promote to corpus-level once the feature pipeline is validated.

### 2.2 Representation: how to turn one [T, T] matrix into a point

Four candidate representations, ranked by interpretability:

| Option | Shape | Interpretability | Length-invariant |
| ------ | ----- | ---------------- | ---------------- |
| A. Raw flatten | R^{T²} | Low | No |
| B. **Structural features** | R^{10} | High | Yes |
| C. Spectral (top-k σ) | R^{k} | Medium | Yes |
| D. Distances to template matrices | R^{d} | High | Yes |

**Recommendation**: option B. The features double as the axes a reader uses to interpret clusters ("cluster 3 is high-diagonal, low-entropy, so: diagonal heads"). A concrete 10-feature set follows in section 3.

### 2.3 Distance

- On structural features (B): **Euclidean on robust z-scored features** (z-score uses median and MAD, not mean and std, to resist outlier contamination). Cosine is a scale-agnostic alternative.
- On raw matrices (A): **mean row-wise Jensen-Shannon divergence**,

  d(A, B) = (1 / T) * Σ_i JS(A[i, :], B[i, :])

  because each row is a softmax distribution. Frobenius norm is defensible as a baseline but treats probability mass like Euclidean mass, which is wrong in principle.

### 2.4 Clustering method

| Method | Pick k? | Outlier concept? | Fit for our setting |
| ------ | ------- | ---------------- | ------------------- |
| k-means | Yes | None | Forces every head into a cluster; hides outliers |
| Hierarchical + Ward | Cut after | Singletons only | Dendrogram is itself a useful deliverable |
| **HDBSCAN** | No | Native (label = -1) | Natural for skewed cluster sizes and explicit noise |
| Spectral on JS-affinity | Yes | None | Elegant but overkill at N=144 |
| Gaussian mixture | Yes (BIC) | Via low posterior | Useful if soft assignments are wanted |

**Recommendation**: HDBSCAN. Noise points are exactly the "outlier" concept we want. No k to tune. Handles the skewed reality where one cluster may have 40 heads (diagonal) and another 5 (syntactic).

Fallback: Ward hierarchical with the linkage dendrogram cut at the knee of sorted merge distances. Use if HDBSCAN collapses everything to noise.

## 3. A concrete 10-feature set

Let A ∈ R^{T × T} be one head's attention matrix (rows sum to 1 after softmax). Let `cls_idx = 0`, `sep_idx = T - 1`, and `PUNCT` be the positions of `.`, `,`, `;`, `!`, `?` in the tokenized sequence.

| Dim | Name | Formula | Captures |
| --- | ---- | ------- | -------- |
| 1 | diag_mass | mean_i A[i, i] | Self-attention |
| 2 | prev_mass | mean_{i≥1} A[i, i-1] | Look-back-one |
| 3 | next_mass | mean_{i≤T-2} A[i, i+1] | Look-forward-one |
| 4 | cls_mass | mean_i A[i, 0] | Global context via `[CLS]` |
| 5 | sep_mass | mean_i A[i, T-1] | `[SEP]`-parking ("no-op") |
| 6 | entropy | mean_i H(A[i, :]) / log(T) | Focus (0 = peaked, 1 = uniform) |
| 7 | local_k3_mass | mean_i Σ_{0<|j-i|≤3} A[i, j] | Local window |
| 8 | punct_mass | mean_i Σ_{j ∈ PUNCT} A[i, j] | Attention to punctuation |
| 9 | row_kl_var | var_i KL(A[i, :] ‖ mean_i A[i, :]) | Row-homogeneity |
| 10 | eff_rank | exp(H(σ² / Σσ²)) where σ are singular values of A | Rank of the attention pattern |

All features except eff_rank lie in [0, 1] and are length-invariant by construction. eff_rank is in [1, T]; dividing by T would normalize it, but keeping it raw preserves ordinal structure.

Features 1 and "1 - off_diag_mass" are identical, so off_diag is omitted. Features 6 and 10 are correlated (entropy and rank both measure spread); keep both because they disagree on peaked-but-scattered patterns.

## 4. Worked example on BERT-base

Sentence S = `"The quick brown fox jumps over the lazy dog."`
BERT-base tokenizer output (T = 12): `[CLS] the quick brown fox jumps over the lazy dog . [SEP]`
`PUNCT = {10}`.

Three illustrative heads (values estimated from Clark et al. 2019, not measured from a live run):

**Head (0, 1), "previous-token" archetype (early layer)**

```
diag=0.05  prev=0.68  next=0.04  cls=0.02  sep=0.03
ent=0.18   loc3=0.85  punc=0.02  rkv=0.02  rank=2.1
```

**Head (10, 10), "SEP-parking" archetype (late layer)**

```
diag=0.06  prev=0.03  next=0.03  cls=0.04  sep=0.76
ent=0.22   loc3=0.12  punc=0.05  rkv=0.08  rank=1.4
```

**Head (5, 5), "broad mixer" archetype (middle layer)**

```
diag=0.09  prev=0.09  next=0.09  cls=0.08  sep=0.08
ent=0.93   loc3=0.28  punc=0.08  rkv=0.01  rank=8.5
```

After robust z-scoring across the full 144-head pool, pairwise Euclidean distances look roughly like:

| | (0,1) | (10,10) | (5,5) | (2,4) |
| --- | --- | --- | --- | --- |
| (0,1) prev-token | 0.0 | 3.8 | 4.1 | 0.6 |
| (10,10) SEP | 3.8 | 0.0 | 4.3 | 3.9 |
| (5,5) mixer | 4.1 | 4.3 | 0.0 | 4.0 |
| (2,4) prev-token | 0.6 | 3.9 | 4.0 | 0.0 |

Two heads of the same archetype are within ~0.6; heads of different archetypes are ~4 apart. The feature space separates semantic roles cleanly. All numbers are illustrative, not measured.

## 5. Expected output on BERT-base

On a sufficiently long sentence (T ≥ 20), we expect HDBSCAN to return something like:

| Cluster | Approx size | Dominant feature | Archetype label |
| ------- | ----------- | ---------------- | --------------- |
| 0 | 30–50 | diag_mass | Diagonal |
| 1 | 20–35 | sep_mass | `[SEP]`-parking (late layers) |
| 2 | 10–20 | prev_mass | Previous-token (early layers) |
| 3 | 5–15 | cls_mass | `[CLS]`-broadcaster |
| 4 | 5–15 | local_k3_mass | Local window |
| 5 | 3–8 | entropy | Broad / uniform mixer |
| -1 | 20–40 | mixed | Noise: syntactic, punctuation, unique |

Sizes are order-of-magnitude estimates grounded in Clark et al.'s head inventory. The noise bucket is where the interesting science is: those are the heads worth rendering individually in IzzyViz.

**Cluster auto-labeling** (optional): for each cluster medoid, label by the feature with the highest z-score. If two features tie within 0.3 σ, chain them: "high-sep + low-entropy → `[SEP]`-parking".

## 6. Pipeline

```
Input: sentence S (or corpus), model M, punctuation set P

 1. Tokenize S          -> tokens (length T)
 2. Forward pass        -> attentions: tuple of L tensors [1, H, T, T]
 3. Flatten grid        -> 144 matrices M_{l, h} of shape [T, T]
 4. Feature extraction  -> f_{l, h} ∈ R^10 for each (l, h)
 5. Stack               -> F ∈ R^{144 × 10}
 6. Robust z-score      -> F' = (F - median(F)) / MAD(F)
 7. HDBSCAN(F', min_cluster_size=3, min_samples=2)
                        -> labels L ∈ {-1, 0, ..., K-1}^144
 8. Medoids             -> for each cluster k ≥ 0,
                          m_k = argmin_i Σ_{j: L_j=k} ‖F'_i - F'_j‖
 9. Outliers            -> all heads with L_i = -1
10. UMAP(F', 2D)        -> Z ∈ R^{144 × 2} (seed pinned; PCA as sanity check)
11. Render:
    a. head_space.pdf            # scatter of Z, colored by L, medoids starred
    b. archetypes/cluster_k.pdf  # IzzyViz on each medoid m_k
    c. outliers/L{l}_H{h}.pdf    # IzzyViz on each noise head
    d. summary.json              # {(l, h): cluster, medoids, noise, labels}
```

Total deliverable: 1 overview plot + K medoid PDFs + |noise| outlier PDFs ≈ 10–20 figures, from an input of 144.

## 7. Validation

Three validation layers, in increasing order of rigor:

1. **Intra- vs inter-cluster separation**. Compute average row-wise JS distance within each cluster and between clusters. Ratio should exceed 2. Silhouette score on the raw-matrix distance should be > 0.2 for a signal.
2. **Archetype labels**. Hand-label 20 well-known BERT heads from Clark et al. (e.g. `(4, 10)` as a coreference head). Check their cluster assignments agree with the published taxonomy.
3. **Stability across inputs**. Run on a second sentence. Check that heads keep their cluster labels under Hungarian matching of cluster IDs. If the per-head assignment is stable above 80 %, the pipeline is robust.

For corpus-level runs, stability becomes a first-class signal: heads with high cross-sentence feature variance are candidates for the stability heatmap.

## 8. Failure modes

1. **Short sentence (T ≤ 5)**: features degenerate; diag_mass saturates because there is nothing else to attend to. Guard: require T ≥ 10 and warn otherwise.
2. **Feature correlation**: entropy ↔ eff_rank, diag_mass ↔ (1 - off_diag). Mitigation: report correlation matrix; optionally PCA with retained variance ≥ 0.9 before clustering (at the cost of interpretable axes).
3. **All-noise HDBSCAN output**: `min_cluster_size` set too high. Fallback: retry with `min_cluster_size=2`, else switch to Ward with a cut at the knee of sorted merge distances.
4. **Outlier contamination of z-score**: one extreme head skews mean and std. Mitigation: robust z-score with median/MAD (already specified).
5. **Layer confound**: early layers attend locally, late layers globally; clustering may mostly recover layer depth. Mitigation: report cluster composition stratified by layer; offer a "residualize by layer" option that subtracts layer-mean features before clustering.
6. **UMAP artifacts**: 2D distances are not faithful, especially for distant pairs. Mitigation: pin random seed; include a PCA scatter as robustness check; never interpret absolute distances in the UMAP plot.
7. **Cluster identity drift across corpus runs**: HDBSCAN cluster IDs are not consistent across runs. Mitigation: Hungarian matching between runs using medoid feature vectors.

## 9. Corpus-level extension

For K sentences:

- Compute `F^{(k)} ∈ R^{144 × 10}` per sentence.
- Aggregate: `F̄ = mean_k F^{(k)}`, `S² = var_k F^{(k)}`.
- Cluster on `F̄` for stable archetype identification.
- Report `S²` as a per-head stability vector: rows with uniformly low variance are reliable archetype members; rows with high variance on a specific dimension (e.g. high variance on `sep_mass` but low variance elsewhere) are context-sensitive heads.
- Optional: concatenate `[F̄ | S²] ∈ R^{144 × 20}` and cluster on the wider feature. This groups heads that are both "similar on average" and "similarly stable".

This corpus-level pipeline is the argument for publication-grade claims about a given model.

## 10. Proposed API

New module `izzyviz/head_clustering.py`:

```python
def extract_head_features(
    attentions,        # tuple from HF output, L × [1, H, T, T]
    tokens,            # list[str] of length T
    cls_idx=0,
    sep_idx=-1,
    punct_tokens=(".", ",", ";", "!", "?"),
) -> np.ndarray:       # shape (L * H, 10)
    ...

def cluster_heads(
    features,          # (N, d) array, already z-scored or not
    method="hdbscan",  # "hdbscan" | "ward" | "kmeans"
    robust_zscore=True,
    **method_kwargs,
) -> HeadClusterResult: # {labels, medoids, noise, linkage, ...}
    ...

def visualize_head_space(
    result,            # HeadClusterResult
    features,          # (N, d) array
    save_path,
    projection="umap", # "umap" | "pca"
    seed=0,
) -> None:
    ...

def render_archetypes_and_outliers(
    result, attentions, tokens, save_dir,
) -> None:
    # dispatches to existing visualize_attention_self_attention
    ...
```

Total implementation footprint estimate: ~400 LOC plus tests. The last function is pure glue around existing IzzyViz renderers.

## 11. What this buys the user

- **Before**: 144 PDFs, manually triaged via BertViz or a thumbnail grid.
- **After**: 1 overview plot that says "here are the 6 patterns, here are 25 weird heads", plus 30 targeted PDFs produced by existing IzzyViz renderers.
- **New**: an explicit `summary.json` that downstream analysis (paper tables, ablation studies) can consume.

The clustering module does not replace any existing IzzyViz function. It is a triage layer that decides *which* heads to pass into them, which is the single gap documented in `LIMITATIONS.md`.

## 12. Open questions

1. **Feature-space vs matrix-space clustering**: run both and report agreement (adjusted Rand index), or pick one? Default: feature-space only, with matrix-space JS as a validation check.
2. **Per-layer vs grid-wide clustering**: per-layer reveals within-layer redundancy (relevant to pruning); grid-wide reveals cross-layer archetypes (relevant to interpretability). Default: grid-wide + a per-layer summary table.
3. **Include variance as a feature?** Only meaningful corpus-level. Default: expose as an opt-in flag.
4. **How to label clusters automatically?** Rule-based on dominant z-scored feature (simple, reliable), or learned from a hand-labeled seed set. Default: rule-based for v1.
5. **Handling cross-attention (encoder-decoder models)**: shape is `[T_dec, T_enc]`, so `diag_mass`, `prev_mass`, `sep_mass` need redefinition. Out of scope for v1; defer to v2.

## 13. Prior art

- Clark, Khandelwal, Levy, Manning, **"What Does BERT Look At?"**, 2019. The canonical taxonomy of BERT attention heads; the source for our validation targets.
- Voita, Talbot, Moiseev, Sennrich, Titov, **"Analyzing Multi-Head Self-Attention"**, 2019. Shows attention head redundancy and motivates pruning; theoretical basis for expecting low-dimensional structure.
- Michel, Levy, Neubig, **"Are Sixteen Heads Really Better Than One?"**, 2019. More evidence that most heads are redundant.
- Vig, **"BertViz"**, 2019. The reference interactive head-browser; IzzyViz + head-clustering is the static-publication complement.
- Abnar, Zuidema, **"Quantifying Attention Flow in Transformers"**, 2020. Attention rollout; related but orthogonal (aggregates across layers rather than clustering across heads).
