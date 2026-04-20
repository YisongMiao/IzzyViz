"""
Pseudo-figure generator for the N_abs head-clustering walkthrough.

Produces five PDFs that illustrate, on synthetic but plausible data, the output
of a medium-complexity pipeline: 6-feature representation + Ward hierarchical
clustering + distance-to-medoid outlier flagging, applied to a 12 x 12 = 144
head grid on a 12-token sentence.

No real transformer is run. Matrices are drawn from archetype templates
(diagonal, previous-token, SEP-parking, CLS-broadcaster, local-window,
uniform-mixer, random) with additive noise.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import median_abs_deviation

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PNG_DPI = 200
np.random.seed(SEED)

def save_both(fig, stem):
    fig.savefig(os.path.join(OUT, f"{stem}.pdf"))
    fig.savefig(os.path.join(OUT, f"preview_{stem}.png"),
                dpi=PNG_DPI, bbox_inches="tight")

T = 12
L, H = 12, 12
N = L * H

TOKENS = ["[CLS]", "the", "quick", "brown", "fox", "jumps",
         "over", "the", "lazy", "dog", ".", "[SEP]"]

# ----------------------------------------------------------------------
# 1. Archetype templates
# ----------------------------------------------------------------------

def _normalize(A):
    A = np.clip(A, 1e-6, None)
    return A / A.sum(axis=1, keepdims=True)

def tpl_diagonal(T, noise=0.04):
    A = np.eye(T) * 0.85 + np.random.rand(T, T) * noise
    return _normalize(A)

def tpl_prev(T, noise=0.04):
    A = np.random.rand(T, T) * noise
    for i in range(1, T):
        A[i, i - 1] = 0.7 + np.random.rand() * 0.1
    A[0, 0] = 0.9
    return _normalize(A)

def tpl_sep(T, noise=0.04):
    A = np.random.rand(T, T) * noise
    A[:, T - 1] = 0.75 + np.random.rand(T) * 0.1
    return _normalize(A)

def tpl_cls(T, noise=0.04):
    A = np.random.rand(T, T) * noise
    A[:, 0] = 0.65 + np.random.rand(T) * 0.1
    return _normalize(A)

def tpl_local(T, noise=0.04, w=3):
    A = np.random.rand(T, T) * noise
    for i in range(T):
        for j in range(max(0, i - w), min(T, i + w + 1)):
            A[i, j] += 1.0 / (abs(i - j) + 1)
    return _normalize(A)

def tpl_uniform(T, noise=0.01):
    A = np.ones((T, T)) / T + np.random.rand(T, T) * noise
    return _normalize(A)

def tpl_random(T):
    return _normalize(np.random.rand(T, T))

GENS = {
    "diagonal": tpl_diagonal,
    "prev":     tpl_prev,
    "sep":      tpl_sep,
    "cls":      tpl_cls,
    "local":    tpl_local,
    "uniform":  tpl_uniform,
    "random":   tpl_random,
}

# ----------------------------------------------------------------------
# 2. Build 144 synthetic heads
# ----------------------------------------------------------------------

ARCH_COUNTS = {
    "diagonal": 42, "sep": 28, "prev": 16, "cls": 12,
    "local": 10, "uniform": 6, "random": 30,
}
assert sum(ARCH_COUNTS.values()) == N

true_archetypes = []
for name, c in ARCH_COUNTS.items():
    true_archetypes.extend([name] * c)
rng = np.random.default_rng(SEED)
rng.shuffle(true_archetypes)

matrices = np.array([GENS[a](T) for a in true_archetypes])   # (144, T, T)
layer_idx = np.array([i // H for i in range(N)])
head_idx  = np.array([i %  H for i in range(N)])

# ----------------------------------------------------------------------
# 3. Five-feature extraction
# ----------------------------------------------------------------------

FEATURE_NAMES = ["diag_mass", "prev_mass", "sep_mass", "cls_mass", "entropy", "local_k3"]

def features(A):
    t = A.shape[0]
    diag = np.mean([A[i, i] for i in range(t)])
    prev = np.mean([A[i, i - 1] for i in range(1, t)])
    sep  = np.mean(A[:, t - 1])
    cls  = np.mean(A[:, 0])
    ent  = -np.sum(A * np.log(A + 1e-12), axis=1).mean() / np.log(t)
    local = 0.0
    for i in range(t):
        for j in range(max(0, i - 3), min(t, i + 4)):
            if j != i:
                local += A[i, j]
    local = local / t
    return np.array([diag, prev, sep, cls, ent, local])

F = np.array([features(A) for A in matrices])   # (144, 5)

med = np.median(F, axis=0)
mad = median_abs_deviation(F, axis=0) + 1e-6
Fz = (F - med) / mad

# ----------------------------------------------------------------------
# 4. Ward clustering + outlier flagging
# ----------------------------------------------------------------------

D_condensed = pdist(Fz, metric="euclidean")
D_square    = squareform(D_condensed)
Zlink       = linkage(D_condensed, method="ward")

K = 6
raw_labels = fcluster(Zlink, t=K, criterion="maxclust")

medoid = {}
for c in np.unique(raw_labels):
    idx = np.where(raw_labels == c)[0]
    sub = D_square[np.ix_(idx, idx)]
    medoid[c] = int(idx[np.argmin(sub.sum(axis=1))])

dist_to_own_medoid = np.array([
    np.linalg.norm(Fz[i] - Fz[medoid[raw_labels[i]]]) for i in range(N)
])

threshold = np.percentile(dist_to_own_medoid, 92)
is_outlier = dist_to_own_medoid > threshold

sizes = {c: int((raw_labels == c).sum()) for c in np.unique(raw_labels)}
for c, s in sizes.items():
    if s <= 3:
        is_outlier[raw_labels == c] = True

display = raw_labels.copy()
display[is_outlier] = -1

# ----------------------------------------------------------------------
# 5. Auto-label each retained cluster by its medoid's dominant feature
# ----------------------------------------------------------------------

DOMINANT_NAME = {
    "diag_mass":  "Diagonal",
    "prev_mass":  "Previous-token",
    "sep_mass":   "SEP-parking",
    "cls_mass":   "CLS-broadcaster",
    "entropy":    "Broad mixer",
    "local_k3":   "Local window",
}

cluster_label_text = {}
retained_clusters = [c for c in sizes if sizes[c] > 3]
pairs = []
for c in retained_clusters:
    mi = medoid[c]
    for f_idx, feat in enumerate(FEATURE_NAMES):
        pairs.append((Fz[mi][f_idx], c, feat))
pairs.sort(reverse=True)
claimed_feat = set()
claimed_cluster = set()
for _, c, feat in pairs:
    if c in claimed_cluster or feat in claimed_feat:
        continue
    cluster_label_text[c] = DOMINANT_NAME.get(feat, f"Cluster {c}")
    claimed_cluster.add(c)
    claimed_feat.add(feat)
for c in retained_clusters:
    if c not in cluster_label_text:
        cluster_label_text[c] = f"Cluster {c}"

# ----------------------------------------------------------------------
# 6. PCA to 2D
# ----------------------------------------------------------------------

Fc = Fz - Fz.mean(axis=0)
U, S_, Vt = np.linalg.svd(Fc, full_matrices=False)
pc = Fc @ Vt[:2].T   # (144, 2)

# ----------------------------------------------------------------------
# 7. Colors
# ----------------------------------------------------------------------

palette = plt.cm.tab10.colors
retained = sorted([c for c in sizes if sizes[c] > 3])
cluster_color = {c: palette[i % 10] for i, c in enumerate(retained)}

def color_for(label):
    if label == -1:
        return (0.55, 0.55, 0.55)
    return cluster_color.get(label, (0.3, 0.3, 0.3))

# ----------------------------------------------------------------------
# Figure 1: head grid overview
# ----------------------------------------------------------------------

def fig1():
    fig, axes = plt.subplots(L, H, figsize=(11, 11))
    fig.suptitle("Figure 1: All 144 attention heads (12 layers x 12 heads)\n"
                 "Thumbnail border color = cluster assignment (same palette as Figs 2 and 3)\n"
                 "Thick black frame = cluster medoid. Gray border = outlier.",
                 fontsize=11)
    medoid_set = set(medoid[c] for c in retained)
    for l in range(L):
        for h in range(H):
            i = l * H + h
            ax = axes[l, h]
            ax.imshow(matrices[i], cmap="Blues", vmin=0, vmax=0.6,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            border_color = color_for(display[i])
            is_med = i in medoid_set
            border_width = 2.5 if is_med else 1.2
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
            if is_med:
                for spine in ax.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(2.5)
                ax.add_patch(mpatches.Rectangle(
                    (-0.5, -0.5), T, T, fill=False,
                    edgecolor=color_for(display[i]), linewidth=4.0,
                    transform=ax.transData, clip_on=False))
        axes[l, 0].set_ylabel(f"L{l}", fontsize=7.5, rotation=0,
                              labelpad=10, fontweight="bold")
    for h in range(H):
        axes[0, h].set_title(f"H{h}", fontsize=7.5, pad=3, fontweight="bold")
    fig.text(0.5, 0.035,
             "Takeaway: once each thumbnail is colored by cluster, the six archetype "
             "regions (and the gray outliers) become visible at a glance.",
             ha="center", fontsize=9, style="italic")
    plt.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.075,
                        wspace=0.12, hspace=0.12)
    save_both(fig, "fig1_head_grid")
    plt.close(fig)

# ----------------------------------------------------------------------
# Figure 2: Ward dendrogram
# ----------------------------------------------------------------------

def fig2():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    from scipy.cluster.hierarchy import set_link_color_palette, leaves_list
    from matplotlib.lines import Line2D

    dendro_palette_tab10 = plt.cm.tab10.colors
    dendro_palette_hex = [
        "#%02x%02x%02x" % tuple(int(255 * x) for x in dendro_palette_tab10[i])
        for i in range(len(retained))
    ]
    set_link_color_palette(dendro_palette_hex)
    cut = Zlink[-(K - 1), 2]

    dendrogram(Zlink, color_threshold=cut, above_threshold_color="#888888",
               no_labels=True, ax=ax)
    cut_line = ax.axhline(cut, color="red", linestyle="--", linewidth=1,
                          label=f"cut at d = {cut:.2f}  (K = {K} clusters)")
    ax.set_title("Figure 2: Ward hierarchical clustering on z-scored features\n"
                 "Each leaf is one of the 144 heads; horizontal cut selects cluster count",
                 fontsize=11)
    ax.set_xlabel("144 heads (ordered by merge)")
    ax.set_ylabel("Ward linkage distance")

    # Map each cluster to the palette color scipy actually painted its subtree.
    # scipy assigns palette colors in the order each below-threshold subtree is
    # encountered during left-to-right drawing; leaves_list gives that order.
    dendro_leaf_order = leaves_list(Zlink)
    cluster_first_drawing_pos = {}
    for pos, original_idx in enumerate(dendro_leaf_order):
        c = raw_labels[original_idx]
        if c not in cluster_first_drawing_pos:
            cluster_first_drawing_pos[c] = pos
    subtree_cluster_order = sorted(
        retained, key=lambda c: cluster_first_drawing_pos[c]
    )
    handles = []
    for idx, c in enumerate(subtree_cluster_order):
        color = dendro_palette_hex[idx % len(dendro_palette_hex)]
        handles.append(Line2D([0], [0], color=color, lw=4,
                              label=cluster_label_text.get(c, f"cluster {c}")))

    legend_clusters = ax.legend(handles=handles, loc="upper left",
                                title="Subtree color  =  cluster",
                                fontsize=9, title_fontsize=9, framealpha=0.92)
    ax.add_artist(legend_clusters)
    ax.legend(handles=[cut_line], loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    save_both(fig, "fig2_dendrogram")
    plt.close(fig)

# ----------------------------------------------------------------------
# Figure 3: 2D PCA scatter
# ----------------------------------------------------------------------

def fig3():
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    rng = np.random.default_rng(SEED + 1)

    JITTER_STD = 1.0
    jitter = rng.normal(0.0, JITTER_STD, size=pc.shape)
    medoid_set = set(medoid[c] for c in retained)
    pc_disp = pc.copy()
    for i in range(N):
        if i not in medoid_set:
            pc_disp[i] = pc[i] + jitter[i]

    for c in retained:
        mi = medoid[c]
        ax.scatter(pc[mi, 0], pc[mi, 1],
                   s=2200, facecolor=cluster_color[c],
                   edgecolor="none", alpha=0.12, zorder=1)

    for c in retained:
        mask = (raw_labels == c) & (~is_outlier)
        mask_nonmed = mask.copy()
        for mi_ in medoid_set:
            mask_nonmed[mi_] = False
        ax.scatter(pc_disp[mask_nonmed, 0], pc_disp[mask_nonmed, 1],
                   color=cluster_color[c], s=48, alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3,
                   label=cluster_label_text.get(c, f"cluster {c}"))

    ax.scatter(pc_disp[is_outlier, 0], pc_disp[is_outlier, 1],
               color="red", marker="x", s=75,
               linewidth=1.5, zorder=4, label="outlier (far from medoid)")

    for c in retained:
        mi = medoid[c]
        ax.scatter(pc[mi, 0], pc[mi, 1],
                   s=360, color=cluster_color[c],
                   edgecolor="black", linewidth=1.8, zorder=5)
        ax.scatter(pc[mi, 0], pc[mi, 1],
                   s=52, color="white",
                   edgecolor="black", linewidth=0.7, zorder=6)
        ax.annotate(
            f"L{layer_idx[mi]}H{head_idx[mi]}   n={sizes[c]}",
            (pc[mi, 0], pc[mi, 1]),
            xytext=(12, 10), textcoords="offset points",
            fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.9,
                      edgecolor=cluster_color[c], linewidth=0.8),
            zorder=7,
        )

    ax.set_title("Figure 3: Head space (PCA on 6 z-scored features)\n"
                 "Jittered dots reveal stacked heads; bullseye = medoid; halo area "
                 "is proportional to cluster size",
                 fontsize=11)
    ax.set_xlabel(f"PC 1  ({100 * S_[0] ** 2 / (S_ ** 2).sum():.1f}% variance)")
    ax.set_ylabel(f"PC 2  ({100 * S_[1] ** 2 / (S_ ** 2).sum():.1f}% variance)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    save_both(fig, "fig3_head_space")
    plt.close(fig)

# ----------------------------------------------------------------------
# Figure 4: archetype gallery (medoids + mean feature signatures)
# ----------------------------------------------------------------------

def fig4():
    n = len(retained)
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, n, height_ratios=[2, 1], hspace=0.55, wspace=0.35)
    for col, c in enumerate(retained):
        mi = medoid[c]
        ax_top = fig.add_subplot(gs[0, col])
        ax_top.imshow(matrices[mi], cmap="Blues", vmin=0, vmax=0.8)
        ax_top.set_title(
            f"{cluster_label_text.get(c, f'Cluster {c}')}\n"
            f"medoid L{layer_idx[mi]}H{head_idx[mi]}   (n = {sizes[c]})",
            fontsize=9)
        ax_top.set_xticks(range(T)); ax_top.set_yticks(range(T))
        ax_top.set_xticklabels(TOKENS, fontsize=5.5, rotation=90)
        ax_top.set_yticklabels(TOKENS, fontsize=5.5)

        ax_bot = fig.add_subplot(gs[1, col])
        mask = (raw_labels == c) & (~is_outlier)
        mean_feat = F[mask].mean(axis=0)
        bars = ax_bot.bar(range(len(FEATURE_NAMES)), mean_feat, color=cluster_color[c],
                          edgecolor="black", linewidth=0.4)
        ax_bot.set_xticks(range(len(FEATURE_NAMES)))
        ax_bot.set_xticklabels(FEATURE_NAMES, fontsize=6, rotation=40, ha="right")
        ax_bot.set_ylim(0, 1.0)
        ax_bot.set_ylabel("mean", fontsize=7)
        ax_bot.tick_params(axis="y", labelsize=6)
        ax_bot.grid(axis="y", alpha=0.2)

    fig.suptitle("Figure 4: Archetype gallery. Top row: medoid heatmap per cluster.\n"
                 "Bottom row: cluster-mean feature signature (the 'why' of the label).",
                 fontsize=11, y=0.98)
    save_both(fig, "fig4_archetypes")
    plt.close(fig)

# ----------------------------------------------------------------------
# Figure 5: outlier gallery
# ----------------------------------------------------------------------

def fig5():
    out_idx = np.where(is_outlier)[0]
    out_sorted = out_idx[np.argsort(-dist_to_own_medoid[out_idx])]
    pick = out_sorted[:6]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(matrices[i], cmap="Oranges", vmin=0, vmax=0.8)
        ax.set_title(f"L{layer_idx[i]}H{head_idx[i]}   "
                     f"dist to medoid = {dist_to_own_medoid[i]:.2f}",
                     fontsize=9)
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(TOKENS, fontsize=5.5, rotation=90)
        ax.set_yticklabels(TOKENS, fontsize=5.5)
    fig.suptitle(f"Figure 5: Outlier heads ({int(is_outlier.sum())} total, "
                 "top 6 by distance to own cluster medoid)\n"
                 "These are the heads that resist compression into an archetype.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_both(fig, "fig5_outliers")
    plt.close(fig)

# ----------------------------------------------------------------------

def summary():
    print("Retained clusters:")
    for c in retained:
        mi = medoid[c]
        n_effective = int(((raw_labels == c) & (~is_outlier)).sum())
        print(f"  cluster {c:>2}  label={cluster_label_text.get(c):<18}  "
              f"medoid=L{layer_idx[mi]}H{head_idx[mi]}  "
              f"size={sizes[c]}  effective(non-outlier)={n_effective}")
    print(f"Outliers flagged: {int(is_outlier.sum())} / {N}")
    print(f"Ward cut distance: {Zlink[-(K - 1), 2]:.3f}")
    print(f"PC1 variance: {100 * S_[0] ** 2 / (S_ ** 2).sum():.1f}%   "
          f"PC2 variance: {100 * S_[1] ** 2 / (S_ ** 2).sum():.1f}%")

    print("\nMedoid feature values (raw, un-z-scored):")
    hdr = "  " + "archetype".ljust(18) + "medoid".ljust(9) + \
          "".join(f.rjust(11) for f in FEATURE_NAMES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in retained:
        mi = medoid[c]
        row = "  " + cluster_label_text.get(c, f"C{c}").ljust(18) + \
              f"L{layer_idx[mi]}H{head_idx[mi]}".ljust(9) + \
              "".join(f"{F[mi][k]:>11.2f}" for k in range(len(FEATURE_NAMES)))
        print(row)

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    summary()
    print(f"Wrote figures to {OUT}")
