"""Figures for the representation report (static PNGs, light theme)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# categorical slots in fixed order; subgroups and feature kinds take slots 1..k
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def _group_colors(groups):
    return {g: SERIES[i] for i, g in enumerate(groups)}


def plot_prevalence(X, feature_kind, path, tau=None):
    """Rank-ordered feature prevalence: the long tail the threshold cuts."""
    prevalence = X.sum(axis=0) / len(X)
    order = np.argsort(-prevalence, kind="stable")
    prevalence, kinds = prevalence[order], np.asarray(feature_kind)[order]
    ranks = np.arange(1, len(prevalence) + 1)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, kind in enumerate(dict.fromkeys(feature_kind)):
        m = kinds == kind
        ax.bar(ranks[m], prevalence[m], width=1.0, color=SERIES[i], label=kind, linewidth=0)
    if tau is not None:
        ax.axhline(tau, color=TEXT_2, linewidth=1, linestyle="--")
        ax.annotate(f"chosen τ = {tau:g}", (len(ranks), tau), xytext=(-4, 4),
                    textcoords="offset points", ha="right", color=TEXT_2, fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlim(0, len(ranks) + 1)
    ax.set_xlabel("Feature rank (most to least prevalent)")
    ax.set_ylabel("Prevalence (share of individuals)")
    ax.set_title("Feature prevalence in the pooled population", loc="left")
    ax.legend(loc="upper right", title="Feature kind", title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_measures_by_group(assessment, path):
    """Small multiples: the three Phase I measures per subgroup."""
    groups = sorted(assessment["group"].unique())
    colors = _group_colors(groups)
    panels = [
        ("pr_i_normalized", "Normalized Pr_i (log scale)", True),
        ("j_pop", "j_pop: mean Jaccard to others", False),
        ("rarity", "Rare(d_i): median rarity (%)", False),
    ]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (col, title, log) in zip(axes, panels):
        data = [assessment.loc[assessment["group"] == g, col].to_numpy() for g in groups]
        if log:
            data = [np.log10(np.clip(d, 1e-300, None)) for d in data]
        ax.boxplot(data, widths=0.5, showfliers=False,
                   medianprops=dict(color=TEXT, linewidth=1.5),
                   boxprops=dict(color=TEXT_2, linewidth=0.8),
                   whiskerprops=dict(color=TEXT_2, linewidth=0.8),
                   capprops=dict(color=TEXT_2, linewidth=0.8))
        for i, (g, d) in enumerate(zip(groups, data)):
            ax.scatter(rng.normal(i + 1, 0.06, len(d)), d, s=10, alpha=0.45,
                       color=colors[g], edgecolor="none")
        ax.set_xticks(range(1, len(groups) + 1),
                      [f"{g}\n(n={len(d)})" for g, d in zip(groups, data)])
        ax.set_title(title, loc="left", fontsize=10)
        if log:
            ax.set_ylabel("log10")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Population heterogeneity by subgroup", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_calibration(summary, chosen_tau, groups, path, min_coverage=None):
    """Coverage-accuracy trade-off (left) and per-group coverage across thresholds (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # left: J_system vs held-out accuracy, one point per threshold
    ax1.errorbar(summary["j_system"], summary["accuracy"], yerr=summary["accuracy_sd"],
                 color=SERIES[0], ecolor=GRID, elinewidth=2, capsize=0, linewidth=2,
                 marker="o", markersize=5)
    if min_coverage is not None:
        ax1.axvline(min_coverage, color=TEXT_2, linewidth=1, linestyle=":")
        ax1.annotate("coverage floor", (min_coverage, ax1.get_ylim()[0]), xytext=(4, 4),
                     textcoords="offset points", color=TEXT_2, fontsize=8)
    chosen = summary[summary["tau"] == chosen_tau].iloc[0]
    ax1.scatter([chosen["j_system"]], [chosen["accuracy"]], s=110, facecolor="none",
                edgecolor=TEXT, linewidth=1.5, zorder=5)
    # label the chosen point plus any point at least 0.06 coverage away from every label so far
    placed = [chosen["j_system"]]
    for _, r in summary.iterrows():
        if r["tau"] != chosen_tau and min(abs(r["j_system"] - p) for p in placed) < 0.06:
            continue
        placed.append(r["j_system"])
        is_chosen = r["tau"] == chosen_tau
        ax1.annotate(f"τ={r['tau']:g}", (r["j_system"], r["accuracy"]),
                     xytext=(0, 10) if is_chosen else (5, -11), textcoords="offset points",
                     ha="center" if is_chosen else "left", fontsize=7.5,
                     color=TEXT if is_chosen else TEXT_2,
                     fontweight="bold" if is_chosen else "normal")
    ax1.set_xlabel("J_system: mean held-out coverage")
    ax1.set_ylabel("Held-out accuracy (5-fold mean ± sd)")
    ax1.set_title("Coverage–accuracy trade-off", loc="left")

    # right: per-group J vs threshold, thresholds on an index axis (τ = 0 has no log position)
    x = np.arange(len(summary))
    colors = _group_colors(groups)
    for g in groups:
        y = summary[f"j_{g}"].to_numpy()
        ax2.plot(x, y, color=colors[g], linewidth=2, marker="o", markersize=4, label=f"group {g}")
    idx = int(np.where(summary["tau"].to_numpy() == chosen_tau)[0][0])
    ax2.axvline(idx, color=TEXT_2, linewidth=1, linestyle="--")
    ax2.set_xticks(x, [f"{t:g}" for t in summary["tau"]], rotation=45)
    ax2.set_xlabel("Prevalence threshold τ")
    ax2.set_ylabel("J_system: mean j_sys (held-out)")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Coverage by subgroup", loc="left")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_fit_scores(reference_j, arrival_df, groups, path):
    """New arrivals' fit scores by group, against the pooled population's distribution."""
    colors = _group_colors(groups)
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ref = np.asarray(reference_j)
    for q, label in ((5, "P5"), (25, "P25")):
        v = np.percentile(ref, q)
        ax.axvline(v, color=TEXT_2, linewidth=1, linestyle="--")
        ax.annotate(f"pooled {label}", (v, len(groups) + 0.35), xytext=(3, 0),
                    textcoords="offset points", fontsize=8, color=TEXT_2)
    for i, g in enumerate(groups):
        v = arrival_df.loc[arrival_df["group"] == g, "fit_score"].to_numpy()
        ax.scatter(v, rng.normal(i + 1, 0.07, len(v)), s=22, alpha=0.7, color=colors[g],
                   edgecolor=SURFACE, linewidth=0.5)
    ax.set_yticks(range(1, len(groups) + 1),
                  [f"group {g} (n={(arrival_df['group'] == g).sum()})" for g in groups])
    ax.set_ylim(0.4, len(groups) + 0.7)
    lowest = min(arrival_df["fit_score"].min(), np.percentile(ref, 5))
    ax.set_xlim(max(0.0, np.floor(lowest * 10) / 10 - 0.05), 1.02)
    ax.set_xlabel("Fit score j_sys (share of reported features the system keeps)")
    ax.set_title("Fit scores of new arrivals", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
