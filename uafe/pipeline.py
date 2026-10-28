"""The six pipeline steps.

Phase I   assess_population, population_summary        (steps 1-2)
Phase II  select_features, calibrate, choose_threshold  (steps 3-4)
Phase III build_system, representation_report, fit_instance, fit_instance_from_names  (steps 5-6)

Two different Jaccard quantities appear:
  j_pop  (Phase I)        mean J(d_i, d_j) over all other individuals j: how much i
                          resembles the rest of the population.
  j_sys  (Phase II / III) J(d_i, r_i), where r_i = d_i restricted to the selected
                          features is the system's representation of i. This equals the
                          share of i's reported features that the system keeps.
                          J_system is the mean of j_sys over individuals.
"""
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .measures import (compute_log10_pr_i, compute_normalized_pr_i, compute_rarity,
                       compute_similarity, compute_system_j, jaccard_matrix,
                       system_representation)


# ======================================================================================
# Phase I: data & assessment
# ======================================================================================

def assess_population(X, group=None):
    """Pr_i, j_pop and Rare(d_i) for every reference representation (step 2)."""
    X = np.asarray(X, dtype=int)
    n = len(X)
    pooled = X.sum(axis=0)

    J = jaccard_matrix(X)
    j_pop = (J.sum(axis=1) - np.diag(J)) / (n - 1)

    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)

    return pd.DataFrame({
        "group": group if group is not None else "all",
        "n_features": X.sum(axis=1),
        "log10_pr_i": [compute_log10_pr_i(pooled, d) for d in X],
        "pr_i_normalized": [compute_normalized_pr_i(pooled, d) for d in X],
        "j_pop": j_pop,
        "rarity": [compute_rarity(d, pooled, n) for d in X],
        "unique_vector": counts[inverse.ravel()] == 1,
    })


def population_summary(X, assessment):
    X = np.asarray(X, dtype=int)
    n, L = X.shape
    counts = X.sum(axis=0)
    return {
        "individuals": n,
        "distinct features": int((counts > 0).sum()),
        "features per individual (median)": float(np.median(X.sum(axis=1))),
        "features held by a single individual": int((counts == 1).sum()),
        "features held by <1% of individuals": int(((counts > 0) & (counts < 0.01 * n)).sum()),
        "individuals with a unique feature vector": f"{assessment['unique_vector'].mean():.0%}",
        "median j_pop": round(float(assessment["j_pop"].median()), 3),
        "median normalized Pr_i": float(f"{assessment['pr_i_normalized'].median():.3g}"),
        "median Rare(d_i) (%)": round(float(assessment["rarity"].median()), 1),
    }


# ======================================================================================
# Phase II: feature selection & training
# ======================================================================================

DEFAULT_THRESHOLDS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,
                      0.1, 0.15, 0.2, 0.3, 0.45)


def feature_prevalence(X):
    X = np.asarray(X)
    return X.sum(axis=0) / len(X)


def select_features(X_ref, tau):
    """Step 3: keep features whose prevalence in the pooled population is >= tau.

    Features never observed in X_ref are always excluded: a system cannot represent
    what it has never seen, even at tau = 0.
    """
    X_ref = np.asarray(X_ref)
    return (X_ref.sum(axis=0) > 0) & (feature_prevalence(X_ref) >= tau)


class SelectedFeatureModel:
    """Logistic regression on the selected features (majority class if none are selected)."""

    def __init__(self, selected, C=1.0):
        self.selected = np.asarray(selected, dtype=bool)
        self.C = C

    def fit(self, X, y):
        if self.selected.any():
            self.clf = LogisticRegression(C=self.C, max_iter=5000)
            self.clf.fit(X[:, self.selected], y)
        else:
            self.clf = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(X), 1)), y)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        if self.selected.any():
            return self.clf.predict(X[:, self.selected])
        return self.clf.predict(np.zeros((len(X), 1)))


@dataclass
class Calibration:
    summary: pd.DataFrame        # one row per threshold
    j_sys: pd.DataFrame          # out-of-fold j_sys per individual (rows) and threshold (columns)


def calibrate(X, y, group, thresholds=DEFAULT_THRESHOLDS, n_splits=5, seed=0, C=1.0):
    """Step 4: sweep the prevalence threshold and record coverage (J_system) and accuracy.

    Selection and training only see the training folds; both J_system and accuracy are
    measured on the held-out fold, so the coverage numbers include individuals whose
    features the system has never seen.
    """
    X, y, group = np.asarray(X, dtype=int), np.asarray(y), np.asarray(group)
    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y))
    groups = sorted(set(group))

    rows, j_by_tau = [], {}
    for tau in thresholds:
        pred = np.empty(len(X), dtype=y.dtype)
        j = np.empty(len(X))
        fold_acc, n_selected = [], []
        for train, test in folds:
            selected = select_features(X[train], tau)
            model = SelectedFeatureModel(selected, C=C).fit(X[train], y[train])
            pred[test] = model.predict(X[test])
            j[test] = compute_system_j(X[test], selected)
            fold_acc.append(np.mean(pred[test] == y[test]))
            n_selected.append(selected.sum())

        correct = pred == y
        row = {
            "tau": tau,
            "n_selected": float(np.mean(n_selected)),
            "j_system": j.mean(),
            "j_p10": np.quantile(j, 0.10),
            "accuracy": np.mean(fold_acc),
            "accuracy_sd": np.std(fold_acc),
        }
        for g in groups:
            row[f"j_{g}"] = j[group == g].mean()
        for g in groups:
            row[f"acc_{g}"] = correct[group == g].mean()
        row["j_worst_group"] = min(row[f"j_{g}"] for g in groups)
        rows.append(row)
        j_by_tau[tau] = j

    return Calibration(pd.DataFrame(rows), pd.DataFrame(j_by_tau))


def choose_threshold(summary, min_coverage=0.6):
    """Highest-accuracy threshold whose J_system is at least `min_coverage`.

    Falls back to the highest-accuracy threshold overall if none meets the floor.
    Ties go to the lower threshold (more coverage).
    """
    candidates = summary[summary["j_system"] >= min_coverage]
    if candidates.empty:
        candidates = summary
    return float(candidates.loc[candidates["accuracy"].idxmax(), "tau"])


# ======================================================================================
# Phase III: deployment
# ======================================================================================

@dataclass
class RepresentationSystem:
    tau: float
    selected: np.ndarray        # (L,) bool mask of the features the system uses
    feature_names: np.ndarray
    pooled: np.ndarray          # pooled population (feature counts)
    n_reference: int
    model: SelectedFeatureModel
    reference: pd.DataFrame     # Phase I measures + out-of-fold j_sys at tau


def build_system(population, assessment, calibration, tau, C=1.0):
    """Freeze the chosen threshold and retrain on the full pooled population."""
    selected = select_features(population.X, tau)
    model = SelectedFeatureModel(selected, C=C).fit(population.X, population.y)
    reference = assessment.copy()
    reference["j_sys"] = calibration.j_sys[tau].to_numpy()
    return RepresentationSystem(tau, selected, population.feature_names,
                                population.X.sum(axis=0), len(population), model, reference)


BANDS = ("poorly represented", "partially represented", "well represented")


@dataclass
class FitReport:
    """Step 6: how well the system represents one individual."""
    instance_id: str
    fit_score: float                 # j_sys: J between d_x and its system representation
    fit_percentile: float            # relative to the pooled population's out-of-fold j_sys
    band: str
    n_reported: int
    n_represented: int
    log10_pr_i: float
    pr_i_normalized: float
    pr_i_normalized_percentile: float
    rarity: float
    rarity_percentile: float
    omitted_features: list = field(default_factory=list)
    unseen_features: list = field(default_factory=list)  # reported by no training individual
    prediction: int | None = None

    def to_dict(self):
        d = asdict(self)
        d["omitted_features"] = ";".join(self.omitted_features)
        d["unseen_features"] = ";".join(self.unseen_features)
        d["n_unseen"] = len(self.unseen_features)
        return d

    def to_markdown(self, max_omitted=15):
        omitted = ", ".join(self.omitted_features[:max_omitted])
        if len(self.omitted_features) > max_omitted:
            omitted += f", ... (+{len(self.omitted_features) - max_omitted} more)"
        return "\n".join([
            f"#### Fit report: {self.instance_id}",
            "",
            f"- **Fit score (j_sys)**: {self.fit_score:.2f}, at the {_ordinal(self.fit_percentile)} percentile "
            f"of the pooled population. **{self.band}**",
            f"- Features the system keeps: {self.n_represented} of {self.n_reported} reported",
            f"- Collision probability: log10 Pr_i = {self.log10_pr_i:.1f}; normalized Pr_i = "
            f"{self.pr_i_normalized:.3g} ({_ordinal(self.pr_i_normalized_percentile)} percentile)",
            f"- Rarity Rare(d_i): {self.rarity:.1f}% ({_ordinal(self.rarity_percentile)} percentile)",
            f"- Features the system omits: {omitted or 'none'}",
            f"- Features no training individual reported: {', '.join(self.unseen_features) or 'none'}",
        ])


def _ordinal(p):
    n = int(round(p))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _percentile(reference, value):
    """Mid-rank percentile of `value` within `reference` (ties count half)."""
    reference = np.asarray(reference)
    return 100.0 * (np.mean(reference < value) + 0.5 * np.mean(reference == value))


def fit_instance(system, x, instance_id="new", band_cutoffs=(5, 25)):
    """Score a new individual x against a deployed system.

    Pr_i and Rare(d_i) treat x as joining the pooled population, the same way each
    training individual is part of the pool it is measured against. This also keeps
    Pr_i defined when x reports features no training individual reported.
    Bands come from the fit percentile: below band_cutoffs[0] is poor, below
    band_cutoffs[1] is partial, anything else is well represented.
    """
    x = np.asarray(x, dtype=int)
    r = system_representation(x, system.selected)
    fit = float(compute_similarity(r, x)) if x.any() else 0.0
    pooled = system.pooled + x
    pr_norm = compute_normalized_pr_i(pooled, x)
    rarity = compute_rarity(x, pooled, system.n_reference + 1)
    ref = system.reference

    pct = _percentile(ref["j_sys"], fit)
    band = BANDS[0] if pct < band_cutoffs[0] else BANDS[1] if pct < band_cutoffs[1] else BANDS[2]
    omitted = system.feature_names[(x == 1) & ~system.selected].tolist()
    unseen = system.feature_names[(x == 1) & (system.pooled == 0)].tolist()

    return FitReport(
        instance_id=str(instance_id),
        fit_score=fit,
        fit_percentile=pct,
        band=band,
        n_reported=int(x.sum()),
        n_represented=int(r.sum()),
        log10_pr_i=compute_log10_pr_i(pooled, x),
        pr_i_normalized=pr_norm,
        pr_i_normalized_percentile=_percentile(ref["pr_i_normalized"], pr_norm),
        rarity=rarity,
        rarity_percentile=_percentile(ref["rarity"], rarity),
        omitted_features=omitted,
        unseen_features=unseen,
        # the model only knows the columns it was trained on (x may be extended, see below)
        prediction=int(system.model.predict(x[:len(system.model.selected)])[0]),
    )


def fit_instance_from_names(system, reported, instance_id="new", band_cutoffs=(5, 25)):
    """Score a new individual given as feature names, including names the system has never seen.

    Unseen features are appended as temporary columns with a training count of 0 that
    are never selected: they lower the fit score, count as held by x alone in Pr_i and
    Rare(d_i), and are ignored by the model. The deployed system itself is unchanged;
    the unseen names are returned so they can be added at the next recalibration.
    Pr_i is then computed over L + k columns rather than the training data's L columns,
    which only adds zero-count columns but means the Pr_i percentile is not on exactly the
    same scale.
    """
    reported = list(dict.fromkeys(reported))
    known = {name: k for k, name in enumerate(system.feature_names)}
    unseen = [f for f in reported if f not in known]

    x = np.zeros(len(system.feature_names) + len(unseen), dtype=int)
    x[[known[f] for f in reported if f in known]] = 1
    x[len(system.feature_names):] = 1

    extended = replace(
        system,
        selected=np.concatenate([system.selected, np.zeros(len(unseen), dtype=bool)]),
        feature_names=np.concatenate([system.feature_names, np.array(unseen, dtype=object)]),
        pooled=np.concatenate([system.pooled, np.zeros(len(unseen), dtype=int)]),
    )
    return fit_instance(extended, x, instance_id, band_cutoffs), unseen


# --------------------------------------------------------------------------------------
# Step 5: structured representation report
# --------------------------------------------------------------------------------------

def _md_table(df, floatfmt=".3f", formats=None):
    """Markdown table; `formats` maps column name -> format spec, overriding `floatfmt`."""
    formats = formats or {}

    def fmt(col, v):
        if isinstance(v, (float, np.floating)):
            return format(v, formats.get(col, floatfmt))
        return str(v)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def _unseen_section(df, arrival_reports):
    """Section 4 lines on features that no training individual reported."""
    has_unseen = df["n_unseen"] > 0
    names = sorted({f for r in arrival_reports for f in r.unseen_features})
    out = ["### Features the system has never seen", ""]
    if not names:
        return out + ["No arrival reported a feature outside the pooled population.", ""]
    out += [f"- **{has_unseen.sum()} of {len(df)}** arrivals ({has_unseen.mean():.0%}) report at least one "
            f"feature no training individual reported; **{len(names)}** distinct features, "
            f"{df['n_unseen'].mean():.1f} per arrival on average (max {df['n_unseen'].max()}).",
            "- The deployed system cannot represent them: each counts as reported but not kept, which "
            "lowers the fit score. In Pr_i and Rare(d_i) each counts as held by that arrival alone, "
            "and the model ignores them.",
            "- They are candidates for the next recalibration, which would add them to the feature "
            "space; until then the system is unchanged.", ""]
    by_unseen = (df.assign(unseen=df["n_unseen"].clip(upper=3).map(lambda k: "3+" if k == 3 else str(k)))
                 .groupby("unseen")
                 .agg(n=("fit_score", "size"), fit_mean=("fit_score", "mean"),
                      fit_percentile_median=("fit_percentile", "median"),
                      **{b: ("band", lambda v, b=b: int((v == b).sum())) for b in BANDS})
                 .reset_index())
    out += ["Fit by number of never-seen features reported:", "",
            _md_table(by_unseen, formats={"fit_percentile_median": ".1f"}), ""]
    counts = pd.Series([f for r in arrival_reports for f in r.unseen_features]).value_counts()
    shared = counts[counts > 1]
    if len(shared):
        out += [f"Never-seen features reported by more than one arrival ({len(shared)}): "
                + ", ".join(f"{f} ({k})" for f, k in shared.head(10).items())
                + (", ..." if len(shared) > 10 else ""), ""]
    out += ["Pr_i for arrivals with never-seen features is computed over the extended feature space, "
            "so its percentile is not on exactly the same scale as the pooled population's.", ""]
    return out


def representation_report(summary, system, calibration, min_coverage, feature_kind=None,
                          arrivals=None, arrival_reports=None, figures=None):
    """Assemble the structured representation report as Markdown."""
    ref = system.reference
    cal = calibration.summary
    chosen = cal[cal["tau"] == system.tau].iloc[0]
    groups = sorted(ref["group"].unique())
    figures = figures or {}
    out = ["# Representation report", ""]

    out += ["## 1. Pooled population (Phase I)", ""]
    out += [f"- **{k}**: {v}" for k, v in summary.items()]
    out += [""]
    if "population" in figures:
        out += [f"![Feature prevalence]({figures['population']})", ""]
    by_group = ref.groupby("group").agg(
        n=("n_features", "size"),
        features_median=("n_features", "median"),
        log10_pr_i_median=("log10_pr_i", "median"),
        log10_pr_i_norm_median=("pr_i_normalized", lambda v: np.log10(np.median(v))),
        j_pop_mean=("j_pop", "mean"),
        rarity_median=("rarity", "median"),
        unique_share=("unique_vector", "mean"),
    ).reset_index()
    out += ["Heterogeneity measures by subgroup:", "", _md_table(by_group, formats={"features_median": ".0f", "log10_pr_i_median": ".1f",
                                                       "log10_pr_i_norm_median": ".1f", "rarity_median": ".1f",
                                                       "unique_share": ".0%"}), ""]
    if "measures" in figures:
        out += [f"![Measures by group]({figures['measures']})", ""]

    out += ["## 2. Feature selection & calibration (Phase II)", ""]
    out += [f"The prevalence threshold was swept with 5-fold cross-validation. The chosen threshold is "
            f"**τ = {system.tau:g}**: it has the highest held-out accuracy among thresholds with "
            f"J_system ≥ {min_coverage:g}.", ""]
    cols = ["tau", "n_selected", "j_system", "j_worst_group", "accuracy", "accuracy_sd"]
    out += [_md_table(cal[cols], formats={"tau": "g", "n_selected": ".0f"}), ""]
    if "calibration" in figures:
        out += [f"![Calibration]({figures['calibration']})", ""]

    out += ["## 3. What the deployed system represents", ""]
    out += [f"- Features used: **{int(system.selected.sum())}** of {len(system.selected)}"]
    if feature_kind is not None:
        for kind in pd.unique(feature_kind):
            m = feature_kind == kind
            out += [f"  - {kind}: {int((system.selected & m).sum())} of {int(m.sum())}"]
    out += [f"- J_system (held-out): **{chosen['j_system']:.3f}**; 10% of individuals are at or below "
            f"{chosen['j_p10']:.3f}",
            f"- Held-out accuracy: **{chosen['accuracy']:.3f}** ± {chosen['accuracy_sd']:.3f}", ""]
    per_group = pd.DataFrame({
        "group": groups,
        "share": [np.mean(ref["group"] == g) for g in groups],
        "j_sys": [chosen[f"j_{g}"] for g in groups],
        "accuracy": [chosen[f"acc_{g}"] for g in groups],
    })
    out += ["Coverage and accuracy by subgroup at the chosen τ (out-of-fold):", "",
            _md_table(per_group), ""]
    worst = per_group.loc[per_group["j_sys"].idxmin()]
    best = per_group.loc[per_group["j_sys"].idxmax()]
    out += [f"Coverage gap: group {worst['group']} keeps {worst['j_sys']:.0%} of its reported features "
            f"on average, compared with {best['j_sys']:.0%} for group {best['group']}.", ""]

    if arrival_reports:
        out += ["## 4. New arrivals (Phase III)", ""]
        df = pd.DataFrame([r.to_dict() for r in arrival_reports])
        if arrivals is not None:
            df["group"] = arrivals.group
            df["correct"] = df["prediction"].to_numpy() == arrivals.y
            out += [f"- {len(df)} new individuals scored; accuracy on them: "
                    f"**{df['correct'].mean():.3f}**", ""]
            bands = pd.crosstab(df["group"], df["band"]).reindex(columns=list(BANDS), fill_value=0)
            out += ["Fit bands by subgroup:", "", _md_table(bands.reset_index()), ""]
        out += _unseen_section(df, arrival_reports)
        lowest = df.nsmallest(5, "fit_score")
        out += ["The 5 lowest-fit arrivals:", ""]
        out += [r.to_markdown() + "\n" for r in arrival_reports if r.instance_id in set(lowest["instance_id"])]
        if "fit" in figures:
            out += [f"![Fit scores]({figures['fit']})", ""]

    return "\n".join(out)
