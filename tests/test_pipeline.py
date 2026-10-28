import numpy as np
import pytest

from uafe.data import arrival_feature_names, generate_population
from uafe.pipeline import (BANDS, assess_population, build_system, calibrate, choose_threshold,
                           fit_instance, fit_instance_from_names, population_summary, representation_report,
                           select_features)


@pytest.fixture(scope="module")
def run():
    full = generate_population(n=150, groups={"A": 0.7, "B": 0.3}, n_idiosyncratic=60, seed=1)
    ref, new = full.subset(np.arange(120)), full.subset(np.arange(120, 150))
    assessment = assess_population(ref.X, ref.group)
    cal = calibrate(ref.X, ref.y, ref.group, thresholds=(0.0, 0.05, 0.2), n_splits=3)
    tau = choose_threshold(cal.summary, min_coverage=0.5)
    system = build_system(ref, assessment, cal, tau)
    return ref, new, assessment, cal, tau, system


def test_population_has_long_tail():
    pop = generate_population(n=200, seed=3)
    prevalence = pop.X.sum(axis=0) / len(pop)
    assert prevalence.max() > 0.4 and (prevalence <= 0.01).sum() > 50
    assert set(pop.y) == {0, 1}


def test_select_features_excludes_unseen_features():
    X = np.array([[1, 0, 1], [1, 0, 0]])
    assert select_features(X, 0.0).tolist() == [True, False, True]
    assert select_features(X, 0.6).tolist() == [True, False, False]


def test_coverage_decreases_with_threshold(run):
    _, _, _, cal, _, _ = run
    j = cal.summary["j_system"].to_numpy()
    assert np.all(np.diff(j) <= 1e-12)


def test_assessment_columns(run):
    _, _, assessment, _, _, _ = run
    assert {"log10_pr_i", "pr_i_normalized", "j_pop", "rarity"} <= set(assessment.columns)
    assert assessment["pr_i_normalized"].between(0, 1 + 1e-9).all()


def test_fit_instance_handles_unseen_feature(run):
    ref, _, _, _, _, system = run
    x = np.zeros(ref.X.shape[1], dtype=int)
    unseen = np.where(ref.X.sum(axis=0) == 0)[0]
    x[:3] = 1
    if len(unseen):
        x[unseen[0]] = 1
    report = fit_instance(system, x, "probe")
    assert report.band in BANDS
    assert 0 <= report.fit_score <= 1
    assert np.isfinite(report.log10_pr_i)


def test_fit_instance_from_names_matches_vector_and_extends(run):
    ref, _, _, _, _, system = run
    names = list(system.feature_names[np.flatnonzero(ref.X[0])])
    report, unseen = fit_instance_from_names(system, names, "same")
    assert unseen == []
    assert report.fit_score == fit_instance(system, ref.X[0], "same").fit_score

    new_report, unseen = fit_instance_from_names(system, names + ["brand_new"], "new")
    assert unseen == ["brand_new"]
    assert new_report.n_reported == report.n_reported + 1
    assert new_report.n_represented == report.n_represented
    assert new_report.fit_score < report.fit_score
    assert "brand_new" in new_report.omitted_features
    assert new_report.prediction == report.prediction
    assert len(system.feature_names) == ref.X.shape[1]  # deployed system unchanged


def test_arrival_feature_names_adds_brand_new_features(run):
    _, new, _, _, _, system = run
    names = arrival_feature_names(new, seed=0)
    assert len(names) == len(new)
    assert all(set(new.feature_names[x == 1]) <= set(n) for x, n in zip(new.X, names))
    brand_new = [[f for f in n if f not in set(system.feature_names)] for n in names]
    assert 0.5 < np.mean([len(b) > 0 for b in brand_new]) < 1.0


def test_report_covers_never_seen_features(run):
    ref, new, assessment, cal, _, system = run
    reports = [fit_instance_from_names(system, n, f"n{i}")[0]
               for i, n in enumerate(arrival_feature_names(new, seed=0))]
    assert any(r.unseen_features for r in reports)
    assert all(set(r.unseen_features) <= set(r.omitted_features) for r in reports)
    md = representation_report(population_summary(ref.X, assessment), system, cal, 0.5,
                               feature_kind=ref.feature_kind, arrivals=new, arrival_reports=reports)
    assert "Features the system has never seen" in md and "Fit by number of never-seen features" in md


def test_report_renders(run):
    ref, new, assessment, cal, tau, system = run
    reports = [fit_instance(system, x, f"n{i}") for i, x in enumerate(new.X)]
    md = representation_report(population_summary(ref.X, assessment), system, cal, 0.5,
                               feature_kind=ref.feature_kind, arrivals=new, arrival_reports=reports)
    assert "# Representation report" in md and "Fit report" in md
