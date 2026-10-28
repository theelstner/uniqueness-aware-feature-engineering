"""Run the full uniqueness-aware feature-engineering pipeline on synthetic data.

    .venv/bin/python run_prototype.py --n 400 --n-new 100 --seed 7 --min-coverage 0.6

Writes to outputs/: report.md, figures, and CSVs (reference measures, calibration
sweep, fit reports for new arrivals).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uafe import plots
from uafe.data import arrival_feature_names, generate_population
from uafe.pipeline import (DEFAULT_THRESHOLDS, assess_population, build_system, calibrate,
                           choose_threshold, fit_instance_from_names, population_summary,
                           representation_report)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=400, help="pooled population size")
    p.add_argument("--n-new", type=int, default=100, help="new arrivals scored in Phase III")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-coverage", type=float, default=0.6, help="J_system floor for choosing τ")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Phase I. Step 1: reference representations (+ held-back new arrivals from the same distribution)
    full = generate_population(n=args.n + args.n_new, seed=args.seed)
    reference = full.subset(np.arange(args.n))
    arrivals = full.subset(np.arange(args.n, args.n + args.n_new))
    # as in practice, arrivals also report features no training individual reported
    arrival_names = arrival_feature_names(arrivals)

    # Step 2: heterogeneity measures
    print("Phase I: computing Pr_i, j_pop, Rare(d_i) ...")
    assessment = assess_population(reference.X, reference.group)
    summary = population_summary(reference.X, assessment)
    assessment.assign(individual=np.arange(args.n)).to_csv(args.out / "reference_measures.csv", index=False)

    # Phase II. Steps 3-4: prevalence-threshold sweep, then choose τ
    print("Phase II: calibrating the prevalence threshold ...")
    calibration = calibrate(reference.X, reference.y, reference.group,
                            thresholds=DEFAULT_THRESHOLDS, n_splits=args.folds, seed=args.seed)
    tau = choose_threshold(calibration.summary, min_coverage=args.min_coverage)
    calibration.summary.to_csv(args.out / "calibration.csv", index=False)

    # Phase III. Step 5 needs the deployed system, step 6 scores each new arrival
    print(f"Phase III: deploying at τ = {tau:g} and scoring {args.n_new} new arrivals ...")
    system = build_system(reference, assessment, calibration, tau)
    reports = [fit_instance_from_names(system, names, instance_id=f"new_{i:03d}")[0]
               for i, names in enumerate(arrival_names)]
    arrival_df = pd.DataFrame([r.to_dict() for r in reports]).assign(group=arrivals.group, y=arrivals.y)
    arrival_df.to_csv(args.out / "new_arrival_fit_reports.csv", index=False)

    groups = sorted(set(reference.group))
    figures = {"population": "fig_prevalence.png", "measures": "fig_measures_by_group.png",
               "calibration": "fig_calibration.png", "fit": "fig_fit_scores.png"}
    plots.plot_prevalence(reference.X, reference.feature_kind, args.out / figures["population"], tau=tau)
    plots.plot_measures_by_group(assessment, args.out / figures["measures"])
    plots.plot_calibration(calibration.summary, tau, groups, args.out / figures["calibration"],
                           min_coverage=args.min_coverage)
    plots.plot_fit_scores(system.reference["j_sys"], arrival_df, groups, args.out / figures["fit"])

    report = representation_report(summary, system, calibration, args.min_coverage,
                                   feature_kind=reference.feature_kind, arrivals=arrivals,
                                   arrival_reports=reports, figures=figures)
    (args.out / "report.md").write_text(report)

    chosen = calibration.summary.set_index("tau").loc[tau]
    print(f"\nτ = {tau:g}: {int(system.selected.sum())} features, J_system = {chosen['j_system']:.3f}, "
          f"held-out accuracy = {chosen['accuracy']:.3f}")
    print("New arrivals by band:", arrival_df["band"].value_counts().to_dict())
    print(f"New arrivals with never-seen features: {(arrival_df['n_unseen'] > 0).mean():.0%}")
    print(f"Wrote {args.out.resolve()}/report.md")


if __name__ == "__main__":
    main()
