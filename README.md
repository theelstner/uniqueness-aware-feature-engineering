# Uniqueness-aware feature engineering: prototype

A prototype of the six-step pipeline, run on synthetic data.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_prototype.py            # writes outputs/report.md, figures and CSVs
.venv/bin/python -m pytest -q                # includes verbatim checks against the draft notebook
```

Or open `prototype.ipynb` with the `.venv` kernel to walk through the steps.

| Phase | Step | Code |
|---|---|---|
| I | 1. Reference representations d_i | `uafe.data.generate_population` |
| I | 2. Pr_i, j_pop, Rare(d_i) | `uafe.pipeline.assess_population` |
| II | 3. Prevalence-based selection | `uafe.pipeline.select_features` |
| II | 4. Coverage–accuracy calibration | `uafe.pipeline.calibrate`, `choose_threshold` |
| III | 5. Representation report | `uafe.pipeline.representation_report` |
| III | 6. Per-individual fit score | `uafe.pipeline.fit_instance`, `fit_instance_from_names` |

## Reused from the draft notebook

`compute_pr_i`, `compute_log_factorial_sum`, `compute_normalized_pr_i`, `compute_similarity`,
`generate_distribution` and `generate_mixed_sparsity_distribution` are copied unchanged into
`uafe/measures.py` and `uafe/data.py`. `tests/test_measures.py` fails if they drift from the
notebook. The rarity logic from the notebook's MAIN cell becomes `compute_rarity`. Two vectorised
or log-space wrappers (`jaccard_matrix`/`compute_system_j` and `compute_log10_pr_i`) are tested
against the originals.

## Modelling choices to check against the paper

- **Synthetic heterogeneity.** `generate_distribution` gives every feature the same expected
  prevalence, so stacking blocks is what creates the long tail. The blocks are four shared tiers
  (sparsity 0.6 → 0.04), 30 features per subgroup that are common only within that subgroup
  (groups A–D are 60/25/10/5 % of the population), and 300 idiosyncratic features that one or two
  people each hold.
- **Outcome y.** A sparse linear score over shared and subgroup features (idiosyncratic features
  carry no signal) plus noise, split at the median.
- **Two Jaccard quantities.** In Phase I, `j_pop` is the mean J(d_i, d_j) against everyone else.
  In Phases II and III, `j_sys` is J(d_i, r_i), where r_i = d_i restricted to the selected
  features. Because r_i ⊆ d_i, this equals the share of i's features the system keeps.
  J_system is the mean of j_sys over individuals.
- **Selection** uses only training folds. Features never seen in training are excluded even at
  τ = 0. J_system and accuracy are both measured out-of-fold.
- **Choosing τ.** The default picks the highest accuracy among thresholds with J_system ≥
  `--min-coverage` (0.6).
- **Fit score** for a new x is j_sys. Its band comes from the percentile within the pooled
  population's out-of-fold j_sys: below P5 is poor, below P25 is partial, otherwise well
  represented. Pr_i and Rare are computed with x added to the pool, matching how training
  individuals are measured, which keeps Pr_i defined when x has unseen features.
