# Changelog

All notable changes to qis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Python 3.14 support.
- `qis.delever_returns`, `qis.lever_returns`, `qis.implied_leverage` in
  `qis.perfstats.returns` for working with levered / unlevered return
  series given leverage and financing rate.
- `qis.unsmooth_returns_ar1_ewma`, `qis.unsmooth_returns_glm`, and
  `qis.compute_ar1_unsmoothed_prices` in `qis.perfstats.unsmoothing` for
  AR(1) EWMA and AR(q) Getmansky-Lo-Makarov unsmoothing of appraisal-based
  NAV series, with severity diagnostics.
- `qis.to_quarterly_returns` in `qis.perfstats.returns` for compounding
  daily / weekly / monthly returns to quarter-end with partial-quarter
  masking.
- Vectorised `qis.compute_risk_table`.
- Reorganised `qis/examples/` into themed sub-packages (`perfstats/`,
  `models/`, `regimes/`, `portfolios/`, `factsheets/`, `plots/`, `utils/`,
  `case_studies/`, `_helpers/`) with a per-folder `README.md` and a
  module-level docstring on every example file.
- New example `qis/examples/perfstats/unsmoothing_and_delevering.py` —
  end-to-end walkthrough of the leverage / unsmoothing functions on a
  bundled OCSL / Oaktree GCF / SPX / US HY / US Agg weekly NAV dataset.
- New example `qis/examples/models/multivariate_ols.py` demonstrating
  `qis.fit_multivariate_ols` directly (separated from the EWM linear-model
  example).
- `bbg-fetch >=2.0.0` listed as optional dependency for examples that
  pull data from a Bloomberg terminal.

### Changed
- Bumped minimum Python from 3.9 to 3.10. (numba 0.61 dropped Python 3.9
  support, and the bump to numba ≥0.63 for Python 3.14 forces the same
  floor here.)
- Bumped minimum numba from 0.60.0 to 0.63.0 (required for Python 3.14
  support; see numba 0.63.0 release notes, Dec 2025).
- Renamed several example files for clarity:
  - `models/ewm_filters.py` → `models/ewm_kernels.py`
  - `models/correlation_matrix.py` → `models/ewm_correlation_table.py`
  - `models/ewma_factor_betas.py` → `models/ewm_linear_model.py`
  - `portfolios/btc_marginal_contribution.py` → `portfolios/balanced_60_40_with_btc.py`
  - `perfstats/perf_excluding_best_worst_days.py` → `perfstats/miss_best_worst_days_impact.py`
- Moved `infrequent_returns_interpolation.py` from `examples/utils/` to
  `examples/perfstats/` (matches the API location:
  `qis.perfstats.timeseries_bfill`).

### Fixed
- `qis.to_quarterly_returns` calendar-QE boundary bug. The previous
  implementation used `returns.reindex(q_returns.index).notna()` to detect
  partial trailing quarters, which silently masked the entire output for
  any input whose timestamps did not land on calendar quarter-end dates
  (W-FRI weekly, business-month-end series). The new implementation uses
  a calendar-month coverage check per column: a quarter ending at QE is
  complete iff the input's last non-NaN observation falls in the same
  calendar month as QE.

## [2.0.1] - 2023-07-08

### Removed
- `qis.portfolio.optimisation` layer, with core functionality moved to a
  stand-alone Python package
  [bop (Backtesting Optimal Portfolio)](https://pypi.org/project/bop/).
  Removes the cvxpy and sklearn dependencies.

### Added
- Factsheet reporting via [pybloqs](https://github.com/man-group/PyBloqs).
- Four factsheet types with examples in `qis.examples.factsheets`:
  - `multi_asset` — cross-sectional comparison
  - `strategy` — performance / risk / trading stats from `PortfolioData`
  - `strategy_benchmark` — strategy vs benchmark
  - `multi_strategy` — parameter sensitivity sweeps

## [1.0.1] - 2022-12-30

Initial public release.

---

Versions between 1.0.1 ↔ 2.0.0 and 2.0.2 onwards (prior to the next
release) have not been backfilled. Run `git log --tags --oneline` for
release-by-release commit history.
