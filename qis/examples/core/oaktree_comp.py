"""
Comparative Risk-Return Analysis: OCSL (BDC) vs Oaktree Global Credit Fund
===========================================================================
Requested by Mika Kastenholz, 15-Apr-2026
Prepared by ISQ (Artur Sepp)

Instruments:
  - OCSL US Equity:    Oaktree Specialty Lending Corp (listed BDC, ~1.07x levered, 6.1% wtd avg debt rate)
  - OTGCADU LX Equity: Oaktree Global Credit Fund (unlisted, unlevered, daily NAV)

Benchmarks:
  - SPX Index:         S&P 500 (equity beta reference)
  - LF98TRUU Index:    Bloomberg US HY Corporate Bond Index (credit beta reference)
  - LBUSTRUU Index:    Bloomberg US Agg Bond Index (duration reference)

Dependencies:
  pip install qis bbg-fetch pandas numpy statsmodels matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

import qis
from qis.perfstats.unsmoothing import unsmooth_returns_glm
from qis.perfstats.returns import delever_returns
from bbg_fetch import fetch_field_timeseries_per_tickers


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

class AnalysisConfig:
    """Central config for the analysis."""

    # Bloomberg tickers
    OCSL_TICKER = 'OCSL US Equity'
    GCF_TICKER = 'OTGCADU LX Equity'

    # Benchmark tickers
    BENCHMARKS = {
        'SPX': 'SPX Index',
        'US HY': 'LF98TRUU Index',
        'US Agg': 'LBUSTRUU Index',
    }

    # Bloomberg fields
    # All tickers use PX_LAST with CshAdjNormal/CshAdjAbnormal:
    #   - OCSL + benchmarks: captures dividends (validated within <5 bps of TRI)
    #   - OTGCADU LX: accumulating SICAV share class, PX_LAST = NAV = total return
    #     (NET_ASSET_VAL field is not available via blpapi for this fund)

    # Date range — common window starts at OTGCADU LX inception
    START_DATE = '2018-11-08'  # Oaktree Global Credit Fund inception
    END_DATE = '2026-04-14'

    # Extended OCSL window for standalone analysis
    OCSL_EXTENDED_START = '2012-10-01'  # OCSL IPO era

    # Analysis frequency
    FREQ = 'W-FRI'  # weekly to reduce noise, align to Friday close

    # OCSL leverage and financing cost (from OCSL Q1 2026 investor presentation):
    # - Net Debt/Equity ratio:                 1.07x (target range 0.90–1.25x)
    # - Weighted average interest rate on debt: 6.1% (inclusive of interest rate swaps)
    OCSL_LEVERAGE = 1.07
    OCSL_FINANCING_RATE = 0.061

    # Risk-free proxy (annualized, approximate current SOFR) — used elsewhere
    RF_RATE = 0.045

    # NAV unsmoothing AR lags
    UNSMOOTH_LAGS = 3

    # Output directory
    OUTPUT_DIR = Path('./output')


# =============================================================================
# 1. DATA FETCHING
# =============================================================================

def fetch_bloomberg_data(config: AnalysisConfig) -> pd.DataFrame:
    """
    Fetch all required time series from Bloomberg via bbg_fetch.

    All tickers use PX_LAST with CshAdjNormal=True, CshAdjAbnormal=True:
      - OCSL: listed BDC, cash adjustment captures ~10% dividend yield
      - OTGCADU LX: accumulating SICAV, PX_LAST = NAV (no distributions)
      - Benchmarks: total-return equivalent via cash adjustment

    Returns a single DataFrame with all price series.
    """
    all_tickers = (
        [config.OCSL_TICKER, config.GCF_TICKER]
        + list(config.BENCHMARKS.values())
    )

    print(f"Fetching PX_LAST (cash-adjusted) for {len(all_tickers)} tickers...")
    prices = fetch_field_timeseries_per_tickers(
        tickers=all_tickers,
        field='PX_LAST',
        CshAdjNormal=True,
        CshAdjAbnormal=True,
        start_date=config.OCSL_EXTENDED_START,
        end_date=config.END_DATE,
    )

    # rename to short names
    rename_map = {
        config.OCSL_TICKER: 'OCSL',
        config.GCF_TICKER: 'Oaktree GCF',
    }
    rename_map.update({v: k for k, v in config.BENCHMARKS.items()})
    prices = prices.rename(columns=rename_map)

    return prices


def build_price_panel(
    prices: pd.DataFrame,
    config: AnalysisConfig,
    use_common_window: bool = True,
) -> pd.DataFrame:
    """
    Align, resample to weekly, and normalize price panel to 100.
    """
    panel = prices.copy()

    # ensure numeric (blpapi can return object columns)
    panel = panel.apply(pd.to_numeric, errors='coerce')

    if use_common_window:
        panel = panel.loc[config.START_DATE:config.END_DATE]

    # forward-fill daily gaps before resampling
    panel = panel.ffill()

    # trim to first date where all columns have data
    panel = panel.dropna()

    print(f"  Panel: {panel.shape[0]} obs, {panel.index[0].date()} to {panel.index[-1].date()}")

    if panel.empty:
        for col in prices.columns:
            print(f"  first valid '{col}': {prices[col].first_valid_index()}")
        raise ValueError("Price panel is empty after alignment")

    # resample to weekly (Friday close)
    panel = panel.resample(config.FREQ).last().dropna()

    # normalize to 100 at start
    panel = 100.0 * panel / panel.iloc[0]

    return panel


# =============================================================================
# 2. NAV UNSMOOTHING & LEVERAGE ADJUSTMENT (imported from qis)
# =============================================================================
# ``unsmooth_returns_glm`` — Getmansky-Lo-Makarov AR(q) unsmoothing for the
#   appraisal-based Global Credit Fund NAV.
# ``delever_returns``      — inverts the constant-leverage identity to recover
#   unlevered asset returns from observed equity returns.
# Both now live in ``qis.perfstats`` and are imported at the top of the module.


# =============================================================================
# 3. QIS TEARSHEET GENERATION
# =============================================================================

def generate_main_tearsheet(
    prices: pd.DataFrame,
    benchmark: str = 'US HY',
    config: AnalysisConfig = AnalysisConfig(),
) -> plt.Figure:
    """
    Generate the standard ISQ multi-asset factsheet using qis.
    """
    perf_params = qis.PerfParams(freq=config.FREQ)
    time_period = qis.TimePeriod(config.START_DATE, config.END_DATE)

    fig = qis.generate_multi_asset_factsheet(
        prices=prices,
        benchmark=benchmark,
        perf_params=perf_params,
        time_period=time_period,
    )
    return fig


def generate_performance_table(
    prices: pd.DataFrame,
    config: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    """Compute risk-adjusted performance metrics table."""
    perf_params = qis.PerfParams(freq=config.FREQ)
    ra_table = qis.compute_ra_perf_table(
        prices=prices,
        perf_params=perf_params,
    )
    return ra_table


# =============================================================================
# 4. FACTOR REGRESSION
# =============================================================================

def run_factor_decomposition(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Regress each strategy on credit market factors.

    Factors:
      - US HY:  high yield credit beta (primary driver for both)
      - SPX:    equity market beta (OCSL should load higher — listed vehicle)
      - US Agg: duration/rates beta

    Returns DataFrame with alpha, betas, R², residual vol per strategy.
    """
    import statsmodels.api as sm

    results = {}
    X = sm.add_constant(factor_returns.dropna())

    for col in returns.columns:
        y = returns[col].dropna()
        common_idx = X.index.intersection(y.index)
        if len(common_idx) < 52:  # need at least 1 year of data
            print(f"  Skipping {col}: insufficient overlapping data ({len(common_idx)} obs)")
            continue

        model = sm.OLS(y.loc[common_idx], X.loc[common_idx]).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

        results[col] = {
            'Alpha (ann. %)': model.params['const'] * 52 * 100,
            'Alpha t-stat': model.tvalues['const'],
            **{f'Beta ({k})': model.params[k] for k in factor_returns.columns},
            **{f't-stat ({k})': model.tvalues[k] for k in factor_returns.columns},
            'R²': model.rsquared,
            'Adj R²': model.rsquared_adj,
            'Residual Vol (ann. %)': model.resid.std() * np.sqrt(52) * 100,
        }

    return pd.DataFrame(results).T


# =============================================================================
# 5. REGIME-CONDITIONAL ANALYSIS
# =============================================================================

def compute_regime_analysis(
    prices: pd.DataFrame,
    vix_prices: pd.Series,
    config: AnalysisConfig,
    n_quantiles: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regime-conditional performance using qis quantile-based classification
    on VIX percentage returns (not absolute level).

    VIX % returns capture vol-of-vol dynamics:
      - Large positive VIX returns = vol spike = stress
      - Large negative VIX returns = vol compression = risk-on
      - Neutral = normal

    Uses qis.BenchmarkReturnsQuantileRegimeSpecs for quantile classification.
    """
    # VIX % returns as regime indicator (not absolute level)
    vix_returns = vix_prices.pct_change().dropna()
    vix_returns.name = 'VIX Returns'

    # resample to match panel frequency
    vix_weekly = vix_returns.resample(config.FREQ).sum()  # sum of daily log-ish returns within week

    # align with price panel
    common_idx = prices.index.intersection(vix_weekly.index)
    prices_aligned = prices.loc[common_idx]
    vix_aligned = vix_weekly.loc[common_idx]

    # qis regime classification via quantiles of VIX returns
    regime_params = qis.BenchmarkReturnsQuantileRegimeSpecs(freq=config.FREQ)

    # compute regime-conditional performance table
    ra_perf_table = qis.compute_ra_perf_table(
        prices=prices_aligned,
        perf_params=qis.PerfParams(freq=config.FREQ),
        regime_benchmark_str=vix_aligned.name,
        regime_params=regime_params,
    )

    # also produce the regime classification for downstream use
    regime_classifier = qis.compute_regime_classifier_for_regime_shifts(
        regime_benchmark=vix_aligned,
        freq=config.FREQ,
        regime_params=regime_params,
    )

    return ra_perf_table, regime_classifier


# =============================================================================
# 6. DRAWDOWN ANALYSIS
# =============================================================================

def compute_drawdown_table(
    prices: pd.DataFrame,
    n_worst: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    For each instrument, compute the N worst drawdown episodes.
    Useful for Mika's point about different drawdown behavior.
    """
    dd_tables = {}
    for col in prices.columns:
        dd_series = prices[col] / prices[col].cummax() - 1.0
        # find the worst N drawdown troughs
        dd_min = dd_series.expanding().min()
        # get unique drawdown episodes
        episodes = []
        remaining = dd_series.copy()
        for _ in range(n_worst):
            trough_date = remaining.idxmin()
            trough_val = remaining.loc[trough_date]
            if trough_val >= 0:
                break
            # find peak before trough
            peak_date = prices[col].loc[:trough_date].idxmax()
            # find recovery after trough
            post_trough = prices[col].loc[trough_date:]
            peak_val = prices[col].loc[peak_date]
            recovery_mask = post_trough >= peak_val
            recovery_date = recovery_mask.idxmax() if recovery_mask.any() else None

            episodes.append({
                'Peak Date': peak_date,
                'Trough Date': trough_date,
                'Recovery Date': recovery_date,
                'Max DD (%)': trough_val * 100,
                'Days to Trough': (trough_date - peak_date).days,
                'Days to Recovery': (recovery_date - trough_date).days if recovery_date else None,
            })
            # mask out this episode to find the next one
            if recovery_date:
                remaining.loc[peak_date:recovery_date] = 0.0
            else:
                remaining.loc[peak_date:] = 0.0

        dd_tables[col] = pd.DataFrame(episodes)

    return dd_tables


# =============================================================================
# 7. DISCOUNT/PREMIUM ANALYSIS (OCSL-specific)
# =============================================================================

def fetch_ocsl_nav_and_discount(config: AnalysisConfig) -> pd.DataFrame:
    """
    Fetch OCSL's reported NAV per share alongside market price
    to compute the premium/discount dynamics.

    BDC discount behavior is a key structural difference:
    - In normal markets, BDCs trade at slight discounts to NAV
    - In stress, discounts widen dramatically (equity-like drawdown)
    - In euphoria, can trade at premium

    This is the main reason OCSL's return profile looks more equity-like
    than the underlying credit portfolio would suggest.
    """
    # raw price (no cash adjustment) — we need the actual market price vs NAV
    price = fetch_field_timeseries_per_tickers(
        tickers=[config.OCSL_TICKER],
        field='PX_LAST',
        CshAdjNormal=False,
        CshAdjAbnormal=False,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    ).rename(columns={config.OCSL_TICKER: 'Price'})

    # BDC NAV: try NET_ASSET_VAL first, fall back to BOOK_VAL_PER_SH
    nav = None
    for nav_field in ['NET_ASSET_VAL', 'BOOK_VAL_PER_SH']:
        try:
            candidate = fetch_field_timeseries_per_tickers(
                tickers=[config.OCSL_TICKER],
                field=nav_field,
                start_date=config.START_DATE,
                end_date=config.END_DATE,
            ).rename(columns={config.OCSL_TICKER: 'NAV'})
            if not candidate.empty and candidate['NAV'].notna().sum() > 0:
                print(f"  OCSL NAV sourced from {nav_field}")
                nav = candidate
                break
        except Exception:
            continue

    if nav is None or nav.empty:
        raise ValueError("Could not fetch OCSL NAV from any field")

    combined = pd.concat([price, nav], axis=1).dropna()
    combined['Discount (%)'] = ((combined['Price'] / combined['NAV']) - 1.0) * 100

    return combined


# =============================================================================
# 8. CORRELATION ANALYSIS
# =============================================================================

def compute_rolling_correlations(
    prices: pd.DataFrame,
    benchmark_col: str = 'US HY',
    window: int = 52,  # 1-year rolling
) -> pd.DataFrame:
    """
    Rolling correlation of each strategy with the HY benchmark.
    Hypothesis: OCSL correlation with HY spikes in stress (convergence),
    while Global Credit maintains steadier correlation.
    """
    returns = prices.pct_change().dropna()
    bench_ret = returns[benchmark_col]

    rolling_corrs = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        if col == benchmark_col:
            continue
        rolling_corrs[col] = returns[col].rolling(window).corr(bench_ret)

    return rolling_corrs.dropna()


# =============================================================================
# 9. MAIN ORCHESTRATOR
# =============================================================================

def run_full_analysis(
    config: Optional[AnalysisConfig] = None,
    use_cached_data: bool = False,
    cache_path: str = './data_cache.parquet',
) -> dict:
    """
    Full analysis pipeline.

    Steps:
      1. Fetch data from Bloomberg (or load cache)
      2. Build price panel, normalize
      3. Generate raw tearsheet (OCSL vs GCF vs benchmarks)
      4. Unsmooth Global Credit NAV returns
      5. De-lever OCSL returns
      6. Generate adjusted tearsheet
      7. Factor decomposition
      8. Regime analysis
      9. Drawdown comparison
     10. OCSL discount/premium dynamics
     11. Rolling correlations
     12. Save all outputs

    Returns dict of all computed artifacts.
    """
    if config is None:
        config = AnalysisConfig()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Data
    # ------------------------------------------------------------------
    if use_cached_data and Path(cache_path).exists():
        print(f"Loading cached data from {cache_path}")
        panel = pd.read_parquet(cache_path)
    else:
        print("=" * 60)
        print("STEP 1: Fetching Bloomberg data")
        print("=" * 60)
        data = fetch_bloomberg_data(config)

        panel = build_price_panel(data, config, use_common_window=True)
        panel.to_parquet(cache_path)
        print(f"  Cached to {cache_path}")

    print(f"\nPrice panel shape: {panel.shape}")
    print(f"Date range: {panel.index[0].date()} to {panel.index[-1].date()}")
    print(f"Columns: {list(panel.columns)}")

    results = {'config': config, 'prices': panel}

    # ------------------------------------------------------------------
    # Step 2: Raw performance table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Raw performance metrics")
    print("=" * 60)
    perf_raw = generate_performance_table(panel, config)
    print(perf_raw.to_string())
    results['perf_raw'] = perf_raw

    # ------------------------------------------------------------------
    # Step 3: Raw tearsheet
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Generating raw tearsheet")
    print("=" * 60)
    fig_raw = generate_main_tearsheet(
        prices=panel,
        benchmark='US HY',
        config=config,
    )
    fig_raw.savefig(config.OUTPUT_DIR / 'tearsheet_raw.pdf', bbox_inches='tight', dpi=150)
    print(f"  Saved to {config.OUTPUT_DIR / 'tearsheet_raw.pdf'}")
    results['fig_raw'] = fig_raw

    # ------------------------------------------------------------------
    # Step 4: Unsmoothing + de-levering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: NAV unsmoothing & leverage adjustment")
    print("=" * 60)

    returns_raw = panel.pct_change().dropna()

    # 4a. Unsmooth Global Credit Fund returns via qis GLM method
    print("\nUnsmoothing Oaktree Global Credit Fund:")
    gcf_unsmoothed, gcf_diag = unsmooth_returns_glm(
        returns=returns_raw['Oaktree GCF'],
        ar_order=config.UNSMOOTH_LAGS,
        return_diagnostics=True,
    )
    print(f"  AR({gcf_diag.ar_order}) coefficients: {np.round(gcf_diag.theta, 4)}")
    print(f"  Smoothing weight sum: {gcf_diag.theta_sum:.4f}")
    print(f"  Vol inflation factor: {gcf_diag.vol_inflation_factor:.2f}x")
    if gcf_diag.is_severe:
        print(f"  WARNING: severe smoothing detected")

    # 4b. De-lever OCSL returns via qis helper.
    # Uses OCSL's actual weighted average debt rate (6.1% from Q1 2026 presentation)
    # rather than the risk-free proxy. A higher financing cost implies a higher
    # implied unlevered asset return, since dr_A/dr_F = L/(1+L) > 0.
    print("\nDe-levering OCSL (leverage = {:.2f}x, financing = {:.1%}):".format(
        config.OCSL_LEVERAGE, config.OCSL_FINANCING_RATE,
    ))
    ocsl_delevered = delever_returns(
        returns=returns_raw['OCSL'],
        leverage=config.OCSL_LEVERAGE,
        financing_rate=config.OCSL_FINANCING_RATE,
    )

    # Build adjusted return series -> reconstruct prices
    adj_returns = returns_raw.copy()
    adj_returns['OCSL (unlev.)'] = ocsl_delevered
    adj_returns['GCF (unsmoothed)'] = gcf_unsmoothed

    # reconstruct adjusted price panel
    adj_prices = (1 + adj_returns).cumprod() * 100
    results['adj_returns'] = adj_returns
    results['adj_prices'] = adj_prices

    # vol comparison
    ann = np.sqrt(52)
    vol_comparison = pd.DataFrame({
        'Raw Vol (%)': returns_raw.std() * ann * 100,
        'OCSL Unlev. Vol (%)': pd.Series({'OCSL': (ocsl_delevered.std() * ann * 100)}),
        'GCF Unsmoothed Vol (%)': pd.Series({'Oaktree GCF': (gcf_unsmoothed.std() * ann * 100)}),
    })
    print("\nVolatility comparison:")
    print(vol_comparison.to_string())
    results['vol_comparison'] = vol_comparison

    # ------------------------------------------------------------------
    # Step 5: Adjusted tearsheet
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Adjusted tearsheet (unlevered OCSL + unsmoothed GCF)")
    print("=" * 60)

    adj_panel = adj_prices[['OCSL (unlev.)', 'GCF (unsmoothed)', 'US HY', 'SPX']].dropna()
    fig_adj = generate_main_tearsheet(
        prices=adj_panel,
        benchmark='US HY',
        config=config,
    )
    fig_adj.savefig(config.OUTPUT_DIR / 'tearsheet_adjusted.pdf', bbox_inches='tight', dpi=150)
    print(f"  Saved to {config.OUTPUT_DIR / 'tearsheet_adjusted.pdf'}")
    results['fig_adj'] = fig_adj

    # ------------------------------------------------------------------
    # Step 6: Factor decomposition
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Factor regression")
    print("=" * 60)

    strategy_returns = returns_raw[['OCSL', 'Oaktree GCF']].copy()
    strategy_returns['OCSL (unlev.)'] = ocsl_delevered
    strategy_returns['GCF (unsmoothed)'] = gcf_unsmoothed

    factor_returns = returns_raw[['US HY', 'SPX', 'US Agg']].copy()

    factor_table = run_factor_decomposition(strategy_returns, factor_returns)
    print(factor_table.round(3).to_string())
    results['factor_table'] = factor_table

    # ------------------------------------------------------------------
    # Step 7: Regime analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Regime-conditional analysis")
    print("=" * 60)

    # Fetch VIX for regime classification
    try:
        vix = fetch_field_timeseries_per_tickers(
            tickers=['VIX Index'],
            field='PX_LAST',
            start_date=config.START_DATE,
            end_date=config.END_DATE,
        ).rename(columns={'VIX Index': 'VIX'})
        vix_daily = vix['VIX'].ffill()

        ra_regime_table, regime_classifier = compute_regime_analysis(
            prices=panel[['OCSL', 'Oaktree GCF', 'US HY']],
            vix_prices=vix_daily,
            config=config,
        )
        print(ra_regime_table.round(3).to_string())
        results['regime_table'] = ra_regime_table
        results['regime_classifier'] = regime_classifier
    except Exception as e:
        print(f"  Regime analysis failed ({e}), skipping")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Step 8: Drawdown comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Drawdown episodes")
    print("=" * 60)

    dd_tables = compute_drawdown_table(
        prices=panel[['OCSL', 'Oaktree GCF', 'US HY']],
        n_worst=5,
    )
    for name, dd_df in dd_tables.items():
        print(f"\n  {name} — Worst drawdowns:")
        print(dd_df.to_string(index=False))
    results['drawdown_tables'] = dd_tables

    # ------------------------------------------------------------------
    # Step 9: OCSL discount/premium
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: OCSL NAV discount/premium dynamics")
    print("=" * 60)

    try:
        ocsl_disc = fetch_ocsl_nav_and_discount(config)
        print(f"  Current discount: {ocsl_disc['Discount (%)'].iloc[-1]:.1f}%")
        print(f"  Average discount: {ocsl_disc['Discount (%)'].mean():.1f}%")
        print(f"  Worst discount:   {ocsl_disc['Discount (%)'].min():.1f}%")
        print(f"  Best premium:     {ocsl_disc['Discount (%)'].max():.1f}%")

        fig_disc, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(ocsl_disc.index, ocsl_disc['Discount (%)'], 0, alpha=0.3,
                        color='steelblue', label='Discount/Premium')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(ocsl_disc['Discount (%)'].mean(), color='red', linestyle='--',
                    linewidth=0.8, label=f"Mean: {ocsl_disc['Discount (%)'].mean():.1f}%")
        ax.set_title('OCSL: Market Price vs NAV (Discount/Premium %)')
        ax.set_ylabel('Premium / Discount (%)')
        ax.legend()
        fig_disc.savefig(config.OUTPUT_DIR / 'ocsl_discount.pdf', bbox_inches='tight', dpi=150)
        results['ocsl_discount'] = ocsl_disc
        results['fig_discount'] = fig_disc
    except Exception as e:
        print(f"  OCSL NAV fetch failed ({e}), skipping discount analysis")

    # ------------------------------------------------------------------
    # Step 10: Rolling correlations
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10: Rolling correlations with US HY")
    print("=" * 60)

    rolling_corrs = compute_rolling_correlations(
        prices=panel,
        benchmark_col='US HY',
        window=52,
    )

    fig_corr, ax = plt.subplots(figsize=(12, 4))
    for col in ['OCSL', 'Oaktree GCF']:
        if col in rolling_corrs.columns:
            ax.plot(rolling_corrs.index, rolling_corrs[col], label=col, linewidth=1.2)
    ax.set_title('52-Week Rolling Correlation with US HY Index')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.5)
    fig_corr.savefig(config.OUTPUT_DIR / 'rolling_correlations.pdf', bbox_inches='tight', dpi=150)
    results['rolling_corrs'] = rolling_corrs
    results['fig_corr'] = fig_corr

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {config.OUTPUT_DIR.resolve()}")
    print("Files:")
    for f in sorted(config.OUTPUT_DIR.glob('*')):
        print(f"  {f.name}")

    return results


# =============================================================================
# 10. EXCEL SUMMARY EXPORT
# =============================================================================

def export_summary_to_excel(results: dict, path: str = './output/ocsl_analysis.xlsx'):
    """Export all tabular results to a single Excel workbook for Mika."""
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        if 'perf_raw' in results:
            results['perf_raw'].to_excel(writer, sheet_name='Perf Summary')

        if 'factor_table' in results:
            results['factor_table'].round(4).to_excel(writer, sheet_name='Factor Regression')

        if 'regime_table' in results:
            results['regime_table'].round(3).to_excel(writer, sheet_name='Regime Analysis')

        if 'vol_comparison' in results:
            results['vol_comparison'].round(2).to_excel(writer, sheet_name='Vol Comparison')

        if 'drawdown_tables' in results:
            row = 0
            for name, dd_df in results['drawdown_tables'].items():
                dd_df.to_excel(writer, sheet_name='Drawdowns', startrow=row, index=False)
                row += len(dd_df) + 3

        if 'ocsl_discount' in results:
            disc = results['ocsl_discount'].resample('ME').last()
            disc.to_excel(writer, sheet_name='OCSL Discount')

    print(f"Excel summary saved to {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    config = AnalysisConfig()
    results = run_full_analysis(config, use_cached_data=False)
    export_summary_to_excel(results)