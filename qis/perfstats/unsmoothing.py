"""
Return unsmoothing for illiquid / appraisal-based NAV series.

Provides two methodologies for recovering "true" returns from observed
(smoothed) NAV returns:

1. ``unsmooth_returns_ar1_ewma``: rolling EWMA AR(1) beta with clipping.
   Recommended for most use cases — adapts to regime changes, handles
   mixed-frequency panels, integrates with qis EWM primitives.

2. ``unsmooth_returns_glm``: static Getmansky-Lo-Makarov AR(q) fit.
   Classical methodology from the 2004 JFE paper. Useful for academic
   reproducibility or when a single smoothing parameter summary is needed.

Both methods assume the observed return is a weighted combination of
current and lagged true returns with weights summing to one:

    r_obs_t = theta_0 * r_true_t + theta_1 * r_true_{t-1} + ...

The rolling AR(1) approximation takes q=1 but allows theta to vary over time.
The GLM method allows q > 1 but assumes theta is constant over the sample.

Reference:
    Getmansky, M., Lo, A.W., and Makarov, I. (2004),
    "An Econometric Model of Serial Correlation and Illiquidity in Hedge Fund Returns,"
    Journal of Financial Economics, 74(3), 529-609.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from qis.utils.np_ops import set_nans_for_warmup_period
from qis.perfstats.returns import to_returns, returns_to_nav
from qis.models.linear.ewm import compute_ewm, MeanAdjType, compute_ewm_beta_alpha_forecast


# =============================================================================
# METHOD 1: ROLLING EWMA AR(1) UNSMOOTHING (preferred default)
# =============================================================================

def unsmooth_returns_ar1_ewma(returns: pd.DataFrame,
                              span: int = 20,
                              mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                              warmup_period: Optional[int] = 10,
                              max_value_for_beta: Optional[float] = 0.75,
                              min_value_for_beta: Optional[float] = -0.25,
                              apply_ewma_mean_smoother: bool = True
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Unsmooth NAV returns using a rolling EWMA AR(1) beta.

    Estimates a time-varying AR(1) coefficient beta_t on observed returns via
    EWMA regression, then inverts:

        r_true_t = (r_obs_t - beta_{t-1} * r_obs_{t-1}) / (1 - beta_{t-1})

    The beta is clipped symmetrically to prevent two failure modes:
      - Upper clip: denominator (1 - beta) approaching zero would blow up the
        unsmoothed series during regime shifts.
      - Lower clip: without a lower bound, sampling noise on a liquid series
        (true beta ≈ 0) produces a clipping asymmetry — negative EWMA estimates
        get truncated at zero while positive estimates pass through, biasing
        the mean beta upward and artificially inflating unsmoothed vol.

    Args:
        returns: Observed period returns (rows = dates, columns = assets).
        span: EWMA span for the rolling beta estimate. Default 20 periods.
            For quarterly data use 40 (10 years); for monthly use 24-36.
        mean_adj_type: How to demean returns before beta estimation.
        warmup_period: Number of initial periods to mask before first valid beta.
            Avoids excessive betas at the start of the sample.
        max_value_for_beta: Upper bound for beta clipping. Default 0.75 keeps
            the inversion denominator bounded away from zero. Pass None to
            disable upper clipping.
        min_value_for_beta: Lower bound for beta clipping. Default -0.25 allows
            symmetric treatment of estimation noise on liquid series while still
            permitting genuine negative autocorrelation (mean reversion). Pass
            None to disable lower clipping; pass 0.0 for one-sided clipping
            (legacy behaviour).
        apply_ewma_mean_smoother: If True, apply an additional EWMA smoother to
            the beta series after clipping.

    Returns:
        Tuple of (unsmoothed_returns, betas, r2) each as DataFrames matching
        the input shape.

    Example:
        >>> # Quarterly HF returns
        >>> unsmoothed, betas, r2 = unsmooth_returns_ar1_ewma(
        ...     returns=hf_returns, span=40,
        ...     max_value_for_beta=0.75, min_value_for_beta=-0.25,
        ... )
    """
    x = returns.shift(1)

    # Rolling EWMA regression: r_t = beta_t * r_{t-1} + noise
    betas, _, _, _, _, ewm_r2 = compute_ewm_beta_alpha_forecast(
        x_data=x,
        y_data=returns,
        mean_adj_type=mean_adj_type,
        span=span,
    )

    # Clip beta symmetrically — see function docstring for rationale.
    if max_value_for_beta is not None or min_value_for_beta is not None:
        betas = betas.clip(lower=min_value_for_beta, upper=max_value_for_beta)

    if apply_ewma_mean_smoother:
        betas = compute_ewm(data=betas, span=span)

    if warmup_period is not None:
        # Mask initial warmup, then backfill from first valid beta to avoid
        # NaN propagation into the unsmoothing step.
        betas = set_nans_for_warmup_period(a=betas, warmup_period=warmup_period)
        betas = betas.reindex(index=returns.index).bfill()

    # Invert the AR(1) smoothing using lagged beta (aligns with the lagged
    # return in the numerator).
    prediction = x.multiply(betas.shift(1))
    unsmoothed = (returns - prediction).divide(1.0 - betas.shift(1))

    return unsmoothed, betas, ewm_r2


def compute_ar1_unsmoothed_prices(prices: pd.DataFrame,
                                  freq: Union[str, pd.Series] = 'QE',
                                  span: int = 40,
                                  mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                                  warmup_period: Optional[int] = 8,
                                  max_value_for_beta: Optional[float] = 0.75,
                                  min_value_for_beta: Optional[float] = -0.25,
                                  is_log_returns: bool = True
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, pd.DataFrame]:
    """Apply AR(1) EWMA unsmoothing to a price panel, optionally mixed-frequency.

    Converts prices to returns at the specified frequency (or per-asset frequencies),
    applies ``unsmooth_returns_ar1_ewma``, converts back to NAVs.

    Args:
        prices: Price level DataFrame.
        freq: Either a single pandas resample frequency string applied to all
            assets, or a Series mapping asset name to frequency for mixed-frequency
            panels (e.g. monthly HFs alongside quarterly PE).
        span: EWMA span for rolling beta.
        mean_adj_type: Mean adjustment type for beta regression.
        warmup_period: Initial warmup periods to mask.
        max_value_for_beta: Upper bound for beta clipping (default 0.75).
        min_value_for_beta: Lower bound for beta clipping (default -0.25). See
            ``unsmooth_returns_ar1_ewma`` for the rationale behind symmetric
            clipping.
        is_log_returns: If True, use log returns for the regression and convert
            back with expm1 at the end. Generally recommended for unsmoothing.

    Returns:
        Tuple of (navs, unsmoothed_returns, betas, r2).
    """
    if isinstance(freq, str):
        y = to_returns(prices, freq=freq, drop_first=False, is_log_returns=is_log_returns)
        unsmoothed, betas, ewm_r2 = unsmooth_returns_ar1_ewma(
            returns=y, span=span, mean_adj_type=mean_adj_type,
            warmup_period=warmup_period,
            max_value_for_beta=max_value_for_beta,
            min_value_for_beta=min_value_for_beta,
        )
    else:
        unsmoothed_dict, betas_dict, r2_dict = {}, {}, {}
        for frequency, assets in freq.groupby(freq):
            asset_list = assets.index.tolist()
            if len(asset_list) == 0:
                continue
            y = to_returns(prices[asset_list], freq=str(frequency),
                               drop_first=False, is_log_returns=is_log_returns)
            u, b, r = unsmooth_returns_ar1_ewma(
                returns=y, span=span, mean_adj_type=mean_adj_type,
                warmup_period=warmup_period,
                max_value_for_beta=max_value_for_beta,
                min_value_for_beta=min_value_for_beta,
            )
            unsmoothed_dict[frequency] = u
            betas_dict[frequency] = b
            r2_dict[frequency] = r
        unsmoothed = pd.concat(unsmoothed_dict.values(), axis=1).reindex(columns=prices.columns)
        betas = pd.concat(betas_dict.values(), axis=1).reindex(columns=prices.columns)
        ewm_r2 = pd.concat(r2_dict.values(), axis=1).reindex(columns=prices.columns)

    if is_log_returns:
        unsmoothed = np.expm1(unsmoothed)
    navs = returns_to_nav(returns=unsmoothed)
    return navs, unsmoothed, betas, ewm_r2


# =============================================================================
# METHOD 2: STATIC GETMANSKY-LO-MAKAROV UNSMOOTHING
# =============================================================================

@dataclass
class GLMUnsmoothingDiagnostics:
    """Diagnostics from a static Getmansky-Lo-Makarov unsmoothing fit.

    Attributes:
        theta: AR coefficients (length q) interpreted as smoothing weights.
        theta_sum: Sum of theta coefficients. Smoothing parameter = 1 - theta_sum.
        vol_inflation_factor: 1 / (1 - theta_sum).
        ar_order: Lag order q used in the fit.
        is_severe: True if |theta_sum| > 0.95 (near-singular inversion).
    """
    theta: np.ndarray
    theta_sum: float
    vol_inflation_factor: float
    ar_order: int
    is_severe: bool


def unsmooth_returns_glm(returns: Union[pd.Series, pd.DataFrame],
                         ar_order: int = 3,
                         return_diagnostics: bool = False
                         ) -> Union[pd.Series, pd.DataFrame,
                                    Tuple[pd.Series, GLMUnsmoothingDiagnostics],
                                    Tuple[pd.DataFrame, dict]]:
    """Apply static AR(q) unsmoothing following Getmansky-Lo-Makarov (2004).

    Fits a single AR(q) model over the entire sample to extract constant
    smoothing weights, then inverts period-by-period. Less adaptive than the
    rolling EWMA method but useful for academic reproducibility.

    Args:
        returns: Observed return Series or DataFrame.
        ar_order: Lag order q. Standard values: 2 for monthly, 3 for higher frequency.
        return_diagnostics: If True, return tuple of (unsmoothed, diagnostics).

    Returns:
        Unsmoothed returns in the same shape as input. If ``return_diagnostics``,
        a tuple with ``GLMUnsmoothingDiagnostics`` (single) or dict (per column).

    Raises:
        ValueError: If returns has fewer than 4*ar_order observations.

    Note:
        Prefer ``unsmooth_returns_ar1_ewma`` for most practical applications.
        Use this method for academic reproducibility or single-parameter summaries.
    """
    if isinstance(returns, pd.Series):
        unsmoothed, diag = _unsmooth_glm_single(returns, ar_order=ar_order)
        if return_diagnostics:
            return unsmoothed, diag
        return unsmoothed

    if isinstance(returns, pd.DataFrame):
        unsmoothed_cols = {}
        diagnostics_dict = {}
        for col in returns.columns:
            us, diag = _unsmooth_glm_single(returns[col], ar_order=ar_order)
            unsmoothed_cols[col] = us
            diagnostics_dict[col] = diag
        unsmoothed_df = pd.DataFrame(unsmoothed_cols)
        if return_diagnostics:
            return unsmoothed_df, diagnostics_dict
        return unsmoothed_df

    raise ValueError(f"returns must be Series or DataFrame, got {type(returns)}")


def _unsmooth_glm_single(returns: pd.Series,
                         ar_order: int
                         ) -> Tuple[pd.Series, GLMUnsmoothingDiagnostics]:
    """Internal single-series worker for ``unsmooth_returns_glm``."""
    from statsmodels.tsa.ar_model import AutoReg

    clean = returns.dropna()
    if len(clean) < 4 * ar_order:
        raise ValueError(
            f"insufficient observations: {len(clean)} returns for AR({ar_order}) "
            f"(need at least {4 * ar_order})"
        )

    model = AutoReg(clean.values, lags=ar_order, old_names=False).fit()
    theta = np.asarray(model.params[1:])
    theta_sum = float(theta.sum())
    vol_inflation = 1.0 / (1.0 - theta_sum) if theta_sum < 1.0 else np.inf
    is_severe = abs(theta_sum) > 0.95

    denom = 1.0 - theta_sum
    if abs(denom) < 1e-10:
        diag = GLMUnsmoothingDiagnostics(theta=theta, theta_sum=theta_sum,
                                         vol_inflation_factor=np.inf,
                                         ar_order=ar_order, is_severe=True)
        return returns.copy(), diag

    vals = returns.values.copy().astype(float)
    out = vals.copy()
    for t in range(ar_order, len(vals)):
        if np.isnan(vals[t]):
            continue
        correction = 0.0
        for i in range(ar_order):
            lag_val = vals[t - i - 1]
            if np.isnan(lag_val):
                correction = np.nan
                break
            correction += theta[i] * lag_val
        if np.isnan(correction):
            out[t] = np.nan
        else:
            out[t] = (vals[t] - correction) / denom

    unsmoothed = pd.Series(out, index=returns.index, name=returns.name)
    diag = GLMUnsmoothingDiagnostics(theta=theta, theta_sum=theta_sum,
                                     vol_inflation_factor=vol_inflation,
                                     ar_order=ar_order, is_severe=is_severe)
    return unsmoothed, diag