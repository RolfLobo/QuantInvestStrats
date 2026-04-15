import pandas as pd
from enum import Enum
from qis.perfstats.config import PerfParams
from qis.perfstats.perf_stats import (compute_ra_perf_table,
                                      compute_ra_perf_table_with_benchmark,
                                      compute_rolling_drawdowns,
                                      compute_drawdowns_stats_table)

class LocalTests(Enum):
    RA_PERF_TABLE = 1
    RA_PERF_TABLE_WITH_BENCHMARK = 2
    DRAWDOWN = 3
    DRAWDOWN_STATS_TABLE = 4
    TOP_BOTTOM = 5


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from qis.test_data import load_etf_data
    prices = load_etf_data() # .dropna()

    if local_test == LocalTests.RA_PERF_TABLE:
        perf_params = PerfParams(freq='B')
        table = compute_ra_perf_table(prices=prices, perf_params=perf_params)
        print(table)

    elif local_test == LocalTests.RA_PERF_TABLE_WITH_BENCHMARK:
        perf_params = PerfParams(freq='ME')
        table = compute_ra_perf_table_with_benchmark(prices=prices, benchmark=prices.columns[0], perf_params=perf_params)
        print(table)

    elif local_test == LocalTests.DRAWDOWN:
        dd_data = compute_rolling_drawdowns(prices=prices['SPY'])
        print(dd_data)

    elif local_test == LocalTests.DRAWDOWN_STATS_TABLE:
        df = compute_drawdowns_stats_table(price=prices['SPY'])
        print(df)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.DRAWDOWN_STATS_TABLE)
