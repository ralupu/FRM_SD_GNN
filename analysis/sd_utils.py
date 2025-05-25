from __future__ import annotations
import numpy as np
import pandas as pd
import pysdtest


def sd_stat_pvalue(
    x,
    y,
    s: int = 1,
    ngrid: int = 100,
    resampling: str = "bootstrap",
    nboot: int = 200,
    b1: int | None = None,
    b2: int | None = None,
    quiet: bool = True,
    debug: bool = False,
) -> tuple[float, float]:
    """
    Wrapper for PySDTest's test_sd function.

    Parameters
    ----------
    x, y         : 1-D array-likes (NumPy arrays, pandas Series)
    s            : SD order (1=FSD, 2=SSD, etc.)
    nboot        : number of bootstrap/subsampling draws
    ngrid        : number of grid points for ECDF evaluation
    resampling   : 'bootstrap' | 'subsampling' | 'paired bootstrap'
    b1, b2       : subsample sizes for sample1 and sample2 (only for subsampling)
    quiet        : if False, prints detailed output from PySDTest

    Returns
    -------
    stat, pvalue : floats
        The test statistic and p-value of the SD test.
    """
    # Convert inputs to numpy arrays
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Guard against empty inputs
    if x_arr.size == 0 or y_arr.size == 0:
        print("[WARN] sd_stat_pvalue called with empty sample(s); returning NaN result")
        return float('nan'), float('nan')
    y_arr = np.asarray(y)

    # Initialize the PySDTest object with the correct signature
    test = pysdtest.test_sd(
        sample1=x_arr,
        sample2=y_arr,
        ngrid=ngrid,
        s=s,
        resampling=resampling,
        b1=b1,
        b2=b2,
        nboot=nboot,
        quiet=quiet,
    )

    # Run the test
    test.testing()

    # Extract statistic and p-value
    stat = getattr(test, 'statistic', None)
    pval = getattr(test, 'pvalue', None)

    if stat is None or pval is None:
        # fall back to result dictionary if attributes not found
        result = getattr(test, 'result', {})
        stat = result.get('statistic') or result.get('test_stat') or 0.0
        pval = result.get('pvalue') or result.get('p_value') or 1.0

    # Ensure returns are numeric defaults if still None
    if stat is None:
        stat = 0.0
    if pval is None:
        pval = 1.0

    # Debug for flat/non-rejection cases

    if debug and pval >= 0.999:
        print("---- SD DEBUG ----")
        print(f"Order s={s}, nboot={nboot}, resampling={resampling}")
        print("Sample1 stats:", np.min(a1), np.mean(a1), np.max(a1))
        print("Sample2 stats:", np.min(a2), np.mean(a2), np.max(a2))
        print("Result dict:", getattr(test, "result", {}))
        print("------------------")

    return float(stat), float(pval)
