"""
sd_utils.py
-----------
Thin shim around the PySDTest package
(https://pypi.org/project/pysdtest).

Exposes just one convenience function:

    sd_stat_pvalue(x, y, s=1, nboot=400, **kw)

…which returns (statistic, p_value) and auto-handles
• pandas Series / NumPy arrays
• choice of bootstrap vs. subsampling
• direction of dominance (+ statistic ⇒ x ≻ y).

Any richer analyses (joint tests, plots) can call
PySDTest directly; import it here so the rest of the
project never touches library internals.
"""

from __future__ import annotations
import numpy as np, pandas as pd
import pysdtest


# ╔═══════════════════════════════════════════════════════╗
# 1. Public helper
# ╚═══════════════════════════════════════════════════════╝
def sd_stat_pvalue(
    x,
    y,
    s: int = 1,
    nboot: int = 400,
    resampling: str = "bootstrap",
    random_state: int | None = 42,
    **kwargs,
) -> tuple[float, float]:
    """
    Parameters
    ----------
    x, y       : 1-D array-likes (pandas Series, NumPy, list)
    s          : SD order (1 = FOSD, 2, 3 …)
    nboot      : # bootstrap / subsample draws
    resampling : 'bootstrap' | 'subsampling'
    **kwargs   : forwarded to pysdtest.test_sd (e.g. bsize)

    Returns
    -------
    statistic  : float  (positive ⇒ X dominates Y)
    p_value    : float  (≤ α → reject H0 of no dominance)

    Notes
    -----
    • PySDTest already centres the bootstrap ⇒ valid even
      under near-crossing distributions (Donald-Hsu, 2016).
    • If you need joint dominance over many assets or
      higher-order integrals, call pysdtest functions
      directly in your notebook / script.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    test = pysdtest.test_sd(
        x,
        y,
        s=s,
        nboot=nboot,
        resampling=resampling,
        random_state=random_state,
        **kwargs,
    )
    return float(test.statistic), float(test.pvalue)
