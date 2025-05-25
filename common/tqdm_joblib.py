"""
common/tqdm_joblib.py
---------------------
Context-manager that patches joblib so each completed batch
updates a tqdm progress-bar.  Use around Parallel(...) calls.

Usage:
    from tqdm.auto import tqdm
    from common.tqdm_joblib import tqdm_joblib

    with tqdm_joblib(tqdm(total=N)) as pbar:
        results = Parallel(n_jobs=4)(delayed(func)(i) for i in items)
"""

import joblib
from joblib.parallel import BatchCompletionCallBack
from contextlib import contextmanager
from tqdm.auto import tqdm

@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Yields a tqdm progress-bar updated inside joblib’s BatchCompletionCallBack.
    Restores the original callback on exit.
    """
    class TqdmCallback(BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # Patch the callback
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmCallback
    try:
        yield tqdm_object
    finally:
        tqdm_object.close()
        joblib.parallel.BatchCompletionCallBack = old_cb
