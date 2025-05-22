"""
common/tqdm_joblib.py
---------------------
Context-manager that patches joblib so each completed batch
increments a given tqdm progress-bar.

Usage
-----
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from common.tqdm_joblib import tqdm_joblib

with tqdm_joblib(tqdm(total=len(tasks))) as pbar:
    results = Parallel(n_jobs=-1)(delayed(func)(t) for t in tasks)
"""

from contextlib import contextmanager
from joblib import Parallel


@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Yields a tqdm progress-bar updated inside joblib’s BatchCompletionCallback.
    Restore original callback on exit.
    """
    class TqdmBatchCompletionCallback(Parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # Patch
    old_callback = Parallel.BatchCompletionCallBack
    Parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        tqdm_object.close()
        Parallel.BatchCompletionCallBack = old_callback
