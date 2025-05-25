import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_recall_curve
from statsmodels.tsa.api import VAR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_logistic_lasso(
    X_train, y_train,
    Cs=10,
    cv=5,
    max_iter=10000
):
    """
    Fit L1-penalized logistic regression with built-in CV,
    handling missing values via mean imputation in a pipeline.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            penalty="l1",
            solver="saga",
            scoring="roc_auc",
            max_iter=max_iter,
            n_jobs=1,
            refit=True
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def eval_logistic(model, X, y):
    """
    Returns AUC-ROC and a precision-recall DataFrame.
    """
    prob = model.predict_proba(X)[:,1]
    auc  = roc_auc_score(y, prob)
    prec, recall, thresh = precision_recall_curve(y, prob)
    pr_df = pd.DataFrame({'precision': prec, 'recall': recall, 'threshold': list(thresh)+[None]})
    return auc, pr_df

def train_var(frm_series, cov_mes, lags=1):
    """
    Fit a VAR on the FRM index plus controls.
    """
    df = pd.concat([frm_series.rename('FRM'), cov_mes], axis=1)
    model = VAR(df)
    res   = model.fit(lags)
    return res

def forecast_var(res, steps):
    """
    Produce h-step ahead forecasts of the FRM index from a fitted VAR,
    then convert those forecasts into jump‐probabilities by comparing the
    predicted change against the historical jump threshold.

    Parameters
    ----------
    res : statsmodels VARResults
        Fitted VARResults object from train_var().
    steps : int
        Number of steps ahead to forecast (should match your jump horizon).

    Returns
    -------
    prob : pandas.Series
        Indexed by the forecast dates, containing the binary jump indicator.
    """
    import pandas as pd

    # 1) Forecast `steps` ahead
    fc = res.forecast(y=res.endog, steps=steps)  # numpy array (steps × n_series)

    # 2) Identify the last in-sample date from the fitted model
    try:
        # Newer statsmodels: model.data.row_labels
        dates_index = res.model.data.row_labels
    except Exception:
        try:
            # Older versions: model.data.dates
            dates_index = res.model.data.dates
        except Exception:
            # Fallback to integer index
            dates_index = pd.RangeIndex(start=0, stop=len(res.endog), step=1)

    last_date = dates_index[-1]

    # 3) Infer frequency if datetime-like
    if hasattr(dates_index, "freq") and dates_index.freq is not None:
        freq = dates_index.freq
    else:
        try:
            freq = pd.infer_freq(dates_index)
        except Exception:
            freq = None

    # 4) Build forecast date index
    if freq:
        # e.g. 'D' or 'B' or 'W'
        dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                              periods=steps, freq=freq)
    else:
        dates = pd.RangeIndex(start=0, stop=steps, step=1)

    # 5) Extract FRM forecasts
    frm_fc = pd.Series(fc[:, 0], index=dates, name="FRM_forecast")

    # 6) Compute h-step ahead change relative to last observed FRM
    frm_last = res.endog[-1, 0]
    delta = frm_fc - frm_last

    # 7) Historical threshold from in-sample FRM changes
    try:
        hist = pd.Series(res.model.endog[:, 0], index=dates_index)
    except Exception:
        # If that fails, rebuild from res.endog and integer index
        hist = pd.Series(res.endog[:, 0], index=pd.RangeIndex(len(res.endog)))

    hist_delta = hist.shift(-steps) - hist
    thr = hist_delta.quantile(0.90)  # or store quantile in res if desired

    # 8) Binary jump indicator
    prob = (delta > thr).astype(int)
    prob.name = "VAR_jump_prob"

    return prob


