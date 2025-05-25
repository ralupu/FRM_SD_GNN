import pandas as pd
from pathlib import Path

def load_controls(path_cov: Path | str,
                  path_mes: Path | str
                 ) -> pd.DataFrame:
    """
    Load CoVaR and MES panels (one column per ticker) from CSV files,
    prefix columns appropriately, and join into a single DataFrame.

    Returns
    -------
    df_controls : DataFrame indexed by Date with columns
                  ['CoVaR_<ticker1>', ..., 'MES_<tickerN>']
    """
    # 1. Read full panels
    cov = pd.read_csv(
        path_cov,
        parse_dates = ['Date'],
        index_col = 'Date',
        dayfirst = True,
    )
    mes = pd.read_csv(
        path_mes,
        parse_dates = ['Date'],
        index_col = 'Date',
        dayfirst = True,
    )

    # 2. Prefix column names
    cov = cov.add_prefix("CoVaR_")
    mes = mes.add_prefix("MES_")

    # 3. Align on dates and forward-fill
    df = pd.concat([cov, mes], axis=1)
    df = df.sort_index().ffill()

    return df
