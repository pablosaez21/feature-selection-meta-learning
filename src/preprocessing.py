

def imputar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    #busca las columnas que contengan todo NaN, y las dropea
    cols_all_nan = df.columns[df.isna().all()]
    if len(cols_all_nan) > 0:
        df = df.drop(columns=cols_all_nan) 
    #busca las columnas que tengan mas de un 95 % de NaNs, y también las dropea 
    pct_nan = df.isna().mean() 
    cols_casi_nan = pct_nan[pct_nan > 0.95].index 
    if len(cols_casi_nan) > 0:
        df = df.drop(columns=cols_casi_nan)
    # Imputa el resto: mediana para numéricas, moda para categóricas
    for col in df.columns:
        serie = df[col]

        if is_numeric_dtype(serie):
            df[col] = serie.fillna(float(serie.median()))
   
        else:
            moda = serie.mode(dropna=True)
            df[col] = serie.fillna(moda.iloc[0])

    return df


