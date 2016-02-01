def prepare_data(df, ignore_columns=[]):
    temp = df.copy()
    active_columns = temp.columns.values.tolist()
    for c in ignore_columns:
        if c in active_columns:
            active_columns.remove(c)
    temp = temp[active_columns]
    return temp