import numpy as np
import pandas as pd

def integrate_rect_col(fx_col, dx_col, init_val):
    """
    Numeric integration from a panda data frame.
    Uses simply rect method
    %fx_col% the fx col to integrate under
    %dx_col% the width of the boxes
    %init_val% the initial constant for integration

    %return% a column that represents the integration results at each row
    """
    assert len(fx_col) == len(dx_col)
    n = len(fx_col)
    fx_l = fxdx_col.tolist()
    dx_l = dx_col.tolist()
    prev_val = init_val
    y = [init_val]*n
    for i in xrange(n):
        y_val = prev_val + float(dx_l[i]) * fx_l[i]
        y[i] = y_val
        prev_val = y_val
        
    return np.array(y)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, "same")

def normalize(arr):
    min_x = min(arr)
    range_x = max(arr) - min_x
    return [ float(x-min_x) / float(range_x) for x in arr ]

def integrate_trapezoid_col(fxdx_col, dx_col, init_val):
    """
    Numeric integration from a panda data frame.
    Uses simply box method
    """
    assert len(fxdx_col) == len(dx_col)
    fxdx_l = fxdx_col.tolist()
    dx_l = dx_col.tolist()
    prev_val = init_val
    y = [init_val]*len(dx_col)
    prev_fx = 0
    for i in xrange(len(dx_col)):
        y_val = prev_val + float(dx_l[i]) * (fxdx_l[i]  + prev_fx) / 2
        y[i] = y_val
        prev_val = y_val
        prev_fx = fxdx_l[i]
        
    return np.array(y)

def generate_windows(df, window=10, ignore_columns = []):
    """
    Apply the future windows to the dataframe
    """
    points = []
    cols = df.columns.values.tolist()
    for ic in ignore_columns:
        if ic in cols:
            cols.remove(ic)
    for i, r in df.iterrows():
        w_start = i
        w_end   = min(i + 100, len(df)-1)
        row = r.to_dict()
        df_w = df.loc[w_start:w_end].reset_index(drop=True)
        for j in xrange(0,window):
            if j < len(df_w):
                window_row = df_w.loc[j].to_dict()
            else:
                window_row = None
            for c in cols:
                name = '%s_%s' % (c, j)
                row[name] = window_row[c] if window_row != None else None
        points.append(row)

    return pd.DataFrame(points)



