import numpy as np

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
