import numpy as np

def rmse(arr):
    a = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(a*a))) if a.size else float("nan")

def touchdown(x, h, x_thresh=0.0, h_thresh=15.0):
    # "touchdown" near runway threshold and near flare height
    return (x >= x_thresh - 10.0) and (h <= h_thresh + 1.0)

def stabilized_gate(y_err, V_err, gamma, h, gate_h=300.0):
    """
    Simple stabilized approach check below gate_h (meters).
    Returns True if stable, False otherwise.
    """
    if h > gate_h:
        return True
    if abs(y_err) > 80.0:   # meters
        return False
    if abs(V_err) > 7.0:    # m/s
        return False
    if abs(gamma) > np.deg2rad(8.0):
        return False
    return True

