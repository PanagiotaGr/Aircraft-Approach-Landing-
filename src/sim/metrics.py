import numpy as np


def rmse(arr):
    a = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else float("nan")


def touchdown(
    x, h,
    x_thresh, h_thresh=None,
    x_window=2000.0,
    h_ground=1.0,
    h_touch=None,
    debug=False,
    **kwargs
):
    """
    Touchdown when:
      - x >= x_thresh
      - x <= x_thresh + x_window
      - h <= h_ground

    Accepts extra kwargs for compatibility.
    """
    if h_touch is not None:
        h_ground = h_touch

    if debug and not hasattr(touchdown, "_printed"):
        print("[touchdown args]",
              "x_thresh=", x_thresh,
              "x_window=", x_window,
              "h_ground=", h_ground,
              "h_touch=", h_touch,
              "h_thresh=", h_thresh,
              "extra=", kwargs)
        touchdown._printed = True

    td = (x >= x_thresh) and (x <= x_thresh + x_window) and (h <= h_ground)

    if debug and td and not hasattr(touchdown, "_hit"):
        print(f"[TOUCHDOWN] x={x:.1f} h={h:.2f} (window {x_thresh:.1f}..{x_thresh + x_window:.1f}, h_ground={h_ground})")
        touchdown._hit = True

    return td

def stabilized_gate(
    y_err,
    V_err,
    gamma,
    h,
    gate_h=150.0,
    y_max=5.0,
    V_max=10.0,
):


    """
    Stabilized approach check below gate_h.

    We check:
      - |y_err| <= y_max  (localizer)
      - |V_err| <= V_max  (speed)

    NOTE: We intentionally do NOT check abs(gamma) around 0,
    because during approach gamma is naturally negative (~ -3 deg).
    """
    if h > gate_h:
        return True

    if abs(y_err) > y_max:
        return False
    if abs(V_err) > V_max:
        return False

    return True
