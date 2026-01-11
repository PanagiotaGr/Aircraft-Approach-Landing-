import yaml
import numpy as np

from sim.dynamics import PointMass3DOF
from sim.guidance import ILSGuidance
from sim.wind import WindModel
from sim.metrics import touchdown, rmse, stabilized_gate
from plots import plot_results

deg = np.pi / 180


def main():
    with open("configs/baseline.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    dt = cfg["simulation"]["dt"]
    T = cfg["simulation"]["total_time"]

    approach = cfg["approach"]
    aircraft_cfg = cfg["aircraft"]
    limits = cfg["limits"]
    refs = cfg["references"]
    gains = cfg["guidance_gains"]

    init = cfg["initial_state"]
    state0 = {
        "x": float(init["x"]),
        "y": float(init["y"]),
        "h": float(init["h"]),
        "V": float(init["V"]),
        "psi": float(init["psi_deg"]) * deg,
        "gamma": float(init["gamma_deg"]) * deg,
        "phi": float(init["phi_deg"]) * deg,
        "throttle": float(init["throttle"]),
        "n": float(init["n"]),
    }

    model = PointMass3DOF(state0, aircraft_cfg, limits)
    guidance = ILSGuidance(gains, approach, limits, refs, aircraft_cfg)
    wind = WindModel(**cfg["wind"])

    log = {k: [] for k in ["x", "y", "h", "V", "psi", "gamma", "phi", "throttle", "n", "href"]}

    y_prev = model.y
    h_prev = model.h

    stable = True
    t = 0.0

    # Touchdown settings
    x_thresh = float(approach["x_threshold"])
    h_thresh = approach.get("h_threshold", approach.get("h_thresh", None))
    x_window = float(approach.get("x_window", 2000.0))
    h_ground = float(approach.get("h_ground", 1.0))

    crossed_threshold_printed = False

    while t < T:
        y_dot = (model.y - y_prev) / dt
        h_dot = (model.h - h_prev) / dt
        y_prev, h_prev = model.y, model.h

        phi_cmd, gamma_cmd, throttle_cmd, href = guidance.compute(
            state={
                "x": model.x,
                "y": model.y,
                "h": model.h,
                "V": model.V,
                "psi": model.psi,
                "gamma": model.gamma,
                "phi": model.phi,
                "throttle": model.throttle,
                "n": model.n,
            },
            rates={"y_dot": y_dot, "h_dot": h_dot},
        )

        out = model.step(dt, phi_cmd, gamma_cmd, throttle_cmd, wind.sample(t))

        # Logging
        log["x"].append(out["x"])
        log["y"].append(out["y"])
        log["h"].append(out["h"])
        log["V"].append(out["V"])
        log["psi"].append(out["psi"])
        log["gamma"].append(out["gamma"])
        log["phi"].append(out["phi"])
        log["throttle"].append(out["throttle"])
        log["n"].append(out["n"])
        log["href"].append(href)

        # Threshold crossing (once)
        if (not crossed_threshold_printed) and (model.x >= x_thresh):
            print(f"Crossed threshold: t={t:.1f}s, x={model.x - x_thresh:.1f} m, h={model.h:.1f} m, y={model.y:.1f} m")
            crossed_threshold_printed = True

        # Stabilized approach gate
        y_err = model.y
        V_err = refs["V_ref"] - model.V
        stable = stable and stabilized_gate(y_err, V_err, model.gamma, model.h)

        # Touchdown
        if touchdown(
            model.x,
            model.h,
            x_thresh=x_thresh,
            h_thresh=h_thresh,
            x_window=x_window,
            h_ground=h_ground,
            debug=True,
        ):
            break

        # Missed approach if we exit runway window without touchdown
        if model.x > x_thresh + x_window:
            print("Missed approach: passed runway window without touchdown.")
            break

        t += dt

    # Metrics
    y_rmse = rmse(log["y"])
    h_err = np.array(log["href"]) - np.array(log["h"])
    h_rmse = rmse(h_err)

    print(f"Finished at t={t:.1f}s, x={model.x:.1f} m, h={model.h:.1f} m")
    print(f"RMSE lateral y: {y_rmse:.2f} m")
    print(f"RMSE glideslope h: {h_rmse:.2f} m")
    print(f"Stabilized approach: {'YES' if stable else 'NO'}")

    plot_results(log)


if __name__ == "__main__":
    main()
