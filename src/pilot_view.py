import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim.dynamics import PointMass3DOF
from sim.guidance import ILSGuidance
from sim.wind import WindModel
from sim.metrics import touchdown

deg = np.pi / 180


def clamp(x, a, b):
    return float(np.clip(x, a, b))


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

    # For numerical derivatives
    y_prev = model.y
    h_prev = model.h

    # --- HUD figure ---
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Pilot View (HUD-style) — Localizer / Glideslope")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Crosshair
    ax.plot([-0.15, 0.15], [0, 0], lw=2)
    ax.plot([0, 0], [-0.15, 0.15], lw=2)

    # Localizer scale (horizontal)
    ax.plot([-0.9, 0.9], [-0.75, -0.75], lw=2)
    for tick in np.linspace(-0.9, 0.9, 7):
        ax.plot([tick, tick], [-0.78, -0.72], lw=1)
    loc_label = ax.text(0, -0.88, "LOCALIZER", ha="center", va="center")

    # Glideslope scale (vertical)
    ax.plot([0.95, 0.95], [-0.8, 0.8], lw=2)
    for tick in np.linspace(-0.8, 0.8, 7):
        ax.plot([0.92, 0.98], [tick, tick], lw=1)
    gs_label = ax.text(0.95, 0.92, "GLIDESLOPE", ha="center", va="center", rotation=90)

    # Indicators (diamonds)
    loc_diamond, = ax.plot([0], [-0.75], marker="D", markersize=10)
    gs_diamond, = ax.plot([0.95], [0], marker="D", markersize=10)

    # Runway "box" (simple perspective)
    runway_poly = plt.Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], closed=True, fill=False, lw=2)
    ax.add_patch(runway_poly)
    runway_centerline, = ax.plot([0, 0], [-0.05, 0.65], lw=1, linestyle="--")

    # Horizon line (proxy of pitch via gamma)
    horizon, = ax.plot([-1.2, 1.2], [0, 0], lw=1)

    # Text overlay
    info = ax.text(-1.15, 0.95, "", ha="left", va="top", family="monospace")

    t = 0.0
    done = {"flag": False}

    def hud_map_localizer(y_m, x_m):
        """
        Convert cross-track error (meters) into HUD position [-0.9..0.9].
        Scale increases near runway (more sensitive).
        """
        dist = max(abs(x_m), 200.0)
        # normalize: 1 "full-scale" ~ 250 m far, ~ 60 m near
        full_scale = np.interp(dist, [200.0, 6000.0], [60.0, 250.0])
        return clamp(-y_m / full_scale, -0.9, 0.9)

    def hud_map_glideslope(h_err_m, x_m):
        """
        Convert altitude error (meters) into HUD vertical position [-0.8..0.8].
        More sensitive near runway.
        """
        dist = max(abs(x_m), 200.0)
        full_scale = np.interp(dist, [200.0, 6000.0], [25.0, 120.0])
        return clamp(h_err_m / full_scale, -0.8, 0.8)

    def runway_shape(y_m, x_m):
        """
        Simple perspective runway box:
        - width grows as x approaches 0
        - lateral shift depends on y/x (like looking left/right)
        """
        dist = max(abs(x_m), 80.0)
        # Perspective scale (closer -> bigger)
        s = np.interp(dist, [80.0, 6000.0], [0.9, 0.15])
        # Lateral visual offset (angle approx)
        off = clamp(-y_m / dist, -0.8, 0.8) * 0.6

        # Define a trapezoid
        near_w = 0.30 * s
        far_w = 0.12 * s
        near_y = -0.05
        far_y = 0.65

        p1 = [off - near_w, near_y]
        p2 = [off + near_w, near_y]
        p3 = [off + far_w, far_y]
        p4 = [off - far_w, far_y]
        return [p1, p2, p3, p4], off

    def step_sim():
        nonlocal t, y_prev, h_prev

        y_dot = (model.y - y_prev) / dt
        h_dot = (model.h - h_prev) / dt
        y_prev, h_prev = model.y, model.h

        phi_cmd, gamma_cmd, throttle_cmd, href = guidance.compute(
            state={
                "x": model.x, "y": model.y, "h": model.h, "V": model.V,
                "psi": model.psi, "gamma": model.gamma, "phi": model.phi,
                "throttle": model.throttle, "n": model.n
            },
            rates={"y_dot": y_dot, "h_dot": h_dot}
        )

        wxy = wind.sample(t)
        model.step(dt, phi_cmd, gamma_cmd, throttle_cmd, wxy)

        # stop conditions
        if touchdown(model.x, model.h, x_thresh=approach["x_threshold"], h_thresh=approach["h_threshold"]):
            done["flag"] = True
        if model.x > approach["x_threshold"] + 80.0:
            done["flag"] = True

        t += dt

    def update(_frame):
        # run a few steps per frame
        if not done["flag"] and t < T:
            for _ in range(4):
                step_sim()
                if done["flag"]:
                    break

        # Guidance references
        href = guidance.h_ref(model.x)
        h_err = href - model.h  # positive means we are below reference (need climb), negative means above

        # HUD indicators
        loc_x = hud_map_localizer(model.y, model.x)
        gs_y = hud_map_glideslope(h_err, model.x)

        loc_diamond.set_data([loc_x], [-0.75])
        gs_diamond.set_data([0.95], [gs_y])

        # Runway
        poly_pts, off = runway_shape(model.y, model.x)
        runway_poly.set_xy(poly_pts)
        runway_centerline.set_data([off, off], [-0.05, 0.65])

        # Horizon (gamma proxy): when descending (negative gamma) horizon goes up a bit
        horizon_y = clamp(-np.rad2deg(model.gamma) / 12.0, -0.6, 0.6)
        horizon.set_data([-1.2, 1.2], [horizon_y, horizon_y])

        # Text
        Vwx, Vwy = wind.sample(t)
        info.set_text(
            f"t={t:6.1f} s\n"
            f"x={model.x:8.1f} m  y={model.y:7.1f} m  h={model.h:7.1f} m\n"
            f"V={model.V:6.1f} m/s  phi={np.rad2deg(model.phi):6.1f} deg  gamma={np.rad2deg(model.gamma):6.1f} deg\n"
            f"h_ref={href:7.1f} m  h_err={h_err:7.1f} m\n"
            f"throttle={model.throttle:4.2f}  n={model.n:4.2f}\n"
            f"wind: Vwx={Vwx:5.1f}  Vwy={Vwy:5.1f}"
        )

        return loc_diamond, gs_diamond, runway_poly, runway_centerline, horizon, info

    _anim = FuncAnimation(fig, update, interval=30, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
