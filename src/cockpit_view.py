import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch

from sim.dynamics import PointMass3DOF
from sim.guidance import ILSGuidance
from sim.wind import WindModel
from sim.metrics import touchdown

deg = np.pi / 180

# -------------------------
# OPTIONS
# -------------------------
SAVE_VIDEO = False  # True για export mp4
VIDEO_PATH = "reports/figures/cockpit_sim.mp4"
FPS = 30

STEPS_PER_FRAME = 6
INTERVAL_MS = 20


def clamp(x, a, b):
    return float(np.clip(x, a, b))


# -------------------------
# Instrument drawing helpers
# -------------------------
def _bezel(ax):
    ax.axis("off")
    ax.add_patch(Circle((0.5, 0.5), 0.50, fill=True, facecolor="#111111", edgecolor="none", alpha=0.92))
    ax.add_patch(Circle((0.5, 0.5), 0.48, fill=False, lw=2, edgecolor="white", alpha=0.85))


def draw_airspeed(ax, V):
    kts = V * 1.94384
    ax.clear()
    _bezel(ax)
    ax.text(0.5, 0.86, "AIRSPEED", ha="center", va="center", fontsize=10, color="white")
    ax.text(0.5, 0.55, f"{kts:5.0f}", ha="center", va="center", fontsize=22, family="monospace", color="white")
    ax.text(0.5, 0.33, "kts", ha="center", va="center", fontsize=10, color="white")


def draw_altimeter(ax, h):
    ft = h * 3.28084
    ax.clear()
    _bezel(ax)
    ax.text(0.5, 0.86, "ALT", ha="center", va="center", fontsize=10, color="white")
    ax.text(0.5, 0.55, f"{ft:6.0f}", ha="center", va="center", fontsize=22, family="monospace", color="white")
    ax.text(0.5, 0.33, "ft", ha="center", va="center", fontsize=10, color="white")


def draw_vspeed(ax, h_dot):
    fpm = h_dot * 196.8504
    ax.clear()
    _bezel(ax)
    ax.text(0.5, 0.86, "V/S", ha="center", va="center", fontsize=10, color="white")
    ax.text(0.5, 0.55, f"{fpm:6.0f}", ha="center", va="center", fontsize=18, family="monospace", color="white")
    ax.text(0.5, 0.33, "fpm", ha="center", va="center", fontsize=10, color="white")


def draw_hsi(ax, psi):
    hdg = (np.rad2deg(psi) % 360.0)
    ax.clear()
    _bezel(ax)
    ax.text(0.5, 0.86, "HDG", ha="center", va="center", fontsize=10, color="white")
    ax.text(0.5, 0.55, f"{hdg:03.0f}°", ha="center", va="center", fontsize=20, family="monospace", color="white")
    ax.plot([0.5, 0.5], [0.50, 0.88], lw=2, color="white")


def draw_attitude(ax, gamma, phi):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    ax.add_patch(Circle((0, 0), 1.00, fill=True, facecolor="#111111", edgecolor="none", alpha=0.92))
    ax.add_patch(Circle((0, 0), 0.98, fill=False, lw=2, edgecolor="white", alpha=0.85))

    pitch = clamp(-np.rad2deg(gamma) / 12.0, -0.6, 0.6)
    roll = phi
    c, s = np.cos(roll), np.sin(roll)

    x1, y1 = -2.0, pitch
    x2, y2 =  2.0, pitch
    X1, Y1 = c*x1 - s*y1, s*x1 + c*y1
    X2, Y2 = c*x2 - s*y2, s*x2 + c*y2
    ax.plot([X1, X2], [Y1, Y2], lw=2, color="white")

    ax.plot([-0.25, 0.25], [0, 0], lw=3, color="white")
    ax.plot([0, 0], [0, -0.15], lw=3, color="white")


def draw_ils(ax, loc, gs):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, fill=True, facecolor="#111111", edgecolor="white", lw=2, alpha=0.92))
    ax.text(0, 0.92, "ILS", ha="center", va="top", fontsize=10, color="white")

    ax.plot([-0.8, 0.8], [0, 0], lw=1, color="white")
    ax.plot([0, 0], [-0.8, 0.8], lw=1, color="white")

    ax.plot([clamp(loc, -0.8, 0.8)], [0], marker="D", markersize=9, color="white")
    ax.plot([0], [clamp(gs, -0.8, 0.8)], marker="D", markersize=9, color="white")

    ax.text(0, -0.95, "LOC ↔   GS ↕", ha="center", va="bottom", fontsize=8, color="white")


# -------------------------
# Main
# -------------------------
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

    y_prev = model.y
    h_prev = model.h
    t = 0.0
    done = {"flag": False}

    fig = plt.figure(figsize=(13, 7))
    grid = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.2], wspace=0.25, hspace=0.2)

    ax_view = fig.add_subplot(grid[0, :])
    ax_view.set_title("Full Cockpit View — Runway + PAPI + Flight Director + Instruments")
    ax_view.set_xlim(-1.2, 1.2)
    ax_view.set_ylim(-0.25, 1.25)
    ax_view.axis("off")

    # background
    sky = Rectangle((-2.0, 0.62), 4.0, 2.0, facecolor="#87CEEB", edgecolor="none", alpha=0.55)
    ground = Rectangle((-2.0, -2.0), 4.0, 2.65, facecolor="#6B8E23", edgecolor="none", alpha=0.55)
    ax_view.add_patch(sky)
    ax_view.add_patch(ground)

    # horizon
    horizon, = ax_view.plot([-2, 2], [0.62, 0.62], lw=2)

    # runway
    runway_outline = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], closed=True, fill=False, lw=3)
    ax_view.add_patch(runway_outline)

    runway_fill = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], closed=True, fill=True,
                          facecolor="#2b2b2b", alpha=0.70, edgecolor="none")
    ax_view.add_patch(runway_fill)

    runway_left, = ax_view.plot([], [], lw=2)
    runway_right, = ax_view.plot([], [], lw=2)
    centerline, = ax_view.plot([0, 0], [0.0, 1.15], lw=1, linestyle="--")
    center_dashes = [ax_view.plot([], [], lw=2, linestyle="--")[0] for _ in range(7)]

    # flight director (thinner)
    fd_h, = ax_view.plot([-0.15, 0.15], [0.58, 0.58], lw=2.2)
    fd_v, = ax_view.plot([0.0, 0.0], [0.43, 0.73], lw=2.2)

    # PAPI (moved closer)
    papi_x = 0.45
    papi_y = 0.18
    papi = [Circle((papi_x + i*0.08, papi_y), 0.018, fill=True) for i in range(4)]
    for c in papi:
        ax_view.add_patch(c)
    ax_view.text(papi_x + 0.12, papi_y - 0.05, "PAPI", ha="center", va="top", fontsize=9)

    # yoke (lower/smaller so it doesn't block runway)
    yoke_column = FancyBboxPatch((-0.05, -0.25), 0.10, 0.22,
                                 boxstyle="round,pad=0.02,rounding_size=0.03",
                                 linewidth=2, edgecolor="black", facecolor="#333333", alpha=0.9)
    ax_view.add_patch(yoke_column)

    yoke_rim = Circle((0.0, -0.10), 0.13, fill=False, lw=8, edgecolor="#1a1a1a", alpha=0.95)
    ax_view.add_patch(yoke_rim)
    yoke_hub = Circle((0.0, -0.10), 0.05, fill=True, facecolor="#2a2a2a", edgecolor="black", lw=2, alpha=0.95)
    ax_view.add_patch(yoke_hub)

    yoke_spoke1, = ax_view.plot([-0.10, -0.02], [-0.10, -0.10], lw=6)
    yoke_spoke2, = ax_view.plot([0.02, 0.10], [-0.10, -0.10], lw=6)
    yoke_spoke3, = ax_view.plot([0.0, 0.0], [-0.20, -0.06], lw=6)

    # info text in a translucent box
    info = ax_view.text(
        -1.18, 1.22, "",
        ha="left", va="top", family="monospace", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.35, edgecolor="none")
    )

    # instruments bottom
    ax_asi = fig.add_subplot(grid[1, 0])
    ax_att = fig.add_subplot(grid[1, 1])
    ax_ils = fig.add_subplot(grid[1, 2])
    ax_alt = fig.add_subplot(grid[1, 3])

    ax_hsi = ax_view.inset_axes([0.02, 0.73, 0.12, 0.24])
    ax_vs = ax_view.inset_axes([0.86, 0.73, 0.12, 0.24])

    # mapping
    def map_loc(y_m, x_m):
        dist = max(abs(x_m), 200.0)
        full_scale = np.interp(dist, [200.0, 6000.0], [60.0, 250.0])
        return clamp(-y_m / full_scale, -1.0, 1.0)

    def map_gs(h_err_m, x_m):
        dist = max(abs(x_m), 200.0)
        full_scale = np.interp(dist, [200.0, 6000.0], [25.0, 120.0])
        return clamp(h_err_m / full_scale, -1.0, 1.0)

    def runway_poly(y_m, x_m):
        dist = max(abs(x_m), 80.0)
        scale = np.interp(dist, [80.0, 6500.0], [1.0, 0.16])
        off = clamp(-y_m / dist, -0.9, 0.9) * 0.55

        near_w = 0.38 * scale
        far_w = 0.14 * scale
        near_y = 0.05
        far_y = 1.12

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

        model.step(dt, phi_cmd, gamma_cmd, throttle_cmd, wind.sample(t))

        if touchdown(model.x, model.h, approach["x_threshold"], approach["h_threshold"]):
            done["flag"] = True
        if model.x > approach["x_threshold"] + 80.0:
            done["flag"] = True

        t += dt
        return y_dot, h_dot, href

    def update(_frame):
        if not done["flag"] and t < T:
            for _ in range(STEPS_PER_FRAME):
                y_dot, h_dot, href = step_sim()
                if done["flag"]:
                    break
        else:
            y_dot, h_dot, href = 0.0, 0.0, guidance.h_ref(model.x)

        h_err = href - model.h
        loc = map_loc(model.y, model.x)
        gs_dev = map_gs(h_err, model.x)

        # horizon rotation
        pitch = clamp(-np.rad2deg(model.gamma) / 14.0, -0.45, 0.45)
        roll = model.phi
        c, s = np.cos(roll), np.sin(roll)

        x1, y1h = -2.0, 0.62 + pitch
        x2, y2h =  2.0, 0.62 + pitch

        def rot_about(x, y, cx=0.0, cy=0.62):
            xr, yr = x - cx, y - cy
            X = c*xr - s*yr + cx
            Y = s*xr + c*yr + cy
            return X, Y

        X1, Y1 = rot_about(x1, y1h)
        X2, Y2 = rot_about(x2, y2h)
        horizon.set_data([X1, X2], [Y1, Y2])

        # runway visuals
        pts, off = runway_poly(model.y, model.x)
        runway_outline.set_xy(pts)
        runway_fill.set_xy(pts)

        runway_left.set_data([pts[0][0], pts[3][0]], [pts[0][1], pts[3][1]])
        runway_right.set_data([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]])

        centerline.set_data([off, off], [0.0, 1.15])
        y0, y1d = 0.10, 1.10
        dash_positions = np.linspace(y0, y1d, len(center_dashes))
        for i, yp in enumerate(dash_positions):
            dash_len = np.interp(yp, [y0, y1d], [0.07, 0.015])
            center_dashes[i].set_data([off, off], [yp, min(yp + dash_len, 1.16)])

        # PAPI from glideslope deviation: above slope -> more red
        dev = clamp(-h_err / 80.0, -3.0, 3.0)
        reds = int(np.clip(2 + np.round(dev), 0, 4))
        for i in range(4):
            if i < reds:
                papi[i].set_facecolor("#cc0000")
                papi[i].set_edgecolor("#550000")
            else:
                papi[i].set_facecolor("#f2f2f2")
                papi[i].set_edgecolor("#999999")

        # instruments
        draw_airspeed(ax_asi, model.V)
        draw_attitude(ax_att, model.gamma, model.phi)
        draw_ils(ax_ils, loc, gs_dev)
        draw_altimeter(ax_alt, model.h)
        draw_vspeed(ax_vs, h_dot)
        draw_hsi(ax_hsi, model.psi)

        Vwx, Vwy = wind.sample(t)
        info.set_text(
            f"t={t:6.1f}s   x={model.x:8.1f}m   y={model.y:7.1f}m   h={model.h:7.1f}m\n"
            f"V={model.V:6.1f}m/s   phi={np.rad2deg(model.phi):6.1f}deg   gamma={np.rad2deg(model.gamma):6.1f}deg\n"
            f"h_ref={href:7.1f}m   h_err={h_err:7.1f}m   throttle={model.throttle:5.2f}   n={model.n:4.2f}\n"
            f"wind: Vwx={Vwx:5.1f}  Vwy={Vwy:5.1f}   PAPI reds={reds}/4"
        )

        return (horizon, runway_outline, runway_fill, runway_left, runway_right, centerline, fd_h, fd_v, info)

    anim = FuncAnimation(fig, update, interval=INTERVAL_MS, blit=False)

    if SAVE_VIDEO:
        writer = FFMpegWriter(fps=FPS, bitrate=2400)
        print(f"Saving video to: {VIDEO_PATH}")
        anim.save(VIDEO_PATH, writer=writer)
        print("Done.")

    plt.show()


if __name__ == "__main__":
    main()

