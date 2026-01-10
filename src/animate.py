import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim.dynamics import PointMass3DOF
from sim.guidance import ILSGuidance
from sim.wind import WindModel
from sim.metrics import touchdown

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

    # Logs for animation
    xs, ys, hs, hrefs = [], [], [], []

    # derivatives
    y_prev = model.y
    h_prev = model.h

    # --- Figure layout ---
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(1, 2, 1)  # ground track
    ax2 = fig.add_subplot(1, 2, 2)  # altitude vs x

    # Ground track artists
    (track_line,) = ax1.plot([], [], lw=2)
    (pos_dot,) = ax1.plot([], [], marker="o")
    ax1.axhline(0, linestyle="--")
    ax1.set_title("Ground track (real time)")
    ax1.set_xlabel("x (m) -> threshold at 0")
    ax1.set_ylabel("y (m)")
    ax1.grid(True)

    # Altitude artists
    (alt_line,) = ax2.plot([], [], lw=2, label="h")
    (ref_line,) = ax2.plot([], [], lw=2, linestyle="--", label="h_ref")
    (alt_dot,) = ax2.plot([], [], marker="o")
    ax2.set_title("Glideslope (real time)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("h (m)")
    ax2.grid(True)
    ax2.legend()

    # Axis ranges (initial)
    ax1.set_xlim(model.x - 500, 200)
    ax1.set_ylim(-1500, 1500)

    ax2.set_xlim(model.x - 500, 200)
    ax2.set_ylim(0, max(model.h, guidance.h_ref(model.x)) + 200)

    # Simulation control
    t = 0.0
    finished = {"done": False}

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

        out = model.step(dt, phi_cmd, gamma_cmd, throttle_cmd, wind.sample(t))

        xs.append(out["x"])
        ys.append(out["y"])
        hs.append(out["h"])
        hrefs.append(href)

        # stop conditions
        if touchdown(model.x, model.h, x_thresh=approach["x_threshold"], h_thresh=approach["h_threshold"]):
            finished["done"] = True
        if model.x > approach["x_threshold"] + 50.0:
            finished["done"] = True

        t += dt

    def init_anim():
        track_line.set_data([], [])
        pos_dot.set_data([], [])
        alt_line.set_data([], [])
        ref_line.set_data([], [])
        alt_dot.set_data([], [])
        return track_line, pos_dot, alt_line, ref_line, alt_dot

    def update(frame):
        # run a few sim steps per frame for smoother motion
        if not finished["done"] and t < T:
            for _ in range(3):
                step_sim()
                if finished["done"]:
                    break

        if len(xs) > 1:
            track_line.set_data(xs, ys)
            pos_dot.set_data([xs[-1]], [ys[-1]])

            alt_line.set_data(xs, hs)
            ref_line.set_data(xs, hrefs)
            alt_dot.set_data([xs[-1]], [hs[-1]])

            # Keep view moving with aircraft (x is increasing toward 0)
            xmin = xs[-1] - 2000
            xmax = 200
            ax1.set_xlim(xmin, xmax)
            ax2.set_xlim(xmin, xmax)

            # Auto y-limits altitude window
            ax2.set_ylim(0, max(max(hs[-200:]), max(hrefs[-200:])) + 200)

        return track_line, pos_dot, alt_line, ref_line, alt_dot

    anim = FuncAnimation(fig, update, init_func=init_anim, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
