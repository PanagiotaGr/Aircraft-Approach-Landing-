# src/sim/run_sim.py
from __future__ import annotations
from dataclasses import dataclass
import time
import math
from typing import Dict, Tuple, List

import numpy as np


@dataclass
class SimConfig:
    dt: float = 0.01          # 100 Hz
    t_final: float = 30.0
    real_time: bool = True
    log_hz: int = 20


@dataclass
class AircraftParams:
    mass: float = 1200.0      # kg
    g: float = 9.80665
    S: float = 16.2           # wing area m^2
    rho: float = 1.225        # sea level density kg/m^3
    CL0: float = 0.2
    CLa: float = 5.5          # per rad
    CD0: float = 0.03
    k: float = 0.05           # induced drag factor
    Tmax: float = 2500.0      # N (toy)
    Iy: float = 2500.0        # kg m^2 (toy pitch inertia)


@dataclass
class Control:
    throttle: float = 0.4     # 0..1
    elevator: float = 0.0     # rad (toy)
    # (later: aileron, rudder)


@dataclass
class State:
    # 2D longitudinal toy model: x,z,u,w,theta,q
    # x forward position, z down position
    x: float = 0.0
    z: float = 0.0
    u: float = 60.0           # forward body velocity
    w: float = 0.0            # vertical body velocity (down)
    theta: float = 0.0        # pitch angle
    q: float = 0.0            # pitch rate


class AircraftModel:
    def __init__(self, p: AircraftParams):
        self.p = p

    def forces_moments(self, s: State, c: Control) -> Tuple[np.ndarray, float]:
        """
        Returns body-axis forces [X, Z] (Z positive down) and pitch moment M about y.
        Toy aero model: CL(alpha), CD(CL), simple elevator pitch moment.
        """
        p = self.p

        V = math.sqrt(s.u*s.u + s.w*s.w)
        V = max(V, 1e-3)

        alpha = math.atan2(s.w, s.u)  # rad

        qbar = 0.5 * p.rho * V*V
        CL = p.CL0 + p.CLa * alpha
        CD = p.CD0 + p.k * (CL**2)

        L = qbar * p.S * CL          # lift (perp to V)
        D = qbar * p.S * CD          # drag (opposite to V)

        # Resolve aerodynamic forces to body axes (X forward, Z down)
        # Using alpha: wind to body rotation
        Xa = -D * math.cos(alpha) + -L * math.sin(alpha)
        Za = -D * math.sin(alpha) +  L * math.cos(alpha)  # down positive

        # Thrust along body X
        T = p.Tmax * float(np.clip(c.throttle, 0.0, 1.0))

        # Weight in body axes: rotate inertial down (g) into body frame by theta
        # In inertial: weight = [0, mg] down (z down). Convert to body:
        Xw = -p.mass * p.g * math.sin(s.theta)
        Zw =  p.mass * p.g * math.cos(s.theta)

        X = Xa + T + Xw
        Z = Za + Zw

        # Toy pitch moment: elevator + damping
        # (later: use Cm(alpha, q, de))
        M = -4000.0 * c.elevator - 800.0 * s.q

        return np.array([X, Z], dtype=float), float(M)

    def dynamics(self, s: State, c: Control) -> np.ndarray:
        """
        Returns time-derivative of state vector [x,z,u,w,theta,q]
        """
        p = self.p
        F, M = self.forces_moments(s, c)

        X, Z = F[0], F[1]

        # Translational equations in body frame (2D)
        du = X / p.mass + s.q * s.w
        dw = Z / p.mass - s.q * s.u

        # Kinematics: convert body velocity to inertial rates
        # inertial x forward, z down; rotate by theta
        dx =  math.cos(s.theta) * s.u - math.sin(s.theta) * s.w
        dz =  math.sin(s.theta) * s.u + math.cos(s.theta) * s.w

        dtheta = s.q
        dq = M / p.Iy

        return np.array([dx, dz, du, dw, dtheta, dq], dtype=float)


def rk4_step(model: AircraftModel, s: State, c: Control, dt: float) -> State:
    y0 = np.array([s.x, s.z, s.u, s.w, s.theta, s.q], dtype=float)

    def f(y: np.ndarray) -> np.ndarray:
        ss = State(*y.tolist())
        return model.dynamics(ss, c)

    k1 = f(y0)
    k2 = f(y0 + 0.5*dt*k1)
    k3 = f(y0 + 0.5*dt*k2)
    k4 = f(y0 + dt*k3)

    y1 = y0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return State(*y1.tolist())


class Logger:
    def __init__(self):
        self.rows: List[Dict[str, float]] = []

    def log(self, t: float, s: State, c: Control):
        V = math.sqrt(s.u*s.u + s.w*s.w)
        alpha = math.atan2(s.w, s.u)
        self.rows.append({
            "t": t,
            "x": s.x,
            "z": s.z,
            "u": s.u,
            "w": s.w,
            "V": V,
            "theta": s.theta,
            "q": s.q,
            "alpha": alpha,
            "throttle": c.throttle,
            "elevator": c.elevator,
        })


def main():
    cfg = SimConfig(dt=0.01, t_final=30.0, real_time=True, log_hz=20)
    p = AircraftParams()
    model = AircraftModel(p)
    s = State(x=0.0, z=0.0, u=60.0, w=0.0, theta=0.0, q=0.0)
    c = Control(throttle=0.5, elevator=0.0)

    logger = Logger()

    t = 0.0
    next_log_t = 0.0
    log_dt = 1.0 / cfg.log_hz

    wall_start = time.perf_counter()
    last_wall = wall_start

    while t < cfg.t_final:
        # Example: simple “pilot input” schedule
        if 3.0 < t < 6.0:
            c.elevator = math.radians(-2.0)  # pull up (negative elevator in this toy sign)
        else:
            c.elevator = 0.0

        s = rk4_step(model, s, c, cfg.dt)
        t += cfg.dt

        if t >= next_log_t:
            logger.log(t, s, c)
            next_log_t += log_dt

        if cfg.real_time:
            # pace to real time
            target = wall_start + t
            now = time.perf_counter()
            sleep_s = target - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            last_wall = now

    # Print a small summary
    last = logger.rows[-1]
    print("Final:", {k: round(v, 3) for k, v in last.items() if k in ["t","x","z","V","theta","alpha"]})
    print(f"Logged {len(logger.rows)} samples.")


if __name__ == "__main__":
    main()
