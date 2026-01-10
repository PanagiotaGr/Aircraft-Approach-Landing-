import numpy as np

g = 9.81

class PointMass3DOF:
    def __init__(self, state: dict, aircraft: dict, limits: dict):
        self.x = state["x"]
        self.y = state["y"]
        self.h = state["h"]
        self.V = state["V"]
        self.psi = state["psi"]
        self.gamma = state["gamma"]
        self.phi = state["phi"]
        self.throttle = float(state["throttle"])
        self.n = float(state["n"])

        self.m = float(aircraft["mass"])
        self.S = float(aircraft["S"])
        self.rho = float(aircraft["rho"])
        self.CD0 = float(aircraft["CD0"])
        self.k = float(aircraft["k"])
        self.Tmax = float(aircraft["Tmax"])

        self.thr_min = float(aircraft["throttle_min"])
        self.thr_max = float(aircraft["throttle_max"])

        self.phi_max = np.deg2rad(float(limits["phi_deg_max"]))
        self.phi_rate = np.deg2rad(float(limits["phi_rate_deg_s"]))

        self.n_min = float(limits["n_min"])
        self.n_max = float(limits["n_max"])
        self.n_rate = float(limits["n_rate"])

        self.gamma_min = np.deg2rad(float(limits["gamma_deg_min"]))
        self.gamma_max = np.deg2rad(float(limits["gamma_deg_max"]))

        # Hard clamp initial values
        self.throttle = float(np.clip(self.throttle, self.thr_min, self.thr_max))
        self.n = float(np.clip(self.n, self.n_min, self.n_max))
        self.gamma = float(np.clip(self.gamma, self.gamma_min, self.gamma_max))
        self.phi = float(np.clip(self.phi, -self.phi_max, self.phi_max))

    def aero_forces(self):
        # Lift via load factor
        L = self.n * self.m * g

        q = 0.5 * self.rho * self.V**2
        qS = max(q * self.S, 1e-6)

        CL = L / qS
        CD = self.CD0 + self.k * (CL**2)
        D = qS * CD

        T = self.throttle * self.Tmax
        return T, D

    def step(self, dt: float, phi_cmd: float, gamma_cmd: float, throttle_cmd: float, wind_xy):
        # --- Commands: clamp ---
        phi_cmd = float(np.clip(phi_cmd, -self.phi_max, self.phi_max))
        gamma_cmd = float(np.clip(gamma_cmd, self.gamma_min, self.gamma_max))
        throttle_cmd = float(np.clip(throttle_cmd, self.thr_min, self.thr_max))

        # --- Actuator-like rate limits ---
        dphi = np.clip(phi_cmd - self.phi, -self.phi_rate*dt, self.phi_rate*dt)
        self.phi = float(np.clip(self.phi + dphi, -self.phi_max, self.phi_max))

        # throttle (rate limited) + HARD clamp after update
        dthr = np.clip(throttle_cmd - self.throttle, -0.8*dt, 0.8*dt)
        self.throttle = float(np.clip(self.throttle + dthr, self.thr_min, self.thr_max))

        # --- Load factor control to track gamma_cmd ---
        tau_g = 1.0
        gamma_dot_cmd = (gamma_cmd - self.gamma) / max(tau_g, 0.2)

        Veff = max(self.V, 1.0)
        n_cmd = ((gamma_dot_cmd * Veff) / g + np.cos(self.gamma)) / max(np.cos(self.phi), 0.2)
        n_cmd = float(np.clip(n_cmd, self.n_min, self.n_max))

        dn = np.clip(n_cmd - self.n, -self.n_rate*dt, self.n_rate*dt)
        self.n = float(np.clip(self.n + dn, self.n_min, self.n_max))

        # --- Forces ---
        T, D = self.aero_forces()

        # --- 3DOF EOM ---
        V_dot = (T - D)/self.m - g*np.sin(self.gamma)
        self.V = float(np.clip(self.V + V_dot*dt, 45.0, 110.0))

        psi_dot = (g / max(self.V, 1.0)) * self.n * np.sin(self.phi) / max(np.cos(self.gamma), 0.2)
        self.psi = float(self.psi + psi_dot*dt)

        gamma_dot = (self.n*g*np.cos(self.phi))/max(self.V, 1.0) - (g*np.cos(self.gamma))/max(self.V, 1.0)
        self.gamma = float(np.clip(self.gamma + gamma_dot*dt, self.gamma_min, self.gamma_max))

        # Kinematics + wind
        Vwx, Vwy = wind_xy
        Vx = self.V*np.cos(self.gamma)*np.cos(self.psi) + Vwx
        Vy = self.V*np.cos(self.gamma)*np.sin(self.psi) + Vwy
        Vh = self.V*np.sin(self.gamma)

        self.x += Vx*dt
        self.y += Vy*dt
        self.h += Vh*dt

        return {
            "x": self.x, "y": self.y, "h": self.h,
            "V": self.V, "psi": self.psi, "gamma": self.gamma,
            "phi": self.phi, "throttle": self.throttle, "n": self.n
        }

