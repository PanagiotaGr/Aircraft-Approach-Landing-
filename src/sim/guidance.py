import numpy as np

deg = np.pi / 180
g = 9.81


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ILSGuidance:
    def __init__(self, gains: dict, approach: dict, limits: dict, refs: dict, aircraft: dict):
        # Lateral gains (kept for compatibility even if not used directly)
        self.K_y_P = gains["K_y_P"]
        self.K_y_D = gains["K_y_D"]
        self.tau_heading = gains["tau_heading"]

        # Vertical + speed gains
        self.K_h_P = gains["K_h_P"]
        self.K_h_D = gains["K_h_D"]
        self.K_V_P = gains["K_V_P"]

        # Approach geometry
        self.gs_deg = approach["glideslope_deg"]
        self.h_thresh = approach["h_threshold"]
        self.x_thresh = approach["x_threshold"]

        # Limits / references
        self.phi_max = np.deg2rad(limits["phi_deg_max"])
        self.V_ref = refs["V_ref"]

    def h_ref(self, x: float) -> float:
        """
        Reference height along glideslope that continues past threshold (so it can reach 0m).
        """
        gs = self.gs_deg * deg
        d = self.x_thresh - x  # allow negative after threshold
        return max(0.0, self.h_thresh + np.tan(gs) * d)

    def compute(self, state: dict, rates: dict):
        x = state["x"]
        y = state["y"]
        h = state["h"]
        V = state["V"]
        psi = state["psi"]
        throttle = state["throttle"]

        y_dot = rates["y_dot"]
        h_dot = rates["h_dot"]

        # -------------------------
        # LATERAL (robust localizer via yaw-rate command)
        # -------------------------
        # Drive y -> 0 and psi -> 0 (runway heading assumed 0 rad)
        k_y = 1.0 / 120.0   # [1/s] per meter
        k_d = 1.0 / 25.0    # damping on y_dot
        k_psi = 1.0 / 1.0   # heading stabilization

        psi_dot_cmd = -(k_y * y + k_d * y_dot + k_psi * wrap_pi(psi))

        # Convert yaw-rate to bank (coordinated turn)
        phi_cmd = np.arctan2(psi_dot_cmd * max(V, 1.0), g)
        phi_cmd = float(np.clip(phi_cmd, -self.phi_max, self.phi_max))

        # -------------------------
        # VERTICAL (glideslope + flare after threshold)
        # -------------------------
        href = self.h_ref(x)
        e_h = h - href

        gamma_gs = -self.gs_deg * deg

        # Stronger vertical correction (you were ~+5-10m high before)
        # Stronger correction BEFORE threshold to hit h_thresh accurately
        pre = 2.0 if x < self.x_thresh else 1.0

        gamma_cmd = gamma_gs - (pre * 4.0 * self.K_h_P * e_h + pre * 2.0 * self.K_h_D * h_dot)

        # Slight down-bias to remove persistent high bias at threshold
        gamma_cmd -= 0.7 * deg

        # Flare ONLY after passing threshold
        flare_h = 80.0
        if x >= self.x_thresh and h < flare_h:
            gamma_cmd = np.interp(h, [0.0, flare_h], [0.0, gamma_cmd])

        # Allow enough descent authority to actually get to the ground
        gamma_cmd = float(np.clip(gamma_cmd, -20 * deg, 3 * deg))

        # -------------------------
        # SPEED
        # -------------------------
        eV = self.V_ref - V
        throttle_cmd = throttle + self.K_V_P * eV
        throttle_cmd = float(np.clip(throttle_cmd, 0.1, 1.0))

        return phi_cmd, gamma_cmd, throttle_cmd, href

