import numpy as np

deg = np.pi / 180
g = 9.81

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class ILSGuidance:
    def __init__(self, gains: dict, approach: dict, limits: dict, refs: dict, aircraft: dict):
        self.K_y_P = gains["K_y_P"]
        self.K_y_D = gains["K_y_D"]
        self.tau_heading = gains["tau_heading"]

        self.K_h_P = gains["K_h_P"]
        self.K_h_D = gains["K_h_D"]
        self.tau_gamma = gains["tau_gamma"]

        self.K_V_P = gains["K_V_P"]

        self.gs_deg = approach["glideslope_deg"]
        self.h_thresh = approach["h_threshold"]
        self.x_thresh = approach["x_threshold"]

        self.phi_max = np.deg2rad(limits["phi_deg_max"])
        self.V_ref = refs["V_ref"]

    def h_ref(self, x):
        gs = self.gs_deg * deg
        return self.h_thresh + np.tan(gs) * (self.x_thresh - x)

    def compute(self, state: dict, rates: dict):
        x, y, h, V = state["x"], state["y"], state["h"], state["V"]
        psi = state["psi"]
        throttle = state["throttle"]

        y_dot = rates["y_dot"]
        h_dot = rates["h_dot"]

        # -------------------------
        # LATERAL (runway capture)
        # Aim at the runway threshold (0,0) so we don't drift past it.
        # Bearing from current position to threshold:
        chi_to_thr = np.arctan2(-y, -x)   # target heading (ground-referenced)
        # Localizer correction (small offset)
        psi_loc = np.clip(self.K_y_P * (-y) - self.K_y_D * y_dot, -10*deg, 10*deg)

        psi_cmd = chi_to_thr + psi_loc
        psi_err = wrap_pi(psi_cmd - psi)

        psi_dot_cmd = psi_err / max(self.tau_heading, 0.2)
        phi_cmd = np.arctan2(psi_dot_cmd * max(V, 1.0), g)
        phi_cmd = float(np.clip(phi_cmd, -self.phi_max, self.phi_max))

        # -------------------------
        # VERTICAL (glideslope)
        # -------------------------
        # VERTICAL (glideslope)
        href = self.h_ref(x)
        e_h = href - h

        # Desired descent angle from geometry (towards the threshold on the slope)
        # gamma_gs is negative (descending)
        gamma_gs = -np.arctan2(max(h - self.h_thresh, 0.0), max(self.x_thresh - x, 1.0))

        # Add correction around the slope (PD in altitude error)
        gamma_cmd = gamma_gs - (self.K_h_P * e_h - self.K_h_D * h_dot)

        gamma_cmd = float(np.clip(gamma_cmd, -8*deg, 2*deg))

        # -------------------------
        # SPEED (throttle)
        eV = self.V_ref - V
        throttle_cmd = throttle + self.K_V_P * eV
        throttle_cmd = float(np.clip(throttle_cmd, 0.05, 1.0))

        return phi_cmd, gamma_cmd, throttle_cmd, href
