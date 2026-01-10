import numpy as np

class WindModel:
    def __init__(self, Vwx=0.0, Vwy=0.0, gust_amp=0.0, gust_freq_hz=0.0,
                 random_gust_std=0.0, seed=0):
        self.Vwx0 = float(Vwx)
        self.Vwy0 = float(Vwy)
        self.gust_amp = float(gust_amp)
        self.w = 2.0 * np.pi * float(gust_freq_hz)
        self.random_gust_std = float(random_gust_std)
        self.rng = np.random.default_rng(int(seed))

    def sample(self, t: float):
        # Smooth gust (sinusoid) + small random component (white-noise-ish)
        gx = self.gust_amp * np.sin(self.w * t)
        gy = self.gust_amp * np.cos(self.w * t)

        rx = self.rng.normal(0.0, self.random_gust_std)
        ry = self.rng.normal(0.0, self.random_gust_std)

        return (self.Vwx0 + gx + rx), (self.Vwy0 + gy + ry)

