import matplotlib.pyplot as plt
import numpy as np

def plot_results(log):
    x = np.array(log["x"])
    y = np.array(log["y"])
    h = np.array(log["h"])
    href = np.array(log["href"])
    V = np.array(log["V"])
    phi = np.array(log["phi"])
    gamma = np.array(log["gamma"])
    thr = np.array(log["throttle"])

    plt.figure()
    plt.plot(x, y)
    plt.axhline(0, linestyle="--")
    plt.gca().invert_xaxis()
    plt.xlabel("x (m) -> threshold")
    plt.ylabel("y (m)")
    plt.title("Ground track (Localizer)")
    plt.grid(True)

    plt.figure()
    plt.plot(x, h, label="h")
    plt.plot(x, href, "--", label="h_ref")
    plt.gca().invert_xaxis()
    plt.xlabel("x (m)")
    plt.ylabel("h (m)")
    plt.title("Glideslope tracking")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(V)
    plt.title("Speed V (m/s)")
    plt.grid(True)

    plt.figure()
    plt.plot(np.rad2deg(phi), label="phi (deg)")
    plt.plot(np.rad2deg(gamma), label="gamma (deg)")
    plt.plot(thr, label="throttle")
    plt.title("Controls / states")
    plt.legend()
    plt.grid(True)

    plt.show()
