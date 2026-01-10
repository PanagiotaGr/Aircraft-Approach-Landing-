# Aircraft Approach and Landing Simulation

This repository implements a simplified aircraft approach and landing simulation using a **3-DOF point-mass flight dynamics model** and an **ILS-like guidance law**.  
The project is intended for **university-level educational use and research prototyping**, emphasizing clear modeling assumptions, reproducible simulation, and interpretable performance metrics rather than high-fidelity aerodynamics.

---

## Overview

The simulation models an aircraft executing an instrument-style final approach to a runway using:

- **Lateral guidance (localizer-like)** for runway centerline capture and tracking  
- **Vertical guidance (glideslope-like)** for descent path capture and tracking  
- **Wind disturbances**, including steady wind and gust components  
- **Real-time visualization**, including runway perspective and cockpit-style instruments  

The model is **3-DOF (point mass)**: translational motion is simulated while attitude dynamics are simplified and implicit through commanded flight-path variables.

---

## Features

- 3-DOF point-mass aircraft dynamics  
- ILS-like guidance:
  - Localizer-like lateral guidance
  - Glideslope-like vertical guidance
- Wind and gust models
- Baseline simulation runner
- Cockpit visualization:
  - Runway perspective view
  - PAPI-like indication
  - Flight instruments
  - Smooth animation
- Optional video export (requires `ffmpeg`)

---

## Repository Structure

```text
aircraft_sim/
├── src/
│   ├── sim/
│   │   ├── dynamics.py      # 3-DOF point-mass aircraft dynamics
│   │   ├── guidance.py      # ILS-like guidance (localizer & glideslope)
│   │   ├── wind.py          # Wind and gust models
│   │   └── metrics.py       # Performance metrics
│   ├── run_sim.py           # Baseline simulation execution
│   └── cockpit_view.py      # Cockpit visualization and animation
├── configs/
│   └── baseline.yaml        # Aircraft, guidance, wind, and simulation parameters
├── reports/
│   └── figures/             # Exported figures and videos
└── README.md






