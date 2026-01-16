# Aircraft Approach and Landing Simulation

### 3-DOF Point-Mass Model with ILS-like Guidance

> This repository implements a simplified **aircraft approach and landing simulation** using a **3-DOF point-mass flight dynamics model** and an **ILS-like guidance law**.
> The project targets **university-level education and research prototyping**, emphasizing clear modeling assumptions, reproducibility, and interpretable performance metrics rather than high-fidelity aerodynamics.

---

## Table of Contents

* [Overview](#overview)
* [Modeling Assumptions](#modeling-assumptions)
* [Guidance Logic](#guidance-logic)
* [Disturbances and Environment](#disturbances-and-environment)
* [Visualization](#visualization)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Configuration](#configuration)
* [Use Cases](#use-cases)
* [Author](#author)

---

## Overview

The simulation models an aircraft executing an **instrument-style final approach** to a runway using:

* **Lateral guidance (localizer-like)** for runway centerline capture and tracking
* **Vertical guidance (glideslope-like)** for descent path capture and tracking
* **Wind disturbances**, including steady wind and gust components
* **Real-time visualization**, including runway perspective and cockpit-style instruments

The aircraft is modeled as a **3-DOF point mass**: only translational motion is simulated, while attitude dynamics are implicit and captured through commanded flight-path variables.

This design choice makes the simulator:

* computationally lightweight
* easy to interpret
* suitable for control and guidance algorithm development

---

## Modeling Assumptions

The aircraft dynamics follow a **3-DOF point-mass formulation**, assuming:

* no explicit rotational dynamics
* coordinated flight
* simplified force balance along the velocity vector

The system state evolves according to position and velocity variables, while control inputs are expressed through guidance-generated commands (e.g., lateral deviation correction and vertical path tracking).

This abstraction is commonly used in:

* flight guidance textbooks
* preliminary control-law validation
* educational simulations

---

## Guidance Logic

The approach is guided by an **ILS-inspired structure**:

### Lateral Guidance (Localizer-like)

* Tracks lateral deviation from the runway centerline
* Commands smooth corrective motion to align with the runway axis

### Vertical Guidance (Glideslope-like)

* Tracks deviation from a predefined descent path
* Ensures stable glide toward the runway threshold

The guidance laws are intentionally simple and interpretable, making them suitable for:

* parameter tuning studies
* robustness analysis
* comparison with more advanced guidance or control schemes

---

## Disturbances and Environment

The simulation includes environmental disturbances to test robustness:

* **Steady wind** (constant horizontal components)
* **Gust models** (time-varying disturbances)
* Configurable wind intensity and direction

These disturbances allow evaluation of:

* tracking performance
* stability margins
* guidance sensitivity under non-ideal conditions

---

## Visualization

The project includes real-time visualization tools:

* **Runway perspective view**
* **Cockpit-style instruments**
* **PAPI-like vertical deviation indicator**
* Smooth animation suitable for demos and presentations

Optional **video export** is supported (requires `ffmpeg`), enabling:

* offline analysis
* report-ready figures
* teaching material creation

---

## Features

* 3-DOF point-mass aircraft dynamics
* ILS-like guidance:

  * Localizer-inspired lateral guidance
  * Glideslope-inspired vertical guidance
* Wind and gust disturbance models
* Baseline simulation runner
* Cockpit visualization:

  * runway perspective
  * PAPI-style indication
  * flight instruments
  * smooth animation
* Optional video export

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
```

---

## Configuration

All key parameters are defined via YAML configuration files:

* aircraft parameters
* guidance gains
* wind profiles
* simulation time step and duration

This enables:

* reproducible experiments
* fast parameter sweeps
* clean separation between code and experimental setup

---

## Use Cases

This project is well suited for:

* flight dynamics and guidance coursework
* MSc-level control and simulation projects
* rapid prototyping of approach guidance laws
* visualization-driven teaching demonstrations
* baseline comparisons before high-fidelity simulation

---

## Author

**Panagiota Grosdouli**

---


* align this README stylistically with your **robotics & RL projects**
* prepare a **“Projects Portfolio” README** that unifies όλα αυτά σε research profile
