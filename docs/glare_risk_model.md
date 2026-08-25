# Glare Risk Model & Mathematical Formulation

## 1. Overview

The Glare Risk Model quantifies the risk of optical glare experienced by oncoming or preceding drivers as a score $R \in [0, 100]$.

## 2. Mathematical Formulation

### Distance Penalty ($S_{dist}$)
Inverse quadratic dropoff with distance threshold $d_{safe} = 150\text{m}$:
$$S_{dist} = \max\left(0, 100 \times \left(1 - \frac{d}{d_{safe}}\right)^2\right)$$

### Speed Penalty ($S_{speed}$)
Higher closing relative velocities shorten reaction windows:
$$S_{speed} = \min\left(1.5, 1.0 + \frac{v_{rel}}{100}\right)$$

### Lane Role Weights ($W_{lane}$)
- Oncoming traffic: $W_{lane} = 1.0$
- Same lane ahead (preceding vehicle): $W_{lane} = 0.8$
- Adjacent lane: $W_{lane} = 0.5$
- Parked / Off-road: $W_{lane} = 0.2$

### Weather Modifier ($\eta_{weather}$)
Atmospheric scattering (fog/rain/snow) redirects high-beam photons back to ego-driver and surrounding traffic:
$$\eta_{weather} = \begin{cases}
1.0 & \text{Clear} \\
1.25 & \text{Rain} \\
1.4 & \text{Fog} \\
1.2 & \text{Snow}
\end{cases}$$

### Total Composite Glare Risk Score
$$R_{glare} = \min\left(100, S_{dist} \times S_{speed} \times W_{lane} \times \eta_{weather}\right)$$

## 3. Spatial Matrix Zone Power Mapping

Target brightness $B_i$ for zone $i$ with risk $R_i$:
$$B_i = B_{min} + (B_{max} - B_{min}) \times \left(1 - \left(\frac{R_i}{100}\right)^\gamma\right)$$
Where:
- $B_{min} = 10$ (minimum safety illumination)
- $B_{max} = 255$ (full PWM power)
- $\gamma = 1.5$ (gamma power curve response)
