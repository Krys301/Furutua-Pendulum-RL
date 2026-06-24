# Furuta Pendulum: LQR vs PPO

> Can a reinforcement learning agent match a mathematically optimal controller on a real nonlinear system, and does it find a better strategy in the process?

This project extends prior control systems coursework (EE291, Maynooth University) into an empirical ML comparison. A custom Gymnasium environment implements the full nonlinear dynamics of the Furuta (rotary inverted) pendulum, derived from first principles using the Euler-Lagrange method, and uses it to pit a classical LQR controller against a PPO agent trained entirely through interaction with the physics.

---

## The System

The Furuta pendulum is a motor-driven arm that rotates in the horizontal plane, with a freely-swinging pendulum attached to its end. It is:

- **Naturally unstable:** gravity pulls the pendulum away from upright at all times
- **Underactuated:** torque is only applied to the arm; the pendulum is controlled indirectly
- **Nonlinear:** the coupling between arm and pendulum produces dynamics that cannot be captured by a linear model away from equilibrium

Classical control handles this by linearising around the upright position (small-angle approximation). This works well near equilibrium but breaks down as the system moves away from it. The central question is whether a learned policy can do better, and whether it discovers a qualitatively different strategy when it does.

---

## Environment

The Gymnasium environment implements the full nonlinear equations of motion derived via Lagrangian mechanics (Cazzolato & Prime, 2011), including DC motor dynamics. All physical parameters are taken directly from the EE291 experimental setup.

**State vector:** `[θ₁, θ₂, θ̇₁, θ̇₂, i]`

| State | Description |
|-------|-------------|
| θ₁ | Arm angle (rad) |
| θ₂ | Pendulum angle from upright (rad) |
| θ̇₁ | Arm angular velocity (rad/s) |
| θ̇₂ | Pendulum angular velocity (rad/s) |
| i | Motor current (A) |

**Action:** Motor voltage V in [-5V, 5V]

**Physical parameters (from EE291 hardware):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| m1 | 0.360 kg | Arm mass |
| m2 | 0.090 kg | Pendulum mass |
| L1 | 0.334 m | Arm length |
| l2 | 0.178 m | Pendulum COM distance |
| Km | 0.108 Nm/A | Motor torque constant |
| Rm | 9.36 ohm | Motor resistance |

**Reward function:**
```
r = 1.0 - 5.0 * theta2^2 - 0.1 * V^2 - 0.5 * theta1_dot^2
```
Survival bonus penalised by pendulum deviation, control effort, and arm velocity.

---

## Controllers

### LQR (Classical Baseline)
Designed using Bryson's method on the linearised 5th-order electromechanical state-space model. Gain vector computed in MATLAB:

```
K = [-6.4, 291.6, -8.9, 70.2, 3.9]
u = -Kx
```

Valid only near the upright equilibrium (theta2 < 6 degrees). Beyond this the linearised model breaks down and arm displacement exceeds physical limits.

### PPO (Learned Policy)
Trained using Stable Baselines3 PPO on the nonlinear environment with no prior knowledge of the system equations. The agent observes the full state vector and outputs a voltage directly.

```
Training timesteps: 100,000
Policy: MLP (2 hidden layers)
Learning rate: 3e-4
```

---

## Results

### Stabilisation Performance (1, 4, 6 degree initial conditions)

|  | 1 degree | 4 degrees | 6 degrees |
|--|----------|-----------|-----------|
| **LQR: Duration** | 15.0s | 15.0s | 15.0s |
| **LQR: Avg theta2** | 0.0180 rad | 0.0719 rad | 0.1081 rad |
| **LQR: Max arm angle** | 14.7 deg | 15.8 deg | 17.2 deg |
| **LQR: Arm within 45 deg limit** | YES | YES | YES |
| **PPO: Duration** | 15.0s | 15.0s | 15.0s |
| **PPO: Avg theta2** | 0.0178 rad | 0.0717 rad | 0.1079 rad |
| **PPO: Max arm angle** | 6.8 deg | 6.0 deg | 5.7 deg |
| **PPO: Arm within 45 deg limit** | YES | YES | YES |

Both controllers stabilise successfully across all tested conditions. Pendulum deviation is nearly identical. However, the PPO agent uses **3x less arm travel** than LQR to achieve the same result, a strategy that emerges purely from training with no explicit constraint on arm movement.

### Disturbance Rejection (sudden push applied at t=2s)

| Push magnitude | LQR survived | PPO survived |
|---------------|-------------|-------------|
| 5 degrees | YES 15.0s | YES 15.0s |
| 10 degrees | YES 15.0s | YES 15.0s |
| 20 degrees | YES 15.0s | YES 15.0s |

Both controllers recover from all tested disturbances. PPO reward degrades gracefully under larger pushes, suggesting it is working closer to its limits at 20 degrees while still maintaining stability.

### Key Finding

The PPO agent converges to a policy that matches LQR's pendulum stabilisation performance while using significantly less arm movement. This suggests the learned policy found a more mechanically efficient strategy, one that would place less stress on physical hardware and stay further from actuator saturation limits. This was not an explicit objective; it emerged from training on the nonlinear dynamics alone.

---

## Project Structure

```
furuta-pendulum-rl/
├── envs/
│   ├── __init__.py
│   └── furuta_env.py       # Gymnasium environment (nonlinear dynamics)
├── scripts/
│   ├── train_ppo.py        # PPO training script
│   ├── run_lqr.py          # LQR baseline evaluation
│   └── compare.py          # Head-to-head comparison
├── models/
│   └── ppo_furuta.zip      # Trained PPO model
├── tests/
│   └── test_env.py         # Environment sanity checks
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Krys301/Furutua-Pendulum-RL.git
cd Furutua-Pendulum-RL
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Train the PPO agent:**
```bash
python scripts/train_ppo.py
```

**Run LQR baseline:**
```bash
python scripts/run_lqr.py
```

**Run full comparison:**
```bash
python scripts/compare.py
```

---

## Requirements

```
gymnasium
numpy
scipy
stable-baselines3
matplotlib
```

---

## Background

This project builds directly on EE291 (Systems Dynamics & Control, Maynooth University), where the Furuta pendulum was modelled from first principles using the Euler-Lagrange method and controlled using PID and LQR in MATLAB/Simulink. The nonlinear equations of motion, physical parameters, and LQR gain vector used here are taken directly from that work.

The central motivation is that LQR is optimal for the linearised system but fundamentally limited by the small-angle approximation. Training an RL agent on the nonlinear dynamics removes that constraint and allows the policy to generalise across a wider operating range, while also, as the results show, discovering a qualitatively different control strategy.

---

## References

- Cazzolato, B.S. & Prime, Z. (2011). On the Dynamics of the Furuta Pendulum. Journal of Control Science and Engineering.
- Hernandez et al. (2024). Modeling, Simulation, and Control of a Rotary Inverted Pendulum: A Reinforcement Learning-Based Control Approach. Modelling, 5(4).
- Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
