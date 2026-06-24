import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.furuta_env import FurutaEnv
from stable_baselines3 import PPO

# LQR gain vector from EE291 report
K = np.array([-6.4, 291.6, -8.9, 70.2, 3.9])

def lqr_action(state):
    V = -K @ state
    return np.clip(V, -5.0, 5.0)

def run_episode(controller, initial_angle_deg, max_steps=1500):
    env = FurutaEnv()
    obs, _ = env.reset()
    obs[1] = np.deg2rad(initial_angle_deg)
    env.state = obs.copy()

    total_reward = 0
    steps = 0
    theta2_history = []
    theta1_history = []

    for _ in range(max_steps):
        if controller == "lqr":
            V = lqr_action(obs)
            action = np.array([V], dtype=np.float32)
        else:
            action, _ = controller.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        theta2_history.append(abs(obs[1]))
        theta1_history.append(abs(obs[0]))

        if terminated or truncated:
            break

    avg_deviation = np.mean(theta2_history)
    max_deviation = np.max(theta2_history)
    max_arm = np.max(theta1_history)
    return steps, total_reward, avg_deviation, max_deviation, max_arm


def run_disturbance_episode(controller, disturbance_deg, max_steps=1500):
    env = FurutaEnv()
    obs, _ = env.reset()
    env.state = obs.copy()

    total_reward = 0
    steps = 0
    disturbance_step = 200

    for step in range(max_steps):
        if step == disturbance_step:
            env.state[1] += np.deg2rad(disturbance_deg)
            obs = env.state.copy()

        if controller == "lqr":
            V = lqr_action(obs)
            action = np.array([V], dtype=np.float32)
        else:
            action, _ = controller.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return steps, total_reward


# Load trained PPO model
ppo_model = PPO.load("models/ppo_furuta")

# ── Main comparison ──────────────────────────────────────────────────
print("=" * 70)
print(f"{'Comparison: LQR vs PPO on Furuta Pendulum':^70}")
print(f"{'EE291 Maynooth University — Nonlinear Environment':^70}")
print("=" * 70)
print(f"{'':22} {'1 degree':^14} {'4 degrees':^14} {'6 degrees':^14}")
print("-" * 70)

for controller_name, controller in [("LQR (from report)", "lqr"),
                                     ("PPO (learned)", ppo_model)]:
    results = []
    for angle in [1, 4, 6]:
        steps, reward, avg_dev, max_dev, max_arm = run_episode(controller, angle)
        results.append((steps, reward, avg_dev, max_dev, max_arm))

    print(f"\n{controller_name}")
    print(f"  Steps survived:      "
          + "   ".join([f"{r[0]:>8}" for r in results]))
    print(f"  Duration (s):        "
          + "   ".join([f"{r[0]*0.01:>8.1f}" for r in results]))
    print(f"  Avg |theta2| (rad):  "
          + "   ".join([f"{r[2]:>8.4f}" for r in results]))
    print(f"  Max |theta2| (rad):  "
          + "   ".join([f"{r[3]:>8.4f}" for r in results]))
    print(f"  Max |theta1| (deg):  "
          + "   ".join([f"{np.rad2deg(r[4]):>8.1f}" for r in results]))
    print(f"  Arm within 45 deg:   "
          + "   ".join([f"{'YES' if np.rad2deg(r[4]) <= 45 else 'NO':>8}"
                        for r in results]))
    print(f"  Total reward:        "
          + "   ".join([f"{r[1]:>8.1f}" for r in results]))

# ── Disturbance test ─────────────────────────────────────────────────
print("\n\nDisturbance Test — Sudden Push Applied at t=2s")
print("=" * 70)
print(f"{'':22} {'5 deg push':^14} {'10 deg push':^14} {'20 deg push':^14}")
print("-" * 70)

for controller_name, controller in [("LQR (from report)", "lqr"),
                                     ("PPO (learned)", ppo_model)]:
    results = []
    for disturbance in [5, 10, 20]:
        steps, reward = run_disturbance_episode(controller, disturbance)
        results.append((steps, reward))

    print(f"\n{controller_name}")
    print(f"  Steps survived:      "
          + "   ".join([f"{r[0]:>8}" for r in results]))
    print(f"  Duration (s):        "
          + "   ".join([f"{r[0]*0.01:>8.1f}" for r in results]))
    print(f"  Total reward:        "
          + "   ".join([f"{r[1]:>8.1f}" for r in results]))

print("\n" + "=" * 70)