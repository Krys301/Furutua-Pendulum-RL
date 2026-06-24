import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.furuta_env import FurutaEnv

# Your exact LQR gain vector from the EE291 report
K = np.array([-6.4, 291.6, -8.9, 70.2, 3.9])

def lqr_action(state):
    """u = -Kx  — full state feedback"""
    V = -K @ state
    # Saturate to motor limits exactly as in your Simulink model
    V = np.clip(V, -5.0, 5.0)
    return np.array([V], dtype=np.float32)

def run_lqr_episode(initial_angle_deg):
    env = FurutaEnv()
    obs, _ = env.reset()

    # Set specific initial condition to match your report tests
    obs[1] = np.deg2rad(initial_angle_deg)  # theta2
    env.state = obs

    total_reward = 0
    steps = 0
    max_steps = 1500  # 15 seconds at 10ms per step

    for _ in range(max_steps):
        action = lqr_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return steps, total_reward

# Test at same initial conditions as your report: 1, 4, 6 degrees
print("LQR Controller Results")
print("=" * 45)

for angle in [1, 4, 6]:
    steps, total_reward = run_lqr_episode(angle)
    duration = steps * 0.01
    print(f"Initial angle: {angle}° | Steps: {steps} | "
          f"Duration: {duration:.1f}s | Total reward: {total_reward:.1f}")