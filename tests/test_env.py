from envs.furuta_env import FurutaEnv

env = FurutaEnv()
obs, _ = env.reset()
print("Initial state:", obs)

for i in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step {i+1} | reward: {reward:.3f} | theta2: {obs[1]:.4f} | terminated: {terminated}")

    if terminated or truncated:
        print(f"Episode ended at step {i+1}, resetting...")
        obs, _ = env.reset()

print("Environment works.")