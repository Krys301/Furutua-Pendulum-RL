import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.furuta_env import FurutaEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Verify environment is valid
env = FurutaEnv()
check_env(env)
print("Environment check passed.")

# Train PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

print("Starting training...")
model.learn(total_timesteps=100_000)

# Save the trained model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_furuta")
print("Model saved to models/ppo_furuta")