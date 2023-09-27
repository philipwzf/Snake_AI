import gym
from stable_baselines3 import A2C, SAC, PPO, TD3
import os

save_dir = "/tmp/gym/"
os.makedirs(save_dir, exist_ok=True)

model = PPO("MlpPolicy", "Pendulum-v1", verbose=0).learn(8000)
model.save(f"{save_dir}/PPO_tutorial")

obs = model.env.observation_space.sample()

print("pre saved", model.predict(obs, deterministic=True))
del model

loaded_model = PPO.load(f"{save_dir}/PPO_tutorial")
# Check that the prediction is the same after loading (for the same observation)
print("loaded", loaded_model.predict(obs, deterministic=True))


