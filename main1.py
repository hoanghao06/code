import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Load data
data_ppo = pd.read_csv(r'C:\Users\dohuy\Desktop\2026. Globecom\output_rural_10\speed_10\0\episode_rewards.csv') 
data_dqn = pd.read_csv(r'C:\Users\dohuy\Desktop\2026. Globecom\episode_rewards.csv') 

episodes_ppo = data_ppo['Episode'].values
rewards_SS = data_ppo['Reward'].values
rewards_HS = data_dqn['reward'].values

# Smooth PPO
window_size = 1
rewards_SS = moving_average(rewards_SS, window_size)

# Align data
episodes_ppo = episodes_ppo[:len(rewards_SS)]
rewards_HS = rewards_HS[:len(episodes_ppo)]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(episodes_ppo, rewards_SS, label="PPO", color='red', linewidth=2)
plt.plot(episodes_ppo, rewards_HS, label="DQN", color='blue', linewidth=2)

plt.xlabel("Episodes", fontsize=16)
plt.ylabel("Reward", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
