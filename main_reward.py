import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ================== FONT SIZE CONFIG ==================
FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICKS = 12
FONT_LEGEND = 12

# ================== LOAD DATA ==================
data_1 = pd.read_csv(r'C:\Users\AVSTC\Desktop\2026.Globecom\output_rural_10\speed_10\0\episode_rewards.csv')
data_2 = pd.read_csv(r'C:\Users\AVSTC\Desktop\2026.Globecom\output_urban_10\speed_10\0\episode_rewards.csv')

episodes_1 = data_1['Episode']
episodes_2 = data_2['Episode']

rewards_raw_1 = data_1['Reward']
rewards_raw_2 = data_2['Reward']

# ================== SMOOTH ==================
window_size = 1
rewards_smooth_1 = moving_average(rewards_raw_1, window_size)
rewards_smooth_2 = moving_average(rewards_raw_2, window_size)

# ================== PLOT ==================
plt.figure(figsize=(10, 5))

# RAW (nhạt)
# plt.plot(episodes_1, rewards_raw_1, color='red', alpha=0.2)
# plt.plot(episodes_2, rewards_raw_2, color='blue', alpha=0.2)

# SMOOTH (đậm)
plt.plot(episodes_1[:len(rewards_smooth_1)], rewards_smooth_1,
         label="vehicle velocity 10 m/s", color='red', linewidth=2)

plt.plot(episodes_2[:len(rewards_smooth_2)], rewards_smooth_2,
         label="vehicle velocity 15 m/s", color='blue', linewidth=2)

# LABELS
plt.xlabel("Episodes", fontsize=FONT_LABEL)
plt.ylabel("Reward", fontsize=FONT_LABEL)
# plt.title("Training Reward", fontsize=FONT_TITLE)

# TICKS
plt.xticks(fontsize=FONT_TICKS)
plt.yticks(fontsize=FONT_TICKS)

# LEGEND
plt.legend(fontsize=FONT_LEGEND)

plt.grid(True)
plt.tight_layout()
plt.show()