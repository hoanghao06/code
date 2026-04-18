# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# import pandas as pd

# data_HS= pd.read_csv(r'C:\Users\dinhk\Desktop\UAV_FSO_RF_DRL-main\output\speed_10\4\episode_rewards.csv') # reward chuyển mạch cứng
# # data_SS= pd.read_csv(r'C:\Users\dinhk\Desktop\UAV_FSO_RF_DRL-main\output\speed_10\0\episode_rewards.csv') # reward chuyển mạch mềm
# episodes = data_HS['Episode Number']
# rewards_HS = data_HS['Episode Reward']
# # rewards_SS = data_SS['Episode Reward']
# # Làm mượt bằng Savitzky-Golay (window_size = 11, poly_order = 2)
# rewards_HS = savgol_filter(rewards_HS, window_length=15, polyorder=2)
# # rewards_SS = savgol_filter(rewards_SS, window_length=11, polyorder=2)

# # Vẽ đồ thị
# plt.figure(figsize=(10, 5))

# plt.plot(episodes, rewards_HS, label="rewards chuyển mạch cứng", color='red', linewidth=2)
# # plt.plot(episodes, rewards_SS, label="rewards chuyển mạch mềm", color='red', linewidth=2)
# plt.xlabel("Episodes")
# plt.ylabel("Reward")
# plt.title("Reward")
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def moving_average(data, window_size):
    """Hàm tính trung bình trượt"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


data_SS= pd.read_csv(r'C:\Users\AVSTC\Desktop\2026.Globecom\output\speed_10\0\episode_rewards.csv') # reward chuyển mạch mềm
# data_HS= pd.read_csv(r'C:\Users\dohuy\Desktop\UAV_FSO_THz\UAV_FSO_RF_DRL-main\output 1.7 HS\speed_10\0\episode_rewards.csv') # reward chuyển mạch cứng
episodes = data_SS['Episode']
rewards_SS = data_SS['Reward']
# Làm mượt với cửa sổ trượt 
window_size = 20
rewards_SS = moving_average(rewards_SS, window_size)


# Vẽ đồ thị
plt.figure(figsize=(10, 5))
# plt.plot(episodes, rewards, label="Raw Reward", alpha=0.5)
plt.plot(episodes[:len(rewards_SS)], rewards_SS, label="Rewards của chuyển mạch mềm", color='red', linewidth=2)
# plt.plot(episodes[:len(rewards_HS)], rewards_HS, label="Rewards của chuyển mạch cứng", color='blue', linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward của 2 cơ chế chuyển mạch")
plt.legend()
plt.grid(True)
plt.show()
