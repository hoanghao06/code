# ================== IMPORT ==================
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD DATA ==================
data = np.load(
    r'C:\Users\AVSTC\Desktop\2026.Globecom\output_rural_10\speed_10\0\flydata\rate_3.npy',
    allow_pickle=True
).item()

fso_rate = data['fso_rate']   # shape (time, UAV)

# ================== XỬ LÝ ==================
rate_mean = np.mean(fso_rate, axis=1)   # trung bình theo thời gian của 3 xe
overall_mean_rate = np.mean(fso_rate)  # trung bình tổng thể (tất cả thời gian + 3 xe)

x = np.arange(len(rate_mean))

# In các thông số
print("=== THỐNG KÊ TỪNG XE ===")
target_rate = 3
for i in range(fso_rate.shape[1]):
    vehicle_rate = fso_rate[:, i]
    mean_vehicle = vehicle_rate.mean()
    count = np.sum(vehicle_rate > target_rate)
    percentage = count / len(vehicle_rate)
    print(f"Vehicle {i+1}:")
    print(f"  - Tốc độ trung bình: {mean_vehicle:.4f}")
    print(f"  - Tỷ lệ > {target_rate} Mbps: {percentage:.2%}")
    print()

print("=== TỔNG HỢP 3 XE ===")
print(f"Tốc độ trung bình của tổng 3 xe (toàn bộ thời gian): {overall_mean_rate:.4f} Mbps")
print()

# ================== VẼ BIỂU ĐỒ ==================
plt.figure(figsize=(12, 6))

# Vẽ từng xe
for i in range(fso_rate.shape[1]):
    vehicle_rate = fso_rate[:, i]
    plt.plot(x, vehicle_rate, linestyle='-', linewidth=1, alpha=0.6,
             label=f'Data rate of vehicle {i+1}')

# Vẽ đường trung bình của cả 3 xe (in đậm)
plt.plot(x, rate_mean, linestyle='-', linewidth=3, color='red',
         label='Average data rate')

FONT_TITLE = 16
FONT_LABEL = 14
FONT_TICKS = 12
FONT_LEGEND = 12

# LABELS
plt.xlabel("Time slot", fontsize=FONT_LABEL)
plt.ylabel("Data rate (Gbps)", fontsize=FONT_LABEL)
# plt.title("Training Reward", fontsize=FONT_TITLE)

# TICKS
plt.xticks(fontsize=FONT_TICKS)
plt.yticks(fontsize=FONT_TICKS)

# LEGEND
plt.legend(fontsize=FONT_LEGEND)

plt.grid(True)
plt.tight_layout()
plt.show()
