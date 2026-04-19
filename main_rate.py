# ================== IMPORT ==================
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD DATA ==================
data = np.load(
    r'C:\Users\AVSTC\Desktop\2026.Globecom\output\speed_10\0\flydata\rate_3.npy',
    allow_pickle=True
).item()

fso_rate = data['fso_rate']   # shape (time, UAV)

# ================== XỬ LÝ ==================
rate_mean = np.mean(fso_rate, axis=1)   # trung bình
x = np.arange(len(rate_mean))

# ================== PLOT ==================
plt.figure(figsize=(12, 6))

# ---- Vẽ từng xe ----
target_rate = 3

for i in range(fso_rate.shape[1]):
    vehicle_rate = fso_rate[:, i]

    plt.plot(
        x,
        vehicle_rate,
        linestyle='-',
        linewidth=1,
        alpha=0.6,
        label=f'vehicle {i+1}'
    )

    # Trung bình
    print(vehicle_rate.mean())

    # Đếm số lần > target
    count = np.sum(vehicle_rate > target_rate)

    # Tỷ lệ %
    percentage = count / len(vehicle_rate)

    print("Tỷ lệ phần trăm:")
    print(percentage)
    print("\n")
