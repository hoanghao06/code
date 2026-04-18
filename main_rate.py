# ================== IMPORT ==================
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD DATA ==================
data = np.load(
    r'C:\Users\AVSTC\Desktop\2026.Globecom\output\speed_10\0\flydata\rate_3.6.npy',
    allow_pickle=True
).item()

fso_rate = data['fso_rate']   # shape (time, UAV)

# ================== XỬ LÝ ==================
rate_mean = np.mean(fso_rate, axis=1)   # trung bình
x = np.arange(len(rate_mean))

# ================== PLOT ==================
plt.figure(figsize=(12, 6))

# ---- Vẽ từng xe ----
for i in range(fso_rate.shape[1]):
    plt.plot(
        x,
        fso_rate[:, i],
        linestyle='-',
        linewidth=1,
        alpha=0.6,
        label=f'vehicle {i+1}'
    )
    print (fso_rate[:, i].mean())

# ---- Vẽ đường trung bình ----
plt.plot(
    x,
    rate_mean,
    color='red',
    linewidth=3,
    label='Average'
)

# ================== FORMAT ==================
plt.title('Tốc độ từng xe và trung bình theo thời gian')
plt.xlabel('Timeslot n')
plt.ylabel('Channel Capacity [Gbps]')

plt.legend(ncol=2)   # legend gọn hơn
plt.grid()
plt.tight_layout()

plt.show()