# ================== IMPORT ==================
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD DATA ==================
data_uav = np.load(
    r'C:\Users\AVSTC\Desktop\2026.Globecom\output_35_Gbps\speed_10\0\flydata\uav_3.5.npy',
    allow_pickle=True
).item()

uav_pos = data_uav['position']   

print("Shape:", uav_pos.shape)

# ================== TÁCH TOẠ ĐỘ ==================
x = uav_pos[:, 0]
y = uav_pos[:, 1]
z = uav_pos[:, 2]

# ================== PLOT 3D ==================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, linewidth=2, label='UAV trajectory')

# đánh dấu điểm đầu & cuối
ax.scatter(x[0], y[0], z[0], marker='o', label='Start')
ax.scatter(x[-1], y[-1], z[-1], marker='x', label='End')

# ================== FORMAT ==================
ax.set_title('Quỹ đạo bay 3D của UAV')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend()
plt.tight_layout()
plt.show()