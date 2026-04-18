import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


csv_path = r"C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\env\data\data_diahinh\rural_1.csv"
try:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
except Exception:
    df = pd.DataFrame()

# Kiểm tra cột cần thiết
required_cols = ['x_pos', 'y_pos', 'avg_building_height_m', 'width', 'length']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Thiếu cột trong CSV: {missing_cols}")

# LẤY DỮ LIỆU TÒA NHÀ TỪ CSV
x_pos = df['x_pos'].values
y_pos = df['y_pos'].values
z_pos = np.zeros(len(df))
dx = df['width'].values
dy = df['length'].values
dz = df['avg_building_height_m'].values

# ĐỌC DỮ LIỆU XE CHẠY
num_vehicles = 3
vehicle_ids = [i + 1 for i in range(num_vehicles)]
data_folder = r"C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\env\data\data_xechay\rural_1"
prefix = "trans_data_Train1th_8ms"

vehicle_data = {}
max_frames = 0
for v_id in vehicle_ids:
    file_xe = os.path.join(data_folder, f"{prefix}_User{v_id}th.csv")
    try:
        df_v = pd.read_csv(file_xe)
        df_v.columns = df_v.columns.str.strip()
        vehicle_data[v_id] = df_v
        max_frames = max(max_frames, len(df_v))
    except FileNotFoundError:
        pass

if max_frames == 0:
    max_frames = 200

time_slots = range(max_frames)
colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_vehicles))

# VẼ 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='#4ca3dd', alpha=0.6, edgecolor='k', linewidth=0.3)

car_height = 1.0
scatters, lines = [], []
for i in range(num_vehicles):
    scat = ax.scatter([], [], [], color=colors[i], s=60, label=f'Xe {vehicle_ids[i]}')
    scatters.append(scat)
    line, = ax.plot([], [], [], color=colors[i], linewidth=2)
    lines.append(line)

ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_zlim(0, 200)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')
ax.set_box_aspect([1, 1, 0.4])
ax.legend()

def update(frame):
    for i, v_id in enumerate(vehicle_ids):
        if v_id not in vehicle_data:
            continue
        df_v = vehicle_data[v_id]
        if frame < len(df_v):
            row = df_v.iloc[frame]
            scatters[i]._offsets3d = ([row['posX']], [row['posY']], [car_height])
            hist = df_v.iloc[:frame+1]
            lines[i].set_data(hist['posX'], hist['posY'])
            lines[i].set_3d_properties(np.full(len(hist), car_height))

    ax.set_title(f"Frame {frame}/{max_frames} | Dữ liệu tòa nhà từ CSV")
    return scatters + lines

ani = FuncAnimation(fig, update, frames=time_slots, interval=100, blit=False)
plt.tight_layout()
plt.show()
