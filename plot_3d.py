import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# =========================================================
# PHẦN 1: ĐỌC DỮ LIỆU TÒA NHÀ (ĐÚNG CHUẨN - KHÔNG GRID)
# =========================================================

csv_path = r"C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\env\data\data_diahinh\rural_1.csv"  # đổi file tùy ý

try:
    df_building = pd.read_csv(csv_path)
    df_building.columns = df_building.columns.str.strip()
except FileNotFoundError:
    print("Không tìm thấy file, dùng dữ liệu giả")
    df_building = pd.DataFrame({
        'x_pos': np.random.uniform(0, 500, 20),
        'y_pos': np.random.uniform(0, 500, 20),
        'avg_building_height_m': np.random.uniform(5, 50, 20)
    })

num_buildings = len(df_building)

# ✅ DÙNG TRỰC TIẾP TỌA ĐỘ TỪ CSV
x_pos = df_building["x_pos"].values
y_pos = df_building["y_pos"].values
z_pos = np.zeros(num_buildings)

# Kích thước tòa nhà (có thể random cho đẹp hơn)
dx = np.random.uniform(10, 25, num_buildings)
dy = np.random.uniform(10, 25, num_buildings)

dz = df_building["avg_building_height_m"].values

# =========================================================
# PHẦN 2: ĐỌC DỮ LIỆU XE
# =========================================================

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
        print(f"Không tìm thấy: {file_xe}")

if max_frames == 0:
    print("⚠️ Không có dữ liệu xe → chạy demo giả")
    max_frames = 200

time_slots = range(max_frames)

# Màu xe
colors = plt.get_cmap('tab10')(np.linspace(0, 1, num_vehicles))

# =========================================================
# PHẦN 3: VẼ 3D
# =========================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Vẽ tòa nhà
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
         color='#4ca3dd', alpha=0.6, edgecolor='k', linewidth=0.3)

# Xe
car_height = 1.0
scatters = []
lines = []

for i in range(num_vehicles):
    scat = ax.scatter([], [], [], color=colors[i], s=60, label=f'Xe {vehicle_ids[i]}')
    scatters.append(scat)
    
    line, = ax.plot([], [], [], color=colors[i], linewidth=2)
    lines.append(line)

# Trục
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Height (m)')

ax.set_xlim(0, 500)
ax.set_ylim(0, 500)
ax.set_zlim(0, 100)

# ✅ FIX méo hình
ax.set_box_aspect([1, 1, 0.4])

ax.legend()

# =========================================================
# ANIMATION
# =========================================================

def update(frame):
    for i, v_id in enumerate(vehicle_ids):
        
        if v_id not in vehicle_data:
            continue
        
        df_v = vehicle_data[v_id]
        
        if frame < len(df_v):
            row = df_v.iloc[frame]
            
            x_car = [row['posX']]
            y_car = [row['posY']]
            z_car = [car_height]
            
            scatters[i]._offsets3d = (x_car, y_car, z_car)
            
            # Vẽ quỹ đạo
            hist = df_v.iloc[:frame+1]
            lines[i].set_data(hist['posX'], hist['posY'])
            lines[i].set_3d_properties(np.full(len(hist), car_height))

    ax.set_title(f"Frame {frame}/{max_frames} | Buildings: {num_buildings}")
    
    return scatters + lines

ani = FuncAnimation(fig, update, frames=time_slots, interval=100, blit=False)

plt.tight_layout()
plt.show()
