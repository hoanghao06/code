import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm # Import colormaps

# =========================================================
# PHẦN 1: ĐỌC DỮ LIỆU VÀ VẼ CÁC TÒA NHÀ (TĨNH)
# =========================================================
# 1. Đọc file tòa nhà
csv_path = r"C:\Users\DELL\Downloads\mountain_1.csv" # Hoặc đường dẫn tuyệt đối của bạn
df_building = pd.read_csv(csv_path).head(25)
df_building.columns = df_building.columns.str.strip() # Xóa khoảng trắng tên cột để tránh lỗi cũ

b_width = 20
b_length = 20
grid_size = 5

x_pos, y_pos = np.zeros(25), np.zeros(25)
current_y = 0
idx = 0

for i in range(grid_size):
    current_x = 0
    row_max_y_step = 0
    for j in range(grid_size):
        if idx >= 25: break
        sep = df_building.iloc[idx]['building_separation_m']
        street_w = df_building.iloc[idx]['street_width_m']
        
        x_pos[idx] = current_x
        y_pos[idx] = current_y
        
        current_x += b_width + sep
        row_max_y_step = max(row_max_y_step, b_length + street_w)
        idx += 1
    current_y += row_max_y_step

# Tính tọa độ Z của tòa nhà dựa trên địa hình (lấy tâm tòa nhà làm mốc)
z_pos = np.zeros(25)

dx = np.ones(25) * b_width
dy = np.ones(25) * b_length
dz = df_building['avg_building_height_m'].values

# =========================================================
# PHẦN 2: ĐỌC DỮ LIỆU QUỸ ĐẠO XE VÀ CHUẨN BỊ ANIMATION
# =========================================================

file_xe = r"vehicle_8ms.csv" 
df_vehicles = pd.read_csv(file_xe)

# Lấy danh sách các giây (time_slot) từ 0 đến 300
time_slots = sorted(df_vehicles['time_slot'].unique())

# Giả định số lượng xe (ví dụ 10 xe)
num_vehicles = 10 
# Tạo một list các ID xe (assuming IDs are 1, 2, ..., num_vehicles)
vehicle_ids = [i + 1 for i in range(num_vehicles)]

# Tạo một bảng màu phân biệt cho 10 xe
# Sử dụng colormap 'tab10' hoặc bất kỳ colormap phân loại nào khác
colors = cm.get_cmap('tab10', num_vehicles)(range(num_vehicles))

# =========================================================
# PHẦN 3: HIỂN THỊ LÊN ĐỒ THỊ 3D (ĐÃ CẬP NHẬT QUỸ ĐẠO)
# =========================================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Vẽ 25 tòa nhà (Màu xanh, hơi trong suốt)
ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='#4ca3dd', alpha=0.5, edgecolor='k', linewidth=0.5)

car_height = 1.0 
scatters = []
lines = [] # TẠO THÊM MỘT LIST CHỨA ĐƯỜNG QUỸ ĐẠO

for i in range(num_vehicles):
    # Tạo scatter cho mỗi xe (vị trí hiện tại)
    scat = ax.scatter([], [], [], color=colors[i], s=70, marker='o', label=f'Xe {vehicle_ids[i]}')
    scatters.append(scat)
    
    # Tạo đối tượng đường thẳng (line) để vẽ lại quỹ đạo xe đã đi
    line, = ax.plot([], [], [], color=colors[i], linewidth=2, alpha=0.6)
    lines.append(line)

ax.set_xlabel('Trục X (m)')
ax.set_ylabel('Trục Y (m)')
ax.set_zlabel('Chiều cao (m)')
ax.set_zlim(0, 100)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), prop={'size': 9})

def update(frame):
    # Lấy dữ liệu của frame hiện tại để vẽ vị trí xe
    current_data = df_vehicles[df_vehicles['time_slot'] == frame]
    
    # Lấy dữ liệu TỪ ĐẦU ĐẾN FRAME HIỆN TẠI để vẽ đường quỹ đạo
    history_data = df_vehicles[df_vehicles['time_slot'] <= frame]
    
    for i, v_id in enumerate(vehicle_ids):
        # Dữ liệu vị trí hiện hành của xe
        v_data = current_data[current_data['vehicle_id'] == v_id]
        
        # Dữ liệu lịch sử đường đi của xe
        v_hist = history_data[history_data['vehicle_id'] == v_id]
        
        if not v_data.empty:
            # 1. Cập nhật vị trí hiện tại (Dấu chấm tròn)
            x_car = v_data['x'].values
            y_car = v_data['y'].values
            z_car = np.full(len(x_car), car_height)
            scatters[i]._offsets3d = (x_car, y_car, z_car)
            
            # 2. Cập nhật đường quỹ đạo (Đường thẳng nối các điểm)
            x_hist = v_hist['x'].values
            y_hist = v_hist['y'].values
            z_hist = np.full(len(x_hist), car_height)
            
            lines[i].set_data(x_hist, y_hist)
            lines[i].set_3d_properties(z_hist)
        else:
            # Ẩn xe đi nếu không có dữ liệu tại giây này
            scatters[i]._offsets3d = ([], [], [])
            # Tùy chọn: có thể để đường quỹ đạo cũ giữ nguyên, hoặc ẩn đi bằng cách:
            # lines[i].set_data([], [])
            # lines[i].set_3d_properties([])
            
    ax.set_title(f'Mô phỏng 3D - Thời gian: {frame} / 300 giây - Kiểm tra {num_vehicles} Xe phân biệt', fontsize=14)
    
    # Trả về cả list scatters và list lines
    return scatters + lines

ani = FuncAnimation(fig, update, frames=time_slots, interval=100, blit=False)

plt.tight_layout()
plt.show()