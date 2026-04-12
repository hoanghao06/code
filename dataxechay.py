import pandas as pd
import numpy as np
import random
import math
import os

# ==========================================
# 1. THIẾT LẬP THÔNG SỐ VÀ ĐỌC DỮ LIỆU
# ==========================================
csv_path = r"C:\Users\DELL\Downloads\mountain_1.csv" # Sửa lại đường dẫn nếu cần
df = pd.read_csv(csv_path).head(25)

b_width = 20
b_length = 20
grid_size = 5

# ==========================================
# 2. XÂY DỰNG BẢN ĐỒ ĐƯỜNG PHỐ (GRAPH)
# ==========================================
x_streets = []
y_streets = []

current_x = 0
for j in range(grid_size):
    sep = df.iloc[j]['building_separation_m']
    if j == 0: x_streets.append(current_x - sep/2)
    current_x += b_width + sep
    x_streets.append(current_x - sep/2)

current_y = 0
for i in range(grid_size):
    row_max_w = df.iloc[i*grid_size : (i+1)*grid_size]['street_width_m'].max()
    if i == 0: y_streets.append(current_y - row_max_w/2)
    current_y += b_length + row_max_w
    y_streets.append(current_y - row_max_w/2)

nodes = [(x, y) for x in x_streets for y in y_streets]

# HÀM TÌM NGÃ RẼ ĐƯỢC NÂNG CẤP (Chống lỗi sai số float và Value Error)
def get_neighbors(current_node, nodes, x_streets, y_streets):
    cx, cy = current_node
    neighbors = []
    
    def get_idx(val, lst):
        for i, v in enumerate(lst):
            if math.isclose(v, val, abs_tol=1e-3): return i
        return -1
        
    c_x_idx = get_idx(cx, x_streets)
    c_y_idx = get_idx(cy, y_streets)
    
    if c_x_idx == -1 or c_y_idx == -1:
        return []
        
    for nx, ny in nodes:
        n_x_idx = get_idx(nx, x_streets)
        n_y_idx = get_idx(ny, y_streets)
        
        if n_x_idx == c_x_idx and n_y_idx != -1:
            if abs(c_y_idx - n_y_idx) == 1: neighbors.append((nx, ny))
        elif n_y_idx == c_y_idx and n_x_idx != -1:
            if abs(c_x_idx - n_x_idx) == 1: neighbors.append((nx, ny))
    return neighbors

# ==========================================
# 3. THUẬT TOÁN DI CHUYỂN & TRÁNH VA CHẠM
# ==========================================
def generate_trajectories(speed, output_filename, time_limit=300, num_vehicles=10):
    vehicles = []
    safe_distance = max(18, speed * 1.5) 
    
    target_x, target_y = 300.0, 100.0
    nodes_sorted = sorted(nodes, key=lambda n: math.hypot(n[0]-target_x, n[1]-target_y))
    start_nodes = nodes_sorted[:num_vehicles]
    
    for i in range(num_vehicles):
        sx, sy = start_nodes[i]
        neighbors = get_neighbors((sx, sy), nodes, x_streets, y_streets)
        first_target = random.choice(neighbors) if neighbors else (sx, sy)
        
        vehicles.append({
            'id': i + 1,
            'x': sx,
            'y': sy,
            'last_node_x': sx,
            'last_node_y': sy,
            'target': first_target, 
            'origin_node': (sx, sy), # BIẾN MỚI: Ghi nhớ ngã tư hợp lệ vừa đi qua
            'distance_covered': 0, 
            'active': True,        
            'wait_time': 0
        })

    records = []

    for t in range(time_limit + 1):
        active_positions = {v['id']: (v['x'], v['y']) for v in vehicles}
        
        for v in vehicles:
            tx, ty = v['target']
            sx, sy = v['last_node_x'], v['last_node_y']
            dx, dy = tx - sx, ty - sy
            segment_length = math.hypot(dx, dy)
            
            intended_distance = v['distance_covered'] + speed
            
            if segment_length == 0 or intended_distance >= segment_length:
                next_x, next_y = tx, ty
            else:
                ratio = intended_distance / segment_length
                next_x = sx + dx * ratio
                next_y = sy + dy * ratio
                
            collision = False
            for other_id, pos in active_positions.items():
                if other_id != v['id']:
                    dist = math.hypot(next_x - pos[0], next_y - pos[1])
                    if dist < safe_distance:
                        collision = True
                        break
            
            if collision:
                v['wait_time'] += 1
                if v['wait_time'] > 3:
                    # GỠ KẸT ĐÃ FIX LỖI: Đảo ngược giữa ngã tư hiện tại và ngã tư gốc
                    old_target = v['target']
                    v['target'] = v['origin_node']
                    v['origin_node'] = old_target
                    
                    # Bắt đầu tính quãng đường mới từ vị trí hiện tại
                    v['last_node_x'], v['last_node_y'] = v['x'], v['y']
                    v['distance_covered'] = 0
                    v['wait_time'] = 0
            else:
                v['wait_time'] = 0
                v['x'], v['y'] = next_x, next_y
                v['distance_covered'] = intended_distance
                
                active_positions[v['id']] = (v['x'], v['y']) 
                
                # XỬ LÝ KHI ĐẾN MỘT NGÃ TƯ MỚI
                if segment_length == 0 or intended_distance >= segment_length:
                    # Cập nhật ngã tư gốc thành ngã tư vừa tới
                    v['origin_node'] = (tx, ty) 
                    v['last_node_x'], v['last_node_y'] = tx, ty
                    v['distance_covered'] = intended_distance - segment_length
                    
                    neighbors = get_neighbors((tx, ty), nodes, x_streets, y_streets)
                    
                    # Lọc bỏ ngã tư gốc để xe không tự động quay đầu đi ngược lại
                    valid_neighbors = [n for n in neighbors if n != v['origin_node']]
                    
                    if valid_neighbors:
                        v['target'] = random.choice(valid_neighbors)
                    elif neighbors:
                        # Nếu đường cụt chỉ có 1 lối ra duy nhất, bắt buộc phải quay lại
                        v['target'] = random.choice(neighbors) 
                    else:
                        v['target'] = (tx, ty)
                        v['distance_covered'] = 0

            records.append({
                'time_slot': t,
                'vehicle_id': v['id'],
                'x': round(v['x'], 2),
                'y': round(v['y'], 2)
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_filename, index=False)
    print(f"Đã tạo thành công quỹ đạo xe chạy đồng loạt: {output_filename} (Vận tốc: {speed}m/s)")

# ==========================================
# 4. CHẠY TẠO FILE KỊCH BẢN
# ==========================================
generate_trajectories(speed=8, output_filename="vehicle_8ms.csv")