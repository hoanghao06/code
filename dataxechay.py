import pandas as pd
import numpy as np
import random
import math
import os

# ==========================================
# 1. THIẾT LẬP THÔNG SỐ VÀ ĐỌC DỮ LIỆU
# ==========================================
# Sửa lại đường dẫn nếu cần thiết
csv_path = r"C:\wisalab\globecom2026\2026.-Tien_Hao-main\env\data\data_diahinh\mountain_1.csv"
try:
    df = pd.read_csv(csv_path).head(25)
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Không tìm thấy file mountain_1.csv, đang tạo dữ liệu giả lập để test...")
    df = pd.DataFrame({'building_separation_m': [10]*25, 'street_width_m': [8]*25})

b_width = 20
b_length = 20
grid_size = 5

# ==========================================
# 2. XÂY DỰNG BẢN ĐỒ ĐƯỜNG PHỐ (GRAPH MẠNG LƯỚI LỆCH)
# ==========================================
# Bản đồ tòa nhà xếp so le nên ta cần tính tọa độ Node cho từng khe hở
y_streets = []
current_y = 0
for i in range(grid_size):
    row_max_w = df.iloc[i*grid_size : (i+1)*grid_size]['street_width_m'].max()
    if i == 0: y_streets.append(current_y - row_max_w/2)
    current_y += b_length + row_max_w
    y_streets.append(current_y - row_max_w/2)

nodes_set = set()
edges_set = set()

idx = 0
for i in range(grid_size):
    current_x = 0
    for j in range(grid_size):
        if idx >= len(df): break
        sep = df.iloc[idx]['building_separation_m']
        
        # Đường dọc bên trái tòa nhà đầu tiên của hàng
        if j == 0:
            x_val = current_x - sep/2
            top_node = (x_val, y_streets[i])
            bot_node = (x_val, y_streets[i+1])
            nodes_set.add(top_node)
            nodes_set.add(bot_node)
            edges_set.add((top_node, bot_node)) # Nối đường dọc
            
        current_x += b_width + sep
        
        # Đường dọc bên phải của tòa nhà hiện tại (khe hở)
        x_val = current_x - sep/2
        top_node = (x_val, y_streets[i])
        bot_node = (x_val, y_streets[i+1])
        nodes_set.add(top_node)
        nodes_set.add(bot_node)
        edges_set.add((top_node, bot_node)) # Nối đường dọc
        
        idx += 1

nodes = list(nodes_set)
# Các đoạn đường ngang (nối các nút trên cùng một đường y_streets)
for y in y_streets:
    nodes_on_y = sorted([n for n in nodes if n[1] == y], key=lambda n: n[0])
    for k in range(len(nodes_on_y) - 1):
        edges_set.add((nodes_on_y[k], nodes_on_y[k+1]))

# Tạo danh sách kề (Adjacency List) để tìm ngã rẽ siêu tốc & chuẩn xác
adj_list = {n: [] for n in nodes}
for u, v in edges_set:
    if v not in adj_list[u]: adj_list[u].append(v)
    if u not in adj_list[v]: adj_list[v].append(u)

def is_same_node(n1, n2):
    return math.isclose(n1[0], n2[0], abs_tol=1e-3) and math.isclose(n1[1], n2[1], abs_tol=1e-3)

def get_neighbors(current_node):
    for n in nodes:
        if is_same_node(n, current_node):
            return adj_list[n]
    return []

# ==========================================
# 3. THUẬT TOÁN DI CHUYỂN & TÍNH TOÁN ĐỘNG HỌC
# ==========================================
def generate_trajectories(speed, output_prefix, time_limit=300, num_vehicles=10):
    vehicles = []
    safe_distance = max(18, speed * 1.5) 
    dt = 1.0 
    
    # Gán điểm tập kết ở góc trên cùng bên phải của bản đồ thực tế
    target_x = max([n[0] for n in nodes])
    target_y = max([n[1] for n in nodes])
    nodes_sorted = sorted(nodes, key=lambda n: math.hypot(n[0]-target_x, n[1]-target_y))
    start_nodes = nodes_sorted[:num_vehicles]
    
    vehicle_records = {i + 1: [] for i in range(num_vehicles)}
    
    for i in range(num_vehicles):
        sx, sy = start_nodes[i]
        neighbors = get_neighbors((sx, sy))
        first_target = random.choice(neighbors) if neighbors else (sx, sy)
        
        vehicles.append({
            'id': i + 1, 'x': sx, 'y': sy, 'last_node_x': sx, 'last_node_y': sy,
            'target': first_target, 'origin_node': (sx, sy), 'distance_covered': 0, 
            'active': True, 'wait_time': 0, 'history_x': sx, 'history_y': sy,
            'last_velX': 0.0, 'last_velY': 0.0
        })

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
                    old_target = v['target']
                    v['target'] = v['origin_node']
                    v['origin_node'] = old_target
                    v['last_node_x'], v['last_node_y'] = v['x'], v['y']
                    v['distance_covered'] = 0
                    v['wait_time'] = 0
            else:
                v['wait_time'] = 0
                v['x'], v['y'] = next_x, next_y
                v['distance_covered'] = intended_distance
                
                active_positions[v['id']] = (v['x'], v['y']) 
                
                if segment_length == 0 or intended_distance >= segment_length:
                    # FIX: Lưu lại nút cũ trước khi cập nhật nút mới để chống quay đầu
                    prev_node = v['origin_node']
                    
                    v['origin_node'] = (tx, ty) 
                    v['last_node_x'], v['last_node_y'] = tx, ty
                    v['distance_covered'] = intended_distance - segment_length
                    
                    neighbors = get_neighbors((tx, ty))
                    # Dùng math.isclose để lọc ngã rẽ chuẩn xác
                    valid_neighbors = [n for n in neighbors if not is_same_node(n, prev_node)]
                    
                    if valid_neighbors:
                        v['target'] = random.choice(valid_neighbors)
                    elif neighbors:
                        v['target'] = random.choice(neighbors) 
                    else:
                        v['target'] = (tx, ty)
                        v['distance_covered'] = 0

            # ---------------------------------------------------
            # TÍNH TOÁN VẬN TỐC VÀ GIA TỐC
            # ---------------------------------------------------
            current_x, current_y = v['x'], v['y']
            velX = (current_x - v['history_x']) / dt
            velY = (current_y - v['history_y']) / dt
            accX = (velX - v['last_velX']) / dt
            accY = (velY - v['last_velY']) / dt
            
            vehicle_records[v['id']].append({
                'posX': f"{current_x:.6f}", 'posY': f"{current_y:.6f}",
                'accX': f"{accX:.6f}", 'accY': f"{accY:.6f}",
                'velX': f"{velX:.6f}", 'velY': f"{velY:.6f}"
            })
            
            v['history_x'] = current_x
            v['history_y'] = current_y
            v['last_velX'] = velX
            v['last_velY'] = velY

    # Xuất file CSV
    for vid, records in vehicle_records.items():
        df_out = pd.DataFrame(records)
        filename = f"{output_prefix}_{speed}ms_User{vid}th.csv"
        df_out.to_csv(filename, index=False)
        
    print(f"Đã tạo thành công {num_vehicles} file kịch bản cho vận tốc {speed}m/s.")

# ==========================================
# 4. CHẠY TẠO FILE KỊCH BẢN
# ==========================================
speeds_to_run = [8,10,15]

for s in speeds_to_run:
    generate_trajectories(speed=s, output_prefix="trans_data_Train1th", time_limit=300, num_vehicles=10)
