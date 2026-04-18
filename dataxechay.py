import pandas as pd
import numpy as np
import random
import math
import os

# ==========================================
# 1. THIẾT LẬP THÔNG SỐ VÀ ĐỌC DỮ LIỆU
# ==========================================
csv_path = r"C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\env\data\data_diahinh\rural_1.csv"

try:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print("Đọc file CSV thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {csv_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

required_cols = ['x_pos', 'y_pos', 'building_separation_m', 'street_width_m', 'avg_building_height_m', 'width', 'length']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Thiếu cột trong CSV: {missing_cols}")

# ===== KHOẢNG CÁCH AN TOÀN =====
SAFE_DIST_TO_BUILDING = 2.0
NODE_CLEARANCE = SAFE_DIST_TO_BUILDING + 5.0

# ===== GIỚI HẠN BẢN ĐỒ =====
MAP_MIN_X, MAP_MAX_X = 0.0, 450.0
MAP_MIN_Y, MAP_MAX_Y = 0.0, 450.0

print(f"Giới hạn bản đồ: X[{MAP_MIN_X:.1f}, {MAP_MAX_X:.1f}], Y[{MAP_MIN_Y:.1f}, {MAP_MAX_Y:.1f}]")

# ==========================================
# 2. XÂY DỰNG DANH SÁCH TÒA NHÀ
# ==========================================
building_positions = []

for _, row in df.iterrows():
    if MAP_MIN_X <= row['x_pos'] <= MAP_MAX_X and MAP_MIN_Y <= row['y_pos'] <= MAP_MAX_Y:
        building_positions.append({
            'x': float(row['x_pos']),
            'y': float(row['y_pos']),
            'width': float(row['width']),
            'length': float(row['length'])
        })

def rect_bounds(rect, expand=0.0):
    xmin = rect['x'] - expand
    xmax = rect['x'] + rect['width'] + expand
    ymin = rect['y'] - expand
    ymax = rect['y'] + rect['length'] + expand
    return xmin, xmax, ymin, ymax

def point_to_rect_distance(px, py, rect):
    xmin, xmax, ymin, ymax = rect_bounds(rect, expand=0.0)
    dx = max(xmin - px, 0.0, px - xmax)
    dy = max(ymin - py, 0.0, py - ymax)
    return math.hypot(dx, dy)

def min_distance_to_buildings(x, y, buildings):
    if not buildings:
        return float('inf')
    return min(point_to_rect_distance(x, y, b) for b in buildings)

def point_inside_rect(px, py, rect, expand=0.0):
    xmin, xmax, ymin, ymax = rect_bounds(rect, expand=expand)
    return xmin <= px <= xmax and ymin <= py <= ymax

def segment_intersects_rect(p1, p2, rect, expand=0.0):
    xmin, xmax, ymin, ymax = rect_bounds(rect, expand=expand)

    x0, y0 = p1
    x1, y1 = p2

    if xmin <= x0 <= xmax and ymin <= y0 <= ymax:
        return True
    if xmin <= x1 <= xmax and ymin <= y1 <= ymax:
        return True

    dx = x1 - x0
    dy = y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t

    return True

def is_path_clear(p1, p2, buildings, safe_dist):
    for b in buildings:
        if segment_intersects_rect(p1, p2, b, expand=safe_dist):
            return False
    return True

# ==========================================
# 3. TẠO MẠNG LƯỚI GIAO THÔNG
# ==========================================
grid_step = 5
nodes_set = set()

for x in np.arange(MAP_MIN_X, MAP_MAX_X + grid_step, grid_step):
    for y in np.arange(MAP_MIN_Y, MAP_MAX_Y + grid_step, grid_step):
        if min_distance_to_buildings(x, y, building_positions) >= NODE_CLEARANCE:
            nodes_set.add((float(x), float(y)))

nodes = list(nodes_set)
adj_list = {n: [] for n in nodes}
edges_set = set()

print("Đang xây dựng mạng lưới đường đi đô thị (Manhattan Grid)...")
for n1 in nodes:
    for n2 in nodes:
        if n1 == n2:
            continue

        same_x = math.isclose(n1[0], n2[0], abs_tol=1e-3)
        same_y = math.isclose(n1[1], n2[1], abs_tol=1e-3)

        if same_x or same_y:
            dist = math.hypot(n1[0] - n2[0], n1[1] - n2[1])
            if dist <= grid_step * 1.1:
                if (n2, n1) not in edges_set:
                    if is_path_clear(n1, n2, building_positions, SAFE_DIST_TO_BUILDING):
                        edges_set.add((n1, n2))
                        adj_list[n1].append(n2)
                        adj_list[n2].append(n1)

def is_same_node(n1, n2):
    if n2 is None:
        return False
    return math.isclose(n1[0], n2[0], abs_tol=1e-3) and math.isclose(n1[1], n2[1], abs_tol=1e-3)

def get_neighbors(current_node):
    return adj_list.get(current_node, [])

def choose_next_target(current_node, previous_node):
    neighbors = get_neighbors(current_node)
    valid_neighbors = [n for n in neighbors if not is_same_node(n, previous_node)]

    if not valid_neighbors:
        if neighbors:
            return random.choice(neighbors)
        return current_node

    if previous_node is not None:
        dir_x = current_node[0] - previous_node[0]
        dir_y = current_node[1] - previous_node[1]

        straight_options = []
        turn_options = []

        for n in valid_neighbors:
            ndx = n[0] - current_node[0]
            ndy = n[1] - current_node[1]

            if (dir_x * ndx + dir_y * ndy) > 0:
                straight_options.append(n)
            else:
                turn_options.append(n)

        if straight_options and random.random() < 0.85:
            return random.choice(straight_options)
        if turn_options:
            return random.choice(turn_options)

    return random.choice(valid_neighbors)

# ==========================================
# 4. THUẬT TOÁN DI CHUYỂN & TRÁNH VA CHẠM
# ==========================================
def generate_trajectories(speed, output_prefix, time_limit=300, num_vehicles=5):
    vehicles = []
    safe_distance_vehicles = max(15.0, speed * 1.5)
    dt = 1.0

    if len(nodes) < num_vehicles:
        print("Lỗi: Bản đồ không đủ chỗ trống để đặt xe!")
        return

    start_nodes = random.sample(nodes, num_vehicles)
    vehicle_records = {i + 1: [] for i in range(num_vehicles)}

    for i in range(num_vehicles):
        sx, sy = start_nodes[i]
        first_target = choose_next_target((sx, sy), None)

        vehicles.append({
            'id': i + 1,
            'x': sx, 'y': sy,
            'last_node_x': sx, 'last_node_y': sy,
            'target': first_target,
            'origin_node': (sx, sy),
            'distance_covered': 0.0,
            'wait_time': 0,
            'history_x': sx, 'history_y': sy,
            'last_velX': 0.0, 'last_velY': 0.0
        })

    for _ in range(time_limit + 1):
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

            # Kiểm tra ra khỏi map
            if not (MAP_MIN_X <= next_x <= MAP_MAX_X and MAP_MIN_Y <= next_y <= MAP_MAX_Y):
                v['target'] = v['origin_node']
                v['origin_node'] = (v['last_node_x'], v['last_node_y'])
                v['last_node_x'], v['last_node_y'] = v['x'], v['y']
                v['distance_covered'] = 0.0
                continue

            # Kiểm tra đoạn di chuyển có cắt vào vùng cấm quanh nhà không
            if not is_path_clear((v['x'], v['y']), (next_x, next_y), building_positions, SAFE_DIST_TO_BUILDING):
                v['wait_time'] += 1
                if v['wait_time'] > 2:
                    v['target'] = v['origin_node']
                    v['origin_node'] = (v['last_node_x'], v['last_node_y'])
                    v['last_node_x'], v['last_node_y'] = v['x'], v['y']
                    v['distance_covered'] = 0.0
                    v['wait_time'] = 0
                continue

            # Kiểm tra điểm đến có quá gần nhà không
            if min_distance_to_buildings(next_x, next_y, building_positions) < SAFE_DIST_TO_BUILDING:
                v['target'] = v['origin_node']
                v['origin_node'] = (v['last_node_x'], v['last_node_y'])
                v['last_node_x'], v['last_node_y'] = v['x'], v['y']
                v['distance_covered'] = 0.0
                continue

            # Kiểm tra va chạm xe khác
            collision = False
            for other_id, pos in active_positions.items():
                if other_id != v['id']:
                    if math.hypot(next_x - pos[0], next_y - pos[1]) < safe_distance_vehicles:
                        collision = True
                        break

            if collision:
                v['wait_time'] += 1
                if v['wait_time'] > 3:
                    old_target = v['target']
                    v['target'] = v['origin_node']
                    v['origin_node'] = old_target
                    v['last_node_x'], v['last_node_y'] = v['x'], v['y']
                    v['distance_covered'] = 0.0
                    v['wait_time'] = 0
            else:
                v['wait_time'] = 0
                v['x'], v['y'] = next_x, next_y
                v['distance_covered'] = intended_distance
                active_positions[v['id']] = (v['x'], v['y'])

                # Khi tới nút
                if segment_length == 0 or intended_distance >= segment_length:
                    prev_node = v['origin_node']
                    v['origin_node'] = (tx, ty)
                    v['last_node_x'], v['last_node_y'] = tx, ty
                    v['distance_covered'] = intended_distance - segment_length

                    v['target'] = choose_next_target((tx, ty), prev_node)

                    # Nếu target mới không an toàn thì tìm lại
                    if v['target'] != (tx, ty):
                        if not is_path_clear((tx, ty), v['target'], building_positions, SAFE_DIST_TO_BUILDING):
                            safe_neighbors = [
                                n for n in get_neighbors((tx, ty))
                                if not is_same_node(n, prev_node)
                                and is_path_clear((tx, ty), n, building_positions, SAFE_DIST_TO_BUILDING)
                            ]
                            if safe_neighbors:
                                v['target'] = random.choice(safe_neighbors)
                            else:
                                v['target'] = prev_node

            # Ghi nhận dữ liệu
            current_x, current_y = v['x'], v['y']
            velX = (current_x - v['history_x']) / dt
            velY = (current_y - v['history_y']) / dt
            accX = (velX - v['last_velX']) / dt
            accY = (velY - v['last_velY']) / dt

            vehicle_records[v['id']].append({
                'posX': f"{current_x:.6f}",
                'posY': f"{current_y:.6f}",
                'accX': f"{accX:.6f}",
                'accY': f"{accY:.6f}",
                'velX': f"{velX:.6f}",
                'velY': f"{velY:.6f}"
            })

            v['history_x'] = current_x
            v['history_y'] = current_y
            v['last_velX'] = velX
            v['last_velY'] = velY

    for vid, records in vehicle_records.items():
        df_out = pd.DataFrame(records)
        filename = f"{output_prefix}_{speed}ms_User{vid}th.csv"
        df_out.to_csv(filename, index=False)

    print(f"Hoàn thành xuất {num_vehicles} file kịch bản cho vận tốc {speed}m/s.")

# ==========================================
# 5. CHẠY KỊCH BẢN
# ==========================================
speeds_to_run = [8, 10, 15]

for s in speeds_to_run:
    generate_trajectories(speed=s, output_prefix="trans_data_Train3th", time_limit=300, num_vehicles=5)
