import numpy as np
import matplotlib.pyplot as plt

# Import các hàm tính toán từ file channel.py của bạn
from channel import total_harvested_energy, get_fso_access, get_snr, data_rate, FSO_bandwidth, UAVEnergyModel

def main():
    print("=== TÍNH TOÁN EE (TỬ SỐ LÀ RATE TRUNG BÌNH CỦA RIÊNG TỪNG TRƯỜNG HỢP) ===")

    num_timeslots = 300
    times = np.arange(num_timeslots)
    
    # Cố định tỷ lệ chia năng lượng
    alpha_fixed = 0.2 
    
    # Độ cao UAV giả định (chỉ lấy 800m)
    altitudes = [800] 
    
    # Tọa độ cơ bản của HAP và IRS
    hap_pos = np.array([500, 500, 20000])
    irs_pos = np.array([0, 0, 80])
    uav_model = UAVEnergyModel()

    # =========================================================
    # 1. LOAD DỮ LIỆU TỪ CÁC FILE NPY
    # =========================================================
    car_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\car_3.npy', allow_pickle=True).item()
    car_trajectory = car_data['car_0'] if 'car_0' in car_data else list(car_data.values())[0]

    uav_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\uav_3.npy', allow_pickle=True).item()
    uav_trajectory = uav_data['position']
    uav_velocities = uav_data['velocity']

    energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
    rate_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\rate_3.npy', allow_pickle=True).item()

    # =========================================================
    # 2. TÍNH TOÁN EE CHO QUỸ ĐẠO TỐI ƯU 
    # =========================================================
    overall_opt_rate_gbps = np.mean(rate_data['mean_rate'])
    overall_opt_rate_mbps = overall_opt_rate_gbps * 1000.0
    print(f"[+] Rate trung bình của quỹ đạo tối ưu: {overall_opt_rate_mbps:.2f} Mbps")

    EE_optimized = []
    for t in range(num_timeslots):
        # Tính vận tốc thực tế tại timeslot t để tính công suất tiêu thụ P_c
        v_vector = uav_velocities[min(t, len(uav_velocities)-1)]
        v_uav = np.linalg.norm(v_vector)
        p_c = uav_model.propulsion_power(v_uav) + uav_model.P_c
        
        # Năng lượng thu hoạch từ file
        p_sol_opt = (energy_data['solar energy'][t] / 0.9) * 0.4
        p_fso_opt = energy_data['fso energy'][t]
        p_h_opt = p_sol_opt + (p_fso_opt * alpha_fixed)
        
        # Tính EE (Lấy tử số là trung bình, mẫu số là tức thời)
        net_power = p_c - p_h_opt
        if net_power > 0:
            ee = overall_opt_rate_mbps / net_power
        else:
            ee = np.nan # Tránh chia cho 0
        EE_optimized.append(ee)

    # =========================================================
    # 3. TÍNH TOÁN EE CHO CỐ ĐỊNH Z = 800m
    # =========================================================
    EE_fixed_alts = {z: [] for z in altitudes}

    print("Đang tính toán cho độ cao Z = 800m...")
    for z in altitudes:
        denominators = [] 
        instant_rates = [] 
        
        for t in range(num_timeslots):
            car_pos = car_trajectory[min(t, len(car_trajectory) - 1)]
            
            uav_pos_raw = uav_trajectory[min(t, len(uav_trajectory) - 1)]
            current_uav_pos = np.array([uav_pos_raw[0], uav_pos_raw[1], z])

            v_vector = uav_velocities[min(t, len(uav_velocities)-1)]
            v_uav = np.sqrt(v_vector[0]**2 + v_vector[1]**2) 
            p_c = uav_model.propulsion_power(v_uav) + uav_model.P_c

            _, p_sol, _, p_batt_fso, p_tx = total_harvested_energy(
                hap_pos, irs_pos, current_uav_pos, duration=1, energy_ratio=alpha_fixed
            )
            p_h = (p_sol / 0.9) * 0.4 + p_batt_fso
            denominators.append(p_c - p_h) 
            
            h_acc, _, _, _ = get_fso_access(current_uav_pos, car_pos)
            gamma = get_snr(h_acc, p_tx, current_uav_pos)
            r_mbps = data_rate(gamma, FSO_bandwidth) * 1000.0
            instant_rates.append(r_mbps) 
            
        overall_z_rate_mbps = np.mean(instant_rates)
        print(f" -> Z = {z} m có Rate trung bình riêng: {overall_z_rate_mbps:.2f} Mbps")
        
        for t in range(num_timeslots):
            net_power = denominators[t]
            if net_power > 0:
                ee = overall_z_rate_mbps / net_power
            else:
                ee = np.nan
            EE_fixed_alts[z].append(ee)

    # =========================================================
    # 4. TÍNH TOÁN EE CHO QUỸ ĐẠO RANDOM (Bắt đầu tại 500, 500, 500)
    # =========================================================
    EE_random = []
    denominators_random = []
    instant_rates_random = []
    harvested_power_random = [] # Mảng lưu năng lượng thu hoạch tức thời để in ra
    
    np.random.seed(1052026) # Cố định seed
    curr_random_pos = np.array([500.0, 500.0, 800.0])
    fixed_uav_velocity = 20.0 
    
    print("Đang tính toán cho quỹ đạo Random 3D...")
    for t in range(num_timeslots):
        car_pos = car_trajectory[min(t, len(car_trajectory) - 1)]
        
        # Tạo hướng di chuyển ngẫu nhiên
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        dx = fixed_uav_velocity * np.sin(phi) * np.cos(theta)
        dy = fixed_uav_velocity * np.sin(phi) * np.sin(theta)
        dz = fixed_uav_velocity * np.cos(phi)
        
        curr_random_pos = curr_random_pos + np.array([dx, dy, dz])
        curr_random_pos[2] = np.maximum(curr_random_pos[2], 50.0) # Tránh đâm xuống đất

        # Tính P_c
        p_c = uav_model.propulsion_power(fixed_uav_velocity) + uav_model.P_c

        # Tính P_h
        _, p_sol, _, p_batt_fso, p_tx = total_harvested_energy(
            hap_pos, irs_pos, curr_random_pos, duration=1, energy_ratio=alpha_fixed
        )
        p_h = (p_sol / 0.9) * 0.4 + p_batt_fso
        denominators_random.append(p_c - p_h)
        harvested_power_random.append(p_h) # Lưu P_h vào mảng
        
        # Tính Rate tức thời
        h_acc, _, _, _ = get_fso_access(curr_random_pos, car_pos)
        gamma = get_snr(h_acc, p_tx, curr_random_pos)
        r_mbps = data_rate(gamma, FSO_bandwidth) * 1000
        instant_rates_random.append(r_mbps)
        
    overall_random_rate_mbps = np.mean(instant_rates_random)
    
    # In ra các thông số của quỹ đạo Random
    print(f" -> Quỹ đạo Random có Rate trung bình: {overall_random_rate_mbps:.2f} Gbps")
    print(f" -> Quỹ đạo Random có Công suất thu hoạch (P_h) trung bình: {np.mean(harvested_power_random):.4f} W")
    print(f" -> Quỹ đạo Random có Tổng năng lượng thu hoạch (300s): {np.sum(harvested_power_random):.2f} Joules")
    
    # Tính EE mảng hoàn chỉnh cho Random
    for t in range(num_timeslots):
        net_power = denominators_random[t]
        if net_power > 0:
            ee = overall_random_rate_mbps / net_power
        else:
            ee = np.nan
        EE_random.append(ee)

    print("Tính toán xong! Đang hiển thị đồ thị...")

    # =========================================================
    # 5. VẼ BIỂU ĐỒ 
    # =========================================================
    fig, ax = plt.subplots(figsize=(11, 9.5))

    # Vẽ đường Tối ưu
    ax.plot(times, EE_optimized, color='red', linewidth=3, 
            label='Optimized Trajectory with Algorithm 1', zorder=5)

    # Vẽ đường Độ cao cố định Z = 800m
    ax.plot(times, EE_fixed_alts[800], color='blue', linestyle='--', linewidth=2, 
            alpha=0.8, label='Fixed UAV position')
    
    # Vẽ đường Random
    ax.plot(times, EE_random, color='darkgreen', linestyle='-.', linewidth=2.5, 
            alpha=0.9, label='Random UAV Trajectory')

    # CẬP NHẬT CỠ CHỮ LÊN 24
    ax.set_xlabel(r'Timeslot ($t$)', fontsize=24)
    ax.set_ylabel(r'Energy Efficiency (Mbps/W)', fontsize=24)
    
    ax.set_xlim(0, num_timeslots)
    
    # CẬP NHẬT CỠ SỐ LÊN 24
    ax.tick_params(axis='both', labelsize=24)
    ax.grid(True, linestyle=':', alpha=0.7)

    # Tăng nhẹ cỡ chữ legend lên 16 cho cân đối
    ax.legend(loc='upper right', fontsize=16, frameon=True, edgecolor='black')
    plt.tight_layout()
    # ... (các dòng code set label, xlim, tick_params, legend của bạn giữ nguyên) ...

    # Thêm dòng này để ép tỷ lệ lõi đồ thị
    # 1.0 = Hình vuông hoàn toàn
    # 1.1 hoặc 1.2 = Khung hình sẽ cao hơn chiều rộng một chút
    ax.set_box_aspect(0.9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
