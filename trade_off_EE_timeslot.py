import numpy as np
import matplotlib.pyplot as plt

# Import các hàm tính toán từ file channel.py của bạn
from channel import total_harvested_energy, get_fso_access, get_snr, data_rate, FSO_bandwidth, UAVEnergyModel

def main():
    print("=== TÍNH TOÁN EE (TỬ SỐ LÀ RATE TRUNG BÌNH CỦA RIÊNG TỪNG ĐỘ CAO) ===")

    num_timeslots = 300
    times = np.arange(num_timeslots)
    
    # Cố định tỷ lệ chia năng lượng
    alpha_fixed = 0.2 
    
    # Các độ cao UAV giả định để vẽ các đường so sánh (m)
    altitudes = [400, 800, 1800] 
    
    # Tọa độ cơ bản của HAP và IRS
    hap_pos = np.array([500, 500, 20000])
    irs_pos = np.array([0, 0, 80])
    uav_model = UAVEnergyModel()

    # =========================================================
    # 1. LOAD DỮ LIỆU TỪ CÁC FILE NPY
    # =========================================================
    # Đã thêm chữ r phía trước đường dẫn để tránh lỗi Unicode
    car_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\car_3.npy', allow_pickle=True).item()
    car_trajectory = car_data['car_0'] if 'car_0' in car_data else list(car_data.values())[0]

    uav_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\uav_3.npy', allow_pickle=True).item()
    uav_trajectory = uav_data['position']
    uav_velocities = uav_data['velocity']

    energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
    rate_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\rate_3.npy', allow_pickle=True).item()

    # =========================================================
    # 2. TÍNH TOÁN EE CHO QUỸ ĐẠO TỐI ƯU (Dùng Rate trung bình của chính nó)
    # =========================================================
    # Tính Rate trung bình của đường tối ưu (hằng số)
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
        p_sol_opt = energy_data['solar energy'][t]
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
    # 3. TÍNH TOÁN EE CHO CÁC ĐỘ CAO CỐ ĐỊNH (Mỗi độ cao tự tính Rate trung bình riêng)
    # =========================================================
    EE_fixed_alts = {z: [] for z in altitudes}

    print("Đang quét tính toán lại cho các độ cao giả định...")
    for z in altitudes:
        denominators = [] # Tạm lưu mẫu số của 300s
        instant_rates = [] # Tạm lưu Rate của 300s để tính trung bình
        
        # BƯỚC 1: Quét 300 giây để lấy các giá trị
        for t in range(num_timeslots):
            car_pos = car_trajectory[min(t, len(car_trajectory) - 1)]
            
            # Lấy X, Y từ quỹ đạo thực tế, ép độ cao bằng z
            uav_pos_raw = uav_trajectory[min(t, len(uav_trajectory) - 1)]
            current_uav_pos = np.array([uav_pos_raw[0], uav_pos_raw[1], z])

            # Tính P_c
            v_vector = uav_velocities[min(t, len(uav_velocities)-1)]
            v_uav = np.sqrt(v_vector[0]**2 + v_vector[1]**2) 
            p_c = uav_model.propulsion_power(v_uav) + uav_model.P_c

            # Tính P_h
            _, p_sol, _, p_batt_fso, p_tx = total_harvested_energy(
                hap_pos, irs_pos, current_uav_pos, duration=1, energy_ratio=alpha_fixed
            )
            p_h = p_sol + p_batt_fso
            denominators.append(p_c - p_h) # Lưu mẫu số lại
            
            # Tính Kênh truyền & Rate tức thời
            h_acc, _, _, _ = get_fso_access(current_uav_pos, car_pos)
            gamma = get_snr(h_acc, p_tx, current_uav_pos)
            r_mbps = data_rate(gamma, FSO_bandwidth) * 1000.0
            instant_rates.append(r_mbps) # Lưu Rate tức thời lại
            
        # BƯỚC 2: Tính Rate trung bình của RIÊNG độ cao z này
        overall_z_rate_mbps = np.mean(instant_rates)
        print(f" -> Z = {z} m có Rate trung bình riêng: {overall_z_rate_mbps:.2f} Mbps")
        
        # BƯỚC 3: Tính mảng EE hoàn chỉnh cho độ cao z
        for t in range(num_timeslots):
            net_power = denominators[t]
            if net_power > 0:
                ee = overall_z_rate_mbps / net_power
            else:
                ee = np.nan
            EE_fixed_alts[z].append(ee)

    print("Tính toán xong! Đang hiển thị đồ thị...")

    # =========================================================
    # 4. VẼ BIỂU ĐỒ 
    # =========================================================
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Vẽ đường Tối ưu (Dữ liệu gốc từ RL)
    ax.plot(times, EE_optimized, color='red', linewidth=3, 
            label='Optimized Trajectory (Dynamic Z)', zorder=5)

    # Vẽ các đường Độ cao cố định
    colors = ['blue', 'darkgreen', 'orange']
    labels = ['Below Cloud', 'Inside Cloud', 'Above Cloud']
    for i, z in enumerate(altitudes):
        ax.plot(times, EE_fixed_alts[z], color=colors[i], linestyle='--', linewidth=2, 
                alpha=0.8, label=rf'Fixed Alt = {z}m ({labels[i]})')

    # Căn chỉnh biểu đồ
    ax.set_xlabel(r'Timeslot ($t$)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Energy Efficiency (Mbps/W)', fontsize=13, fontweight='bold')
    
    ax.set_xlim(0, num_timeslots)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)

    ax.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='black')

    plt.title(rf'Energy Efficiency over Time ($\alpha_{{split}} = {alpha_fixed}$)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()