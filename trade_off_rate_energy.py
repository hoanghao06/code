import numpy as np
import matplotlib.pyplot as plt

# NHÚNG THÊM CÁC HÀM TỪ FILE CHANNEL CỦA BẠN
from channel import total_harvested_energy, get_fso_access, get_snr, data_rate, FSO_bandwidth

def main():
    print("=== ĐANG TÍNH TOÁN VÀ VẼ BIỂU ĐỒ VỚI ALPHA = 0.2 ===")

    # CHỈ SỬ DỤNG DUY NHẤT 1 GIÁ TRỊ ALPHA_SPLIT NHƯ BẠN YÊU CẦU
    energy_ratio = 0.2 
    
    # Tọa độ cơ bản (Sử dụng hệ tọa độ 3D: [X, Y, Z])
    hap_pos = np.array([0, 0, 20000])
    irs_pos = np.array([0, 0, 80])

    # =========================================================
    # 1. ĐỌC DỮ LIỆU TỪ CÁC FILE NPY
    # =========================================================
    # 1.1 Đọc dữ liệu Energy (Kịch bản Proposed)
    try:
        energy_path = r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy'
        energy_data = np.load(energy_path, allow_pickle=True).item()
        
        power_from_file = energy_data['solar energy'] + (energy_data['fso energy'] * energy_ratio)
        num_timeslots = len(power_from_file)
        print(f"[+] Đã tải dữ liệu Power ({num_timeslots} timeslots)")
    except Exception as e:
        print(f"[!] Lỗi khi đọc file energy: {e}")
        num_timeslots = 300 
        power_from_file = None

    # 1.2 Đọc dữ liệu Rate (Kịch bản Proposed)
    try:
        rate_path = r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\rate_3.npy'
        rate_data = np.load(rate_path, allow_pickle=True).item()
        
        if 'mean_rate' in rate_data:
            rate_from_file = rate_data['mean_rate']
        elif 'fso_rate' in rate_data:
            rate_from_file = rate_data['fso_rate']
        else:
            rate_from_file = list(rate_data.values())[0]
            
        print(f"[+] Đã tải dữ liệu Data Rate")
    except Exception as e:
        print(f"[!] Lỗi khi đọc file rate: {e}")
        rate_from_file = None

    # 1.3 Đọc dữ liệu quỹ đạo Xe (Car)
    try:
        car_path = r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\car_3.npy'
        car_data_dict = np.load(car_path, allow_pickle=True).item()
        
        if 'car_0' in car_data_dict:
            car_trajectory = car_data_dict['car_0']
        else:
            car_trajectory = list(car_data_dict.values())[0]
            
        print(f"[+] Đã tải dữ liệu Quỹ đạo xe (Dài: {len(car_trajectory)} bước)")
    except Exception as e:
        print(f"[!] Lỗi khi đọc file car_3.npy: {e}")
        print("[!] Đang sử dụng tọa độ xe cố định làm dự phòng.")
        car_trajectory = np.tile(np.array([10, 10, 2]), (num_timeslots, 1))

    # 1.4 Đọc dữ liệu quỹ đạo UAV
    try:
        uav_path = r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\uav_3.npy'
        uav_data_dict = np.load(uav_path, allow_pickle=True).item()
        
        if 'position' in uav_data_dict:
            uav_trajectory = uav_data_dict['position']
        else:
            uav_trajectory = list(uav_data_dict.values())[0]
            
        print(f"[+] Đã tải dữ liệu Quỹ đạo UAV (Dài: {len(uav_trajectory)} bước)")
    except Exception as e:
        print(f"[!] Lỗi khi đọc file uav_3.npy: {e}")
        print("[!] Đang sử dụng tọa độ UAV cố định làm dự phòng.")
        uav_trajectory = np.tile(np.array([400, 100, 500]), (num_timeslots, 1))

    # =========================================================
    # 2. TÍNH TOÁN KÊNH TRUYỀN VÀ NĂNG LƯỢNG THỰC TẾ
    # =========================================================
    Z_ABOVE = 1800
    Z_IN = 500
    Z_BELOW = 200

    power_above, power_in, power_below = np.zeros(num_timeslots), np.zeros(num_timeslots), np.zeros(num_timeslots)
    rate_above, rate_in, rate_below = np.zeros(num_timeslots), np.zeros(num_timeslots), np.zeros(num_timeslots)

    print("Đang chạy mô phỏng... Vui lòng đợi...")
    for t in range(num_timeslots):
        current_car_pos = car_trajectory[t] if t < len(car_trajectory) else car_trajectory[-1]
        
        current_uav_pos_raw = uav_trajectory[t] if t < len(uav_trajectory) else uav_trajectory[-1]
        uav_x = current_uav_pos_raw[0]
        uav_y = current_uav_pos_raw[1]

        pos_above = np.array([uav_x, uav_y, Z_ABOVE])
        pos_in    = np.array([uav_x, uav_y, Z_IN])
        pos_below = np.array([uav_x, uav_y, Z_BELOW])

        # --- TRÊN MÂY ---
        _, p_sol_a, _, p_batt_a, p_tx_a = total_harvested_energy(hap_pos, irs_pos, pos_above, duration=1, energy_ratio=energy_ratio)
        power_above[t] = p_sol_a + p_batt_a
        h_acc_a, _, _, _ = get_fso_access(pos_above, current_car_pos)
        gamma_a = get_snr(h_acc_a, p_tx_a, pos_above)
        rate_above[t] = data_rate(gamma_a, FSO_bandwidth) 

        # --- TRONG MÂY ---
        _, p_sol_i, _, p_batt_i, p_tx_i = total_harvested_energy(hap_pos, irs_pos, pos_in, duration=1, energy_ratio=energy_ratio)
        power_in[t] = p_sol_i + p_batt_i
        h_acc_i, _, _, _ = get_fso_access(pos_in, current_car_pos)
        gamma_i = get_snr(h_acc_i, p_tx_i, pos_in)
        rate_in[t] = data_rate(gamma_i, FSO_bandwidth)

        # --- DƯỚI MÂY ---
        _, p_sol_b, _, p_batt_b, p_tx_b = total_harvested_energy(hap_pos, irs_pos, pos_below, duration=1, energy_ratio=energy_ratio)
        power_below[t] = p_sol_b + p_batt_b
        h_acc_b, _, _, _ = get_fso_access(pos_below, current_car_pos)
        gamma_b = get_snr(h_acc_b, p_tx_b, pos_below)
        rate_below[t] = data_rate(gamma_b, FSO_bandwidth)

    print("Tính toán xong! Đang hiển thị đồ thị...")

    # =========================================================
    # 3. VẼ BIỂU ĐỒ 
    # =========================================================
    labels = ['Above Cloud', 'Inside Cloud', 'Below Cloud']
    
    # Tính toán phần trăm vượt ngưỡng (threshold = 4.0 Gbps)
    def calc_percentage_above_threshold(rate_array, threshold=3.8):
        arr = np.array(rate_array)
        return (np.sum(arr >= threshold) / len(arr)) * 100.0

    avg_power = [np.mean(power_above), np.mean(power_in), np.mean(power_below)]
    pct_rate = [
        calc_percentage_above_threshold(rate_above), 
        calc_percentage_above_threshold(rate_in), 
        calc_percentage_above_threshold(rate_below)
    ]

    if power_from_file is not None and rate_from_file is not None:
        labels.append('Proposed')
        avg_power.append(np.mean(power_from_file))
        pct_rate.append(calc_percentage_above_threshold(rate_from_file))

        labels = [labels[0], labels[3], labels[1], labels[2]]
        avg_power = [avg_power[0], avg_power[3], avg_power[1], avg_power[2]]
        pct_rate = [pct_rate[0], pct_rate[3], pct_rate[1], pct_rate[2]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 1. Vẽ CỘT cho HARVESTED POWER (Trục trái) ---
    color_bar1 = 'steelblue' 
    bars1 = ax1.bar(x - width/2, avg_power, width, color=color_bar1, 
                    label='Average Harvested Power', edgecolor='black', hatch='////')
    
    ax1.set_ylabel('Average Power [W]', color='black', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    
    # Gắn Text lên các cột của Power 
    for i, val in enumerate(avg_power):
        ax1.text(x[i] - width/2, val + max(avg_power)*0.02, f'{val:,.1f}', 
                 ha='center', va='bottom', fontweight='bold', color='black', fontsize=10)

    # --- 2. Vẽ CỘT cho RATE THRESHOLD (Trục phải) ---
    ax2 = ax1.twinx()
    color_bar2 = 'tab:red' 

    bars2 = ax2.bar(x + width/2, pct_rate, width, color=color_bar2, 
                    label='Rate $\geq$ 3.8 Gbps (%)', edgecolor='black', hatch=r'\\\\')

    ax2.set_ylabel('Rate $\geq$ 3.8 Gbps [%]', color='black', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')

    # Gắn Text lên các cột của Rate 
    for i, val in enumerate(pct_rate):
        ax2.text(x[i] + width/2, val + 2, f'{val:.1f}%', 
                 ha='center', va='bottom', fontweight='bold', color='black', fontsize=10)

    # --- ĐỒNG BỘ 2 TRỤC Y ---
    max_p = max(avg_power)
    y_max_power = np.ceil(max_p / 100.0) * 100.0 
    if y_max_power < max_p * 1.15:
        y_max_power += 100.0
    
    num_ticks = 7
    ax2.set_ylim(0, 100)
    ax2.set_yticks(np.linspace(0, 100, num_ticks))

    ax1.set_ylim(0, y_max_power)  
    ax1.set_yticks(np.linspace(0, y_max_power, num_ticks))

    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    
    # --- GỘP LEGEND VÀ ĐẶT RA BÊN NGOÀI KHUNG (PHÍA TRÊN) ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    # Đưa legend ra ngoài bằng bbox_to_anchor
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, 
               fontsize=11, frameon=True, edgecolor='black')

    # Đẩy title lên cao hơn một chút để không đè vào legend
    plt.title(r'Performance with $\alpha_{split} = 0.2$', y=1.15, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()