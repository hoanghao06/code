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
    car_pos = np.array([10, 10, 2])

    # =========================================================
    # 1. ĐỌC DỮ LIỆU TỪ FILE NPY CHO KỊCH BẢN PROPOSED
    # =========================================================
    try:
        energy_path = r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy'
        energy_data = np.load(energy_path, allow_pickle=True).item()
        
        # Power tính chuẩn với energy_ratio = 0.2
        power_from_file = energy_data['solar energy'] + (energy_data['fso energy'] * energy_ratio)
        num_timeslots = len(power_from_file)
        print(f"[+] Đã tải dữ liệu Power ({num_timeslots} timeslots)")
    except Exception as e:
        print(f"[!] Lỗi khi đọc file energy: {e}")
        num_timeslots = 300 
        power_from_file = None

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

    # =========================================================
    # 2. TÍNH TOÁN KÊNH TRUYỀN VÀ NĂNG LƯỢNG THỰC TẾ
    # =========================================================
    pos_above = np.array([400, 100, 2000])
    pos_in    = np.array([400, 100, 500])
    pos_below = np.array([400, 100, 200])

    power_above, power_in, power_below = np.zeros(num_timeslots), np.zeros(num_timeslots), np.zeros(num_timeslots)
    rate_above, rate_in, rate_below = np.zeros(num_timeslots), np.zeros(num_timeslots), np.zeros(num_timeslots)

    print("Đang chạy mô phỏng... Vui lòng đợi...")
    for t in range(num_timeslots):
        # --- TRÊN MÂY ---
        _, p_sol_a, _, p_batt_a, p_tx_a = total_harvested_energy(hap_pos, irs_pos, pos_above, duration=1, energy_ratio=energy_ratio)
        power_above[t] = p_sol_a + p_batt_a
        h_acc_a, _, _, _ = get_fso_access(pos_above, car_pos)
        gamma_a = get_snr(h_acc_a, p_tx_a, pos_above)
        rate_above[t] = data_rate(gamma_a, FSO_bandwidth) 

        # --- TRONG MÂY ---
        _, p_sol_i, _, p_batt_i, p_tx_i = total_harvested_energy(hap_pos, irs_pos, pos_in, duration=1, energy_ratio=energy_ratio)
        power_in[t] = p_sol_i + p_batt_i
        h_acc_i, _, _, _ = get_fso_access(pos_in, car_pos)
        gamma_i = get_snr(h_acc_i, p_tx_i, pos_in)
        rate_in[t] = data_rate(gamma_i, FSO_bandwidth)

        # --- DƯỚI MÂY ---
        _, p_sol_b, _, p_batt_b, p_tx_b = total_harvested_energy(hap_pos, irs_pos, pos_below, duration=1, energy_ratio=energy_ratio)
        power_below[t] = p_sol_b + p_batt_b
        h_acc_b, _, _, _ = get_fso_access(pos_below, car_pos)
        gamma_b = get_snr(h_acc_b, p_tx_b, pos_below)
        rate_below[t] = data_rate(gamma_b, FSO_bandwidth)

    print("Tính toán xong! Đang hiển thị đồ thị...")

    # =========================================================
    # 3. VẼ BIỂU ĐỒ
    # =========================================================
    labels = ['Above Cloud', 'Inside Cloud', 'Below Cloud']
    
    # Dùng np.mean để tính TRUNG BÌNH Công suất (Watt)
    avg_power = [np.mean(power_above), np.mean(power_in), np.mean(power_below)]
    avg_rate = [np.mean(rate_above), np.mean(rate_in), np.mean(rate_below)]

    # Thêm dữ liệu Proposed
    if power_from_file is not None and rate_from_file is not None:
        labels.append('Proposed')
        avg_power.append(np.mean(power_from_file))
        avg_rate.append(np.mean(rate_from_file))

        # --- SẮP XẾP LẠI THỨ TỰ: Above (0) -> Proposed (3) -> Inside (1) -> Below (2) ---
        labels = [labels[0], labels[3], labels[1], labels[2]]
        avg_power = [avg_power[0], avg_power[3], avg_power[1], avg_power[2]]
        avg_rate = [avg_rate[0], avg_rate[3], avg_rate[1], avg_rate[2]]

    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 1. Vẽ ĐƯỜNG cho HARVESTED POWER (Trục trái) ---
    color_line1 = 'blue' 
    
    line1 = ax1.plot(x, avg_power, color=color_line1, marker='o', linestyle='-', 
                     linewidth=3, markersize=8, label='Average Harvested Power')
    
    ax1.set_ylabel('Average Power [W]', color='black', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    
    # Gắn Text lên các điểm của đường Power (hiển thị 1 chữ số thập phân)
    for i, val in enumerate(avg_power):
        ax1.text(x[i], val + max(avg_power)*0.03, f'{val:,.1f}', 
                 ha='center', va='bottom', fontweight='bold', color=color_line1, fontsize=10)

    # --- 2. Vẽ ĐƯỜNG cho DATA RATE (Trục phải) ---
    ax2 = ax1.twinx()
    color_line2 = 'darkred'

    line2 = ax2.plot(x, avg_rate, color=color_line2, marker='s', linestyle='-', 
                     linewidth=3, markersize=8, label='Data Rate')

    ax2.set_ylabel('Data Rate [Gbps]', color='black', fontsize=12, fontweight='bold')

    # Gắn Text lên các điểm của đường Rate
    for i, val in enumerate(avg_rate):
        ax2.text(x[i], val + max(avg_rate)*0.03, f'{val:.2f}', 
                 ha='center', va='bottom', fontweight='bold', color=color_line2, fontsize=10)

    # --- ĐỒNG BỘ 2 TRỤC Y ---
    max_p = max(avg_power)
    max_r = max(avg_rate)
    
    # Tỷ lệ scale đã được trả về 100 do dùng trung bình (Watt) thay vì tổng (Joule)
    scale_ratio = 100.0
    
    # Giữ khoảng trống 25% phía trên cho Legend
    y_max_target = max(max_r, max_p / scale_ratio) * 1.25 
    y_max_target = np.ceil(y_max_target)
    
    ax2.set_ylim(0, y_max_target)
    ax1.set_ylim(0, y_max_target * scale_ratio)

    num_ticks = int(y_max_target + 1)
    ticks_rate = np.linspace(0, y_max_target, num_ticks)
    
    ax2.set_yticks(ticks_rate)
    ax1.set_yticks(ticks_rate * scale_ratio)

    ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    
    # --- GỘP LEGEND VÀ ĐẶT VÀO BÊN TRONG ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper center', ncol=2, 
               fontsize=11, frameon=True, edgecolor='black')

    plt.title(r'Performance with $\alpha_{split} = 0.2$', y=1.02, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()