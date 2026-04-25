import numpy as np
import matplotlib.pyplot as plt

# NHÚNG THÊM CÁC HÀM TỪ FILE CHANNEL CỦA BẠN
from channel import total_harvested_energy, get_fso_access, get_snr, data_rate, FSO_bandwidth

def main():
    print("=== TÍNH TOÁN TRADE-OFF THEO ALPHA_SPLIT (CÓ PROPOSED) ===")

    # Khởi tạo các giá trị alpha_split để khảo sát (Từ 0.1 đến 0.9)
    alphas = np.arange(0.1, 1.0, 0.1) 
    num_timeslots = 300
    
    # Tọa độ cơ bản
    hap_pos = np.array([500, 500, 20000])
    irs_pos = np.array([0, 0, 80])
    
    pos_above = np.array([600, 0, 2000])
    pos_in    = np.array([600, 0, 700])
    pos_below = np.array([600, 0, 200])

    # =========================================================
    # 0. LOAD TỌA ĐỘ XE TỪ FILE
    # =========================================================
    try:
        car_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\car_3.npy', allow_pickle=True).item()
        # Mặc định chọn quỹ đạo của 'car_0', bạn có thể đổi thành 'car_1' hoặc 'car_2' tùy ý
        car_trajectory = car_data['car_0'] 
        print("[+] Đã tải dữ liệu quỹ đạo xe thành công từ car_3.npy")
    except Exception as e:
        print(f"[!] Lỗi đọc file car_3.npy: {e}")
        # Fallback: Nếu lỗi không đọc được file, xe đứng yên tại gốc tọa độ
        car_trajectory = [np.array([0, 0, 0])] * num_timeslots

    # Mảng lưu kết quả trung bình cho từng alpha
    avg_power_above, avg_power_in, avg_power_below, avg_power_prop = [], [], [], []
    avg_rate_above, avg_rate_in, avg_rate_below, avg_rate_prop = [], [], [], []

    # =========================================================
    # 1. XỬ LÝ DỮ LIỆU PROPOSED TỪ FILE
    # =========================================================
    has_proposed = False
    try:
        energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
        solar_prop = np.array(energy_data['solar energy'])
        fso_prop = np.array(energy_data['fso energy'])
        
        rate_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\rate_3.npy', allow_pickle=True).item()
        fso_rate_old = np.array(rate_data['fso_rate'])
        
        # --- Back-calculation logic (Kỹ thuật suy ngược SNR) ---
        gamma_old = 2**(2 * fso_rate_old) - 1
        has_proposed = True
        print("[+] Đã tải dữ liệu Proposed (Tối ưu tại alpha = 0.2). Đang chuẩn bị nội suy...")
    except Exception as e:
        print(f"[!] Lỗi đọc file Proposed: {e}")

    # =========================================================
    # 2. CHẠY MÔ PHỎNG VÀ NỘI SUY THEO TỪNG ALPHA
    # =========================================================
    print(f"Đang chạy mô phỏng cho {len(alphas)} giá trị alpha_split... Vui lòng đợi...")
    
    for alpha in alphas:
        # --- TÍNH TOÁN PROPOSED (Nội suy toán học siêu tốc) ---
        if has_proposed:
            p_mean = np.mean(solar_prop + (fso_prop * alpha))
            avg_power_prop.append(p_mean)
            
            ratio_tx = (1 - alpha) / 0.8
            gamma_new = gamma_old * (ratio_tx ** 2)
            
            rate_new = 0.5 * np.log2(1 + gamma_new)
            r_mean = np.mean(rate_new) 
            avg_rate_prop.append(r_mean)

        # --- TÍNH TOÁN BENCHMARK THÔNG THƯỜNG ---
        p_above, p_in, p_below = 0, 0, 0
        r_above, r_in, r_below = 0, 0, 0
        
        for t in range(num_timeslots):
            # Lấy tọa độ xe tại thời điểm t. 
            # Dùng min(t, len-1) để tránh out-of-bounds nếu file npy ngắn hơn num_timeslots
            car_pos = car_trajectory[min(t, len(car_trajectory) - 1)]

            # TRÊN MÂY
            _, p_sol_a, _, p_batt_a, p_tx_a = total_harvested_energy(hap_pos, irs_pos, pos_above, duration=1, energy_ratio=alpha)
            p_above += (p_sol_a + p_batt_a)
            h_acc_a, _, _, _ = get_fso_access(pos_above, car_pos)
            gamma_a = get_snr(h_acc_a, p_tx_a, pos_above)
            r_above += data_rate(gamma_a, FSO_bandwidth)

            # TRONG MÂY
            _, p_sol_i, _, p_batt_i, p_tx_i = total_harvested_energy(hap_pos, irs_pos, pos_in, duration=1, energy_ratio=alpha)
            p_in += (p_sol_i + p_batt_i)
            h_acc_i, _, _, _ = get_fso_access(pos_in, car_pos)
            gamma_i = get_snr(h_acc_i, p_tx_i, pos_in)
            r_in += data_rate(gamma_i, FSO_bandwidth)

            # DƯỚI MÂY
            _, p_sol_b, _, p_batt_b, p_tx_b = total_harvested_energy(hap_pos, irs_pos, pos_below, duration=1, energy_ratio=alpha)
            p_below += (p_sol_b + p_batt_b)
            h_acc_b, _, _, _ = get_fso_access(pos_below, car_pos)
            gamma_b = get_snr(h_acc_b, p_tx_b, pos_below)
            r_below += data_rate(gamma_b, FSO_bandwidth)

        avg_power_above.append(p_above / num_timeslots)
        avg_power_in.append(p_in / num_timeslots)
        avg_power_below.append(p_below / num_timeslots)

        avg_rate_above.append(r_above / num_timeslots)
        avg_rate_in.append(r_in / num_timeslots)
        avg_rate_below.append(r_below / num_timeslots)
        
        print(f" [+] Xong mô phỏng alpha = {alpha:.1f}")

    # =========================================================
    # 3. VẼ BIỂU ĐỒ TRADE-OFF HOÀN CHỈNH
    # =========================================================
    fig, ax1 = plt.subplots(figsize=(12, 7.5))

    # --- TRỤC Y TRÁI: AVERAGE TOTAL POWER ---
    if has_proposed:
        ax1.plot(alphas, avg_power_prop, color='purple', marker='*', linestyle='-', linewidth=3, markersize=12, label='Power (Proposed)')
    
    ax1.plot(alphas, avg_power_above, color='blue', marker='o', linestyle='-', linewidth=2, markersize=6, label='Power (Above)')
    ax1.plot(alphas, avg_power_in, color='green', marker='o', linestyle='-', linewidth=2, markersize=6, label='Power (Inside)')
    ax1.plot(alphas, avg_power_below, color='red', marker='o', linestyle='-', linewidth=2, markersize=6, label='Power (Below)')

    ax1.set_xlabel(r'$\alpha_{split}$ (Power Splitting Ratio)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Total Power (Solar + FSO) [W]', color='black', fontsize=12, fontweight='bold')
    ax1.set_xticks(alphas)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- TRỤC Y PHẢI: DATA RATE ---
    ax2 = ax1.twinx()
    
    if has_proposed:
        ax2.plot(alphas, avg_rate_prop, color='purple', marker='*', linestyle='--', linewidth=3, markersize=12, label='Rate (Proposed)')
        
    ax2.plot(alphas, avg_rate_above, color='blue', marker='s', linestyle='--', linewidth=2, markersize=6, label='Rate (Above)')
    ax2.plot(alphas, avg_rate_in, color='green', marker='s', linestyle='--', linewidth=2, markersize=6, label='Rate (Inside)')
    ax2.plot(alphas, avg_rate_below, color='red', marker='s', linestyle='--', linewidth=2, markersize=6, label='Rate (Below)')

    ax2.set_ylabel('Data Rate [Gbps]', color='black', fontsize=12, fontweight='bold')

    # --- GỘP LEGEND ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    num_cols = 4 if has_proposed else 3
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, 
               loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=num_cols, 
               fontsize=10, frameon=True, edgecolor='black')

    plt.title(r'Harvested Power vs Data Rate over $\alpha_{split}$', y=1.28, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
