import numpy as np
from channel import get_fso, get_fso_backhaul
# =====================================================================
# THAM SỐ THU HOẠCH NĂNG LƯỢNG
# =====================================================================

# --- Tham số Năng lượng Mặt trời ---
eta_p = 0.9           # Hiệu suất chuyển đổi quang điện 
A_solar = 2.0         # Diện tích tấm pin mặt trời UAV (m2)
G_r = 1361            # Bức xạ mặt trời trung bình (W/m2)
A_0 = 0.8978          # Giá trị truyền qua khí quyển tối đa
B_0 = 0.2804          # Hệ số dập tắt của không khí (m^-1)
delta = 8000          # Chiều cao thang đo Trái Đất (m)
beta_c = 0.01         # Hệ số suy hao năng lượng mặt trời qua mây (Ví dụ: 0.001 - 0.02)

# --- Tham số Năng lượng FSO ---
V_t = 0.025           # Điện áp nhiệt (Thermal voltage ~ 25mV)
I_0 = 1e-10           # Dòng bão hòa tối của P-D (Dark saturation current, Ampere)
eta_eo = 0.9          # Hiệu suất chuyển đổi quang-điện
R_xi = 0.6            # Độ nhạy P-D (A/W)
A_rx_fso = 0.1        # Diện tích thu FSO (m2) - Lấy theo a_rx ở trên
P_s = 1               # Công suất phát FSO từ HAP (Watts)
B_bias = 0.01         # Dòng DC Bias (Ampere)
H_cloud_min = 500
H_cloud_max = 1000

# =====================================================================
# CÁC HÀM TÍNH TOÁN NĂNG LƯỢNG (POWER)
# =====================================================================

def get_solar_power(z):
    """
    Tính công suất năng lượng mặt trời thu được tại độ cao z (W).
    """
    # Tính độ truyền qua của khí quyển alpha_a(z)
    alpha_a = A_0 - B_0 * np.exp(-z / delta)
    
    if z >= H_cloud_max:
        # UAV bay trên mây
        return eta_p * A_solar * G_r * alpha_a
    elif H_cloud_min <= z < H_cloud_max:
        # UAV bay trong mây (Bị suy hao quãng đường mây từ đỉnh mây xuống UAV)
        # alpha_c = exp(-beta_c * d_cloud)
        alpha_c = np.exp(-beta_c * (H_cloud_max - z))
        return eta_p * A_solar * G_r * alpha_a * alpha_c
    else:
        # UAV bay dưới mây 
        return 0.0

def get_fso_harvested_power(h_total):
    """
    Tính công suất thu hoạch được từ sóng mang FSO 
    """
    # Công thức: P_R = 1.5 * V_t * (eta * h_total * R_xi * A * sqrt(P_s) * B)^2 / I_0
    core_term = eta_eo * h_total * R_xi * A_rx_fso * np.sqrt(P_s) * B_bias
    p_fso = (1.5 * V_t * (core_term ** 2)) / I_0
    return p_fso

# =====================================================================
# HÀM TỔNG HỢP: TÍNH TỔNG NĂNG LƯỢNG TRONG 300 GIÂY
# =====================================================================

def calculate_total_harvested_energy(hap_pos, irs_pos, uav_pos, duration=300):
    """
    Tính tổng năng lượng UAV thu hoạch được trong khoảng thời gian duration (Joules).
    Xét 3 trường hợp: Trên mây, Trong mây, Dưới mây.
    """
    z_uav = uav_pos[-1]
    
    # Tính năng lượng mặt trời
    P_solar = get_solar_power(z_uav)
    
    # Tính kênh FSO và năng lượng FSO
    if z_uav >= H_cloud_max:
        # 1. UAV TRÊN MÂY: Nhận trực tiếp FSO từ HAP (HAP - UAV)
        # Tái sử dụng hàm get_fso nhưng truyền uav_pos thay vì irs_pos
        h_total_fso, _, _, _ = get_fso(hap_pos, uav_pos)
        
    elif H_cloud_min <= z_uav < H_cloud_max:
        # 2. UAV TRONG MÂY: Nhận FSO qua IRS (HAP - IRS - UAV)
        h_hap_irs, _, _, _ = get_fso(hap_pos, irs_pos)
        h_irs_uav, _, _, _ = get_fso_backhaul(uav_pos, irs_pos)
        # Giả sử IRS khuếch đại/phản xạ toàn bộ (Cascade channel)
        h_total_fso = h_hap_irs * h_irs_uav
        
    else:
        # 3. UAV DƯỚI MÂY: Nhận FSO qua IRS (HAP - IRS - UAV)
        h_hap_irs, _, _, _ = get_fso(hap_pos, irs_pos)
        h_irs_uav, _, _, _ = get_fso_backhaul(uav_pos, irs_pos)
        h_total_fso = h_hap_irs * h_irs_uav
        
    P_fso = get_fso_harvested_power(h_total_fso)
    
    # Tổng công suất (W)
    P_total = P_solar + P_fso
    
    # Tổng năng lượng (Joules = Watts * seconds)
    E_total = P_total * duration
    
    return E_total, P_solar, P_fso

# --- Ví dụ chạy thử ---
if __name__ == "__main__":
    hap_pos = np.array([0, 0, 20000])     # HAP ở độ cao 20km
    irs_pos = np.array([0, 0, 80])   # IRS treo ở độ cao 800m (đang ở trong mây)
    
    # Test UAV ở 3 trường hợp khác nhau
    uav_pos_above = np.array([100, 100, 1000])  # Trên mây
    uav_pos_inside = np.array([100, 100, 700])  # Trong mây
    uav_pos_below = np.array([100, 100, 300])   # Dưới mây
    
    for state, pos in zip(["TRÊN MÂY", "TRONG MÂY", "DƯỚI MÂY"], 
                          [uav_pos_above, uav_pos_inside, uav_pos_below]):
        
        E_tot, P_sol, P_fso = calculate_total_harvested_energy(hap_pos, irs_pos, pos, duration=300)
        
        print(f"--- TRƯỜNG HỢP {state} (z = {pos[-1]}m) ---")
        print(f"Công suất Mặt trời: {P_sol:.4f} W")
        print(f"Công suất FSO:      {P_fso:.4e} W") 
        print(f"TỔNG NĂNG LƯỢNG THU ĐƯỢC SAU 300s: {E_tot:.4f} Joules\n")