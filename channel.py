import numpy as np
import scipy.integrate as integrate
from scipy import special
from scipy.special import erf
import pandas as pd                                  
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import fsolve

lamda = 1550e-9
k = 2*np.pi/lamda
v_wind = 21 #[m/s]
Cn2_0 = 1.7e-14
a_irs = 1
a_rx = 0.1 # [m]
ps_jitter = 2 # [m]
sigma_p = 0.1
CLWC = 10 ** -2 # mật độ nước lỏng trong mây (g/m3)
r_cloud = 3.33 # Bán kính trung bình giọt mây Stratus (μm) 
rho_water = 1.0 # Mật độ nước (g/cm3)
N_c = (CLWC / ((4/3) * np.pi * (r_cloud**3) * rho_water * 1e-6)) 
# Tọa độ lớp mây
H_cloud_min = 500 # [m]
H_cloud_max = 1000 # [m]
# --- Tham số hệ thống ---
c_speed = 3e8                  # Tốc độ ánh sáng (m/s)
q_charge = 1.6e-19             # Điện tích electron (C)
E_0 = 0.26961                  # Bức xạ quang phổ mặt trời nền (kW/m2.um)
# B_0 = 250e9                    # Băng thông quang (Optical bandwidth) (Hz)
k_B = 1.38e-23                 # Hằng số Boltzmann (J/K)
R_L = 50                       # Điện trở tải (Load resistance) (Ohm)
T_thermal = 298
FSO_bandwidth = 1 # GHz             
Delta_f = FSO_bandwidth*1e9 / 2 # Efficient BW
Delta_lamda = (FSO_bandwidth*1e9 * (lamda ** 2)) / c_speed
number_car = 3                 # Số lượng xe dưới mặt đất
# =====================================================================
# THAM SỐ THU HOẠCH NĂNG LƯỢNG
# =====================================================================

# --- Tham số Năng lượng Mặt trời ---
eta_p = 0.9           # Hiệu suất chuyển đổi quang điện 
A_solar = 0.5         # Diện tích tấm pin mặt trời UAV (m2)
G_r = 1361            # Bức xạ mặt trời trung bình (W/m2)
A_0_solar = 0.8978    # Giá trị truyền qua khí quyển tối đa 
B_0_solar = 0.2804    # Hệ số dập tắt của không khí (m^-1)
delta = 8000          # Chiều cao thang đo Trái Đất (m)
beta_c = 0.01         # Hệ số suy hao năng lượng mặt trời qua mây (0.001 - 0.02)
beta_a = 0.5

# --- Tham số Năng lượng FSO ---
V_t = 0.025           # Điện áp nhiệt (Thermal voltage ~ 25mV)
I_0 = 1e-9           # Dòng bão hòa tối của P-D (Dark saturation current, Ampere)
eta_eo = 0.9          # Hiệu suất chuyển đổi quang-điện
R_xi = 0.6            # Độ nhạy P-D (A/W)
P_s = (10 ** (25 / 10)) / 1000               # Công suất phát FSO từ HAP (Watts)
B_bias = 0.04         # Dòng DC Bias (A)
irs_gain_dB = 2
irs_gain = 10 ** (irs_gain_dB / 10) 

def transmittance(file_path, environment='Nông thôn (T)'):
    df = pd.read_excel(file_path)
    df = df[df['h_sensor (km)'].astype(str).str.isnumeric()].copy()
    df['V (km)'] = df['V (km)'].ffill().astype(float)
    df['h_sensor (km)'] = df['h_sensor (km)'].astype(float)
    df[environment] = df[environment].astype(float)
    points = df[['V (km)', 'h_sensor (km)']].values
    values = df[environment].values
    return LinearNDInterpolator(points, values)

csv_file_path = r"C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\env\data\data_khihau\Tropical.csv.xlsx"
env_type = 'Nông thôn (T)'
t_interpolator = transmittance(csv_file_path, env_type)

# -------------------
#=========== HAP-IRS channel ===============
# -------------------

def get_fso(hap_pos: np.ndarray, irs_pos: np.ndarray) -> np.ndarray:
# uav_pos là tọa độ UAV
# irs_pos là tọa độ irs
    
    H_HAP = hap_pos[-1]
    H_IRS = irs_pos[-1]
# Tính góc Zenith 
    delta_h = np.abs(H_IRS - H_HAP)
    r_dis = np.linalg.norm(irs_pos[0:-1] - hap_pos[0:-1])
    r_dis = np.maximum(r_dis, 1e-6) # Tránh lỗi chia cho 0
    
    phi_beam_z = np.arctan(r_dis / delta_h) 
    sec = 1 / np.cos(phi_beam_z)
    L = delta_h * sec

# -------------------
# Atmospheric attenuation
# -------------------

    # Tìm phần giao nhau giữa tia laser và lớp mây theo trục Z
    
    Z_min = np.maximum(H_IRS, H_cloud_min)
    Z_max = np.minimum(H_HAP, H_cloud_max)
    
    # Độ dày lớp mây (theo phương thẳng đứng) mà tia đi qua
    delta_hc = np.maximum(Z_max - Z_min, 0)
    
    L_c = delta_hc * sec # quãng đường thực tế xuyên qua mây
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    q = 1.6
    cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
    cloud_coefficient = cloud_coefficient_dB /(1e4 * np.log10(np.exp(1)))
    clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
    clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
    L_clear = delta_h - L_c
    hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

# -------------------
# Cn^2 model (HVB)
# -------------------

    def Cn2(h):

        term1 = 0.00594*(v_wind/27)**2*(1e-5*h)**10*np.exp(-h/1000)
        term2 = 2.7e-16*np.exp(-h/1500)
        term3 = Cn2_0*np.exp(-h/100)

        return term1 + term2 + term3
# -------------------
# Rytov variance
# -------------------
    def integrand(h):
        return Cn2(h) * np.abs(h - H_IRS) ** (5/6)
    integral_value,_ = integrate.quad(integrand, H_IRS, H_HAP)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec) ** (11/6) * integral_value

# -------------------
# Log-normal turbulence
# -------------------

    def lognormal_turbulence():

        mu = -sigma_R2/2
        sigma = np.sqrt(sigma_R2)

        ha = np.random.lognormal(mu,sigma)

        return ha
    
    ha = lognormal_turbulence()

# -------------------
# Pointing error loss
# -------------------
    theta_HI = 1e-3 #rad
    rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5) # [m]
    omega_0 = 2*lamda / (np.pi*theta_HI)
    omega_l = omega_0 * \
              np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) *
                      ((lamda * L) /
                       (np.pi * omega_0 ** 2)) ** 2)
    v = (a_irs * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
    A_0 = special.erf(v)**2
    den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
    rho_l = 0
    # rho_l = np.random.rayleigh(ps_jitter)
    hs = A_0*np.exp(-2*(rho_l/omega_l2_eeq)**2)
# -------------------
# Total channel HAP-IRS
# -------------------

    h_total = hc*ha*hs

    return h_total, hc, ha, hs

# -------------------
#=========== IRS-UAV channel ===============
# -------------------

def get_fso_backhaul(uav_pos: np.ndarray, irs_pos: np.ndarray) -> np.ndarray:
# uav_pos là tọa độ UAV
# irs_pos là tọa độ irs
    
    H_UAV = uav_pos[-1]
    H_IRS = irs_pos[-1]
# Tính góc Zenith 
    delta_h = np.maximum(np.abs(H_UAV - H_IRS), 1e-6)
    r_dis = np.linalg.norm(irs_pos[0:-1] - uav_pos[0:-1])
    r_dis = np.maximum(r_dis, 1e-6) # Tránh lỗi chia cho 0
    
    phi_beam_z = np.arctan(r_dis / delta_h) 
    sec = 1 / np.cos(phi_beam_z)
    L = delta_h * sec

# -------------------
# Atmospheric attenuation
# -------------------

    # Tìm phần giao nhau giữa tia laser và lớp mây theo trục Z
    
    Z_min = np.maximum(H_IRS, H_cloud_min)
    Z_max = np.minimum(H_UAV, H_cloud_max)
    
    # Độ dày lớp mây (theo phương thẳng đứng) mà tia đi qua
    delta_hc = np.maximum(Z_max - Z_min, 0)
    
    L_c = delta_hc * sec # quãng đường thực tế xuyên qua mây
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    if fso_visibility > 6:
        # Clear sky
        q = 1.6
    elif 1 < fso_visibility <= 6:
        # Hazy (sương mù nhẹ / mù)
        q = 0.16 * fso_visibility + 0.34
    else: 
        # Foggy (sương mù đặc / mây) (fso_visibility <= 1)
        q = fso_visibility - 0.5
        if q < 0:
            q = 0
    cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
    cloud_coefficient = cloud_coefficient_dB /(1e4 * np.log10(np.exp(1)))
    clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
    clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
    L_clear = delta_h - L_c
    hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

# -------------------
# Rytov variance
# -------------------

    def integrand (h):
        return (0.00594 * (v_wind / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
            2.7 * 10 ** -16 * np.exp(-h / 1500) + Cn2_0 * np.exp(-h / 100)) * \
                np.abs(h - H_UAV) ** (5/6)
    z_lower = np.minimum(H_IRS, H_UAV)
    z_upper = np.maximum(H_IRS, H_UAV)
    integral_value,_ = integrate.quad(integrand, z_lower, z_upper)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

# -------------------
# Gamma-Gamma turbulence
# -------------------

    denom_alpha = np.maximum(np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1, 1e-6)
    alpha_f = 1.0 / denom_alpha
    denom_beta = np.maximum(np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1, 1e-6)
    beta_f = 1.0 / denom_beta
    X = np.random.gamma(shape=alpha_f, scale=1.0/alpha_f)
    Y = np.random.gamma(shape=beta_f, scale=1.0/beta_f)
    ha = X * Y

# -------------------
# Pointing error loss
# -------------------
    omega_0 = 0.01
    rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5) # [m]
    omega_l = omega_0 * \
              np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) *
                      ((lamda * L) /
                       (np.pi * omega_0 ** 2)) ** 2)
    v = (a_rx * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
    A_0 = special.erf(v)**2
    den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den

# Mô phỏng dao động theo phân bố Rice

    x_rpe = np.random.normal(0, sigma_p)
    y_rpe = np.random.normal(0, sigma_p)
    r_pe2 = x_rpe**2 + y_rpe**2
    hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))

# -------------------
# Total channel IRS-UAV
# -------------------

    h_total = hc*ha*hs

    return h_total, hc, ha, hs

# -------------------
#=========== UAV-Vehices channel ===============
# -------------------

def get_fso_access(uav_pos: np.ndarray, car_pos: np.ndarray) -> np.ndarray:
# uav_pos là tọa độ UAV
# irs_pos là tọa độ irs
    
    H_UAV = uav_pos[-1]
    H_CAR = car_pos[-1]
# Tính góc Zenith 
    delta_h = np.maximum(np.abs(H_CAR - H_UAV), 1e-6) # Thêm np.maximum để chặn delta_h = 0
    r_dis = np.linalg.norm(uav_pos[0:-1] - car_pos[0:-1])
    r_dis = np.maximum(r_dis, 1e-6) # Tránh lỗi chia cho 0
    
    phi_beam_z = np.arctan(r_dis / delta_h) 
    sec = 1 / np.cos(phi_beam_z)
    L = delta_h * sec

# -------------------
# Atmospheric attenuation
# -------------------

    # Tìm phần giao nhau giữa tia laser và lớp mây theo trục Z
    
    Z_min = np.maximum(H_CAR, H_cloud_min)
    Z_max = np.minimum(H_UAV, H_cloud_max)
    
    # Độ dày lớp mây (theo phương thẳng đứng) mà tia đi qua
    delta_hc = np.maximum(Z_max - Z_min, 0)
    
    L_c = delta_hc * sec # quãng đường thực tế xuyên qua mây
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    if fso_visibility > 6:
        # Clear sky
        q = 1.6
    elif 1 < fso_visibility <= 6:
        # Hazy (sương mù nhẹ / mù)
        q = 0.16 * fso_visibility + 0.34
    else: 
        # Foggy (sương mù đặc / mây) (fso_visibility <= 1)
        q = fso_visibility - 0.5
        if q < 0:
            q = 0
    cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
    cloud_coefficient = cloud_coefficient_dB /(1e4 * np.log10(np.exp(1)))
    clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
    clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
    L_clear = delta_h - L_c
    hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

# -------------------
# Rytov variance
# -------------------

    def integrand (h):
        return (0.00594 * (v_wind / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
            2.7 * 10 ** -16 * np.exp(-h / 1500) + Cn2_0 * np.exp(-h / 100)) * \
                np.abs(h - H_UAV) ** (5/6)
    z_lower = np.minimum(H_CAR, H_UAV)
    z_upper = np.maximum(H_CAR, H_UAV)
    integral_value,_ = integrate.quad(integrand, z_lower, z_upper)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

# -------------------
# Gamma-Gamma turbulence
# -------------------

    denom_alpha = np.maximum(np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1, 1e-6)
    alpha_f = 1.0 / denom_alpha
    denom_beta = np.maximum(np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1, 1e-6)
    beta_f = 1.0 / denom_beta
    X = np.random.gamma(shape=alpha_f, scale=1.0/alpha_f)
    Y = np.random.gamma(shape=beta_f, scale=1.0/beta_f)
    ha = X * Y

# -------------------
# Pointing error loss
# -------------------
    omega_0 = 0.01
    rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5) # [m]
    omega_l = omega_0 * \
              np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) *
                      ((lamda * L) /
                       (np.pi * omega_0 ** 2)) ** 2)
    v = (a_rx * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
    A_0 = special.erf(v)**2
    den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den

# Mô phỏng dao động theo phân bố Rice

    x_rpe = np.random.normal(0, sigma_p)
    y_rpe = np.random.normal(0, sigma_p)
    r_pe2 = x_rpe**2 + y_rpe**2
    hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))
    h_total = hc*ha*hs
    return h_total, hc, ha, hs

# =====================================================================
# CÁC HÀM TÍNH TOÁN NĂNG LƯỢNG (POWER)
# =====================================================================

def get_solar_power(z):
    """
    Tính công suất năng lượng mặt trời thu được tại độ cao z (W).
    """
    # Tính độ truyền qua của khí quyển alpha_a(z)
    alpha_a = A_0_solar - B_0_solar * np.exp(-z / delta)
    
    if z >= H_cloud_max:
        # UAV bay trên mây
        return eta_p * A_solar * G_r * alpha_a
    elif H_cloud_min <= z < H_cloud_max:
        # UAV bay trong mây 
        alpha_c = np.exp(-beta_c * (H_cloud_max - z))
        return eta_p * A_solar * G_r * alpha_a * alpha_c
    else:
        # UAV bay dưới mây 
        alpha_c = np.exp(-beta_c * (H_cloud_max - H_cloud_min))
        alpha_d = np.exp(-beta_a * (H_cloud_min - z))
        return eta_p * A_solar * G_r * alpha_a * alpha_c * alpha_d

def get_fso_harvested_power(h_total, gain_factor=1):
    """
    Tính công suất thu hoạch được từ sóng mang FSO 
    """
    # Công thức: P_R = 1.5 * V_t * (eta * h_total * R_xi * A * sqrt(P_s) * B)^2 / I_0
    core_term = eta_eo * h_total * R_xi * a_rx * np.sqrt(P_s) * B_bias
    p_fso = ((0.75 * V_t * (core_term ** 2)) / I_0) * gain_factor
    return p_fso

# =====================================================================
# HÀM TỔNG HỢP: TÍNH TỔNG NĂNG LƯỢNG TRONG DURATION
# =====================================================================

def total_harvested_energy(hap_pos, irs_pos, uav_pos, duration=300, energy_ratio=0.2):
    """
    Tính tổng năng lượng UAV thu hoạch được trong khoảng thời gian duration (Joules).
    Xét 3 trường hợp: Trên mây, Trong mây, Dưới mây.
    """
    z_uav = uav_pos[-1]
    
    P_solar = get_solar_power(z_uav) 
    
    # Tính kênh FSO và năng lượng FSO
    if z_uav >= H_cloud_max:
        # 1. UAV TRÊN MÂY: Nhận trực tiếp FSO từ HAP (HAP - UAV)
        h_total_fso, _, _, _ = get_fso(hap_pos, uav_pos)
        P_fso = get_fso_harvested_power(h_total_fso, gain_factor=1)
        
    elif H_cloud_min <= z_uav < H_cloud_max:
        # 2. UAV TRONG MÂY: Nhận FSO qua IRS (HAP - IRS - UAV)
        h_hap_irs, _, _, _ = get_fso(hap_pos, irs_pos)
        h_irs_uav, _, _, _ = get_fso_backhaul(uav_pos, irs_pos)
        h_total_fso = h_hap_irs * h_irs_uav
        P_fso = get_fso_harvested_power(h_total_fso, gain_factor=irs_gain)
        
    else:
        # 3. UAV DƯỚI MÂY: Nhận FSO qua IRS (HAP - IRS - UAV)
        h_hap_irs, _, _, _ = get_fso(hap_pos, irs_pos)
        h_irs_uav, _, _, _ = get_fso_backhaul(uav_pos, irs_pos)
        h_total_fso = h_hap_irs * h_irs_uav
        P_fso = get_fso_harvested_power(h_total_fso, gain_factor=irs_gain)

    P_battery_fso = P_fso * energy_ratio               # 20% của P_fso sạc vào pin
    P_transmit_total = P_fso * (1 - energy_ratio)      # 80% của P_fso để phát tín hiệu
    P_transmit_per_car = P_transmit_total / number_car # Chia đều cho số xe
    
    # Tổng năng lượng (Joules = Watts * seconds)
    E_total = (P_solar + P_battery_fso) * duration
    
    return E_total, P_solar, P_fso, P_battery_fso, P_transmit_per_car
# -------------------
# Total channel UAV-Vehices and SNR 
# -------------------
def get_snr(h_total, P_transmit_total, uav_pos):
    H_UAV = uav_pos[-1]
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    sigma2_FSO = 1e-17
    R_acc = 1
    if fso_visibility < 18.0:
        V_snapped = 5.0
    else:
        V_snapped = 30.0
    T_trans_raw = t_interpolator(V_snapped, H_UAV/1000)
    
    # THÊM ĐOẠN NÀY ĐỂ BẪY LỖI: Nếu nội suy ra NaN (do vượt quá dữ liệu file Excel)
    if np.isnan(T_trans_raw):
        # Fallback về 1 giá trị mặc định hợp lý (ví dụ 0.5) 
        # Hoặc bạn có thể tự chỉnh con số này dựa theo dữ liệu Excel của bạn
        T_trans = 0.5 
    else:
        T_trans = float(T_trans_raw)
    P_b = (E_0*1e9) * T_trans * np.pi * (a_rx ** 2) * Delta_lamda
    shot_noise_term = 2 * q_charge * R_xi * P_b
    thermal_noise_term = (4 * k_B * T_thermal / R_L) * Delta_f
    sigma_N_square = shot_noise_term + thermal_noise_term
    gamma_F = (P_transmit_total * R_acc * h_total)**2 / ((sigma2_FSO + sigma_N_square) * FSO_bandwidth)   
    return gamma_F

# =====================================================================
# DATA RATE
# =====================================================================
def data_rate(gamma_F, bandwidth_ghz):

    B_hz = bandwidth_ghz 
    data_rate_bps = (B_hz / 2) * np.log2(1 + gamma_F) # Gbps
    
    return data_rate_bps

# =====================================================================
# CLASS MÔ HÌNH NĂNG LƯỢNG TIÊU THỤ 
# =====================================================================
class UAVEnergyModel:
    def __init__(self):
        # --- CÁC THÔNG SỐ TỪ BẢNG I (TABLE I) ---
        self.W = 100.45         # Trọng lượng UAV (N) (2.45 là trọng lượng tấm pin mặt trời dạng vải dệt)
        self.rho = 1.225        # Mật độ không khí (kg/m^3)
        self.R = 0.5            # Bán kính cánh quạt (m)
        self.A = 0.79           # Diện tích đĩa cánh quạt (m^2)
        self.U_tip = 200.0      # Tốc độ mũi cánh quạt (m/s)
        self.s = 0.05           # Độ đặc của rotor (Rotor solidity)
        self.d_0 = 0.3          # Hệ số cản thân máy bay (Fuselage drag ratio)
        self.v_0 = 7.2          # Vận tốc cảm ứng trung bình khi bay lơ lửng (m/s)
        self.delta = 0.012      # Hệ số cản biên dạng (Profile drag coefficient)
        self.k = 0.1            # Hệ số hiệu chỉnh công suất cảm ứng
        
        # --- TÍNH TOÁN P_0 VÀ P_i ---
        self.P_0 = (self.delta / 8) * self.rho * self.s * self.A * (self.U_tip ** 3)
        self.P_i = (1 + self.k) * (self.W ** 1.5) / np.sqrt(2 * self.rho * self.A)
        self.P_c = 50.0 # Công suất tiêu thụ cho mạch truyền thông (W)

    def velocity_3d(self, v_x, v_y, v_z):
        """Tính độ lớn vận tốc V từ các vector vận tốc."""
        return np.sqrt(v_x**2 + v_y**2 + v_z**2)

    def propulsion_power(self, V):
        """Tính toán công suất đẩy P(V)"""
        V = np.maximum(V, 0.0)
        term1 = self.P_0 * (1 + (3 * V**2) / (self.U_tip**2))
        inner_sqrt = np.sqrt(1 + (V**4) / (4 * self.v_0**4))
        term2 = self.P_i * np.sqrt(inner_sqrt - (V**2) / (2 * self.v_0**2))
        term3 = 0.5 * self.d_0 * self.rho * self.s * self.A * (V**3)
        return term1 + term2 + term3

    def total_energy(self, velocities, dt, include_communication=True):
        """Tính tổng năng lượng tiêu thụ (Joule)"""
        P_prop = self.propulsion_power(velocities)
        E_prop = np.sum(P_prop) * dt
        if include_communication:
            E_comm = self.P_c * len(velocities) * dt
            return E_prop + E_comm
        return E_prop

def uav_lifespan(uav_model, velocity, p_solar, p_battery_fso, e_b_wh=276.0):
    """
    Tính toán tuổi thọ của UAV dựa trên công suất tiêu thụ và thu hoạch.
    
    Tham số:
    - uav_model: Thực thể của class UAVEnergyModel đã có trong code.
    - velocity: Vận tốc hiện tại của UAV (m/s).
    - p_solar: Công suất thu hoạch từ mặt trời (Watts).
    - p_battery_fso: Công suất FSO thực tế nạp vào pin (20% của P_fso).
    - e_b_wh: Năng lượng pin khả dụng (mặc định 276 Wh theo bài báo).
    
    Trả về:
    - lifespan_seconds: Tuổi thọ UAV tính bằng giây.
    """
    # 1. Tính tổng công suất tiêu thụ Pc = P_propulsion + P_communication
    # P_propulsion tính từ hàm propulsion_power trong code của bạn
    p_propulsion = uav_model.propulsion_power(velocity)
    p_c = p_propulsion + uav_model.P_c # [cite: 969, 970]
    
    # 2. Tính tổng công suất thu hoạch được nạp vào pin P_H
    p_h = p_solar + p_battery_fso
    
    # 3. Kiểm tra trạng thái cân bằng năng lượng
    if p_h >= p_c:
        return float('inf')
    
    # 4. Tính Lifespan (Ls) theo công thức (34)
    e_b_joules = e_b_wh * 3600 
    lifespan_seconds = e_b_joules / (p_c - p_h) 
    return lifespan_seconds, p_c, p_h

# if __name__ == "__main__":
#     print("=== ĐANG TEST KÊNH TRUYỀN VÀ TUỔI THỌ (LIFESPAN) UAV ===")

#     uav_energy_model = UAVEnergyModel()

#     v_uav = 5.0      # Giả định UAV bay với vận tốc 5 m/s
#     eb_value = 276.0 # Wh
#     energy_ratio = 0.2

#     # 1. Đọc dữ liệu từ file energy_4.0.npy
#     try:
#         energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
#         P_solar_mean = np.mean(energy_data['solar energy'])
#         P_fso_mean = np.mean(energy_data['fso energy'])
#         P_battery_fso_mean = P_fso_mean * energy_ratio
#     except Exception as e:
#         print(f"Lỗi khi đọc file energy_4.0.npy: {e}")
#         exit()

#     # 2. Gọi HÀM DUY NHẤT để tính toán tất cả
#     # Hàm giờ đây trả về 3 giá trị, ta gán vào 3 biến tương ứng
#     ls_seconds, p_c, p_h = uav_lifespan(uav_energy_model, v_uav, P_solar_mean, P_battery_fso_mean, eb_value)

#     # 3. In kết quả trực tiếp từ các biến đã được hàm trả về
#     print(f"\n{'='*60}")
#     print(">> THÔNG SỐ NĂNG LƯỢNG (POWER & ENERGY):")
#     print(f"   - Tổng tiêu thụ:                  {p_c:.4f} W")
#     print(f"   - Tổng thu hoạch:                 {p_h:.4f} W")
#     print(f"      + solar:                       {P_solar_mean:.4f} W")
#     print(f"      + fso:                         {P_battery_fso_mean:.4f} W")
    
#     if ls_seconds == float('inf'):
#         print("   => TUỔI THỌ UAV (Lifespan L_s):         Vô hạn (Thu hoạch >= Tiêu thụ)")
#     else:
#         ls_hours = ls_seconds / 3600
#         print(f"   => TUỔI THỌ UAV (Lifespan L_s):         {ls_hours:.2f} giờ ({ls_seconds:.2f} giây)")

    
