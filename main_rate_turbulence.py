# import numpy as np
# import scipy.integrate as integrate
# from scipy import special
# from scipy.special import erf
# import pandas as pd
# from scipy.interpolate import LinearNDInterpolator
# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# import random

# # ================== ĐẶT SEED CỐ ĐỊNH ==================
# SEED = 10520264
# # SEED = 16
# np.random.seed(SEED)
# random.seed(SEED)

# # ================== THAM SỐ HỆ THỐNG ==================
# lamda = 1550e-9
# k = 2 * np.pi / lamda
# v_wind = 21
# Cn2_0 = 1e-14          # giá trị tạm, sẽ bị ghi đè trong vòng lặp
# a_irs = 1
# a_rx = 0.1
# ps_jitter = 2
# sigma_p = 0.1
# CLWC = 10 ** -2
# r_cloud = 3.33
# rho_water = 1.0
# N_c = (CLWC / ((4/3) * np.pi * (r_cloud**3) * rho_water * 1e-6))

# H_cloud_min = 500
# H_cloud_max = 1000

# c_speed = 3e8
# q_charge = 1.6e-19
# E_0 = 0.26961
# k_B = 1.38e-23
# R_L = 50
# T_thermal = 298
# FSO_bandwidth = 1
# Delta_f = FSO_bandwidth * 1e9 / 2
# Delta_lamda = (FSO_bandwidth * 1e9 * (lamda ** 2)) / c_speed
# number_car = 3
# energy_ratio = 0.2
# tx_ratio = 1 - energy_ratio

# V_t = 0.025
# I_0 = 1e-9
# eta_eo = 0.9
# R_xi = 0.6
# P_s = (10 ** (25 / 10)) / 1000
# B_bias = 0.04
# irs_gain_dB = 2
# irs_gain = 10 ** (irs_gain_dB / 10)

# # ================== HÀM NỘI SUY KHÍ QUYỂN ==================
# def transmittance(file_path, environment='Nông thôn (T)'):
#     df = pd.read_excel(file_path)
#     df = df[df['h_sensor (km)'].astype(str).str.isnumeric()].copy()
#     df['V (km)'] = df['V (km)'].ffill().astype(float)
#     df['h_sensor (km)'] = df['h_sensor (km)'].astype(float)
#     df[environment] = df[environment].astype(float)
#     points = df[['V (km)', 'h_sensor (km)']].values
#     values = df[environment].values
#     return LinearNDInterpolator(points, values)

# csv_file_path = r"C:\Users\AVSTC\Desktop\2026.Globecom\data_khihau\Tropical.csv.xlsx"
# env_type = 'Đô thị (T)'
# t_interpolator = transmittance(csv_file_path, env_type)

# # ================== CÁC HÀM KÊNH FSO ==================
# def get_fso(hap_pos: np.ndarray, irs_pos: np.ndarray):
#     H_HAP = hap_pos[-1]
#     H_IRS = irs_pos[-1]
#     delta_h = np.abs(H_IRS - H_HAP)
#     r_dis = np.linalg.norm(irs_pos[0:-1] - hap_pos[0:-1])
#     r_dis = np.maximum(r_dis, 1e-6)
#     phi_beam_z = np.arctan(r_dis / delta_h)
#     sec = 1 / np.cos(phi_beam_z)
#     L = delta_h * sec

#     Z_min = np.maximum(H_IRS, H_cloud_min)
#     Z_max = np.minimum(H_HAP, H_cloud_max)
#     delta_hc = np.maximum(Z_max - Z_min, 0)
#     L_c = delta_hc * sec
#     fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
#     q = 1.6
#     cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
#     cloud_coefficient = cloud_coefficient_dB / (1e4 * np.log10(np.exp(1)))
#     clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
#     clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
#     L_clear = delta_h - L_c
#     hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

#     def Cn2(h):
#         term1 = 0.00594*(v_wind/27)**2*(1e-5*h)**10*np.exp(-h/1000)
#         term2 = 2.7e-16*np.exp(-h/1500)
#         term3 = Cn2_0*np.exp(-h/100)
#         return term1 + term2 + term3

#     def integrand(h):
#         return Cn2(h) * np.abs(h - H_IRS) ** (5/6)
#     integral_value, _ = integrate.quad(integrand, H_IRS, H_HAP)
#     sigma_R2 = 2.25 * (k ** (7/6)) * (sec) ** (11/6) * integral_value

#     def lognormal_turbulence():
#         mu = -sigma_R2/2
#         sigma = np.sqrt(sigma_R2)
#         ha = np.random.lognormal(mu, sigma)
#         return ha
#     ha = lognormal_turbulence()

#     theta_HI = 1e-3
#     rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5)
#     omega_0 = 2*lamda / (np.pi*theta_HI)
#     omega_l = omega_0 * np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) * ((lamda * L) / (np.pi * omega_0 ** 2)) ** 2)
#     v = (a_irs * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
#     A_0 = special.erf(v)**2
#     den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
#     omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
#     rho_l = 0
#     hs = A_0 * np.exp(-2*(rho_l/omega_l2_eeq)**2)
#     h_total = hc * ha * hs
#     return h_total, hc, ha, hs

# def get_fso_backhaul(uav_pos: np.ndarray, irs_pos: np.ndarray):
#     H_UAV = uav_pos[-1]
#     H_IRS = irs_pos[-1]
#     delta_h = np.maximum(np.abs(H_UAV - H_IRS), 1e-6)
#     r_dis = np.linalg.norm(irs_pos[0:-1] - uav_pos[0:-1])
#     r_dis = np.maximum(r_dis, 1e-6)
#     phi_beam_z = np.arctan(r_dis / delta_h)
#     sec = 1 / np.cos(phi_beam_z)
#     L = delta_h * sec

#     Z_min = np.maximum(H_IRS, H_cloud_min)
#     Z_max = np.minimum(H_UAV, H_cloud_max)
#     delta_hc = np.maximum(Z_max - Z_min, 0)
#     L_c = delta_hc * sec
#     fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
#     if fso_visibility > 6:
#         q = 1.6
#     elif 1 < fso_visibility <= 6:
#         q = 0.16 * fso_visibility + 0.34
#     else:
#         q = fso_visibility - 0.5
#         if q < 0:
#             q = 0
#     cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
#     cloud_coefficient = cloud_coefficient_dB / (1e4 * np.log10(np.exp(1)))
#     clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
#     clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
#     L_clear = delta_h - L_c
#     hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

#     def integrand(h):
#         return (0.00594 * (v_wind / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
#                 2.7 * 10 ** -16 * np.exp(-h / 1500) + Cn2_0 * np.exp(-h / 100)) * np.abs(h - H_UAV) ** (5/6)
#     z_lower = np.minimum(H_IRS, H_UAV)
#     z_upper = np.maximum(H_IRS, H_UAV)
#     integral_value, _ = integrate.quad(integrand, z_lower, z_upper)
#     sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

#     denom_alpha = np.maximum(np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1, 1e-6)
#     alpha_f = 1.0 / denom_alpha
#     denom_beta = np.maximum(np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1, 1e-6)
#     beta_f = 1.0 / denom_beta
#     X = np.random.gamma(shape=alpha_f, scale=1.0/alpha_f)
#     Y = np.random.gamma(shape=beta_f, scale=1.0/beta_f)
#     ha = X * Y

#     omega_0 = 0.01
#     rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5)
#     omega_l = omega_0 * np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) * ((lamda * L) / (np.pi * omega_0 ** 2)) ** 2)
#     v = (a_rx * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
#     A_0 = special.erf(v)**2
#     den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
#     omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
#     x_rpe = np.random.normal(0, sigma_p)
#     y_rpe = np.random.normal(0, sigma_p)
#     r_pe2 = x_rpe**2 + y_rpe**2
#     hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))
#     h_total = hc * ha * hs
#     return h_total, hc, ha, hs

# def get_fso_access(uav_pos: np.ndarray, car_pos: np.ndarray):
#     H_UAV = uav_pos[-1]
#     H_CAR = car_pos[-1]
#     delta_h = np.maximum(np.abs(H_CAR - H_UAV), 1e-6)
#     r_dis = np.linalg.norm(uav_pos[0:-1] - car_pos[0:-1])
#     r_dis = np.maximum(r_dis, 1e-6)
#     phi_beam_z = np.arctan(r_dis / delta_h)
#     sec = 1 / np.cos(phi_beam_z)
#     L = delta_h * sec

#     Z_min = np.maximum(H_CAR, H_cloud_min)
#     Z_max = np.minimum(H_UAV, H_cloud_max)
#     delta_hc = np.maximum(Z_max - Z_min, 0)
#     L_c = delta_hc * sec
#     fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
#     if fso_visibility > 6:
#         q = 1.6
#     elif 1 < fso_visibility <= 6:
#         q = 0.16 * fso_visibility + 0.34
#     else:
#         q = fso_visibility - 0.5
#         if q < 0:
#             q = 0
#     cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
#     cloud_coefficient = cloud_coefficient_dB / (1e4 * np.log10(np.exp(1)))
#     clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
#     clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
#     L_clear = delta_h - L_c
#     hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

#     def integrand(h):
#         return (0.00594 * (v_wind / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
#                 2.7 * 10 ** -16 * np.exp(-h / 1500) + Cn2_0 * np.exp(-h / 100)) * np.abs(h - H_UAV) ** (5/6)
#     z_lower = np.minimum(H_CAR, H_UAV)
#     z_upper = np.maximum(H_CAR, H_UAV)
#     integral_value, _ = integrate.quad(integrand, z_lower, z_upper)
#     sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

#     denom_alpha = np.maximum(np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1, 1e-6)
#     alpha_f = 1.0 / denom_alpha
#     denom_beta = np.maximum(np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1, 1e-6)
#     beta_f = 1.0 / denom_beta
#     X = np.random.gamma(shape=alpha_f, scale=1.0/alpha_f)
#     Y = np.random.gamma(shape=beta_f, scale=1.0/beta_f)
#     ha = X * Y

#     omega_0 = 0.01
#     rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5)
#     omega_l = omega_0 * np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) * ((lamda * L) / (np.pi * omega_0 ** 2)) ** 2)
#     v = (a_rx * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
#     A_0 = special.erf(v)**2
#     den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
#     omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
#     x_rpe = np.random.normal(0, sigma_p)
#     y_rpe = np.random.normal(0, sigma_p)
#     r_pe2 = x_rpe**2 + y_rpe**2
#     hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))
#     h_total = hc * ha * hs
#     return h_total, hc, ha, hs

# def get_fso_harvested_power(h_total, gain_factor=1):
#     core_term = eta_eo * h_total * R_xi * a_rx * np.sqrt(P_s) * B_bias
#     p_fso = ((0.75 * V_t * (core_term ** 2)) / I_0) * gain_factor
#     return p_fso

# def get_snr(h_total, P_transmit_total, uav_pos):
#     H_UAV = uav_pos[-1]
#     fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
#     sigma2_FSO = 1e-17
#     R_acc = 1
#     if fso_visibility < 18.0:
#         V_snapped = 5.0
#     else:
#         V_snapped = 30.0
#     T_trans_raw = t_interpolator(V_snapped, H_UAV / 1000)
#     if np.isnan(T_trans_raw):
#         T_trans = 0.5
#     else:
#         T_trans = float(T_trans_raw)
#     P_b = (E_0 * 1e9) * T_trans * np.pi * (a_rx ** 2) * Delta_lamda
#     shot_noise_term = 2 * q_charge * R_xi * P_b
#     thermal_noise_term = (4 * k_B * T_thermal / R_L) * Delta_f
#     sigma_N_square = shot_noise_term + thermal_noise_term
#     gamma_F = (P_transmit_total * R_acc * h_total) ** 2 / ((sigma2_FSO + sigma_N_square) * FSO_bandwidth)
#     return gamma_F

# def data_rate(gamma_F, bandwidth_ghz):
#     B_hz = bandwidth_ghz * 1e9
#     data_rate_bps = (B_hz / 2) * np.log2(1 + gamma_F)
#     return data_rate_bps

# # ================== ĐỌC DỮ LIỆU QUỸ ĐẠO (MỘT LẦN) ==================
# data_uav = np.load(r'C:\Users\AVSTC\Desktop\2026.Globecom\output_rural_10\speed_10\0\flydata\uav_3.npy', allow_pickle=True).item()
# uav_traj_full = data_uav['position']

# car_traj_full = []
# for i in range(1, number_car + 1):
#     file_path = rf'C:\Users\AVSTC\Desktop\2026.Globecom\data_xechay\rural_1\trans_data_Train1th_15ms_User{i}th.csv'
#     df_car = pd.read_csv(file_path, skiprows=1, header=None)
#     car_traj_full.append(df_car.iloc[:, :3].values.astype(float))

# min_T = min(uav_traj_full.shape[0], min(t.shape[0] for t in car_traj_full))
# print(f"Đồng bộ về {min_T} bước thời gian")
# uav_traj = uav_traj_full[:min_T, :]
# car_traj = [t[:min_T, :] for t in car_traj_full]
# T = min_T


# hap_pos = np.array([0, 0, 20000])
# irs_pos = np.array([0, 0, 80])
# uav_fixed_pos = np.array([0, 0, 1200])


# # ================== THAM SỐ QUÉT ==================
# Cn2_values = np.linspace(1e-15, 1e-13, 40)
# target_rates_mbps = np.arange(3, 8, 0.1)

# # Hàm tính tỷ lệ phần trăm cho một chế độ
# def compute_percentages(mode='dynamic'):
#     # Reset seed để mỗi chế độ có cùng điều kiện khởi tạo
#     np.random.seed(SEED)
#     random.seed(SEED)
    
#     percentage_results = np.zeros((len(Cn2_values), len(target_rates_mbps)))
    
#     for idx_cn, cn_val in enumerate(Cn2_values):
#         global Cn2_0
#         Cn2_0 = cn_val
#         # Reset seed tại mỗi giá trị Cn2 (giữ nguyên logic cũ)
#         np.random.seed(SEED)
#         random.seed(SEED)
#         print(f"[{mode}] Đang chạy với Cn2_0 = {cn_val:.2e}")
        
#         h_hap_irs, _, _, _ = get_fso(hap_pos, irs_pos)
#         rates = np.zeros((T, number_car))
        
#         for t in range(T):
#             if mode == 'dynamic':
#                 uav_pos = uav_traj[t]
#             else:   # fixed
#                 uav_pos = uav_fixed_pos
            
#             h_irs_uav, _, _, _ = get_fso_backhaul(uav_pos, irs_pos)
#             h_eq = h_hap_irs * h_irs_uav * np.sqrt(irs_gain)
#             P_harvest = get_fso_harvested_power(h_eq, gain_factor=1)
#             P_tx_uav = tx_ratio * P_harvest
            
#             for i in range(number_car):
#                 car_pos = car_traj[i][t].copy()
#                 car_pos[-1] = 0.0
#                 h_uav_car, _, _, _ = get_fso_access(uav_pos, car_pos)
#                 gamma = get_snr(h_uav_car, P_tx_uav, uav_pos)
#                 rate = data_rate(gamma, FSO_bandwidth)
#                 rates[t, i] = rate
        
#         for idx_tar, tar_mbps in enumerate(target_rates_mbps):
#             tar_bps = tar_mbps * 1e9
#             percentages = []
#             for i in range(number_car):
#                 count = np.sum(rates[:, i] > tar_bps)
#                 percentages.append(count / T * 100)
#             percentage_results[idx_cn, idx_tar] = np.mean(percentages)
    
#     return percentage_results

# # Tính cho hai chế độ
# print("=== UAV DI CHUYỂN (dynamic) ===")
# percentage_dynamic = compute_percentages(mode='dynamic')

# print("\n=== UAV CỐ ĐỊNH (fixed) ===")
# percentage_fixed = compute_percentages(mode='fixed')

# # ================== VẼ HAI HÌNH CÙNG MỘT ẢNH (HÀNG NGANG) ==================
# from matplotlib.ticker import ScalarFormatter

# # --- CỠ CHỮ CÓ THỂ TÙY CHỈNH ---
# FONTSIZE_LABEL = 14      # chữ trên nhãn trục (Target rate, Cn^2,...)
# FONTSIZE_TITLE = 14      # tiêu đề mỗi subplot
# FONTSIZE_CBAR_LABEL = 14 # chữ "Percentage (%)" bên cạnh thanh màu
# FONTSIZE_TICK = 12       # chữ trên các vạch chia (tick) của trục và colorbar

# X, Y = np.meshgrid(target_rates_mbps, Cn2_values)

# # Tạo figure với 1 hàng, 2 cột
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), subplot_kw={'projection': '3d'})

# # ---- Hình thứ nhất: UAV di chuyển ----
# surf1 = ax1.plot_surface(X, Y, percentage_dynamic, cmap='viridis', 
#                          edgecolor='black', linewidth=0.3, alpha=0.8)
# cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20)
# cbar1.set_label('Percentage (%)', fontsize=FONTSIZE_CBAR_LABEL)
# cbar1.ax.tick_params(labelsize=FONTSIZE_TICK)           # cỡ chữ ticks trên colorbar

# ax1.set_xlabel('Target rate (Gbps)', fontsize=FONTSIZE_LABEL)
# ax1.set_ylabel(r'$C_n^2(0)$', fontsize=FONTSIZE_LABEL)
# ax1.set_zlabel('Percentage of target achieved (%)', fontsize=FONTSIZE_LABEL)
# # ax1.set_title('UAV di chuyển theo quỹ đạo', fontsize=FONTSIZE_TITLE)

# # Chỉnh cỡ chữ ticks cho các trục
# ax1.xaxis.set_tick_params(labelsize=FONTSIZE_TICK)
# ax1.yaxis.set_tick_params(labelsize=FONTSIZE_TICK)
# ax1.zaxis.set_tick_params(labelsize=FONTSIZE_TICK)

# # Định dạng trục y (hiển thị số mũ)
# ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax1.yaxis.get_major_formatter().set_powerlimits((0, 0))

# ax1.view_init(elev=25, azim=-60)
# ax1.grid(True, linewidth=0.5, linestyle='--', color='gray')

# # ---- Hình thứ hai: UAV cố định ----
# surf2 = ax2.plot_surface(X, Y, percentage_fixed, cmap='viridis', 
#                          edgecolor='black', linewidth=0.3, alpha=0.8)
# cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20)
# cbar2.set_label('Percentage (%)', fontsize=FONTSIZE_CBAR_LABEL)
# cbar2.ax.tick_params(labelsize=FONTSIZE_TICK)           # cỡ chữ ticks trên colorbar

# ax2.set_xlabel('Target rate (Gbps)', fontsize=FONTSIZE_LABEL)
# ax2.set_ylabel(r'$C_n^2(0)$', fontsize=FONTSIZE_LABEL)
# ax2.set_zlabel('Percentage of target achieved (%)', fontsize=FONTSIZE_LABEL)
# # ax2.set_title('UAV cố định tại (0,0,1200m)', fontsize=FONTSIZE_TITLE)

# ax2.xaxis.set_tick_params(labelsize=FONTSIZE_TICK)
# ax2.yaxis.set_tick_params(labelsize=FONTSIZE_TICK)
# ax2.zaxis.set_tick_params(labelsize=FONTSIZE_TICK)

# ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))

# ax2.view_init(elev=25, azim=-60)
# ax2.grid(True, linewidth=0.5, linestyle='--', color='gray')

# plt.tight_layout()
# plt.show()







import numpy as np
import scipy.integrate as integrate
from scipy import special
from scipy.special import erf
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import random

# ================== ĐẶT SEED CỐ ĐỊNH ==================
SEED = 10520264
# SEED = 16
np.random.seed(SEED)
random.seed(SEED)

# ================== THAM SỐ HỆ THỐNG ==================
lamda = 1550e-9
k = 2 * np.pi / lamda
v_wind = 21
Cn2_0 = 1e-14          # giá trị tạm, sẽ bị ghi đè trong vòng lặp
a_irs = 1
a_rx = 0.1
ps_jitter = 2
sigma_p = 0.1
CLWC = 10 ** -2
r_cloud = 3.33
rho_water = 1.0
N_c = (CLWC / ((4/3) * np.pi * (r_cloud**3) * rho_water * 1e-6))

H_cloud_min = 500
H_cloud_max = 1000

c_speed = 3e8
q_charge = 1.6e-19
E_0 = 0.26961
k_B = 1.38e-23
R_L = 50
T_thermal = 298
FSO_bandwidth = 1
Delta_f = FSO_bandwidth * 1e9 / 2
Delta_lamda = (FSO_bandwidth * 1e9 * (lamda ** 2)) / c_speed
number_car = 3
energy_ratio = 0.2
tx_ratio = 1 - energy_ratio

V_t = 0.025
I_0 = 1e-9
eta_eo = 0.9
R_xi = 0.6
P_s = (10 ** (25 / 10)) / 1000
B_bias = 0.04
irs_gain_dB = 2
irs_gain = 10 ** (irs_gain_dB / 10)

# ================== HÀM NỘI SUY KHÍ QUYỂN ==================
def transmittance(file_path, environment='Nông thôn (T)'):
    df = pd.read_excel(file_path)
    df = df[df['h_sensor (km)'].astype(str).str.isnumeric()].copy()
    df['V (km)'] = df['V (km)'].ffill().astype(float)
    df['h_sensor (km)'] = df['h_sensor (km)'].astype(float)
    df[environment] = df[environment].astype(float)
    points = df[['V (km)', 'h_sensor (km)']].values
    values = df[environment].values
    return LinearNDInterpolator(points, values)

csv_file_path = r"C:\Users\AVSTC\Desktop\2026.Globecom\data_khihau\Tropical.csv.xlsx"
env_type = 'Đô thị (T)'
t_interpolator = transmittance(csv_file_path, env_type)

# ================== CÁC HÀM KÊNH FSO ==================
def get_fso(hap_pos: np.ndarray, irs_pos: np.ndarray):
    H_HAP = hap_pos[-1]
    H_IRS = irs_pos[-1]
    delta_h = np.abs(H_IRS - H_HAP)
    r_dis = np.linalg.norm(irs_pos[0:-1] - hap_pos[0:-1])
    r_dis = np.maximum(r_dis, 1e-6)
    phi_beam_z = np.arctan(r_dis / delta_h)
    sec = 1 / np.cos(phi_beam_z)
    L = delta_h * sec

    Z_min = np.maximum(H_IRS, H_cloud_min)
    Z_max = np.minimum(H_HAP, H_cloud_max)
    delta_hc = np.maximum(Z_max - Z_min, 0)
    L_c = delta_hc * sec
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    q = 1.6
    cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
    cloud_coefficient = cloud_coefficient_dB / (1e4 * np.log10(np.exp(1)))
    clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
    clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
    L_clear = delta_h - L_c
    hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

    def Cn2(h):
        term1 = 0.00594*(v_wind/27)**2*(1e-5*h)**10*np.exp(-h/1000)
        term2 = 2.7e-16*np.exp(-h/1500)
        term3 = Cn2_0*np.exp(-h/100)
        return term1 + term2 + term3

    def integrand(h):
        return Cn2(h) * np.abs(h - H_IRS) ** (5/6)
    integral_value, _ = integrate.quad(integrand, H_IRS, H_HAP)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec) ** (11/6) * integral_value

    def lognormal_turbulence():
        mu = -sigma_R2/2
        sigma = np.sqrt(sigma_R2)
        ha = np.random.lognormal(mu, sigma)
        return ha
    ha = lognormal_turbulence()

    theta_HI = 1e-3
    rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5)
    omega_0 = 2*lamda / (np.pi*theta_HI)
    omega_l = omega_0 * np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) * ((lamda * L) / (np.pi * omega_0 ** 2)) ** 2)
    v = (a_irs * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
    A_0 = special.erf(v)**2
    den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
    rho_l = 0
    hs = A_0 * np.exp(-2*(rho_l/omega_l2_eeq)**2)
    h_total = hc * ha * hs
    return h_total, hc, ha, hs

def get_fso_backhaul(uav_pos: np.ndarray, irs_pos: np.ndarray):
    H_UAV = uav_pos[-1]
    H_IRS = irs_pos[-1]
    delta_h = np.maximum(np.abs(H_UAV - H_IRS), 1e-6)
    r_dis = np.linalg.norm(irs_pos[0:-1] - uav_pos[0:-1])
    r_dis = np.maximum(r_dis, 1e-6)
    phi_beam_z = np.arctan(r_dis / delta_h)
    sec = 1 / np.cos(phi_beam_z)
    L = delta_h * sec

    Z_min = np.maximum(H_IRS, H_cloud_min)
    Z_max = np.minimum(H_UAV, H_cloud_max)
    delta_hc = np.maximum(Z_max - Z_min, 0)
    L_c = delta_hc * sec
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    if fso_visibility > 6:
        q = 1.6
    elif 1 < fso_visibility <= 6:
        q = 0.16 * fso_visibility + 0.34
    else:
        q = fso_visibility - 0.5
        if q < 0:
            q = 0
    cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
    cloud_coefficient = cloud_coefficient_dB / (1e4 * np.log10(np.exp(1)))
    clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
    clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
    L_clear = delta_h - L_c
    hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

    def integrand(h):
        return (0.00594 * (v_wind / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
                2.7 * 10 ** -16 * np.exp(-h / 1500) + Cn2_0 * np.exp(-h / 100)) * np.abs(h - H_UAV) ** (5/6)
    z_lower = np.minimum(H_IRS, H_UAV)
    z_upper = np.maximum(H_IRS, H_UAV)
    integral_value, _ = integrate.quad(integrand, z_lower, z_upper)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

    denom_alpha = np.maximum(np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1, 1e-6)
    alpha_f = 1.0 / denom_alpha
    denom_beta = np.maximum(np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1, 1e-6)
    beta_f = 1.0 / denom_beta
    X = np.random.gamma(shape=alpha_f, scale=1.0/alpha_f)
    Y = np.random.gamma(shape=beta_f, scale=1.0/beta_f)
    ha = X * Y

    omega_0 = 0.01
    rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5)
    omega_l = omega_0 * np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) * ((lamda * L) / (np.pi * omega_0 ** 2)) ** 2)
    v = (a_rx * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
    A_0 = special.erf(v)**2
    den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
    x_rpe = np.random.normal(0, sigma_p)
    y_rpe = np.random.normal(0, sigma_p)
    r_pe2 = x_rpe**2 + y_rpe**2
    hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))
    h_total = hc * ha * hs
    return h_total, hc, ha, hs

def get_fso_access(uav_pos: np.ndarray, car_pos: np.ndarray):
    H_UAV = uav_pos[-1]
    H_CAR = car_pos[-1]
    delta_h = np.maximum(np.abs(H_CAR - H_UAV), 1e-6)
    r_dis = np.linalg.norm(uav_pos[0:-1] - car_pos[0:-1])
    r_dis = np.maximum(r_dis, 1e-6)
    phi_beam_z = np.arctan(r_dis / delta_h)
    sec = 1 / np.cos(phi_beam_z)
    L = delta_h * sec

    Z_min = np.maximum(H_CAR, H_cloud_min)
    Z_max = np.minimum(H_UAV, H_cloud_max)
    delta_hc = np.maximum(Z_max - Z_min, 0)
    L_c = delta_hc * sec
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    if fso_visibility > 6:
        q = 1.6
    elif 1 < fso_visibility <= 6:
        q = 0.16 * fso_visibility + 0.34
    else:
        q = fso_visibility - 0.5
        if q < 0:
            q = 0
    cloud_coefficient_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-q)
    cloud_coefficient = cloud_coefficient_dB / (1e4 * np.log10(np.exp(1)))
    clear_coe_dB = (3.91 / fso_visibility) * (lamda * 1e9 / 550) ** (-1.6)
    clear_coe = clear_coe_dB / (1e4 * np.log10(np.exp(1)))
    L_clear = delta_h - L_c
    hc = np.exp(-cloud_coefficient * L_c - clear_coe * L_clear)

    def integrand(h):
        return (0.00594 * (v_wind / 27) ** 2 * (h * 10 ** -5) ** 10 * np.exp(-h / 1000) + \
                2.7 * 10 ** -16 * np.exp(-h / 1500) + Cn2_0 * np.exp(-h / 100)) * np.abs(h - H_UAV) ** (5/6)
    z_lower = np.minimum(H_CAR, H_UAV)
    z_upper = np.maximum(H_CAR, H_UAV)
    integral_value, _ = integrate.quad(integrand, z_lower, z_upper)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

    denom_alpha = np.maximum(np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1, 1e-6)
    alpha_f = 1.0 / denom_alpha
    denom_beta = np.maximum(np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1, 1e-6)
    beta_f = 1.0 / denom_beta
    X = np.random.gamma(shape=alpha_f, scale=1.0/alpha_f)
    Y = np.random.gamma(shape=beta_f, scale=1.0/beta_f)
    ha = X * Y

    omega_0 = 0.01
    rho_0 = (0.55 * Cn2_0 * L * k ** 2) ** (-3 / 5)
    omega_l = omega_0 * np.sqrt(1 + (1 + (2 * omega_0 ** 2 / (L * rho_0 ** 2))) * ((lamda * L) / (np.pi * omega_0 ** 2)) ** 2)
    v = (a_rx * np.sqrt(np.pi)) / (omega_l * np.sqrt(2))
    A_0 = special.erf(v)**2
    den = np.maximum(2 * v * np.exp(-v**2), 1e-12)
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / den
    x_rpe = np.random.normal(0, sigma_p)
    y_rpe = np.random.normal(0, sigma_p)
    r_pe2 = x_rpe**2 + y_rpe**2
    hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))
    h_total = hc * ha * hs
    return h_total, hc, ha, hs

def get_fso_harvested_power(h_total, gain_factor=1):
    core_term = eta_eo * h_total * R_xi * a_rx * np.sqrt(P_s) * B_bias
    p_fso = ((0.75 * V_t * (core_term ** 2)) / I_0) * gain_factor
    return p_fso

def get_snr(h_total, P_transmit_total, uav_pos):
    H_UAV = uav_pos[-1]
    fso_visibility = (1002 / (CLWC * N_c) ** 0.6473) / 1000
    sigma2_FSO = 1e-17
    R_acc = 1
    if fso_visibility < 18.0:
        V_snapped = 5.0
    else:
        V_snapped = 30.0
    T_trans_raw = t_interpolator(V_snapped, H_UAV / 1000)
    if np.isnan(T_trans_raw):
        T_trans = 0.5
    else:
        T_trans = float(T_trans_raw)
    P_b = (E_0 * 1e9) * T_trans * np.pi * (a_rx ** 2) * Delta_lamda
    shot_noise_term = 2 * q_charge * R_xi * P_b
    thermal_noise_term = (4 * k_B * T_thermal / R_L) * Delta_f
    sigma_N_square = shot_noise_term + thermal_noise_term
    gamma_F = (P_transmit_total * R_acc * h_total) ** 2 / ((sigma2_FSO + sigma_N_square) * FSO_bandwidth)
    return gamma_F

def data_rate(gamma_F, bandwidth_ghz):
    B_hz = bandwidth_ghz * 1e9
    data_rate_bps = (B_hz / 2) * np.log2(1 + gamma_F)
    return data_rate_bps

# ================== ĐỌC DỮ LIỆU QUỸ ĐẠO (MỘT LẦN) ==================
data_uav = np.load(r'C:\Users\AVSTC\Desktop\2026.Globecom\output_rural_10\speed_10\0\flydata\uav_3.npy', allow_pickle=True).item()
uav_traj_full = data_uav['position']

car_traj_full = []
for i in range(1, number_car + 1):
    file_path = rf'C:\Users\AVSTC\Desktop\2026.Globecom\data_xechay\rural_1\trans_data_Train1th_15ms_User{i}th.csv'
    df_car = pd.read_csv(file_path, skiprows=1, header=None)
    car_traj_full.append(df_car.iloc[:, :3].values.astype(float))

min_T = min(uav_traj_full.shape[0], min(t.shape[0] for t in car_traj_full))
print(f"Đồng bộ về {min_T} bước thời gian")
uav_traj = uav_traj_full[:min_T, :]
car_traj = [t[:min_T, :] for t in car_traj_full]
T = min_T


hap_pos = np.array([0, 0, 20000])
irs_pos = np.array([0, 0, 80])
uav_fixed_pos = np.array([0, 0, 1200])


# ================== THAM SỐ QUÉT ==================
Cn2_values = np.linspace(1e-15, 1e-13, 40)      # [1e-15, 2.5e-14?] Thực tế 5 điểm cách đều
target_rates_mbps = np.arange(3, 8, 0.1)         # 3,4,5,6,7

percentage_results = np.zeros((len(Cn2_values), len(target_rates_mbps)))

# ================== VÒNG LẶP QUA Cn2_0 ==================
for idx_cn, cn_val in enumerate(Cn2_values):
    # global Cn2_0
    Cn2_0 = cn_val
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Đang chạy với Cn2_0 = {cn_val:.2e}")
    
    h_hap_irs, _, _, _ = get_fso(hap_pos, irs_pos)
    rates = np.zeros((T, number_car))
    
    for t in range(T):
        uav_pos = uav_traj[t]
        uav_pos = uav_fixed_pos
        h_irs_uav, _, _, _ = get_fso_backhaul(uav_pos, irs_pos)
        h_eq = h_hap_irs * h_irs_uav * np.sqrt(irs_gain)
        P_harvest = get_fso_harvested_power(h_eq, gain_factor=1)
        P_tx_uav = tx_ratio * P_harvest
        
        for i in range(number_car):
            car_pos = car_traj[i][t].copy()
            car_pos[-1] = 0.0
            h_uav_car, _, _, _ = get_fso_access(uav_pos, car_pos)
            gamma = get_snr(h_uav_car, P_tx_uav, uav_pos)
            rate = data_rate(gamma, FSO_bandwidth)
            rates[t, i] = rate
    
    for idx_tar, tar_mbps in enumerate(target_rates_mbps):
        tar_bps = tar_mbps * 1e9
        percentages = []
        for i in range(number_car):
            count = np.sum(rates[:, i] > tar_bps)
            percentages.append(count / T * 100)
        percentage_results[idx_cn, idx_tar] = np.mean(percentages)

# ================== VẼ BIỂU ĐỒ 3D ==================
# Tạo lưới dữ liệu cho surface plot
X, Y = np.meshgrid(target_rates_mbps, Cn2_values)
Z = percentage_results  # shape (len(Cn2), len(target_rates))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Vẽ surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', linewidth=0.5, alpha=0.8)
# cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20)
# cbar.set_label('Percentage (%)', fontsize=14)  # cỡ chữ mong muốn


# Thêm thanh màu
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Percentage (%)')
FONT_LABEL = 14
FONT_TICKS = 12
FONT_LEGEND = 14

# Đặt nhãn trục
plt.xlabel("Time slot", fontsize=FONT_LABEL)
ax.set_xlabel('Target rate (Gbps)', fontsize=FONT_LABEL)
ax.set_ylabel(r'$C_n^2(0)$', fontsize=FONT_LABEL )   # Ký hiệu Cn^2
ax.set_zlabel('Percentage of target achieved (%)',fontsize=FONT_LABEL)

# Định dạng tick cho trục y hiển thị dạng 10^{-15}
from matplotlib.ticker import ScalarFormatter
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

# Tùy chỉnh góc nhìn
ax.view_init(elev=25, azim=-60)

# TICKS
plt.xticks(fontsize=FONT_TICKS)
plt.yticks(fontsize=FONT_TICKS)

# LEGEND
plt.legend(fontsize=FONT_LEGEND)

plt.tight_layout()
plt.show() 