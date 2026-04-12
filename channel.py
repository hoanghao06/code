import numpy as np
import scipy.integrate as integrate
from scipy import special
from scipy.special import erf

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
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * erf(v)) / (2 * v * np.exp(-v**2))
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
    delta_h = np.abs(H_IRS - H_UAV)
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
    integral_value,_ = integrate.quad(integrand, H_IRS, H_UAV)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

# -------------------
# Gamma-Gamma turbulence
# -------------------

    alpha_f = 1.0 / (np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1)
    beta_f = 1.0 / (np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1)
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
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / (2 * v * np.exp(-v**2))

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
    delta_h = np.abs(H_CAR - H_UAV)
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
    integral_value,_ = integrate.quad(integrand, H_CAR, H_UAV)
    sigma_R2 = 2.25 * (k ** (7/6)) * (sec ** (11/6)) * integral_value

# -------------------
# Gamma-Gamma turbulence
# -------------------

    alpha_f = 1.0 / (np.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2**(12/5))**(7/6)) - 1)
    beta_f = 1.0 / (np.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2**(12/5))**(5/6)) - 1)
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
    omega_l2_eeq = omega_l**2 * (np.sqrt(np.pi) * special.erf(v)) / (2 * v * np.exp(-v**2))

# Mô phỏng dao động theo phân bố Rice

    x_rpe = np.random.normal(0, sigma_p)
    y_rpe = np.random.normal(0, sigma_p)
    r_pe2 = x_rpe**2 + y_rpe**2
    hs = A_0 * np.exp(-2 * (r_pe2 / omega_l2_eeq))

# -------------------
# Total channel UAV-Vehices
# -------------------

    h_total = hc*ha*hs

    return h_total, hc, ha, hs
