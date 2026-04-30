import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# MÔ HÌNH NĂNG LƯỢNG 
# =====================================================================
class UAVEnergyModelDynamicW:
    def __init__(self, W):
        self.W = W  # Tổng trọng lượng UAV (Newtons) - mảng (array) để tính công suất
        
        # Các thông số khí động học
        self.rho = 1.225        
        self.R = 0.5            
        self.A = 0.79           
        self.U_tip = 200.0      
        self.s = 0.05           
        self.d_0 = 0.3          
        self.v_0 = 7.2          
        self.delta = 0.012      
        self.k = 0.1            
        
        self.P_c = 50.0  # Công suất tiêu thụ cho mạch truyền thông (W)
        
        # Tính toán các thành phần công suất cơ bản phụ thuộc vào trọng lượng W
        self.P_0 = (self.delta / 8) * self.rho * self.s * self.A * (self.U_tip ** 3)
        self.P_i = (1 + self.k) * (self.W ** 1.5) / np.sqrt(2 * self.rho * self.A)

    def propulsion_power(self, V):
        V = np.maximum(V, 0.0)
        term1 = self.P_0 * (1 + (3 * V**2) / (self.U_tip**2))
        inner_sqrt = np.sqrt(1 + (V**4) / (4 * self.v_0**4))
        term2 = self.P_i * np.sqrt(inner_sqrt - (V**2) / (2 * self.v_0**2))
        term3 = 0.5 * self.d_0 * self.rho * self.s * self.A * (V**3)
        return term1 + term2 + term3

def get_dynamic_solar_power(SC_mass, z_array):
    """
    Quy đổi khối lượng (kg) sang diện tích (m^2).
    A_solar = SC_mass * 2.0 
    """
    A_solar = SC_mass * 2.0 
    
    eta_p = 0.4           
    G_r = 1361            
    A_0_solar = 0.8978    
    B_0_solar = 0.2804    
    delta = 8000          
    
    alpha_a_array = A_0_solar - B_0_solar * np.exp(-z_array / delta)
    mean_alpha_a = np.mean(alpha_a_array)
    
    return eta_p * A_solar * G_r * mean_alpha_a

# =====================================================================
# CHẠY THỬ NGHIỆM VÀ VẼ ĐỒ THỊ
# =====================================================================

U_mass = 10.0                 # Khối lượng UAV cố định (kg)
g = 9.81                      # Gia tốc trọng trường (m/s^2)
velocity = 20.0               # Vận tốc cố định = 20 m/s

# Trích xuất độ cao Z
try:
    uav_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\uav_3.npy', allow_pickle=True).item()
    positions = uav_data['position']   
    z_array = positions[:, 2] if (positions.ndim > 1 and positions.shape[1] >= 3) else positions
except:
    z_array = np.array([100.0])  

# Trích xuất năng lượng FSO
try:
    energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
    P_battery_fso = np.mean(energy_data['fso energy'])
except:
    P_battery_fso = 0.0

battery_configs = [
    {"E_b": 291, "color": "blue",    "marker": "o"}, 
    {"E_b": 276, "color": "teal",    "marker": "s"}, 
    {"E_b": 246, "color": "red",   "marker": "^"}, 
]

# SỬA Ở ĐÂY: Bắt đầu từ 0.0 thay vì 0.05
SC_mass_array = np.linspace(0.0, 0.5, 20)
M_array = U_mass + SC_mass_array
W_array = M_array * g
A_solar_array = SC_mass_array * 2.0 

# KHỞI TẠO KHUNG HÌNH (Mình tăng chiều ngang lên 11 để có chỗ kéo giãn)
plt.figure(figsize=(11, 10)) 

uav_model = UAVEnergyModelDynamicW(W=W_array)
P_propulsion = uav_model.propulsion_power(velocity)
P_c = P_propulsion + uav_model.P_c
P_solar = get_dynamic_solar_power(SC_mass_array, z_array)
P_h = P_solar + P_battery_fso 

handles_harvest = []
labels_harvest = []
handles_no = []
labels_no = []

for config in battery_configs:
    E_b_joules = config["E_b"] * 3600
    Ls_harvest = np.zeros_like(M_array)
    Ls_no_harvest = np.zeros_like(M_array)
    
    for i in range(len(M_array)):
        Ls_harvest[i] = E_b_joules / (P_c[i] - P_h[i]) if P_h[i] < P_c[i] else np.nan
        Ls_no_harvest[i] = E_b_joules / P_c[i]
            
    line_h, = plt.plot(A_solar_array, Ls_harvest, 
             color=config["color"], marker=config["marker"], linestyle='-', markersize=6)
    handles_harvest.append(line_h)
    labels_harvest.append(f"$E_b$= {config['E_b']} Wh (with HSF-EH scheme)")
    
    line_no, = plt.plot(A_solar_array, Ls_no_harvest, 
             color=config["color"], marker=config["marker"], linestyle='--', markersize=6, alpha=0.6)
    handles_no.append(line_no)
    labels_no.append(f"$E_b$= {config['E_b']} Wh (without EH)")

handles = handles_harvest + handles_no
labels = labels_harvest + labels_no

# --- ĐỊNH DẠNG TRỤC ---
plt.xlabel("Solar Panel Area, $A_{solar}$ ($m^2$)", fontsize=24) 
plt.ylabel("Lifespan, $L_s$ (s)", fontsize=24) 
plt.tick_params(axis='both', which='major', labelsize=20) 
plt.grid(True, linestyle='--', alpha=0.6)

# THÊM Ở ĐÂY: Ép giới hạn trục x bắt đầu từ 0
plt.xlim(left=0)

# Đưa legend vào trong hình
plt.legend(handles, labels, fontsize=14.5, loc='upper left', 
           frameon=True, edgecolor='black', ncol=1)

# =====================================================================
# LỆNH ÉP KHUNG ĐỒ THỊ VỚI TỈ LỆ MỚI (0.85 = Chiều cao bằng 85% chiều rộng)
# =====================================================================
plt.gca().set_box_aspect(0.85)

# --- CHỈNH LỀ THỦ CÔNG ĐỂ KHÔNG BỊ CẮT CHỮ TRỤC X ---
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

plt.show()
