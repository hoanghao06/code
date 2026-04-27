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
    Nhận vào TOÀN BỘ mảng độ cao z_array từ quỹ đạo.
    Tính toán năng lượng ở từng độ cao, sau đó lấy trung bình.
    """
    A_solar = SC_mass * 2.0 
    
    eta_p = 0.4           
    G_r = 1361            
    A_0_solar = 0.8978    
    B_0_solar = 0.2804    
    delta = 8000          
    
    # 1. Tính mảng hệ số alpha_a cho TỪNG ĐIỂM độ cao trên quỹ đạo
    # z_array là mảng N phần tử -> alpha_a_array cũng là mảng N phần tử
    alpha_a_array = A_0_solar - B_0_solar * np.exp(-z_array / delta)
    
    # 2. Lấy trung bình hệ số thu hoạch trên toàn bộ thời gian bay
    mean_alpha_a = np.mean(alpha_a_array)
    
    # 3. Tính công suất thu hoạch trung bình cuối cùng (W)
    return eta_p * A_solar * G_r * mean_alpha_a

# =====================================================================
# CHẠY THỬ NGHIỆM VÀ VẼ ĐỒ THỊ
# =====================================================================

# 1. Các thông số đầu vào cố định
U_mass = 10.0                 # Khối lượng UAV cố định (kg)
g = 9.81                      # Gia tốc trọng trường (m/s^2)
velocity = 20.0               # Vận tốc cố định = 20 m/s

print(f"[+] Đã cố định vận tốc UAV: {velocity} m/s")

# ---------------------------------------------------------
# TRÍCH XUẤT TOÀN BỘ MẢNG ĐỘ CAO Z TỪ FILE UAV_3.NPY
# ---------------------------------------------------------
try:
    uav_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\uav_3.npy', allow_pickle=True).item()
    positions = uav_data['position']   
    
    # Lấy toàn bộ mảng tọa độ z (không dùng np.mean ở đây nữa)
    if positions.ndim > 1 and positions.shape[1] >= 3:
        z_array = positions[:, 2]
    else:
        z_array = positions
        
    print(f"[+] Trích xuất quỹ đạo tối ưu thành công.")
    print(f"[+] Số lượng điểm độ cao (z) thu thập được để tính toán: {len(z_array)} điểm")

except Exception as e:
    print(f"[-] Lỗi khi đọc file uav_3.npy: {e}. Đang dùng giá trị fallback.")
    z_array = np.array([100.0])  

# ---------------------------------------------------------
# TRÍCH XUẤT NĂNG LƯỢNG FSO THỰC TỪ FILE ENERGY_3.NPY
# ---------------------------------------------------------
try:
    energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
    
    fso_energy_array = energy_data['fso energy']
    
    P_battery_fso = np.mean(fso_energy_array)
    
    print(f"[+] Trích xuất dữ liệu năng lượng thành công.")
    print(f"[+] Công suất FSO thu hoạch trung bình: {P_battery_fso:.4e} Watt")
except Exception as e:
    print(f"[-] Lỗi khi đọc file energy_3.npy: {e}")
    P_battery_fso = 0.0

# ---------------------------------------------------------
# TÍNH TOÁN TUỔI THỌ VÀ VẼ ĐỒ THỊ
# ---------------------------------------------------------

battery_configs = [
    {"E_b": 291.72, "color": "blue",    "marker": "o"}, 
    {"E_b": 276.00, "color": "teal",    "marker": "s"}, 
    {"E_b": 246.84, "color": "black",   "marker": "^"}, 
]

SC_mass_array = np.linspace(0.05, 0.5, 20)
M_array = U_mass + SC_mass_array
W_array = M_array * g

plt.figure(figsize=(12, 8)) # Tăng nhẹ size để chứa chú thích dài hơn

# Khởi tạo mô hình Năng lượng 1 lần
uav_model = UAVEnergyModelDynamicW(W=W_array)

# Tính toán tiêu thụ
P_propulsion = uav_model.propulsion_power(velocity)
P_c = P_propulsion + uav_model.P_c

# Truyền nguyên chuỗi z_array vào hàm tính năng lượng mặt trời
P_solar = get_dynamic_solar_power(SC_mass_array, z_array)
P_h = P_solar + P_battery_fso 

# Khởi tạo các danh sách để hứng các đường (handles) và nhãn (labels)
handles_harvest = []
labels_harvest = []
handles_no = []
labels_no = []

# Vẽ đồ thị 
for config in battery_configs:
    E_b_joules = config["E_b"] * 3600 # Đổi Wh sang Joules
    
    Ls_harvest = np.zeros_like(M_array)
    Ls_no_harvest = np.zeros_like(M_array)
    
    for i in range(len(M_array)):
        # 1. Trường hợp CÓ thu hoạch năng lượng
        if P_h[i] >= P_c[i]:
            Ls_harvest[i] = np.nan  
        else:
            Ls_harvest[i] = E_b_joules / (P_c[i] - P_h[i])
            
        # 2. Trường hợp KHÔNG thu hoạch năng lượng (P_h = 0)
        Ls_no_harvest[i] = E_b_joules / P_c[i]
            
    # Vẽ đường có thu hoạch (nét liền) và lưu handle lại
    label_harvest = f"$E_b$= {config['E_b']} Wh (with HSF-EH scheme)"
    line_h, = plt.plot(M_array, Ls_harvest, 
             color=config["color"], marker=config["marker"], linestyle='-', markersize=5)
    handles_harvest.append(line_h)
    labels_harvest.append(label_harvest)
    
    # Vẽ đường KHÔNG thu hoạch (nét đứt) và lưu handle lại
    label_no_harvest = f"$E_b$= {config['E_b']} Wh (No Harvesting)"
    line_no, = plt.plot(M_array, Ls_no_harvest, 
             color=config["color"], marker=config["marker"], linestyle='--', markersize=5, alpha=0.6)
    handles_no.append(line_no)
    labels_no.append(label_no_harvest)

# Gộp danh sách lại: nửa đầu tiên (đầy cột 1) là có thu hoạch, nửa sau (đầy cột 2) là không thu hoạch
handles = handles_harvest + handles_no
labels = labels_harvest + labels_no

plt.title("Impact of Total UAV Mass ($M$) on Lifespan", fontsize=14, pad=15)
plt.xlabel("Total UAV Mass, $M = 10~kg + SC_{mass}$ (kg)", fontsize=12)
plt.ylabel("Lifespan, $L_s$ (s)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Truyền mảng đã sắp xếp gọn gàng vào legend
plt.legend(handles, labels, title="", fontsize=10, ncol=2)
plt.tight_layout()

plt.show()
