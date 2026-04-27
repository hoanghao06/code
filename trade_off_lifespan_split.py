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

def get_dynamic_solar_power(SC_mass):
    """
    Quy đổi khối lượng (kg) sang diện tích (m^2). 
    Dựa trên code gốc: Trọng lượng 2.45 N (~0.25 kg) tương ứng với 0.5 m^2.
    => 1 kg tương ứng với 2.0 m^2.
    """
    A_solar = SC_mass * 2.0 
    
    eta_p = 0.9           
    G_r = 1361            
    A_0_solar = 0.8978    
    B_0_solar = 0.2804    
    delta = 8000          
    
    # Độ truyền qua khí quyển (giả định ở độ cao z = 100m)
    alpha_a = A_0_solar - B_0_solar * np.exp(-100 / delta)
    
    return eta_p * A_solar * G_r * alpha_a

# =====================================================================
# CHẠY THỬ NGHIỆM VÀ VẼ ĐỒ THỊ
# =====================================================================

# 1. Các thông số đầu vào cố định
U_mass = 10.0                 # Khối lượng UAV cố định (kg)
g = 9.81                      # Gia tốc trọng trường (m/s^2)

# ---------------------------------------------------------
# CỐ ĐỊNH VẬN TỐC UAV 
# ---------------------------------------------------------
velocity = 20.0
print(f"[+] Đã cố định vận tốc UAV: {velocity} m/s")

# ---------------------------------------------------------
# TRÍCH XUẤT NĂNG LƯỢNG FSO THỰC TỪ FILE ENERGY_3.NPY
# ---------------------------------------------------------
try:
    energy_data = np.load(r'C:\Users\DELL\Desktop\nckh\prj1\2026.-Tien_Hao-main\2026.-Tien_Hao-main\main_output\output_rural_10\speed_10\0\flydata\energy_3.npy', allow_pickle=True).item()
    fso_energy_array = energy_data['fso energy']
    
    # Lấy giá trị trung bình trên toàn quỹ đạo làm công suất thực tế
    P_battery_fso = np.mean(fso_energy_array)
    
    print(f"[+] Trích xuất dữ liệu năng lượng thành công.")
    print(f"[+] Công suất FSO sạc vào pin trung bình (P_battery_fso): {P_battery_fso:.4e} Watt")
except Exception as e:
    print(f"[-] Lỗi khi đọc file energy_3.npy: {e}")
    P_battery_fso = 0.0

# ---------------------------------------------------------
# TÍNH TOÁN TUỔI THỌ VÀ VẼ ĐỒ THỊ
# ---------------------------------------------------------

# 3 cấu hình dung lượng pin E_b (Wh)
battery_configs = [
    {"E_b": 291.72, "color": "blue",    "marker": "o"}, # Kịch bản 1
    {"E_b": 276.00, "color": "teal",    "marker": "s"}, # Kịch bản 2
    {"E_b": 246.84, "color": "black",   "marker": "^"}, # Kịch bản 3
]

# Trục X: Khối lượng tấm pin (SC_mass) thay đổi từ 0.05 kg đến 0.5 kg
SC_mass_array = np.linspace(0.05, 0.5, 20)

# Mảng Tổng khối lượng UAV (Mass) - CHÍNH LÀ TRỤC X
M_array = U_mass + SC_mass_array
W_array = M_array * g

plt.figure(figsize=(10, 7))

# Khởi tạo mô hình Năng lượng 1 lần
uav_model = UAVEnergyModelDynamicW(W=W_array)

# Tính công suất tiêu thụ P_c
P_propulsion = uav_model.propulsion_power(velocity)
P_c = P_propulsion + uav_model.P_c

# Tính công suất thu hoạch P_h
P_solar = get_dynamic_solar_power(SC_mass_array)
# Cộng công suất FSO trung bình từ file vào tổng công suất thu hoạch
P_h = P_solar + P_battery_fso 

# 2. Vòng lặp vẽ đồ thị cho từng dung lượng pin Eb
for config in battery_configs:
    E_b_joules = config["E_b"] * 3600 # Đổi Wh sang Joules (Watt-giây)
    
    # Tính Lifespan 
    Ls = np.zeros_like(M_array)
    for i in range(len(M_array)):
        if P_h[i] >= P_c[i]:
            Ls[i] = np.nan  # Thu hoạch nhiều hơn tiêu thụ -> Vô hạn
        else:
            Ls[i] = E_b_joules / (P_c[i] - P_h[i])
            
    # Vẽ đường cong theo khối lượng M_array
    label = f"$E_b$= {config['E_b']} Wh"
    plt.plot(M_array, Ls, label=label, 
             color=config["color"], marker=config["marker"], linestyle='-', markersize=5)

# 3. Tinh chỉnh đồ thị
plt.title("Impact of Total UAV Mass ($M$) on Lifespan", fontsize=14, pad=15)
plt.xlabel("Total UAV Mass, $M = 10~kg + SC_{mass}$ (kg)", fontsize=12)
plt.ylabel("Lifespan, $L_s$ (s)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Battery Capacity", fontsize=10)
plt.tight_layout()

# Hiển thị
plt.show()
