import numpy as np

class UAVEnergyModel:
    def __init__(self):
        # --- CÁC THÔNG SỐ TỪ BẢNG I (TABLE I) ---
        self.W = 100.0          # Trọng lượng UAV (N)
        self.rho = 1.225        # Mật độ không khí (kg/m^3)
        self.R = 0.5            # Bán kính cánh quạt (m)
        self.A = 0.79           # Diện tích đĩa cánh quạt (m^2)
        self.U_tip = 200.0      # Tốc độ mũi cánh quạt (m/s)
        self.s = 0.05           # Độ đặc của rotor (Rotor solidity)
        self.d_0 = 0.3          # Hệ số cản thân máy bay (Fuselage drag ratio)
        self.v_0 = 7.2          # Vận tốc cảm ứng trung bình khi bay lơ lửng (m/s)
        self.delta = 0.012      # Hệ số cản biên dạng (Profile drag coefficient)
        self.k = 0.1            # Hệ số hiệu chỉnh công suất cảm ứng
        
        # --- TÍNH TOÁN P_0 VÀ P_i THEO PHỤ LỤC A, PHƯƠNG TRÌNH (61) ---
        # P_0 = (delta / 8) * rho * s * A * (Omega^3 * R^3) -> (Omega*R) chính là U_tip
        self.P_0 = (self.delta / 8) * self.rho * self.s * self.A * (self.U_tip ** 3)
        
        # P_i = (1 + k) * (W^(3/2)) / sqrt(2 * rho * A)
        self.P_i = (1 + self.k) * (self.W ** 1.5) / np.sqrt(2 * self.rho * self.A)

        # Công suất tiêu thụ cho mạch truyền thông (Mục V. Numerical Results)
        self.P_c = 50.0 

    def velocity_3d(self, v_x, v_y, v_z):
        """Tính độ lớn vận tốc V (scalar) từ các vector vận tốc trong tọa độ 3D."""
        return np.sqrt(v_x**2 + v_y**2 + v_z**2)

    def propulsion_power(self, V):
        """
        Tính toán công suất đẩy P(V) dựa trên phương trình (37) trong ảnh
        và phương trình (6) trong bài báo.
        """
        # Tránh lỗi chia cho 0 hoặc căn số âm
        V = np.maximum(V, 0.0)
        
        # Thành phần 1: Blade profile power
        term1 = self.P_0 * (1 + (3 * V**2) / (self.U_tip**2))
        
        # Thành phần 2: Induced power
        # Công thức: P_i * sqrt( sqrt(1 + V^4 / 4*v_0^4) - V^2 / 2*v_0^2 )
        inner_sqrt = np.sqrt(1 + (V**4) / (4 * self.v_0**4))
        term2 = self.P_i * np.sqrt(inner_sqrt - (V**2) / (2 * self.v_0**2))
        
        # Thành phần 3: Parasite power
        # Lưu ý: S trong ảnh = A (Rotor disc area) trong bài báo
        term3 = 0.5 * self.d_0 * self.rho * self.s * self.A * (V**3)
        
        return term1 + term2 + term3

    def total_energy(self, velocities, dt, include_communication=True):
        """
        Tính tổng năng lượng tiêu thụ (Joule) qua một chuỗi các vận tốc theo thời gian.
        
        Tham số:
        - velocities: Mảng numpy chứa độ lớn vận tốc tại từng mốc thời gian.
        - dt: Bước thời gian mô phỏng (giây).
        """
        # Tính công suất đẩy tại mọi thời điểm
        P_prop = self.propulsion_power(velocities)
        
        # Năng lượng = Công suất * thời gian
        E_prop = np.sum(P_prop) * dt
        
        if include_communication:
            E_comm = self.P_c * len(velocities) * dt
            return E_prop + E_comm
            
        return E_prop

# # --- VÍ DỤ CÁCH SỬ DỤNG ---
# if __name__ == "__main__":
#     uav_model = UAVEnergyModel()
    
#     # Giả lập UAV bay trong 100 bước thời gian, bước nhảy 0.5 giây
#     # Vận tốc 3D (ví dụ: đang bay nghiêng)
#     v_x = np.full(300, 10.0) # Vận tốc trục x là 10 m/s
#     v_y = np.full(300, 5.0)  # Vận tốc trục y là 5 m/s
#     v_z = np.full(300, 0.0)  # Vận tốc trục z là 0 m/s (bay bằng)
    
#     # Quy đổi ra độ lớn vận tốc vô hướng
#     V_magnitude = uav_model.velocity_3d(v_x, v_y, v_z)
    
#     # Tính năng lượng
#     total_energy = uav_model.total_energy(V_magnitude, dt=0.5)
#     print(f"Tổng năng lượng tiêu thụ: {total_energy:.2f} Joules")