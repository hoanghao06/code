import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from arg_data import CarsPath
from channel import get_fso_access, data_rate, get_solar_power, get_fso, get_fso_backhaul, get_fso_harvested_power, get_snr, irs_gain, H_cloud_max, H_cloud_min
from store_file import Buffer

# car_speed = 15  # m/s
car_force = 5  # gia tốc tối đa của xe (m/s^2)
# uav_height = 100  # m
# target_rate = 4.0e2  # Mbs
slot_time = 1  # s
fso_power = 15  # dBm
energy_ratio = 0.2 


class MakeEnv(gym.Env):
    def __init__(self, set_num, car_speed, target_rate):
        self.car_num = 3
        self.car_speed = car_speed
        # load
        self.cars_path = CarsPath()
        self._max_episode_steps = self.cars_path.max_time
        # store
        self.buffer = Buffer(max_time=self._max_episode_steps + 1, car_num=self.car_num)
        # self.p_fso_max = fso_power * np.ones(shape=(self.car_num,))  # average power in dBm
        self.p_fso_max = fso_power
        self.target_rate = target_rate # R_E
        self.alpha = 10  # Hệ số phạt (lớn)
        self.beta = 5       # Trọng số rate (ưu tiên cao)
        self.gamma = 0.01  # Trọng số energy (nhỏ hơn)
        self.hap_pos = np.array([0, 0, 20000]) 
        self.irs_pos = np.array([0, 0, 80])
        # edge constraint
        self.target_rate = target_rate
        self.delta_rate = target_rate * 1.0  # Mbps
        # [-500, -500, 0] -> [500, 500, 100]
        self.uav_acc_edge = np.array([0, 10], dtype=np.float32)  # Giới hạn độ lớn gia tốc của UAV m/s^2

        self.uav_velocity_edge = np.array([0, 30], dtype=np.float32)  # Giới hạn tốc độ bay tối đa m/s
        
        self.env_edge = np.array([[-500, 500], [-500, 500], [0, 5000]], dtype=np.float32)  # Không gian hoạt động của hệ thống m
        self.max_env_distance = np.sqrt((self.env_edge[0][1] - self.env_edge[0][0])**2 + 
                                        (self.env_edge[1][1] - self.env_edge[1][0])**2 + 
                                        (self.env_edge[2][1] - self.env_edge[2][0])**2)
        # [0, 1], advice: 2 ** n
        observation_spaces = gym.spaces.Box(low=np.zeros(shape=(self.car_num + 4,), dtype=np.float32),
                                            high=np.ones(shape=(self.car_num + 4,), dtype=np.float32))
        self.observation_space = observation_spaces
        # [-1, 1]
        action_spaces = gym.spaces.Box(low=-1 * np.ones(shape=(3,), dtype=np.float32),
                                       high=np.ones(shape=(3,), dtype=np.float32))
        self.action_space = action_spaces

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0  # slot = 1s, max_time = 1000s
        self.buffer.clear()  # reset Buffer
        temp_car_init_pos = self.cars_path.load(speed=self.car_speed, force=car_force, num=self.car_num)
        self.obj_point = self.cars_path.obj_pos
        # [-200, 200], uav-pos in m
        # self.uav_pos = np.mean(temp_car_init_pos, axis=0) + np.array([0, 0., uav_height])
        self.uav_pos = np.array([0.0, 0.0, 700.0], dtype=np.float32)
        self.uav_acc_xyz = np.zeros(shape=(3,), dtype=np.float32) # khởi tạo gia tốc
        self.pre_acc_xyz = self.uav_acc_xyz
        # velocity [0, pi]
        self.vel_theta = 0. # góc lệch ban đầu
        self.uav_velocity_xyz = np.zeros(shape=(3,), dtype=np.float32)


        state = self.deal_data()
        info = {}
        return state, info

    def step(self, action):
        self.time += 1
        info = {}
        terminated = False
        truncated = False
        if self.time >= self._max_episode_steps:
            truncated = True

        acc_theta = action[0] * np.pi            # Góc phương vị (XY): [-pi, pi]
        acc_phi = action[1] * (np.pi / 2)        # Góc ngẩng (Trục Z): [-pi/2, pi/2]
        acc_mod = self.uav_acc_edge[1] * (action[2] + 1) / 2  # Độ lớn: [0, 5]
        if acc_mod < self.uav_acc_edge[1] * 0.01:
            acc_mod *= 0.

        # 2. TÍNH GIA TỐC 3 TRỤC
        a_x = acc_mod * np.cos(acc_phi) * np.cos(acc_theta)
        a_y = acc_mod * np.cos(acc_phi) * np.sin(acc_theta)
        a_z = acc_mod * np.sin(acc_phi)

        self.pre_acc_xyz = self.uav_acc_xyz
        self.uav_acc_xyz = np.array([a_x, a_y, a_z], dtype=np.float32)

        # 3. CẬP NHẬT VỊ TRÍ 3D
        delta_tran = self.uav_velocity_xyz * slot_time + 0.5 * self.uav_acc_xyz * (slot_time ** 2)
        self.uav_pos += delta_tran

        # 4. CẬP NHẬT VẬN TỐC 3D
        self.uav_velocity_xyz += self.uav_acc_xyz * slot_time

        # Phạt và nảy lại nếu đụng ranh giới bản đồ
        reward_penalty = self.rectify_pos()

        # Giới hạn tốc độ bay tối đa
        self.vel_mod = np.linalg.norm(self.uav_velocity_xyz)
        if self.vel_mod > self.uav_velocity_edge[1]:
            ratio = self.uav_velocity_edge[1] / self.vel_mod
            self.uav_velocity_xyz *= ratio

        state = self.deal_data()
        reward = self.get_reward() + reward_penalty # Cộng thêm điểm phạt nếu đập tường

        return state, reward, terminated, truncated, info

    def render(self):
        now_time = self.time
        if not now_time % 10:
            plt.ion()  # 将画图模式改为交互模式
            plt.draw()
            # 设置三维图形模式
            ax = plt.axes(projection='3d')
            ax.set(xlabel='X', ylabel='Y', zlabel='Z',
                   title='Trans', xlim=self.env_edge[0],
                   ylim=self.env_edge[1], zlim=self.env_edge[2])
            # X = np.linspace(-R * 2, R * 2, 10)
            # Y = np.linspace(-R * 2, R * 2, 10)
            # X, Y = np.meshgrid(X, Y)
            # ax.plot_surface(X, Y, X * 0 + uav_height, alpha=0.2)
            # 线条
            for i in range(self.obj_point.shape[0]):
                left_point = self.obj_point[i, 0:3]
                right_point = self.obj_point[i, 3:]
                x = left_point[0]
                y = left_point[1]
                z = left_point[2]
                dx = right_point[0] - x
                dy = right_point[1] - y
                dz = right_point[2] - z
                
                xx = np.linspace(x, x+dx, 2)
                yy = np.linspace(y, y+dy, 2)
                zz = np.linspace(z, z+dz, 2)
                
                yy2, zz2 = np.meshgrid(yy, zz)
                ax.plot_surface(np.full_like(yy2, x), yy2, zz2, color='xkcd:light blue')
                ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2, color='xkcd:light blue')
                
                xx2, yy2 = np.meshgrid(xx, yy)
                ax.plot_surface(xx2, yy2, np.full_like(xx2, z), color='xkcd:light blue')
                ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz), color='xkcd:light blue')
                
                xx2, zz2= np.meshgrid(xx, zz)
                ax.plot_surface(xx2, np.full_like(yy2, y), zz2, color='xkcd:light blue')
                ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2, color='xkcd:light blue')

            ax.scatter3D(
                self.uav_pos[0], self.uav_pos[1], self.uav_pos[2], 'r*')
            # print(self.uav_pos[2])
            trans = self.buffer.uav_info['position'][:now_time]
            ax.plot3D(trans[:, 0], trans[:, 1], trans[:, 2], 'r')
            data = self.buffer.car_info
            for i in range(self.car_num):
                name = 'car_' + str(i)
                temp = data[name][:now_time]
                ax.plot3D(temp[:, 0], temp[:, 1], temp[:, 2], '--')
                # if self.h_fso[i] == 0:
                #     temp = self.cars_positions_xy[i][now_time, :]
                #     temp = np.c_[self.uav_position, np.r_[temp, 0]]
                #     ax.plot3D(temp[0], temp[1], temp[2])
                #     # print(temp)
                #     plt.pause(1)
            # plt.view
            plt.pause(0.1)
            plt.ioff()
            plt.clf()

    def seed(self, seed=None):
        seed = np.random.seed(seed)
        return [seed]

    def deal_data(self):
        self.delta_acc_acc_xyz = self.uav_acc_xyz - self.pre_acc_xyz
        self.pre_acc_xyz = self.uav_acc_xyz
        inter_index, self.cars_pos_list, self.distance = self.cars_path.get_inter_distance(time=self.time,
                                                                                           point=self.uav_pos)
        z_uav = self.uav_pos[2]
        self.P_solar = get_solar_power(z_uav) 
        # Tính năng lượng FSO nhận từ Backhaul theo 3 kịch bản mây
        if z_uav >= H_cloud_max:
            # 1. UAV bay trên mây: Nhận trực tiếp từ HAP, không có IRS khuếch đại
            h_total_fso, _, _, _ = get_fso(self.hap_pos, self.uav_pos)
            self.P_R = get_fso_harvested_power(h_total_fso, gain_factor=1)
            
        elif H_cloud_min <= z_uav < H_cloud_max:
            # 2. UAV bay trong mây: Nhận qua IRS, có khuếch đại
            h_hap_irs, _, _, _ = get_fso(self.hap_pos, self.irs_pos)
            h_irs_uav, _, _, _ = get_fso_backhaul(self.uav_pos, self.irs_pos)
            h_total_fso = h_hap_irs * h_irs_uav
            self.P_R = get_fso_harvested_power(h_total_fso, gain_factor=irs_gain)
            
        else:
            # 3. UAV bay dưới mây: Nhận qua IRS, có khuếch đại
            h_hap_irs, _, _, _ = get_fso(self.hap_pos, self.irs_pos)
            h_irs_uav, _, _, _ = get_fso_backhaul(self.uav_pos, self.irs_pos)
            h_total_fso = h_hap_irs * h_irs_uav
            self.P_R = get_fso_harvested_power(h_total_fso, gain_factor=irs_gain) 
        P_transmit_total = self.P_R * (1 - energy_ratio)     # 80% để phát
        P_transmit_per_car = P_transmit_total / self.car_num # Chia đều cho các xe
        # NĂNG LƯỢNG THỰC TẾ SẠC VÀO PIN 
        self.P_battery = self.P_solar + (self.P_R * energy_ratio)

        h_fso_list = []
        gamma_F_list = []
        fso_rate_list = []
        FSO_bandwidth= 1.0  # fso bandwidth

        for car_pos in self.cars_pos_list:
            if len(car_pos) == 2:
                car_pos_3d = np.append(car_pos, 2.0)
            else:
                car_pos_3d = car_pos
            # Gọi hàm tính kênh FSO từ UAV xuống Vehicle
            h_total, hc, ha, hs = get_fso_access(self.uav_pos, car_pos_3d)
            h_fso_list.append(h_total)
            gamma_F = get_snr(h_total, P_transmit_per_car, self.uav_pos)
            gamma_F_list.append(gamma_F)
            
            # Tính tốc độ dữ liệu (Data Rate) và chuyển sang Mbps
            rate_bps = data_rate(gamma_F, FSO_bandwidth)
            rate_bps = min(rate_bps, 4)
            fso_rate_list.append(rate_bps) # Gbps

        self.h_fso = np.array(h_fso_list)
        self.gamma_F = np.array(gamma_F_list)
        self.FSO_rate = np.array(fso_rate_list)
        # Giả sử tốc độ tổng hiện tại chỉ có FSO
        self.r_all = self.FSO_rate 
        self.real_rate = self.r_all 
        self.store()
        
        # 3. TẠO TRẠNG THÁI (STATE)
        # Chuẩn hóa khoảng cách có nhiễu Gaussian
        # Xác định khoảng cách đường chéo lớn nhất trong môi trường hoạt độn

        # Chia cho max_distance để đảm bảo dist_noisy luôn nằm trong [0, 1] trước khi vào hàm clip
        dist_noisy = (self.distance + np.random.normal(loc=0, scale=2, size=(self.car_num,))) / self.max_env_distance
        # Chuẩn hóa vận tốc
        vel_norm = (self.uav_velocity_xyz / self.uav_velocity_edge[1] + 1) / 2
        # Chuẩn hóa năng lượng (để đưa vào mạng neural)
        max_energy_expected = 500.0 
        energy_norm = np.array([self.P_battery / max_energy_expected])

        # Ghép lại thành vector state
        state = np.clip(np.r_[vel_norm, energy_norm, dist_noisy], 0, 1)
        if np.isnan(state).any() or np.isinf(state).any():
            print("PHÁT HIỆN LỖI NaN TẠI STATE!")
            print("Khoảng cách:", self.distance)
            print("Hệ số kênh truyền FSO:", self.h_fso)
            print("Năng lượng P_battery:", self.P_battery)
        return state.astype(np.float32)

    def get_reward(self):
        R_min = self.target_rate
        R_t = self.r_all  # Mảng tốc độ dữ liệu của các phương tiện
        E_t = self.P_battery # Chuẩn hóa năng lượng thu được tại bước t
        
        # Phân tách 2 trường hợp R_t < R_min và R_t >= R_min bằng boolean mask
        penalty_mask = R_t < R_min
        bonus_mask = R_t >= R_min
        
        # Khởi tạo mảng reward cùng kích thước với R_t
        reward_array = np.zeros_like(R_t)
        
        # Tính reward cho trường hợp R_t < R_min
        reward_array[penalty_mask] = -self.alpha * (R_min - R_t[penalty_mask])
        
        # Tính reward cho trường hợp R_t >= R_min
        reward_array[bonus_mask] = self.beta * (R_t[bonus_mask] - R_min) + self.gamma * E_t
        
        # Lấy giá trị trung bình làm phần thưởng tổng thể cho môi trường
        reward = np.mean(reward_array)
        
        return float(reward)


    def store(self):
        uav = [self.uav_pos, self.uav_velocity_xyz, self.uav_acc_xyz]
        car = self.cars_pos_list
        rate = [self.r_all, self.r_all.mean()]
        energy = [self.P_battery, self.P_solar, self.P_R * energy_ratio]
        channel = [self.h_fso]
        self.buffer.update(uav_info=uav, car_info=car, rate_info=rate, channel_info=channel, energy_info=energy)

    def rectify_pos(self):
        # [x, y, h]
        reward = 0
        coeff = np.array([1, -1])
        delta_x = (self.uav_pos[0] - self.env_edge[0]) * coeff
        delta_y = (self.uav_pos[1] - self.env_edge[1]) * coeff
        delta_z = (self.uav_pos[2] - self.env_edge[2]) * coeff
        if (delta_x < 0).any():
            # Giữ vị trí cách biên 1 mét thay vì nhân 0.9 để tránh dịch chuyển tức thời
            self.uav_pos[0] = np.clip(self.uav_pos[0], self.env_edge[0][0] + 1.0, self.env_edge[0][1] - 1.0)
            self.uav_velocity_xyz[0] *= -0.5
            reward -= 0.5
            
        # Xử lý bật tường trục Y
        if (delta_y < 0).any():
            self.uav_pos[1] = np.clip(self.uav_pos[1], self.env_edge[1][0] + 1.0, self.env_edge[1][1] - 1.0)
            self.uav_velocity_xyz[1] *= -0.5
            reward -= 0.5
            
        # Nảy lại ở mặt đất (Z=0) và trần bay (Z=5000)
        if (delta_z < 0).any():
            if self.uav_pos[2] < self.env_edge[2][0]:
                self.uav_pos[2] = 10.0 # Không cho chìm xuống đất, bật lên 10m
            elif self.uav_pos[2] > self.env_edge[2][1]:
                # Giữ cách trần 1 mét
                self.uav_pos[2] = self.env_edge[2][1] - 1
            self.uav_velocity_xyz[2] *= -0.5
            reward -= 1.0 # Phạt nặng hơn nếu rớt

        return reward

    def numpy_cube_one(x=0, y=0, z=0, dx=50, dy=50, dz=50):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xx = np.linspace(x, x+dx, 2)
        yy = np.linspace(y, y+dy, 2)
        zz = np.linspace(z, z+dz, 2)
        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, z))
        ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz))
        yy2, zz2 = np.meshgrid(yy, zz)
        ax.plot_surface(np.full_like(yy2, x), yy2, zz2)
        ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2)
        xx2, zz2= np.meshgrid(xx, zz)
        ax.plot_surface(xx2, np.full_like(yy2, y), zz2)
        ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2)
        #坐标及其刻度隐藏
        plt.title("Cube")
        plt.show()

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
