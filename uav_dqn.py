import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from arg_data import CarsPath
from channel import get_fso_access, data_rate, get_solar_power, get_fso, get_fso_backhaul, get_fso_harvested_power, get_snr, irs_gain, H_cloud_max, H_cloud_min
from store_file import Buffer

car_force = 5
slot_time = 1
fso_power = 15  # dBm
energy_ratio = 0.2

class MakeEnv(gym.Env):
    def __init__(self, set_num, car_speed, target_rate):
        self.car_num = set_num
        self.car_speed = car_speed
        self.cars_path = CarsPath()
        self._max_episode_steps = self.cars_path.max_time
        self.buffer = Buffer(max_time=self._max_episode_steps + 1, car_num=self.car_num)
        self.p_fso_max = fso_power
        self.target_rate = target_rate
        self.alpha = 10
        self.beta = 5
        self.gamma_energy = 0.01
        self.hap_pos = np.array([0, 0, 20000])
        self.irs_pos = np.array([0, 0, 80])

        self.uav_acc_edge = np.array([0, 10], dtype=np.float32)
        self.uav_velocity_edge = np.array([0, 20], dtype=np.float32)
        self.env_edge = np.array([[0, 600], [0, 600], [0, 3000]], dtype=np.float32)
        self.max_env_distance = np.sqrt((self.env_edge[0][1] - self.env_edge[0][0])**2 +
                                        (self.env_edge[1][1] - self.env_edge[1][0])**2 +
                                        (self.env_edge[2][1] - self.env_edge[2][0])**2)

        # Action space: 9 discrete actions (0: đứng yên, 1..8: 8 hướng với tốc độ 80% max_acc)
        self.n_actions = 9
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # Observation space: [vx_norm, vy_norm, vz_norm, energy_norm, dist1_norm, ..., distN_norm]
        obs_dim = 3 + 1 + self.car_num   # vel(3) + energy(1) + distances(car_num)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.buffer.clear()
        temp_car_init_pos = self.cars_path.load(speed=self.car_speed, force=car_force, num=self.car_num)
        self.obj_point = self.cars_path.obj_pos
        self.uav_pos = np.array([0.0, 0.0, 500.0], dtype=np.float32)
        self.uav_acc_xyz = np.zeros(3, dtype=np.float32)
        self.pre_acc_xyz = np.zeros(3, dtype=np.float32)
        self.uav_velocity_xyz = np.zeros(3, dtype=np.float32)
        state = self._get_state()
        return state, {}

    def step(self, action_idx):
        self.time += 1
        truncated = (self.time >= self._max_episode_steps)
        terminated = False

        # Map action_idx (0..8) -> (acc_theta, acc_phi, acc_mod)
        # action 0: dừng, action 1..8: 8 hướng (45° bước) trên mặt phẳng XY, góc ngẩng=0, mod=80% max_acc
        max_acc = self.uav_acc_edge[1]
        if action_idx == 0:
            acc_theta = 0.0
            acc_phi = 0.0
            acc_mod = 0.0
        else:
            angle_deg = (action_idx - 1) * 45   # 0,45,90,...,315 độ
            acc_theta = np.deg2rad(angle_deg)
            acc_phi = 0.0   # giữ nguyên độ cao
            acc_mod = 0.8 * max_acc

        # Tính gia tốc 3D
        a_x = acc_mod * np.cos(acc_phi) * np.cos(acc_theta)
        a_y = acc_mod * np.cos(acc_phi) * np.sin(acc_theta)
        a_z = acc_mod * np.sin(acc_phi)

        self.pre_acc_xyz = self.uav_acc_xyz.copy()
        self.uav_acc_xyz = np.array([a_x, a_y, a_z], dtype=np.float32)

        # Cập nhật vị trí và vận tốc
        self.uav_pos += self.uav_velocity_xyz * slot_time + 0.5 * self.uav_acc_xyz * (slot_time**2)
        self.uav_velocity_xyz += self.uav_acc_xyz * slot_time

        # Xử lý biên và phạt
        reward_penalty = self._rectify_pos()

        # Giới hạn tốc độ
        vel_norm = np.linalg.norm(self.uav_velocity_xyz)
        if vel_norm > self.uav_velocity_edge[1]:
            self.uav_velocity_xyz *= (self.uav_velocity_edge[1] / vel_norm)

        state = self._get_state()
        reward = self._get_reward() + reward_penalty

        return state, reward, terminated, truncated, {}

    def _get_state(self):
        # Khoảng cách đến các xe (có nhiễu Gaussian)
        _, _, dist = self.cars_path.get_inter_distance(time=self.time, point=self.uav_pos)
        dist_noisy = (dist + np.random.normal(0, 2, size=self.car_num)) / self.max_env_distance
        dist_noisy = np.clip(dist_noisy, 0, 1)

        # Vận tốc chuẩn hóa
        vel_norm = (self.uav_velocity_xyz / self.uav_velocity_edge[1] + 1) / 2
        vel_norm = np.clip(vel_norm, 0, 1)

        # Năng lượng hiện tại (ước lượng)
        z_uav = self.uav_pos[2]
        p_solar = get_solar_power(z_uav)
        # Tính P_R từ backhaul (giống như trong deal_data)
        if z_uav >= H_cloud_max:
            h_total, _, _, _ = get_fso(self.hap_pos, self.uav_pos)
            p_r = get_fso_harvested_power(h_total, gain_factor=1)
        elif H_cloud_min <= z_uav < H_cloud_max:
            h_hap_irs, _, _, _ = get_fso(self.hap_pos, self.irs_pos)
            h_irs_uav, _, _, _ = get_fso_backhaul(self.uav_pos, self.irs_pos)
            h_total = h_hap_irs * h_irs_uav
            p_r = get_fso_harvested_power(h_total, gain_factor=irs_gain)
        else:
            h_hap_irs, _, _, _ = get_fso(self.hap_pos, self.irs_pos)
            h_irs_uav, _, _, _ = get_fso_backhaul(self.uav_pos, self.irs_pos)
            h_total = h_hap_irs * h_irs_uav
            p_r = get_fso_harvested_power(h_total, gain_factor=irs_gain)
        p_battery = p_solar + (p_r * energy_ratio)
        max_energy_expected = 1000.0
        energy_norm = np.array([p_battery / max_energy_expected], dtype=np.float32)

        state = np.concatenate([vel_norm, energy_norm, dist_noisy])
        state = np.clip(state, 0, 1).astype(np.float32)
        return state

    def _get_reward(self):
        # Tính tốc độ FSO xuống các xe
        _, _, dist = self.cars_path.get_inter_distance(time=self.time, point=self.uav_pos)
        z_uav = self.uav_pos[2]
        p_solar = get_solar_power(z_uav)
        # P_R từ backhaul (tính lại hoặc lưu từ _get_state)
        if z_uav >= H_cloud_max:
            h_total, _, _, _ = get_fso(self.hap_pos, self.uav_pos)
            p_r = get_fso_harvested_power(h_total, gain_factor=1)
        elif H_cloud_min <= z_uav < H_cloud_max:
            h_hap_irs, _, _, _ = get_fso(self.hap_pos, self.irs_pos)
            h_irs_uav, _, _, _ = get_fso_backhaul(self.uav_pos, self.irs_pos)
            h_total = h_hap_irs * h_irs_uav
            p_r = get_fso_harvested_power(h_total, gain_factor=irs_gain)
        else:
            h_hap_irs, _, _, _ = get_fso(self.hap_pos, self.irs_pos)
            h_irs_uav, _, _, _ = get_fso_backhaul(self.uav_pos, self.irs_pos)
            h_total = h_hap_irs * h_irs_uav
            p_r = get_fso_harvested_power(h_total, gain_factor=irs_gain)
        p_battery = p_solar + (p_r * energy_ratio)

        p_tx_total = p_r * (1 - energy_ratio)
        p_tx_per_car = p_tx_total / self.car_num

        fso_rates = []
        car_positions = self.cars_path.get_inter_distance(time=self.time, point=self.uav_pos)[1]  # list of car positions
        for car_pos in car_positions:
            if len(car_pos) == 2:
                car_pos = np.append(car_pos, 2.0)
            h_fso, _, _, _ = get_fso_access(self.uav_pos, car_pos)
            gamma = get_snr(h_fso, p_tx_per_car, self.uav_pos)
            rate_bps = data_rate(gamma, 1.0)  # bandwidth 1 GHz
            fso_rates.append(rate_bps)
        rates = np.array(fso_rates)  # Gbps
        R_min = self.target_rate
        penalty_mask = rates < R_min
        bonus_mask = rates >= R_min
        reward_array = np.zeros_like(rates)
        reward_array[penalty_mask] = -self.alpha * (R_min - rates[penalty_mask])
        reward_array[bonus_mask] = self.beta + self.gamma_energy * p_battery
        return float(np.mean(reward_array))

    def _rectify_pos(self):
        reward = 0
        # X
        if self.uav_pos[0] < self.env_edge[0][0]:
            self.uav_pos[0] = self.env_edge[0][0] + 1.0
            self.uav_velocity_xyz[0] *= -0.5
            reward -= 0.5
        elif self.uav_pos[0] > self.env_edge[0][1]:
            self.uav_pos[0] = self.env_edge[0][1] - 1.0
            self.uav_velocity_xyz[0] *= -0.5
            reward -= 0.5
        # Y
        if self.uav_pos[1] < self.env_edge[1][0]:
            self.uav_pos[1] = self.env_edge[1][0] + 1.0
            self.uav_velocity_xyz[1] *= -0.5
            reward -= 0.5
        elif self.uav_pos[1] > self.env_edge[1][1]:
            self.uav_pos[1] = self.env_edge[1][1] - 1.0
            self.uav_velocity_xyz[1] *= -0.5
            reward -= 0.5
        # Z
        if self.uav_pos[2] < self.env_edge[2][0]:
            self.uav_pos[2] = self.env_edge[2][0] + 10.0
            self.uav_velocity_xyz[2] *= -0.5
            reward -= 1.0
        elif self.uav_pos[2] > self.env_edge[2][1]:
            self.uav_pos[2] = self.env_edge[2][1] - 1.0
            self.uav_velocity_xyz[2] *= -0.5
            reward -= 1.0
        return reward

    def store(self):
        # Giữ lại cách lưu như cũ, nhưng có thể cần sửa store_file.py cho phù hợp
        # Ở đây tạm thời bỏ qua để tránh lỗi, nếu cần thì giữ nguyên
        pass

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def render(self):
        # Có thể giữ nguyên hoặc bỏ qua
        pass