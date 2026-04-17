# ================== IMPORT ==================
import numpy as np
import matplotlib.pyplot as plt

# ================== LOAD DATA ==================
data_energy = np.load(
    r'C:\Users\AVSTC\Desktop\2026.Globecom\output_35_Gbps\speed_10\0\flydata\energy_3.5.npy',
    allow_pickle=True
).item()

# ================== LẤY DỮ LIỆU ==================
total_energy = data_energy['total energy']
solar_energy = data_energy['solar energy']
fso_energy   = data_energy['fso energy']

# ================== HÀM XỬ LÝ ==================
def process_energy(energy):
    if len(energy.shape) > 1:
        return np.mean(energy, axis=1)
    return energy

total_energy = process_energy(total_energy)
solar_energy = process_energy(solar_energy)
fso_energy   = process_energy(fso_energy)

x = np.arange(len(total_energy))

# ================== PLOT 1: TOTAL ==================
plt.figure(figsize=(12, 6))
plt.plot(x, total_energy, linewidth=2, color='blue')
plt.title('Total Energy theo thời gian')
plt.xlabel('Timeslot n')
plt.ylabel('Energy')
plt.grid()
plt.tight_layout()
plt.show()

# ================== PLOT 2: SOLAR ==================
plt.figure(figsize=(12, 6))
plt.plot(x, solar_energy, linewidth=2, color='orange')
plt.title('Solar Energy theo thời gian')
plt.xlabel('Timeslot n')
plt.ylabel('Energy')
plt.grid()
plt.tight_layout()
plt.show()

# ================== PLOT 3: FSO ==================
plt.figure(figsize=(12, 6))
plt.plot(x, fso_energy, linewidth=2, color='green')
plt.title('FSO Energy theo thời gian')
plt.xlabel('Timeslot n')
plt.ylabel('Energy')
plt.grid()
plt.tight_layout()
plt.show()