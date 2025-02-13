import numpy as np
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, Nx, Ny, J=1, h=0, beta=1, p_init=0.5):
        self.Nx, self.Ny = Nx, Ny
        self.J = J
        self.h = h
        self.beta = beta
        self.spin_field = np.sign(np.random.rand(Nx, Ny) - p_init)
        self.rng = np.random.default_rng()

    def compute_energy_full(self):
        energy = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                S = self.spin_field[i, j]
                neighbors = (
                    self.spin_field[(i + 1) % self.Nx, j] +
                    self.spin_field[(i - 1) % self.Nx, j] +
                    self.spin_field[i, (j + 1) % self.Ny] +
                    self.spin_field[i, (j - 1) % self.Ny]
                )
                energy += -self.J * S * neighbors - self.h * S
        return energy / 2  # Each pair counted twice

    def pick_random_site(self):
        return self.rng.integers(self.Nx), self.rng.integers(self.Ny)

    def compute_energy_change(self, x, y):
        S = self.spin_field[x, y]
        neighbors = (
            self.spin_field[(x + 1) % self.Nx, y] +
            self.spin_field[(x - 1) % self.Nx, y] +
            self.spin_field[x, (y + 1) % self.Ny] +
            self.spin_field[x, (y - 1) % self.Ny]
        )
        return 2 * S * (self.J * neighbors + self.h)

    def simulate_step(self):
        x, y = self.pick_random_site()
        delta_energy = self.compute_energy_change(x, y)
        if delta_energy < 0 or self.rng.random() < np.exp(-self.beta * delta_energy):
            self.spin_field[x, y] *= -1

    def simulate_multistep(self, num_flip_steps):
        for _ in range(num_flip_steps):
            self.simulate_step()

    def compute_magnetization(self):
        return np.abs(np.sum(self.spin_field)) / (self.Nx * self.Ny)

def compute_specific_heat(energies, beta):
    E_mean = np.mean(energies)
    E2_mean = np.mean(np.array(energies) ** 2)
    return beta ** 2 * (E2_mean - E_mean ** 2)

Nx, Ny = 10, 10
num_steps = Nx * Ny * 100  # Ensure sufficient thermalization
T_range = np.linspace(1.5, 3.5, 20)
magnetizations = []
specific_heats = []

for T in T_range:
    beta = 1 / T
    model = IsingModel(Nx, Ny, beta=beta)
    energies = []
    
    for _ in range(500):  # Take multiple measurements
        model.simulate_multistep(num_steps)
        energies.append(model.compute_energy_full())
    
    magnetizations.append(model.compute_magnetization())
    specific_heats.append(compute_specific_heat(energies, beta))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(T_range, magnetizations, marker='o')
plt.xlabel('Temperature T')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Temperature')

plt.subplot(1, 2, 2)
plt.plot(T_range, specific_heats, marker='s', color='r')
plt.xlabel('Temperature T')
plt.ylabel('Specific Heat')
plt.title('Specific Heat vs Temperature')

plt.tight_layout()
plt.show()
