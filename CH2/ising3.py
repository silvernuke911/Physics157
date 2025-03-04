import numpy as np
import matplotlib.pyplot as plt

def science_plot(fontsize = 11):
    import scienceplots
    plt.style.use(['science','grid','notebook'])
    plt.rcParams.update({
        'font.size'       : fontsize,    # General font size
        'axes.titlesize'  : fontsize,    # Font size of the axes title
        'axes.labelsize'  : fontsize,    # Font size of the axes labels
        'xtick.labelsize' : fontsize,    # Font size of the x-axis tick labels
        'ytick.labelsize' : fontsize,    # Font size of the y-axis tick labels
        'legend.fontsize' : fontsize,    # Font size of the legend
        'figure.titlesize': fontsize,    # Font size of the figure title
        'legend.fancybox' : False,       # Disable the fancy box for legend
        'legend.edgecolor': 'k',         # Set legend border color to black
        'text.usetex'     : True,        # Use LaTeX for text rendering
        'font.family'     : 'serif'      # Set font family to serif
    })
science_plot()

class IsingModel:
    Nx, Ny = 100, 100
    J, h, beta = 1, 0, 1
    spin_field = np.zeros((Nx, Ny))
    rng = np.random.default_rng()

    def __init__(self, Nx, Ny, J=1, h=0, beta=1, p_init=0.5):
        self.Nx, self.Ny = Nx, Ny
        self.spin_field = np.sign(self.rng.random((Nx, Ny)) - p_init)
        self.J = J
        self.beta = beta
        self.h = h

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
        return energy / 2

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

    def get_spin_configuration(self):
        return self.spin_field

Nx, Ny = 100, 100
num_steps = Nx * Ny * 100
params = [(1, 0.1), (-1, 0.1), (1, 10), (-1, 10)]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, (J, beta) in zip(axes.flatten(), params):
    model = IsingModel(Nx, Ny, J=J, beta=beta)
    model.simulate_multistep(num_steps)
    ax.imshow(model.get_spin_configuration(), cmap='gray')
    ax.set_title(f'J={J}, $\\beta$={beta}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

plt.tight_layout()
plt.savefig('Ising.png')
plt.show()

