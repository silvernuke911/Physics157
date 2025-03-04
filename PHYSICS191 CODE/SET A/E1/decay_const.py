import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,100,0.01)
p = 1/10
y1 = (1-p)**x



halflife = -np.log2(1-p)**(-1)

print(halflife)
y2 = 2**(- x / halflife)

plt.plot(x,y2, color = 'r', linewidth = 3)
plt.plot(x,y1, color = 'b')
plt.show()