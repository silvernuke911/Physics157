import numpy as np
import matplotlib.pyplot as plt

N = 100
p = 0.105
t_max = 100
t = np.arange(t_max)  # Use np.arange for proper indexing
samples = np.ones(N)
samplist = np.zeros(t_max)
diffs = []
samplist[0] = np.sum(samples)  # Set initial value
for ti in range(1, t_max):  # Start from 1
    difs = 0
    for i in range(N):
        choice = np.random.choice([0, 1], p=[p, 1 - p])
        if samples[i] == 1 and choice == 0:
            samples[i] = 0  
            difs +=1
    diffs.append(difs)
    samplist[ti] = np.sum(samples)  
    
print(samplist)
for i in range(68):
    print(diffs[i])
with open("results.txt", "w") as f:
    for i in range(70):
        f.write(str(diffs[i]) + "\n")  # Write each diffs[i] value on a new line
# plt.step(t, samplist)
# plt.xlabel("Time Step")
# plt.ylabel("Number of Active Samples")
# plt.title("Decay Process Simulation")
# plt.show()
