import numpy as np
import pandas as pd
# uniform

# N = 300
# d = 8
# dice_list = list(range(d))
# total_list = np.zeros(N, dtype = int)
# num_list = np.zeros(d, dtype = int)

# for i in range(N):
#     j = np.random.choice(dice_list)
#     num_list[j] += 1
#     total_list[i] = j

# for i in range(len(num_list)):
#     print(num_list[i])
        
print(np.sqrt(200), np.sqrt(100), np.sqrt(50))
c
## means CLT
def clt(d, n, N = 1000, l=200):
    print(N//n)
    means_list = np.full(l, np.nan)
    for i in range(N//n):
        j = np.random.randint(1,d+1,n)
        p = np.average(j)
        means_list[i] = p

    for i in range(len(means_list)):
        print(means_list[i])
    
    return means_list

np.random.seed(34)

d6_05 = clt(6,5)
d6_10 = clt(6,10)
d6_20 = clt(6,20)
data = {
    '5':d6_05,
    '10':d6_10,
    '15':d6_20
}
data = pd.DataFrame(data)
print(data)
data.to_csv(r'C:\Users\verci\Documents\Python Code\Physics157\PHYSICS191 CODE\SET A\E2\diffclt6.csv', index = False)

d8_05 = clt(8,5)
d8_10 = clt(8,10)
d8_20 = clt(8,20)
data = {
    '5':d8_05,
    '10':d8_10,
    '15':d8_20
}
data = pd.DataFrame(data)
print(data)
data.to_csv(r'C:\Users\verci\Documents\Python Code\Physics157\PHYSICS191 CODE\SET A\E2\diffclt8.csv', index = False)