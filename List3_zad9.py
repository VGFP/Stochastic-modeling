import numpy as np
from io import StringIO 
import matplotlib.pyplot as plt
import scipy.stats as stats
# Zadanie 4

loaded_data = []

with open("GEFCOM.txt", "r") as file:
    for line in file:
        strintIO_line = StringIO(line)
        npLine = np.loadtxt(strintIO_line)
        loaded_data.append(npLine)

loaded_data = np.array(loaded_data)

data_361_1082 = loaded_data[361*24:1082*24,2]

# Naive forecasting
data_361_1082_naive_pred = np.zeros((np.shape(data_361_1082)))
data_361_1082_naive_pred[0:24] = data_361_1082[0:24]

for i in range(24,17304):
    data_361_1082_naive_pred[i] = data_361_1082[i-24]

# HW

s = 4
alpha = 0.1
beta = 0.1
gamma = 0.1

L = np.zeros(17304)
T = L.copy()
S = L.copy()
fx = L.copy()
# Set initial values of L, T and S
L[s-1] = np.sum(data_361_1082[:s]) / s
T[s-1] = np.sum(data_361_1082[s:2*s] - data_361_1082[:s]) / (s ** 2)
S[:s] = data_361_1082[:s] - L[s-1]

# Iterate to compute L(t), T(t), S(t) and FX(t)
for t in range(s, len(data_361_1082)-1):
    L[t] = alpha * (data_361_1082[t] - S[t-s]) + (1 - alpha) * (L[t-1] + T[t-1])
    T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
    S[t] = gamma * (data_361_1082[t] - L[t]) + (1 - gamma) * S[t-s]
    fx[t+1] = L[t] + T[t] + S[t - s + 1]

def dmtest(e1, e2, h=1, lossf='AE'):
    e1 = np.array(e1)
    e2 = np.array(e2)
    T = len(e1)
    # AE: r=1
    if lossf == 'AE':
        d = np.abs(e1) - np.abs(e2)
    # AE: r=2
    else: # lossf == 'SE'
        d = e1**2 - e2**2
    dMean = np.mean(d)
    gamma0 = np.var(d)
    if h > 1:
        raise NotImplementedError()
    else:
        varD = gamma0

    DM = dMean / np.sqrt((1 / T) * varD)
    return DM

# Naive vs HW

naive_hw_DM_r_1 = dmtest(data_361_1082_naive_pred[2*s:], fx[2*s:])
naive_hw_DM_r_2 = dmtest(data_361_1082_naive_pred[2*s+8::24], fx[2*s+8::24], lossf="R2")

print("Metoda naive vs HW dla 8 rano: " + str(naive_hw_DM_r_2))
naive_hw_DM_r_2 = np.array(naive_hw_DM_r_2)
p_naive_hw_DM_r_2 = (naive_hw_DM_r_2, naive_hw_DM_r_2.mean(), naive_hw_DM_r_2.std())
print("Metoda naive vs HW dla 8 rano p value: " + str(p_naive_hw_DM_r_2))
print("Metoda naive vs HW dla 24h: " + str(naive_hw_DM_r_1))

# HW vs Naive

hw_naive_DM_r_1 = dmtest(fx[2*s:], data_361_1082_naive_pred[2*s:])
hw_naive_DM_r_2 = dmtest(fx[2*s+8::24],data_361_1082_naive_pred[2*s+8::24],  lossf="R2")

print("Metoda HW vs naive dla 8 rano: " + str(hw_naive_DM_r_2))
print("Metoda HW vs naive dla 24h: " + str(hw_naive_DM_r_1))
