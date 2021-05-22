import numpy as np
from scipy.optimize import minimize
from holtwinters import holtwinters
from christof import christof

d = np.loadtxt('GEFCOM.txt')

# Select one hour for analysis ...
hour = 8
p = d[hour::24, 2]
# ... or take the daily average(uncomment)
# pd = d[:, 2].reshape((len(d) // 24, 24)).mean(-1)

# Period(weekly, i.e., 7 day, for daily data)
s = 7

# Last day of the calibration period
T = 360

# Estimate Holt-Winters parameters
initial_param = np.array([.5, .5, .5])
param = minimize(holtwinters, initial_param, args=(s, p[:T])).x
print('Holt-Winters')
print(f'[alpha, beta, gamma] = {param.tolist()}')
# param = [.8804, .0265, .864] # Params from Matlab's fminsearch
pf = holtwinters(param, s, p, return_fx=True)
print(len(p[T:]))
print(f'MAE: {np.mean(np.abs(p[T:] - pf[T:]))}')
print(f'RMSE: {np.sqrt(np.mean((p[T:] - pf[T:])**2))}')

np.savetxt('holtwinters.txt', pf)

# Holt-Winters method - empirical PIs
PI50 = np.zeros((len(p), 2)) * np.nan
PI90 = PI50.copy()
cover50 = np.zeros_like(p) * np.nan
cover90 = cover50.copy()

for j in range(T, len(p)):
    er = p[j - T + 7:j] - pf[j - T + 7:j]
    PI50[j, :] = pf[j] + np.quantile(er, [.25, .75])
    cover50[j] = p[j] > PI50[j, 0] and p[j] < PI50[j, 1]
    PI90[j, :] = pf[j] + np.quantile(er, [.05, .95])
    cover90[j] = p[j] > PI90[j, 0] and p[j] < PI90[j, 1]

# Crop the NaNs from the beginning
PI50 = PI50[T:, :]
cover50 = cover50[T:]
PI90 = PI90[T:, :]
cover90 = cover90[T:]

print('[PI (%), #hits, coverage (%)]:')
print(f'{[50, np.sum(cover50[T:]), 100 * np.sum(cover50[T:])/(len(p) - T)]}')
print(f'{[90, np.sum(cover90[T:]), 100 * np.sum(cover90[T:])/(len(p) - T)]}')

# Tests for coverage of the 50% intervals
LR_uc, LR_i, LR_cc, LR_uc_p, LR_i_p, LR_cc_p = christof(cover50, .5)
print(f'p-values for Chistoffersen\'s test - 50% PI:')
print('\t\tUC\t\t\t\t\tInd\t\t\t\t\tCC')
print(LR_uc_p, LR_i_p, LR_cc_p)
