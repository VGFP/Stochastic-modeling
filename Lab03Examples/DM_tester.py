import numpy as np
from scipy.stats import norm
from dmtest import dmtest

T = 360
hour = 8
real = np.loadtxt('GEFCOM.txt', usecols=2)[hour::24]

pf_naive = np.loadtxt('res_naive.txt')[hour::24, 3]
pf_HW = np.loadtxt('holtwinters.txt')[T:]

err_naive = real[T:] - pf_naive
err_HW = real[T:] - pf_HW

print(f'MAE naive: {np.mean(np.abs(err_naive))}')
print(f'MAE HW: {np.mean(np.abs(err_HW))}')

# Is HW better than naive?
DM = dmtest(err_naive, err_HW, 1, 'AE')
DM1_pval = 1 - norm.cdf(DM)
DM = dmtest(err_naive, err_HW, 1, 'SE')
DM2_pval = 1 - norm.cdf(DM)

print(f'Diebold-Mariano test for hour {hour}, HW significantly better if p-value <0.05')
print('\t\tAE\t\t\t\t\tSE')
print(DM1_pval, DM2_pval)

# Is naive better than HW?
DM = dmtest(err_HW, err_naive, 1, 'AE')
DM1_pval = 1 - norm.cdf(DM)
DM = dmtest(err_HW, err_naive, 1, 'SE')
DM2_pval = 1 - norm.cdf(DM)

print(f'Diebold-Mariano test for hour {hour}, naive significantly better if p-value <0.05')
print('\t\tAE\t\t\t\t\tSE')
print(DM1_pval, DM2_pval)
