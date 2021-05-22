import numpy as np
from scipy.stats import chi2

def christof(I, p):
    n1 = np.sum(I[1:])
    n0 = len(I) - 1 - n1
    pihat = n1 / (n0 + n1)

    LR_uc = -2 * (n1 * np.log(p) + n0 * np.log(1-p)) + 2 * (n1 * np.log(pihat) + n0 * np.log(1 - pihat))
    LR_uc_p = 1 - chi2.cdf(LR_uc, 1)

    # Independence
    value_from = I[:-1]
    value_to = I[1:]
    transitions = np.stack([value_from, value_to])
    n01 = np.sum([1 if value_from[i] == 0 and value_to[i] == 1 else 0 for i in range(len(value_to))])
    n11 = np.sum([1 if value_from[i] == 1 and value_to[i] == 1 else 0 for i in range(len(value_to))])
    n10 = np.sum([1 if value_from[i] == 1 and value_to[i] == 0 else 0 for i in range(len(value_to))])
    n00 = np.sum([1 if value_from[i] == 0 and value_to[i] == 0 else 0 for i in range(len(value_to))])
    pihat01 = n01 / (n00 + n01)
    pihat11 = n11 / (n10 + n11)
    pihat2 = (n01 + n11) / (n00 + n01 + n10 + n11)

    LR_i = -2 * ((n00 + n10)*np.log(1 - pihat2) + (n01 + n11) * np.log(pihat2)) + \
           2 * (n00 * np.log(1 - pihat01) + n01 * np.log(pihat01) + n10 * np.log(1 - pihat11) + n11 * np.log(pihat11))
    LR_i_p = 1 - chi2.cdf(LR_i, 1)

    # Conditional coverage
    LR_cc = LR_uc + LR_i
    LR_cc_p = 1 - chi2.cdf(LR_cc, 2)

    return LR_uc, LR_i, LR_cc, LR_uc_p, LR_i_p, LR_cc_p
