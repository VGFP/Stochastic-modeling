import numpy as np

def holtwinters(param, s, x, return_fx=False):
    # Recover alpha, beta and gamma from PARAM
    alpha, beta, gamma = param
    # Initialize L, T, S and FX vectors
    L = np.zeros_like(x)
    T = L.copy()
    S = L.copy()
    fx = L.copy()
    # Set initial values of L, T and S
    L[s-1] = np.sum(x[:s]) / s
    T[s-1] = np.sum(x[s:2*s] - x[:s]) / (s ** 2)
    S[:s] = x[:s] - L[s-1]

    # Iterate to compute L(t), T(t), S(t) and FX(t)
    for t in range(s, len(x)-1):
        L[t] = alpha * (x[t] - S[t-s]) + (1 - alpha) * (L[t-1] + T[t-1])
        T[t] = beta * (L[t] - L[t-1]) + (1 - beta) * T[t-1]
        S[t] = gamma * (x[t] - L[t]) + (1 - gamma) * S[t-s]
        fx[t+1] = L[t] + T[t] + S[t - s + 1]

    # Compute MAE + a penalty for parameters beyond the admitted range, i.e.,
    # 0 < alpha, beta, gamma < 1. The latter is required for parameter estimation.
    maxx = np.max(x)
    MAE = np.mean(np.abs(x[2*s:] - fx[2*s:])) + maxx * (
            (alpha <= 0 or alpha >= 1) + (beta <= 0 or beta >= 1) + (gamma <= 0 or beta >= 1))
    if return_fx:
       return fx
    return MAE
