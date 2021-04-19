import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def SIR_ode(t, y, params):
    beta, gamma, N = params
    S0, I0 = y
    S1 = -beta * I0 * S0 / N
    I1 = beta * I0 * S0 / N - gamma * I0
    return [S1, I1]

# parametry
beta = .5
gamma = .1
N = 1000
T = 100
dt = 1 # krok czasowy na wykresie - nie ma wpływu na dokładność rozwiązania  
steps = int(T / dt)

T_ode = np.zeros(steps)
Y = np.zeros((steps, 2))
Y[0, :] = [N - 1, 1] # [S_0, I_0]
for T in range(steps-1):
    ode_system = RK45(lambda t, y: SIR_ode(t, y, [beta, gamma, N]), T_ode[T], Y[T, :], T_ode[T]+dt)
    while ode_system.status == 'running':
        ode_system.step()
    Y[T + 1, :] = ode_system.y
    T_ode[T + 1] = ode_system.t

S = Y[:, 0]
I = Y[:, 1]
R = N - S - I

plt.plot(T_ode, S, label=r'$S_t^{DOPRI}$')
plt.plot(T_ode, I, label=r'$I_t^{DOPRI}$')
plt.plot(T_ode, R, label=r'$R_t^{DOPRI}$')
plt.xlabel('Czas [np. dni]')
plt.ylabel('Liczba przypadków')
plt.title(f'Ewolucja SIR dla N={N}, $I_0$={I[0]}, ' + '$\\beta = ' + f'{beta}$, ' + '$\\gamma = ' + f'{gamma}$')
plt.legend()
# plt.savefig('SIR.png', dpi=300)
plt.show()