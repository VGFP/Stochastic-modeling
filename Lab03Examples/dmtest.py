import numpy as np

def dmtest(e1, e2, h=1, lossf='AE'):
    e1 = np.array(e1)
    e2 = np.array(e2)
    T = len(e1)
    if lossf == 'AE':
        d = np.abs(e1) - np.abs(e2)
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
