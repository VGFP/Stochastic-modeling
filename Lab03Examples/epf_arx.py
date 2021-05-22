import numpy as np
from calendar import weekday
from forecast import forecast_arx, forecast_naive
from time import time as t

def task(argtup):
    '''Helper function for multi-core NARX'''
    from forecast import forecast_narx
    data, startd, endd, j, hour = argtup
    data_h = data[hour::24, :]
    ts = t()
    task_output = forecast_narx(data_h[startd + j:endd + j + 1, :])
    print(f'{j}\t{hour}\t{t() - ts}')
    return task_output 

def epf_arx(data, Ndays, startd, endd, forecast_type='naive'):
    if forecast_type.lower() == 'narx':      # forecst_narx imports additional libraries, importing
        from forecast import forecast_narx   # here ensures that they are not needed for ARX or naive
    elif forecast_type.lower() == 'narx_mc': # multi-core variant
        from multiprocessing import Pool
    # DATA:   4-column matrix (date, hour, price, load forecast)
    # RESULT: 4-column matrix (date, hour, price, forecasted price)
    first_day = str(int(data[0, 0]))
    first_day = (int(e) for e in (first_day[:4], first_day[4:6], first_day[6:]))
    i = weekday(*first_day) # Weekday of starting day: 0 - Monday, ..., 6 - Sunday
    N = len(data) // 24
    data = np.hstack([data, np.zeros((N*24, 4))]) # Append 'data' matrix with daily dummies & p_min
    for j in range(N):
        if i % 7 == 5:
            data[24*j:24*(j+1), 4] = 1 # Saturday dummy in 5th (index 4) column
        elif i % 7 == 6:
            data[24*j:24*(j+1), 5] = 1 # Sunday dummy in 6th column
        elif i % 7 == 0:
            data[24*j:24*(j+1), 6] = 1 # Monday dummy in 7th column
        i += 1
        data[24*j:24*(j+1), 7] = np.min(data[24*j:24*(j+1), 2]) # p_min in 8th column
    result = np.zeros((Ndays * 24, 4)) # Initialize `result` matrix
    result[:, :3] = data[endd*24:(endd + Ndays) * 24, :3]
    if forecast_type.lower() == 'narx_mc': # multi-core invocation of NARX model
        argtups = [(data, startd, endd, j, h) for j in range(Ndays) for h in range(24)]
        with Pool() as pool:   # Pool(N) uses N simultaneous processes
            res = pool.map(task, argtups)
        result[:, 3] = res
        return result
    for j in range(Ndays):     # For all days ...
        for hour in range(24): # ... compute 1-day ahead forecasts for each hour
            data_h = data[hour::24, :]
            # Compute forecasts for the hour
            if forecast_type.lower() == 'narx':
                ts = t()
                result[j * 24 + hour, 3] = forecast_narx(data_h[startd + j:endd + j + 1, :])
                print(f'{j}\t{hour}\t{t() - ts}')
            elif forecast_type.lower() == 'arx':
                result[j * 24 + hour, 3] = forecast_arx(data_h[startd + j:endd + j + 1, :])
            elif forecast_type.lower() == 'naive':
                result[j * 24 + hour, 3] = forecast_naive(data_h[startd + j:endd + j + 1, :])

    return result
