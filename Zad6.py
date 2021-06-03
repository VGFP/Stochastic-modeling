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

def forecast_arx(DATA):
    # DATA: 8-column matrix (date, hour, price, load forecast, Sat, Sun, Mon dummy, p_min)
    # Select data to be used
    # print(DATA[-1, :])
    price = DATA[:-1, 2]             # For day d (d-1, ...)
    price_min = DATA[:-1, 11]         # For day d
    Dummies = DATA[1:, 4:11]          # Dummies for day d+1
    loadr = DATA[1:, 3]              # Load for day d+1

    # Take logarithms
    price = np.log(price)
    mc = np.mean(price)
    price -= mc                      # Remove mean(price)
    price_min = np.log(price_min)
    price_min -= np.mean(price_min)  # Remove mean(price)
    loadr = np.log(loadr)

    # Calibrate the ARX model
    y = price[7:]                    # For day d, d-1, ...
    # Define explanatory variables for calibration
    # without intercept
    X = np.vstack([price[6:-1], price[5:-2], price[:-7], price_min[6:-1],
                   loadr[6:-1], Dummies[6:-1, 0], Dummies[6:-1, 1], Dummies[6:-1, 2], Dummies[6:-1, 3], Dummies[6:-1, 4], Dummies[6:-1, 5], Dummies[6:-1, 6]]).T
    # with intercept
    # X = np.vstack([np.ones(len(y)), price[6:-1], price[5:-2], price[:-7], price_min[6:-1],
    #                loadr[6:-1], Dummies[6:-1, 0], Dummies[6:-1, 1], Dummies[6:-1, 2]]).T
    # Define explanatory variables for day d+1
    # without intercept
    X_fut = np.hstack([price[-1], price[-2], price[-7], price_min[-1],
                    loadr[-1], Dummies[-1, 0], Dummies[-1, 1], Dummies[-1, 2], Dummies[-1, 3], Dummies[-1, 4], Dummies[-1, 5], Dummies[-1, 6]])
    # with intercept
    # X_fut = np.hstack([[1], price[-1], price[-2], price[-7], price_min[-1],
    #                    loadr[-1], Dummies[-1, 0], Dummies[-1, 1], Dummies[-1, 2]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]  # Estimate the ARX model
    prog = np.dot(beta, X_fut)                   # Compute a step-ahead forecast
    return np.exp(prog + mc)                     # Convert to price level

def forecast_narx(DATA):
    import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.callbacks import EarlyStopping
    # DATA: 8-column matrix (date, hour, price, load forecast, Sat, Sun, Mon dummy, p_min)
    # Select data to be used
    # print(DATA[-1, :])
    price = DATA[:-1, 2]             # For day d (d-1, ...)
    price_min = DATA[:-1, 7]         # For day d
    Dummies = DATA[1:, 4:7]          # Dummies for day d+1
    loadr = DATA[1:, 3]              # Load for day d+1

    # Take logarithms
    price = np.log(price)
    mc = np.mean(price)
    price -= mc                      # Remove mean(price)
    price_min = np.log(price_min)
    price_min -= np.mean(price_min)  # Remove mean(price)
    loadr = np.log(loadr)

    # Calibrate the ARX model
    y = price[7:]                    # For day d, d-1, ...
    # Define explanatory variables for calibration
    X = np.vstack([price[6:-1], price[5:-2], price[:-7], price_min[6:-1],
                   loadr[6:-1], Dummies[6:-1, 0], Dummies[6:-1, 1], Dummies[6:-1, 2]]).T
    # Define explanatory variables for day d+1
    X_fut = np.hstack([price[-1], price[-2], price[-7], price_min[-1],
                       loadr[-1], Dummies[-1, 0], Dummies[-1, 1], Dummies[-1, 2]])

    # Define Neural Network model
    inputs = Input(shape=(X.shape[1], ))                  # Input layer
    hidden = Dense(units=20, activation='sigmoid')(inputs)# Hidden layer (20 neurons)
    outputs = Dense(units=1, activation='linear')(hidden) # Output layer
    model = keras.Model(inputs=inputs, outputs=outputs)
    # callbacks = [EarlyStopping(patience=20, restore_best_weights=True)]
    callbacks = []
    model.compile(loss='MAE', optimizer='ADAM')           # Compile model
    model.fit(X, y, batch_size=64, epochs=500, verbose=0, # Fit to data
              validation_split=.0, shuffle=False, callbacks=callbacks)
    prog = model.predict(np.array(X_fut, ndmin=2))        # Compute a step-ahead forecast

    return np.exp(prog + mc)                     # Convert to price level

def forecast_naive(DATA):
    if np.sum(DATA[-1, 4:7]) > 0:
        return DATA[-8, 2]
    return DATA[-2, 2]

import numpy as np
from calendar import weekday
from time import time as t

def task(argtup):
    '''Helper function for multi-core NARX'''
    data, startd, endd, j, hour = argtup
    data_h = data[hour::24, :]
    ts = t()
    task_output = forecast_narx(data_h[startd + j:endd + j + 1, :])
    print(f'{j}\t{hour}\t{t() - ts}')
    return task_output 

def epf_arx(data, Ndays, startd, endd, forecast_type='naive'):
    if forecast_type.lower() == 'narx':      # forecst_narx imports additional libraries, importing
        print(":)")
        # from forecast import forecast_narx   # here ensures that they are not needed for ARX or naive
    elif forecast_type.lower() == 'narx_mc': # multi-core variant
        from multiprocessing import Pool
    # DATA:   4-column matrix (date, hour, price, load forecast)
    # RESULT: 4-column matrix (date, hour, price, forecasted price)
    first_day = str(int(data[0, 0]))
    first_day = (int(e) for e in (first_day[:4], first_day[4:6], first_day[6:]))
    i = weekday(*first_day) # Weekday of starting day: 0 - Monday, ..., 6 - Sunday
    N = len(data) // 24
    data = np.hstack([data, np.zeros((N*24, 9))]) # Append 'data' matrix with daily dummies & p_min
    for j in range(N):
        if i % 7 == 5:
            data[24*j:24*(j+1), 4] = 1 # Saturday dummy in 5th (index 4) column
        elif i % 7 == 6:
            data[24*j:24*(j+1), 5] = 1 # Sunday dummy in 6th column
        elif i % 7 == 0:
            data[24*j:24*(j+1), 6] = 1 # Monday dummy in 7th column
        elif i % 7 == 1:
            data[24*j:24*(j+1), 7] = 1 # wtorek
        elif i % 7 == 2:
            data[24*j:24*(j+1), 8] = 1 # sroda
        elif i % 7 == 3:
            data[24*j:24*(j+1), 9] = 1 # czwartek
        elif i % 7 == 4:
            data[24*j:24*(j+1), 10] = 1 # piatek

        '''
        # Dummies dla pozostałych dni
        '''
        i += 1
        data[24*j:24*(j+1), 11] = np.min(data[24*j:24*(j+1), 2]) # p_min in 8th column
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
                result[j * 24 + hour, 3] = forecast_arx(data_h[startd:endd + j + 1, :])
            elif forecast_type.lower() == 'naive':
                result[j * 24 + hour, 3] = forecast_naive(data_h[startd + j:endd + j + 1, :])

    return result

import matplotlib.pyplot as plt

data = np.loadtxt('GEFCOM.txt')
startd = 0   # First day of the calibration window (startd from Matlab minus 1)
endd = 360   # First day to forecast (equal to endd from Matlab)
Ndays = 722  # user provided number of days to be predicted
             # (max is 722 for GEFCom with endd=360)

# naive, arx, narx or narx_mc
forecast_type = 'arx'

# Estimate and compute forecasts of the ARX model
res = epf_arx(data[:, :4], Ndays, startd, endd, forecast_type)

dane_arx_8_rano = res[8::24,3]
dane_8_rano = res[8::24,2]
# print("Test DM dla 8 rano: " + str(dmtest(dane_8_rano,dane_arx_8_rano)))
# print("Test DM dla wszystkich godzin: " + str(dmtest(res[:,2],res[:,3])))

plt.figure(1)
x_h = np.linspace(1,17328,17328)
plt.plot(x_h, res[:,3], label='ARX Prediction')
plt.plot(x_h, data[24*360:17328+24*360,2], label='Real Data')
plt.xlabel('Godzina')
# plt.xticks(range(1,25,1))
plt.ylabel('Cena strefowa')
plt.title('Wykres sezonowy')
plt.legend()
plt.savefig('List_3_zad_6.png', dpi=300)
plt.show()

# np.savetxt(f'res_{forecast_type.lower()}.txt', res)

# Compute and display MAE
print(f'MAE for days {endd} to {endd+Ndays} across all hours')
print(f'(length of the calibration window for point forecasts = {endd} days)')
print(f'{forecast_type} MAE: {np.mean(np.abs(res[:, 2] - res[:, 3]))}')
print(f'{forecast_type} RMSE: { np.power(np.mean(np.power(res[:, 2] - res[:, 3],2)),0.5)}')

plt.figure(1)
x_h = np.linspace(1,60,60)
# plt.plot(x_h,data_361_1082_naive_pred, label='Naive Prediction')
# plt.plot(x_h,data_361_1082[4800:5301], label='Real Data', alpha=1)
plt.plot(x_h, res[4860:4920,3], label='ARX Prediction')
plt.plot(x_h, data[24*360+4860:24*360+4920,2], label='Real Data')
plt.xlabel('Godzina')
# plt.xticks(range(1,25,1))
plt.ylabel('Cena strefowa')
plt.title('Wykres sezonowy zbliżenie')
plt.legend()
plt.savefig('List_3_zad_6_c_z.png', dpi=300)
plt.show()

# Zadanie 8

data = np.loadtxt('NPdata_2013-2016.txt')
startd = 0   # First day of the calibration window (startd from Matlab minus 1)
endd = 360   # First day to forecast (equal to endd from Matlab)
Ndays = 722  # user provided number of days to be predicted
             # (max is 722 for GEFCom with endd=360)

# naive, arx, narx or narx_mc
forecast_type = 'arx'

# Estimate and compute forecasts of the ARX model
res = epf_arx(data[:, :4], Ndays, startd, endd, forecast_type)

dane_arx_8_rano = res[8::24,3]
dane_8_rano = res[8::24,2]
# print("Test DM dla 8 rano: " + str(dmtest(dane_8_rano,dane_arx_8_rano)))
# print("Test DM dla wszystkich godzin: " + str(dmtest(res[:,2],res[:,3])))

plt.figure(1)
x_h = np.linspace(1,17328,17328)
plt.plot(x_h, res[:,3], label='ARX Prediction')
plt.plot(x_h, data[24*360:17328+24*360,2], label='Real Data')
plt.xlabel('Godzina')
# plt.xticks(range(1,25,1))
plt.ylabel('Cena strefowa')
plt.title('Wykres sezonowy')
plt.legend()
plt.savefig('List_3_zad_8_6.png', dpi=300)
plt.show()

# np.savetxt(f'res_{forecast_type.lower()}.txt', res)

# Compute and display MAE
print(f'MAE for days {endd} to {endd+Ndays} across all hours')
print(f'(length of the calibration window for point forecasts = {endd} days)')
print(f'{forecast_type} MAE: {np.mean(np.abs(res[:, 2] - res[:, 3]))}')
print(f'{forecast_type} RMSE: { np.power(np.mean(np.power(res[:, 2] - res[:, 3],2)),0.5)}')

plt.figure(1)
x_h = np.linspace(1,60,60)
# plt.plot(x_h,data_361_1082_naive_pred, label='Naive Prediction')
# plt.plot(x_h,data_361_1082[4800:5301], label='Real Data', alpha=1)
plt.plot(x_h, res[4860:4920,3], label='ARX Prediction')
plt.plot(x_h, data[24*360+4860:24*360+4920,2], label='Real Data')
plt.xlabel('Godzina')
# plt.xticks(range(1,25,1))
plt.ylabel('Cena strefowa')
plt.title('Wykres sezonowy zbliżenie')
plt.legend()
plt.savefig('List_3_zad_8_6_z.png', dpi=300)
plt.show()
