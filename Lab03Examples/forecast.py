import numpy as np

def forecast_arx(DATA):
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
    # without intercept
    X = np.vstack([price[6:-1], price[5:-2], price[:-7], price_min[6:-1],
                   loadr[6:-1], Dummies[6:-1, 0], Dummies[6:-1, 1], Dummies[6:-1, 2]]).T
    # with intercept
    # X = np.vstack([np.ones(len(y)), price[6:-1], price[5:-2], price[:-7], price_min[6:-1],
    #                loadr[6:-1], Dummies[6:-1, 0], Dummies[6:-1, 1], Dummies[6:-1, 2]]).T
    # Define explanatory variables for day d+1
    # without intercept
    X_fut = np.hstack([price[-1], price[-2], price[-7], price_min[-1],
                       loadr[-1], Dummies[-1, 0], Dummies[-1, 1], Dummies[-1, 2]])
    # with intercept
    # X_fut = np.hstack([[1], price[-1], price[-2], price[-7], price_min[-1],
    #                    loadr[-1], Dummies[-1, 0], Dummies[-1, 1], Dummies[-1, 2]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]  # Estimate the ARX model
    prog = np.dot(beta, X_fut)                   # Compute a step-ahead forecast
    return np.exp(prog + mc)                     # Convert to price level

def forecast_naive(DATA):
    if np.sum(DATA[-1, 4:7]) > 0:
        return DATA[-8, 2]
    return DATA[-2, 2]

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
