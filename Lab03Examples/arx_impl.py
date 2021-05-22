import numpy as np
from epf_arx import epf_arx

data = np.loadtxt('GEFCOM.txt')
startd = 0   # First day of the calibration window (startd from Matlab minus 1)
endd = 360   # First day to forecast (equal to endd from Matlab)
Ndays = 722  # user provided number of days to be predicted
             # (max is 722 for GEFCom with endd=360)

# naive, arx, narx or narx_mc
forecast_type = 'arx'

# Estimate and compute forecasts of the ARX model
res = epf_arx(data[:, :4], Ndays, startd, endd, forecast_type)
np.savetxt(f'res_{forecast_type.lower()}.txt', res)

# Compute and display MAE
print(f'MAE for days {endd} to {endd+Ndays} across all hours')
print(f'(length of the calibration window for point forecasts = {endd} days)')
print(f'{forecast_type} MAE: {np.mean(np.abs(res[:, 2] - res[:, 3]))}')
