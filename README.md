# DEVELOPED BY:LISIANA T
# REG NO:212222240053
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
# Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the WeatherAUS dataset
data = pd.read_csv('/content/weatherAUS.csv')

# Extract and clean the Rainfall data
X = data['Rainfall'].dropna().reset_index(drop=True)  # Drop NaNs and reset index

# Set figure size and plot the original Rainfall data
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X)
plt.title('Original Rainfall Data')
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X) / 4), ax=plt.gca())
plt.title('Rainfall Data - ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=30, method='ywm', ax=plt.gca())  # Reduced lags for faster calculation
plt.title('Rainfall Data - PACF')
plt.tight_layout()
plt.show()

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Use a smaller subset of the data (optional)
X_subset = X[:1000]  # First 1000 data points for faster fitting

# Fit ARMA(1,1) model
arma11_model = ARIMA(X_subset, order=(1, 0, 1)).fit()

# Extract parameters
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Print results
print(f"\nARMA(1,1) Parameters (Subset of Data):\nAR (phi1): {phi1_arma11}\nMA (theta1): {theta1_arma11}")

# Simulate ARMA(1,1) Process
N = 1000
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process (Rainfall)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim([0, 500])
plt.grid(True)
plt.show()

# ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.title('ACF of Simulated ARMA(1,1) - Rainfall')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of Simulated ARMA(1,1) - Rainfall')
plt.show()

# -----------------------------
# Fit ARMA(2,2) model
# -----------------------------
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

print(f"\nARMA(2,2) Parameters:\nAR (phi1, phi2): {phi1_arma22}, {phi2_arma22}")
print(f"MA (theta1, theta2): {theta1_arma22}, {theta2_arma22}")

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process (Rainfall)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim([0, 500])
plt.grid(True)
plt.show()

# ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.title('ACF of Simulated ARMA(2,2) - Rainfall')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of Simulated ARMA(2,2) - Rainfall')
plt.show()

```
### OUTPUT:
Original Rainfall Data:
![image](https://github.com/user-attachments/assets/ccbf9d0c-1d1e-4eed-beb8-f4b409b63891)

Autocorrelation:
![image](https://github.com/user-attachments/assets/320ca725-541d-4abe-afb9-eae41dfc7b2e)

Partial Autocorrelation:
![image](https://github.com/user-attachments/assets/bf85af3a-8fb1-4685-bc9f-e37bc96a1237)



SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/cbe57f99-5204-4de7-bbbc-5e2a66385046)


Partial Autocorrelation:
![image](https://github.com/user-attachments/assets/c40ddc68-d9d0-4c3d-9730-4719d18dc2e4)

Autocorrelation

![image](https://github.com/user-attachments/assets/543048a0-0991-4204-aeca-34f22dafed2c)



SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/617f4ad6-d6b3-4938-a53a-99ff49e8e389)


Partial Autocorrelation

![image](https://github.com/user-attachments/assets/c2a3c9e8-bdc9-4cf8-9284-33b673e41498)


Autocorrelation
![image](https://github.com/user-attachments/assets/b16e5e6d-ed12-43c7-8558-730e1bfef19c)


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
