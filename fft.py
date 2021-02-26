import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from utilities import line_plot, bar_plot, scatter_plot

# pumps 1 and 8 did not have values for AMP so they were left out
data = pd.read_csv("raw data.csv").fillna(value=0)
data["Datetime"] = pd.to_datetime(data["Datetime"])

# choose a pump to model
PUMP = "02"  # 02, 03, 04, 05, 06, 07
pump = data[
    [
        "Datetime",
        f"PUMP_{PUMP}_MOTOR_VIBRATION_DRIVE_END",
        f"PUMP_{PUMP}_MOTOR_VIBRATION_NON_DRIVE_END",
        f"PUMP_{PUMP}_PUMP_VIBRATION_DRIVE_END",
        f"PUMP_{PUMP}_PUMP_VIBRATION_NON_DRIVE_END",
    ]
].copy()

# choose a variable to model
VARIABLE = f"PUMP_{PUMP}_PUMP_VIBRATION_DRIVE_END"

# take a slice of the data where there is consistent behavior
# Dec 22, 2020 1:45 - Jan 7, 2021 1:45
pump = pump.loc[
    (pump["Datetime"] >= "2020-12-22 01:45:00")
    & (pump["Datetime"] <= "2021-01-07 01:45:00")
].reset_index(drop=True)

# determine the seasonality in the data
## raw plot
line_plot(pump, x="Datetime", y=VARIABLE, name=f"{VARIABLE} raw plot")

## difference plot
diff = pump.copy()
diff[VARIABLE] = diff[VARIABLE].diff()
line_plot(diff, x="Datetime", y=VARIABLE, name=f"{VARIABLE} difference plot")

## autocorrelation plot
autocor = []
lag = []
for n in range(1, 100):
    autocor.append(pump[VARIABLE].diff().autocorr(lag=n))
    lag.append(n)
autocorrelation = pd.DataFrame({"Autocorrelation": autocor, "Lag": lag})

bar_plot(
    autocorrelation,
    x="Lag",
    y="Autocorrelation",
    title=VARIABLE,
    name=f"{VARIABLE} autocorrelation",
)

## power spectrum plot
power = pd.DataFrame()
power["Power"] = np.abs(np.fft.fft(pump[[VARIABLE]].diff().fillna(value=0))).ravel()
power["Frequency"] = np.fft.fftfreq(pump.shape[0])
power = power.loc[power["Frequency"] >= 0].reset_index(drop=True)

line_plot(
    power, x="Frequency", y="Power", title=VARIABLE, name=f"{VARIABLE} power spectrum"
)

## seasonality
WINDOW = 9  # Frequency at max Power is 0.11 -> 1 / 0.11 = 9.09 samples

# define training and testing data
train = pump.iloc[:WINDOW].copy().reset_index(drop=True)
test = pump.iloc[WINDOW:].copy().reset_index(drop=True)

# compute the FFT power on training data
train_power = np.abs(np.fft.fft(train[[VARIABLE]].diff().fillna(value=0))).ravel()

# compute the distance between training and testing for FFT power
DTW = True  # should dynamic time warping be used to measure the distance? - Euclidean otherwise
distance = []
timestamps = []
values = []
for step in range(0, test.shape[0] - WINDOW, WINDOW):
    test_set = test.iloc[step : (step + WINDOW)].copy().reset_index(drop=True)
    test_power = np.abs(np.fft.fft(test_set[[VARIABLE]].diff().fillna(value=0))).ravel()

    if DTW:
        dist, path = fastdtw(train_power, test_power, dist=euclidean)
    else:
        dist = np.linalg.norm(train_power - test_power, ord=2)

    distance = np.append(distance, [dist for _ in range(WINDOW)])
    timestamps = np.append(timestamps, test_set["Datetime"].tolist())
    values = np.append(values, test_set[VARIABLE].tolist())

# choose a distance threshold to identify an anomaly warning
THRESHOLD = 1

# put the results into a data frame
distance = pd.DataFrame(
    {"Distance": distance, "Datetime": timestamps, VARIABLE: values}
)
distance["Difference"] = distance[VARIABLE].diff().fillna(value=0)
distance["Warning"] = distance["Distance"] >= THRESHOLD

# plot the distance
scatter_plot(
    distance,
    x="Datetime",
    y="Distance",
    color="Warning",
    title=VARIABLE,
    name=f"{VARIABLE} fft distance",
)

scatter_plot(
    distance,
    x="Datetime",
    y=VARIABLE,
    color="Warning",
    title=VARIABLE,
    name=f"{VARIABLE} fft detection",
)

scatter_plot(
    distance,
    x="Datetime",
    y="Difference",
    color="Warning",
    title=VARIABLE,
    name=f"{VARIABLE} fft detection diff",
)
