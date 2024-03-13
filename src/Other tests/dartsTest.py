import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.datasets import ElectricityDataset
from darts.models import VARIMA, RNNModel
from darts.metrics import mae
from darts.dataprocessing.transformers import Scaler
import numpy as np
import pandas as pd
from darts.metrics import mape


full_dataUnpickled = pd.read_pickle("dataMultipleOrbits.pkl")
# full_dataUnpickled['time_idx'] = range(1, len(full_dataUnpickled) + 1)
full_dataUnpickled.sort_values(
["orbit_id", "time"],
ascending=[True, True],
inplace=True
)
#remove orbit with orbit ids 120 and 136 (wrong orbit generated???)
full_dataUnpickled = full_dataUnpickled[full_dataUnpickled["orbit_id"] != "120"]
full_dataUnpickled = full_dataUnpickled[full_dataUnpickled["orbit_id"] != "136"]

full_dataUnpickled["time_idx"] = full_dataUnpickled.groupby("orbit_id").cumcount()
#divide x,y,z,vx,vy,vz by 1000 to get km and km/s
full_dataUnpickled[["x", "y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "alt"]] = full_dataUnpickled[["x", "y", "z", "xdot", "ydot",
                                                                                                                    "zdot", "xdotdot", "ydotdot", "zdotdot", "alt"]]/1000
#drop all columns except time_idx, x, y, z, xdot, ydot, zdot
full_dataUnpickled = full_dataUnpickled[["orbit_id", "time_idx", "x", "y", "z", "xdot", "ydot", "zdot"]]
#filter for only 1 orbit
full_dataUnpickled = full_dataUnpickled[full_dataUnpickled["orbit_id"] == "1"]
full_dataUnpickled = full_dataUnpickled[["time_idx", "x", "y", "z", "xdot", "ydot", "zdot"]]
#convert time_idx to datetime skipping 5 seconds
full_dataUnpickled["time_idx"] = pd.to_datetime(full_dataUnpickled["time_idx"], unit="s", origin=pd.Timestamp("2020-01-01"))


#
print(full_dataUnpickled.head())
series = TimeSeries.from_dataframe(full_dataUnpickled, "time_idx")
# multi_serie_elec = ElectricityDataset().load()
# print(multi_serie_elec)

# retaining only three components in different ranges
retained_components = ["x", "y", "z"]
multi_serie_elec = series[retained_components]
# resampling the multivariate time serie
multi_serie_elec = multi_serie_elec.resample(freq="5S")

print(np.shape(multi_serie_elec))
multi_serie_elec.plot()
plt.show()

# split in train/validation sets
training_set, validation_set = multi_serie_elec[:-200], multi_serie_elec[-200:]

# define a scaler, by default, normalize each component between 0 and 1
scaler_dataset = Scaler()
# scaler is fit on training set only to avoid leakage
training_scaled = scaler_dataset.fit_transform(training_set)
validation_scaled = scaler_dataset.transform(validation_set)


def fit_and_pred(model, training, validation):
    model.fit(training)
    forecast = model.predict(len(validation))
    return forecast

model_VARIMA = VARIMA(p=20, d=0, q=0, trend="n")

model_GRU = RNNModel(
    input_chunk_length=24,
    model="LSTM",
    hidden_dim=25,
    n_rnn_layers=3,
    training_length=36,
    n_epochs=50,
)

# training and prediction with the VARIMA model
forecast_VARIMA = fit_and_pred(model_VARIMA, training_scaled, validation_scaled)
print("MAE (VARIMA) = {:.2f}".format(mae(validation_scaled, forecast_VARIMA)))

# training and prediction with the RNN model
forecast_RNN = fit_and_pred(model_GRU, training_scaled, validation_scaled)
print("MAE (RNN) = {:.2f}".format(mae(validation_scaled, forecast_RNN)))

forecast_VARIMA = scaler_dataset.inverse_transform(forecast_VARIMA)
forecast_RNN = scaler_dataset.inverse_transform(forecast_RNN)

labels = [f"forecast {component}" for component in retained_components]
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
validation_set.plot(ax=axs[0])

forecast_VARIMA.plot(label=labels, ax=axs[0])
axs[0].set_ylim(-4000, 8000)
axs[0].set_title("VARIMA model forecast")
axs[0].legend(loc="upper left")
validation_set.plot(ax=axs[1])

forecast_RNN.plot(label=labels, ax=axs[1])
axs[1].set_ylim(-4000, 8000)
axs[1].set_title("RNN model forecast")
axs[1].legend(loc="upper left")
plt.show()

pred = model_GRU.predict(n=400, series=multi_serie_elec)

training_scaled.plot(label="actual")
pred.plot(label="forecast")
plt.legend()
plt.show()
# print("MAPE = {:.2f}%".format(mape(training_scaled, pred)))