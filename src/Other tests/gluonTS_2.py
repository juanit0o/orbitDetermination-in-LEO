import pandas as pd
import numpy as np
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
# from gluonts.dataset.split import split
from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation import Evaluator, backtest


full_dataUnpickled = pd.read_pickle("dataMultipleOrbits.pkl")
full_dataUnpickled = full_dataUnpickled.drop(columns=["xdotdot", "ydotdot", "zdotdot", "ecc",  "inc", "alt"], axis =1)
#select data from one orbit id
full_dataUnpickled = full_dataUnpickled[full_dataUnpickled['orbit_id'] == "1"]
full_dataUnpickled[["x", "y", "z", "xdot", "ydot", "zdot"]] = full_dataUnpickled[["x", "y", "z", "xdot", "ydot", "zdot"]]/1000

full_dataUnpickled['timeAux'] = full_dataUnpickled.groupby('orbit_id')['time'].transform(lambda x: x - x.min())
full_dataUnpickled['timeAux'] = full_dataUnpickled['timeAux'] + 1577836800
full_dataUnpickled['datetime'] = pd.to_datetime(full_dataUnpickled['timeAux'], unit='s')
full_dataUnpickled = full_dataUnpickled.drop(columns=["orbit_id", "timeAux", "time"], axis =1)
full_dataUnpickled = full_dataUnpickled.set_index('datetime')
print(full_dataUnpickled.head())
print("N rows:", full_dataUnpickled.shape[0])
print("N cols:", full_dataUnpickled.shape[1])
print("Column types:", full_dataUnpickled.dtypes.unique())

metadata = {
    'prediction_length': 1000,
    'freq': '1min'
}

train_data = [{"start": full_dataUnpickled.index[0], "target": full_dataUnpickled[i].values[:-metadata['prediction_length']]} for i in full_dataUnpickled.columns]
test_data = [{"start": full_dataUnpickled.index[0], "target": full_dataUnpickled[i].values} for i in full_dataUnpickled.columns]

train_ds = ListDataset(
    data_iter=train_data,
    freq=metadata['freq']
)

test_ds = ListDataset(
    data_iter=test_data,
    freq=metadata['freq']
)

train_entry = next(iter(train_ds))
test_entry = next(iter(test_ds))

test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.suptitle("Note the testing data contains the training data plus inference period", fontsize=14)
plt.show()

estimator = DeepAREstimator(
    prediction_length=metadata['prediction_length'],
    context_length=2*metadata['prediction_length'],
    freq=metadata['freq'],
    trainer=Trainer(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-3,
        hybridize=False,
        num_batches_per_epoch=100
    )
)

predictor = estimator.train(train_ds)

forecast_it, ts_it = backtest.make_evaluation_predictions(
    dataset=test_ds,  
    predictor=predictor, 
    num_samples=1000,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)

print("N forecasted time series:", len(forecasts))

def plot_prob_forecasts(ts, forecast, title):
    plot_length = 100
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts[-plot_length:].plot(ax=ax)  # plot the time series
    forecast.plot(prediction_intervals=prediction_intervals, color='g')
    plt.legend(legend, loc="upper left")
    fig.suptitle("Time series:" + title)
    plt.show()

ts_sample = list(range(0,19))

[plot_prob_forecasts(tss[i], forecasts[i], full_dataUnpickled.columns[i]) for i in ts_sample]