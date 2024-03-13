import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
import utils as utils
import pysindy as ps
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from pysindy.utils.odes import lorenz
from scipy.integrate import odeint
import sys
from sklearn import preprocessing
import plot as plot
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
import datetime

from ns_sfd import constants
from ns_sfd import datautils
from ns_sfd import orbital_state

from common import orbits
import traceback

import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    datautils.data.set_data_dir("Notebooks/data/orekit-data")
    # m = 100.0
    # step = 5.0
    # duration = 4*3600.0

    # inclinations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    # altitudes = [200e3, 400e3, 600e3, 800e3, 1000e3, 1200e3]
    # eccs = [0.01, 0.02, 0.03]
    # # eccs = [0.001, 0.01, 0.01]
    # times = []
    # trajectories = []
    # scalers = []
    # times_norm = []
    # trajectories_st = []
    # trajectories_dot = []
    # trajectories_dot_st = []
    # scaler_dot = []
    # iteration = 0
    # full_data = pd.DataFrame(columns=["time", "x", "y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "ecc", "inc", "alt", "orbit_id"])
    # for i in inclinations:
    #     for a in altitudes:
    #         for ecc in eccs:
    #             kep = orbital_state.KeplerianElements(
    #                 sma=constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS + a,
    #                 ecc=ecc,
    #                 inc=i*np.pi/180,
    #                 arg_perigee=0.0,
    #                 raan=0.0,
    #                 true_anomaly=0.0,
    #                 epoch=datetime.datetime(2023, 1, 27, 0, 0, 0, 0),
    #                 mu=constants.EIGEN5C_EARTH_MU
    #             )

    #             #full dynamics    
    #             #exception that can be thrown due to funky dynamics
    #             try:
    #                 data = orbits.full_dynamics_training_data_from_kep(kep, m, step, duration)
    #                 times.append(data["time"].to_numpy())
    #                 times_norm.append(data["time"].to_numpy()/data["time"].to_numpy()[-1])
                    
    #                 trj = data[["x","y","z",
    #                             "xdot","ydot","zdot",
    #                             "xdotdot","ydotdot","zdotdot"]].to_numpy()
    #                 trajectories.append(trj[:,:6].copy())
                    
    #                 scaler = preprocessing.StandardScaler()
    #                 scaler.fit(trj[:,:6].copy())
    #                 scaledData = scaler.transform(trj[:,:6].copy())
    #                 scalers.append(scaler)
    #                 trajectories_st.append(scaledData)


    #                 trajectories_dot.append(trj[:, 3:9].copy())
    #                 scaler_xdot = preprocessing.StandardScaler()
    #                 scaler_xdot.fit(trj[:, 3:9].copy())
    #                 trajectories_dot_st.append(trj[:, 3:9].copy())
    #                 scaler_dot.append(scaler_xdot)
    #                 #add to data matrix the eccentricity, inclination and altitude
    #                 data["ecc"] = float(ecc)
    #                 data["inc"] = float(i)
    #                 data["alt"] = float(a)
    #                 #add orbit id as an integer
    #                 data["orbit_id"] = iteration
    #                 full_data = pd.concat([data, full_data], ignore_index=True)
    #                 iteration += 1
    #             except Exception as e:
    #                 print("Inclination: " + str(i) + " Altitude: " + str(a) + " Eccentricity: " + str(ecc))
    #                 print(e)

    # full_data['time'] = full_data.time.astype("int64")
    # full_data['orbit_id']=full_data.orbit_id.astype("str")

    # # #write data to pickle
    # full_data.to_pickle("dataMultipleOrbits.pkl")

    ##############################################################
    ##############################################################
    ##############################################################

    #read data from pickle
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
    full_dataUnpickled[["x", "y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "alt"]] = full_dataUnpickled[["x", "y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "alt"]]/1000
    full_dataUnpickled = full_dataUnpickled.drop(columns=["y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "alt", "inc",  "ecc"], axis =1)
    print(full_dataUnpickled.head())
    # plt.plot(full_dataUnpickled["orbit_id"], full_dataUnpickled["time_idx"], label="x")
    # plt.show()
    
    #predict for the next 4 hours
    max_prediction_length = 20
    max_encoder_length = 10
    training_cutoff = full_dataUnpickled["time_idx"].max() - max_prediction_length
    
    # #list of orbit ids
    # orbit_ids = full_dataUnpickled["orbit_id"].unique()
    # print(orbit_ids)
    # for i in orbit_ids:
    #     plt.plot(full_dataUnpickled[full_dataUnpickled["orbit_id"] == i]["time"], full_dataUnpickled[full_dataUnpickled["orbit_id"] == i]["x"], label="x")
    #     plt.title("Orbit " + str(i))
    #     plt.show()
    # plt.scatter(full_dataUnpickled["time"], full_dataUnpickled["x"])
    # plt.show()
    #check if full_dataUnpickled has nan or inf values
    # print(full_dataUnpickled.isin([np.nan, np.inf, -np.inf]).sum())

    trainingData = TimeSeriesDataSet(
        full_dataUnpickled[lambda x: x.time_idx <= training_cutoff],
        target="x",
        # target = "x",
        time_idx="time_idx",
        group_ids=["orbit_id"],
        #defines the lookback period
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length= max_encoder_length,
        #specifies how many datapoints will be predicted
        min_prediction_length=1,
        max_prediction_length= max_prediction_length,
        time_varying_known_reals=["time_idx", "time"],
        time_varying_unknown_reals= ["x"],
        static_categoricals=["orbit_id"],
        # target_normalizer = GroupNormalizer(groups=["orbit_id"], transformation="softplus"),
        # target_normalizer=MultiNormalizer([EncoderNormalizer(), TorchNormalizer()]),
        categorical_encoders={
        'orbit_id':NaNLabelEncoder(add_nan=True)
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    

    validation = TimeSeriesDataSet.from_dataset(trainingData, full_dataUnpickled, predict=True, stop_randomization=True)
    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = trainingData.to_dataloader(train=True, batch_size=batch_size, num_workers=5)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=5)
    #get the data from the dataloader
    x, y = next(iter(train_dataloader))
    print(np.shape(x))
    print(np.shape(y))

    # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    # baseline_predictions = Baseline().predict(val_dataloader)
    # (actuals - baseline_predictions).abs().mean().item()

    ###################################
    ###################################
    # configure network and trainer
    # pl.seed_everything(42)
    trainer = pl.Trainer(
        gpus=0,
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )

    print(trainingData)
    tft = TemporalFusionTransformer.from_dataset(
        trainingData,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        hidden_size=16,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        # output_size=7,  # 7 quantiles by default
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=4,
        optimizer = "adam"
    )

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    res = trainer.tuner.lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logsReal")  # logging results to a tensorboard

    
    #define callback path for saving model
    checkpoint_callback = ModelCheckpoint(dirpath = 'callbacksTFTReal/', filename = 'best-checkpoint',
                        save_top_k = 1, verbose = True, monitor = 'val_loss', mode = 'min')
    trainer = pl.Trainer(
        max_epochs=3,
        # max_epochs=30,
        gpus=0,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
    )


    tft = TemporalFusionTransformer.from_dataset(
        trainingData,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    import pickle
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_testrEAL",
        # n_trials=200,
        n_trials=2,
        # max_epochs=50,
        max_epochs=2,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("tft_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)

    
    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("=================================")
    print(best_model_path)
    print("=================================")
    best_model_path = "C:/Users/jpfun/Desktop/Gitlab Neuraspace/joao-funenga/src/callbacksTFT/.lr_find_47ce0246-bfd5-452b-a471-a948a994964d.ckpt"
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calcualte mean absolute error on validation set
    
    # print("Len val_dataloader: ", len(val_dataloader))
    # print("Shape val_dataloader: ", np.shape(val_dataloader))
    predictions = best_tft.predict(val_dataloader)
    print("Shape predictions: ", np.shape(predictions))


    # calcualte mean absolute error on validation set
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader)
    (actuals - predictions).abs().mean()

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
    print(np.shape(raw_predictions))
    for idx in range(2):  # plot 2 examples
        print(f"Plotting example {idx}")
        best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        figu = best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        # figu.savefig(f"fig{idx}.png")
        # plt.show()

        # figu = best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
        figu.savefig(f"fig{idx}.png")
        plt.show()



if __name__ == "__main__":
    main()
