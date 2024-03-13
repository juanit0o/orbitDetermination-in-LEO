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
    m = 100.0
    step = 60
    duration = 4*3600.0

    inclinations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    altitudes = [200e3, 400e3, 600e3, 800e3, 1000e3, 1200e3]
    eccs = [0.01, 0.02, 0.03]
    # eccs = [0.001, 0.01, 0.01]
    times = []
    trajectories = []
    scalers = []
    times_norm = []
    trajectories_st = []
    trajectories_dot = []
    trajectories_dot_st = []
    scaler_dot = []
    iteration = 0
    full_data = pd.DataFrame(columns=["time", "x", "y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "density", "drag_coefficient", "drag_area", "ecc", "inc", "alt", "orbit_id"])
    for i in inclinations:
        for a in altitudes:
            for ecc in eccs:
                kep = orbital_state.KeplerianElements(
                    sma=constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS + a,
                    ecc=ecc,
                    inc=i*np.pi/180,
                    arg_perigee=0.0,
                    raan=0.0,
                    true_anomaly=0.0,
                    epoch=datetime.datetime(2023, 1, 27, 0, 0, 0, 0),
                    mu=constants.EIGEN5C_EARTH_MU
                )

                #full dynamics    
                #exception that can be thrown due to funky dynamics
                try:
                    data = orbits.full_dynamics_training_data_from_kep(kep, m, step, duration)

                    #add to data matrix the eccentricity, inclination and altitude
                    data["ecc"] = float(ecc)
                    data["inc"] = float(i)
                    data["alt"] = float(a)
                    data["mass"] = float(m)
                    data["orbit_id"] = iteration


                    times.append(data["time"].to_numpy())
                    times_norm.append(data["time"].to_numpy()/data["time"].to_numpy()[-1])
                    
                    trj = data[["x","y","z",
                                "xdot","ydot","zdot",
                                "xdotdot","ydotdot","zdotdot"]].to_numpy()
                    trajectories.append(trj[:,:6].copy())
                    
                    scaler = preprocessing.StandardScaler()
                    scaler.fit(trj[:,:6].copy())
                    scaledData = scaler.transform(trj[:,:6].copy())
                    scalers.append(scaler)
                    trajectories_st.append(scaledData)


                    trajectories_dot.append(trj[:, 3:9].copy())
                    scaler_xdot = preprocessing.StandardScaler()
                    scaler_xdot.fit(trj[:, 3:9].copy())
                    trajectories_dot_st.append(trj[:, 3:9].copy())
                    scaler_dot.append(scaler_xdot)

                    full_data = pd.concat([data, full_data], ignore_index=True)
                    iteration += 1
                except Exception as e:
                    print("Inclination: " + str(i) + " Altitude: " + str(a) + " Eccentricity: " + str(ecc))
                    print(e)

    full_data['time'] = full_data.time.astype("int64")
    full_data['orbit_id']=full_data.orbit_id.astype("str")

    print(full_data.head())
    # #write data to pickle
    full_data.to_pickle("dataMultipleOrbits_Density.pkl")

if __name__ == "__main__":
    main()
