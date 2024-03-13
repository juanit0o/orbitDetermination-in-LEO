import numpy as np
import pandas as pd
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


def main():
    datautils.data.set_data_dir("Notebooks/data/orekit-data")
    m = 100.0
    # step = 5.0
    step = 60.0
    duration = 14*24*3600.0

    #dataSingleOrbitMinute, 3weeks
    # inclination = 30
    # altitude = 500e3
    # ecc = 0.02

    #dataSingleOrbitMinuteV2
    inclination = 80
    altitude = 1000e3
    ecc = 0.01

    
    data = pd.DataFrame(columns=["time", "x", "y", "z", "xdot", "ydot", "zdot", "xdotdot", "ydotdot", "zdotdot", "density", "drag_coefficient", "drag_area", "ecc", "inc", "alt"])

    kep = orbital_state.KeplerianElements(
        sma=constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS + altitude,
        ecc=ecc,
        inc=inclination*np.pi/180,
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
        data["inc"] = float(inclination)
        data["alt"] = float(altitude)
        data["mass"] = float(m)

    except Exception as e:
        print("Inclination: " + str(inclination) + " Altitude: " + str(altitude) + " Eccentricity: " + str(ecc))
        print(e)

    data['time'] = data.time.astype("int64")

    # #write data to pickle
    # data.to_pickle("dataSingleOrbitMinuteV2_Density.pkl")
    data.to_pickle("dataSingleOrbitMinuteV2Week_Density.pkl")
    print(data.head(10))
    #plot x of dataframe data
    plt.plot(data["time"], data["x"])
    plt.show()

if __name__ == "__main__":
    main()
