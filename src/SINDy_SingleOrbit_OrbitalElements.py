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
import typing

from ns_sfd import constants
from ns_sfd import datautils
from ns_sfd import orbital_state

from common import orbits
import traceback
import datetime


datautils.data.set_data_dir("data/orekit-data")


def main():

    if len(sys.argv) > 1:

        m = 100.0
        step = 5.0
        duration = 4*3600.0
        date0 = datetime.datetime(2023, 1, 27, 0, 0, 0, 0)
        inc = 60
        alt = 400e3
        ecc = 0.01

        kep = orbital_state.KeplerianElements(
                    sma=constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS + alt,
                    ecc=ecc,
                    inc=inc*np.pi/180,
                    arg_perigee=0.0,
                    raan=0.0,
                    true_anomaly=0.0,
                    epoch=datetime.datetime(2023, 1, 27, 0, 0, 0, 0),
                    mu=constants.EIGEN5C_EARTH_MU
        )

        if sys.argv[1] == "nodynamics":
            data = orbits.training_data_from_kep_with_orbital_elements(
                                            kep, m, step, duration)
                                            
            infoForPlot = tuple([alt, inc, ecc, "W/O Dynamics", "SO_OO"])
        else:
            data = orbits.full_dynamics_training_data_from_kep_with_orbital_elements(
                                            kep, m, step, duration)

            infoForPlot = tuple([alt, inc, ecc, "W/ Dynamics", "SO_OO"])

    
        time = data["t"]
        training_data = data["kep"]

        optimizer = ps.FROLS(alpha=1e-25, normalize_columns=True)

        model = ps.SINDy(
            feature_names=[
                "sma", "ecc", "inc", "arg_perigee", "raan", "true_anomaly"],
            optimizer=optimizer
        )
        model.fit(training_data, t=time)
        model.print()

        #####################################
        altToSimulate = 500e3
        eccToSimulate = 0.015
        incToSimulate = 65
        kepToSimulate = orbital_state.KeplerianElements(
                        sma=constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS + altToSimulate,
                        ecc=eccToSimulate,
                        inc= incToSimulate *np.pi/180,
                        arg_perigee=0.0,
                        raan=0.0,
                        true_anomaly=0.0,
                        epoch=datetime.datetime(2023, 1, 27, 0, 0, 0, 0),
                        mu=constants.EIGEN5C_EARTH_MU
                    )
        if sys.argv[1] == "dynamics":
            dataToSimulate = orbits.full_dynamics_training_data_from_kep_with_orbital_elements(kepToSimulate, m, step, duration)
        else:
            dataToSimulate = orbits.training_data_from_kep_with_orbital_elements(kepToSimulate, m, step, duration)
        
        timeToSimulate = dataToSimulate["t"]
        training_dataToSimulate = dataToSimulate["kep"]

        model_data = model.simulate(training_dataToSimulate[0], t=timeToSimulate, integrator="odeint")
        ######################################

        plot.plot_orbital_elements(timeToSimulate, training_dataToSimulate, model_data, infoForPlot)
        
        dates = add_deltas_to_initial_date(date0, timeToSimulate)
        plot_states = orbits.osculating_elements_to_cartesian(
            model_data, dates, m)

        test_data = dataToSimulate["cartesian"]
        plot.plot_single_orbit_run(timeToSimulate, test_data, plot_states, infoForPlot)


def add_deltas_to_initial_date(
    date0: datetime.datetime, deltas: typing.List[float]
) -> typing.List[datetime.datetime]:
    return [date0 + datetime.timedelta(seconds=d) for d in deltas]

    


if __name__ == "__main__":
    main()
