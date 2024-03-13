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

def main():
    datautils.data.set_data_dir("Notebooks/data/orekit-data")
    if len(sys.argv) > 1:

        m = 100.0
        step = 5.0
        duration = 4*3600.0

        date0 = datetime.datetime(2023, 1, 27, 0, 0, 0, 0)

        inclinations = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        altitudes = [200e3, 400e3, 600e3, 800e3, 1000e3, 1200e3]
        eccs = [0.01, 0.02, 0.03]
        # eccs = [0.001, 0.01, 0.01]
        times = []
        trajectories_orbital_elements = []
        trajectories_cartesian = []
        infoForPlot = []
        iteration = 0
        dataList = []
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

                    
                    if sys.argv[1] == "nodynamics":
                        if iteration == 0: 
                            print("NO DYNAMICS")
                            iteration += 1
                        else:
                            pass
                        data = orbits.training_data_from_kep_with_orbital_elements(kep, m, step, duration)
                        dataList.append(data)
                        
                        times.append(data["t"])
                        trajectories_cartesian.append(data["cartesian"])
                        trajectories_orbital_elements.append(data["kep"])


                    #dynamics    
                    else:
                        if iteration == 0: 
                            print("DYNAMICS")
                            iteration += 1
                        else:
                            pass

                        #exception that can be thrown due to funky dynamics
                        try:
                            data = orbits.full_dynamics_training_data_from_kep_with_orbital_elements(kep, m, step, duration)
                            dataList.append(data)

                            times.append(data["t"])
                            trajectories_cartesian.append(data["cartesian"])
                            trajectories_orbital_elements.append(data["kep"])
                            
                        except Exception as e:
                            print("Inclination: " + str(i) + " Altitude: " + str(a) + " Eccentricity: " + str(ecc))
                            print(e)
                        

                    if sys.argv[1] == "dynamics":
                        infoForPlot.append(tuple([a, i, ecc, "W/ Dynamics", "MO_OO"]))
                    else:
                        infoForPlot.append(tuple([a, i, ecc, "W/O Dynamics", "MO_OO"]))
                        

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        #, normalize_columns=True
        optimizer = ps.FROLS(alpha=1e-3, normalize_columns=True)
        model = ps.SINDy(
            feature_names=[
                "sma", "ecc", "inc", "arg_perigee", "raan", "true_anomaly"],
            optimizer=optimizer
        )

        model.fit(trajectories_orbital_elements, t=times, multiple_trajectories=True)
        model.print()

        # for n, _ in enumerate(inclinations):
        #     time = dataList[n]["t"]
        #     kep_data = dataList[n]["kep"]
            
        #     model_data = model.simulate(
        #         kep_data[0], t=time, integrator="odeint")
            
        #     time = dataList[n]["t"]
        #     plot.plot_orbital_elements(time, kep_data, model_data, infoForPlot[n])
            
        #     dates = add_deltas_to_initial_date(date0, time)
        #     plot_states = orbits.osculating_elements_to_cartesian(
        #         model_data, dates, m)

        #     test_data = dataList[n]["cartesian"]
        #     plot.plot_single_orbit_run(time, test_data, plot_states, infoForPlot[n])

        
        ##############################################
        #simular com uma condição inicial de uma nova orbita (nao usada para treinar o modelo)
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
        
        if sys.argv[1] == "dynamics":
            infoForPlotSimulate = tuple([altToSimulate, incToSimulate, eccToSimulate, "W/ Dinamics", "MO"])
        else:
            infoForPlotSimulate = tuple([altToSimulate, incToSimulate, eccToSimulate, "W/O Dinamics", "MO"])
        
        plot.plot_orbital_elements(timeToSimulate, training_dataToSimulate, model_data, infoForPlotSimulate)
        
        dates = add_deltas_to_initial_date(date0, timeToSimulate)
        plot_states = orbits.osculating_elements_to_cartesian(
            model_data, dates, m)

        test_data = dataToSimulate["cartesian"]
        plot.plot_single_orbit_run(timeToSimulate, test_data, plot_states, infoForPlotSimulate)


def add_deltas_to_initial_date(
    date0: datetime.datetime, deltas: typing.List[float]
) -> typing.List[datetime.datetime]:
    return [date0 + datetime.timedelta(seconds=d) for d in deltas]


if __name__ == "__main__":
    main()

