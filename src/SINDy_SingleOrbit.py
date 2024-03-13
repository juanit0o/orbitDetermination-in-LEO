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

from ns_sfd import constants
from ns_sfd import datautils
from ns_sfd import orbital_state

from common import orbits
import traceback
import datetime


#arguments: (nodynamics, dynamics) (custom, poly) (notstandardized, normalized, standardized) (xdot, noxdot) (losses, noLosses) (TAKES SOME TIME lossesMultipleAlphasStandardized, noLossesMultipleAlphasStandardized)
def main():

    if len(sys.argv) > 1:

        m = 100.0
        step = 5.0
        duration = 4*3600.0

        inc = 65
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
            print("NO DYNAMICS")
            # listOrbitsStatesOdeint, listOrbitsDerivs, listOrbitTimesOdeint = utils.prepare_data_multiple_orbits()
            data = orbits.data_from_kep(kep, m, step, duration)
        else:
            print("DYNAMICS")
            data = orbits.full_dynamics_training_data_from_kep(kep, m, step, duration)


        time = data["time"].to_numpy()
        time_norm = time/time[-1]

        trj = data[["x","y","z",
                    "xdot","ydot","zdot",
                    "xdotdot","ydotdot","zdotdot"]].to_numpy()

        trajectory = trj[:,:6].copy()
        scaler = preprocessing.StandardScaler()
        scaler.fit(trajectory)
        trajectory_st = scaler.transform(trajectory)

        trajectory_dot = trj[:, 3:9].copy()
        scaler_xdot = preprocessing.StandardScaler()
        scaler_xdot.fit(trajectory_dot)
        trajectory_dot_st = scaler_xdot.transform(trajectory_dot)
    

        if sys.argv[1] == "dynamics":
            infoForPlot = tuple([alt, inc, ecc, "W/ Dynamics", "SO"])
        else:
            infoForPlot = tuple([alt, inc, ecc, "W/O Dynamics", "SO"])

        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # # Seed the random number generators for reproducibility
        # np.random.seed(100)

        # integration keywords for solve_ivp, typically needed for chaotic systems
        integrator_keywords = {}
        integrator_keywords['rtol'] = 1e-12
        integrator_keywords['method'] = 'LSODA'
        integrator_keywords['atol'] = 1e-12

        #CUSTOM LIBRARY
        if sys.argv[2] == "custom":
            library_functions1stDeriv = [lambda x: x,
                                lambda a, b, c : a/(a**2 + b**2 + c**2)**(3/2),
                                lambda a, b, c : b/(a**2 + b**2 + c**2)**(3/2),
                                lambda a, b, c : c/(a**2 + b**2 + c**2)**(3/2)]
            library_function_names1stDeriv = [lambda x: "*" + x, 
                                                lambda a, b, c: "*" + a + "/((" + a + "**2 + " + b + "**2 + " + c + "**2)**(3/2))",
                                                lambda a, b, c: "*" + b + "/((" + a + "**2 + " + b + "**2 + " + c + "**2)**(3/2))",
                                                lambda a, b, c: "*" + c + "/((" + a + "**2 + " + b + "**2 + " + c + "**2)**(3/2))"]
            pde_lib1stDeriv = ps.WeakPDELibrary(
                library_functions=library_functions1stDeriv,
                function_names=library_function_names1stDeriv,
                spatiotemporal_grid=time,
                derivative_order=1,
                is_uniform=True,
                K=500
            )
            # if argument is notstandardized
            if len(sys.argv) >= 3:

                if sys.argv[3] == "nonstandardized":
                    print("RUNNING CUSTOM LIBRARY, NONSTANARDIZED")

                    optimizer = ps.FROLS(alpha=1e-3)
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library=pde_lib1stDeriv)
                    
                    
                    if sys.argv[4] == "xdot":
                        print("CUSTOM WEAK LIBRARY ISN'T ABLE TO USE XDOT, IT IS CALCULATED AUTOMATICALLY")
                        x_dotCalculated = utils.calculateXdot(trajectory)
                        model.fit(trajectory, t=time, ensemble=True, quiet=True)
                        model.print()
                    else:
                        model.fit(trajectory, t=time, ensemble=True, quiet=True)
                        model.print()

                    
                    # dataSimulated = utils.useSindyEquations(model, trajectory[0], time, "kilometers")


                # if argument is normalized
                elif sys.argv[3] == "normalized":
                    print("RUNNING CUSTOM LIBRARY, NORMALIZED")
                    # # Normalize data
                    #axis = 0 => normaliza por coluna
                    normalizedOdeintData = preprocessing.normalize(trajectory, axis=0)
                    normalizedOdeintData = pd.DataFrame(normalizedOdeintData, columns=["posX", "posY", "posZ", "velX", "velY", "velZ"])

                    optimizer = ps.FROLS()
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library=pde_lib1stDeriv)
                   

                    if sys.argv[4] == "xdot":
                        print("CUSTOM WEAK LIBRARY ISN'T ABLE TO USE XDOT, IT IS CALCULATED AUTOMATICALLY")
                        normalizedXdotCalculated = preprocessing.normalize(x_dotCalculated, axis=0)
                        normalizedXdotCalculated = pd.DataFrame(normalizedXdotCalculated, columns=["velX", "velY", "velZ", "accX", "accY", "accZ"])
                        model.fit(normalizedOdeintData, t=time, ensemble=True, quiet=True)
                        model.print()
                    else:
                        model.fit(normalizedOdeintData, t=time_norm, ensemble=True, quiet=True)
                        model.print()

                    
                    # dataSimulated = utils.useSindyEquations(model,normalizedOdeintData.iloc[0].to_numpy(), time_norm, "kilometers")

                # if argument is standardized
                else:
                    print("RUNNING CUSTOM LIBRARY, STANDARDIZED")  
                    optimizer = ps.FROLS(verbose=False)
                    scalerTime = StandardScaler()
                    scaledTime = scalerTime.fit_transform(time.reshape(-1,1)).flatten()
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library=pde_lib1stDeriv)
                    
                    if sys.argv[4] == "xdot":
                        print("CUSTOM WEAK LIBRARY ISN'T ABLE TO USE XDOT, IT IS CALCULATED AUTOMATICALLY")
                        standardizedXdotScaler1st = StandardScaler()
                        x_dotCalculated = utils.calculateXdot(trajectory)
                        standardizedXdot1st = standardizedXdotScaler1st.fit_transform(x_dotCalculated)
                        model.fit(trajectory_st, t=time_norm, ensemble=True, quiet=True)
                        model.print()
                    else:
                        model.fit(trajectory_st, t=time_norm, ensemble=True, quiet=True)
                        model.print()
                    
                    # dataSimulatedScaled = utils.useSindyEquations(model,trajectory_st[0], time, "kilometers")
                    # dataSimulated = scalerData.inverse_transform(dataSimulatedScaled)



        #POLYNOMIAL LIBRARY
        else:
            polynomialLib = ps.PolynomialLibrary(degree=4)
            if len(sys.argv) >= 2:

                if sys.argv[3] == "nonstandardized":
                    
                    if sys.argv[1] == "dynamics":
                        optimizer = ps.FROLS(alpha=1e-15)
                    else:
                        optimizer = ps.FROLS(alpha=1e-3)
                    
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library= polynomialLib)
                    if sys.argv[4] == "xdot":
                        print("RUNNING POLYNOMIAL LIBRARY, NONSTANARDIZED, XDOT")
                        model.fit(trajectory, t=time, ensemble=True, quiet=True, x_dot = trajectory_dot)
                        model.print()
                    else:
                        print("RUNNING POLYNOMIAL LIBRARY, NONSTANARDIZED, NOXDOT")
                        model.fit(trajectory, t=time, ensemble=True, quiet=True)
                        model.print()

                    
                    # dataSimulated = model1stNotStandardized.simulate(x0=data[0], t=time, integrator='odeint')


                # if argument is normalized
                elif sys.argv[3] == "normalized":
                    
                    # # Normalize data
                    #axis = 0 => normaliza por coluna
                    normalizedOdeintData = preprocessing.normalize(trajectory, axis=0)
                    normalizedOdeintData = pd.DataFrame(normalizedOdeintData, columns=["posX", "posY", "posZ", "velX", "velY", "velZ"])

                    optimizer = ps.FROLS()
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library= polynomialLib)
                    

                    if sys.argv[4] == "xdot":
                        print("RUNNING POLYNOMIAL LIBRARY, NORMALIZED, XDOT")
                        normalizedXdotCalculated = preprocessing.normalize(trajectory_dot, axis=0)
                        normalizedXdotCalculated = pd.DataFrame(normalizedXdotCalculated, columns=["velX", "velY", "velZ", "accX", "accY", "accZ"])
                        model.fit(normalizedOdeintData, t=time, ensemble=True, quiet=True, x_dot = normalizedXdotCalculated)
                        model.print()
                    else:
                        print("RUNNING POLYNOMIAL LIBRARY, NORMALIZED, NOXDOT")
                        model.fit(normalizedOdeintData, t=time, ensemble=True, quiet=True)
                        model.print()

                    
                    # dataSimulated = model.simulate(x0=normalizedOdeintData.iloc[0].to_numpy(), t=time, integrator='odeint')

                # if argument is standardized
                else:
                    if sys.argv[1] == "dynamics":
                        #funciona +-
                        optimizer = ps.FROLS(alpha = 1e3)
                    else:
                        #funciona +-
                        optimizer = ps.FROLS(alpha = 1e-9)

                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library = polynomialLib)
                    
                    if sys.argv[4] == "xdot":
                        print("RUNNING POLYNOMIAL LIBRARY, STANARDIZED, XDOT")
                        model.fit(trajectory_st, t=time_norm, ensemble=True, quiet=True, x_dot = trajectory_dot_st)
                        model.print()
                    else:
                        print("RUNNING POLYNOMIAL LIBRARY, STANARDIZED, NOXDOT")
                        model.fit(trajectory_st, t=time_norm, ensemble=True, quiet=True)
                        model.print()
                    
                    # data1stSimulatedScaled = model.simulate(x0 = scaledOdeintData[0], t=time, integrator='odeint')
                    # dataSimulated = scalerData.inverse_transform(data1stSimulatedScaled)
        
        
        ########################################################################################
        #simular com uma condição inicial de uma nova orbita (nao usada para treinar o modelo)
        altToSimulate = 500e3
        eccToSimulate = 0.03
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
            dataToSimulate = orbits.full_dynamics_training_data_from_kep(kepToSimulate, m, step, duration)
        else:
            dataToSimulate = orbits.data_from_kep(kepToSimulate, m, step, duration)
        
        timeToSimulate = dataToSimulate["time"].to_numpy()
        time_normToSimulate = timeToSimulate/timeToSimulate[-1]

        trjToSimulate = dataToSimulate[["x","y","z",
                    "xdot","ydot","zdot",
                    "xdotdot","ydotdot","zdotdot"]].to_numpy()

        
        trajectoryToSimulate = trjToSimulate[:,:6].copy()
        scalerToSimulate = preprocessing.StandardScaler()
        scalerToSimulate.fit(trajectoryToSimulate)
        scaledDataToSimulate = scalerToSimulate.transform(trajectoryToSimulate)
        
        if sys.argv[3] == "standardized":
            model_data = model.simulate(scaledDataToSimulate[0, :], t=time_normToSimulate, integrator="odeint")  
            #if sys.argv[4] == "xdot":
                #model_physical_data = scalerToSimulate.inverse_transform(np.column_stack([model_data, trajectory_dot_st[:,3:]]))
            #else:
                #model_physical_data = scalerToSimulate.inverse_transform(model_data)
            model_physical_data = scalerToSimulate.inverse_transform(model_data)
            #model_physical_data = scalerToSimulate.inverse_transform(model_data)
        else:
            model_physical_data = model.simulate(trajectoryToSimulate[0, :], t=timeToSimulate, integrator="odeint")  

        if sys.argv[1] == "dynamics":
            infoForPlotSimulate = tuple([altToSimulate, incToSimulate, eccToSimulate, "W/ Dynamics", "SO"])
        else:
            infoForPlotSimulate = tuple([altToSimulate, incToSimulate, eccToSimulate, "W/O Dynamics", "SO"])
        plot.plot_single_orbit_run(timeToSimulate, trajectoryToSimulate, model_physical_data, infoForPlotSimulate)
        
        
        ########################################################################################
        # if sys.argv[3] == "standardized":
        #     model_data = model.simulate(trajectory_st[0, :], t=time_norm, integrator="odeint")
        #     model_physical_data = scaler.inverse_transform(model_data)
        #     plot.plot_single_orbit_run(time, trajectory, model_physical_data, infoForPlot)
        # else:
        #     model_physical_data = model.simulate(trajectory[0, :], t=time, integrator="odeint")
        #     plot.plot_single_orbit_run(time, trajectory, model_physical_data, infoForPlot)

        


        if sys.argv[5] == "losses":
            plot.plotSingleLosses(optimizer)
        else:
            pass

        if sys.argv[6] == "lossesMultipleAlphasStandardized":
            # standardizedXdotScaler1st = StandardScaler()
            # x_dotCalculated = utils.calculateXdot(sol1stDerivativeKm)
            # standardizedXdot1st = standardizedXdotScaler1st.fit_transform(np.array(x_dotCalculated))

            #create a list with alphas ranging from 1e-1 to 1e-40
            alphas = np.logspace(10, -20, 25)
            matrixLossesLoop = []
            
            for i in range(len(alphas)):
                optimizerKms = ps.FROLS(alpha=alphas[i])
                model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizerKms, feature_library=polynomialLib)

                if sys.argv[3] == "standardized":
                    if sys.argv[4] == "xdot":
                        model.fit(trajectory_st, t=time_norm, x_dot = trajectory_dot_st, ensemble=True, quiet=True)
                    else:
                        model.fit(trajectory_st, t=time_norm, ensemble=True, quiet=True)
                #nonstandardized
                else:
                    if sys.argv[4] == "xdot":
                        model.fit(trajectory, t=time, x_dot = trajectory_dot, ensemble=True, quiet=True)
                    else:
                        model.fit(trajectory, t=time, ensemble=True, quiet=True)

                matrixLossesLoop.append(optimizerKms.get_loss_matrix())
            
            plot.plotMultipleLosses(matrixLossesLoop, alphas, "Single Orbit (" + infoForPlot[-2] + ")")
        else:
            pass
    else:
        print("Check README file for instructions on the arguments to use")




if __name__ == "__main__":
    main()
