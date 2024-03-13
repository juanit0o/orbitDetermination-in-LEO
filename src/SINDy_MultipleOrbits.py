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
import numpy as np
import pysindy

from ns_sfd import constants
from ns_sfd import datautils
from ns_sfd import orbital_state

from common import orbits
import traceback


#arguments: (dynamics, nodynamics) (custom, poly) (notstandardized, normalized, standardized) (xdot, noxdot) (losses, noLosses) (TAKES SOME TIME lossesMultipleAlphasStandardized, noLossesMultipleAlphasStandardized)
def main():
    datautils.data.set_data_dir("Notebooks/data/orekit-data")
    if len(sys.argv) > 1:

        m = 100.0
        step = 5.0
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
        infoForPlot = []
        trajectories_dot = []
        trajectories_dot_st = []
        scaler_dot = []
        iteration = 0
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
                        # listOrbitsStatesOdeint, listOrbitsDerivs, listOrbitTimesOdeint = utils.prepare_data_multiple_orbits()
                        data = orbits.data_from_kep(kep, m, step, duration)
                        
                        times.append(data["time"].to_numpy())
                        times_norm.append(data["time"].to_numpy()/data["time"].to_numpy()[-1])
                        
                        trj = data[["x","y","z",
                                    "xdot","ydot","zdot",
                                    "xdotdot","ydotdot","zdotdot"]].to_numpy()
                        trajectories.append(trj[:,:6].copy())
                        
                        scaler = preprocessing.StandardScaler()
                        scaler.fit(trj)
                        scaledData = scaler.transform(trj)
                        scalers.append(scaler)
                        trajectories_st.append(scaledData)


                        
                        trajectories_dot.append(trj[:, 3:9].copy())
                        scaler_xdot = preprocessing.StandardScaler()
                        scaler_xdot.fit(trj[:, 3:9].copy())
                        trajectories_dot_st.append(trj[:, 3:9].copy())
                        scaler_dot.append(scaler_xdot)

                    #dynamics    
                    else:
                        if iteration == 0: 
                            print("DYNAMICS")
                            iteration += 1
                        else:
                            pass

                        #exception that can be thrown due to funky dynamics
                        try:
                            data = orbits.full_dynamics_training_data_from_kep(kep, m, step, duration)
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
                        except Exception as e:
                            print("Inclination: " + str(i) + " Altitude: " + str(a) + " Eccentricity: " + str(ecc))
                            print(e)
                        

                    

                    if sys.argv[1] == "dynamics":
                        infoForPlot.append(tuple([a, i, ecc, "W/ Dynamics", "MO"]))
                    else:
                        infoForPlot.append(tuple([a, i, ecc, "W/O Dynamics", "MO"]))
                        

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
            
            # library_functions1stDeriv = [lambda x: x,
            #                     lambda a, b, c : a/(a**2 + b**2 + c**2)**(3/2),
            #                     lambda a, b, c : b/(a**2 + b**2 + c**2)**(3/2),
            #                     lambda a, b, c : c/(a**2 + b**2 + c**2)**(3/2)]
            # library_function_names1stDeriv = [lambda x: "*" + x, 
            #                                     lambda a, b, c: "*" + a + "/((" + a + "**2 + " + b + "**2 + " + c + "**2)**(3/2))",
            #                                     lambda a, b, c: "*" + b + "/((" + a + "**2 + " + b + "**2 + " + c + "**2)**(3/2))",
            #                                     lambda a, b, c: "*" + c + "/((" + a + "**2 + " + b + "**2 + " + c + "**2)**(3/2))"]
            # pde_lib1stDeriv = ps.WeakPDELibrary(
            #     library_functions=library_functions1stDeriv,
            #     function_names=library_function_names1stDeriv,
            #     spatiotemporal_grid=time,
            #     derivative_order=1,
            #     is_uniform=True,
            #     K=500
            # )
            # # if argument is notstandardized
            # if len(sys.argv) >= 2:

            #     if sys.argv[2] == "nonstandardized":
            #         print("RUNNING CUSTOM LIBRARY, NONSTANARDIZED")

            #         optimizer = ps.FROLS(alpha=1e-3)
            #         model1stNotStandardized = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library=pde_lib1stDeriv)
                    
                    
            #         if sys.argv[3] == "xdot":
            #             print("CUSTOM WEAK LIBRARY ISN'T ABLE TO USE XDOT, IT IS CALCULATED AUTOMATICALLY")
            #             x_dotCalculated = utils.calculateXdot(sol1stDerivativeKm)
            #             model1stNotStandardized.fit(sol1stDerivativeKm, t=time, ensemble=True, quiet=True)
            #             model1stNotStandardized.print()
            #         else:
            #             model1stNotStandardized.fit(sol1stDerivativeKm, t=time, ensemble=True, quiet=True)
            #             model1stNotStandardized.print()

                    
            #         dataSimulated = utils.useSindyEquations(model1stNotStandardized, sol1stDerivativeKm[0], time, "kilometers")


            #     # if argument is normalized
            #     elif sys.argv[2] == "normalized":
            #         print("RUNNING CUSTOM LIBRARY, NORMALIZED")
            #         # # Normalize data
            #         #axis = 0 => normaliza por coluna
            #         normalizedOdeintData = preprocessing.normalize(sol1stDerivativeKm, axis=0)
            #         normalizedOdeintData = pd.DataFrame(normalizedOdeintData, columns=["posX", "posY", "posZ", "velX", "velY", "velZ"])

            #         optimizer = ps.FROLS()
            #         model1stNormalized = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library=pde_lib1stDeriv)
                   

            #         if sys.argv[3] == "xdot":
            #             print("CUSTOM WEAK LIBRARY ISN'T ABLE TO USE XDOT, IT IS CALCULATED AUTOMATICALLY")
            #             normalizedXdotCalculated = preprocessing.normalize(x_dotCalculated, axis=0)
            #             normalizedXdotCalculated = pd.DataFrame(normalizedXdotCalculated, columns=["velX", "velY", "velZ", "accX", "accY", "accZ"])
            #             model1stNormalized.fit(normalizedOdeintData, t=time, ensemble=True, quiet=True)
            #             model1stNormalized.print()
            #         else:
            #             model1stNormalized.fit(normalizedOdeintData, t=time, ensemble=True, quiet=True)
            #             model1stNormalized.print()

                    
            #         dataSimulated = utils.useSindyEquations(model1stNormalized,normalizedOdeintData.iloc[0].to_numpy(), time, "kilometers")

            #     # if argument is standardized
            #     else:
            #         print("RUNNING CUSTOM LIBRARY, STANDARDIZED")  
            #         optimizer = ps.FROLS(verbose=False)
            #         scalerData = StandardScaler()
            #         scaledOdeintData = scalerData.fit_transform(sol1stDerivativeKm)
            #         scalerTime = StandardScaler()
            #         scaledTime = scalerTime.fit_transform(time.reshape(-1,1)).flatten()
            #         model1stStandardized = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizer, feature_library=pde_lib1stDeriv)
                    
            #         if sys.argv[3] == "xdot":
            #             print("CUSTOM WEAK LIBRARY ISN'T ABLE TO USE XDOT, IT IS CALCULATED AUTOMATICALLY")
            #             standardizedXdotScaler1st = StandardScaler()
            #             x_dotCalculated = utils.calculateXdot(sol1stDerivativeKm)
            #             standardizedXdot1st = standardizedXdotScaler1st.fit_transform(x_dotCalculated)
            #             model1stStandardized.fit(scaledOdeintData, t=time, ensemble=True, quiet=True)
            #             model1stStandardized.print()
            #         else:
            #             model1stStandardized.fit(scaledOdeintData, t=time, ensemble=True, quiet=True)
            #             model1stStandardized.print()
                    
            #         dataSimulatedScaled = utils.useSindyEquations(model1stStandardized,scaledOdeintData[0], time, "kilometers")
            #         dataSimulated = scalerData.inverse_transform(dataSimulatedScaled)
            pass



        #POLYNOMIAL LIBRARY
        else:
            polynomialLib = ps.PolynomialLibrary(degree=4)
            if len(sys.argv) >= 3:

                if sys.argv[3] == "nonstandardized":
                    
                    if sys.argv[1] == "dynamics":
                        optimizer = ps.FROLS(alpha=1e-39)
                    else:
                        optimizer = ps.FROLS(alpha=1e-39)
                    
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer=optimizer, feature_library=polynomialLib)
                    if sys.argv[4] == "xdot":
                        print("RUNNING POLYNOMIAL, NONSTANDARDIZED, XDOT")
                        model.fit(trajectories, t=times, multiple_trajectories=True, quiet = True, x_dot = trajectories_dot)
                        model.print()
                    else:
                        print("RUNNING POLYNOMIAL, NONSTANDARDIZED")
                        model.fit(trajectories, t=times, multiple_trajectories=True, quiet = True)
                        model.print()
                    #simulate for one orbit
                    # dataSimulated = model1stMultipleNotStandardized.simulate(listOrbitsStatesOdeint[2][0], t=listOrbitTimesOdeint[2], integrator="odeint")


                # if argument is normalized
                elif sys.argv[3] == "normalized":
                    pass

                # if argument is standardized
                else:
                    

                    #listOrbitsDerivsStandardized, listOrbitsStatesOdeintStandardized, listScalersDerivs, listScalersData, listOrbitsTimesOdeintStandardized, listScalersTime = utils.standardizeInputs(listOrbitsDerivs, listOrbitsStatesOdeint, listOrbitTimesOdeint)
                    if sys.argv[1] == "dynamics":
                        #funciona +-
                        optimizer = ps.FROLS(alpha = 1e4)
                    else:
                        #funciona +-
                        #-10
                        optimizer = ps.FROLS(alpha = 1e-10)
                    
                    # optimizerTestStandardized = ps.FROLS()
                    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer=optimizer, feature_library = polynomialLib)
                    if sys.argv[4] == "xdot":
                        print("RUNNING POLYNOMIAL, STANDARDIZED, XDOT")
                        model.fit(trajectories_st, t=times_norm, multiple_trajectories=True, quiet = True, x_dot = trajectories_dot_st)
                        model.print()
                    else:
                        print("RUNNING POLYNOMIAL, STANDARDIZED, NOXDOT")
                        model.fit(trajectories_st, t=times_norm, multiple_trajectories=True, quiet = True)
                        print("====================================")
                        model.print()
                        print("====================================")



                    # # #TENTATIVA QUE DÁ ERRO
                    # # # simulatedTest = model.simulate(trajectories_st[0][0], t=times_norm[0], integrator="odeint")
                    # # #É DITO QUE ESTE SCALER NAO FOI FITTED AINDA
                    # # # dataSimulated = scalers[0].inverse_transform(simulatedTest)

                    # # TENTATIVA DE RESOLVER PROB
                    # sindy_stateStandardized = model.simulate(trajectories_st[2][0], t=listOrbitTimesOdeint[2], integrator="odeint")
                    # dataSimulated = listScalersData[2].inverse_transform(sindy_stateStandardized)
                    

        
        #Uncomment to simulate for every orbit
        # if sys.argv[3] == "standardized":
        #     for n, scaler in enumerate(scalers):
        #         time = times_norm[n]
        #         test_data = trajectories_st[n]
        #         test_physical_data = trajectories[n]
                
        #         try:
        #             model_data = model.simulate(test_data[0, :], t=time, integrator="odeint")
        #             model_physical_data = scaler.inverse_transform(model_data)
        #             plot.plot_single_orbit_run(times[n], test_physical_data, model_physical_data, infoForPlot[n])

        #         except Exception as e:
        #             print(e)
        #             print("Error in simulate()/inverse_transform()")
        #             continue
        # else:
        #     for n, scaler in enumerate(scalers):
        #         test_physical_data = trajectories[n]
        #         try:
        #             model_physical_data = model.simulate(trajectories[n][0, :], t=times[n], integrator="odeint")
        #             plot.plot_single_orbit_run(times[n], test_physical_data, model_physical_data, infoForPlot[n])
        #         except Exception as e:
        #             print(e)
        #             print("Error in simulate()/inverse_transform()")
        #             continue

        
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
            model_physical_data = scalerToSimulate.inverse_transform(model_data)
        else:
            model_physical_data = model.simulate(trajectoryToSimulate[0, :], t=timeToSimulate, integrator="odeint")
        
        if sys.argv[1] == "dynamics":
            infoForPlotSimulate = tuple([altToSimulate, incToSimulate, eccToSimulate, "W/ Dinamics", "MO"])
        else:
            infoForPlotSimulate = tuple([altToSimulate, incToSimulate, eccToSimulate, "W/O Dinamics", "MO"])
        plot.plot_single_orbit_run(timeToSimulate, trajectoryToSimulate, model_physical_data, infoForPlotSimulate)

        ##############################################

        if sys.argv[5] == "losses":
            plot.plotSingleLosses(optimizer)
        else:
            pass

        if sys.argv[6] == "lossesMultipleAlphasStandardized":
            alphas = np.logspace(20, -30, 25)
            matrixLossesLoopMultipleFirst = []
            
            for i in range(len(alphas)):

                # listOrbitsDerivsStandardized, listOrbitsStatesOdeintStandardized, listScalersDerivs, listScalersData, listOrbitsTimesOdeintStandardized, listScalersTime = utils.standardizeInputs(listOrbitsDerivs, listOrbitsStatesOdeint, listOrbitTimesOdeint)
                optimizer = ps.FROLS(alpha=alphas[i])
                model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer=optimizer, feature_library = ps.PolynomialLibrary(degree=4))

                if sys.argv[3] == "standardized":
                    if sys.argv[4] == "xdot":
                        model.fit(trajectories_st, t=times_norm, multiple_trajectories=True, quiet = True, x_dot = trajectories_dot_st)
                    else:
                        model.fit(trajectories_st, t=times_norm, multiple_trajectories=True, quiet = True)
                else:
                    if sys.argv[4] == "xdot":
                        model.fit(trajectories, t=times, multiple_trajectories=True, quiet = True, x_dot = trajectories_dot)
                    else:
                        model.fit(trajectories, t=times, multiple_trajectories=True, quiet = True)

                # model1stMultipleStandardized.print()
                matrixLossesLoopMultipleFirst.append(optimizer.get_loss_matrix())
            
            plot.plotMultipleLosses(matrixLossesLoopMultipleFirst, alphas, "Multiple Orbits")
        else:
            pass
    else:
        print("Check README file for instructions on the arguments to use")




if __name__ == "__main__":
    main()
