import pysindy as ps
import scipy as sp
import pandas as pd
import utils as utils
from sklearn.preprocessing import StandardScaler
import numpy as np
import plot as plot

def main(): 
    # Load data
    dataUniqueSatellite, allFilesOrbitalUnpickled, realEpochSinceUniqueSatellite = utils.loadData()
    initialData = dataUniqueSatellite.iloc[0].to_list()
    #TEMPO DE 10 EM 10 SEGUNDOS DURANTE 3 HORAS
    time = realEpochSinceUniqueSatellite.copy().tolist()[0] + np.linspace(0, 10800, 573)

    sol1stDerivative = utils.applyOdeint(initialData, time, derivative = "first")
    #divide all collumns by 1000 to get km
    sol1stDerivativeKm = sol1stDerivative.copy()/1000

    scalerData = StandardScaler()
    scaledOdeintData = scalerData.fit_transform(sol1stDerivativeKm)
    gaussianSimulatedNoisyDataNotStandardized, modelGaussianNoisy = sindyNoise(scalerData, sol1stDerivativeKm , time, "gaussian", False)
    plot.plot_noiseVSnoNoise(sol1stDerivativeKm, gaussianSimulatedNoisyDataNotStandardized, time, "gaussian")




def applySindy(data, t_train, scaler):
    polynomialLib = ps.PolynomialLibrary(degree=4)
    #standardizar outra vez (ja com o noise)
    optimizerKms = ps.FROLS()
    scaledOdeintData = scaler.fit_transform(data)
    modelStandardizedNoisy = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizerKms, feature_library=polynomialLib)
    modelStandardizedNoisy.fit(scaledOdeintData, t=t_train, ensemble=True, quiet=True)
    modelStandardizedNoisy.print()
    #de-standardize
    data1stSimulatedScaled = modelStandardizedNoisy.simulate(x0=scaledOdeintData[0], t=t_train, integrator='odeint')
    deStandardizedDataNoisySimulated = scaler.inverse_transform(data1stSimulatedScaled)
    return modelStandardizedNoisy, deStandardizedDataNoisySimulated

def applySindyNotStandardized(data, t_train):
    optimizerKms = ps.FROLS(alpha=1e-3)
    model = ps.SINDy(feature_names=["posX", "posY", "posZ", "velX", "velY", "velZ"], optimizer= optimizerKms)
    model.fit(data, t=t_train, ensemble=True, quiet=True)
    model.print()
    data1stNoisySimulated = model.simulate(x0=data[0], t=t_train, integrator='odeint')
    return model, data1stNoisySimulated
        
def sindyNoise(scaler, dataSindy, t_train, noise = "gaussian", standardizedInput = True):

    if standardizedInput == True:
        # de-standardize to calculate norms
        dataSindy = scaler.inverse_transform(dataSindy)

    normPositions = np.linalg.norm(dataSindy[0][0:3])
    normVelocities = np.linalg.norm(dataSindy[0][3:6])

    #create list of standard deviations for positions and velocities with logarithm scale between 0 and 10 percent of norms
    sigmasPosition = []
    for i in range(-10, 5):
        if 10**i*normPositions >= 0.1*normPositions:
            break
        else:
            sigmasPosition.append(10**i*normPositions)        
    sigmasPosition.insert(0, 0)

    sigmasVelocities = []
    for i in range(-10, 5):
        if 10**i*normVelocities >= 0.1*normVelocities:
            break
        else:
            sigmasVelocities.append(10**i*normVelocities)  
    sigmasVelocities.insert(0, 0)
    

    if noise == "gaussian":
        # #noise nos dois
        # dataSindyPos = dataSindy[:,0:3] + np.random.normal(0, 26, dataSindy[:,0:3].shape)
        # dataSindyVel = dataSindy[:,3:6] + np.random.normal(0, 0.07, dataSindy[:,3:6].shape)
        # dataGaussianNoisyToReturn = np.concatenate((dataSindyPos, dataSindyVel), axis=1) 
        # if standardizedInput == True:
        #     modelToReturn, simulatedDataToReturn = applySindy(dataGaussianNoisyToReturn, t_train, scaler)
        # else:
        #     modelToReturn, simulatedDataToReturn = applySindyNotStandardized(dataGaussianNoisyToReturn, t_train)


        # for sigPos in sigmasPosition:
        #     for sigVel in sigmasVelocities:
        #         dataSindyPos = dataSindy[:,0:3] + np.random.normal(0, sigPos, dataSindy[:,0:3].shape)
        #         dataSindyVel = dataSindy[:,3:6] + np.random.normal(0, sigVel, dataSindy[:,3:6].shape)
        #         dataSindyNoisy = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        #         if standardizedInput == True:
        #             model, simulatedData  = applySindy(dataSindyNoisy, t_train, scaler)
        #         else:
        #             model, simulatedData = applySindyNotStandardized(dataSindyNoisy, t_train)
        #         print("Noise Position: ", sigPos)
        #         print("Noise Velocity: ", sigVel)
        #         print("=========================================")

        #noise só na posição
        # 26km é 0.37% da norma da posição
        dataSindyPos = dataSindy[:,0:3] + np.random.normal(0, 0, dataSindy[:,0:3].shape)
        dataSindyVel = dataSindy[:,3:6]
        dataSindyNoisy = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        if standardizedInput == True:
            modelToReturn, simulatedDataToReturn  = applySindy(dataSindyNoisy, t_train, scaler)
        else:
            modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyNoisy, t_train)
        
        #noise só na velocidade
        #0.06km é 0.79% da norma da velocidade
        # dataSindyPos = dataSindy[:,0:3]
        # dataSindyVel = dataSindy[:,3:6] + np.random.normal(0, 0.06, dataSindy[:,3:6].shape)
        # dataSindyNoisy = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        
        # if standardizedInput == True:
        #     modelToReturn, simulatedDataToReturn  = applySindy(dataSindyNoisy, t_train, scaler)
        # else:
        #     modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyNoisy, t_train)

        return simulatedDataToReturn, modelToReturn


    elif noise == "laplacian":
        dataSindyPos = dataSindy[:,0:3] + np.random.laplace(0, 27, dataSindy[:,0:3].shape)
        dataSindyVel = dataSindy[:,3:6]
        dataSindyLaplacianToReturn = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        if standardizedInput == True:
            modelToReturn, simulatedDataToReturn  = applySindy(dataSindyLaplacianToReturn, t_train, scaler)
        else:
            modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyLaplacianToReturn, t_train)

        # dataSindyPos = dataSindy[:,0:3] 
        # dataSindyVel = dataSindy[:,3:6] + np.random.laplace(0, 0.05, dataSindy[:,3:6].shape)
        # dataSindyLaplacianToReturn = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        # if standardizedInput == True:
        #     modelToReturn, simulatedDataToReturn  = applySindy(dataSindyLaplacianToReturn, t_train, scaler)
        # else:
        #     modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyLaplacianToReturn, t_train)

        # dataSindyPos = dataSindy[:,0:3] + np.random.laplace(0, 29, dataSindy[:,0:3].shape)
        # dataSindyVel = dataSindy[:,3:6] + np.random.laplace(0, 0.04, dataSindy[:,3:6].shape)
        # dataSindyLaplacianToReturn = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        # if standardizedInput == True:
        #     modelToReturn, simulatedDataToReturn  = applySindy(dataSindyLaplacianToReturn, t_train, scaler)
        # else:
        #     modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyLaplacianToReturn, t_train)

        # for sigPos in sigmasPosition:
        #     for sigVel in sigmasVelocities:
        #         dataSindyPos = dataSindy[:,0:3] + np.random.laplace(0, sigPos, dataSindy[:,0:3].shape)
        #         dataSindyVel = dataSindy[:,3:6] + np.random.laplace(0, sigVel, dataSindy[:,3:6].shape)
        #         dataSindyNoisy = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        #         if standardizedInput == True:
        #             model, simulatedData  = applySindy(dataSindyNoisy, t_train, scaler)
        #         else:
        #             model, simulatedData = applySindyNotStandardized(dataSindyNoisy, t_train)
        #         print("Noise Position: ", sigPos)
        #         print("Noise Velocity: ", sigVel)
        #         print("=========================================")

        return simulatedDataToReturn, modelToReturn

    elif noise == "uniform":
        dataSindyUniformToReturn = dataSindy + np.random.uniform(0, 10, dataSindy.shape)
        if standardizedInput == True:
            modelToReturn, simulatedDataToReturn  = applySindy(dataSindyUniformToReturn, t_train, scaler)
        else:
            modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyUniformToReturn, t_train)

        # for i in range(-10, 5):
        #     dataSindy = dataSindy + np.random.uniform(0, 10**i, dataSindy.shape)
        #     if standardizedInput == True:
        #         model, simulatedData  = applySindy(dataSindy, t_train, scaler)
        #     else:
        #         model, simulatedData = applySindyNotStandardized(dataSindy, t_train)
        #     print("Noise: ", 10**i)
        #     print("=========================================")

        return simulatedDataToReturn, modelToReturn

    elif noise == "cauchy":
        #limite posicao 180
        # 0.01% da norma da posicao
        percentageOfNormPosition = int(0.0001 *normPositions)
        dataSindyPos = dataSindy[:,0:3] + sp.stats.cauchy.rvs(loc=0, scale= percentageOfNormPosition, size=dataSindy[:,0:3].shape)
        dataSindyVel = dataSindy[:,3:6]
        dataSindyCauchyToReturn = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        if standardizedInput == True:
            modelToReturn, simulatedDataToReturn  = applySindy(dataSindyCauchyToReturn, t_train, scaler)
        else:
            modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyCauchyToReturn, t_train)

        #0.1% da norma da velocidade
        # percentageOfNormVelocity = int(0.001*normVelocities)
        # dataSindyPos = dataSindy[:,0:3]
        # dataSindyVel = dataSindy[:,3:6] + sp.stats.cauchy.rvs(loc=0, scale= percentageOfNormVelocity, size=dataSindy[:,3:6].shape)
        # dataSindyCauchyToReturn = np.concatenate((dataSindyPos, dataSindyVel), axis=1)
        # if standardizedInput == True:
        #     modelToReturn, simulatedDataToReturn  = applySindy(dataSindyCauchyToReturn, t_train, scaler)
        # else:
        #     modelToReturn, simulatedDataToReturn  = applySindyNotStandardized(dataSindyCauchyToReturn, t_train) 

        return simulatedDataToReturn, modelToReturn
    else:
        print("Invalid noise type")
        return



#standardized
# gaussianSimulatedNoisyData, modelGaussianNoisyStandardized = sindyNoise(scalerData, scaledOdeintData , time, "gaussian", True)
# laplacianSimulatedNoisyData, modelLaplacianNoisyStandardized = sindyNoise(scalerData, scaledOdeintData , time, "laplacian", True)
# cauchySimulatedNoisyData, modelCauchyNoisyStandardized = sindyNoise(scalerData, scaledOdeintData , time, "cauchy", True)
#sem estar standardizado








if __name__ == "__main__":
    main()
