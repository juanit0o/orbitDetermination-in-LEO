import pandas as pd
from scipy.integrate import odeint
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def loadData():
    allFilesOrbitalUnpickled = pd.read_pickle("C:/Users/jpfun/Desktop/Gitlab Neuraspace/joao-funenga/data/allFilesOrbital.pkl")
    allIds = allFilesOrbitalUnpickled["satID"].unique()
    #para um satelite em especifico [index 1], comparar os valores dos parametros orbitais (entre os varios dias)
    dataUniqueSatellite = allFilesOrbitalUnpickled[allFilesOrbitalUnpickled["satID"] == allIds[2]][["posX", "posY", "posZ", "velX", "velY", "velZ"]]  
    realEpochSinceUniqueSatellite = allFilesOrbitalUnpickled[allFilesOrbitalUnpickled["satID"] == allIds[2]]["epochSince"]
    return dataUniqueSatellite, allFilesOrbitalUnpickled, realEpochSinceUniqueSatellite

def prepare_data_multiple_orbits():
    allFilesOrbitalUnpickled = pd.read_pickle("./data/allFilesOrbital.pkl")
    #how many different sattelite ids
    allIds = allFilesOrbitalUnpickled["satID"].unique()
    allFilesOrbitalUnpickled = allFilesOrbitalUnpickled.sort_values(by=['satID', 'epochSince'])[["satID", "epochSince", "posX", "posY", "posZ", "velX", "velY", "velZ"]]
    print(allIds[2])
    #separate dataframe into a list of dataframes, one for each satellite
    listOrbitsAllInfo = [allFilesOrbitalUnpickled[allFilesOrbitalUnpickled["satID"] == id] for id in allIds]

    #one list only with positionsX, positionsY, positionsZ, velocitiesX, velocitiesY, velocitiesZ
    listOrbitsStates = [orbit[["epochSince", "posX", "posY", "posZ", "velX", "velY", "velZ"]].values for orbit in listOrbitsAllInfo]


    listOrbitsStatesOdeint = []
    listOrbitTimesOdeint = []
    for orbit in listOrbitsStates:
        #posXYZ velXYZ da primeira orbita
        initialData = orbit[0][1:]
        #comecar no timestamp da primeira e gerar dados para 3 horas
        time = orbit[0][0] + np.linspace(0, 10800, 573)
        sol1stDerivativeKm = applyOdeint(initialData, time, derivative = "first")/1000

        listOrbitsStatesOdeint.append(sol1stDerivativeKm)
        listOrbitTimesOdeint.append(time)

    #drop epochSince column from listOrbitsStates
    # listOrbitsStates = [orbit[:,1:] for orbit in listOrbitsStates]

    def functionToIntegrate(valuesList):
        posX = valuesList[0]
        posY = valuesList[1]
        posZ = valuesList[2]
        velX = valuesList[3]
        velY = valuesList[4]
        velZ = valuesList[5]
        standardGravitationalParameter = 3.986004418 * 10**(14)

        return [velX, 
                velY,
                velZ, 
                -standardGravitationalParameter * posX/(posX**2 + posY**2 + posZ**2)**(3/2), 
                -standardGravitationalParameter * posY/(posX**2 + posY**2 + posZ**2)**(3/2), 
                -standardGravitationalParameter * posZ/(posX**2 + posY**2 + posZ**2)**(3/2)]

    #calculate list with the derivatives of each orbit
    listOrbitsDerivs = []
    for orbit in listOrbitsStatesOdeint:
        listDerivsSingleOrbit = []
        for state in orbit:
            listDerivsSingleOrbit.append(np.array(functionToIntegrate(state)))
            # np.append(listOrbitsDerivs,np.array(functionToIntegrate(state, 0)))
        listOrbitsDerivs.append(np.array(listDerivsSingleOrbit))
    return listOrbitsStatesOdeint, listOrbitsDerivs, listOrbitTimesOdeint

def applyOdeint(initialState, time, order="meter", derivative = "first"):
    def functionToIntegrate(valuesList, time):
        posX = valuesList[0]
        posY = valuesList[1]
        posZ = valuesList[2]
        velX = valuesList[3]
        velY = valuesList[4]
        velZ = valuesList[5]
        standardGravitationalParameter = 3.986004418 * 10**(14)
        
        return [velX, 
                velY,
                velZ, 
                -standardGravitationalParameter * posX/(posX**2 + posY**2 + posZ**2)**(3/2), 
                -standardGravitationalParameter * posY/(posX**2 + posY**2 + posZ**2)**(3/2), 
                -standardGravitationalParameter * posZ/(posX**2 + posY**2 + posZ**2)**(3/2)]


    # intialState = dataSindy.iloc[0].to_list()
    # time = t_train.copy().flatten().tolist()
    #TEMPO DE 10 EM 10 SEGUNDOS DURANTE 3 HORAS

    #o que sai do odeint sao os state vectors ao longo do tempo (posicoes e velocidades)
    #rows = timesteps
    #columns = variables
    return odeint(functionToIntegrate, y0=initialState, t = time, tfirst=False)

def applyOdeint2ndDerivative(model, initialState, time, order="meter"):
    
    def getEquations(model):
        equationsModel = model.print()
        newEquations = []
        for equation in equationsModel:
            newEquations.append(equation.replace("+ -", "- ").replace("+-", "-"))
        return newEquations

    equations = getEquations(model)

    def functionToIntegrate(valuesList, time):
        posX = valuesList[0]
        posY = valuesList[1]
        posZ = valuesList[2]
        
        if order == "meter":
            standardGravitationalParameter = 3.986004418 * 10**(14)
        else :
            #em kilometer
            standardGravitationalParameter = 3.986004418 * 10**5

        return [eval(equations[0]),eval(equations[1]),
                eval(equations[2])]


    solOdeintNoise = odeint(functionToIntegrate, y0=initialState, t = time, tfirst=False)
    return solOdeintNoise

def applyOdeint1stDerivDrag(initialState, time, order = "meter"):
    def functionToIntegrate(valuesList, time):
        # standardGravitationalParameter = 3.986004418 * 10**(14)
        posX = valuesList[0]
        posY = valuesList[1]
        posZ = valuesList[2]
        velX = valuesList[3]
        velY = valuesList[4]
        velZ = valuesList[5]
        if order == "meter":
            #m3 / s2
            standardGravitationalParameter = 3.986004418 * 10**(14)
            #kg / m3
            ro = 10**(-8.7)
            dragCoef = 2
            #m2
            area = 10
            #kg
            mass = 15
        else :
            #em kilometer
            standardGravitationalParameter = 3.986004418 * 10**5
            #kg / km3
            ro = 10**(-8.7*3)
            dragCoef = 2
            #km2
            area = 10/1000
            #kg
            mass = 15
        
        # # confirmar com henrique unidades
        # standardGravitationalParameter = 3.986004418 * 10**(14)
        # #multiplicar o exppoente por 3 caso sejam kms
        # ro = 10**(-8.7)
        # dragCoef = 2
        # area = 10
        # mass = 15

        
        return [velX, 
                velY,
                velZ,
                -standardGravitationalParameter * posX/(posX**2 + posY**2 + posZ**2)**(3/2) - 1/2 * ro * dragCoef * (area/mass) * np.linalg.norm([velX, velY, velZ]) * velX,
                -standardGravitationalParameter * posY/(posX**2 + posY**2 + posZ**2)**(3/2) - 1/2 * ro * dragCoef * (area/mass) * np.linalg.norm([velX, velY, velZ]) * velY , 
                -standardGravitationalParameter * posZ/(posX**2 + posY**2 + posZ**2)**(3/2) - 1/2 * ro * dragCoef * (area/mass) * np.linalg.norm([velX, velY, velZ]) * velZ,
                ]

    #o que sai do odeint sao os state vectors ao longo do tempo (posicoes e velocidades)
    #rows = timesteps
    #columns = variables
    return odeint(functionToIntegrate, y0=initialState, t = time, tfirst=False)

def useSindyEquations(model, initialState, time, order = "meter"):

    
    def getEquations(model):
        equationsModel = model.print()
        newEquations = []
        for equation in equationsModel:
            newEquations.append(equation.replace("+ -", "- ").replace("+-", "-"))
        return newEquations

    noiseEquations = getEquations(model)

    def functionToIntegrate(valuesList, time):
        posX = valuesList[0]
        posY = valuesList[1]
        posZ = valuesList[2]
        velX = valuesList[3]
        velY = valuesList[4]
        velZ = valuesList[5]
        
        if order == "meter":
            standardGravitationalParameter = 3.986004418 * 10**(14)
        else :
            #em kilometer
            standardGravitationalParameter = 3.986004418 * 10**5

        return [eval(noiseEquations[0]),eval(noiseEquations[1]),
                eval(noiseEquations[2]),eval(noiseEquations[3]),
                eval(noiseEquations[4]),eval(noiseEquations[5])]



    #o que sai do odeint sao os state vectors ao longo do tempo (posicoes e velocidades)
    #rows = timesteps
    #columns = variables

    solOdeintNoise = odeint(functionToIntegrate, y0=initialState, t = time, tfirst=False)
    return solOdeintNoise

def calculateXdot(dataMatrix):

    def functionToIntegrate(valuesList):
        posX = valuesList[0]
        posY = valuesList[1]
        posZ = valuesList[2]
        velX = valuesList[3]
        velY = valuesList[4]
        velZ = valuesList[5]
        #METERS
        # standardGravitationalParameter = 3.986004418 * 10**(14)
        #KMS
        standardGravitationalParameter = 3.986004418 * 10**5

        return [velX, 
                velY,
                velZ, 
                -standardGravitationalParameter * posX/(posX**2 + posY**2 + posZ**2)**(3/2), 
                -standardGravitationalParameter * posY/(posX**2 + posY**2 + posZ**2)**(3/2), 
                -standardGravitationalParameter * posZ/(posX**2 + posY**2 + posZ**2)**(3/2)]
        
    x_dotCalculated = []
    for x in dataMatrix: 
        x_dotCalculated.append(np.array(functionToIntegrate(x)))
    return np.array(x_dotCalculated)

def calculate2ndXdot(xdotMatrix):
    
    def functionToIntegrate2nd(valuesList, units = "meters"):
            posX = valuesList[0]
            posY = valuesList[1]
            posZ = valuesList[2]
            if units == "meters":
                standardGravitationalParameter = 3.986004418 * 10**(14)
            else:
                standardGravitationalParameter = 3.986004418 * 10**(5)


            return [-standardGravitationalParameter * posX/(posX**2 + posY**2 + posZ**2)**(3/2), 
                    -standardGravitationalParameter * posY/(posX**2 + posY**2 + posZ**2)**(3/2), 
                    -standardGravitationalParameter * posZ/(posX**2 + posY**2 + posZ**2)**(3/2)]

    x_dotCalculated1st = np.array(np.copy(xdotMatrix))[:,0:3]
    x_dotCalculated2nd = []
    for x in x_dotCalculated1st: 
        x_dotCalculated2nd.append(np.array(functionToIntegrate2nd(x, "kilometers")))
    return np.array(x_dotCalculated2nd)

def standardizeInputs(listOrbitsDerivs, listOrbitsStatesOdeint, listOrbitsTimesOdeint):
    listOrbitsDerivsStandardized = []
    listScalersDerivs = []
    for orbit in listOrbitsDerivs:
        scalerDerivsAux = StandardScaler()
        scaledOrbitDerivs = scalerDerivsAux.fit_transform(orbit)
        standardizedDerivsAux = pd.DataFrame(scaledOrbitDerivs, columns=["posX", "posY", "posZ", "velX", "velY", "velZ"])
        listOrbitsDerivsStandardized.append(standardizedDerivsAux.to_numpy())
        listScalersDerivs.append(scalerDerivsAux)

    listOrbitsStatesOdeintStandardized = []
    listScalersData = []
    for orbit in listOrbitsStatesOdeint:
        scalerDataAux = StandardScaler()
        scaledOrbitData = scalerDataAux.fit_transform(orbit)
        standardizedOdeintDataAux = pd.DataFrame(scaledOrbitData, columns=["posX", "posY", "posZ", "velX", "velY", "velZ"])
        listOrbitsStatesOdeintStandardized.append(standardizedOdeintDataAux.to_numpy())
        listScalersData.append(scalerDataAux)


    listOrbitsTimesOdeintStandardized = []
    listScalersTime = []
    for timeList in listOrbitsTimesOdeint:
        scalerTime = StandardScaler()
        scaledTime = scalerTime.fit_transform(timeList.reshape(-1,1)).flatten()
        listOrbitsTimesOdeintStandardized.append(scaledTime)
        listScalersTime.append(scalerTime)
    return listOrbitsDerivsStandardized, listOrbitsStatesOdeintStandardized, listScalersDerivs, listScalersData, listOrbitsTimesOdeintStandardized, listScalersTime





def integrateVelocityMatrix(time, sol1stDerivativeKm, velocitiesMatrix):
    # integrar matriz dos velocidades simulados para ter posicoes
    # p0 = initialPosition
    # p1 = p0 + v0 * (t1-t0)
    # p2 = p1 + v1 * (t2-t1)
    # p3 = p2 + v2 * (t3-t2)

    initialPosition = sol1stDerivativeKm[0][0:3]
    timesAux = time.copy()
    positionMatrix = []
    iteration = 0
    for i in velocitiesMatrix:
        if iteration == 0:
            positionMatrix.append(initialPosition)
            iteration += 1
        else:
            positionMatrix.append(positionMatrix[iteration-1] + velocitiesMatrix[iteration-1] * (timesAux[iteration] - timesAux[iteration-1]))
            iteration += 1
    return positionMatrix



def prepare_data_drag():
    allFilesOrbitalUnpickled = pd.read_pickle("./data/allFilesOrbital.pkl")
    #how many different sattelite ids
    allIds = allFilesOrbitalUnpickled["satID"].unique()
    allFilesOrbitalUnpickled = allFilesOrbitalUnpickled.sort_values(by=['satID', 'epochSince'])[["satID", "epochSince", "posX", "posY", "posZ", "velX", "velY", "velZ"]]

    #separate dataframe into a list of dataframes, one for each satellite
    listOrbitsAllInfo = [allFilesOrbitalUnpickled[allFilesOrbitalUnpickled["satID"] == id] for id in allIds]

    #one list only with positionsX, positionsY, positionsZ, velocitiesX, velocitiesY, velocitiesZ
    listOrbitsStates = [orbit[["epochSince", "posX", "posY", "posZ", "velX", "velY", "velZ"]].values for orbit in listOrbitsAllInfo]


    listOrbitsStatesOdeintDrag = []
    listOrbitsWithoutDrag = []
    listOrbitTimesOdeintDrag = []

    for orbit in listOrbitsStates:
        #posXYZ velXYZ da primeira orbita
        initialDataDrag = orbit[0][1:]
        #comecar no timestamp da primeira e gerar dados para 3 horas
        time = orbit[0][0] + np.linspace(0, 10800, 573)
        listOrbitTimesOdeintDrag.append(time)

        sol1stDerivativeKmDrag = applyOdeint1stDerivDrag(initialDataDrag, time)/1000
        listOrbitsStatesOdeintDrag.append(sol1stDerivativeKmDrag)

        withoutDrag = applyOdeint(initialDataDrag, time)/1000
        listOrbitsWithoutDrag.append(withoutDrag)
    return listOrbitsStatesOdeintDrag, listOrbitsWithoutDrag, listOrbitTimesOdeintDrag

def calculateXdotOrbitsDrag(listOrbitsStatesOdeintDrag):
        def functToIntegrateDrag(valuesList, order = "meter"):
                # standardGravitationalParameter = 3.986004418 * 10**(14)
                posX = valuesList[0]
                posY = valuesList[1]
                posZ = valuesList[2]
                velX = valuesList[3]
                velY = valuesList[4]
                velZ = valuesList[5]
                if order == "meter":
                        #m3 / s2
                        standardGravitationalParameter = 3.986004418 * 10**(14)
                        #kg / m3
                        ro = 10**(-8.7)
                        dragCoef = 2
                        #m2
                        area = 10
                        #kg
                        mass = 15
                else :
                        #em kilometer
                        standardGravitationalParameter = 3.986004418 * 10**5
                        #kg / km3
                        ro = 10**(-8.7*3)
                        dragCoef = 2
                        #km2
                        area = 10/1000
                        #kg
                        mass = 15
                

                return [velX, 
                        velY,
                        velZ,
                        -standardGravitationalParameter * posX/(posX**2 + posY**2 + posZ**2)**(3/2) - 1/2 * ro * dragCoef * (area/mass) * np.linalg.norm([velX, velY, velZ]) * velX,
                        -standardGravitationalParameter * posY/(posX**2 + posY**2 + posZ**2)**(3/2) - 1/2 * ro * dragCoef * (area/mass) * np.linalg.norm([velX, velY, velZ]) * velY , 
                        -standardGravitationalParameter * posZ/(posX**2 + posY**2 + posZ**2)**(3/2) - 1/2 * ro * dragCoef * (area/mass) * np.linalg.norm([velX, velY, velZ]) * velZ,
                        ]

        xDotDragOrbits = []
        for orbit in listOrbitsStatesOdeintDrag:
                listXdotSingleOrbit = []
                for state in orbit:
                        listXdotSingleOrbit.append(np.array(functToIntegrateDrag(state, 0))/1000)
                xDotDragOrbits.append(np.array(listXdotSingleOrbit))
        return xDotDragOrbits