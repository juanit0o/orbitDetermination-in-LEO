from matplotlib import pyplot as plt
import numpy as np
import pathlib
import typing
import pandas as pd
import re

from ns_plots import orbitplot
from ns_plots import timeseries


def axes_three_subplots():
    fig, ax = plt.subplots(3)
    return fig, ax


def cartesian_time_series(
        a: typing.List[float], b: typing.List[float], c: typing.List[float],
        time: typing.List[float], time_unit: str = "seconds",
        title: str = "Cartesian time series",
        labels: typing.List[str] = ["x", "y", "z"],
        filepath: pathlib.Path = None):

    fig, axes = axes_three_subplots()
    _add_cartesian_time_series(axes, a, b, c, time)

    fig.suptitle(title)

    if len(labels) == 3:
        for ax, label in zip(axes, labels):
            ax.set_ylabel(label)
    else:
        raise ValueError("Length of labels different than 3.")


    axes[2].set_xlabel(f"Time [{time_unit}]")

    if filepath is None:
        plt.show()


def _add_cartesian_time_series(
        axes: plt.Axes, a: typing.List[float], b: typing.List[float],
        c: typing.List[float], time: typing.List[float]):

    axes[0].plot(time, a)
    axes[1].plot(time, b)
    axes[2].plot(time, c)


def plot_orbit(
        x: typing.List[float],
        y: typing.List[float],
        z: typing.List[float],
        title: str = "Orbit",
        filepath: pathlib.Path = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    fig.suptitle(title)

    # Plot orbit
    ax.plot(x, y, z, color='r', label='orbit')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Label center of the Earth
    ax.scatter(0, 0, 0, label='Earth Centre')
    set_3d_axes_equal(ax)

    ax.legend()
    
    if filepath is not None:
        fig.savefig(filepath, bbox_inches='tight')
    else:
        plt.show()
    
def set_3d_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1]-x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*np.max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle+ plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle+ plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle+ plot_radius])

def plotPositionsVelocities(time, dataMatrix, title = " - odeint generated"):
    # plot.plot_orbit(realDrag[:,0], realDrag[:,1], realDrag[:,2],title="SINDy orbit")
    # plot.plot_orbit(dragSimulated[:,0], dragSimulated[:,1], dragSimulated[:,2],title="SINDy orbit")

    ###################
    ### POSITIONS #####
    ###################

    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0.4, wspace=0)

    axs[0].set_title('PosX', y=1.0, pad=5)
    axs[1].set_title('PosY', y=1.0, pad=5)
    axs[2].set_title('PosZ', y=1.0, pad=5)

    #define image siz
    fig.set_size_inches(20, 7)
    axs[0].plot(time, dataMatrix[:,0], ":", label="Odeint Data")
    axs[1].plot(time, dataMatrix[:,1], ":", label="Odeint Data")
    axs[2].plot(time, dataMatrix[:,2], ":", label="Odeint Data")

    fig.suptitle("Positions" + title)
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:3], loc='lower center', ncol=2, bbox_transform=fig.transFigure)

    
    ###################
    ### VELOCITIES#####
    ###################

    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0.4, wspace=0)

    axs[0].set_title('PosX', y=1.0, pad=5)
    axs[1].set_title('PosY', y=1.0, pad=5)
    axs[2].set_title('PosZ', y=1.0, pad=5)

    #define image siz
    fig.set_size_inches(20, 7)
    axs[0].plot(time, dataMatrix[:,3], ":", label="Odeint Data")
    axs[1].plot(time, dataMatrix[:,4], ":", label="Odeint Data")
    axs[2].plot(time, dataMatrix[:,5], ":", label="Odeint Data")

    fig.suptitle("Velocities" + title)
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:3], loc='lower center', ncol=2, bbox_transform=fig.transFigure)


def plot_orbits_versus(time, originalData, simulatedData, title="SINDy orbit"):
    plot_orbit(simulatedData[:,0], simulatedData[:,1], simulatedData[:,2], "Simulated Data")
    plot_orbit(originalData[:,0], originalData[:,1], originalData[:,2], "Original ODEINT Data")

    ###################
    ### POSITIONS #####
    ###################

    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0.4, wspace=0)

    axs[0].set_title('PosX', y=1.0, pad=5)
    axs[1].set_title('PosY', y=1.0, pad=5)
    axs[2].set_title('PosZ', y=1.0, pad=5)

    #define image siz
    fig.set_size_inches(20, 7)
    axs[0].plot(time, originalData[:,0], "-", label="Odeint Data")
    axs[0].plot(time, simulatedData[:,0], ":", label="Simulated Data", linewidth=4)
    axs[1].plot(time, originalData[:,1], "-", label="Odeint Data")
    axs[1].plot(time, simulatedData[:,1], ":", label="Simulated Data", linewidth=4)
    axs[2].plot(time, originalData[:,2], "-", label="Odeint Data")
    axs[2].plot(time, simulatedData[:,2], ":", label="Simulated Data", linewidth=4)

    fig.suptitle('Original data VS Simulated data (positions)')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:2], loc='lower center', ncol=2, bbox_transform=fig.transFigure)
    plt.show()

    # ###################
    # ### VELOCITIES#####
    # ###################

    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0.4, wspace=0)

    axs[0].set_title('VelX', y=1.0, pad=5)
    axs[1].set_title('VelY', y=1.0, pad=5)
    axs[2].set_title('VelZ', y=1.0, pad=5)

    #define image siz
    fig.set_size_inches(20, 7)
    axs[0].plot(time, originalData[:,3], "-", label="Odeint Data")
    axs[0].plot(time, simulatedData[:,3], ":", label="Simulated Data", linewidth=4)
    axs[1].plot(time, originalData[:,4], "-", label="Odeint Data")
    axs[1].plot(time, simulatedData[:,4], ":", label="Simulated Data", linewidth=4)
    axs[2].plot(time, originalData[:,5], "-", label="Odeint Data")
    axs[2].plot(time, simulatedData[:,5], ":", label="Simulated Data", linewidth=4)

    fig.suptitle('Original data vs Simulated data (velocities)')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:2], loc='lower center', ncol=2, bbox_transform=fig.transFigure)
    plt.show()


def plot_orbits_versus_2(t, training_data, model_data, title="SINDy orbit"):
    orbitplot.plot_orbits(
        x=[training_data[:, 0]/1e3, model_data[:, 0]/1e3],
        y=[training_data[:, 1]/1e3, model_data[:, 1]/1e3],
        z=[training_data[:, 2]/1e3, model_data[:, 2]/1e3],
        labels=["training data", "model fit"],
        xlabel="x [km]",
        ylabel="y [km]",
        zlabel="z [km]",
        title="Training data vs. model simulation",
        filepath = "Training data vs. model simulationBB.pdf"
    )

    time_hours = t/3600.0

    diff = training_data - model_data

    timeseries.plot_series3(
        x=[time_hours, time_hours],
        y1=[training_data[:, 0]/1e3, model_data[:, 0]/1e3],
        y2=[training_data[:, 1]/1e3, model_data[:, 1]/1e3],
        y3=[training_data[:, 2]/1e3, model_data[:, 2]/1e3],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        y1label="x [km]",
        y2label="y [km]",
        y3label="z [km]",
        title="Training data vs. model simulation (Position)",
        filepath = "Training data vs. model simulation (Position)BB.pdf"
    )
    
    
    print("Position Mean Errors: ")
    print("x: " + str(np.mean(diff[:, 0]/1e3)))
    print("y: " + str(np.mean(diff[:, 1]/1e3)))
    print("z: " + str(np.mean(diff[:, 2]/1e3)))

    timeseries.plot_series3(
        x=[time_hours],
        y1=[diff[:, 0]/1e3],
        y2=[diff[:, 1]/1e3],
        y3=[diff[:, 2]/1e3],
        labels=["model error"],
        xlabel="time [hour]",
        y1label="error x [km]",
        y2label="error y [km]",
        y3label="error z [km]",
        title="Difference between training data and model simulation (Position)",
        filepath = "Difference between training data and model simulation (Position)BB.pdf"
    )

    timeseries.plot_series3(
        x=[time_hours, time_hours],
        y1=[training_data[:, 3]/1e3, model_data[:, 3]/1e3],
        y2=[training_data[:, 4]/1e3, model_data[:, 4]/1e3],
        y3=[training_data[:, 5]/1e3, model_data[:, 5]/1e3],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        y1label="vx [km/s]",
        y2label="vy [km/s]",
        y3label="vz [km/s]",
        title="Training data vs. model simulation (Velocity)",
        filepath = "Training data vs. model simulation (Velocity)BB.pdf"
    )

    print("Velocity Mean Errors: ")
    print("vx: " + str(np.mean(diff[:, 3]/1e3)))
    print("vy: " + str(np.mean(diff[:, 4]/1e3)))
    print("vz: " + str(np.mean(diff[:, 5]/1e3)))
    
    timeseries.plot_series3(
        x=[time_hours],
        y1=[diff[:, 3]/1e3],
        y2=[diff[:, 4]/1e3],
        y3=[diff[:, 5]/1e3],
        labels=["model error"],
        xlabel="time [hour]",
        y1label="error vx [km/s]",
        y2label="error vy [km/s]",
        y3label="error vz [km/s]",
        title="Difference between training data and model simulation (Velocity)",
        filepath = "Difference between training data and model simulation (Velocity)BB.pdf"
    )

    #plot the density (last column)
    timeseries.plot_series(
        x=[time_hours, time_hours],
        y=[training_data[:, 6], model_data[:, 6]],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        ylabel="density [kg/m^3]",
        title="Training data vs. model simulation (Density)",
        filepath = "Training data vs. model simulation (Density)BB.pdf"
    )

    print("Density Mean Errors: ")
    print("Density: " + str(np.mean(diff[:, 6])))

    #difference between training data and model simulation density
    timeseries.plot_series(
        x=[time_hours],
        y=[diff[:, 6]],
        labels=["model error"],
        xlabel="time [hour]",
        ylabel="error density [kg/m^3]",
        title="Difference between training data and model simulation (Density)",
        filepath = "Difference between training data and model simulation (Density)BB.pdf"
    )

    

def plotSingleLosses(optimizer):
    # losses = optimizer.get_loss()
    # l2ErrorModel = losses[0]
    # l0_normErrorModel = losses[1]
    # l0_penaltyModel = losses[2]

    # plt.plot(l2ErrorModel)
    # plt.figure()
    # plt.plot(l0_normErrorModel)
    # plt.figure()

    lossesMatrix = optimizer.get_loss_matrix()
    #[i, k, R2, L2, L0, l0_penalty * L0, R2 + L2 + l0_penalty * L0]
    lossesMatrix = pd.DataFrame(lossesMatrix, columns=["iteration", "index", "dataFidelityR2", "weightVectorl2", "penaltyL0", "penaltyL0TimesL0", "totalLoss"])

    listNumbers = list(range(0, len(lossesMatrix)))
    plt.figure()
    plt.plot(listNumbers, lossesMatrix["dataFidelityR2"].values, "o")
    plt.title("Data Fidelity Loss")
    plt.show()
    plt.figure()
    plt.plot(listNumbers, lossesMatrix["weightVectorl2"].values, "o")
    plt.title("L2 Loss")
    plt.show()
    plt.figure()

def plotMultipleLosses(matrixLossesLoop, alphas, title = ""): 
    lastTotalError = []
    lastDataFidelityError = []
    lastL2Error = []
    lastL0Error = []
    for matrix in matrixLossesLoop:
        lastDataFidelityError.append(matrix[-1][2])
        lastL2Error.append(matrix[-1][3])
        lastL0Error.append(matrix[-1][4])
        lastTotalError.append(matrix[-1][-1])
        


    ########################
    ### Data Fidelity Loss #
    ########################

    plt.figure()
    plt.plot(alphas, lastDataFidelityError, "o")
    plt.xlabel("Alpha")
    plt.ylabel("Data Fidelity Loss")
    plt.title("Data Fidelity Loss vs Alphas (Regularization Factor)" + " "+ title)
    plt.yscale('log')
    plt.xscale('log')
    filePath = "DataFidelityLoss_" + title
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)
    plt.savefig(filePathSafe + ".png")
    plt.show()



    ##############
    ### L2 Loss #
    ##############

    plt.figure()
    plt.plot(alphas, lastL2Error, "o")
    plt.xlabel("Alpha")
    plt.ylabel("L2 Loss")
    plt.title("L2 Loss vs Alphas (Regularization Factor)" + " "+ title)
    plt.yscale('log')
    plt.xscale('log')
    filePath = 'L2Loss_' + title
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)
    plt.savefig(filePathSafe + ".png")
    plt.show()

    #################
    ### Total Loss #
    #################

    plt.figure()
    plt.plot(alphas, lastTotalError, "o")
    plt.xlabel("Alpha")
    plt.ylabel("Total Loss")
    plt.title("Total Loss vs Alphas (Regularization Factor)" + " "+ title)
    plt.yscale('log')
    plt.xscale('log')
    filePath = 'TotalLoss_' + title
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)
    plt.savefig(filePathSafe + ".png")
    plt.show()


    #################
    ### y = Data Fidelity, x = alpha*L2loss (l2 term)  #
    #################

    plt.figure()
    plt.plot(lastL2Error, lastDataFidelityError, "o")
    plt.xlabel("Alpha x L2 Loss")
    plt.ylabel("Data Fidelity Loss")
    plt.title("Data Fidelity Loss vs Alpha x L2 Loss" + " "+ title)
    plt.yscale('log')
    plt.xscale('log')
    filePath = 'DataFidelity_VS_alphaL2' + title
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)
    plt.savefig(filePathSafe + ".png")
    plt.show()

    #################
    ### y = Data Fidelity, x = Cardinality Theta  #
    #################

    plt.figure()
    plt.plot(lastL0Error, lastDataFidelityError, "o")
    plt.xlabel("# NonZero Coefficients")
    plt.ylabel("Data Fidelity Loss")
    plt.title("Data Fidelity Loss vs # NonZero Coefficients" + " "+ title)
    plt.yscale('log')
    plt.xscale('log')
    filePath = 'DataFidelity_VS_alphaNonZero' + title
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)
    plt.savefig(filePathSafe + ".png")
    plt.show()

def plot_noiseVSnoNoise(originalData, noisyData, time, noiseType):
    # plot.plot_orbit(gaussianSimulatedNoisyData[:,0], gaussianSimulatedNoisyData[:,1], gaussianSimulatedNoisyData[:,2],title="SINDy orbit with gaussian noise")
    # plot.plot_orbit(laplacianSimulatedNoisyData[:,0], laplacianSimulatedNoisyData[:,1], laplacianSimulatedNoisyData[:,2],title="SINDy orbit with gaussian noise")
    # plot.plot_orbit(cauchySimulatedNoisyData[:,0], cauchySimulatedNoisyData[:,1], cauchySimulatedNoisyData[:,2],title="SINDy orbit with gaussian noise")
    plot_orbit(noisyData[:,0], noisyData[:,1], noisyData[:,2], title="SINDy orbit with " + noiseType + " noise")


    # Positions
    fig, axs = plt.subplots(ncols=3, nrows=1, sharey=True)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0, wspace=0)

    axs[0].set_title('PosX', y=1.0, pad=5)
    axs[1].set_title('PosY', y=1.0, pad=5)
    axs[2].set_title('PosZ', y=1.0, pad=5)

    for col in range(3):
        #define image siz
        fig.set_size_inches(20, 7)
        axs[0].plot(time, originalData[:,0], ":", label="No noise", linewidth=3)
        axs[0].plot(time, noisyData[:,0], label="Noise")
        axs[1].plot(time, originalData[:,1], ":", label="No noise", linewidth=3)
        axs[1].plot(time, noisyData[:,1], label="Noise")
        axs[2].plot(time, originalData[:,2], ":", label="No noise", linewidth=3)
        axs[2].plot(time, noisyData[:,2], label="Noise")
    fig.suptitle('Positions w Noise VS Positions w/o Noise')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:2], loc='lower center', ncol=2, bbox_transform=fig.transFigure)
    plt.show()

    # ###############################################################

    fig, axs = plt.subplots(ncols=3, nrows=1, sharey=True)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0, wspace=0)

    axs[0].set_title('VelX', y=1.0, pad=5)
    axs[1].set_title('VelY', y=1.0, pad=5)
    axs[2].set_title('VelZ', y=1.0, pad=5)

    for col in range(3):
        #define image siz
        fig.set_size_inches(20, 7)
        axs[0].plot(time, originalData[:,3], ":", label="No noise", linewidth=3)
        axs[0].plot(time, noisyData[:,3], label="Noise")
        axs[1].plot(time, originalData[:,4], ":", label="No noise", linewidth=3)
        axs[1].plot(time, noisyData[:,4], label="Noise")
        axs[2].plot(time, originalData[:,5], ":", label="No noise", linewidth=3)
        axs[2].plot(time, noisyData[:,5], label="Noise")
    fig.suptitle('Velocities w Noise VS Positions w/o Noise')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:2], loc='lower center', ncol=2, bbox_transform=fig.transFigure)
    plt.show()

def plot_positionIntegrated_VS_positions(positionMatrix, sol1stDerivativeKm, time):
    #VELOCIDADES
    # plot.plot_orbit(dataModel2ndSimulated[:,0], dataModel2ndSimulated[:,1], dataModel2ndSimulated[:,2], title="SINDy orbit")
    #POSICOES
    plot_orbit(np.array(positionMatrix)[:,0], np.array(positionMatrix)[:,1],np.array(positionMatrix)[:,2] ,title="SINDy orbit")

    #################################
    ############ PLOTS ##############
    #################################

    ##########################
    fig, axs = plt.subplots(ncols=3, nrows=1, sharey=True)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0, wspace=0)

    axs[0].set_title('(PosX)\'\'', y=1.0, pad=5)
    axs[1].set_title('(PosY)\'\'', y=1.0, pad=5)
    axs[2].set_title('(PosZ)\'\'', y=1.0, pad=5)


    for col in range(3):
        #define image siz
        fig.set_size_inches(20, 7)
        # + 1450
        axs[0].plot(time, sol1stDerivativeKm[:,0], ":", label="Positions", linewidth=3)
        axs[0].plot(time, np.array(positionMatrix)[:,0], label="IntegratedVelocities")
        axs[1].plot(time, sol1stDerivativeKm[:,1], ":", label="Positions", linewidth=3)
        axs[1].plot(time, np.array(positionMatrix)[:,1], label="IntegratedVelocities")
        axs[2].plot(time, sol1stDerivativeKm[:,2], ":", label="Positions", linewidth=3)
        axs[2].plot(time, np.array(positionMatrix)[:,2], label="IntegratedVelocities")
    fig.suptitle('Odeint generated positions VS Integrated velocities from 2nd derivatives')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:2], loc='lower center', ncol=2, bbox_transform=fig.transFigure)

def plotDragvsNoDrag(time, noDrag, realDrag, dragSimulated, title="Sindy Orbit"):

    plot_orbit(realDrag[:,0], realDrag[:,1], realDrag[:,2],title="SINDy orbit")
    plot_orbit(dragSimulated[:,0], dragSimulated[:,1], dragSimulated[:,2],title="SINDy orbit")


    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0.4, wspace=0)

    axs[0].set_title('PosX', y=1.0, pad=5)
    axs[1].set_title('PosY', y=1.0, pad=5)
    axs[2].set_title('PosZ', y=1.0, pad=5)

    #define image siz
    fig.set_size_inches(20, 7)
    axs[0].plot(time, realDrag[:,0], ":", label="Drag real")
    axs[0].plot(time, dragSimulated[:,0], "-.", label="Drag das equações da sindy")
    axs[0].plot(time, noDrag[:,0], label="Sem Drag")

    axs[1].plot(time, realDrag[:,1], ":", label="Drag real")
    axs[1].plot(time, dragSimulated[:,1], "-.",  label="Drag das equações da sindy")
    axs[1].plot(time, noDrag[:,1], label="Sem Drag")

    axs[2].plot(time, realDrag[:,2], ":", label="Drag real")
    axs[2].plot(time, dragSimulated[:,2], "-.", label="Drag das equações da sindy")
    axs[2].plot(time, noDrag[:,2], label="Sem Drag")
    fig.suptitle('Positions w Drag VS Positions w Drag from sindy')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:3], loc='lower center', ncol=2, bbox_transform=fig.transFigure)
    plt.show()
    
    fig, axs = plt.subplots(ncols=1, nrows=3, sharey=False)
    #make graphics side by side together
    plt.subplots_adjust(hspace=0.4, wspace=0)

    axs[0].set_title('VelX', y=1.0, pad=5)
    axs[1].set_title('VelY', y=1.0, pad=5)
    axs[2].set_title('VelZ', y=1.0, pad=5)

    #define image siz
    fig.set_size_inches(20, 7)
    axs[0].plot(time, realDrag[:,3], ":", label="Drag real")
    axs[0].plot(time, dragSimulated[:,3], "-.", label="Drag das equações da sindy")
    axs[0].plot(time, noDrag[:,3], label="Sem Drag")

    axs[1].plot(time, realDrag[:,4], ":", label="Drag real")
    axs[1].plot(time, dragSimulated[:,4],"-.", label="Drag das equações da sindy")
    axs[1].plot(time, noDrag[:,4], label="Sem Drag")

    axs[2].plot(time, realDrag[:,5], ":", label="Drag real")
    axs[2].plot(time, dragSimulated[:,5],"-.", label="Drag das equações da sindy")
    axs[2].plot(time, noDrag[:,5], label="Sem Drag")
    fig.suptitle('Velocities w Drag VS Velocities w Drag from sindy')
    handles, labels = axs[2].get_legend_handles_labels()
    fig.legend(labels[:3], loc='lower center', ncol=2, bbox_transform=fig.transFigure)
    plt.show()

def plot_single_orbit_run(t, training_data, model_data, plotInfo = None) -> None:
    
    filePath = " (A:" + str(plotInfo[0]) + " I:" + str(plotInfo[1]) + " E:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4])
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)

    orbitplot.plot_orbits(
        x=[training_data[:, 0]/1e3, model_data[:, 0]/1e3],
        y=[training_data[:, 1]/1e3, model_data[:, 1]/1e3],
        z=[training_data[:, 2]/1e3, model_data[:, 2]/1e3],
        labels=["training data", "model fit"],
        xlabel="x [km]",
        ylabel="y [km]",
        zlabel="z [km]",
        title="Training data vs. model simulation" + " (A:" + str(plotInfo[0]) + " I:" + str(plotInfo[1]) + " E:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Training data vs. model simulation" + filePathSafe + ".png"
    )

    time_hours = t/3600.0

    diff = training_data - model_data

    timeseries.plot_series3(
        x=[time_hours, time_hours],
        y1=[training_data[:, 0]/1e3, model_data[:, 0]/1e3],
        y2=[training_data[:, 1]/1e3, model_data[:, 1]/1e3],
        y3=[training_data[:, 2]/1e3, model_data[:, 2]/1e3],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        y1label="x [km]",
        y2label="y [km]",
        y3label="z [km]",
        title="Training data vs. model simulation (Position)"+ " (Alt:" + str(plotInfo[0]) + " Inc:" + str(plotInfo[1]) + " Ecc:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Training data vs. model simulation (Position)" + filePathSafe + ".png"
    )
    
    print("Position Mean Errors: ")
    print("x: " + str(np.mean(diff[:, 0]/1e3)))
    print("y: " + str(np.mean(diff[:, 1]/1e3)))
    print("z: " + str(np.mean(diff[:, 2]/1e3)))

    timeseries.plot_series3(
        x=[time_hours],
        y1=[diff[:, 0]/1e3],
        y2=[diff[:, 1]/1e3],
        y3=[diff[:, 2]/1e3],
        labels=["model error"],
        xlabel="time [hour]",
        y1label="error x [km]",
        y2label="error y [km]",
        y3label="error z [km]",
        title="Difference between training data and model simulation (Position)"+ " (Alt:" + str(plotInfo[0]) + " Inc:" + str(plotInfo[1]) + " Ecc:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Difference between training data and model simulation (Position)" + filePathSafe + ".png"
    )

    timeseries.plot_series3(
        x=[time_hours, time_hours],
        y1=[training_data[:, 3]/1e3, model_data[:, 3]/1e3],
        y2=[training_data[:, 4]/1e3, model_data[:, 4]/1e3],
        y3=[training_data[:, 5]/1e3, model_data[:, 5]/1e3],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        y1label="vx [km/s]",
        y2label="vy [km/s]",
        y3label="vz [km/s]",
        title="Training data vs. model simulation (Velocity)"+ " (Alt:" + str(plotInfo[0]) + " Inc:" + str(plotInfo[1]) + " Ecc:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Training data vs. model simulation (Velocity)" + filePathSafe + ".png"
    )

    print("Velocity Mean Errors: ")
    print("vx: " + str(np.mean(diff[:, 3]/1e3)))
    print("vy: " + str(np.mean(diff[:, 4]/1e3)))
    print("vz: " + str(np.mean(diff[:, 5]/1e3)))
    
    timeseries.plot_series3(
        x=[time_hours],
        y1=[diff[:, 3]/1e3],
        y2=[diff[:, 4]/1e3],
        y3=[diff[:, 5]/1e3],
        labels=["model error"],
        xlabel="time [hour]",
        y1label="error vx [km/s]",
        y2label="error vy [km/s]",
        y3label="error vz [km/s]",
        title="Difference between training data and model simulation (Velocity)"+ " (Alt:" + str(plotInfo[0]) + " Inc:" + str(plotInfo[1]) + " Ecc:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Difference between training data and model simulation (Velocity)"+ filePathSafe + ".png"
    )

def plot_orbital_elements(t, training_data, model_data, plotInfo = None) -> None:
    
    filePath = " (A:" + str(plotInfo[0]) + " I:" + str(plotInfo[1]) + " E:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4])
    filePathSafe = re.sub(r'[^\w\d-]','_', filePath)

    time_hours = t/3600.0

    timeseries.plot_series3(
        x=[time_hours, time_hours],
        y1=[training_data[:, 0]/1e3, model_data[:, 0]/1e3],
        y2=[training_data[:, 1], model_data[:, 1]],
        y3=[training_data[:, 2]*180/np.pi, model_data[:, 2]*180/np.pi],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        y1label="sma [km]",
        y2label="ecc [-]",
        y3label="inc [deg]",
        title="Training data vs. model simulation (2)" + " (Alt:" + str(plotInfo[0]) + " Inc:" + str(plotInfo[1]) + " Ecc:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Training data vs. model simulation (1)" + filePathSafe + ".png"
    )

    timeseries.plot_series3(
        x=[time_hours, time_hours],
        y1=[training_data[:, 3]*180/np.pi, model_data[:, 3]*180/np.pi],
        y2=[training_data[:, 4]*180/np.pi, model_data[:, 4]*180/np.pi],
        y3=[training_data[:, 5]*180/np.pi, model_data[:, 5]*180/np.pi],
        labels=["training data", "model fit"],
        xlabel="time [hour]",
        y1label="arg_perigee [deg]",
        y2label="raan [deg]",
        y3label="true_anomaly [deg]",
        title="Training data vs. model simulation (2)" + " (Alt:" + str(plotInfo[0]) + " Inc:" + str(plotInfo[1]) + " Ecc:" + str(plotInfo[2]) + ") " + str(plotInfo[3]) + "_" + str(plotInfo[4]),
        filepath = "Training data vs. model simulation (2)" + filePathSafe + ".png"
    )

