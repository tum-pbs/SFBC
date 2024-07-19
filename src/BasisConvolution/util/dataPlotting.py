import torch
import h5py
from BasisConvolution.util.datautils import getFrameCount, getFrames
from BasisConvolution.util.plotting import prepVisualizationState, visualizeParticleQuantity
import matplotlib.pyplot as plt
import numpy as np
import copy
from BasisConvolution.util.augment import augmentState, loadAugmentedFrame

@torch.jit.script
def mod(x, min : float, max : float):
    return torch.where(torch.abs(x) > (max - min) / 2, torch.sgn(x) * ((torch.abs(x) + min) % (max - min) + min), x)
    
def preparePlot_testCase_I(train_ds, hyperParameterDict):
        
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(0, train_ds, hyperParameterDict)

    fig, axis = plt.subplots(1,2, figsize = (10,4), squeeze=False)

    pos = currentState['fluid']['positions'][:,0]
    pos = mod(pos, -1, 1)
    pos = pos.detach().cpu().numpy()

    scx = axis[0,0].scatter(pos, currentState['fluid']['densities'].detach().cpu().numpy(), c = currentState['fluid']['features'][:,0], s = 0.5)
    fig.colorbar(scx, ax = axis[0,0])
    axis[0,0].set_xlabel('x')
    axis[0,0].set_ylabel('density')
    axis[0,0].set_title('Feature')


    scy = axis[0,1].scatter(pos, currentState['fluid']['densities'].detach().cpu().numpy(), c = currentState['fluid']['velocities'][:,0], s = 0.5, cmap = 'RdBu')
    axis[0,1].set_xlabel('x')
    axis[0,1].set_ylabel('density')
    axis[0,1].set_title('Velocity')

    fig.colorbar(scy, ax = axis[0,1])
    fig.tight_layout()
    return fig, axis, scx, scy

def updatePlot_testCase_I(plotState, train_ds, hyperParameterDict, frame):
    fig, axis, scx, scy = plotState
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(frame, train_ds, hyperParameterDict)

    pos = currentState['fluid']['positions'][:,0]
    pos = mod(pos, -1, 1)
    pos = pos.detach().cpu().numpy()

    scx.set_offsets(np.c_[pos, currentState['fluid']['densities'].detach().cpu().numpy()])
    scx.set_array(currentState['fluid']['features'][:,0].detach().cpu().numpy())

    scy.set_offsets(np.c_[pos, currentState['fluid']['densities'].detach().cpu().numpy()])
    scy.set_array(currentState['fluid']['velocities'][:,0].detach().cpu().numpy())

    if axis[0,0].get_ylim()[0] > currentState['fluid']['densities'].min() or axis[0,0].get_ylim()[1] < currentState['fluid']['densities'].max():
        axis[0,0].set_ylim(currentState['fluid']['densities'].detach().cpu().numpy().min(), currentState['fluid']['densities'].max())
    if axis[0,1].get_ylim()[0] > currentState['fluid']['densities'].min() or axis[0,1].get_ylim()[1] < currentState['fluid']['densities'].detach().cpu().numpy().max():
        axis[0,1].set_ylim(currentState['fluid']['densities'].detach().cpu().numpy().min(), currentState['fluid']['densities'].detach().cpu().numpy().max())


    fig.canvas.draw()

from BasisConvolution.util.plotting import updatePlot
def preparePlot_testCase_II(train_ds, hyperParameterDict):
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(0, train_ds, hyperParameterDict)
        
    deAugmentedState = augmentState(copy.deepcopy(currentState), augRotation = currentState['augmentRotation'].T if 'augmentRotation' in currentState else None, augmentFeatures=False)
    visualizationState = prepVisualizationState(deAugmentedState, config)

    s = 4
    fig, axis = plt.subplot_mosaic('''ABC''', figsize=(13.5,5), sharex = False, sharey = False)

    indexPlot = visualizeParticleQuantity(fig, axis['A'], config, visualizationState, quantity = 'indices', mapping = '.x', s = s, 
                            scaling = 'lin', gridVisualization=False, cmap = 'twilight', title = 'Particle Index', plotBoth=True, which = 'fluid')

    xVelocityPlot = visualizeParticleQuantity(fig, axis['B'], config, visualizationState, quantity = 'velocities', mapping = '.x', s = s, 
                            scaling = 'lin', gridVisualization=False, cmap = 'twilight', title = 'Particle x-Velcotiy', plotBoth=True, which = 'fluid')
    yVelocityPlot = visualizeParticleQuantity(fig, axis['C'], config, visualizationState, quantity = 'velocities', mapping = '.y', s = s, 
                            scaling = 'lin', gridVisualization=False, cmap = 'twilight', title = 'Particle y-Velocity', plotBoth=True, which = 'fluid')


    fig.tight_layout()

    return fig, axis, indexPlot, xVelocityPlot, yVelocityPlot

def updatePlot_testCase_II(plotState, train_ds, hyperParameterDict, frame):
    fig, axis, indexPlot, xVelocityPlot, yVelocityPlot = plotState
    # print(frame)

    frame[1] = '%05d' % frame[1]
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(frame, train_ds, hyperParameterDict)
    deAugmentedState = augmentState(copy.deepcopy(currentState), augRotation = currentState['augmentRotation'].T if 'augmentRotation' in currentState else None, augmentFeatures=False)

    visualizationState = prepVisualizationState(deAugmentedState, config)

    updatePlot(indexPlot, visualizationState, quantity = 'indices')
    updatePlot(xVelocityPlot, visualizationState, quantity = 'velocities')
    updatePlot(yVelocityPlot, visualizationState, quantity = 'velocities')
    fig.canvas.draw()

def preparePlot_testCase_IV(train_ds, hyperParameterDict):
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(0, train_ds, hyperParameterDict)

    # import matplotlib.pyplot as plt

    # Extract the x, y, and z coordinates from the currentState variable
    x = currentState['fluid']['positions'][:, 0].cpu().numpy()
    y = currentState['fluid']['positions'][:, 1].cpu().numpy()
    z = currentState['fluid']['positions'][:, 2].cpu().numpy()

    # Create a 3D plot
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(x, y, z, c = currentState['fluid']['areas'].cpu().numpy())

    fig.colorbar(sc, ax = ax)

    # Set labels for the x, y, and z axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')
    ax.set_title('Particle Area')
    axb = fig.add_subplot(122, projection='3d')

    scb = axb.scatter(x, y, z, c = currentState['fluid']['neighborhood']['numNeighbors'].cpu().numpy())

    fig.colorbar(scb, ax = axb)

    axb.set_xlabel('X')
    axb.set_ylabel('Y')
    axb.set_zlabel('Z')

    axb.set_aspect('equal')
    axb.set_title('Neighbor Count')

    fig.tight_layout()
    # Show the plot
    plt.show()

    return fig, ax, axb, sc, scb

def updatePlot_testCase_IV(plotState, train_ds, hyperParameterDict, frame):
    fig, ax, axb, sc, scb = plotState
    inFile = h5py.File(frame[0], 'r')
    frames =  getFrames(inFile)[0]
    # print(len(frames), ' - ', frame[1])
    # print(frames)
    frame[1] = str(frames[frame[1]])
    # print('Updated Frame: ', frame[1])
    inFile.close()
    # frame[1] = '%05d' % frame[1]
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(frame, train_ds, hyperParameterDict)

    x = currentState['fluid']['positions'][:, 0].cpu().numpy()
    y = currentState['fluid']['positions'][:, 1].cpu().numpy()
    z = currentState['fluid']['positions'][:, 2].cpu().numpy()


    sc._offsets3d = (x, y, z)
    sc.set_array(currentState['fluid']['areas'].cpu().numpy())

    scb._offsets3d = (x, y, z)
    scb.set_array(currentState['fluid']['neighborhood']['numNeighbors'].cpu().numpy())


    fig.canvas.draw()


def getFileCount(file):
    tf = h5py.File(file, 'r')
    currentFrameCount = getFrameCount(tf)
    tf.close()
    return currentFrameCount - 1


def getPreparePlotFunction(datasetStyle):
    if datasetStyle == 'testcase_I':
        return preparePlot_testCase_I
    elif datasetStyle == 'testcase_II':
        return preparePlot_testCase_II
    elif datasetStyle == 'testcase_IV':
        return preparePlot_testCase_IV
    else:
        raise ValueError(f"Unknown dataset style: {datasetStyle}")
    
def getUpdatePlotFunction(datasetStyle):
    if datasetStyle == 'testcase_I':
        return updatePlot_testCase_I
    elif datasetStyle == 'testcase_II' or datasetStyle == 'testcase_III' or datasetStyle == 'newFormat':
        return updatePlot_testCase_II
    elif datasetStyle == 'testcase_IV':
        return updatePlot_testCase_IV
    else:
        raise ValueError(f"Unknown dataset style: {datasetStyle}")