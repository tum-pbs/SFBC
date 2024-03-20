# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
    
# sph related imports
# from BasisConvolution.oneDimensionalSPH.sph import *
# from BasisConvolution.oneDimensionalSPH.perlin import *
# neural network rlated imports
# from torch.optim import Adam
# from BasisConvolution.oneDimensionalSPH.rbfConv import *
# from torch_geometric.loader import DataLoader
# from BasisConvolution.oneDimensionalSPH.trainingHelper import *
# plotting/UI related imports
# from BasisConvolution.oneDimensionalSPH.plotting import *
# import matplotlib as mpl
# plt.style.use('dark_background')
# cmap = mpl.colormaps['viridis']
# from tqdm.autonotebook import trange, tqdm
# from IPython.display import display, Latex
# from datetime import datetime
# from BasisConvolution.oneDimensionalSPH.rbfNet import *
# from BasisConvolution.convNet import RbfNet
# import h5py
# import matplotlib.colors as colors
import torch
# import torch.nn as nn
# %matplotlib notebook


from BasisConvolution.detail.radius import batchedNeighborsearch
# from BasisConvolution.oneDimensionalSPH.util import *

def flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity):
    return torch.hstack(positions), torch.hstack(velocities), torch.hstack(areas), torch.hstack(density), torch.hstack(dudts), torch.hstack(inputVelocity), torch.hstack(outputVelocity)


def loadBatch(particleData, settings, dataSet, bdata, device, offset):
    dataEntries = [dataSet[b] for b in bdata]
    
    positions = [particleData[f]['positions'][t,:].to(device) for f,t in dataEntries]
    velocities = [particleData[f]['velocity'][t,:].to(device) for f,t in dataEntries]
    areas = [particleData[f]['area'][t,:].to(device) for f,t in dataEntries]
    dudts = [particleData[f]['dudt'][t,:].to(device) for f,t in dataEntries]
    densities = [particleData[f]['density'][t,:].to(device) for f,t in dataEntries]

    inputVelocity = [(particleData[f]['positions'][t,:] - particleData[f]['positions'][max(0, t - (offset -1 ))]).to(device) / settings[f]['dt'] for f,t in dataEntries] 
    outputVelocity = [particleData[f]['stacked'][t,:].to(device) for f,t in dataEntries] 

    setup = [settings[f] for f,t in dataEntries]
    
    return positions, velocities, areas, dudts, densities, inputVelocity, outputVelocity, setup
    

def getTestcase(testingData, settings, f, frames, device, offset):
    positions = [testingData[f]['positions'][t,:].to(device) for t in frames]
    velocities = [testingData[f]['velocity'][t,:].to(device) for t in frames]
    areas = [testingData[f]['area'][t,:].to(device) for t in frames]
    dudts = [testingData[f]['dudt'][t,:].to(device) for t in frames]
    densities = [testingData[f]['density'][t,:].to(device) for t in frames]
    setup = [settings[f] for t in frames]
    inputVelocity = [(testingData[f]['positions'][t,:] - testingData[f]['positions'][max(0, t - (offset -1 ))]).to(device)  / settings[f]['dt'] for t in frames] 
    outputVelocity = [testingData[f]['stacked'][t,:].to(device) for t in frames] 
    
    return positions, velocities, areas, dudts, densities, inputVelocity, outputVelocity, setup
def loadTestcase(testingData, settings, f, frames, device, groundTruthFn, featureFn, offset):
    positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = getTestcase(testingData, settings, f, frames, device, offset)

    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

    x = x[:,None].to(device)    
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, settings[f]['particleSupport']).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
#     print(groundTruth)
    return positions, velocities, areas, density, dudts, features, i, j, distance, groundTruth


def loadTestcase(testingData, settings, f, frames, device, groundTruthFn, featureFn, offset, particleSupport):
    positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = getTestcase(testingData, settings, f, frames, device, offset)

    i, j, distance, direction = batchedNeighborsearch(positions, setup)
    x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

    x = x[:,None].to(device)    
    groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction, particleSupport).to(device)
    distance = (distance * direction)[:,None].to(device)
    features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
#     print(groundTruth)
    return positions, velocities, areas, density, dudts, features, i, j, distance, groundTruth, x, u
