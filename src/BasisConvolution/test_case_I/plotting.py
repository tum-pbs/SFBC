# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
    
# sph related imports
# from BasisConvolution.oneDimensionalSPH.sph import *
# from BasisConvolution.oneDimensionalSPH.perlin import *
# neural network rlated imports
from torch.optim import Adam
# from BasisConvolution.oneDimensionalSPH.rbfConv import *
# from torch_geometric.loader import DataLoader
# from BasisConvolution.oneDimensionalSPH.trainingHelper import *
# plotting/UI related imports
from BasisConvolution.test_case_I.plotting import *
import matplotlib as mpl
# plt.style.use('dark_background')
cmap = mpl.colormaps['viridis']
from tqdm.autonotebook import trange, tqdm
from IPython.display import display, Latex
from datetime import datetime
# from BasisConvolution.oneDimensionalSPH.rbfNet import *
# from BasisConvolution.convNet import RbfNet
import h5py
import matplotlib.colors as colors
import torch
import torch.nn as nn
# %matplotlib notebook

from BasisConvolution.test_case_I.io import loadFile

def plotTrainingFiles(trainingFiles, numParticles, dt, timesteps):
    ns = int(np.sqrt(len(trainingFiles)))
    fig, axis = plt.subplots(ns, ns, figsize=(ns*6,ns * 2), sharex = True, sharey = True, squeeze = False)

    def plot(fig, axis, mat, title, cmap = 'viridis'):
        im = axis.imshow(mat, extent = [0,dt * timesteps,numParticles,0], cmap = cmap)
        axis.axis('auto')
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="2%", pad="6%")
        cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
        cb1.ax.tick_params(labelsize=8) 
        axis.set_title(title)

    for i in range(ns):
        for j in range(ns):
            data = loadFile(trainingFiles[ns * j + i], False)
            plot(fig,axis[i,j], data['velocity'].mT, trainingFiles[ns * j + i].split('/')[-1].split('.')[0].split('_')[2], 'RdBu')
    #         plot(fig,axis[i,j], data['dudt'].mT, trainingFiles[ns * j + i].split('/')[-1].split('.')[0].split('_')[2], 'RdBu')

    fig.tight_layout()



# def plotBatch(trainingData, settings, dataSet, bdata, device, offset, particleSupport, groundTruthFn, featureFn, model = None):
#     positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
#     i, j, distance, direction = batchedNeighborsearch(positions, setup)
#     x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

#     x = x[:,None].to(device)    
#     groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction, particleSupport).to(device)
#     distance = (distance * direction)[:,None].to(device)
#     features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)

# #     optimizer.zero_grad()
# #     prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
# #     lossTerm = lossFn(prediction, groundTruth)
# #     loss = torch.mean(lossTerm)
    
#     fig, axis = plt.subplot_mosaic('''AF
#     BC
#     DE''', figsize=(12,8), sharey = False, sharex = False)
    
#     positions = torch.vstack(positions).mT.detach().cpu().numpy()
#     vel = u.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     area = v.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     dudt = dudt.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     density = rho.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     inVel = inVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     outVel = outVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     gt = groundTruth.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     ft = features[:,0].reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    
#     axis['A'].set_title('Position')
#     axis['A'].plot(positions)
#     axis['B'].set_title('Density')
#     axis['B'].plot(positions, density)
#     axis['C'].set_title('Difference')
#     axis['C'].plot(positions, gt - ft)
#     axis['D'].set_title('Instantenous Velocity')
#     axis['D'].plot(positions, vel)
#     axis['E'].set_title('Ground Truth')
#     axis['E'].plot(positions, gt)
#     axis['F'].set_title('Features[:,0]')
#     axis['F'].plot(positions, ft)
    
#     fig.tight_layout()
    
# def plotTrainedBatch(trainingData, settings, dataSet, bdata, device, offset, modelState, groundTruthFn, featureFn, lossFn):
#     positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
#     i, j, distance, direction = batchedNeighborsearch(positions, setup)
#     x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

#     x = x[:,None].to(device)    
#     groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
#     distance = (distance * direction)[:,None].to(device)
#     features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
    
#     with torch.no_grad():
#         prediction = modelState['model'](features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
#         lossTerm = lossFn(prediction, groundTruth)
#         loss = torch.mean(lossTerm)
    
#     fig, axis = plt.subplot_mosaic('''ABC
#     DEF''', figsize=(16,5), sharey = False, sharex = True)
    
#     positions = torch.vstack(positions).mT.detach().cpu().numpy()
#     vel = u.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     area = v.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     dudt = dudt.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     density = rho.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     inVel = inVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     outVel = outVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     gt = groundTruth.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     ft = features[:,0].reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     loss = lossTerm.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     pred = prediction.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    
#     axis['A'].set_title('Density')
#     axis['A'].plot(positions, density)
#     axis['E'].set_title('Ground Truth - Features[:,0]')
#     axis['E'].plot(positions, gt - ft)
#     axis['B'].set_title('Ground Truth')
#     axis['B'].plot(positions, gt)
#     axis['D'].set_title('Features[:,0]')
#     axis['D'].plot(positions, ft)
#     axis['C'].set_title('Prediction')
#     axis['C'].plot(positions, pred)
#     axis['F'].set_title('Loss')
#     axis['F'].plot(positions, loss)
    
#     fig.tight_layout()
    
# # plotBatch(trainingData, settings, dataSet, bdata, device, offset)
    
    
# Copyright 2023 <COPYRIGHT HOLDER>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Math/parallelization library includes
import numpy as np
import torch

# Plotting includes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import NearestNDInterpolator
import scipy
from .sph import createGhostParticles, computeDensity
from BasisConvolution.detail.radius import findNeighborhoods

# Plots the given simulation (via simulationStates) and the given timesteps
# This function plots both density and velocity
def plotSimulationState(simulationStates, minDomain, maxDomain, dt, timepoints = []):
    fig, axis = plt.subplots(2, 1, figsize=(9,6), sharex = True, sharey = False, squeeze = False)

    axis[0,0].axvline(minDomain, color = 'black', ls = '--')
    axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
    axis[1,0].axvline(minDomain, color = 'black', ls = '--')
    axis[1,0].axvline(maxDomain, color = 'black', ls = '--')

    axis[1,0].set_xlabel('Position')
    axis[1,0].set_ylabel('Velocity[m/s]')
    axis[0,0].set_ylabel('Density[1/m]')

    def plotTimePoint(i, c, simulationStates, axis, minDomain, maxDomain):
        x = simulationStates[i,0,:]
        xPos = torch.remainder(x + minDomain, maxDomain - minDomain) - maxDomain
        y = simulationStates[i,c,:]
        idx = torch.argsort(xPos)
        axis.plot(xPos[idx].detach().cpu().numpy(), y[idx].detach().cpu().numpy(), label = 't = %1.2g' % (i * dt))
    if timepoints == []:
        plotTimePoint(0,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(0,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]//4,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]//4,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]//4*2,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]//4*2,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]//4*3,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]//4*3,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]-1,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]-1,2, simulationStates, axis[0,0], minDomain, maxDomain)
    else:
        for t in timepoints:
            plotTimePoint(t,1, simulationStates, axis[1,0], minDomain, maxDomain)
            plotTimePoint(t,2, simulationStates, axis[0,0], minDomain, maxDomain)

    axis[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axis[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

# # Plots the density of a set of particles, as determined using SPH, as well as the FFT and PSD of the density field
# # Used for plotting the initial density field/sampling
def plotDensityField(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport):
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)

    xs = fluidPositions.detach().cpu().numpy()
    densityField = fluidDensity.detach().cpu().numpy()
    fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
    numSamples = densityField.shape[-1]
    fs = numSamples/2
    fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
    x = densityField
    y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
    axis[0,0].plot(xs, densityField)
    axis[0,1].loglog(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'baseTarget')
    f, Pxx_den = scipy.signal.welch(densityField, fs, nperseg=len(x)//32)
    axis[0,2].loglog(f, Pxx_den, label = 'baseTarget')
    axis[0,2].set_xlabel('frequency [Hz]')
    axis[0,2].set_ylabel('PSD [V**2/Hz]')
    fig.tight_layout()
    return fluidDensity
# Plots a pseudo 2D plot of the given simulation showing both the density and velocity (color mapped) with
# time on the y-axis and position on the x-axis to demonstrate how the simulations evolve over time.
# This function is relatively slow as it does a resampling from the particle data to a regular grid, of size
# nx * ny, using NearestNDInterpolator.
def regularPlot(simulationStates, minDomain, maxDomain, dt, nx = 512, ny = 2048):
    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]
    positionArray = torch.remainder(positionArray + minDomain, maxDomain - minDomain) - maxDomain
    
    xys = torch.vstack((timeArray.flatten().to(positionArray.device).type(positionArray.dtype), positionArray.flatten())).mT.detach().cpu().numpy()

    interpVelocity = NearestNDInterpolator(xys, simulationStates[:,1].flatten().detach().cpu().numpy())
    interpDensity = NearestNDInterpolator(xys, simulationStates[:,2].flatten().detach().cpu().numpy())

    X = torch.linspace(torch.min(timeArray), torch.max(timeArray), ny).detach().cpu().numpy()
    Y = torch.linspace(minDomain, maxDomain, nx).detach().cpu().numpy()
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    # Z = interp(X, Y)

    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)


    im = axis[0,0].pcolormesh(X,Y,interpDensity(X,Y), cmap = 'viridis', vmin = torch.min(torch.abs(simulationStates[:,2])),vmax = torch.max(torch.abs(simulationStates[:,2])))
    # im = axis[0,0].imshow(simulationStates[:,2].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')

    im = axis[1,0].pcolormesh(X,Y,interpVelocity(X,Y), cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    # im = axis[1,0].imshow(simulationStates[:,1].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()