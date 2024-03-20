import h5py
import torch
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def loadFile(t, plot = False):
    inFile = h5py.File(t,'r')
    fluidPositions = np.array(inFile['simulationData']['fluidPosition'])
    
    fluidVelocities = np.array(inFile['simulationData']['fluidVelocities'])
    fluidDensity = np.array(inFile['simulationData']['fluidDensity'])
    fluidPressure = np.array(inFile['simulationData']['fluidPressure'])
    fluidAreas = np.array(inFile['simulationData']['fluidAreas'])
    dudt = np.array(inFile['simulationData']['dudt'])
    numParticles = inFile.attrs['numParticles']
    timesteps = inFile.attrs['timesteps']
    dt = inFile.attrs['dt']

    if plot:
        fig, axis = plt.subplots(1, 5, figsize=(16,6), sharex = False, sharey = False, squeeze = False)

        def plot(fig, axis, mat, title, cmap = 'viridis'):
            im = axis.imshow(mat, extent = [0,numParticles,dt * timesteps,0], cmap = cmap)
            axis.axis('auto')
            ax1_divider = make_axes_locatable(axis)
            cax1 = ax1_divider.append_axes("bottom", size="2%", pad="6%")
            cb1 = fig.colorbar(im, cax=cax1,orientation='horizontal')
            cb1.ax.tick_params(labelsize=8) 
            axis.set_title(title)
        plot(fig,axis[0,0], fluidPositions, 'position')
        plot(fig,axis[0,1], fluidDensity, 'density')
        plot(fig,axis[0,2], fluidPressure, 'pressure')
        plot(fig,axis[0,3], fluidVelocities, 'velocity', 'RdBu')
        plot(fig,axis[0,4], dudt, 'dudt', 'RdBu')

        fig.suptitle(t)
        fig.tight_layout()
    inFile.close()
    return {
        'positions': torch.tensor(fluidPositions).type(torch.float32), 
        'density': torch.tensor(fluidDensity).type(torch.float32), 
        'pressure':torch.tensor(fluidPressure).type(torch.float32), 
        'area': torch.tensor(fluidAreas).type(torch.float32), 
        'velocity': torch.tensor(fluidVelocities).type(torch.float32), 
        'dudt' : torch.tensor(dudt).type(torch.float32)}


def export(simulationStates, 
           numParticles, timesteps, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleRadius, baseArea, particleSupport, dt, 
           generator, generatorSettings, folder = 'test_case_I', nameOverride = None):
    
    if nameOverride is None:
        if not os.path.exists('../../datasets/%s/' % folder):
            os.makedirs('../../datasets/%s/' % folder)
    else:
        if not os.path.exists('%s/' % folder):
            os.makedirs('%s/' % folder)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if nameOverride is None:
        outFile = h5py.File('../../datasets/%s/out_%s_%08d_%s.hdf5' % (folder, generator, generatorSettings['seed'], timestamp),'w')
    else:
        outFile = h5py.File('%s/out_%s_%08d_%s.hdf5' % (folder, generator, generatorSettings['seed'], timestamp),'w')

    outFile.attrs['minDomain'] = minDomain
    outFile.attrs['maxDomain'] = maxDomain

    outFile.attrs['baseArea'] = baseArea
    outFile.attrs['particleRadius'] = particleRadius
    outFile.attrs['particleSupport'] = particleSupport

    outFile.attrs['xsphConstant'] = xsphConstant
    outFile.attrs['diffusionAlpha'] = diffusionAlpha
    outFile.attrs['diffusionBeta'] = diffusionBeta
    outFile.attrs['kappa'] = kappa
    outFile.attrs['restDensity'] = restDensity
    outFile.attrs['c0'] = c0
    outFile.attrs['dt'] = dt

    outFile.attrs['numParticles'] = numParticles
    outFile.attrs['timesteps'] = timesteps

    outFile.attrs['generator'] = generator

    grp = outFile.create_group('generatorSettings')
    grp.attrs.update(generatorSettings)

    grp = outFile.create_group('simulationData')

    grp.create_dataset('fluidPosition', data = simulationStates[:,0].detach().cpu().numpy())
    grp.create_dataset('fluidVelocities', data = simulationStates[:,1].detach().cpu().numpy())
    grp.create_dataset('fluidDensity', data = simulationStates[:,2].detach().cpu().numpy())
    grp.create_dataset('fluidPressure', data = simulationStates[:,3].detach().cpu().numpy())
    grp.create_dataset('fluidAreas', data = simulationStates[:,13].detach().cpu().numpy())

    grp.create_dataset('dudt', data = simulationStates[:,4].detach().cpu().numpy())
    grp.create_dataset('dudt_k1', data = simulationStates[:,5].detach().cpu().numpy())
    grp.create_dataset('dudt_k2', data = simulationStates[:,6].detach().cpu().numpy())
    grp.create_dataset('dudt_k3', data = simulationStates[:,7].detach().cpu().numpy())
    grp.create_dataset('dudt_k4', data = simulationStates[:,8].detach().cpu().numpy())

    grp.create_dataset('dxdt_k1', data = simulationStates[:,9].detach().cpu().numpy())
    grp.create_dataset('dxdt_k2', data = simulationStates[:,10].detach().cpu().numpy())
    grp.create_dataset('dxdt_k3', data = simulationStates[:,11].detach().cpu().numpy())
    grp.create_dataset('dxdt_k4', data = simulationStates[:,12].detach().cpu().numpy())
    outFile.close()
