import torch

def constructFluidFeatures(attributes, inputData):
    fluidFeatures = torch.hstack(\
                (torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1), \
                 inputData['fluidVelocity'].type(torch.float32), 
                 torch.zeros(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)))
                #  inputData['fluidGravity'].type(torch.float32)))

                #  torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)))

    # fluidFeatures = torch.ones(inputData['fluidArea'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    # fluidFeatures[:,0] *= 7 / np.pi * inputData['fluidArea']  / attributes['support']**2
    
    boundaryFeatures = torch.hstack((inputData['boundaryNormal'].type(torch.float32), torch.zeros(inputData['boundaryNormal'].shape[0]).type(torch.float32).unsqueeze(dim=1)))
    # boundaryFeatures = torch.ones(inputData['boundaryNormal'].shape[0]).type(torch.float32).unsqueeze(dim=1)
    # boundaryFeatures[:,0] *=  7 / np.pi * inputData['boundaryArea']  / attributes['support']**2
    
    return inputData['fluidPosition'].type(torch.float32), inputData['boundaryPosition'].type(torch.float32), fluidFeatures, boundaryFeatures

import numpy as np
import random
def setSeeds(seed, verbose = False):
    if verbose:
        print('Setting all rng seeds to %d' % seed)


    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

import os
from .datautils import splitFile
from BasisConvolution.detail.util import debugPrint

def loadDataset(path, limitData = 0, frameDistance = 16, maxUnroll = 10, adjustForFrameDistance = True, verbose = False):
    basePath = os.path.expanduser(path)
    trainingFiles = [basePath + f for f in os.listdir(basePath) if f.endswith('.hdf5')]

    training = []
    validation = []
    testing = []

    
    if limitData > 0:
        files = []
        for i in range(max(len(trainingFiles), limitData)):
            files.append(trainingFiles[i])
        simulationFiles = files
    # simulationFiles = [simulationFiles[0]]
    if verbose:
        print('Input files:')
        for i, c in enumerate(trainingFiles):
            print('\t', i ,c)

    training = []
    validation = []
    testing = []

    for s in trainingFiles:
        f, s, u = splitFile(s, split = False, cutoff = -frameDistance * maxUnroll, skip = frameDistance if adjustForFrameDistance else 0)
        training.append((f, (s,u)))
    # for s in tqdm(validationFiles):
    #     f, s, u = splitFile(s, split = False, cutoff = -4, skip = 0)
    #     validation.append((f, (s,u)))

    if verbose:
        print('Processed data into datasets:')
        debugPrint(training)
    return training, trainingFiles

#     print(trainingFiles)


from BasisConvolution.test_case_II.datautils import datasetLoader
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

def getDataLoader(data, batch_size, shuffle = True, verbose = False):
    if verbose:
        print('Setting up data loaders')
    train_ds = datasetLoader(data)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler
    return train_ds, train_dataloader

def getFeatureSizes(constructFluidFeatures):    
    fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = constructFluidFeatures(attributes = {}, inputData = {'fluidArea': torch.ones(16), 'fluidVelocity': torch.ones(16,2), 'boundaryNormal': torch.ones(16, 2), 'fluidPosition': torch.ones(16,2), 'boundaryPosition': torch.ones(16,2)})
    return fluidFeatures.shape[1], boundaryFeatures.shape[1] 


from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



import numpy as np

@torch.jit.script
def wendland(q, h : float):
    C = 21 / (2 * np.pi)
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**3    
# Wendland 2 Kernel function and its derivative
@torch.jit.script
def kernelScalar(q :float, h : float):
    C = 21 / (2 * np.pi)
    b1 = (1. - q)**4
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**3    
@torch.jit.script
def wendlandGrad(q,r,h : float):
    C = 21 / (2 * np.pi)
    return - r * C / h**4 * (20. * q * (1. -q)**3)[:,None]
# support = 
from BasisConvolution.test_case_IV.radius import periodicNeighborSearchXYZ
from BasisConvolution.detail.scatter import scatter_sum
def generateGrid(nx,ny,nz):
    dx = 2 / nx
    x = torch.tensor(np.linspace(-1 + dx / 2,1 - dx / 2,nx, endpoint = True))
    y = torch.tensor(np.linspace(-1 + dx / 2,1 - dx / 2,nx, endpoint = True))
    z = torch.tensor(np.linspace(-1 + dx / 2,1 - dx / 2,nx, endpoint = True))

    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    gridPositions = torch.stack((xx,yy,zz), axis = -1).flatten().reshape(-1,3)

    return gridPositions, x, y, z

def optimizeVolume(gridPositions, minDomain, maxDomain, support, steps = 16):
    volume = 8 / gridPositions.shape[0]
    for i in range(steps):

        fi, fj, rij, dist = periodicNeighborSearchXYZ(gridPositions, gridPositions, minDomain, maxDomain, support, True, True )
        # volume = v
        rho = scatter_sum(volume * wendland(dist, support), fi, dim = 0, dim_size = gridPositions.shape[0])

        i, ni = torch.unique(fi, return_counts = True)
    #     print(torch.min(ni), torch.mean(ni.type(torch.float32)), torch.max(ni))
        #     print(torch.min(rho), torch.mean(rho), torch.max(rho))
    #     print(n, support, torch.max(ni), torch.mean(rho))

        volCorrection = torch.mean(rho)**(1/3)
        volume = volume / volCorrection
    # fi, fj, rij, dist = periodicNeighborSearchXYZ(gridPositions, gridPositions, minDomain, maxDomain, support, True, True )
    # i, ni = torch.unique(fi, return_counts = True)
    # rho = scatter_sum(volume * wendland(dist, support), fi, dim = 0, dim_size = gridPositions.shape[0])
    return volume


from .simplex import _init, getSimplexNoisePeriodic3
from noise.generator import generateOctaveNoise

def genData(xx, yy, zz, gridPositions, seed, parameterDict, simplexFrequency = None):
#     grp = grp.create_group('%d' % seed)
    
    torch.manual_seed(seed)
    # perm, _perm_grad_index3 = _init(seed)    
    jitter = torch.normal(mean = torch.zeros_like(gridPositions) + parameterDict['jitterMean'], std = torch.ones_like(gridPositions) * parameterDict['jitterAmount'])
    
    # print('simplexFrequency:', simplexFrequency)
    # print('nx', parameterDict['nx'])
    # print('octaves', int(np.floor(np.emath.logn(simplexFrequency, parameterDict['nx']))))

    octaveLimit = int(np.floor(np.emath.logn(2, parameterDict['nx'] / simplexFrequency) + 1)) if simplexFrequency is not None else 2
    # print(seed, simplexFrequency)

    _,_,_,vol = generateOctaveNoise(parameterDict['nx'], dim = 3, octaves = octaveLimit, lacunarity = 2, persistence = 0.5, baseFrequency = 2 if simplexFrequency is None else simplexFrequency, tileable = True, kind = 'perlin', device = gridPositions.device, dtype = gridPositions.dtype, seed = seed, normalized = True)

    # vol = getSimplexNoisePeriodic3(xx.numpy(),yy.numpy(),zz.numpy(), res = parameterDict['simplexFrequency'] if simplexFrequency is None else simplexFrequency, perm = perm, perm_grad_index3 = _perm_grad_index3) * parameterDict['simplexScale']+ 1
#     print(volume)
#     print(vol.shape)
    vols = parameterDict['volume'] *  torch.tensor(vol.flatten())

    fi, fj, rij, dist = periodicNeighborSearchXYZ(gridPositions + jitter, gridPositions + jitter, parameterDict['minDomain'], parameterDict['maxDomain'], parameterDict['support'], True, True )
    i, ni = torch.unique(fi, return_counts = True)
    rho = scatter_sum(vols[fj] * wendland(dist, parameterDict['support']), fi, dim = 0, dim_size = gridPositions.shape[0])
    
    x = (gridPositions + jitter).type(torch.float32)
    gradRhoSymmetric = rho[:,None] * scatter_sum(vols[fj,None] * (1/rho[fi,None] + 1/ rho[fj,None]) * wendlandGrad(dist, -rij, parameterDict['support']), fi, dim = 0, dim_size = gridPositions.shape[0])
    gradRhoDifference = scatter_sum(vols[fj,None]  / rho[fj,None] * (rho[fj,None] - rho[fi,None]) * wendlandGrad(dist, -rij, parameterDict['support']), fi, dim = 0, dim_size = gridPositions.shape[0])
    gradRhoNaive = scatter_sum(vols[fj,None] * wendlandGrad(dist, -rij, parameterDict['support']), fi, dim = 0, dim_size = gridPositions.shape[0])
    
    return {'x': x, 'vols': vols, 'rho': rho, 'ni':ni, 'gradRhoNaive': gradRhoNaive, 'gradRhoDifference': gradRhoDifference, 'gradRhoSymmetric': gradRhoSymmetric, 'jitter': jitter}
