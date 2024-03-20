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
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader

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

