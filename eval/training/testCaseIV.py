import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)# import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import time
import torch
from torch_geometric.loader import DataLoader
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

# from rbfConv import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
%matplotlib notebook
import copy

import time
import torch
from torch_geometric.loader import DataLoader
from tqdm.notebook import trange, tqdm
import argparse
import yaml
from torch_geometric.nn import radius
from torch.optim import Adam
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

from BasisConvolution.convLayer import RbfConv
# from dataset import compressedFluidDataset, prepareData

import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))


import tomli
from scipy.optimize import minimize
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

seed = 0


import random 
import numpy as np
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
# print(torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('running on: ', device)
torch.set_num_threads(1)

from joblib import Parallel, delayed

# from cutlass import *
# from rbfConv import *
from tqdm.autonotebook import tqdm
import random 
import numpy as np
from BasisConvolution.test_case_II.datautils import splitFile
from BasisConvolution.test_case_II.datautils import datasetLoader, loadFrame
from BasisConvolution.detail.windows import getWindowFunction
from BasisConvolution.test_case_II.util import constructFluidFeatures
from BasisConvolution.convNet import RbfNet
from datetime import datetime
import portalocker
from BasisConvolution.detail.augment import augment
from BasisConvolution.test_case_II.training import processBatch
# from datautils import *
# from sphUtils import *
# from lossFunctions import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from BasisConvolution.test_case_II.util import setSeeds, loadDataset, getDataLoader, getFeatureSizes
from BasisConvolution.detail.util import count_parameters
import json
from BasisConvolution.test_case_II.util import NumpyArrayEncoder
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from BasisConvolution.test_case_IV.training import loadFrame, runNetwork
from BasisConvolution.detail.scatter import scatter_sum
from BasisConvolution.test_case_IV.eval import getUnrollFrame

cm = mpl.colormaps['viridis']
import matplotlib.colors as colors
import h5py


from BasisConvolution.test_case_IV.radius import periodicNeighborSearchXYZ
from BasisConvolution.detail.scatter import scatter_sum
from BasisConvolution.test_case_IV.util import generateGrid, optimizeVolume
from BasisConvolution.test_case_IV.simplex import getSimplexNoisePeriodic3, _init
from BasisConvolution.test_case_IV.util import wendland, wendlandGrad
from BasisConvolution.test_case_IV.arguments import parser

args = parser.parse_args()

perm, _perm_grad_index3 = _init(1234)

setSeeds(args.seed, args.verbose)

if args.verbose:
    print('Available cuda devices:', torch.cuda.device_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.verbose:
    print('Running on Device %s' % device)
torch.set_num_threads(1)

inFile = h5py.File('../../datasets/test_case_IV/lowJitter.hdf5', 'r')
# inFile.close()

dataParameters = {}
for k in inFile.attrs.keys():
    dataParameters[k] = inFile.attrs[k]
dataEntries = len(inFile['simulationData'].keys())
trainingEntries = np.arange(0, dataEntries * 0.9)
testingEntries = np.arange(dataEntries * 0.9, dataEntries)

train_dataloader = DataLoader(list(inFile['simulationData'].keys()), shuffle=True, batch_size = args.batch_size)
train_iter = iter(train_dataloader)

bdata = next(train_iter)
# print(bdata)

def loadBatch(bdata, inFile):
    with record_function("Load Batch"): 
        processed = []
        for b in bdata:
            dataGrp = inFile['simulationData'][b]
            loaded = {}
            for g in dataGrp:
                loaded[g] = torch.tensor(np.array(dataGrp[g]))
    #             print(loaded[g])abs
            for a in dataGrp.attrs:
                loaded[a] = dataGrp.attrs[a]
            processed.append(loaded)
        return processed
    
batchData = loadBatch(bdata, inFile)

from BasisConvolution.test_case_IV.plotting import plotSlices
from BasisConvolution.test_case_IV.util import genData

def constructInputs(bdata, inFile, device, features = ['one', 'volume'], target = 'gradRhoDifference', dataOverride = None):
    with record_function("Construct Inputs"): 
        batchData = loadBatch(bdata, inFile) if dataOverride is None else dataOverride
        positions = torch.vstack([b['x'] for b in batchData])
        batchIndex = torch.hstack([torch.ones_like(b['rho']) * i for i, b in enumerate(batchData)])

        collectedFeatures = []
        for f in features:
            if f == 'one':
                collectedFeatures.append(torch.hstack([torch.ones_like(b['rho']) for b in batchData])[:,None])
            elif f == 'zero':
                collectedFeatures.append(torch.hstack([torch.zeros_like(b['rho']) for b in batchData])[:,None])
            elif f == 'volume':
                collectedFeatures.append(torch.hstack([b['vols'] for b in batchData])[:,None])
            elif f == 'normalizedVolume':
                collectedFeatures.append(torch.hstack([b['vols'] for b in batchData])[:,None] / (4/3 * np.pi * inFile.attrs['support']**3) * inFile.attrs['numNeighbors'])
            elif f == 'ni':
                collectedFeatures.append(torch.hstack([b['ni'] for b in batchData])[:,None])
            elif f == 'rho':
                collectedFeatures.append(torch.hstack([b['rho'] for b in batchData])[:,None])
            elif f == 'rhoTimesVolume':
                collectedFeatures.append(torch.hstack([b['rho'] * b['vols'] for b in batchData])[:,None])
            elif f == 'rhoTimesNormalizedVolume':
                collectedFeatures.append(torch.hstack([b['rho'] * b['vols'] for b in batchData])[:,None] / (4/3 * np.pi * inFile.attrs['support']**3) * inFile.attrs['numNeighbors'])
            elif f == 'rhoOverVolume':
                collectedFeatures.append(torch.hstack([b['rho'] * b['vols'] for b in batchData])[:,None])
            elif f == 'rhoOverNormalizedVolume':
                collectedFeatures.append(torch.hstack([b['rho'] / b['vols'] for b in batchData])[:,None]* (4/3 * np.pi * inFile.attrs['support']**3) * inFile.attrs['numNeighbors'])
            elif f == 'volumeOverRho':
                collectedFeatures.append(torch.hstack([b['vols'] * b['rho'] for b in batchData])[:,None])
            elif f == 'normalizedVolumeOverRho':
                collectedFeatures.append(torch.hstack([b['vols'] / b['rho'] for b in batchData])[:,None]/ (4/3 * np.pi * inFile.attrs['support']**3) * inFile.attrs['numNeighbors'])
            elif f == 'frequency':
                collectedFeatures.append(torch.hstack([torch.ones_like(b['rho']) * b['frequency'] for b in batchData])[:,None])
            elif f == 'seed':
                collectedFeatures.append(torch.hstack([torch.ones_like(b['rho']) * b['seed'] for b in batchData])[:,None])
            else:
                collectedFeatures.append(torch.vstack([b[f] for b in batchData]))        

        features = torch.hstack(collectedFeatures).type(torch.float32)
        if 'grad' in target:
            gt = torch.vstack([b[target] for b in batchData])
        else:
            gt = torch.hstack([b[target] for b in batchData])[:,None]
        return positions.to(device), features.to(device), gt.to(device), batchIndex.to(device)
def batchNeighborsearch(bdata, inFile, device, dataOverride = None):    
    with record_function("Batched Neighborsearch"): 
        batchData = loadBatch(bdata, inFile) if dataOverride is None else dataOverride        
        positions = []
        settings = []
        for batch in batchData:
            positions.append(torch.tensor(np.array(batch['x'])).to(device))
            settings.append({'minDomain': inFile.attrs['minDomain'], 'maxDomain': inFile.attrs['maxDomain'], 'support': inFile.attrs['support']})

        with record_function("Batch Neighborsearch Processing"): 
            neighborLists = [periodicNeighborSearchXYZ(x,x, parameterDict['minDomain'], parameterDict['maxDomain'], parameterDict['support'], True, True ) for x, parameterDict in zip(positions, settings)]

        neigh_i = [n[0] for n in neighborLists]
        neigh_j = [n[1] for n in neighborLists]
        neigh_direction = [n[2] for n in neighborLists]
        neigh_distance = [n[3] for n in neighborLists]

        for i in range(len(neighborLists) - 1):
            neigh_i[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
            neigh_j[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])

        neigh_i = torch.hstack(neigh_i)
        neigh_j = torch.hstack(neigh_j)
        neigh_distance = torch.hstack(neigh_distance)
        neigh_direction = torch.vstack(neigh_direction)

        return neigh_i, neigh_j, neigh_distance, neigh_direction
def processBatch(bdata, model, inFile, features = ['vols', 'one'], target = 'gradRhoDifference', li = False, dataOverride = None):
    x, features, gt, batchIndex = constructInputs(bdata, inFile, device, features = features, target = target, dataOverride = dataOverride)
    fi, fj, rij, direction = batchNeighborsearch(bdata, inFile, device, dataOverride = dataOverride)  
#     print('Features:', features)
#     print('Connectivity:', fi, fj)
    ffi, ni = torch.unique(fi, return_counts = True)
    ffj, nj = torch.unique(fj, return_counts = True)
#     print('Neighbors i:', torch.min(ni), torch.mean(ni.type(torch.float32)), torch.max(ni))
#     print('Neighbors j:', torch.min(nj), torch.mean(nj.type(torch.float32)), torch.max(nj))
#     print(torch.min(direction), torch.max(direction))
#     print('Direction:', direction)
    prediction = model(features, fi, fj, direction * rij[:,None], None, None, None)
#     print('Network Prediction:', prediction)
#     print('Target:', gt)
#     print('Ratio:', prediction / gt)
    loss = torch.linalg.norm(prediction - gt, dim = -1)
#     print('Loss:', loss)
#     print('Li:', model.li)
    if dataOverride is None:
        return (loss * (1 if not li else model.li)).reshape(len(bdata), -1), features, prediction, gt
    else:
        return (loss * (1 if not li else model.li)).reshape(len(dataOverride), -1), features, prediction, gt

def processDataLoaderIter(hyperParameterDict, e, inFile, dataLoader, dataIter, model, optimizer, scheduler, train, gtqdms, pb, prefix = '', gpu = 0, gpus = 1, dataOverride = None):
    with record_function("prcess data loader"): 
        pbl = gtqdms[gpu + gpus]
        losses = []
        batchIndices = []

        if train:
            model = model.train(True)
        else:
            model = model.train(False)

        with portalocker.Lock('README.md', flags = 0x2, timeout = None):
            pbl.reset(total=hyperParameterDict['iterations'])
        i = 0
        for b in range(hyperParameterDict['iterations']):
            try:
                bdata = next(dataIter)
            except:
                dataIter = iter(dataLoader)
                bdata = next(dataIter)
#             bdata = [list(inFile['simulationData'].keys())[0]]
            with record_function("prcess data loader[batch]"): 
                if train:
                    optimizer.zero_grad()
                loss, _, _, _ = processBatch(bdata, model, inFile, features = hyperParameterDict['features'], target = hyperParameterDict['target'], li = hyperParameterDict['liLoss'] == 'yes', dataOverride = dataOverride)
                batchLosses = torch.mean(loss, dim = -1)
                loss = torch.mean(loss)
                
                batchIndices.append(np.array(bdata))
                losses.append(batchLosses.detach().cpu().numpy())

                with record_function("prcess data loader[batch] - backward"): 
                    if train:
                        loss.backward()
                        optimizer.step()
                        
                lossString = np.array2string(batchLosses.detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.2e" % x})
                batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%04d' % x}))

                with portalocker.Lock('README.md', flags = 0x2, timeout = None):
                    pbl.set_description('%8s[gpu %d]: %3d @ %1.1e: :  %s -> %.2e' %(prefix, gpu, e, optimizer.param_groups[0]['lr'], batchString, loss.detach().cpu().numpy()))
                    pbl.update()
                    if prefix == 'training':
                        # pb.set_description('[gpu %d] Learning: %1.4e Validation: %1.4e' %(args.gpu, np.mean(np.mean(np.vstack(losses)[:,:,0], axis = 1)), 0))
                        pb.set_description('[gpu %d] %90s - Learning: %1.4e |  %1.4e' %(gpu, hyperParameterDict['shortLabel'], np.mean(np.hstack(losses)),  np.mean(np.hstack(losses if len(losses) < 100 else losses[-100:]))))
                    pb.update()
                ii = e * hyperParameterDict['epochs'] + b
                if ii % hyperParameterDict['lrStep'] == 0 and ii > 0:
                    scheduler.step()
#                 i = i + 1
#                 if i > 100:
#                     break
        bIndices  = np.hstack(batchIndices)
        losses = np.vstack(losses)

        # idx = np.argsort(bIndices)
        # bIndices = bIndices[idx]
        # losses = losses[idx]

        epochLoss = losses
        return epochLoss
    
# processDataLoaderIter(hyperParameterDict, 0, inFile, train_dataloader, train_iter, model, optimizer, True, gtqdms, prefix = 'training', gpu = args.gpu, gpus = args.gpus)

def initializeTraining(args, hyperParameterDict, writeData):
    if args.verbose:
        print('Writing output to ./%s/%s' % (hyperParameterDict['output'], hyperParameterDict['exportString']))
    if writeData:
    # exportPath = './trainingData/%s - %s.hdf5' %(self.config['export']['prefix'], timestamp)
        if not os.path.exists('./%s/%s' % (hyperParameterDict['output'], hyperParameterDict['exportString'])):
            os.makedirs('./%s/%s' % (hyperParameterDict['output'], hyperParameterDict['exportString']))


    # self.outFile = h5py.File(self.exportPath,'w')

    gtqdms = []
    if args.verbose:
        print('Setting up tqdm progress bars')

    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        for g in range(args.gpus):
            gtqdms.append(tqdm(range(0, (hyperParameterDict['epochs']) * hyperParameterDict['iterations']), position = g, leave = False))
        for g in range(args.gpus):
            gtqdms.append(tqdm(range(1, hyperParameterDict['epochs'] + 1), position = args.gpus + g, leave = False))
    # print(torch.cuda.current_device())

    gpu = args.gpu
    pb = gtqdms[gpu]

    training = {}
    # training_fwd = {}
    validation = {}
    testing = {}

    with portalocker.Lock('README.md', flags = 0x2, timeout = None):
        pb.set_description('[gpu %d]' %(gpu))

    trainLoss = 0
    train_iter = iter(train_dataloader)

    trainingEpochLosses = []
    setSeeds(hyperParameterDict['seed'], verbose = args.verbose)
    
    return gtqdms, trainingEpochLosses

def traininingLoop(model, optimizer, scheduler, hyperParameterDict, inFile, train_dataloader, train_iter, gtqdms, args, writeData, dataOverride = None):
    trainingEpochLosses = []
    for epoch in range(hyperParameterDict['epochs']):
        losses = []
        trainingEpochLoss = processDataLoaderIter(hyperParameterDict, epoch, inFile, train_dataloader, train_iter, model, optimizer, scheduler, True, gtqdms, pb = gtqdms[args.gpu], prefix = 'training', gpu = args.gpu, gpus = args.gpus, dataOverride = dataOverride)
        trainingEpochLosses.append(trainingEpochLoss)
#         if epoch % 5 == 0 and epoch > 0:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = 0.5 * param_group['lr']
        if writeData:
            torch.save(model.state_dict(), './%s/%s/model_%03d.torch' % (hyperParameterDict['output'], hyperParameterDict['exportString'], epoch))
    return trainingEpochLosses 

def plotPredictionWithoutLossCurve(model, inFile, hyperParameterDict, generatedData):
    bdata = [list(inFile['simulationData'].keys())[0]]
    loss, features, prediction, gt = processBatch(bdata, model, inFile, features = hyperParameterDict['features'], target = hyperParameterDict['target'], li = hyperParameterDict['liLoss'] == 'yes', dataOverride = [generatedData])
    batchedLoss = torch.mean(loss, dim = -1)
    loss = torch.mean(loss)
#     print('batchedLoss:', batchedLoss)

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = features[:,0].detach().cpu(), s = 32)
    # fig.colorbar(sc, ax=ax)
    fig.colorbar(sc, ax=ax,orientation = 'horizontal')
    ax.set_box_aspect([1,1,1]) 
    ax.set_title('Input Feature')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = gt.detach().cpu(), s = 32)
    # fig.colorbar(sc, ax=ax)
    ax.set_box_aspect([1,1,1]) 
    fig.colorbar(sc, ax=ax,orientation = 'horizontal')
    ax.set_title('Ground Truth Density')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = prediction.detach().cpu(), s = 32)
    fig.colorbar(sc, ax=ax,orientation = 'horizontal')
    ax.set_box_aspect([1,1,1]) 
    ax.set_title('Predicted Density')

    fig.tight_layout()

def initModel(args, inFile):    
    nx = inFile.attrs['nx']
    ny = inFile.attrs['ny']
    nz = inFile.attrs['nz']
    support = inFile.attrs['support']
    volume = inFile.attrs['volume']
    maxDomain = inFile.attrs['maxDomain']
    minDomain = inFile.attrs['minDomain']
    numNeighbors = inFile.attrs['numNeighbors']

    gridPositions, xx, yy, zz = generateGrid(nx, ny, nz)
    
    parameterDict = {
        'nx' : nx, 'ny' : ny, 'nz': nz,
        'support': support, 'minDomain': minDomain, 'maxDomain': maxDomain,
        'numNeighbors': numNeighbors, 'dx': 2 / nx, 'volume': volume,
        'simplexFrequency': 0.1, 'simplexScale': 1.05 / 2, 'jitterMean': 0,'jitterAmount': 0.005
    }
    generatedData = genData(xx,yy,zz,gridPositions, 1234567, parameterDict)
    _, features, _, _ = constructInputs('12345', inFile, device, args.features.split(' '), dataOverride = [generatedData])
    
    totalIterations = args.iterations * args.epochs
    lrSteps = int(np.ceil((totalIterations - args.lrStep) / args.lrStep))
    gamma = np.power(args.finalLR / args.initialLR, 1/lrSteps)

    hyperParameterDict = {}
    hyperParameterDict['nx'] = args.nx
    hyperParameterDict['ny'] = args.ny
    hyperParameterDict['nz'] = args.nz
    hyperParameterDict['coordinateMapping'] = args.coordinateMapping
    hyperParameterDict['rbf_x'] = args.rbf_x
    hyperParameterDict['rbf_y'] = args.rbf_y
    hyperParameterDict['rbf_z'] = args.rbf_z
    hyperParameterDict['windowFunction'] =  args.windowFunction
    hyperParameterDict['liLoss'] = 'yes' if args.li else 'no'
    hyperParameterDict['initialLR'] = args.initialLR
    hyperParameterDict['finalLR'] = args.finalLR
    hyperParameterDict['lrStep'] = args.lrStep
    hyperParameterDict['lrSteps'] = lrSteps
    hyperParameterDict['gamma'] = gamma
    hyperParameterDict['epochs'] = args.epochs
    hyperParameterDict['iterations'] = args.iterations
    hyperParameterDict['totalIterations'] = totalIterations    
    hyperParameterDict['arch'] =  args.arch
    hyperParameterDict['seed'] =  args.seed
    hyperParameterDict['augmentAngle'] =  args.augmentAngle
    hyperParameterDict['augmentJitter'] =  args.augmentJitter
    hyperParameterDict['jitterAmount'] =  args.jitterAmount
    hyperParameterDict['networkSeed'] =  args.networkseed
    hyperParameterDict['network'] = args.network
    hyperParameterDict['normalized'] = args.normalized
    hyperParameterDict['adjustForFrameDistance'] = args.adjustForFrameDistance
    hyperParameterDict['fluidFeatures'] = features.shape[1]
    hyperParameterDict['boundaryFeatures'] = 0
    hyperParameterDict['cutlassBatchSize'] = args.cutlassBatchSize
    hyperParameterDict['normalized'] = args.normalized
    hyperParameterDict['weight_decay'] = args.weight_decay
    hyperParameterDict['input'] = args.input
    hyperParameterDict['output'] = args.output
    hyperParameterDict['iterations'] = args.iterations
    hyperParameterDict['trainingFiles'] = list(inFile['simulationData'].keys())

    hyperParameterDict['target'] = args.target
    hyperParameterDict['features'] = args.features.split(' ')

    hyperParameterDict['liLoss'] = False

    widths = hyperParameterDict['arch'].strip().split(' ')
    layers = [int(s) for s in widths]
    # debugPrint(layers)
    if args.verbose:
        print('Building Network')
    setSeeds(hyperParameterDict['networkSeed'], verbose = args.verbose)
    
    model = None
    if args.network == 'default':
        model = RbfNet(hyperParameterDict['fluidFeatures'], hyperParameterDict['boundaryFeatures'], layers = layers, 
                       coordinateMapping = hyperParameterDict['coordinateMapping'], dims = [hyperParameterDict['nx'], hyperParameterDict['ny'], hyperParameterDict['nz']], windowFn = getWindowFunction(hyperParameterDict['windowFunction']), 
                       rbfs = [hyperParameterDict['rbf_x'], hyperParameterDict['rbf_y'], hyperParameterDict['rbf_z']], batchSize = hyperParameterDict['cutlassBatchSize'], normalized = hyperParameterDict['normalized'])

    optimizer = Adam(model.parameters(), lr=hyperParameterDict['initialLR'], weight_decay = hyperParameterDict['weight_decay'])
    model = model.to(device)

    optimizer.zero_grad()
    model = model.train()
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma)
    
    hyperParameterDict['parameterCount'] = count_parameters(model)
    hyperParameterDict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hyperParameterDict['networkPrefix'] = hyperParameterDict['network']
    hyperParameterDict['exportString'] = '%s - n=[%2d,%2d,%2d] rbf=[%s,%s,%s] map = %s window = %s e = %2d arch %s - %s seed %s' % (
        hyperParameterDict['networkPrefix'], hyperParameterDict['nx'], hyperParameterDict['ny'], hyperParameterDict['nz'], hyperParameterDict['rbf_x'], hyperParameterDict['rbf_y'], hyperParameterDict['rbf_z'], hyperParameterDict['coordinateMapping'], 
        hyperParameterDict['windowFunction'], hyperParameterDict['epochs'], 
        hyperParameterDict['arch'], hyperParameterDict['timestamp'], hyperParameterDict['networkSeed'])
    hyperParameterDict['shortLabel'] = '%14s [%14s] - %s -> [%16s, %16s, %16s] x [%2d, %2d, %2d] @ %2s ' % (
        hyperParameterDict['windowFunction'], hyperParameterDict['arch'], hyperParameterDict['coordinateMapping'], 
        hyperParameterDict['rbf_x'], hyperParameterDict['rbf_y'], hyperParameterDict['rbf_z'], hyperParameterDict['nx'], 
        hyperParameterDict['ny'], hyperParameterDict['nz'],hyperParameterDict['networkSeed'])
    if args.verbose:
        for k in hyperParameterDict.keys():
            print('%14s = ' % k, hyperParameterDict[k] if k!='trainingFiles'else '...')

    return model, optimizer, scheduler, hyperParameterDict, parameterDict, xx, yy, zz, gridPositions, generatedData
def plotPredictionWithLossCurve(model, trainingEpochLosses, inFile, hyperParameterDict, generatedData):
    bdata = [list(inFile['simulationData'].keys())[0]]
    loss, features, prediction, gt = processBatch(bdata, model, inFile, features = hyperParameterDict['features'], target = hyperParameterDict['target'], li = hyperParameterDict['liLoss'] == 'yes', dataOverride = [generatedData])
    batchedLoss = torch.mean(loss, dim = -1)
    loss = torch.mean(loss)
    losses = np.vstack(trainingEpochLosses)
#     print('batchedLoss:', batchedLoss)

    fig = plt.figure(figsize=(14,5))
    ax = fig.add_subplot(1, 4, 1)
    ax.plot(np.mean(losses, axis = -1))
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_title('Training Loss')


    ax = fig.add_subplot(1, 4, 2, projection='3d')
    sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = gt.detach().cpu(), s = 32)
    # fig.colorbar(sc, ax=ax)
    fig.colorbar(sc, ax=ax,orientation = 'horizontal')
    ax.set_box_aspect([1,1,1]) 
    ax.set_title('Ground Truth Density')

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = prediction.detach().cpu(), s = 32)
    # fig.colorbar(sc, ax=ax)
    ax.set_box_aspect([1,1,1]) 
    fig.colorbar(sc, ax=ax,orientation = 'horizontal')
    ax.set_title('Predicted Density')

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = ((gt-prediction)**2).detach().cpu(), s = 32)
    fig.colorbar(sc, ax=ax,orientation = 'horizontal')
    ax.set_box_aspect([1,1,1]) 
    ax.set_title('Loss')

    fig.subplots_adjust(
        top=0.981,
        bottom=0.049,
        left=0.042,
        right=0.981,
        hspace=0.2,
        wspace=0.2
    )
    # fig.tight_layout()
    

import pandas
from noise.generator import generateOctaveNoise, generateSimplex, generatePerlin

def getParameterDict(inFile):
    nx = inFile.attrs['nx']
    ny = inFile.attrs['ny']
    nz = inFile.attrs['nz']
    support = inFile.attrs['support']
    volume = inFile.attrs['volume']
    maxDomain = inFile.attrs['maxDomain']
    minDomain = inFile.attrs['minDomain']
    numNeighbors = inFile.attrs['numNeighbors']

    
    parameterDict = {
        'nx' : nx, 'ny' : ny, 'nz': nz,
        'support': support, 'minDomain': minDomain, 'maxDomain': maxDomain,
        'numNeighbors': numNeighbors, 'dx': 2 / nx, 'volume': volume,
        'simplexFrequency': 0.1, 'simplexScale': inFile.attrs['simplexScale'], 'jitterMean': inFile.attrs['jitterMean'] ,'jitterAmount': inFile.attrs['jitterAmount']
    }
    return parameterDict

def generateTestingDataset(inFile, plot = False):
    # manually generated seeds by hitting the numpad randomly
    seeds = [894263481,918464356,319562498,2741982453,89123968,32898147,49325386124,39818259468124,595381246,3912845398145,2512459812,463236,23874594,1563928,519984,2912741]
    frequencies = [2**np.random.default_rng(s).integers(low = 0, high = 2) for s in seeds]
    parameterDict = getParameterDict(inFile)
    gridPositions, xx, yy, zz = generateGrid(parameterDict['nx'], parameterDict['ny'], parameterDict['nz'])
    testingData = [genData(xx,yy,zz,gridPositions, s, parameterDict, simplexFrequency = f) for s,f in zip(seeds, frequencies)]

    if plot:
        fig = plt.figure(figsize=(10,10))
        for i in range(len(seeds)):
            data = testingData[i]
            ax = fig.add_subplot(4, 4, i + 1, projection='3d')
            sc = ax.scatter(data['x'][:,0], data['x'][:,1], data['x'][:,2], c = data['rho'], s = 32)
            ax.set_box_aspect([1,1,1]) 
            ax.set_title('%d' % seeds[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        #     break
        fig.subplots_adjust(
            top=0.95,
            bottom=0.05,
            left=0.05,
            right=0.95,
            hspace=0.2,
            wspace=0.2
        )
    return testingData, seeds
    
def trainNetworkUsingArgs(args, inFile, plotPreTrain = False, plotPostTrain = True, writeData = False, dataOverride = None, testingData = None):
    model, optimizer, scheduler, hyperParameterDict, parameterDict, _, _, _, _, _ = initModel(args, inFile)


    parameterDict['simplexFrequency'] = 0.1
    parameterDict['simplexScale'] = 1.05/2
    parameterDict['jitterMean'] = 0.0
    parameterDict['jitterAmount'] = 0.005

    gridPositions, xx, yy, zz = generateGrid(parameterDict['nx'], parameterDict['ny'], parameterDict['nz'])
    generatedData = genData(xx,yy,zz,gridPositions, 1234567, parameterDict)
    if plotPreTrain:
        plotPredictionWithoutLossCurve(model, inFile, hyperParameterDict, generatedData)
    gtqdms, trainingEpochLosses = initializeTraining(args, hyperParameterDict,writeData)
    
    trainingEpochLosses = traininingLoop(model, optimizer, scheduler, hyperParameterDict, inFile, train_dataloader, train_iter, gtqdms, args, writeData, dataOverride = dataOverride)
    if plotPostTrain:
        plotPredictionWithLossCurve(model, trainingEpochLosses, inFile, hyperParameterDict, generatedData)
    
    
    inferenceModel = model.train(False)

    testDataset = pandas.DataFrame()
    if testingData is not None:
        for i, t in enumerate(testingData):
            with torch.no_grad():
                loss, features, prediction, gt = processBatch(bdata, inferenceModel, inFile, features = hyperParameterDict['features'], target = hyperParameterDict['target'], li = hyperParameterDict['liLoss'] == 'yes', dataOverride = [t])
        #         batchedLoss = torch.mean(loss, dim = -1)
                meanLoss = torch.mean(loss)
        #     print(meanLoss, loss)
        #     df = pandas.DataFrame({'seed': seeds[i], 'meanLoss': meanLoss.item(), 'loss': pandas.Series(loss.cpu().numpy()), 'prediction': prediction.cpu().numpy(), 'gt': gt.cpu().numpy()}, index = [1])
            seeds = [894263481,918464356,319562498,2741982453,89123968,32898147,49325386124,39818259468124,595381246,3912845398145,2512459812,463236,23874594,1563928,519984,2912741]
    
            df = pandas.DataFrame({'base': hyperParameterDict['rbf_x'], 'n': hyperParameterDict['nx'], 'arch': hyperParameterDict['arch'], 'map': hyperParameterDict['coordinateMapping'], 'window': hyperParameterDict['windowFunction'], 'seed': seeds[i], 'networkSeed': hyperParameterDict['networkSeed'],
                                   'meanLoss': meanLoss.item(), 'loss': [loss.cpu().numpy().flatten()], 'prediction': [prediction.cpu().numpy().flatten()], 'gt': [gt.cpu().numpy().flatten()], 'features': args.features, 'target': args.target, 'mapping': args.coordinateMapping, 'window':args.windowFunction}, index = [1])
            testDataset = pandas.concat((df, testDataset), ignore_index = True)
    gtqdms[args.gpu].close()
    gtqdms[args.gpu + args.gpus].close()
    return model, optimizer, scheduler, hyperParameterDict, parameterDict, trainingEpochLosses, generatedData, testDataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

testingData, seeds = generateTestingDataset(inFile)

ns = [1,2,3,4,5,6,7,8,16]
# ns = [6]
bases = ['linear', 'rbf square', 'ffourier', 'fourier even', 'fourier odd']
# bases = ['ffourier']
mappings = ['cartesian']
seeds = [ 409567468, 230187756, 78465423, 39981938]
seeds = [ 409567468, 230187756, 78465423, 39981938]
seeds = [ 409567468, 230187756]
# targets = ['rho']
targets = ['rho', 'gradRhoNaive']
# features = ['one', 'volume', 'normalizedVolume', 'rho', 'rhoTimesVolume', 'rhoTimesNormalizedVolume', 'rhoOverVolume', 'rhoOverNormalizedVolume', 'volumeOverRho', 'normalizedVolumeOverRho']
features = ['normalizedVolume']

ablationStudy = pandas.DataFrame()
    
args.iterations = 1000
for seed in tqdm(seeds, leave = False):
    for n in tqdm(ns):
        for base in tqdm(bases, leave = False):
            for cmap in tqdm(mappings, leave = False):
                for target in tqdm(targets, leave = False):
                    for feature in tqdm(features, leave = False):
                        args.nx = args.ny = args.nz = n
                        args.rbf_x = args.rbf_y = args.rbf_z = base
                        if 'grad' in target:
                            args.arch[-1] = '3'
                        else:
                            args.arch[-1] = '1'
                        args.target = target
                        args.features = feature
                        args.coordinateMapping = cmap
                        args.networkseed = seed
                        args.seed = seed
                        model, optimizer, scheduler, hyperParameterDict, parameterDict, trainingEpochLosses, generatedData, testDataset = trainNetworkUsingArgs(args,inFile, dataOverride = None, testingData = testingData, plotPostTrain = False)
                        ablationStudy = pandas.concat((ablationStudy, testDataset), ignore_index = True)
                        ablationStudy.to_csv('ablationStudyBseFunctions.csv')
