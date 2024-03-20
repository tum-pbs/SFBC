import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

import time
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange, tqdm
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


import tomli
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as mticker

# from cutlass import *
# from rbfConv import *
from tqdm.autonotebook import tqdm
from BasisConvolution.detail.cutlass import cutlass
from torch.nn.parameter import Parameter
import numpy as np
from BasisConvolution.convNet import RbfNet

device = 'cuda'
dtype = torch.float32

def generateParticles(nx, dim = 1):
    dx = 2 / nx
#     nx = int(numParticles ** (1/dim))
#     print(nx)
    x = torch.linspace(-1 + dx / 2, 1 - dx/2, nx, device = device, dtype = dtype)
    y = torch.linspace(-1 + dx / 2, 1 - dx/2, nx, device = device, dtype = dtype)
    z = torch.linspace(-1 + dx / 2, 1 - dx/2, nx, device = device, dtype = dtype)
    if dim == 1:
        return x[:,None], dx
    if dim == 2:
        xx,yy = torch.meshgrid(x,y, indexing = 'xy')
        return torch.stack((xx,yy), axis = -1).flatten().reshape((-1,2)), dx
    if dim == 3:
        xx,yy,zz = torch.meshgrid(x,y,z, indexing = 'xy')
        return torch.stack((xx,yy,zz), axis = -1).flatten().reshape((-1,3)), dx

@torch.jit.script
def supportFromVolume(v, dim: int, numNeighbors: int):
#     print(v,dim,numNeighbors)
    if dim == 1:
        return (numNeighbors / 2) * v
    if dim == 2: 
        return (numNeighbors * v / np.pi) ** (1/2)
    if dim == 3:
        return (numNeighbors * v *3 / (4 * np.pi))**(1/3)
    return torch.ones_like(v) * np.nan

def neighSearch(x, y, support, max_num_neighbors):
    j,i = radius(x, y, support, max_num_neighbors = numNeighbors * 4)
    fluidDistances = y[i] - x[j]
    fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

    fluidDistances[fluidRadialDistances < 1e-4 * support,:] = 0
    fluidDistances[fluidRadialDistances >= 1e-4 * support,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-4 * support,None]
    fluidRadialDistances /= support
    return i, j, fluidDistances, fluidRadialDistances
def batchedNeighSearch(positions, h, numNeighbors):
    neighborLists = [neighSearch(xs, xs, h[0], max_num_neighbors = numNeighbors * 4) for xs in positions]
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

def stepGPUwNeighborhood(convNet, optimizer, xstar, h, numNeighbors, f, gt):    
    step_start = torch.cuda.Event(enable_timing=True)
    neigh_start = torch.cuda.Event(enable_timing=True)
    prediction_start = torch.cuda.Event(enable_timing=True)
    loss_start = torch.cuda.Event(enable_timing=True)
    backwards_start = torch.cuda.Event(enable_timing=True)  
    
    step_end = torch.cuda.Event(enable_timing=True)
    neigh_end = torch.cuda.Event(enable_timing=True)
    prediction_end = torch.cuda.Event(enable_timing=True)
    loss_end = torch.cuda.Event(enable_timing=True)
    backwards_end = torch.cuda.Event(enable_timing=True)
    
    
    step_start.record()
    optimizer.zero_grad()
    neigh_start.record()
    fi, fj, rij, direction = batchedNeighSearch(xstar, h[0], numNeighbors)
    neigh_end.record()
    
    prediction_start.record()
    prediction = convNet(f, fi, fj, direction * rij[:,None], None, None, None)
    prediction_end.record()
    
    loss_start.record()
    loss = (prediction - gt)**2
    meanLoss = torch.mean(loss)
    loss_end.record()
    
    backwards_start.record()
    meanLoss.backward()
    backwards_end.record()
    
    step_end.record()
    torch.cuda.synchronize()
    
    return step_start.elapsed_time(step_end), neigh_start.elapsed_time(neigh_end), prediction_start.elapsed_time(prediction_end), loss_start.elapsed_time(loss_end), backwards_start.elapsed_time(backwards_end)
def stepGPU(convNet, optimizer, xstar, h, numNeighbors, f, gt, fi, fj, rij, direction):    
    step_start = torch.cuda.Event(enable_timing=True)
    prediction_start = torch.cuda.Event(enable_timing=True)
    loss_start = torch.cuda.Event(enable_timing=True)
    backwards_start = torch.cuda.Event(enable_timing=True)  
    
    step_end = torch.cuda.Event(enable_timing=True)
    prediction_end = torch.cuda.Event(enable_timing=True)
    loss_end = torch.cuda.Event(enable_timing=True)
    backwards_end = torch.cuda.Event(enable_timing=True)
    
    
    step_start.record()
    optimizer.zero_grad()
    
    prediction_start.record()
    prediction = convNet(f, fi, fj, direction * rij[:,None], None, None, None)
    prediction_end.record()
    
    loss_start.record()
    loss = (prediction - gt)**2
    meanLoss = torch.mean(loss)
    loss_end.record()
    
    backwards_start.record()
    meanLoss.backward()
    backwards_end.record()
    
    step_end.record()
    torch.cuda.synchronize()
    
    return step_start.elapsed_time(step_end), prediction_start.elapsed_time(prediction_end), loss_start.elapsed_time(loss_end), backwards_start.elapsed_time(backwards_end)

def stepCPUwNeighborhood(convNet, optimizer, xstar, h, numNeighbors, f, gtn):    
    step_start = time.time()
    optimizer.zero_grad()

    neigh_start = time.time()
    fi, fj, rij, direction = batchedNeighSearch(xstar, h[0], numNeighbors)
    neigh_end = time.time()
    
    prediction_start = time.time()
    prediction = convNet(f, fi, fj, direction * rij[:,None], None, None, None)
    prediction_end = time.time()
    
    loss_start = time.time()
    loss = (prediction - gt)**2
    meanLoss = torch.mean(loss)
    loss_end = time.time()
    
    backwards_start = time.time()
    meanLoss.backward()
    backwards_end = time.time()
    
    step_end = time.time()
    
    return step_end - step_start, neigh_end - neigh_start, prediction_end - prediction_start, loss_end - loss_start, backwards_end - backwards_start

def stepCPU(convNet, optimizer, xstar, h, numNeighbors, f, gt, fi, fj, rij, direction):    
    step_start = time.time()
    optimizer.zero_grad()
    
    prediction_start = time.time()
    prediction = convNet(f, fi, fj, direction * rij[:,None], None, None, None)
    prediction_end = time.time()
    
    loss_start = time.time()
    loss = (prediction - gt)**2
    meanLoss = torch.mean(loss)
    loss_end = time.time()
    
    backwards_start = time.time()
    meanLoss.backward()
    backwards_end = time.time()
    
    step_end = time.time()
    
    return step_end - step_start, prediction_end - prediction_start, loss_end - loss_start, backwards_end - backwards_start


import pandas as pd
from BasisConvolution.detail.windows import getWindowFunction

def getBasisLabel(b):
    if b == 'fourier even':
        return 'Fourier (even)'
    if b == 'fourier odd':
        return 'Fourier (odd)'
    if b == 'fourier odd lin':
        return 'Fourier (odd) + x'
    if b == 'fourier odd sgn':
        return 'Fourier (odd) + sgn(x)'
    if b == 'fourier':
        return 'SFBC'
    if b == 'ffourier':
        return 'Fourier (4-Terms)'
    if b == 'ffourier 5':
        return 'Fourier (5-Terms)'
    if b == 'linear':
        return 'LinCConv'
    if b == 'rbf linear':
        return 'LinCConv'
    if b == 'abf cubic_spline':
        return 'SplineConv'
    if b == 'dmcf':
        return 'DMCF'
    if b == 'rbf square':
        return 'Nearest Neighbor'
    if b == 'chebyshev':
        return 'Chebyshev'
    if b == 'chebyshev 2':
        return 'Chebyshev (2nd kind)'
    if b == 'ubf cubic_spline':
        return 'Normalized SplineConv'
    if b == 'ubf quartic_spline':
        return 'Normalized Quartic Spline'
    if b == 'abf bump':
        return 'Bump RBF'
    if b == 'rbf linear':
        return 'CConv'
    if b == 'ubf wendland2':
        return 'Normalized Wendland-2'
    if b == 'ubf poly6':
        return 'Normalized Müller Kernel'
    if b == 'ubf gaussian':
        return 'Normalized Gaussian Kernel'
    if b == 'abf cubic_spline':
        return 'SplineConv'
    if b == 'abf gaussian':
        return 'Gaussian Kernel'
    if b == 'abf quartic_spline':
        return 'Quartic Spline'
    if b == 'abf poly6':
        return 'Müller Kernel'
    if b == 'abf wendland2':
        return 'Wendland-2'
    if b == 'rbf spiky':
        return 'Spiky'
    print('unknown basis function', b)
def getWindowLabel(w):
    if w == 'poly6':
        return 'Spiky'
    if w == 'Spiky':
        return 'Spiky Kernel'
    if w == 'Parabola':
        return r'$1-x^2$'
    if w == 'cubicSpline':
        return 'Cubic Spline'
    if w == 'quarticSpline':
        return 'Quartic Spline'
    if w == 'Linear':
        return r'$1-x$'
    if w == 'None':
        return 'None'
    if w == 'Mueller':
        return 'Müller'
    print('unknown window function', w)
    if np.isnan(w):
        return 'None'
    
def getMapLabel(w):
    if w == 'cartesian':
        return 'Cartesian'
    if w == 'polar':
        return 'Polar'
    if w == 'preserving':
        return 'Ummenhofer et al.'
    print('unknown mapping function', w)
    
import copy
def pivotDS(dataset, metrics = None):
    baseData = pd.DataFrame()
    for k in dataset.keys():
        if k not in metrics:
            baseData[k] = dataset[k]
# baseData['label'] = dataset[['rbf_x','n','map']].apply(lambda x: getBasisLabel(x[0])  + " x " + str(x[1]) + ' @ ' + getMapLabel(x[2]), axis = 1)
# baseData['Basis'] = dataset['rbf_x'].apply(getBasisLabel)
# baseData['map'] = dataset['map'].apply(getMapLabel)
# baseData['n'] = dataset['n']
# baseData['seed']= dataset['seed']
# baseData['window']= dataset['window'].apply(getWindowLabel)
# baseData['Configuration'] = baseData[['window','map']].apply(lambda x: 'Window: ' + x[0] + ' @ Map: ' + x[1], axis = 1)
# baseData['arch']=dataset['arch']
# baseData['testFile']= dataset['testFile']
# baseData['initialFrame']= dataset['initialFrame']
# baseData['unrollStep']= dataset['unrollStep']

    processedData = pd.DataFrame()
    for metric in tqdm(metrics):
        tempData = copy.deepcopy(baseData)
        tempData['metric'] = metric
        tempData['value'] = dataset[metric]
        processedData = pd.concat((processedData, tempData), ignore_index = True)
    return processedData

def benchmarkNetwork(nx, dim, numNeighbors, fluidFeatures, cmap, rbf, n, batch_size, arch, window_fn, performanceSamples = 256, benchmarkNeighSearch = False, seed = 12345):
    torch.manual_seed = seed
    x, dx = generateParticles(nx, dim = dim)
    v = torch.ones_like(x) * dx**dim
    h = supportFromVolume(v, dim, 32)
#     print(x.shape)
    jitter = [torch.normal(torch.zeros_like(x), torch.ones_like(x) * h * jitterAmount).type(dtype).to(device) for b in range(batch_size)]
    f = torch.vstack([torch.normal(torch.zeros([x.shape[0], fluidFeatures]), torch.ones([x.shape[0], fluidFeatures])).type(dtype).to(device) for b in range(batch_size)])
    gt = torch.vstack([torch.normal(torch.zeros([x.shape[0], 1]), torch.ones([x.shape[0], 1])).type(dtype).to(device) for b in range(batch_size)])
    xstar = [x + j for j in jitter]

    convNet = RbfNet(fluidFeatures, 0, layers = arch, coordinateMapping = cmap, dims = [n] * dim, rbfs = [rbf] * dim, batchSize = 32, ignoreCenter = True, normalized = False, windowFn = getWindowFunction(window_fn)).to(device)
    optimizer = Adam(convNet.parameters(), lr=1e-2, weight_decay = 0)
    performanceDataset = pd.DataFrame()
    
    neigh_start = time.time()
    fi, fj, rij, direction = batchedNeighSearch(xstar, h[0], numNeighbors)
    neigh_end = time.time()
    neighTime = neigh_end - neigh_start
    
    for i in tqdm(range(performanceSamples), leave = False):
        if benchmarkNeighSearch:
            if device == 'cpu':
                stepTime, neighTime, predTime, lossTime, backwardsTime = stepCPUwNeighborhood(convNet, optimizer, xstar, h, numNeighbors, f, gt)
            else:
                stepTime, neighTime, predTime, lossTime, backwardsTime = stepGPUwNeighborhood(convNet, optimizer, xstar, h, numNeighbors, f, gt)
        else:
            if device == 'cpu':
                stepTime, predTime, lossTime, backwardsTime = stepCPU(convNet, optimizer, xstar, h, numNeighbors, f, gt,fi,fj,rij,direction)
            else:
                stepTime, predTime, lossTime, backwardsTime = stepGPU(convNet, optimizer, xstar, h, numNeighbors, f, gt,fi,fj,rij,direction)
        frame = pd.DataFrame({
            'Sample': i,
            'Number of Particles': nx**dim,
            'Dimensionality': dim,
            'Neighborhood Size': numNeighbors,
            'Fluid Features': fluidFeatures,
            'Coordinate Mapping': cmap,
            'Window Function': window_fn,
            'Base Function': rbf,
            'Terms':n,
            'Batch Size': batch_size,
            'Network Architecture':'[%s]'% (', '.join([str(a) for a in arch])),
            'Gradient Update': stepTime,
            'Neighborsearch': neighTime,
            'Forward Pass': predTime,
            'Loss Computation': lossTime,
            'Backward Pass': backwardsTime,        
        }, [0])
        performanceDataset = pd.concat((performanceDataset, frame), ignore_index=True)
    return performanceDataset

# nx = 16
# dim = 3
jitterAmount = 0.05
numNeighbors = 32
fluidFeatures = 1
# cmap = 'cartesian'
# rbf = 'linear'
# n = 4
# arch = [1]
# batch_size = 4
performanceSamples = 96
# window_fn = 'None'



overallDataset = pd.DataFrame()
for arch in tqdm([[1], [32, 1], [32, 32, 1]]):
    for [nx, dim] in tqdm([[4096, 1], [64, 2], [16, 3]], leave = False):
        performanceDataset = pd.DataFrame()
        for n in tqdm([1,2,3,4,5,6,7,8,16,32], leave = False):
            for batch_size in tqdm([2], leave = False):
                for window_fn in tqdm(['None', 'Mueller'], leave = False):
                    for cmap in tqdm(['cartesian', 'polar', 'preserving'] if dim != 1 else ['cartesian'], leave = False):
                        for rbf in tqdm(['linear', 'rbf square', 'chebyshev', 'fourier', 'dmcf'], leave = False):
                            frame = benchmarkNetwork(nx, dim, numNeighbors, fluidFeatures, cmap, rbf, n, batch_size, arch, window_fn, performanceSamples)
                            performanceDataset = pd.concat((performanceDataset, frame), ignore_index = True)
                            performanceDataset.to_csv('performance v2 [nx = %4d, dim = %d, arch = %s].csv' % (nx, dim, arch))

                            overallDataset = pd.concat((overallDataset, frame), ignore_index = True)
                            overallDataset.to_csv('performance v2.csv')



# display(performanceDataset)
# performanceDataset['Base Function'] =  performanceDataset['Base Function'].apply(getBasisLabel)
# performanceDataset['Coordinate Mapping'] =  performanceDataset['Coordinate Mapping'].apply(getMapLabel)
# performanceDataset['Window Function'] =  performanceDataset['Window Function'].apply(getWindowLabel)