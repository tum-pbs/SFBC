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
from torch.utils.data import DataLoader
# from BasisConvolution.oneDimensionalSPH.trainingHelper import *
# plotting/UI related imports
# from BasisConvolution.oneDimensionalSPH.plotting import *
# import matplotlib as mpl
# plt.style.use('dark_background')
# cmap = mpl.colormaps['viridis']
from tqdm.autonotebook import tqdm
# from IPython.display import display, Latex
# from datetime import datetime
# from BasisConvolution.oneDimensionalSPH.rbfNet import *
from BasisConvolution.convNetv2 import BasisNetwork
from BasisConvolution.detail.windows import getWindowFunction
# import h5py
# import matplotlib.colors as colors
# %matplotlib notebook
from BasisConvolution.test_case_I.dataset import loadBatch, flatten
from BasisConvolution.detail.radius import batchedNeighborsearch
import torch
import numpy as np
# import torch.nn as nn


def trainModel(particleData, testData, settings, dataSet, trainingFiles, offset, seed, particleSupport, fluidFeatures = 1, n = 16, basis = 'linear', layers = [1], window = None, windowNorm = None, epochs = 5, iterations = 1000, testInterval = 10, initialLR = 1e-2, groundTruthFn = None, featureFn = None, lossFn = None, device = 'cpu', weightOverride = None,batchSize = 4):   
    dataLoader = DataLoader(dataSet, shuffle=True, batch_size = batchSize).batch_sampler
#     random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    windowFn = getWindowFunction(window, norm = windowNorm) if window is not None else None
    model = BasisNetwork(fluidFeatures = fluidFeatures, 
                   layers = layers, 
                   denseLayer = True, activation = 'relu', coordinateMapping = 'cartesian', 
                   dims = [n], windowFn = windowFn, rbfs = [basis], batchSize = 32, ignoreCenter = True, normalized = False).to(device)   
    lr = initialLR
    with torch.no_grad():
        if weightOverride is not None:
            model.convs[0].weight[:,0,0] = torch.tensor(weightOverride).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    losses = []
    lossValues = []
    testLosses = {}
    testPredictions = {}
    for e in (pb := tqdm(range(epochs), leave = False)):
        for b in (pbl := tqdm(range(iterations), leave=False)):
            
            try:
                bdata = next(dataIter)
                if len(bdata) < batchSize :
                    raise Exception('batch too short')
            except:
                dataIter = iter(dataLoader)
                bdata = next(dataIter)

            positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(particleData, settings, dataSet, bdata, device, offset)
            i, j, distance, direction = batchedNeighborsearch(positions, setup)
            x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

            x = x[:,None].to(device)    
            groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction, particleSupport).to(device)
            distance = (distance * direction)[:,None].to(device)
            features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)

            optimizer.zero_grad()
            prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
            lossTerm = lossFn(prediction, groundTruth)
            loss = torch.mean(lossTerm)

            loss.backward()
            optimizer.step()
            losses.append(lossTerm.detach().cpu())
            lossValues.append(loss.detach().cpu().item())

            lossString = np.array2string(torch.mean(lossTerm.reshape(batchSize, positions[0].shape[0]),dim=1).detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.4e" % x})
            batchString = str(np.array2string(np.array(bdata), formatter={'float_kind':lambda x: "%.2f" % x, 'int':lambda x:'%6d' % x}))

            pbl.set_description('%s:  %s -> %.4e' %(batchString, lossString, loss.detach().cpu().numpy()))
            pb.set_description('epoch %2dd, lr %6.4g: loss %.4e [rolling %.4e]' % (e, lr, np.mean(lossValues), np.mean(lossValues[-100:] if len(lossValues) > 100 else lossValues)))
            
            it = e * iterations + b
            if it % testInterval == 0:
                with torch.no_grad():
                    testLossDict = {}
                    testPreds = {}
                    for i, k in  enumerate(testData.keys()):
                        gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
                        prediction = model(testData[k][5], testData[k][6], testData[k][7], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0])
                        arr = []
                        for i, (xs, pred, gt) in enumerate(zip(testData[k][0], prediction, gt)):
                             arr.append(lossFn(pred, gt).detach().cpu().numpy())
                        testLossDict[k] = arr
                        testPreds[k] = prediction.detach().cpu()
#                         print(testLossDict[k])
                    testLosses[it] = testLossDict
                    testPredictions[it] = testPreds
#             break
#         break
        
        lr = lr * 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
            
    return {'model': model, 'optimizer': optimizer, 'finalLR': lr, 'losses': losses, 'testLosses': testLosses, 'testPredictions': testPredictions, 'seed': seed,
            'window': window, 'windowNorm': windowNorm, 'n':n, 'basis':basis, 'layers':layers, 'epochs': epochs, 'iterations': iterations, 'fluidFeatures': fluidFeatures          
           }