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
from tqdm.autonotebook import tqdm
# from IPython.display import display, Latex
# from datetime import datetime
# from BasisConvolution.oneDimensionalSPH.rbfNet import *
# from BasisConvolution.convNet import RbfNet
# import h5py
# import matplotlib.colors as colors
import torch
import torch.nn as nn
# %matplotlib notebook

from .util import getGroundTruthKernel, getGroundTruthKernelGradient, getGroundTruthPhysics
from .util import getFeaturesKernel, getFeaturesPhysics, lossFunction
import pandas
from BasisConvolution.test_case_I.train import trainModel
from BasisConvolution.test_case_I.eval import getTestingLossFrame
from BasisConvolution.test_case_I.dataset import loadTestcase

def runAblationStudyMLPOneLayer(basis, testCase, hiddenLayout, widths, depths, messages, seeds, device, offset, particleSupport, trainingFiles, trainingData, dataSet, testingFiles, testingData, settings,):
    global testData
    if testCase == 'kernel':
        groundTruthFn = getGroundTruthKernel
        featureFn = getFeaturesKernel
        inputFeatures = 1
    elif testCase == 'kernelGradient':
        groundTruthFn = getGroundTruthKernelGradient   
        featureFn = getFeaturesKernel 
        inputFeatures = 1
    elif testCase == 'physicsUpdate':
        groundTruthFn = getGroundTruthPhysics
        featureFn = getFeaturesPhysics
        inputFeatures = 2

    testData = {}
    for i in range(len(testingFiles)):
        testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset, particleSupport)
        
    layouts = []
    for d in depths:
        for w in widths:
            l = [w] * d
            if l not in layouts:
                layouts.append(l)
#     print(len(messages) * len(seeds) * len(layouts))

    dataset = pandas.DataFrame()
    for l in tqdm(layouts, leave = False):
        for m in tqdm(messages, leave = False):
            for s in tqdm(seeds, leave = False):
                if basis == 'PointNet':
                    net = buildPointNet(inputFeatures, feedThroughVertexFeatures = 1, vertexFeatures = l, vertexMLPLayout = hiddenLayout, seed = s)
                if basis == 'DPCConv':
                    net = buildMessagePassingNetwork(inputFeatures, feedThroughEdgeFeatures = 0, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = False, edgeFeatures = [1], seed = s)
                if basis == 'GNS':
                    net = buildGNS(inputFeatures, feedThroughVertexFeatures = 0, feedThroughEdgeFeatures = 0, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = True, edgeFeatures = [1] + l, messageFeatures = [m], seed = s)
                if basis == 'MP-PDE':
                    net = buildGNS(inputFeatures, feedThroughVertexFeatures = 1, feedThroughEdgeFeatures = 1, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = True, edgeFeatures = [1] + l, messageFeatures = [m], seed = s)
                net = net.to(device)

                modelstate = trainMLP(trainingData, testData, settings, dataSet, trainingFiles, offset, net,
                    epochs = 5, iterations = 1000, initialLR = 1e-3, device = device, testInterval = 100,
                    groundTruthFn = groundTruthFn, featureFn = featureFn, lossFn = lossFunction, particleSupport = particleSupport)  
                net.train(False)
                df = getTestingLossFrameMLP(net, testData)

                dataset = pandas.concat([dataset, df])


                # dataset.to_csv('ablationStudy_%s_%s ws %s ds %s seeds %s messsages %s.csv' % (testCase, basis, '[' + ' '.join([str(d) for d in widths]) + ']', '[' + ' '.join([str(d) for d in depths]) + ']', '[' + ' '.join([str(d) for d in seeds]) + ']','[' + ' '.join([str(d) for d in messages]) + ']'))
    return dataset


def runAblationStudyMLP(basis, testCase, hiddenLayout, widths, depths, messages,device,offset, particleSupport ):
    global testData
    # basis = 'MP-PDE'
    # testCase = 'physicsUpdate'
    if testCase == 'kernel':
        groundTruthFn = getGroundTruthKernel
        featureFn = getFeaturesKernel
    elif testCase == 'kernelGradient':
        groundTruthFn = getGroundTruthKernelGradient   
        featureFn = getFeaturesKernel 
    elif testCase == 'physicsUpdate':
        groundTruthFn = getGroundTruthPhysics
        featureFn = getFeaturesPhysics

    testData = {}
    for i in range(len(testingFiles)):
        testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset, particleSupport, offset)

    layouts = []
    for d in depths:
        for w in widths:
            l = [w] * d
            if l not in layouts:
                layouts.append(l)
    #     print(len(messages) * len(seeds) * len(layouts))

    # print(layouts)
    l = layouts[1]
    l = layouts[-1]
    # hiddenLayout = [32] * 3
    m = [8] * (1 + len(l))
    # print(m)
    # s = 12345


    dataset = pandas.DataFrame()
    for l in tqdm(layouts, leave = False):
        for m in tqdm(messages, leave = False):
            for s in tqdm(seeds, leave = False):
                if basis == 'PointNet':
                    net = buildPointNet(3 if testCase == 'physicsUpdat' else 1, feedThroughVertexFeatures = 1, vertexFeatures = l, vertexMLPLayout = hiddenLayout, seed = s)
                if basis == 'DPCConv':
                    net = buildMessagePassingNetwork(3 if testCase == 'physicsUpdat' else 1, feedThroughEdgeFeatures = 0, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = False, edgeFeatures = [1], seed = s)
                if basis == 'GNS':
                    net = buildGNS(3 if testCase == 'physicsUpdat' else 1, feedThroughVertexFeatures = 0, feedThroughEdgeFeatures = 0, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = True, edgeFeatures = [1] + l, messageFeatures = [m], seed = s)
                if basis == 'MP-PDE':
                    net = buildGNS(3 if testCase == 'physicsUpdat' else 1, feedThroughVertexFeatures = 1, feedThroughEdgeFeatures = 1, vertexFeatures = l, vertexMLPLayout = hiddenLayout, edgeMLP = True, edgeFeatures = [1] + l, messageFeatures = [m], seed = s)
                net = net.to(device)
                modelstate = trainMLP(particleData, testData, settings, dataSet, trainingFiles, offset, net,
                    epochs = 5, iterations = 1000, initialLR = 1e-3, device = device, testInterval = 100,
                    groundTruthFn = groundTruthFn, featureFn = featureFn, lossFn = lossFunction, particleSupport = particleSupport)  
                net.train(False)
                df = getTestingLossFrameMLP(net, testData)

                dataset = pandas.concat([dataset, df])


def trainRBFNetwork(basis, testCase, ns, widths, depths, seeds, window, device, particleSupport, trainingFiles, trainingData, dataSet, testingFiles, testingData, settings, offset):
    global testData
    if testCase == 'kernel':
        groundTruthFn = getGroundTruthKernel
        featureFn = getFeaturesKernel
    elif testCase == 'kernelGradient':
        groundTruthFn = getGroundTruthKernelGradient   
        featureFn = getFeaturesKernel 
    elif testCase == 'physicsUpdate':
        groundTruthFn = getGroundTruthPhysics
        featureFn = getFeaturesPhysics

    testData = {}
    for i in range(len(testingFiles)):
        testData['_'.join(testingFiles[i].split('/')[-1].split('.')[0].split('_')[1:3])] = loadTestcase(testingData, settings, testingFiles[i], [offset, 128, 256, 1024], device, groundTruthFn, featureFn, offset, particleSupport = particleSupport)
        
    layouts = []
    for d in depths:
        for w in widths:
            l = [w] * d + [1]
            if l not in layouts:
                layouts.append(l)
    dataset = pandas.DataFrame()

#     for basis in tqdm(bases, leave = False):
    dataset = pandas.DataFrame()
    for n in tqdm(ns, leave = False):
        for l in tqdm(layouts, leave = False):
            for s in tqdm(seeds, leave = False):
                trainedModel = trainModel(trainingData, testData, settings, dataSet, trainingFiles, fluidFeatures = 3 if testCase == 'physicsUpdate' else 1, offset = offset,
                                  n = n, basis = basis, layers = l, seed = s, particleSupport = particleSupport,
                                 window = window, windowNorm = 'integral',
                                 epochs = 5, iterations = 1000, initialLR = 1e-3, device = device, testInterval = 1000,
                                 groundTruthFn = groundTruthFn, featureFn = featureFn, lossFn = lossFunction)
        #         models.append(trainedModel)
                trainedModel['model'].train(False)
                df = getTestingLossFrame(trainedModel, testData, plot = False)
                dataset = pandas.concat([dataset, df])
    return dataset
#                 dataset.to_csv('ablationStudy_%s_%s window %s ns %s ws %s ds %s seeds %s.csv' % (testCase, basis, 'None' if window is None else window, '[' + ' '.join([str(d) for d in ns]) + ']', '[' + ' '.join([str(d) for d in widths]) + ']', '[' + ' '.join([str(d) for d in depths]) + ']', '[' + ' '.join([str(d) for d in seeds]) + ']'))

