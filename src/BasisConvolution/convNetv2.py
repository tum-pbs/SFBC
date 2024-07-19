# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import inspect
# import re
# def debugPrint(x):
#     frame = inspect.currentframe().f_back
#     s = inspect.getframeinfo(frame).code_context[0]
#     r = re.search(r"\((.*)\)", s).group(1)
#     print("{} [{}] = {}".format(r,type(x).__name__, x))
import torch
# from torch_geometric.loader import DataLoader
# import argparse
# from BasisConvolution.detail.radius import radius
# from torch.optim import Adam
import copy
import torch
# from torch_geometric.loader import DataLoader
# import argparse
# from BasisConvolution.detail.radius import radius
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# import portalocker
# import seaborn as sns
import torch
import torch.nn as nn



def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationLayer(function: str):
    if function == 'elu':
        return nn.ELU()
    elif function == 'relu':
        return nn.ReLU()
    elif function == 'hardtanh':
        return nn.Hardtanh()
    elif function == 'hardswish':
        return nn.Hardswish()
    elif function == 'selu':
        return nn.SELU()
    elif function == 'celu':
        return nn.CELU()
    elif function == 'leaky_relu':
        return nn.LeakyReLU()
    elif function == 'prelu':
        return nn.PReLU()
    elif function == 'rrelu':
        return nn.RReLU()
    elif function == 'glu':
        return nn.GLU()
    elif function == 'gelu':
        return nn.GELU()
    elif function == 'logsigmoid':
        return nn.LogSigmoid()
    elif function == 'hardshrink':
        return nn.Hardshrink()
    elif function == 'tanhshrink':
        return nn.Tanhshrink()
    elif function == 'softsign':
        return nn.Softsign()
    elif function == 'softplus':
        return nn.Softplus()
    elif function == 'softmin':
        return nn.Softmin()
    elif function == 'softmax':
        return nn.Softmax()
    elif function == 'softshrink':
        return nn.Softshrink()
    elif function == 'log_softmax':
        return nn.LogSoftmax()
    elif function == 'tanh':
        return nn.Tanh()
    elif function == 'sigmoid':
        return nn.Sigmoid()
    elif function == 'hardsigmoid':
        return nn.Hardsigmoid()
    elif function == 'silu':
        return nn.SiLU()
    elif function == 'mish':
        return nn.Mish()
    else:
        raise ValueError(f'Unknown activation function: {function}')
    
# for activation in getActivationFunctions():
#     print(activation, getActivationLayer(activation), getActivationFunction(activation))

class TransposeLayer(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)

import numpy as np
def buildMLPwActivation(layers, inputFeatures = 1, gain = 1/np.sqrt(34), activation = 'gelu', norm = False, groups = 1, preNorm = False, postNorm = False, noLinear = False):
    # print(f'layers: {layers}, inputFeatures: {inputFeatures}, gain: {gain}, activation: {activation}, norm: {norm}, channels: {channels}, preNorm: {preNorm}, postNorm: {postNorm}, noLinear: {noLinear}')
    activationFn = getActivationLayer(activation)
    modules = []
    if preNorm:
        modules.append(TransposeLayer(1,2))
        # print(f'groups: {groups[0] if isinstance(groups, list) else groups}, inputFeatures: {inputFeatures}')
        modules.append(nn.GroupNorm(groups[0] if isinstance(groups,list) else groups, inputFeatures))
        modules.append(TransposeLayer(1,2))
    if not noLinear:
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                modules.append(nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i]))
    #             torch.nn.init.uniform_(modules[-1].weight,-0.5, 0.5)
                torch.nn.init.xavier_normal_(modules[-1].weight,1)
        #         torch.nn.init.zeros_(modules[-1].weight)
                torch.nn.init.zeros_(modules[-1].bias)
                # modules.append(nn.BatchNorm1d(layers[i]))
                if norm:
                    modules.append(TransposeLayer(1,2))
                    # print(f'groups: {groups}, layers[i]: {layers[i]}')
                    modules.append(nn.GroupNorm(groups[i] if isinstance(groups,list) else groups, layers[i]))
                    modules.append(TransposeLayer(1,2))
                modules.append(activationFn)
            modules.append(nn.Linear(layers[-2],layers[-1]))
        else:
            modules.append(nn.Linear(inputFeatures,layers[-1]))  
        torch.nn.init.xavier_normal_(modules[-1].weight,gain)
        torch.nn.init.zeros_(modules[-1].bias)     
    if postNorm:
        modules.append(TransposeLayer(1,2))
        # print(f'groups: {channels}, layers[-1]: {layers[-1]}')
        # print(f'groups: {groups[-1] if isinstance(groups,list) else groups}, layers[-1]: {layers[-1]}')
        modules.append(nn.GroupNorm(groups[-1] if isinstance(groups,list) else groups, layers[-1]))
        modules.append(TransposeLayer(1,2)) 
    return nn.Sequential(*modules)

def buildMLPwDict(properties : dict):
    layout = properties['layout'] if 'layout' in properties else []
    output = properties['output']
    inputFeatures = properties['inputFeatures']
    groups = properties['channels'] if 'channels' in properties else 1
    gain = properties['gain'] if 'gain' in properties else 1/np.sqrt(34)
    activation = properties['activation'] if 'activation' in properties else 'celu'
    norm = properties['norm'] if 'norm' in properties else True
    preNorm = properties['preNorm'] if 'preNorm' in properties else False
    postNorm = properties['postNorm'] if 'postNorm' in properties else False
    noLinear = properties['noLinear'] if 'noLinear' in properties else False

    mlp = buildMLPwActivation(layout + [output], inputFeatures, gain = gain, activation = activation, norm = norm, groups = groups, preNorm = preNorm, postNorm = postNorm, noLinear = noLinear)
    return mlp

# from .detail.cutlass import cutlass
from .convLayerv2 import BasisConv
# from datautils import *
# from plotting import *

# Use dark theme
# from tqdm.autonotebook import trange, tqdm
# import os

from .detail.mapping import mapToSpherePreserving, mapToSpherical

import numpy as np
def process(edge_index_i, edge_index_j, edge_attr, centerIgnore = True, coordinateMapping = 'cartesian', windowFn = None):
    if centerIgnore:
        nequals = edge_index_i != edge_index_j

    i, ni = torch.unique(edge_index_i, return_counts = True)
    
    if centerIgnore:
        fluidEdgeIndex = torch.stack([edge_index_i[nequals], edge_index_j[nequals]], dim = 0)
    else:
        fluidEdgeIndex = torch.stack([edge_index_i, edge_index_j], dim = 0)
        
    if centerIgnore:
        fluidEdgeLengths = edge_attr[nequals]
    else:
        fluidEdgeLengths = edge_attr
    fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
    
    if not(windowFn is None):
        edge_weights = windowFn(torch.linalg.norm(fluidEdgeLengths, axis = 1))
    else:
        edge_weights = None

    mapped = fluidEdgeLengths

    # positions = torch.hstack((edge_attr, torch.zeros(edge_attr.shape[0],1, device = edge_attr.device, dtype = edge_attr.dtype)))
    if fluidEdgeLengths.shape[1] > 1:
        expanded = torch.hstack((fluidEdgeLengths, torch.zeros_like(fluidEdgeLengths[:,0])[:,None])) if edge_attr.shape[1] == 2 else fluidEdgeLengths
        if coordinateMapping == 'polar':
            spherical = mapToSpherical(expanded)
            if fluidEdgeLengths.shape[1] == 2:
                mapped = torch.vstack((spherical[:,0] * 2. - 1.,spherical[:,1] / np.pi)).mT
            else:
                mapped = torch.vstack((spherical[:,0] * 2. - 1.,spherical[:,1] / np.pi,spherical[:,2] / np.pi)).mT
        if coordinateMapping == 'cartesian':
            mapped = fluidEdgeLengths
        if coordinateMapping == 'preserving':
            cubePositions = mapToSpherePreserving(expanded)
            mapped = cubePositions
    return ni, i, fluidEdgeIndex, mapped, edge_weights
        

def buildMLP(layers, inputFeatures = 1, gain = 1/np.sqrt(34)):
    modules = []
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i]))
#             torch.nn.init.uniform_(modules[-1].weight,-0.5, 0.5)
            torch.nn.init.xavier_normal_(modules[-1].weight,1)
    #         torch.nn.init.zeros_(modules[-1].weight)
            torch.nn.init.zeros_(modules[-1].bias)
            # modules.append(nn.BatchNorm1d(layers[i]))
            modules.append(nn.GELU())
        modules.append(nn.Linear(layers[-2],layers[-1]))
    else:
        modules.append(nn.Linear(inputFeatures,layers[-1]))        
    torch.nn.init.xavier_normal_(modules[-1].weight,gain)
    torch.nn.init.zeros_(modules[-1].bias)
    return nn.Sequential(*modules)

import torch.nn as nn
def getActivationFunctions():
    return ['elu', 'relu', 'hardtanh', 'hardswish', 'selu', 'celu', 'leaky_relu', 'prelu', 'rrelu', 'glu', 'gelu', 'logsigmoid', 'hardshrink', 'tanhshrink', 'softsign', 'softplus', 'softmin', 'softmax', 'softshrink', 'gumbel_softmax', 'log_softmax', 'tanh', 'sigmoid', 'hardsigmoid', 'silu', 'mish']
def getActivationFunction(function : str):
    return getattr(nn.functional, function)

class BasisNetwork(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures = 0, layers = [32,64,64,2], denseLayer = True, activation = 'relu',
                coordinateMapping = 'cartesian', dims = [8], windowFn = None, rbfs = ['linear', 'linear'],batchSize = 32, ignoreCenter = True, normalized = False, outputScaling = 1/128, layerMLP = False, MLPLayout = [32,32], convBias = False, outputBias = True, initializer = 'uniform', optimizeWeights = False, exponentialDecay = True, inputEncoder = None, outputDecoder = None, edgeMLP = None, vertexMLP = None):
        super().__init__()
        self.centerIgnore = ignoreCenter
        self.features = copy.copy(layers)
        self.convs = torch.nn.ModuleList()
        self.mlps = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, activation)
        self.layers = layers
        self.dims = dims
        self.rbfs = rbfs
        self.dim = len(dims)
        self.normalized = normalized
        self.hasBoundaryLayers = boundaryFeatures != 0
        self.coordinateMapping = coordinateMapping
        self.windowFn = windowFn
        self.outputScaling = outputScaling
        self.layerMLP = layerMLP
        self.MLPLayout = MLPLayout
        self.inputEncoderProperties = inputEncoder
        self.outputDecoderProperties = outputDecoder
        self.edgeMLP = edgeMLP
        self.vertexMLP = vertexMLP
        self.vertexMLPDicts = []

        periodic = [False] * len(dims)
        if coordinateMapping == 'polar':
            if len(dims) == 1:
                periodic = [True]
            if len(dims) == 2:
                periodic = [False, True]
            if len(dims) == 3:
                periodic = [False, True, True]

        if self.inputEncoderProperties is not None:
            if 'inputFeatures' not in self.inputEncoderProperties:
                self.inputEncoderProperties['inputFeatures'] = fluidFeatures
            if 'output' not in self.inputEncoderProperties:
                self.inputEncoderProperties['output'] = fluidFeatures
            self.inputEncoder = buildMLPwDict(self.inputEncoderProperties)
        else:
            self.inputEncoder = None
        if self.outputDecoderProperties is not None:
            if 'inputFeatures' not in self.outputDecoderProperties:
                self.outputDecoderProperties['inputFeatures'] = self.features[-1]
            if 'output' not in self.outputDecoderProperties:
                self.outputDecoderProperties['output'] = self.features[-1]
            self.outputDecoder = buildMLPwDict(self.outputDecoderProperties)
        else:
            self.outputDecoder = None
        if self.edgeMLP is not None:
            self.edgeMLP['inputFeatures'] = len(dims)
            self.edgeMLP['output'] = len(dims)

        if len(layers) == 1:
            self.convs.append(BasisConv(
                inputFeatures=  fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output'], 
                outputFeatures= self.features[0] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures'],
                dim = len(dims), linearLayerActive= False, biasActive= convBias, feedThrough= False,
                basisTerms = dims, basisFunction = rbfs, basisPeriodicity= periodic, cutlassBatchSize= batchSize, 
                initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay))
            self.centerIgnore = False
            if boundaryFeatures != 0:
                self.convs.append(BasisConv(
                    inputFeatures=  boundaryFeatures, 
                    outputFeatures= self.features[0] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures'],
                    dim = len(dims), linearLayerActive= False, biasActive= convBias, feedThrough= False,
                    basisTerms = dims, basisFunction = rbfs, basisPeriodicity= periodic, cutlassBatchSize= batchSize, 
                    initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay))

            if self.layerMLP:
                self.mlps.append(buildMLP(self.MLPLayout + [self.features[0]], fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output'], gain = 1))
            return

        self.convs.append(BasisConv(
                inputFeatures=  fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output'], 
                outputFeatures= self.features[0],
                dim = len(dims), linearLayerActive= False, biasActive= convBias, feedThrough= False,
                basisTerms = dims, basisFunction = rbfs, basisPeriodicity= periodic, cutlassBatchSize= batchSize, 
                initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay))
        if boundaryFeatures != 0:
            self.convs.append(BasisConv(
                inputFeatures=  boundaryFeatures, 
                outputFeatures= self.features[0],
                dim = len(dims), linearLayerActive= False, biasActive= convBias, feedThrough= False,
                basisTerms = dims, basisFunction = rbfs, basisPeriodicity= periodic, cutlassBatchSize= batchSize, 
                initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay))

        self.fcs.append(nn.Linear(in_features=fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output'],out_features= layers[0],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)
    
        self.vertexMLPs = torch.nn.ModuleList()
        self.edgeMLPs = torch.nn.ModuleList()
        if self.vertexMLP is not None:
            newDict = copy.copy(self.vertexMLP)
            newDict['inputFeatures'] = 3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]
            newDict['output'] = newDict['inputFeatures'] if 'outputFeatures' not in newDict else newDict['outputFeatures']
            self.vertexMLPs.append(buildMLPwDict(newDict))
            self.vertexMLPDicts.append(newDict)
        if self.edgeMLP is not None:
            self.edgeMLPs.append(buildMLPwDict(self.edgeMLP))


        if self.layerMLP:
            width = (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0])
            self.mlps.append(buildMLP(self.MLPLayout + [width], width, gain = 1))

        self.features[0] = self.features[0]
        for i, l in enumerate(layers[1:-1]):
            inputFeatures = (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]) if i == 0 else self.features[i]
            if self.vertexMLP is not None:
                inputFeatures = newDict['output']

            self.convs.append(BasisConv(
                inputFeatures = inputFeatures, 
                outputFeatures = layers[i+1],

                dim = len(dims), linearLayerActive= False, biasActive= convBias, feedThrough= False,
                basisTerms = dims, basisFunction = rbfs, basisPeriodicity= periodic, cutlassBatchSize= batchSize, 
                initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay))
            self.fcs.append(nn.Linear(in_features=(3 * layers[0] if boundaryFeatures != 0 else 2 * self.features[0]) if i == 0 else layers[i],out_features=layers[i+1],bias=True))
            torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
            torch.nn.init.zeros_(self.fcs[-1].bias)

            if self.vertexMLP is not None:
                newDict = copy.copy(self.vertexMLP)
                newDict['inputFeatures'] = layers[i+1]
                newDict['output'] = newDict['inputFeatures'] if 'outputFeatures' not in newDict else newDict['outputFeatures']
                self.vertexMLPs.append(buildMLPwDict(newDict))
                self.vertexMLPDicts.append(newDict)
            if self.edgeMLP is not None:
                self.edgeMLPs.append(buildMLPwDict(self.edgeMLP))

            if self.layerMLP:
                self.mlps.append(buildMLP(self.MLPLayout + [layers[i+1]], layers[i+1], gain = 1))
            
        self.convs.append(BasisConv(
            inputFeatures = self.features[-2] if len(layers) > 2 else (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]), 
            outputFeatures = self.features[-1] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures'],
                dim = len(dims), linearLayerActive= False, biasActive= convBias, feedThrough= False,
                basisTerms = dims, basisFunction = rbfs, basisPeriodicity= periodic, cutlassBatchSize= batchSize, 
                initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay))
        self.fcs.append(nn.Linear(in_features=self.features[-2] if len(layers) > 2 else (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]),out_features=self.features[-1] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures'],bias=outputBias))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        if outputBias:
            torch.nn.init.zeros_(self.fcs[-1].bias)

        if self.layerMLP:
            self.mlps.append(buildMLP(self.MLPLayout + [self.features[-1]], self.features[-1], gain = 1))



    def forward(self, \
                fluidFeatures, \
                fluid_edge_index_i, fluid_edge_index_j, distances, boundaryFeatures = None, bf = None, bb = None, boundaryDistances = None, batches = 1, verbose = False):
        
        ni, i, fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights = process(
            fluid_edge_index_i, fluid_edge_index_j, distances, self.centerIgnore, self.coordinateMapping, self.windowFn)
        
        # print(f'(pre encoder) fluidFeatures: {fluidFeatures.shape}')
        if self.inputEncoder is not None:
            if verbose:
                print(f'(pre encoder) fluidFeatures: {fluidFeatures.shape}')
            transposedFeatures = fluidFeatures.view(batches,-1, *fluidFeatures.shape[1:])
            # print(f'(pre encoder) transposedFeatures: {transposedFeatures.shape}')
            fluidFeatures = self.inputEncoder(transposedFeatures)
            # print(f'(post encoder) fluidFeatures: {fluidFeatures.shape}')
            fluidFeatures = fluidFeatures.view(-1, *fluidFeatures.shape[2:])
            if verbose:
                print(f'(post encoder) fluidFeatures: {fluidFeatures.shape}')
        # print(f'(post encoder) fluidFeatures: {fluidFeatures.shape}')

        self.ni = ni
        if self.hasBoundaryLayers:
            nb, b, boundaryEdgeIndex, boundaryEdgeLengths, boundaryEdgeWeights = process(
                bf, bb, boundaryDistances, False, self.coordinateMapping, self.windowFn)
            self.nb = nb

            ni[i[b]] += nb
            if verbose:
                print(f'ni: {ni}, i: {i}, fluidEdgeIndex: {fluidEdgeIndex.shape}, fluidEdgeLengths: {fluidEdgeLengths.shape}, fluidEdgeWeights: {fluidEdgeWeights.shape}')
                print(f'nb: {nb}, b: {b}, boundaryEdgeIndex: {boundaryEdgeIndex.shape}, boundaryEdgeLengths: {boundaryEdgeLengths.shape}, boundaryEdgeWeights: {boundaryEdgeWeights.shape}')
        else:
            if verbose:
                print(f'ni: {ni}, i: {i}, fluidEdgeIndex: {fluidEdgeIndex.shape}, fluidEdgeLengths: {fluidEdgeLengths.shape}, fluidEdgeWeights: {fluidEdgeWeights.shape}')
        self.li = torch.exp(-1 / 16 * ni)
        if len(self.rbfs) > 2:
            self.li = torch.exp(-1 / 50 * ni)

        if verbose:
            print(f'Running Convolution (FTF) {self.convs[0].inputFeatures} -> {self.convs[0].outputFeatures} features')
        fluidConvolution = (self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights))
#         fluidConvolution = scatter_sum(baseArea * fluidFeatures[fluidEdgeIndex[1]] * kernelGradient(torch.abs(fluidEdgeLengths), torch.sign(fluidEdgeLengths), particleSupport), fluidEdgeIndex[0], dim = 0, dim_size = fluidFeatures.shape[0])
        
        if len(self.layers) == 1:
            if self.hasBoundaryLayers:
                if verbose:
                    print(f'Running Convolution (BTF) {self.convs[1].inputFeatures} -> {self.convs[1].outputFeatures} features')
                boundaryConvolution = (self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths, boundaryEdgeWeights))
                fluidConvolution += boundaryConvolution

            if self.layerMLP:
                fluidConvolution = self.mlps[0](fluidConvolution)
            if self.outputDecoder is not None:
                if verbose:
                    print(f'(pre outputDecoder) fluidConvolution: {fluidConvolution.shape}')
                fluidConvolution = self.outputDecoder(fluidConvolution.view(batches,-1, *fluidConvolution.shape[1:]))
                fluidConvolution = fluidConvolution.view(-1, *fluidConvolution.shape[2:])
                if verbose:
                    print(f'(post outputDecoder) fluidConvolution: {fluidConvolution.shape}')
            
            return fluidConvolution 
        if verbose:
            print(f'Running Linear {self.fcs[0].in_features} -> {self.fcs[0].out_features} features')
        linearOutput = (self.fcs[0](fluidFeatures))
        if self.hasBoundaryLayers:
            if verbose:
                print(f'Running Convolution {self.convs[1].inputFeatures} -> {self.convs[1].outputFeatures} features')
            boundaryConvolution = (self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths, boundaryEdgeWeights))
            ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
        else:
            ans = torch.hstack((linearOutput, fluidConvolution))
        if verbose:
            print(f'ans: {ans.shape}')
        
        if self.layerMLP:
            ans = self.mlps[0](ans)
        if self.edgeMLP is not None:
            if verbose:
                print(f'Running Edge MLP {self.edgeMLP["inputFeatures"]} -> {self.edgeMLP["output"]} features')
            fluidEdgeLengths = self.edgeMLPs[0](fluidEdgeLengths)
            fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
        if self.vertexMLP is not None:
            if verbose:
                print(f'Running Vertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features')
            transposedFeatures = ans.view(batches,-1, *ans.shape[1:])
            ans = self.vertexMLPs[0](transposedFeatures)
            ans = ans.view(-1, *ans.shape[2:])
        layers = len(self.convs)
        for i in range(1 if not self.hasBoundaryLayers else 2,layers):
            if verbose:
                print(f'Relu {ans.shape}')
            ansc = self.relu(ans)
            if verbose:
                print(f'Running Convolution {self.convs[i].inputFeatures} -> {self.convs[i].outputFeatures} features')
            ansConv = self.convs[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights)
            if verbose:
                print(f'Running Linear {self.fcs[i - (1 if self.hasBoundaryLayers else 0)].in_features} -> {self.fcs[i - (1 if self.hasBoundaryLayers else 0)].out_features} features') 
            ansDense = self.fcs[i - (1 if self.hasBoundaryLayers else 0)](ansc)
            
            
            if self.features[i- (2 if self.hasBoundaryLayers else 1)] == self.features[i-(1 if self.hasBoundaryLayers else 0)] and ans.shape == ansConv.shape:
                ans = ansConv + ansDense + ans
            else:
                ans = ansConv + ansDense
            if self.layerMLP:
                ans = self.mlps[i](ans)
            if self.edgeMLP is not None and i < layers - 1:
                if verbose:
                    print(f'Running Edge MLP {self.edgeMLP["inputFeatures"]} -> {self.edgeMLP["output"]} features')
                fluidEdgeLengths = self.edgeMLPs[i](fluidEdgeLengths)
                fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
            if self.vertexMLP is not None and i < layers - 1:
                if verbose:
                    print(f'Running Vertex MLP {self.vertexMLPDicts[i]["inputFeatures"]} -> {self.vertexMLPDicts[i]["output"]} features')
                transposedFeatures = ans.view(batches,-1, *ans.shape[1:])
                ans = self.vertexMLPs[i](transposedFeatures)
                ans = ans.view(-1, *ans.shape[2:])
        if self.outputDecoder is not None:
            if verbose:
                print(f'(pre outputDecoder) ans: {ans.shape}')
            ans = self.outputDecoder(ans.view(batches,-1, *ans.shape[1:]))
            ans = ans.view(-1, *ans.shape[2:])
            if verbose:
                print(f'(post outputDecoder) ans: {ans.shape}')

        return ans * self.outputScaling #(ans * outputScaling) if self.dim == 2 else ans
    