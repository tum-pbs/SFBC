
import torch
import torch.nn as nn
from .activation import getActivationLayer
from collections import OrderedDict

class TransposeLayer(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)

def runMLP_(mlp : torch.nn.Module, features : torch.Tensor, batches : int, verbose : bool = False):  
    if verbose:
        print(f'MLP {features.shape} -> {mlp[-1].out_features} features')
    transposedFeatures = features.view(batches,-1, *features.shape[1:])
    
    processedFeatures = mlp(transposedFeatures)
    processedFeatures = processedFeatures.view(-1, *processedFeatures.shape[2:])
    if verbose:
        print(f'\tFeatures: {processedFeatures.shape} [min: {torch.min(processedFeatures)}, max: {torch.max(processedFeatures)}, mean: {torch.mean(processedFeatures)}]')
    return processedFeatures


# @torch.jit.script
def runMLP(mlp : torch.nn.Module, features : torch.Tensor, batches : int, verbose : bool = False, checkpoint : bool = True):      
    if checkpoint:
        return torch.utils.checkpoint.checkpoint(runMLP_, mlp, features, batches, verbose, use_reentrant = False)
    else:
        return runMLP_(mlp, features, batches, verbose)

import numpy as np
def buildMLPwActivation(layers, inputFeatures = 1, gain = 1/np.sqrt(34), activation = 'gelu', norm = False, groups = 1, preNorm = False, postNorm = False, noLinear = False, bias = True):
    # print(f'layers: {layers}, inputFeatures: {inputFeatures}, gain: {gain}, activation: {activation}, norm: {norm}, channels: {channels}, preNorm: {preNorm}, postNorm: {postNorm}, noLinear: {noLinear}')
    activationFn = getActivationLayer(activation)
    transposeCounter = 0
    normCounter = 0
    linear = 0
    modules = []
    if preNorm:
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1
        # print(f'groups: {groups[0] if isinstance(groups, list) else groups}, inputFeatures: {inputFeatures}')
        # print(f'PreNorm: {groups} | {inputFeatures}')
        if isinstance(groups,list):
            numGroups = groups[0]
        if numGroups == -1:
            numGroups = inputFeatures
        modules.append((f'norm{normCounter}', nn.GroupNorm(numGroups, inputFeatures)))
        normCounter += 1
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1

    if not noLinear:
        if len(layers) > 1:
            for i in range(len(layers) - 1):
                modules.append((f'linear{linear}', nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i])))
                linear += 1

    #             torch.nn.init.uniform_(modules[-1].weight,-0.5, 0.5)
                torch.nn.init.xavier_normal_(modules[-1][1].weight,1)
        #         torch.nn.init.zeros_(modules[-1].weight)
                torch.nn.init.zeros_(modules[-1][1].bias)
                # modules.append(nn.BatchNorm1d(layers[i]))
                if norm:
                    modules.append((f'transposeLayer{transposeCounter}',TransposeLayer(1,2)))
                    transposeCounter += 1
                    # print(f'groups: {groups}, layers[i]: {layers[i]}')

                    numGroups = groups[(i + 1) if preNorm else i] if isinstance(groups,list) else groups
                    if numGroups == -1:
                        numGroups = layers[i]
                    modules.append((f'norm{normCounter}', nn.GroupNorm(numGroups, layers[i])))
                    normCounter += 1
                    modules.append((f'transposeLayer{transposeCounter}',TransposeLayer(1,2)))
                    transposeCounter += 1
                modules.append((f'activation{linear-1}', activationFn))
            modules.append((f'linear{linear}', nn.Linear(layers[-2],layers[-1], bias = bias)))
        else:
            modules.append((f'linear{linear}', nn.Linear(inputFeatures,layers[-1], bias = bias))  )
        torch.nn.init.xavier_normal_(modules[-1][1].weight,gain)
        if bias:
            torch.nn.init.zeros_(modules[-1][1].bias)     
    if postNorm:
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1
        # print(f'groups: {channels}, layers[-1]: {layers[-1]}')
        # print(f'groups: {groups[-1] if isinstance(groups,list) else groups}, layers[-1]: {layers[-1]}')
        numGroups = groups[-1] if isinstance(groups,list) else groups
        if numGroups == -1:
            numGroups = layers[-1]
        modules.append((f'norm{normCounter}', nn.GroupNorm(numGroups, layers[-1])))
        normCounter += 1
        modules.append((f'transposeLayer{transposeCounter}', TransposeLayer(1,2)))
        transposeCounter += 1
    moduleDict = OrderedDict()
    for i, module in enumerate(modules):
        moduleDict[module[0]] = module[1]
    return nn.Sequential(moduleDict)

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
    
    numberOfNorms = 0
    if preNorm:
        numberOfNorms += 1
    if postNorm:
        numberOfNorms += 1
    if norm and not noLinear:
        numberOfNorms += len(layout)
    if numberOfNorms >0 and (isinstance(groups,list) and numberOfNorms != len(groups)):
        raise ValueError(f'Number of groups {len(groups)} does not match number of norms {numberOfNorms}')

    mlp = buildMLPwActivation(layout + [output], inputFeatures, gain = gain, activation = activation, norm = norm, groups = groups, preNorm = preNorm, postNorm = postNorm, noLinear = noLinear, bias = properties['bias'] if 'bias' in properties else True)
    return mlp


# def buildMLP(layers, inputFeatures = 1, gain = 1/np.sqrt(34)):
#     modules = []
#     if len(layers) > 1:
#         for i in range(len(layers) - 1):
#             modules.append(nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i]))
# #             torch.nn.init.uniform_(modules[-1].weight,-0.5, 0.5)
#             torch.nn.init.xavier_normal_(modules[-1].weight,1)
#     #         torch.nn.init.zeros_(modules[-1].weight)
#             torch.nn.init.zeros_(modules[-1].bias)
#             # modules.append(nn.BatchNorm1d(layers[i]))
#             modules.append(nn.GELU())
#         modules.append(nn.Linear(layers[-2],layers[-1]))
#     else:
#         modules.append(nn.Linear(inputFeatures,layers[-1]))        
#     torch.nn.init.xavier_normal_(modules[-1].weight,gain)
#     torch.nn.init.zeros_(modules[-1].bias)
#     return nn.Sequential(*modules)