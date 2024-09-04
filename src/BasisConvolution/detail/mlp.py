
import torch
import torch.nn as nn
from .activation import getActivationLayer

class TransposeLayer(nn.Module):
    def __init__(self, dim1=0, dim2=1):
        super(TransposeLayer, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, input):
        return torch.transpose(input, self.dim1, self.dim2)

import numpy as np
def buildMLPwActivation(layers, inputFeatures = 1, gain = 1/np.sqrt(34), activation = 'gelu', norm = False, groups = 1, preNorm = False, postNorm = False, noLinear = False, bias = True):
    # print(f'layers: {layers}, inputFeatures: {inputFeatures}, gain: {gain}, activation: {activation}, norm: {norm}, channels: {channels}, preNorm: {preNorm}, postNorm: {postNorm}, noLinear: {noLinear}')
    activationFn = getActivationLayer(activation)
    modules = []
    if preNorm:
        modules.append(TransposeLayer(1,2))
        # print(f'groups: {groups[0] if isinstance(groups, list) else groups}, inputFeatures: {inputFeatures}')
        if isinstance(groups,list):
            numGroups = groups[0]
        if numGroups == -1:
            numGroups = inputFeatures
        modules.append(nn.GroupNorm(numGroups, inputFeatures))
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

                    numGroups = groups[(i + 1) if preNorm else i] if isinstance(groups,list) else groups
                    if numGroups == -1:
                        numGroups = layers[i]
                    modules.append(nn.GroupNorm(numGroups, layers[i]))
                    modules.append(TransposeLayer(1,2))
                modules.append(activationFn)
            modules.append(nn.Linear(layers[-2],layers[-1], bias = bias))
        else:
            modules.append(nn.Linear(inputFeatures,layers[-1], bias = bias)  )
        torch.nn.init.xavier_normal_(modules[-1].weight,gain)
        if bias:
            torch.nn.init.zeros_(modules[-1].bias)     
    if postNorm:
        modules.append(TransposeLayer(1,2))
        # print(f'groups: {channels}, layers[-1]: {layers[-1]}')
        # print(f'groups: {groups[-1] if isinstance(groups,list) else groups}, layers[-1]: {layers[-1]}')
        numGroups = groups[-1] if isinstance(groups,list) else groups
        if numGroups == -1:
            numGroups = layers[-1]
        modules.append(nn.GroupNorm(numGroups, layers[-1]))
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
    
    numberOfNorms = 0
    if preNorm:
        numberOfNorms += 1
    if postNorm:
        numberOfNorms += 1
    if norm and not noLinear:
        numberOfNorms += len(layout)
    if isinstance(groups,list) and numberOfNorms != len(groups):
        raise ValueError(f'Number of groups {len(groups)} does not match number of norms {numberOfNorms}')

    mlp = buildMLPwActivation(layout + [output], inputFeatures, gain = gain, activation = activation, norm = norm, groups = groups, preNorm = preNorm, postNorm = postNorm, noLinear = noLinear, bias = properties['bias'] if 'bias' in properties else True)
    return mlp


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