import torch
import numpy as np
import copy

def augmentState(perennialState, augJitter = None, augRotation = None, augmentFeatures = True):
    augmentedState = perennialState.copy()
    if augJitter is not None:
        augmentedState['fluid']['positions'] += augJitter
    if augRotation is not None:
        for k in augmentedState['fluid'].keys():
            if not isinstance(augmentedState['fluid'][k], torch.Tensor):
                continue
            if augmentedState['fluid'][k].dim() == 2 and augmentedState['fluid'][k].shape[1] == augmentedState['fluid']['positions'].shape[1]:
                if not augmentFeatures and k == 'features':
                    continue                
                augmentedState['fluid'][k] = augmentedState['fluid'][k].clone() @ augRotation
        if 'boundary' in augmentedState and augmentedState['boundary'] is not None:
            for k in augmentedState['boundary'].keys():
                if not isinstance(augmentedState['boundary'][k], torch.Tensor):
                    continue                
                if not augmentFeatures and k == 'features':
                    continue  
                if augmentedState['boundary'][k].dim() == 2 and augmentedState['boundary'][k].shape[1] == augmentedState['boundary']['positions'].shape[1]:
                    augmentedState['boundary'][k] = augmentedState['boundary'][k].clone() @ augRotation
    return augmentedState

def augmentStates(attributes, states, hyperParameterDict):
    if hyperParameterDict['augmentJitter']:
        jitterAmount = hyperParameterDict['jitterAmount']
        augJitter = torch.normal(0, jitterAmount * attributes['support'], states[0]['fluid']['positions'].shape, device = states[0]['fluid']['positions'].device, dtype = states[0]['fluid']['positions'].dtype)
    else:
        augJitter = None
    if hyperParameterDict['augmentAngle']:
        dim = states[0]['fluid']['positions'].shape[1]
        if dim == 1:
            raise ValueError('Cannot rotate 1D data')
        if dim == 2:
            angle = torch.rand(1) * 2 *  np.pi
            augRotation = torch.tensor([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]], device = states[0]['fluid']['positions'].device, dtype = states[0]['fluid']['positions'].dtype)
        if dim == 3:
            angle_phi = torch.rand(1) * 2 *  np.pi
            angle_theta = torch.rand(1) * 2 *  np.pi
            augRotation = torch.tensor([
                [np.cos(angle_phi) * np.sin(angle_theta), -np.sin(angle_phi), np.cos(angle_phi) * np.cos(angle_theta)],
                [np.sin(angle_phi) * np.sin(angle_theta), np.cos(angle_phi), np.sin(angle_phi) * np.cos(angle_theta)],
                [np.cos(angle_theta), 0, -np.sin(angle_theta)]
            ], device = states[0]['fluid']['positions'].device, dtype = states[0]['fluid']['positions'].dtype)
    else:
        augRotation = None

    if hyperParameterDict['augmentJitter'] or hyperParameterDict['augmentAngle']:
        states = [copy.deepcopy(state) for state in states]

        states = [augmentState(s, augJitter = augJitter, augRotation = augRotation) for s in states]
        for state in states:
            state['augmentJitter'] = augJitter
            state['augmentRotation'] = augRotation
    return states
        
from BasisConvolution.util.testcases import loadFrame

from BasisConvolution.util.features import getFeatures
from BasisConvolution.util.radius import searchNeighbors

def loadAugmentedFrame(index, dataset, hyperParameterDict, unrollLength = 8):
    if unrollLength > hyperParameterDict['maxUnroll']:
        unrollLength = hyperParameterDict['maxUnroll']
    config, attributes, currentState, priorState, trajectoryStates = loadFrame(index, dataset, hyperParameterDict, unrollLength = unrollLength)
    # print(currentState)
    # print(priorState)
    # print(trajectoryStates)
    
    combinedStates = []
    combinedStates.append(currentState)

    if priorState is not None:
        combinedStates.append(priorState)

    combinedStates += trajectoryStates
    
    augmentedStates = augmentStates(attributes, combinedStates, hyperParameterDict,)

    # config['neighborhood']['verletScale'] = 1.0
    # config['neighborhood']['scheme'] = 'compact'

    searchNeighbors(augmentedStates[0], config, computeKernels = True)

    currentState = augmentedStates[0]
    priorState = augmentedStates[1] if priorState is not None else None
    trajectoryStates = augmentedStates[2:] if priorState is not None else augmentedStates[1:]

    if 'compute' in hyperParameterDict['groundTruth']:
        for state  in trajectoryStates:
            if hyperParameterDict['frameDistance'] == 0:
                state['fluid']['neighborhood'] = currentState['fluid']['neighborhood']
                if 'boundary' in currentState and currentState['boundary'] is not None:
                    state['boundary']['neighborhood'] = currentState['boundary']['neighborhood']
                    state['fluidToBoundaryNeighborhood'] = currentState['fluidToBoundaryNeighborhood']
                    state['boundaryToFluidNeighborhood'] = currentState['boundaryToFluidNeighborhood']
            else:
                searchNeighbors(state, config, computeKernels = True)
    currentState['fluid']['features'] = getFeatures(hyperParameterDict['fluidFeatures'].split(' '), currentState, priorState if priorState is not None else None, 'fluid', config, currentState['time'] - priorState['time'] if priorState is not None else 0.0, verbose = False, includeOther = 'boundary' in currentState and currentState['boundary'] is not None)
    
    # print('boundary')
    if 'boundary' in currentState and currentState['boundary'] is not None:
        currentState['boundary']['features'] = getFeatures(hyperParameterDict['boundaryFeatures'].split(' '), currentState, priorState if priorState is not None else None, 'boundary', config, currentState['time'] - priorState['time'] if priorState is not None else 0.0, verbose = False, includeOther = True)
    # print('gt')
    cState = currentState
    for state in trajectoryStates:
        state['fluid']['target'] = getFeatures(hyperParameterDict['groundTruth'].split(' '), state, cState, 'fluid', config, state['time'] - cState['time'], verbose = False, includeOther = 'boundary' in currentState and currentState['boundary'] is not None,)
        cState = state

    return config, attributes, augmentedStates[0], augmentedStates[1] if priorState is not None else None, augmentedStates[2:] if priorState is not None else augmentedStates[1:]


def loadAugmentedBatch(bdata, dataset, hyperParameterDict, unrollLength = 8):
    if unrollLength > hyperParameterDict['maxUnroll']:
        unrollLength = hyperParameterDict['maxUnroll']
    data = [loadAugmentedFrame(index, dataset, hyperParameterDict, unrollLength = unrollLength) for index in bdata]
    return [data[0] for data in data], [data[1] for data in data], [data[2] for data in data], [data[3] for data in data], [data[4] for data in data]