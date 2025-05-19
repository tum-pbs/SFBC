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
    # if hyperParameterDict['augmentJitter']:
    #     jitterAmount = hyperParameterDict['jitterAmount']
    #     augJitter = torch.normal(0, jitterAmount * attributes['support'], states[0]['fluid']['positions'].shape, device = states[0]['fluid']['positions'].device, dtype = states[0]['fluid']['positions'].dtype)
    # else:
    #     augJitter = None
    # if hyperParameterDict['augmentAngle']:
    #     dim = states[0]['fluid']['positions'].shape[1]
    #     if dim == 1:
    #         raise ValueError('Cannot rotate 1D data')
    #     if dim == 2:
    #         angle = torch.rand(1) * 2 *  np.pi
    #         augRotation = torch.tensor([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]], device = states[0]['fluid']['positions'].device, dtype = states[0]['fluid']['positions'].dtype)
    #     if dim == 3:
    #         angle_phi = torch.rand(1) * 2 *  np.pi
    #         angle_theta = torch.rand(1) * 2 *  np.pi
    #         augRotation = torch.tensor([
    #             [np.cos(angle_phi) * np.sin(angle_theta), -np.sin(angle_phi), np.cos(angle_phi) * np.cos(angle_theta)],
    #             [np.sin(angle_phi) * np.sin(angle_theta), np.cos(angle_phi), np.sin(angle_phi) * np.cos(angle_theta)],
    #             [np.cos(angle_theta), 0, -np.sin(angle_theta)]
    #         ], device = states[0]['fluid']['positions'].device, dtype = states[0]['fluid']['positions'].dtype)
    # else:
    #     augRotation = None

    # if hyperParameterDict['augmentJitter'] or hyperParameterDict['augmentAngle']:
    #     states = [copy.deepcopy(state) for state in states]

    #     states = [augmentState(s, augJitter = augJitter, augRotation = augRotation) for s in states]
    #     for state in states:
    #         state['augmentJitter'] = augJitter
    #         state['augmentRotation'] = augRotation

    if hyperParameterDict['velocityNoise']:
        u_mag = torch.norm(states[0]['fluid']['velocities'], dim = -1)
        states[0]['fluid']['velocities'] += torch.randn_like(states[0]['fluid']['velocities']) * (u_mag if hyperParameterDict['velocityNoiseScaling'] == 'rel' else 1.0)[:,None] * hyperParameterDict['velocityNoiseMagnitude']

    if hyperParameterDict['positionNoise']:
        states[0]['fluid']['positions'] += torch.randn_like(states[0]['fluid']['positions']) * (hyperParameterDict['positionNoiseMagnitude'] * attributes['support'] / 2)

    return states
        
from BasisConvolution.util.testcases import loadFrame

from BasisConvolution.util.features import getFeatures
from BasisConvolution.util.radius import searchNeighbors

def loadAugmentedFrame(index, dataset, hyperParameterDict, unrollLength = 8, skipAssembly = False, limitUnroll = True, skipAugment = False):
    if unrollLength > hyperParameterDict['maxUnroll'] and limitUnroll:
        print('Unroll length ', unrollLength, ' exceeds maximum, limiting to', hyperParameterDict["maxUnroll"])
        unrollLength = hyperParameterDict['maxUnroll']
        # print('Unroll length exceeds maximum, limiting to', unrollLength)
    # print('Loading frame ', index, ' with unroll length ', unrollLength)
    config, attributes, currentState, priorStates, trajectoryStates = loadFrame(index, dataset, hyperParameterDict, unrollLength = unrollLength)
    # print(len(priorStates), len(trajectoryStates))
    # print(currentState)
    # print(priorState)
    # print(trajectoryStates)
    
    combinedStates = []
    combinedStates.append(currentState)

    # print('history Length', hyperParameterDict['historyLength'])

    if priorStates is not None and len(priorStates) > 0:
        # print('Prior states', len(priorStates))
        combinedStates += priorStates
    # if len(trajectoryStates) > 0:
        # print('Trajectory states', len(trajectoryStates))

    combinedStates += trajectoryStates
    # print('Combined states', len(combinedStates))

    # for si, state in enumerate(combinedStates):
        # if not isinstance(state, dict):
            # print('????????', state)
        # print('State', si, state.keys())


    if skipAugment:
        augmentedStates = combinedStates
    else:
        augmentedStates = augmentStates(attributes, combinedStates, hyperParameterDict,)

    # config['neighborhood']['verletScale'] = 1.0
    # config['neighborhood']['scheme'] = 'compact'
    searchNeighbors(augmentedStates[0], config, computeKernels = True)

    currentState = augmentedStates[0]
    if priorStates is not None:
        priorStates = augmentedStates[1:1 + len(priorStates)] if len(priorStates) > 0 else priorStates
        trajectoryStates = augmentedStates[1 + len(priorStates):] #if priorState is not None else augmentedStates[1 + len(priorStates):]
    # print('prior', len(priorStates), 'trajectory', len(trajectoryStates))

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
    if skipAssembly:
        return config, attributes, currentState, priorStates, trajectoryStates

    # print(currentState.keys())
    # print(priorStates)

    # print(currentState)
    # print(priorStates)

    currentState['fluid']['features'] = getFeatures(hyperParameterDict['fluidFeatures'].split(' '), currentState, priorStates, 'fluid', config, currentState['time'] - priorStates[-1]['time'] if priorStates is not None and len(priorStates) > 0 else 0.0, verbose = False, includeOther = 'boundary' in currentState and currentState['boundary'] is not None, historyLength=hyperParameterDict['historyLength'], normalizeRho=hyperParameterDict['normalizeDensity'])
    
    # print('boundary')
    if 'boundary' in currentState and currentState['boundary'] is not None:
        currentState['boundary']['features'] = getFeatures(hyperParameterDict['boundaryFeatures'].split(' '), currentState, priorStates, 'boundary', config, currentState['time'] - priorStates[-1]['time'] if priorStates is not None and len(priorStates) > 0 else 0.0, verbose = False, includeOther = True, historyLength=hyperParameterDict['historyLength'], normalizeRho=hyperParameterDict['normalizeDensity'])
    # print('gt')
    cState = currentState
    for state in trajectoryStates:
        state['fluid']['target'] = getFeatures(hyperParameterDict['groundTruth'].split(' '), state, [cState], 'fluid', config, state['time'] - cState['time'] if state['time'] != cState['time'] else 1, verbose = False, includeOther = 'boundary' in currentState and currentState['boundary'] is not None, historyLength=0, normalizeRho=hyperParameterDict['normalizeDensity'])
        cState = state
    # print('done' , len(priorStates), len(trajectoryStates))
    return config, attributes, currentState, priorStates, trajectoryStates


def loadAugmentedBatch(bdata, dataset, hyperParameterDict, unrollLength = 8, skipAssembly = False, limitUnroll = True, skipAugment = False):
    # print('Loading batch with length ', len(bdata), ' and unroll length ', unrollLength, ' limited to ', hyperParameterDict['maxUnroll'] if limitUnroll else 'unlimited')
    if unrollLength > hyperParameterDict['maxUnroll'] and limitUnroll:
        print('Unroll length ', unrollLength, ' exceeds maximum, limiting to', hyperParameterDict["maxUnroll"], '[batch]')
        unrollLength = hyperParameterDict['maxUnroll']
    
    data = [loadAugmentedFrame(index, dataset, hyperParameterDict, unrollLength = unrollLength, skipAssembly=skipAssembly, limitUnroll=limitUnroll, skipAugment=skipAugment) for index in bdata]
    return [data[0] for data in data], [data[1] for data in data], [data[2] for data in data], [data[3] for data in data], [data[4] for data in data]