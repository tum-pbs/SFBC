import torch
import numpy as np
import h5py
import warnings
from BasisConvolution.util.datautils import isTemporalData
from BasisConvolution.sph.kernels import getKernel
# import warnings

def loadAdditional(inGrp, state, additionalData, device, dtype):
    for dataKey in additionalData:
        if dataKey in inGrp:
            state[dataKey] = torch.from_numpy(inGrp[dataKey][:]).to(device = device, dtype = dtype)
        else:
            warnings.warn('Additional data key %s not found in group' % dataKey)
    return state

def loadFrame_testcaseI(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    attributes = {
        'support': inFile.attrs['particleSupport'],
        'targetNeighbors': inFile.attrs['particleSupport'] / inFile.attrs['particleRadius'],
        'restDensity': inFile.attrs['restDensity'],
        'dt': inFile.attrs['dt'],
        'time': inFile.attrs['dt'] * key,
        'radius': inFile.attrs['particleRadius'],
        'area': inFile.attrs['baseArea'],
    }

    config = {
        'domain':{
            'dim': 1,
            'minExtent': torch.tensor([-1], device = device, dtype = dtype),
            'maxExtent': torch.tensor([1], device = device, dtype = dtype),
            'periodicity': torch.tensor([True], device = device, dtype = torch.bool),
            'periodic': True
        },
        'neighborhood':{
            'scheme': 'compact',
            'verletScale': 1.4
        },
        'compute':{
            'device': device,
            'dtype': dtype,
            'precision': 'float32' if dtype == torch.float32 else 'float64',
        },
        'kernel':{
            'name': 'Wendland2',
            'targetNeighbors': attributes['targetNeighbors'],
            'function': getKernel('Wendland2')
        },
        'boundary':{
            'active': False
        },
        'fluid':{
            'rho0': 1000,
            'cs': 20,
        },
        'particle':{
            'support': attributes['support']
        }
    }

    grp = inFile['simulationData']
    areas = torch.from_numpy(np.array(grp['fluidAreas'][key,:])).to(device = device, dtype = dtype)

    priorKey = key - hyperParameterDict['frameDistance']

    state = {
        'fluid': {
            'positions': torch.from_numpy(np.array(grp['fluidPosition'][key,:])).to(device = device, dtype = dtype).view(-1,1),
            'velocities': torch.from_numpy(np.array(grp['fluidVelocities'][key,:])).to(device = device, dtype = dtype).view(-1,1),
            'gravityAcceleration': torch.zeros_like(areas, device = device, dtype = dtype),
            'densities': torch.from_numpy(np.array(grp['fluidDensity'][key,:])).to(device = device, dtype = dtype) * config['fluid']['rho0'],
            'areas': areas,
            'masses': areas * inFile.attrs['restDensity'],
            'supports': torch.ones_like(areas) * attributes['support'],
            'indices': torch.arange(areas.shape[0], device = device, dtype = torch.int64),
            'numParticles': len(areas)
        },
        'boundary': None,
        'time': inFile.attrs['dt'] * key,
        'dt': inFile.attrs['dt'],
        'timestep': key,
    }
    loadAdditional(grp, state['fluid'], additionalData, device, dtype)


    # for dataKey in additionalData:
    #     state['fluid'][dataKey] = torch.from_numpy(np.array(grp[dataKey][key,:])).to(device = device, dtype = dtype)
    priorState = None
    if buildPriorState:
        if priorKey < 0 or hyperParameterDict['frameDistance'] == 0:
            priorState = copy.deepcopy(state)
        else:
            priorState = {
                'fluid': {
                    'positions': torch.from_numpy(np.array(grp['fluidPosition'][priorKey,:])).to(device = device, dtype = dtype).view(-1,1),
                    'velocities': torch.from_numpy(np.array(grp['fluidVelocities'][priorKey,:])).to(device = device, dtype = dtype).view(-1,1),
                    'gravityAcceleration': torch.zeros_like(areas, device = device, dtype = dtype),
                    'densities': torch.from_numpy(np.array(grp['fluidDensity'][priorKey,:])).to(device = device, dtype = dtype) * config['fluid']['rho0'],
                    'areas': areas,
                    'masses': areas * inFile.attrs['restDensity'],
                    'supports': torch.ones_like(areas) * attributes['support'],
                    'indices': torch.arange(areas.shape[0], device = device, dtype = torch.int64),
                    'numParticles': len(areas)
                },
                'boundary': None,
                'time': inFile.attrs['dt'] * priorKey,
                'dt': inFile.attrs['dt'],
                'timestep': priorKey,
            }
            loadAdditional(grp, priorState['fluid'], additionalData, device, dtype)
            # for dataKey in additionalData:
            #     priorState['fluid'][dataKey] = torch.from_numpy(np.array(grp[dataKey][priorKey,:])).to(device = device, dtype = dtype)
    nextStates = []
    if buildNextState:
        if unrollLength == 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)]
        if unrollLength == 0 and hyperParameterDict['frameDistance'] != 0:
            nextStates = [copy.deepcopy(state)]
            warnings.warn('Unroll length is zero, but frame distance is not zero')
        if unrollLength != 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)] * unrollLength
        if unrollLength != 0 and hyperParameterDict['frameDistance'] != 0:
            for u in range(unrollLength):
                unrollKey = key + hyperParameterDict['frameDistance'] * (u + 1)
                nextState = {
                    'fluid': {
                        'positions': torch.from_numpy(np.array(grp['fluidPosition'][unrollKey,:])).to(device = device, dtype = dtype).view(-1,1),
                        'velocities': torch.from_numpy(np.array(grp['fluidVelocities'][unrollKey,:])).to(device = device, dtype = dtype).view(-1,1),
                        'gravityAcceleration': torch.zeros_like(areas, device = device, dtype = dtype),
                        'densities': torch.from_numpy(np.array(grp['fluidDensity'][unrollKey,:])).to(device = device, dtype = dtype) * config['fluid']['rho0'],
                        'areas': areas,
                        'masses': areas * inFile.attrs['restDensity'],
                        'supports': torch.ones_like(areas) * attributes['support'],
                        'indices': torch.arange(areas.shape[0], device = device, dtype = torch.int64),
                        'numParticles': len(areas)
                    },
                    'boundary': None,
                    'time': inFile.attrs['dt'] * unrollKey,
                    'dt': inFile.attrs['dt'],
                    'timestep': unrollKey,
                }
                loadAdditional(grp, nextState['fluid'], additionalData, device, dtype)
                # for dataKey in additionalData:
                    # nextState['fluid'][dataKey] = torch.from_numpy(np.array(grp[dataKey][unrollKey,:])).to(device = device, dtype = dtype)
                # nextStates.append(nextState)            



    return config, attributes, state, priorState, nextStates

def loadGroup_testcaseII(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    if 'boundaryInformation' in inFile:
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]
    else:
        dynamicBoundaryData = None

    areas = torch.from_numpy(inGrp['fluidArea'][:]).to(device = device, dtype = dtype)
    state = {
        'fluid': {
            'positions': torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype) if 'fluidGravity' not in inFile.attrs else torch.from_numpy(inFile.attrs['fluidGravity']).to(device = device, dtype = dtype) * torch.ones(inGrp['fluidDensity'][:].shape[0]).to(device = device, dtype = dtype)[:,None],
            'densities': torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype) * inFile.attrs['restDensity'],
            'areas': areas,
            'masses': areas * inFile.attrs['restDensity'],
            'supports': torch.from_numpy(inGrp['fluidSupport'][:]).to(device = device, dtype = dtype),
            'indices': torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(areas)
        },
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state


def loadFrame_testcaseII(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(key)

    inGrp = inFile['simulationExport'][key]

    attributes = {
        'support': np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support'],
        'targetNeighbors': inFile.attrs['targetNeighbors'],
        'restDensity': inFile.attrs['restDensity'],
        'dt': inGrp.attrs['dt'],
        'time': inGrp.attrs['time'],
        'radius': inFile.attrs['radius'] if 'radius' in inFile.attrs else inGrp.attrs['radius'],
        'area': inFile.attrs['radius'] **2 * np.pi if 'area' not in inFile.attrs else inFile.attrs['area'],
    }
    config = {
        'domain':{
            'dim': 2,
            'minExtent': torch.tensor([-1.2, -1.2], device = device, dtype = dtype),
            'maxExtent': torch.tensor([1.2, 1.2], device = device, dtype = dtype),
            'periodicity': torch.tensor([False, False], device = device, dtype = torch.bool),
            'periodic': False
        },
        'neighborhood':{
            'scheme': 'compact',
            'verletScale': 1.4
        },
        'compute':{
            'device': device,
            'dtype': dtype,
            'precision': 'float32' if dtype == torch.float32 else 'float64',
        },
        'kernel':{
            'name': 'Wendland2',
            'targetNeighbors': 20,
            'function': getKernel('Wendland2')
        },
        'boundary':{
            'active': True
        },
        'fluid':{
            'rho0': 1000,
            'cs': 20,
        },
        'particle':{
            'support': attributes['support']
        }
    }

    if 'boundaryInformation' in inFile:
        staticBoundaryData = {
                'indices': torch.arange(0, inFile['boundaryInformation']['boundaryPosition'].shape[0], device = device, dtype = torch.int64),
                'positions': torch.from_numpy(inFile['boundaryInformation']['boundaryPosition'][:]).to(device = device, dtype = dtype),
                'normals': torch.from_numpy(inFile['boundaryInformation']['boundaryNormals'][:]).to(device = device, dtype = dtype),
                'areas': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype),
                'masses': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype) * config['fluid']['rho0'],
                'velocities': torch.from_numpy(inFile['boundaryInformation']['boundaryVelocity'][:]).to(device = device, dtype = dtype),
                'densities': torch.from_numpy(inFile['boundaryInformation']['boundaryRestDensity'][:]).to(device = device, dtype = dtype),
                'supports': torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:]).to(device = device, dtype = dtype),
                'bodyIDs': torch.from_numpy(inFile['boundaryInformation']['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64),
                'numParticles': len(inFile['boundaryInformation']['boundaryPosition'][:]),
            } if 'boundaryInformation' in inFile else None
    else:
        staticBoundaryData = None

    if 'boundaryInformation' in inFile:
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]

        dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype) if 'boundaryPosition' in inGrp else dynamicBoundaryData['positions']
        dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype) if 'boundaryNormals' in inGrp else dynamicBoundaryData['normals']
        dynamicBoundaryData['areas'] = torch.from_numpy(inGrp['boundaryArea'][:]).to(device = device, dtype = dtype) if 'boundaryArea' in inGrp else dynamicBoundaryData['areas']
        dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype) if 'boundaryVelocity' in inGrp else dynamicBoundaryData['velocities']
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype) if 'boundaryDensity' in inGrp else dynamicBoundaryData['densities']
        dynamicBoundaryData['supports'] = torch.from_numpy(inGrp['boundarySupport'][:]).to(device = device, dtype = dtype) if 'boundarySupport' in inGrp else dynamicBoundaryData['supports']
        dynamicBoundaryData['bodyIDs'] = torch.from_numpy(inGrp['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64) if 'boundaryBodyAssociation' in inGrp else dynamicBoundaryData['bodyIDs']
    else:
        dynamicBoundaryData = None

    state = loadGroup_testcaseII(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)


    iPriorKey = int(key) - hyperParameterDict['frameDistance']

    priorState = None
    if buildPriorState:
        if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
            priorState = copy.deepcopy(state)
        else:
            priorState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
            
    nextStates = []
    if buildNextState:
        if unrollLength == 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)]
        if unrollLength == 0 and hyperParameterDict['frameDistance'] != 0:
            nextStates = [copy.deepcopy(state)]
            warnings.warn('Unroll length is zero, but frame distance is not zero')
        if unrollLength != 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)] * unrollLength
        if unrollLength != 0 and hyperParameterDict['frameDistance'] != 0:
            for u in range(unrollLength):
                unrollKey = int(key) + hyperParameterDict['frameDistance'] * (u + 1)
                nextState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % unrollKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            




    return config, attributes, state, priorState, nextStates

def loadFrame_testcaseIV(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    attributes = {
        'support': inFile.attrs['support'],
        'targetNeighbors': inFile.attrs['numNeighbors'],
        'restDensity': 1,
        'dt': 0,
        'time': 0,
        'radius': 2 / inFile.attrs['nx'],
        'area': inFile.attrs['volume'],
    }
    inGrp = inFile['simulationData'][key]
    
    positions = torch.from_numpy(inGrp['x'][:]).to(device = device, dtype = dtype)
    areas = torch.from_numpy(inGrp['vols'][:]).to(device = device, dtype = dtype)

    config = {
        'domain':{
            'dim': 2,
            'minExtent': torch.tensor([-1, -1, -1], device = device, dtype = dtype),
            'maxExtent': torch.tensor([1, 1, 1], device = device, dtype = dtype),
            'periodicity': torch.tensor([True, True, True], device = device, dtype = torch.bool),
            'periodic': True
        },
        'neighborhood':{
            'scheme': 'compact',
            'verletScale': 1.0
        },
        'compute':{
            'device': device,
            'dtype': dtype,
            'precision': 'float32' if dtype == torch.float32 else 'float64',
        },
        'kernel':{
            'name': 'Wendland2',
            'targetNeighbors': 50,
            'function': getKernel('Wendland2')
        },
        'boundary':{
            'active': False
        },
        'fluid':{
            'rho0': 1,
            'cs': 20,
        },
        'particle':{
            'support': attributes['support']
        }
    }

    state = {
        'fluid': {
            'positions': positions,
            'velocities': torch.zeros_like(positions),
            'gravityAcceleration': torch.zeros_like(positions),
            'densities':  torch.from_numpy(inGrp['rho'][:]).to(device = device, dtype = dtype),
            'areas': areas,
            'masses': areas,
            'supports': torch.ones_like(areas) * attributes['support'],
            'indices': torch.arange(areas.shape[0], device = device, dtype = torch.int64),
            'numParticles': len(areas)
        },
        'boundary': None,
        'time': 0,
        'dt': 0,
        'timestep': 0,
    }

    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for k in additionalData:
        # if k in inGrp:
            # state['fluid'][k] = torch.from_numpy(inGrp[k][:]).to(device = device, dtype = dtype)
        # else:
            # warnings.warn('Additional data key %s not found in group' % k)

    priorState = None
    nextStates = [copy.deepcopy(state)]

    return config, attributes, state, priorState, nextStates



try:
    from diffSPH.v2.parameters import parseDefaultParameters, parseModuleParameters
    # from torchCompactRadius import radiusSearch
    hasDiffSPH = True
except ModuleNotFoundError:
    # from BasisConvolution.neighborhoodFallback.neighborhood import radiusSearch
    hasDiffSPH = False
    # pass

# from diffSPH.v2.parameters import parseDefaultParameters, parseModuleParameters
import copy

def parseSPHConfig(inFile, device, dtype):
    # if not hasDiffSPH:
        # raise ModuleNotFoundError('diffSPH is not installed, cannot parse SPH config')
    config = {}
    for key in inFile['config'].keys():
        config[key] = {}
        for subkey in inFile['config'][key].attrs.keys():
            # print(key,subkey)
            config[key][subkey] = inFile['config'][key].attrs[subkey]
        # print(key, config[key])

    if 'domain' in config:
        if 'minExtent' in config['domain']:
            config['domain']['minExtent'] = config['domain']['minExtent'].tolist()
        if 'maxExtent' in config['domain']:
            # print(config['domain']['maxExtent'])
            config['domain']['maxExtent'] = config['domain']['maxExtent'].tolist()
        if 'periodicity' in config['domain']:
            config['domain']['periodicity'] = config['domain']['periodicity'].tolist()
        if 'periodic' in config['domain']:
            config['domain']['periodic'] = bool(config['domain']['periodic'])
    config['compute']['device'] = device
    config['compute']['dtype'] = dtype
    config['simulation']['correctArea'] = False

    if hasDiffSPH:
        parseDefaultParameters(config)
        parseModuleParameters(config)
    
    return config

def loadGroup_newFormat(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    
    if 'boundaryInformation' in inFile:
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]
    else:
        dynamicBoundaryData = None

    # for k in inGrp.keys():
        # print(k, inGrp[k])

    rho = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    areas = torch.ones_like(rho) * inFile.attrs['area']
    state = {
        'fluid': {
            'positions': torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype) if 'fluidGravity' not in inFile.attrs else torch.from_numpy(inFile.attrs['fluidGravity']).to(device = device, dtype = dtype) * torch.ones(inGrp['fluidDensity'][:].shape[0]).to(device = device, dtype = dtype)[:,None],
            'densities': rho * inFile.attrs['restDensity'],
            'areas': areas,
            'masses': areas * inFile.attrs['restDensity'],
            'supports': torch.ones_like(rho) * inFile.attrs['support'],
            'indices': torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(rho)
        },
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state

def loadFrame_newFormat(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(key)

    inGrp = inFile['simulationExport'][key]

    # print(inFile.attrs.keys())
    # for k in inFile.attrs.keys():
        # print(k, inFile.attrs[k])

    config = parseSPHConfig(inFile, device, dtype)

    attributes = {
        'support': np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support'],
        'targetNeighbors': inFile.attrs['targetNeighbors'],
        'restDensity': inFile.attrs['restDensity'],
        'dt': inGrp.attrs['dt'],
        'time': inGrp.attrs['time'],
        'radius': inFile.attrs['radius'] if 'radius' in inFile.attrs else inGrp.attrs['radius'],
        'area': inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area'],
    }
    if 'boundaryInformation' in inFile:
        staticBoundaryData = {
                'indices': torch.arange(0, inFile['boundaryInformation']['boundaryPosition'].shape[0], device = device, dtype = torch.int64),
                'positions': torch.from_numpy(inFile['boundaryInformation']['boundaryPosition'][:]).to(device = device, dtype = dtype),
                'normals': torch.from_numpy(inFile['boundaryInformation']['boundaryNormals'][:]).to(device = device, dtype = dtype),
                'areas': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype),
                'masses': torch.from_numpy(inFile['boundaryInformation']['boundaryArea'][:]).to(device = device, dtype = dtype) * config['fluid']['rho0'],
                'velocities': torch.from_numpy(inFile['boundaryInformation']['boundaryVelocity'][:]).to(device = device, dtype = dtype),
                'densities': torch.from_numpy(inFile['boundaryInformation']['boundaryRestDensity'][:]).to(device = device, dtype = dtype),
                'supports': torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:]).to(device = device, dtype = dtype),
                'bodyIDs': torch.from_numpy(inFile['boundaryInformation']['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64),
                'numParticles': len(inFile['boundaryInformation']['boundaryPosition'][:]),
            } if 'boundaryInformation' in inFile else None
    else:
        staticBoundaryData = None

    if 'boundaryInformation' in inFile:
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]

        dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype) if 'boundaryPosition' in inGrp else dynamicBoundaryData['positions']
        dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype) if 'boundaryNormals' in inGrp else dynamicBoundaryData['normals']
        dynamicBoundaryData['areas'] = torch.from_numpy(inGrp['boundaryArea'][:]).to(device = device, dtype = dtype) if 'boundaryArea' in inGrp else dynamicBoundaryData['areas']
        dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype) if 'boundaryVelocity' in inGrp else dynamicBoundaryData['velocities']
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype) if 'boundaryDensity' in inGrp else dynamicBoundaryData['densities']
        dynamicBoundaryData['supports'] = torch.from_numpy(inGrp['boundarySupport'][:]).to(device = device, dtype = dtype) if 'boundarySupport' in inGrp else dynamicBoundaryData['supports']
        dynamicBoundaryData['bodyIDs'] = torch.from_numpy(inGrp['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64) if 'boundaryBodyAssociation' in inGrp else dynamicBoundaryData['bodyIDs']
    else:
        dynamicBoundaryData = None

    state = loadGroup_newFormat(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)

    iPriorKey = int(key) - hyperParameterDict['frameDistance']

    priorState = None
    if buildPriorState:
        if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
            priorState = copy.deepcopy(state)
        else:
            priorState = loadGroup_newFormat(inFile, inFile['simulationExport']['%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
            
    nextStates = []
    if buildNextState:
        if unrollLength == 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)]
        if unrollLength == 0 and hyperParameterDict['frameDistance'] != 0:
            nextStates = [copy.deepcopy(state)]
            warnings.warn('Unroll length is zero, but frame distance is not zero')
        if unrollLength != 0 and hyperParameterDict['frameDistance'] == 0:
            nextStates = [copy.deepcopy(state)] * unrollLength
        if unrollLength != 0 and hyperParameterDict['frameDistance'] != 0:
            for u in range(unrollLength):
                unrollKey = int(key) + hyperParameterDict['frameDistance'] * (u + 1)
                nextState = loadGroup_newFormat(inFile, inFile['simulationExport']['%05d' % unrollKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            




    return config, attributes, state, priorState, nextStates



def loadFrame(index, dataset, hyperParameterDict, unrollLength = 8):

    fileName, key, fileData, fileIndex, fileOffset = dataset[index] if isinstance(index, int) else index

    # print(fileName)
    # print(key)
    # print(fileData)
    # print(fileIndex)
    # print(fileOffset)

    inFile = h5py.File(fileName, 'r')
    try:
        if 'simulationExport' in inFile:
            attributes = {
                'support': None,
                'targetNeighbors': None,
                'restDensity': None,
                'dt': None,
                'time': None,
                'radius': None,
                'area': None,
            }
            state = {
                'fluid': {
                    'positions': None,
                    'velocities': None,
                    'gravityAcceleration': None,
                    'densities': None,
                    'areas': None,
                    'masses': None,
                    'supports': None,
                    'indices': None,
                    'numParticles': 0
                },
                'boundary':{
                    'positions': None,
                    'normals': None,
                    'areas': None,
                    'velocities': None,
                    'densities': None,
                    'supports': None,
                    'numParticles': 0
                },
                'time': 0,
                'dt': 0,
                'timestep': 0,
            }

            if 'config' in inFile: # New format
                return loadFrame_newFormat(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])

                raise ValueError('New format not supported')
            if 'config' not in inFile:
                if isTemporalData(inFile): # temporal old format data, test case II/III
                    return loadFrame_testcaseII(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])
                else:
                    raise ValueError('Unsupported Format for file')


            print(inFile['simulationExport'][key])
        else:
            # This should be test case I with flat 1D data
            if isTemporalData(inFile):
                return loadFrame_testcaseI(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])
            else:
                # print('Test case IV')
                return loadFrame_testcaseIV(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])


            print(inFile['simulationData'].keys())
        # print(inFile['simulationExport'])
    except Exception as e:
        inFile.close()
        raise e
    inFile.close()

