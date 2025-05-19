import torch
import numpy as np
import h5py
import warnings
from BasisConvolution.util.datautils import isTemporalData
from BasisConvolution.sph.kernels import getKernel
# import warnings

def computeSupport(area, targetNumNeighbors, dim):
    if dim == 1:
        return targetNumNeighbors * area
    if dim == 2:
        if (isinstance(targetNumNeighbors, int) or isinstance(targetNumNeighbors, float)) and not isinstance(area, torch.Tensor):
            return np.sqrt(targetNumNeighbors * area / np.pi)
        return torch.sqrt(targetNumNeighbors * area / np.pi)
    if dim == 3:
        return (3 * targetNumNeighbors * area / (4 * np.pi))**(1/3)
    else:
        raise ValueError('Unsupported dimension %d' % dim)

def loadAdditional(inGrp, state, additionalData, device, dtype):
    for dataKey in additionalData:
        if dataKey in inGrp:
            state[dataKey] = torch.from_numpy(inGrp[dataKey][:]).to(device = device, dtype = dtype)
        else:
            warnings.warn('Additional data key %s not found in group' % dataKey)
    return state

def loadFrame_testcaseI(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    attributes = {
        'support': inFile.attrs['particleSupport'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['baseArea'], hyperParameterDict['numNeighbors'], 1),
        'targetNeighbors':( inFile.attrs['particleSupport'] / inFile.attrs['particleRadius']) if hyperParameterDict['numNeighbors'] < 0 else hyperParameterDict['numNeighbors'],
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


    
    return config, attributes, state, [priorState], nextStates

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
    support = computeSupport(inGrp['fluidArea'][0], hyperParameterDict['numNeighbors'], 2) if hyperParameterDict['numNeighbors'] > 0 else inGrp['fluidSupport'][0]
    state = {
        'fluid': {
            'positions': torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype) if 'fluidGravity' not in inFile.attrs else torch.from_numpy(inFile.attrs['fluidGravity']).to(device = device, dtype = dtype) * torch.ones(inGrp['fluidDensity'][:].shape[0]).to(device = device, dtype = dtype)[:,None],
            'densities': torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype),
            'areas': areas,
            'masses': areas * inFile.attrs['restDensity'],
            'supports': torch.ones_like(areas) * support, #torch.from_numpy(inGrp['fluidSupport'][:]).to(device = device, dtype = dtype),
            'indices': torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(areas)
        },
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    if hyperParameterDict['normalizeDensity']:
        state['fluid']['densities'] = (state['fluid']['densities'] - 1) * inFile.attrs['restDensity']
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state


def loadFrame_testcaseII(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(key)

    inGrp = inFile['simulationExport'][key]
    support = np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support']
    if hyperParameterDict['numNeighbors'] > 0:
        support = computeSupport(inGrp['fluidArea'][0], hyperParameterDict['numNeighbors'], 2)
    attributes = {
        'support': support,
        'targetNeighbors': inFile.attrs['targetNeighbors'],
        'restDensity': inFile.attrs['restDensity'],
        'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
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
        },
        'shifting':{
            'CFL': 1.5
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
                'supports': torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:]).to(device = device, dtype = dtype) if hyperParameterDict['numNeighbors'] < 0 else torch.ones_like(torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:]).to(device = device, dtype = dtype)) * support,
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
        dynamicBoundaryData['supports'] = (torch.from_numpy(inGrp['boundarySupport'][:]).to(device = device, dtype = dtype) if hyperParameterDict['numNeighbors'] < 0 else torch.from_numpy(inFile['boundaryInformation']['boundarySupport'][:].to(device = device, dtype = dtype)) * support) if 'boundarySupport' in inGrp else dynamicBoundaryData['supports']
        dynamicBoundaryData['bodyIDs'] = torch.from_numpy(inGrp['boundaryBodyAssociation'][:]).to(device = device, dtype = torch.int64) if 'boundaryBodyAssociation' in inGrp else dynamicBoundaryData['bodyIDs']
    else:
        dynamicBoundaryData = None

    state = loadGroup_testcaseII(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)


    iPriorKey = int(key) - hyperParameterDict['frameDistance']


    priorStates = []
    # print(f'Loading prior states [{max(hyperParameterDict["historyLength"], 1)}]')
    for h in range(max(hyperParameterDict['historyLength'], 1)):
        priorState = None        
        iPriorKey = int(key) - hyperParameterDict['frameDistance'] * (h + 1)

        if buildPriorState or hyperParameterDict['adjustForFrameDistance']:
            if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
                priorState = copy.deepcopy(state)
            else:
                grp = inFile['simulationExport']['%05d' % iPriorKey] if '%05d' % iPriorKey in inFile['simulationExport'] else None
                # if grp is None:
                    # print('Key %s not found in file' % iPriorKey)
                priorState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
        # print('Loaded prior state %s' % iPriorKey)
        priorStates.append(priorState)

    # priorState = None
    # print(iPriorKey)
    # if buildPriorState:
    #     if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
    #         priorState = copy.deepcopy(state)
    #         print('copying state')
    #     else:
    #         priorState = loadGroup_testcaseII(inFile, inFile['simulationExport']['%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
    #         print('loading prior state')
            
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




    return config, attributes, state, priorStates, nextStates

def loadFrame_testcaseIV(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    print('loading frame', key, 'using testcase IV')
    support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['volume'], hyperParameterDict['numNeighbors'], 3)
    attributes = {
        'support': support,
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
    else:
        raise ModuleNotFoundError('diffSPH is not installed, cannot parse SPH config')
    
    return config

def loadGroup_newFormat(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    if staticFluidData is not None and int(key) == 0 or inGrp is None:
        state = {
            'fluid': staticFluidData,
            'boundary': staticBoundaryData,
            'time': 0.0,
            'dt': inFile['config']['timestep'].attrs['dt'] * hyperParameterDict['frameDistance'],
            'timestep': 0,
        }
        loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
        return state


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
    elif 'initial' in inFile:
        dynamicBoundaryData = {} if staticBoundaryData is not None else None
        if staticBoundaryData is not None:
            for k in staticBoundaryData.keys():
                if isinstance(staticBoundaryData[k], torch.Tensor):
                    dynamicBoundaryData[k] = staticBoundaryData[k].clone()
                else:
                    dynamicBoundaryData[k] = staticBoundaryData[k]

        if 'boundaryDensity' in inGrp:
            dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype)
        if 'boundaryVelocity' in inGrp:
            dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype)
        if 'boundaryPosition' in inGrp:
            dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype)
        if 'boundaryNormals' in inGrp:
            dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype)
    else:
        dynamicBoundaryData = None
    if 'boundaryDensity' in inGrp:
        dynamicBoundaryData['densities'] = torch.from_numpy(inGrp['boundaryDensity'][:]).to(device = device, dtype = dtype)
    if 'boundaryVelocity' in inGrp:
        dynamicBoundaryData['velocities'] = torch.from_numpy(inGrp['boundaryVelocity'][:]).to(device = device, dtype = dtype)
    if 'boundaryPosition' in inGrp:
        dynamicBoundaryData['positions'] = torch.from_numpy(inGrp['boundaryPosition'][:]).to(device = device, dtype = dtype)
    if 'boundaryNormals' in inGrp:
        dynamicBoundaryData['normals'] = torch.from_numpy(inGrp['boundaryNormals'][:]).to(device = device, dtype = dtype)


    fluidState = {}

    if staticFluidData is not None:
        for k in staticFluidData.keys():
            if isinstance(staticFluidData[k], torch.Tensor):
                fluidState[k] = staticFluidData[k].clone()
            else:
                fluidState[k] = staticFluidData[k]
    
    if 'fluidPosition' in inGrp:
        fluidState['positions'] = torch.from_numpy(inGrp['fluidPosition'][:]).to(device = device, dtype = dtype)
    if 'fluidVelocity' in inGrp:
        fluidState['velocities'] = torch.from_numpy(inGrp['fluidVelocity'][:]).to(device = device, dtype = dtype)
    if 'fluidDensity' in inGrp:
        fluidState['densities'] = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    if 'fluidGravity' in inGrp:
        fluidState['gravityAcceleration'] = torch.from_numpy(inGrp['fluidGravity'][:]).to(device = device, dtype = dtype)
    
    support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    rho = fluidState['densities']
    areas = torch.ones_like(rho) * inFile.attrs['area']

    fluidState['densities'] = rho #- rho.mean()#* inFile.attrs['restDensity']
    if hyperParameterDict['normalizeDensity']:
        fluidState['densities'] = (fluidState['densities'] - 1.0) * inFile.attrs['restDensity']
    fluidState['areas'] = areas
    fluidState['masses'] = areas * inFile.attrs['restDensity']
    fluidState['supports'] = torch.ones_like(rho) * support
    fluidState['indices'] = torch.from_numpy(inGrp['UID'][:]).to(device = device, dtype = torch.int64)
    fluidState['numParticles'] = len(rho)

    # for k in inGrp.keys():
        # print(k, inGrp[k])

    # support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    # rho = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    # areas = torch.ones_like(rho) * inFile.attrs['area']
    state = {
        'fluid': fluidState,
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state

def loadFrame_newFormat(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(f'Loading frame {key} from {fileName} ')
    # print(key)

    if 'initial' in inFile:
        targetNeighbors = inFile.attrs['targetNeighbors']

        staticFluidData = {
            'positions': torch.from_numpy(inFile['initial']['fluid']['positions'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initial']['fluid']['velocities'][:]).to(device = device, dtype = dtype),
            'gravityAcceleration': torch.zeros_like(torch.from_numpy(inFile['initial']['fluid']['velocities'][:]).to(device = device, dtype = dtype)),
            'densities': torch.from_numpy(inFile['initial']['fluid']['densities'][:]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initial']['fluid']['areas'][:]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initial']['fluid']['masses'][:]).to(device = device, dtype = dtype),
            'supports': computeSupport(torch.from_numpy(inFile['initial']['fluid']['areas'][:]).to(device = device, dtype = dtype), targetNeighbors, 2),
            'indices': torch.from_numpy(inFile['initial']['fluid']['UID'][:]).to(device = device, dtype = torch.int32),
            'numParticles': len(inFile['initial']['fluid']['positions'][:]),                                  
        }

        config = parseSPHConfig(inFile, device, dtype)
        area = inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area']
        support = np.max(staticFluidData['supports'].detach().cpu().numpy()) if 'support' not in inFile.attrs else inFile.attrs['support']
        if hyperParameterDict['numNeighbors'] > 0:
            support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
        attributes = {
            'support': support,
            'targetNeighbors': inFile.attrs['targetNeighbors'],
            'restDensity': inFile.attrs['restDensity'],
            'dt': config['timestep']['dt'] * hyperParameterDict['frameDistance'],
            'time': 0.0,
            'radius': inFile.attrs['radius'],
            'area': area,
        }
        if key == '00000':
            inGrp = None
        else:
            inGrp = inFile['simulationExport'][key] 

    else:
        staticFluidData = None
        inGrp = inFile['simulationExport'][key]

        # print(inFile.attrs.keys())
        # for k in inFile.attrs.keys():
            # print(k, inFile.attrs[k])

        config = parseSPHConfig(inFile, device, dtype)
        area = inFile.attrs['radius'] **2 if 'area' not in inFile.attrs else inFile.attrs['area']
        support = np.max(inGrp['fluidSupport'][:]) if 'support' not in inFile.attrs else inFile.attrs['support']
        if hyperParameterDict['numNeighbors'] > 0:
            support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
        attributes = {
            'support': support,
            'targetNeighbors': inFile.attrs['targetNeighbors'],
            'restDensity': inFile.attrs['restDensity'],
            'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
            'time': inGrp.attrs['time'],
            'radius': inFile.attrs['radius'] if 'radius' in inFile.attrs else inGrp.attrs['radius'],
            'area': area,
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
    elif 'initial' in inFile:
        staticBoundaryData = {
            'indices': torch.from_numpy(inFile['initial']['boundary']['UID'][:]).to(device = device, dtype = torch.int64),
            'positions': torch.from_numpy(inFile['initial']['boundary']['positions'][:]).to(device = device, dtype = dtype),
            'normals': torch.from_numpy(inFile['initial']['boundary']['normals'][:]).to(device = device, dtype = dtype),
            'distances': torch.from_numpy(inFile['initial']['boundary']['distances'][:]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initial']['boundary']['areas'][:]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initial']['boundary']['masses'][:]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initial']['boundary']['velocities'][:]).to(device = device, dtype = dtype),
            'densities': torch.from_numpy(inFile['initial']['boundary']['densities'][:]).to(device = device, dtype = dtype),
            'supports': computeSupport(torch.from_numpy(inFile['initial']['boundary']['areas'][:]).to(device = device, dtype = dtype), inFile.attrs['targetNeighbors'], 2),
            'bodyIDs': torch.from_numpy(inFile['initial']['boundary']['bodyIDs'][:]).to(device = device, dtype = torch.int64),
            'numParticles': len(inFile['initial']['boundary']['UID'][:]),

        } if 'boundary' in inFile['initial'] else None
    else:
        staticBoundaryData = None

    # if 'boundaryInformation' in inFile:
    #     dynamicBoundaryData = {}
    #     for k in staticBoundaryData.keys():
    #         if isinstance(staticBoundaryData[k], torch.Tensor):
    #             dynamicBoundaryData[k] = staticBoundaryData[k].clone()
    #         else:
    #             dynamicBoundaryData[k] = staticBoundaryData[k]


    # else:
    #     dynamicBoundaryData = None

    state = loadGroup_newFormat(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)


    priorStates = []
    # print(f'Loading prior states [{max(hyperParameterDict["historyLength"], 1)}]')
    historyLength = max(hyperParameterDict['historyLength'], 1)
    if 'dt' in hyperParameterDict['fluidFeatures'] or 'ddt' in hyperParameterDict['fluidFeatures'] or 'diff' in hyperParameterDict['fluidFeatures']:
        historyLength += 1
    for h in range(historyLength):
        priorState = None        
        iPriorKey = int(key) - hyperParameterDict['frameDistance'] * (h + 1)

        if buildPriorState or hyperParameterDict['adjustForFrameDistance']:
            if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
                priorState = copy.deepcopy(state)
            else:
                grp = inFile['simulationExport']['%05d' % iPriorKey] if '%05d' % iPriorKey in inFile['simulationExport'] else None
                # if grp is None:
                    # print('Key %s not found in file' % iPriorKey)
                priorState = loadGroup_newFormat(inFile, grp, staticFluidData, staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
        # print('Loaded prior state %s' % iPriorKey)
        priorStates.append(priorState)

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
                nextState = loadGroup_newFormat(inFile, inFile['simulationExport']['%05d' % unrollKey], staticFluidData, staticBoundaryData, fileName, unrollKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            

    # if hyperParameterDict['adjustForFrameDistance']:

    config['particle']['support'] = support

    # print('Loaded frame %s' % key)

    return config, attributes, state, priorStates, nextStates

from BasisConvolution.sph.kernels import getKernel
import numpy as np

def loadAdditional(inGrp, state, additionalData, device, dtype):
    for dataKey in additionalData:
        if dataKey in inGrp:
            state[dataKey] = torch.from_numpy(inGrp[dataKey][:]).to(device = device, dtype = dtype)
        else:
            warnings.warn('Additional data key %s not found in group' % dataKey)
    return state

def loadGroup_waveEqn(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    dynamicBoundaryData = None

    rho = torch.from_numpy(inFile['gridState']['densities'][:]).to(device = device, dtype = dtype)
    areas = torch.ones_like(rho) * inFile.attrs['dx']
    state = {
        'fluid': {
            'positions': torch.from_numpy(inFile['gridState']['positions'][:]).to(device = device, dtype = dtype),
            'densities': rho * 1,
            'areas': areas,
            'masses': areas * 1,
            'supports': torch.from_numpy(inFile['gridState']['supports'][:]).to(device = device, dtype = dtype),
            'indices': torch.arange(len(rho), device = device, dtype = torch.int64),
            'numParticles': len(rho)
        },
        'boundary': None,
        'time': inGrp.attrs['time'],
        'dt': inFile.attrs['dt']
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state

def loadFrame_waveEqn(inFile, fileName, key_, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print('Loading frame %s' % key_)
    # print(key)
    if isinstance(key_, str) and '_' in key_:
        key = int(key_.split('_')[1])
    else:
        key = int(key_)

    inGrp = inFile['simulation'][key_]

    # print(inGrp.keys())
    # for k in inFile.attrs.keys():
    #     print(k, inFile.attrs[k])
    # pass

    # print(inFile.attrs.keys())
    # for k in inFile.attrs.keys():
        # print(k, inFile.attrs[k])

    config = {
        'domain':{
            'dim': 2,
            'minExtent': torch.tensor([-1, -1], device = device, dtype = dtype),
            'maxExtent': torch.tensor([1, 1], device = device, dtype = dtype),
            'periodic': torch.tensor([inFile.attrs['periodic'], inFile.attrs['periodic']], device = device, dtype = torch.bool),
            'periodicity': torch.tensor([inFile.attrs['periodic'], inFile.attrs['periodic']], device = device, dtype = torch.bool),
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
            'name': inFile.attrs['kernel'],
            'targetNeighbors': inFile.attrs['resampledNeighbors'],
            'function': getKernel(inFile.attrs['kernel'])
        },
        'boundary':{
            'active': False
        },
        'fluid':{
            'rho0': 1,
            'cs': 20,
        },
        'particle':{
            'support': inFile.attrs['h']
        }
    }

    attributes = {
        'support': inFile.attrs['h'],
        'targetNeighbors': inFile.attrs['resampledNeighbors'],
        'restDensity': 1,
        'dt': inFile.attrs['dt'],
        'time': inGrp.attrs['time'],
        'radius': inFile.attrs['radius'],
        'area': inFile.attrs['dx'],
    }
    staticBoundaryData = None

    state = loadGroup_waveEqn(inFile, inGrp, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)

    iPriorKey = int(key) - hyperParameterDict['frameDistance']

    priorStates = []
    for h in hyperParameterDict['historyLength']:
        priorState = None        
        if buildPriorState or hyperParameterDict['adjustForFrameDistance']:
            if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
                priorState = copy.deepcopy(state)
            else:
                priorState = loadGroup_waveEqn(inFile, inFile['simulation']['timestep_%05d' % iPriorKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
            priorStates.append(priorState)

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
                nextState = loadGroup_waveEqn(inFile, inFile['simulation']['timestep_%05d' % unrollKey], staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            

    # if hyperParameterDict['adjustForFrameDistance']:

    return config, attributes, state, priorState, nextStates



def loadGroup_cuMath(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    if staticFluidData is not None and int(key) == 0 or inGrp is None:
        state = {
            'fluid': staticFluidData,
            'boundary': staticBoundaryData,
            'time': 0.0,
            'dt': inFile.attrs['targetDt'] * hyperParameterDict['frameDistance'],
            'timestep': 0,
        }
        loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
        return state

    kinds = inGrp['kinds'][:] if 'kinds' in inGrp else inFile['initialState']['kinds'][:]
    hasBoundary = np.any(kinds == 1)
    if hasBoundary:
        boundaryMask = kinds == 1
        dynamicBoundaryData = {}
        for k in staticBoundaryData.keys():
            if isinstance(staticBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = staticBoundaryData[k].clone()
            else:
                dynamicBoundaryData[k] = staticBoundaryData[k]

        for k in staticFluidData.keys():
            if isinstance(dynamicBoundaryData[k], torch.Tensor):
                dynamicBoundaryData[k] = torch.from_numpy(inGrp[k][boundaryMask]).to(device = device, dtype = dtype) if k in inGrp else dynamicBoundaryData[k]
    else:
        dynamicBoundaryData = None

    fluidState = {}

    if staticFluidData is not None:
        for k in staticFluidData.keys():
            if isinstance(staticFluidData[k], torch.Tensor):
                fluidState[k] = staticFluidData[k].clone()
            else:
                fluidState[k] = staticFluidData[k]
    fluidMask = kinds == 0
    for k in staticFluidData.keys():
        if isinstance(fluidState[k], torch.Tensor):
            fluidState[k] = torch.from_numpy(inGrp[k][fluidMask]).to(device = device, dtype = dtype) if k in inGrp else fluidState[k]
    fluidState['numParticles'] = len(fluidState['densities'])

    # for k in inGrp.keys():
        # print(k, inGrp[k])

    # support = inFile.attrs['support'] if hyperParameterDict['numNeighbors'] < 0 else computeSupport(inFile.attrs['area'], hyperParameterDict['numNeighbors'], 2)
    # rho = torch.from_numpy(inGrp['fluidDensity'][:]).to(device = device, dtype = dtype)
    # areas = torch.ones_like(rho) * inFile.attrs['area']
    state = {
        'fluid': fluidState,
        'boundary': dynamicBoundaryData if dynamicBoundaryData is not None else staticBoundaryData,
        'time': inGrp.attrs['time'],
        'dt': inGrp.attrs['dt'] * hyperParameterDict['frameDistance'],
        'timestep': inGrp.attrs['timestep'],
    }
    loadAdditional(inGrp, state['fluid'], additionalData, device, dtype)
    # for dataKey in additionalData:
        # state['fluid'][dataKey] = torch.from_numpy(np.array(inGrp[dataKey])).to(device = device, dtype = dtype)
    
    return state

def loadFrame_cuMath(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = 8, device = 'cpu', dtype = torch.float32, additionalData = [], buildPriorState = True, buildNextState = True):
    # print(f'Loading frame {key} from {fileName} ')
    # print(key)

    initialKinds = inFile['initialState']['fluid']['kinds'][:]
    fluidMask = initialKinds == 0
    boundaryMask = initialKinds == 1


    staticFluidData = {
        'positions': torch.from_numpy(inFile['initialState']['fluid']['positions'][fluidMask]).to(device = device, dtype = dtype),
        'velocities': torch.from_numpy(inFile['initialState']['fluid']['velocities'][fluidMask]).to(device = device, dtype = dtype),
        # 'gravityAcceleration': torch.zeros_like(torch.from_numpy(inFile['initialState']['fluid']['velocities'][fluidMask]).to(device = device, dtype = dtype)),
        'densities': torch.from_numpy(inFile['initialState']['fluid']['densities'][fluidMask]).to(device = device, dtype = dtype),
        'areas': torch.from_numpy(inFile['initialState']['fluid']['areas'][fluidMask]).to(device = device, dtype = dtype),
        'masses': torch.from_numpy(inFile['initialState']['fluid']['masses'][fluidMask]).to(device = device, dtype = dtype),
        'supports': torch.from_numpy(inFile['initialState']['fluid']['supports'][fluidMask]).to(device = device, dtype = dtype),
        'indices': torch.from_numpy(inFile['initialState']['fluid']['UIDs'][fluidMask]).to(device = device, dtype = torch.int32),
        'numParticles': len(inFile['initialState']['fluid']['positions'][fluidMask]),                                  
    }

    area = np.max(staticFluidData['areas'].detach().cpu().numpy())
    support = np.max(staticFluidData['supports'].detach().cpu().numpy())
    if hyperParameterDict['numNeighbors'] > 0:
        support = computeSupport(area, hyperParameterDict['numNeighbors'], 2)
    attributes = {
        'support': support,
        'targetNeighbors': inFile.attrs['targetNeighbors'],
        'restDensity': inFile.attrs['rho0'],
        'dt': inFile.attrs['targetDt'] * hyperParameterDict['frameDistance'],
        'time': 0.0,
        'radius': inFile.attrs['dx'],
        'area': area,
    }
    if key == '000000':
        inGrp = None
    else:
        inGrp = inFile['simulationData'][key] 


    initialKinds = inFile['initialState']['boundary']['kinds'][:]
    fluidMask = initialKinds == 0
    boundaryMask = initialKinds == 1

    staticBoundaryData = {
            'indices': torch.from_numpy(inFile['initialState']['boundary']['UIDs'][boundaryMask]).to(device = device, dtype = torch.int64),
            'positions': torch.from_numpy(inFile['initialState']['boundary']['positions'][boundaryMask]).to(device = device, dtype = dtype),
            'normals': torch.from_numpy(inFile['initialState']['boundary']['normals'][boundaryMask]).to(device = device, dtype = dtype),
            'areas': torch.from_numpy(inFile['initialState']['boundary']['areas'][boundaryMask]).to(device = device, dtype = dtype),
            'masses': torch.from_numpy(inFile['initialState']['boundary']['masses'][boundaryMask]).to(device = device, dtype = dtype),
            'velocities': torch.from_numpy(inFile['initialState']['boundary']['velocities'][boundaryMask]).to(device = device, dtype = dtype),
            'densities': torch.from_numpy(inFile['initialState']['boundary']['densities'][boundaryMask]).to(device = device, dtype = dtype),
            'supports': torch.from_numpy(inFile['initialState']['boundary']['supports'][boundaryMask]).to(device = device, dtype = dtype),
            'bodyIDs': torch.from_numpy(inFile['initialState']['boundary']['materials'][boundaryMask]).to(device = device, dtype = torch.int64),
            'numParticles': len(inFile['initialState']['boundary']['positions'][boundaryMask]),
        } if np.sum(boundaryMask)>0  else None

    # print(staticBoundaryData['numParticles'])
    # print(f'Fluid Particles: {staticFluidData["numParticles"]}, Boundary Particles: {staticBoundaryData["numParticles"] if staticBoundaryData is not None else 0}')

    # if 'boundaryInformation' in inFile:
    #     dynamicBoundaryData = {}
    #     for k in staticBoundaryData.keys():
    #         if isinstance(staticBoundaryData[k], torch.Tensor):
    #             dynamicBoundaryData[k] = staticBoundaryData[k].clone()
    #         else:
    #             dynamicBoundaryData[k] = staticBoundaryData[k]


    # else:
    #     dynamicBoundaryData = None

    state = loadGroup_cuMath(inFile, inGrp, staticFluidData, staticBoundaryData, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = buildPriorState, buildNextState = buildNextState)


    priorStates = []
    # print(f'Loading prior states [{max(hyperParameterDict["historyLength"], 1)}]')
    historyLength = max(hyperParameterDict['historyLength'], 1)
    if 'dt' in hyperParameterDict['fluidFeatures'] or 'ddt' in hyperParameterDict['fluidFeatures'] or 'diff' in hyperParameterDict['fluidFeatures']:
        historyLength += 1
    for h in range(historyLength):
        priorState = None        
        iPriorKey = int(key) - hyperParameterDict['frameDistance'] * (h + 1)

        if buildPriorState or hyperParameterDict['adjustForFrameDistance']:
            if iPriorKey < 0 or hyperParameterDict['frameDistance'] == 0:
                priorState = copy.deepcopy(state)
            else:
                grp = inFile['simulationData']['%06d' % iPriorKey] if '%06d' % iPriorKey in inFile['simulationData'] else None
                # if grp is None:
                    # print('Key %s not found in file' % iPriorKey)
                priorState = loadGroup_cuMath(inFile, grp, staticFluidData, staticBoundaryData, fileName, iPriorKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)
        # print('Loaded prior state %s' % iPriorKey)
        priorStates.append(priorState)

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
                nextState = loadGroup_cuMath(inFile, inFile['simulationData']['%05d' % unrollKey], staticFluidData, staticBoundaryData, fileName, unrollKey, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, device = device, dtype = dtype, additionalData = additionalData, buildPriorState = False, buildNextState = False)                
                nextStates.append(nextState)            

    # if hyperParameterDict['adjustForFrameDistance']:

    # config['particle']['support'] = support

    # print('Loaded frame %s' % key)

    config = {
        'domain':{
            'dim': 2,
            'minExtent': torch.tensor(inFile.attrs['domainMin'], device = device, dtype = dtype),
            'maxExtent': torch.tensor(inFile.attrs['domainMax'], device = device, dtype = dtype),
            'periodic': torch.tensor(inFile.attrs['domainPeriodic'], device = device, dtype = torch.bool),
            'periodicity': torch.tensor(inFile.attrs['domainPeriodic'], device = device, dtype = torch.bool),
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
            'name': inFile.attrs['kernel'],
            'targetNeighbors': inFile.attrs['targetNeighbors'],
            'function': getKernel(inFile.attrs['kernel'])
        },
        'boundary':{
            'active': np.any(initialKinds == 1),
        },
        'fluid':{
            'rho0': inFile.attrs['rho0'],
            'cs': inFile.attrs['c_s'],
        },
        'particle':{
            'support': inFile.attrs['support']
        }
    }
    return config, attributes, state, priorStates, nextStates


def loadFrame(index, dataset, hyperParameterDict, unrollLength = 8):

    fileName, key, fileData, fileIndex, fileOffset = dataset[index] if isinstance(index, int) else index

    # print(f'Loading frame {fileName.split("/")[-1]}:"{key}" w/ unroll length {unrollLength}')

    # print(fileName)
    # print(key)
    # print(fileData)
    # print(fileIndex)
    # print(fileOffset)

    inFile = h5py.File(fileName, 'r')
    try:
        if dataset.fileFormat == 'waveEquation':
            return loadFrame_waveEqn(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])
        if 'simulationData'in inFile and 'simulator' in inFile.attrs:
            if dataset.fileFormat == 'cuMath':
                # print('cuMath format')
                return loadFrame_cuMath(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])

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
                # print('New format')
                return loadFrame_newFormat(inFile, fileName, key, fileData, fileIndex, fileOffset, dataset, hyperParameterDict, unrollLength = unrollLength, additionalData= [] if 'additionalData' not in hyperParameterDict else hyperParameterDict['additionalData'], device = hyperParameterDict['device'], dtype = hyperParameterDict['dtype'])

                raise ValueError('New format not supported')
            if 'config' not in inFile:
                # print('Old format')
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

