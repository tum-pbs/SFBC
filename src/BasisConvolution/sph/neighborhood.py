from typing import List, Optional
import torch
@torch.jit.script
def mod(x, min : float, max : float):
    return torch.where(torch.abs(x) > (max - min) / 2, torch.sgn(x) * ((torch.abs(x) + min) % (max - min) + min), x)
@torch.jit.script
def countUniqueEntries(indices, positions):
    """
    Count the number of unique entries in the indices tensor and return the unique indices and their counts.

    Args:
        indices (torch.Tensor): Tensor containing the indices.
        positions (torch.Tensor): Tensor containing the positions.

    Returns:
        tuple: A tuple containing the unique indices and their counts.
    """
    ii, nit = torch.unique(indices, return_counts=True)
    ni = torch.zeros(positions.shape[0], dtype=nit.dtype, device=positions.device)
    ni[ii] = nit
    return ii, ni


from torch.profiler import record_function

try:
    from torchCompactRadius.neighborhood import neighborSearch, buildDataStructure, neighborSearchExisting, radiusSearch
    hasCompactRadius = True
except:
    from BasisConvolution.neighborhoodFallback.neighborhood import radiusSearch
    hasCompactRadius = False


def getNeighborSearchDataStructure(particleState, config):
    if hasCompactRadius:
        return buildDataStructure(particleState['positions'], particleState['supports'] * config['neighborhood']['verletScale'], particleState['supports'].max(), config['domain']['minExtent'], config['domain']['maxExtent'], config['domain']['periodic'], particleState['positions'].shape[0], verbose = False)
    else:
        return None


def getPeriodicPosition(x, config):
    periodic = config['domain']['periodicity']
    minDomain = config['domain']['minExtent']
    maxDomain = config['domain']['maxExtent']

    # dim = config['domain']['dim']
    # kernel = config['kernel']['function']

    with record_function("NeighborSearch [adjust Domain]"):
        periodicity = torch.tensor([False] * x.shape[1], dtype = torch.bool).to(x.device)
        if isinstance(periodic, torch.Tensor):
            periodicity = periodic
        if isinstance(periodic, bool):
            periodicity = torch.tensor([periodic] * x.shape[1], dtype = torch.bool).to(x.device)
        # if minDomain is not None and isinstance(minDomain, list):
            # minD = torch.tensor(minDomain).to(x.device).type(x.dtype)
        # else:
        minD = minDomain
        # if maxDomain is not None and isinstance(minDomain, list):
            # maxD = torch.tensor(maxDomain).to(x.device).type(x.dtype)
        # else:
        maxD = maxDomain
    with record_function("NeighborSearch [periodicity]"):
        return torch.stack([x[:,i] if not periodic_i else torch.remainder(x[:,i] - minDomain[i], maxDomain[i] - minDomain[i]) + minDomain[i] for i, periodic_i in enumerate(periodicity)], dim = 1)
    

def computeNeighborhood(neighborhood, pos_x, pos_y, h_i, h_j, config, mode):
    i,j = neighborhood

    hij = None
    if mode == 'scatter':
        hij = h_j[j]
    elif mode == 'gather':
        hij = h_i[i]
    elif mode == 'symmetric':
        hij = 0.5 * (h_i[i] + h_j[j])


    xij = pos_x[i,:] - pos_y[j,:]
    xij = torch.stack([xij[:,i] if not periodic_i else mod(xij[:,i], config['domain']['minExtent'][i], config['domain']['maxExtent'][i]) for i, periodic_i in enumerate(config['domain']['periodicity'])], dim = -1)
    # rij = torch.sqrt((xij**2).sum(-1))
    rij = torch.linalg.norm(xij, dim = -1)
    xij = xij / (rij + 1e-7).view(-1,1)

    rij = rij / hij

    mask = rij <= 1.0
    rij = rij[mask]
    xij = xij[mask,:]
    iFiltered = i[mask]
    jFiltered = j[mask]
    hijFiltered = hij[mask]

    return (iFiltered, jFiltered), hijFiltered, rij, xij

def updateDataStructure(referenceState, config, priorDatastructure, verbose = False):
    if not hasCompactRadius:
        return priorDatastructure, False
    if priorDatastructure['referencePositions'].shape != referenceState['positions'].shape:
        if verbose:
            print(f'Updating Datastructure because of shape mismatch ({priorDatastructure["referencePositions"].shape} != {referenceState["positions"].shape})')
        return getNeighborSearchDataStructure(referenceState, config), True

    maxDistance = torch.linalg.norm(priorDatastructure['referencePositions'] - referenceState['positions'], dim = -1).max()
    minSupport = priorDatastructure['referenceSupports'].min()

    if maxDistance * 2 > (config['neighborhood']['verletScale'] - 1) * minSupport:    
        if verbose:
            print(f'Updating Datastructure because of distance mismatch ({maxDistance} > {(config["neighborhood"]["verletScale"] - 1) * minSupport})')
        return getNeighborSearchDataStructure(referenceState, config), True

    priorDatastructure['sortedPositions'] = referenceState['positions'][priorDatastructure['sortIndex'],:]
    priorDatastructure['sortedSupports'] = referenceState['supports'][priorDatastructure['sortIndex']]
    return priorDatastructure, False


def neighborSearch(queryState, referenceState, config, computeKernels = True, priorState = None, neighborDatastructure = None, verbose = False):
    if neighborDatastructure is None:
        if verbose:
            print(f'Building Datastructure because prior state is None')
        neighborDatastructure = getNeighborSearchDataStructure(referenceState, config)
        dirty = True
    else:
        neighborDatastructure, dirty = updateDataStructure(referenceState, config, neighborDatastructure, verbose)
    if neighborDatastructure is None:

        if verbose:
            print(f'Building Neighborlist because prior state is None')
        dirty = True
    
    if priorState is not None:
        if priorState['initialPositions'][0].shape != queryState['positions'].shape:
            if verbose:
                print(f'Updating Neighborsearch because of shape mismatch ({priorState["initialPositions"][0].shape} != {queryState["positions"].shape})')
            dirty = True
        else:

            maxDistance = torch.linalg.norm(priorState['initialPositions'][0] - queryState['positions'], dim = -1).max()
            minSupport = queryState['supports'].min()
            if maxDistance * 2 > (config['neighborhood']['verletScale'] - 1) * minSupport:    
                if verbose:
                    print(f'Updating Neighborsearch because of distance mismatch ({maxDistance} > {(config["neighborhood"]["verletScale"] - 1) * minSupport})')
                dirty = True
        
    else:
        if verbose:
            print(f'Updating Neighborsearch because priorState is None')
        dirty = True


    if dirty:
        if hasCompactRadius:
            neighborhood = neighborSearchExisting(queryState['positions'], queryState['supports'], neighborDatastructure, 'scatter', 1, 'cpp')
        else:
            neighborhood = radiusSearch(queryState['positions'], referenceState['positions'], support = queryState['supports'] * config['neighborhood']['verletScale'], mode = 'gather', domainMin = config['domain']['minExtent'], domainMax = config['domain']['maxExtent'], periodicity = config['domain']['periodicity'], algorithm = 'naive')
        numNeighbors_full = countUniqueEntries(neighborhood[0], queryState['positions'])[1].to(torch.int32)
        neighborOffsets_full = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors_full.device), torch.cumsum(numNeighbors_full, dim = 0).to(torch.int32)))[:-1]\

    
    else:
        neighborhood = priorState['fullIndices']
        numNeighbors_full = priorState['fullNumNeighbors']
        neighborOffsets_full = priorState['fullNeighborOffsets']

    pos_x = getPeriodicPosition(queryState['positions'], config)
    pos_y = getPeriodicPosition(referenceState['positions'], config)

    h_i = queryState['supports']
    h_j = referenceState['supports']

    neighborhood_actual, hij_actual, rij, xij = computeNeighborhood(neighborhood, pos_x, pos_y, h_i, h_j, config, 'scatter')

    numNeighbors = countUniqueEntries(neighborhood_actual[0], pos_x)[1].to(torch.int32)
    neighborOffsets = torch.hstack((torch.tensor([0], dtype = torch.int32, device = numNeighbors.device), torch.cumsum(numNeighbors, dim = 0).to(torch.int32)))[:-1]\

    # numNeighbors, neighborOffsets, i, j, rij, xij, hij_actual = computeNeighborhood_cpp(
    # neighborhood, 
    # pos_x.shape[0],
    # numNeighbors_full, neighborOffsets_full,
    # (pos_x, pos_y), 
    # (h_i, h_j),
    # config['domain']['minExtent'], config['domain']['maxExtent'], config['domain']['periodicity'])

    # neighborhood_actual = (i, j)
# torch.cuda.synchronize()

    if dirty:
        neighborDict = {
            'indices': neighborhood_actual,
            'fullIndices': neighborhood,
            'distances': rij,
            'vectors': xij,
            'supports': hij_actual,
            'initialPositions': (queryState['positions'], referenceState['positions']),
            'numNeighbors': numNeighbors,
            'neighborOffsets': neighborOffsets,
            'fullNumNeighbors': numNeighbors_full,
            'fullNeighborOffsets': neighborOffsets_full
        }
    else:
        neighborDict = {
            'indices': neighborhood_actual,
            'fullIndices': priorState['fullIndices'],
            'distances': rij,
            'vectors': xij,
            'supports': hij_actual,
            'initialPositions': priorState['initialPositions'],
            'numNeighbors': numNeighbors,
            'neighborOffsets': neighborOffsets,
            'fullNumNeighbors': numNeighbors_full,
            'fullNeighborOffsets': neighborOffsets_full
        }


    if computeKernels:
        dim = config['domain']['dim']
        kernel = config['kernel']['function']
        neighborDict['kernels'] = kernel.kernel(rij, hij_actual, dim)
        neighborDict['gradients'] = kernel.kernelGradient(rij, xij, hij_actual, dim) 
    
    return neighborDatastructure, neighborDict
