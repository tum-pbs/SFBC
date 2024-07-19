import torch


@torch.jit.script
def countUnique(indices, numEntries : int):
    """
    Count the number of unique entries in the indices tensor and return the unique indices and their counts.

    Args:
        indices (torch.Tensor): Tensor containing the indices.
        positions (torch.Tensor): Tensor containing the positions.

    Returns:
        tuple: A tuple containing the unique indices and their counts.
    """
    ii, nit = torch.unique(indices, return_counts=True)
    ni = torch.zeros(numEntries, dtype=nit.dtype, device=indices.device)
    ni[ii] = nit
    return ii, ni

# from BasisConvolution.detail.radius import radiusSearch
# from diffSPH.v2.modules.neighborhood import neighborSearchVerlet
from BasisConvolution.sph.neighborhood import neighborSearch

def neighborSearchStates(stateA, stateB, config, augR = None, priorState = None, priorDatastructure = None, computeKernels = False):
    x = stateA['positions']
    y = stateB['positions']

    domainMin = config['domain']['minExtent']
    domainMax = config['domain']['maxExtent']
    periodicity = config['domain']['periodic']
    tempPositionsA = stateA['positions']
    tempPositionsB = stateB['positions']

    if augR is not None:
        x = x @ augR.T
        y = y @ augR.T

    stateA['positions'] = x
    stateB['positions'] = y

    ds, neighborDict = neighborSearch(stateA, stateB, config, 
            computeKernels = computeKernels, 
            priorState = priorState,
            neighborDatastructure = priorDatastructure,
            verbose = False)

    stateA['positions'] = tempPositionsA
    stateB['positions'] = tempPositionsB

    if augR is not None:
        neighborDict['vectors'] = neighborDict['vectors'] @ augR
        if 'gradients' in neighborDict:
            neighborDict['gradients'] = neighborDict['gradients'] @ augR

    return ds, neighborDict


def searchNeighbors(state, config, computeKernels = False):    
    # print('fluid - fluid neighbor search')
    state['fluid']['datastructure'], state['fluid']['neighborhood'] = neighborSearchStates(state['fluid'], state['fluid'], config, augR = state['augmentRotation'] if 'augmentRotation' in state else None, priorState = state['fluid']['neighborhood'] if 'neighborhood' in state['fluid'] else None, priorDatastructure = state['fluid']['datastructure'] if 'datastructure' in state['fluid'] else None, computeKernels = computeKernels)
    state['fluid']['numNeighbors'] = state['fluid']['neighborhood']['numNeighbors']
    
    if 'boundary' in state and state['boundary'] is not None:
        state['boundary']['datastructure'], state['boundary']['neighborhood'] = neighborSearchStates(state['boundary'], state['boundary'], config, augR = state['augmentRotation'] if 'augmentRotation' in state else None, priorState = state['boundary']['neighborhood'] if 'neighborhood' in state['boundary'] else None, priorDatastructure = state['boundary']['datastructure'] if 'datastructure' in state['boundary'] else None, computeKernels = computeKernels)
        state['boundary']['numNeighbors'] = state['fluid']['neighborhood']['numNeighbors']
    
        _, state['fluidToBoundaryNeighborhood'] = neighborSearchStates(state['boundary'], state['fluid'], config, augR = state['augmentRotation'] if 'augmentRotation' in state else None, priorState = state['fluidToBoundaryNeighborhood'] if 'fluidToBoundaryNeighborhood' in state else None, priorDatastructure = state['fluid']['datastructure'] if 'datastructure' in state['fluid'] else None, computeKernels = computeKernels)
        
        _, state['boundaryToFluidNeighborhood'] = neighborSearchStates(state['fluid'], state['boundary'], config, augR = state['augmentRotation'] if 'augmentRotation' in state else None, priorState = state['boundaryToFluidNeighborhood'] if 'boundaryToFluidNeighborhood' in state else None, priorDatastructure = state['boundary']['datastructure'] if 'datastructure' in state['boundary'] else None, computeKernels = computeKernels)
