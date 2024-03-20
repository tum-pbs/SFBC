
# Math/parallelization library includes
import numpy as np
import torch

# Imports for neighborhood searches later on
# from torch_geometric.nn import radius

try:
    from torchCompactRadius import radiusSearch
    hasClusterRadius = True
except ModuleNotFoundError:
    from BasisConvolution.neighborhoodFallback.neighborhood import radiusSearch
    hasClusterRadius = False
    # pass


# def radius(x, y, batch_x = None, batch_y = None, batch_size = None):
#     # This is a simple radius search function that is used to find the neighbors of particles
#     # x and y are the coordinates of the particles
#     # batch_x and batch_y are the batch indices of the particles
    


def radius(data_x, data_y, r, batch_x = None, batch_y = None, max_num_neighbors = 32, neighborhoodArguments = None):
    # print(neighborhoodArguments)
    if batch_x == None and batch_y == None:
        with torch.no_grad():
            if neighborhoodArguments == None:
                i, j = radiusSearch(data_x, data_y, fixedSupport = torch.tensor(r, device =data_x.device, dtype = data_x.dtype), algorithm = 'small' if hasClusterRadius else 'naive')
            else:
                i, j = radiusSearch(data_x, data_y, fixedSupport = torch.tensor(r, device =data_x.device, dtype = data_x.dtype), **neighborhoodArguments)
            return j, i
    if batch_x != None and batch_y != None:
        batchIndices = torch.unique(batch_x)
        i = []
        j = []
        cum_x = 0
        cum_y = 0
        for b in batchIndices:
            batchData_x = data_x[batch_x == b]
            batchData_y = data_y[batch_y == b]
            with torch.no_grad():
                if neighborhoodArguments == None:
                    i_, j_ = radiusSearch(batchData_x, batchData_y, fixedSupport = torch.tensor(r, device =data_x.device, dtype = data_x.dtype), algorithm = 'small' if hasClusterRadius else 'naive')
                else:
                    i_, j_ = radiusSearch(batchData_x, batchData_y, fixedSupport = torch.tensor(r, device =data_x.device, dtype = data_x.dtype), **neighborhoodArguments)
            i.append(i_ + cum_x)
            j.append(j_ + cum_y)
            cum_x += batchData_x.shape[0]
            cum_y += batchData_y.shape[0]
        i = torch.cat(i)
        j = torch.cat(j)
        return j, i
    raise ValueError('Incompatible batches (batch_x = None and batch_y != None) or (batch_x != None and batch_y = None)')



# Neighborhood search
def findNeighborhoods(particles, allParticles, support):
    # Call the external neighborhood search function
    row, col = radiusSearch(allParticles, particles, fixedSupport = support)
    fluidNeighbors = torch.stack([row, col], dim = 0)
        
    # Compute the distances of all particle pairings
    fluidDistances = (allParticles[fluidNeighbors[1]] - particles[fluidNeighbors[0]])
    # This could also be done with an absolute value function
    fluidRadialDistances = torch.abs(fluidDistances)# torch.sqrt(fluidDistances**2)

    # Compute the direction, in 1D this is either 0 (i == j) or +-1 depending on the relative position
    fluidDistances[fluidRadialDistances < 1e-7] = 0
    fluidDistances[fluidRadialDistances >= 1e-7] /= fluidRadialDistances[fluidRadialDistances >= 1e-7]
    fluidRadialDistances /= support
    
    # Modify the neighbor list so that everything points to the original particles
    particleIndices = torch.arange(particles.shape[0]).to(particles.device)
    stackedIndices = torch.hstack((particleIndices, particleIndices, particleIndices))
    fluidNeighbors[1,:] = stackedIndices[fluidNeighbors[1,:]]    
    
    return fluidNeighbors, fluidRadialDistances, fluidDistances

def periodicNeighborSearch(fluidPositions, particleSupport, minDomain, maxDomain):
    distanceMat = fluidPositions[:,None] - fluidPositions
    distanceMat = torch.remainder(distanceMat + minDomain, maxDomain - minDomain) - maxDomain
    neighs = torch.abs(distanceMat) < particleSupport
    n0 = torch.sum(neighs, dim = 0)
    indices = torch.arange(fluidPositions.shape[0]).to(fluidPositions.device)
    indexMat = indices.expand(fluidPositions.shape[0], fluidPositions.shape[0])
    j, i = indexMat[neighs], indexMat.mT[neighs]
    distances = -distanceMat[neighs]
    directions = torch.sign(distances)    
    return torch.vstack((i, j)), torch.abs(distances)  / particleSupport, directions

    
def batchedNeighborsearch(positions, setup):
    neighborLists = [periodicNeighborSearch(p, s['particleSupport'], s['minDomain'], s['maxDomain']) for p, s in zip(positions, setup)]
    
    neigh_i = [n[0][0] for n in neighborLists]
    neigh_j = [n[0][1] for n in neighborLists]
    neigh_distance = [n[1] for n in neighborLists]
    neigh_direction = [n[2] for n in neighborLists]
    
    for i in range(len(neighborLists) - 1):
        neigh_i[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
        neigh_j[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
        
    neigh_i = torch.hstack(neigh_i)
    neigh_j = torch.hstack(neigh_j)
    neigh_distance = torch.hstack(neigh_distance)
    neigh_direction = torch.hstack(neigh_direction)
    
    return neigh_i, neigh_j, neigh_distance, neigh_direction
