import torch
# from diffSPH.v2.math import scatter_sum
from typing import Dict, Optional, Union
from torch.profiler import record_function

# ------ Beginning of scatter functionality ------ #
# Scatter summation functionality based on pytorch geometric scatter functionality
# This is included here to make the code independent of pytorch geometric for portability
# Note that pytorch geometric is licensed under an MIT licenses for the PyG Team <team@pyg.org>
@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
# ------ End of scatter functionality ------ #

@torch.jit.script 
def sphInterpolation(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], kernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int):                                                        # Ancillary information
    j = neighborhood[1]
    k = masses[1][j] / densities[1][j] * kernels
    kq = torch.einsum('n..., n -> n...', quantities[1][j] if isinstance(quantities,tuple) else quantities, k)
    
    return scatter_sum(kq, neighborhood[0], dim = 0, dim_size = numParticles)

@torch.jit.script 
def sphDensityInterpolation(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], kernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int):                                                        # Ancillary information
    j = neighborhood[1]
    kq = masses[1][j] * kernels
    
    return scatter_sum(kq, neighborhood[0], dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphGradient(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int, type : str = 'naive'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]
    if type == 'symmetric':
        assert isinstance(quantities, tuple), 'Symmetric gradient only supports two inputs for quantities!'
        k = masses[1][j].view(-1,1) * gradKernels
        Ai = torch.einsum('n..., n -> n...', quantities[0][i], 1.0 / densities[0][i]**2)
        Aj = torch.einsum('n..., n -> n...', quantities[1][j], 1.0 / densities[1][j]**2)
        kq = torch.einsum('n... , nd -> n...d', Ai + Aj, k)

        return torch.einsum('n, n... -> n...', densities[0], scatter_sum(kq, i, dim = 0, dim_size = numParticles))
    elif type == 'difference':
        k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
        qij = (quantities[0][i] - quantities[1][j]) if isinstance(quantities, tuple) else quantities
        kq = torch.einsum('n... , nd -> n...d', qij, k)
    elif type == 'summation':
        k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
        qij = (quantities[0][i] + quantities[1][j]) if isinstance(quantities, tuple) else quantities
        kq = torch.einsum('n... , nd -> n...d', qij, k)
    else:
        k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
        qij = (quantities[1][j]) if isinstance(quantities, tuple) else quantities
        kq = torch.einsum('n... , nd -> n...d', qij, k)
    
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphDivergence(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int, type : str = 'naive', mode : str = 'div'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]

    assert (isinstance(quantities, tuple) and quantities[0].dim() > 1) or (isinstance(quantities, torch.Tensor) and quantities.dim() > 1), 'Cannot compute divergence on non vector fields!'
    assert (mode in ['div','dot']), 'Only supports div F and nabla dot F'

    if type == 'symmetric':
        assert isinstance(quantities, tuple), 'Symmetric divergence only supports two inputs for quantities!'
        k = masses[1][j].view(-1,1) * gradKernels
        Ai = torch.einsum('n..., n -> n...', quantities[0][i], 1.0 / densities[0][i]**2)
        Aj = torch.einsum('n..., n -> n...', quantities[1][j], 1.0 / densities[1][j]**2)
        q = Ai + Aj
            
        if mode == 'div':
            kq = torch.einsum('n...d, nd -> n...', q, k)
        else:
            kq = torch.einsum('nd..., nd -> n...', q, k)

        return torch.einsum('n, n... -> n...', densities[0], scatter_sum(kq, i, dim = 0, dim_size = numParticles))
        
    q = quantities[1][j] if isinstance(quantities, tuple) else quantities
    k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
    
    if type == 'difference':
        q = (quantities[1][j] - quantities[0][i]) if isinstance(quantities, tuple) else quantities
    elif type == 'summation':
        q = (quantities[1][j] + quantities[0][i]) if isinstance(quantities, tuple) else quantities
        
    if mode == 'div':
        kq = torch.einsum('n...d, nd -> n...', q, k)
    else:
        kq = torch.einsum('nd..., nd -> n...', q, k)
            
    
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphCurl(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        numParticles : int, type : str = 'naive'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]

    assert (isinstance(quantities, tuple) and quantities[0].dim() > 1), 'Cannot compute curl on non vector fields!'
    assert gradKernels.shape[1] > 1, 'Cannot compute curl on one-dimensional fields!'

    if type == 'symmetric':
        assert isinstance(quantities, tuple), 'Symmetric curl only supports two inputs for quantities!'
        k = masses[1][j].view(-1,1) * gradKernels
        Ai = torch.einsum('n..., n -> n...', quantities[0][i], 1.0 / densities[0][i]**2)
        Aj = torch.einsum('n..., n -> n...', quantities[1][j], 1.0 / densities[1][j]**2)
        q = Ai + Aj
            
        if quantities[1].dim() == 2:
            kq = q[:,1] * k[:,0] - q[:,0] * k[:,1]
        else:
            kq = torch.cross(q, k, dim = -1)        
        
        return torch.einsum('n, n... -> n...', densities[0], scatter_sum(kq, i, dim = 0, dim_size = numParticles))
        
    q = (quantities[1][j])
    k = (masses[1][j] / densities[1][j]).view(-1,1) * gradKernels
    
    if type == 'difference':
        q = (quantities[1][j] - quantities[0][i]) if isinstance(quantities, tuple) else quantities
    elif type == 'summation':
        q = (quantities[1][j] + quantities[0][i]) if isinstance(quantities, tuple) else quantities
        
    if q.dim() == 2:
        kq = q[:,1] * k[:,0] - q[:,0] * k[:,1]
    else:
        kq = torch.cross(q, k, dim = -1)            
    
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script 
def sphLaplacian(
        masses : tuple[torch.Tensor, torch.Tensor],                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], gradKernels : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels ij
        laplaceKernels : Optional[torch.Tensor],    
        rij: torch.Tensor, xij:  torch.Tensor, hij : torch.Tensor,
        numParticles : int, type : str = 'naive'):    # Ancillary information
    i = neighborhood[0]                                                    
    j = neighborhood[1]

    if (isinstance(quantities, tuple) and quantities[0].dim() > 2) or (not isinstance(quantities, tuple) and quantities.dim() > 2):
        grad = sphGradient(masses, densities, quantities, neighborhood, gradKernels, numParticles, type = 'difference')
        div = sphDivergence(masses, densities, (grad, grad), neighborhood, gradKernels, numParticles, type = 'difference', mode = 'div')
        return div
    if type == 'naive':     
        assert laplaceKernels is not None, 'Laplace Kernel Values required for naive sph Laplacian operation'
        if laplaceKernels is not None:   
            print('naive')
            lk = -(masses[1][j] / densities[1][j]) * laplaceKernels
            qij = (quantities[0][i] - quantities[1][j]) if isinstance(quantities, tuple) else quantities
            kq = torch.einsum('n, n... -> n...', lk, qij)
            # kq = torch.einsum('n, n... -> n...', lk, quantities[1][j])
        
            return scatter_sum(kq, i, dim = 0, dim_size = numParticles)
            
    quotient = (rij * hij + 1e-7 * hij)
    kernelApproximation = torch.linalg.norm(gradKernels, dim = -1) /  quotient
    kernelApproximation = torch.einsum('nd, nd -> n', gradKernels, -xij)/  quotient# * rij * hij
    
    Aij = (quantities[0][i] - quantities[1][j]) if isinstance(quantities, tuple) else quantities
    if Aij.dim() == 1:
        kq = -Aij * (masses[1][j] / densities[1][j]) * 2 * kernelApproximation
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)
    
    if type == 'conserving':
        dot = torch.einsum('nd, nd -> n', Aij, xij) 
        q = (masses[1][j] / densities[1][j]) * kernelApproximation * dot# * rij
        kq = -q.view(-1, 1) * xij 
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)
        
    if type == 'divergenceFree':
        dot = torch.einsum('nd, nd -> n', Aij, xij) / (rij * hij + 1e-7 * hij)
        q = 2 * (xij.shape[1] + 2) *  (masses[1][j] / densities[1][j]) * dot
        kq = q.view(-1, 1) * gradKernels
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)

    if type == 'dot':
        term = -(xij.shape[1] + 2) * torch.einsum('nd, nd -> n', Aij, xij).view(-1,1) * xij - Aij
        kq = term * (masses[1][j] / densities[1][j] * kernelApproximation).view(-1,1)
        return scatter_sum(kq, i, dim = 0, dim_size = numParticles)

    q = -2 * (masses[1][j] / densities[1][j]) * kernelApproximation
    kq = Aij * q.view(-1,1)
    return scatter_sum(kq, i, dim = 0, dim_size = numParticles)


@torch.jit.script
def sphOperation(
        masses : tuple[torch.Tensor, torch.Tensor],                                                                 # Tuple of particle masses for (i,j)
        densities : tuple[torch.Tensor, torch.Tensor],                                                              # Tuple of particle densities for (i,j)
        quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],                             # Tuple of particle quantities for (i,j)
        neighborhood : tuple[torch.Tensor, torch.Tensor], kernels : torch.Tensor, kernelGradients : torch.Tensor,   # Neighborhood information (i,j) and precomupted kernels and kernelGradients ij
        radialDistances : torch.Tensor, directions : torch.Tensor, supports : torch.Tensor,                         # Graph information of |x_j - x_i| / hij, (x_j - x_i) / |x_j - x_i| and hij
        numParticles : int,                                                                                         # Ancillary information
        operation : str = 'interpolate', gradientMode : str = 'symmetric', divergenceMode : str = 'div',
        kernelLaplacians : Optional[torch.Tensor] = None) -> torch.Tensor:           # Operation to perform
    with record_function("[SPH] - Operation [%s]" % operation):
        if operation == 'density':
            return sphDensityInterpolation(masses, densities, quantities, neighborhood, kernels, numParticles)
        if operation == 'interpolate':
            return sphInterpolation(masses, densities, quantities, neighborhood, kernels, numParticles)
        if operation == 'gradient':
            return sphGradient(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode)
        if operation == 'divergence':
            return sphDivergence(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode, mode = divergenceMode)
        if operation == 'curl':
            return sphCurl(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode)
        if operation == 'laplacian':
            return sphLaplacian(masses, densities, quantities, neighborhood, kernelGradients, kernelLaplacians, radialDistances, directions, supports, numParticles, type = gradientMode)
        if operation == 'directLaplacian':
            grad = sphGradient(masses, densities, quantities, neighborhood, kernelGradients, numParticles, type = gradientMode)
            div = sphDivergence(masses, densities, (grad, grad), neighborhood, kernelGradients, numParticles, type = gradientMode, mode = divergenceMode)
            return div
    raise ValueError('Operation %s not supported!' % operation)
    
# @torch.jit.script
def sphOperationStates(stateA, stateB, quantities : Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], neighborhood: dict, operation : str = 'interpolate', gradientMode : str = 'symmetric', divergenceMode : str = 'div'):
    if operation == 'density':
        return sphDensityInterpolation(
            (stateA['masses'], stateB['masses']), 
            (stateA['masses'], stateB['masses']),
            (stateA['masses'], stateB['masses']), 
            neighborhood['indices'], 
            neighborhood['kernels'], 
            stateA['numParticles'])
    return sphOperation(
        (stateA['masses'], stateB['masses']), 
        (stateA['densities'], stateB['densities']),
        quantities, 
        neighborhood['indices'], 
        neighborhood['kernels'], neighborhood['gradients'], 
        neighborhood['distances'], neighborhood['vectors'], neighborhood['supports'], 
        stateA['numParticles'], 
        operation = operation, gradientMode = gradientMode, divergenceMode = divergenceMode, 
        kernelLaplacians = neighborhood['laplacians'] if 'laplacians' in neighborhood else None)



def adjunctMatrix(M, c, i):
    res = torch.empty_like(M)

    for j in range(c.shape[1]):
        res[:,:,j] = c if j == i else M[:,:,j]
    return res
def LiuLiuConsistent(ghostState, fluidState, q):
    b_scalar = sphOperationStates(ghostState, fluidState, (ghostState['masses'] *0, q), operation = 'interpolate', neighborhood = ghostState['neighborhood'])
    b_grad = sphOperationStates(ghostState, fluidState, (ghostState['masses'] *0, q), operation = 'gradient', neighborhood = ghostState['neighborhood'], gradientMode = 'naive')
    b = torch.cat([b_scalar.view(-1,1), b_grad], dim = 1)

    xij = -ghostState['neighborhood']['vectors'] * ghostState['neighborhood']['distances'].view(-1,1) * ghostState['neighborhood']['supports'].view(-1,1)
    M_0 = sphOperationStates(ghostState, fluidState, (torch.ones_like(q), torch.ones_like(q)), operation = 'interpolate', neighborhood = ghostState['neighborhood'])
    M_grad = sphOperationStates(ghostState, fluidState, (torch.ones_like(q), torch.ones_like(q)), operation = 'gradient', neighborhood = ghostState['neighborhood'], gradientMode = 'naive')

    M_x = sphOperationStates(ghostState, fluidState, xij, operation = 'interpolate', neighborhood = ghostState['neighborhood'])
    M_x_grad = sphOperationStates(ghostState, fluidState, xij, operation = 'gradient', neighborhood = ghostState['neighborhood'], gradientMode = 'naive')

    M = []
    M.append(torch.cat([M_0.view(-1,1), M_x], dim = 1))
    for i in range(M_grad.shape[1]):
        M.append(torch.cat([M_grad[:,i].view(-1,1), M_x_grad[:,i,:]], dim = 1))
        

    M = torch.stack([row.unsqueeze(2) for row in M], dim = 2)[:,:,:,0].mT
    # solution = torch.linalg.solve(M, b)

    # d0 = torch.linalg.det(M)
    d0 = torch.linalg.det(M)
    adjunctMatrices = [torch.linalg.det(adjunctMatrix(M, b, i)) for i in range(M.shape[1])]
    solution = torch.stack([adj/(d0 + 1e-7) for adj in adjunctMatrices], dim = 1)

    # dets = []

    return solution, M, b