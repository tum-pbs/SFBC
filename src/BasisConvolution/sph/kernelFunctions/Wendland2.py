import numpy as np
from BasisConvolution.sph.kernels import cpow 
import torch

@torch.jit.script
def k(q, dim: int = 2):        
    if dim == 1:
        return cpow(1 - q, 3) * (1 + 3 * q)
    else:
        return cpow(1 - q, 4) * (1 + 4 * q)
@torch.jit.script
def dkdq(q, dim: int = 2):        
    if dim == 1:
        return -12 * q * cpow(1 - q, 2)
    else:
        return -20 * q * cpow(1 - q, 3)
@torch.jit.script
def d2kdq2(q, dim: int = 2):        
    if dim == 1:
        return -12 * (3 * q **2 - 4 * q + 1)
    else:
        return 20 * (4 * q - 1) * cpow(1-q, 2)
    
@torch.jit.script
def C_d(dim : int):
    if dim == 1: return      5 / 4
    elif dim == 2: return    7 / np.pi
    else: return            21 / (2 * np.pi)

@torch.jit.script
def kernel(rij, hij, dim : int = 2):
    return k(rij, dim) * C_d(dim) / hij**dim
    
@torch.jit.script
def kernelGradient(rij, xij, hij, dim : int = 2):
    return xij * (dkdq(rij, dim) * C_d(dim) / hij**(dim + 1))[:,None]

@torch.jit.script
def kernelLaplacian(rij, hij, dim : int = 2):
    return ((torch.where(rij > -1e-7, ((dim - 1) / (rij *hij + 1e-7 * hij)) * dkdq(rij + 1e-7, dim) * hij, 0) + d2kdq2(rij, dim)) * C_d(dim)) / hij**(dim + 2)

@torch.jit.script # See Dehnen & Aly: Improving convergence in smoothed particle hydrodynamics simulations
def kernelScale(dim: int = 2):
    if dim == 1: return 1.620185
    elif dim == 2: return 1.897367
    else: return 1.936492