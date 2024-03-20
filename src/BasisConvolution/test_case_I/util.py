# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
    
# sph related imports
from BasisConvolution.test_case_I.sph import kernel, kernelGradient
from BasisConvolution.detail.scatter import scatter_sum
import torch



def getStackedUpdates(positions, velocities, accelerations, offset):
    dx = (velocities + accelerations)
    x = dx.mT
    cumsum = torch.cumsum(x, axis = 1)
    s = torch.sum(x, axis = 1, keepdims=True)
    r2lcumsum = x + s - cumsum
    stacked = torch.hstack((r2lcumsum[:,:-offset] - r2lcumsum[:,offset:], r2lcumsum[:,-offset:]))
    return stacked.mT

def getGroundTruthKernel(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction, particleSupport):
    return scatter_sum(torch.hstack(areas)[j] * kernel(torch.abs(distance), particleSupport), i, dim = 0, dim_size = torch.hstack(areas).shape[0])
def getGroundTruthKernelGradient(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction, particleSupport):
    return scatter_sum(torch.hstack(areas)[j] * kernelGradient(torch.abs(distance), direction, particleSupport), i, dim = 0, dim_size = torch.hstack(areas).shape[0])
def getGroundTruthPhysics(positions, velocities, areas, densities, dudts, inVel, outVel, i, j, distance, direction, particleSupport):
    return torch.hstack(outVel)
def getFeaturesKernel(positions, velocities, areas, densities, dudts, inVel, outVel):
    return torch.ones_like(areas )[:,None]
def getFeaturesPhysics(positions, velocities, areas, densities, dudts, inVel, outVel):
    return torch.vstack((inVel, densities,torch.ones_like(areas))).mT
def lossFunction(prediction, groundTruth):
    return (prediction - groundTruth)**2 # MSE