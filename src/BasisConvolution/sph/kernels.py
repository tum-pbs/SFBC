import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
    
import numpy as np
import matplotlib.pyplot as plt

import torch

# Wendland 2 Kernel function and its derivative
@torch.jit.script
def wendland(q, h : float):
    C = 7 / np.pi
    b1 = torch.pow(1. - q, 4)
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    
# Wendland 2 Kernel function and its derivative
@torch.jit.script
def kernelScalar(q :float, h : float):
    C = 7 / np.pi
    b1 = (1. - q)**4
    b2 = 1.0 + 4.0 * q
    return b1 * b2 * C / h**2    
@torch.jit.script
def wendlandGrad(q,r,h : float):
    C = 7 / np.pi    
    return - r * C / h**3 * (20. * q * (1. -q)**3)[:,None]
# Spiky kernel function used mainly in DFSPH to avoid particle clustering
@torch.jit.script
def spikyGrad(q,r,h : float):
    return -r * 30 / np.pi / h**3 * ((1 - q)**2)[:,None]
# Cohesion kernel is used for the akinci surface tension module
@torch.jit.script
def cohesionKernel(q, h : float):
    res = q.new_zeros(q.shape)
    Cd = -1 / (2 * np.pi) * 1 / h**2
    k1 = 128 * (q-1)**3 * q**3 + 1
    k2 = 64 * (q-1)**3 * q**3
    
    res[q <= 0.5] = k1[q<=0.5]
    res[q > 0.5] = k2[q>0.5]
    
    return -res
# Convenient alias functions for easier usage
@torch.jit.script
def kernel(q , h : float):
    return wendland(q,h)


@torch.jit.script
def kernelGradient(q ,r,h : float):
    return wendlandGrad(q,r,h)

# This function was inteded to be used to swap to different kernel functions
# However, pytorch SPH makes this overly cumbersome so this is not implemented
# TODO: Someday this should be possible in torch script.
def getKernelFunctions(kernel):
    if kernel == 'wendland2':
        return wendland, wendlandGrad



@torch.jit.script
def cpow(q, p : int):
    return torch.clamp(q, 0, 1)**p
# @torch.jit.script
# def wendland4(q, h):
#     C = 7 / np.pi
#     b1 = torch.pow(1. - q, 4)
#     b2 = 1.0 + 4.0 * q
#     return b1 * b2 * C / h**2  
    
# class Wendland2:
#     C = [5/4, 7 / np.pi, 21/ (2 * np.pi)]
#     @staticmethod
#     @torch.jit.script
#     def k(q, dim: int = 2):        
#         if dim == 1:
#             return cpow(1 - q, 3) * (1 + 3 * q)
#         else:
#             return cpow(1 - q, 4) * (1 + 4 * q)
#     @staticmethod
#     @torch.jit.script
#     def dkdq(q, dim: int = 2):        
#         if dim == 1:
#             return -12 * q * cpow(1 - q, 2)
#         else:
#             return -20 * q * cpow(1 - q, 3)
#     @staticmethod
#     @torch.jit.script
#     def d2kdq2(q, dim: int = 2):        
#         if dim == 1:
#             return -12 * (3 * q **2 - 4 * q + 1)
#         else:
#             return 20 * (4 * q - 1) * cpow(1-q, 2)
#     @staticmethod
#     @torch.jit.script
#     def kernel(rij, hij, dim : int = 2):
#         return Wendland2.k(rij, dim) * Wendland2.C[dim - 1] / hij**dim
        
#     @staticmethod
#     @torch.jit.script
#     def kernelGradient(rij, xij, hij, dim : int = 2):
#         return xij * (Wendland2.dkdq(rij, dim) * Wendland2.C[dim - 1] / hij**(dim + 1))[:,None]
    
#     @staticmethod
#     @torch.jit.script
#     def kernelLaplacian(rij, hij, dim : int = 2):
#         return (((dim - 1) / (rij + 1e-7 * hij)) * Wendland2.dkdq(rij, dim) * hij + Wendland2.d2kdq2(rij, dim)) * Wendland2.C[dim - 1] / hij**(dim + 2)

import BasisConvolution.sph.kernelFunctions.Wendland2 as Wendland2

# class Wendland4:
#     @staticmethod
#     @torch.jit.script
#     def kernel(rij, hij, dim : int = 2):
#         C = [3/2, 9 / np.pi, 495/ (32 * np.pi)]
        
#         if dim == 1:
#             k = cpow(1 - rij, 5) * (1 + 5 * rij + 8 * rij**2)
#         else:
#             k = cpow(1 - rij, 6) * (1 + 6 * rij + 35/3 * rij **2)
#         return k * C[dim - 1] / hij**dim
        
#     @staticmethod
#     @torch.jit.script
#     def kernelGradient(rij, xij, hij, dim : int = 2):
#         C = [3/2, 9 / np.pi, 495/ (32 * np.pi)]
        
#         if dim == 1:
#             k = -14 * rij * (4 *rij + 1) * (1 - rij)**4
#         else:
#             k = -56/3 * rij * (5 * rij + 1) * (1 - rij)**5
#         return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]        
# class Wendland6:
#     @staticmethod
#     @torch.jit.script
#     def kernel(rij, hij, dim : int = 2):
#         C = [55/32, 78 / (7 * np.pi), 1365/ (64 * np.pi)]
        
#         if dim == 1:
#             k = cpow(1 - rij, 7) * (1 + 7 * rij + 19 * rij**2 + 21 * rij**3)
#         else:
#             k = cpow(1 - rij, 8) * (1 + 8 * rij + 25 * rij**2 + 32 * rij**3)
#         return k * C[dim - 1] / hij**dim
        
#     @staticmethod
#     @torch.jit.script
#     def kernelGradient(rij, xij, hij, dim : int = 2):
#         C = [55/32, 78 / (7 * np.pi), 1365/ (64 * np.pi)]
        
#         if dim == 1:
#             k = -6 * rij * (35 * rij**2 + 18 * rij + 3) * (1 - rij)**6
#         else:
#             k = -22 * rij * (16 * rij**2 + 7 *rij + 1) * (1 - rij)**7
#         return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
# class CubicSpline:
#     @staticmethod
#     @torch.jit.script
#     def kernel(rij, hij, dim : int = 2):
#         C = [8/3, 80 / (7 * np.pi), 16/ (np.pi)]
#         k = cpow(1-rij, 3) - 4 * cpow(1/2 - rij,3)
#         return k * C[dim - 1] / hij**dim
        
#     @staticmethod
#     @torch.jit.script
#     def kernelGradient(rij, xij, hij, dim : int = 2):
#         C = [8/3, 80 / (7 * np.pi), 16/ (np.pi)]
#         k = -3 * cpow(1-rij, 2) + 12 * cpow(1/2 - rij,2)
#         return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
# class QuarticSpline:
#     @staticmethod
#     @torch.jit.script
#     def kernel(rij, hij, dim : int = 2):
#         C = [5**5/768, 5**6 * 3 / (2398 * np.pi), 5**6/ (512 * np.pi)]
#         k = cpow(1-rij, 4) - 5 * cpow(3/5 - rij, 4) + 10 * cpow(1/5 - rij, 4)
#         return k * C[dim - 1] / hij**dim
        
#     @staticmethod
#     @torch.jit.script
#     def kernelGradient(rij, xij, hij, dim : int = 2):
#         C = [5**5/768, 5**6 * 3 / (2398 * np.pi), 5**6/ (512 * np.pi)]
#         k = -4 * cpow(1-rij, 3) + 20 * cpow(3/5 - rij, 3) - 40 * cpow(1/5 - rij, 3)
#         return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
# class QuinticSpline:
#     @staticmethod
#     @torch.jit.script
#     def kernel(rij, hij, dim : int = 2):
#         C = [3**5/40, 3**7 * 7 / (478 * np.pi), 3**7/ (40 * np.pi)]
#         k = cpow(1-rij, 5) - 6 * cpow(2/3 - rij, 5) + 15 * cpow(1/3 - rij, 5)
#         return k * C[dim - 1] / hij**dim
        
#     @staticmethod
#     @torch.jit.script
#     def kernelGradient(rij, xij, hij, dim : int = 2):
#         C = [3**5/40, 3**7 * 7 / (478 * np.pi), 3**7/ (40 * np.pi)]
#         k = -5 * cpow(1-rij, 4) + 30 * cpow(2/3 - rij, 4) - 75 * cpow(1/3 - rij, 4)
#         return xij * (k * C[dim - 1] / hij**(dim + 1))[:,None]
    
# def getKernel(kernel: str = 'Wendland2'):
#     if kernel == 'Wendland2':
#         return Wendland2
#     elif kernel == 'Wendland4':
#         return Wendland4
#     elif kernel == 'Wendland6':
#         return Wendland6
#     elif kernel == 'CubicSpline':
#         return CubicSpline
#     elif kernel == 'QuarticSpline':
#         return QuarticSpline
#     elif kernel == 'QuinticSpline':
#         return QuinticSpline
#     else: return Wendland2


# For Kernel Normalization:
# in 3D: https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BIntegrate%5BIntegrate%5BPower%5B%5C%2840%291-q%5C%2841%29%2C3%5DPower%5Bq%2C2%5DSin%5B%CF%86%5D%2C%7B%CF%86%2C0%2C%CF%80%7D%5D%2C%7B%CE%98%2C0%2C2%CF%80%7D%5D%2C%7Bq%2C0%2C1%7D%5D
# in 2D: https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BIntegrate%5BPower%5B%5C%2840%291-q%5C%2841%29%2C3%5D%2C%7B%CE%98%2C0%2C2%CF%80%7D%5D%2C%7Bq%2C0%2C1%7D%5D
# in 1D: https://www.wolframalpha.com/input?i2d=true&i=Integrate%5BPower%5B%5C%2840%291-q%5C%2841%29%2C3%5D%2C%7Bq%2C0%2C1%7D%5D

import BasisConvolution.sph.kernelFunctions.Wendland2 as Wendland2
import BasisConvolution.sph.kernelFunctions.Wendland4 as Wendland4
import BasisConvolution.sph.kernelFunctions.Wendland6 as Wendland6
import BasisConvolution.sph.kernelFunctions.CubicSpline as CubicSpline
import BasisConvolution.sph.kernelFunctions.QuarticSpline as QuarticSpline
import BasisConvolution.sph.kernelFunctions.QuinticSpline as QuinticSpline

import BasisConvolution.sph.kernelFunctions.Spiky as Spiky
import BasisConvolution.sph.kernelFunctions.ViscosityKernel as ViscosityKernel
import BasisConvolution.sph.kernelFunctions.Poly6 as Poly6
import BasisConvolution.sph.kernelFunctions.CohesionKernel as CohesionKernel
import BasisConvolution.sph.kernelFunctions.AdhesionKernel as AdhesionKernel
class KernelWrapper:
    def __init__(self, module):
        self.module = module
        # for attr_name in dir(module):
        #     attr = getattr(module, attr_name)
        #     if callable(attr):
        #         setattr(self, attr_name, attr)
    def __getattr__(self, name):
        return getattr(self.module, name)
    
    def C_d(self, dim : int):
        return self.module.C_d(dim)

    def kernel(self, rij, hij, dim : int = 2):
        return self.module.kernel(rij, hij, dim)
    
    def kernelGradient(self, rij, xij, hij, dim : int = 2):
        return self.module.kernelGradient(rij, xij, hij, dim)
    
    def kernelLaplacian(self, rij, hij, dim : int = 2):
        return self.module.kernelLaplacian(rij, hij, dim)
    
    def Jacobi(self, rij, xij, hij, dim : int = 2):
        return self.module.kernelGradient(rij, xij, hij, dim)

    def Hessian2D(self, rij, xij, hij, dim : int = 2):
        hessian = torch.zeros(rij.shape[0], 2, 2, device=rij.device, dtype=rij.dtype)
        factor = self.module.C_d(dim) / hij**(dim)

        r_ij = rij * hij
        x_ij = xij * r_ij.unsqueeze(-1)
        q_ij = rij

        termA_x = x_ij[:,0]**2 / (hij * r_ij    + 1e-5 * hij)**2 * self.module.d2kdq2(q_ij, dim = dim)
        termA_y = x_ij[:,1]**2 / (hij * r_ij    + 1e-5 * hij)**2 * self.module.d2kdq2(q_ij, dim = dim)

        termB_x = torch.where(r_ij / hij > 1e-5, 1 /(hij * r_ij + 1e-6 * hij),0) * self.module.dkdq(q_ij, dim = dim)
        termB_y = torch.where(r_ij / hij > 1e-5, 1 /(hij * r_ij + 1e-6 * hij),0) * self.module.dkdq(q_ij, dim = dim)

        termC_x = - (x_ij[:,0]**2) / (hij * r_ij**3 + 1e-5 * hij) * self.module.dkdq(q_ij, dim = dim)
        termC_y = - (x_ij[:,1]**2) / (hij * r_ij**3 + 1e-5 * hij) * self.module.dkdq(q_ij, dim = dim)
                
        d2Wdx2 = factor * (termA_x + termB_x + termC_x + 1/hij**2 * torch.where(q_ij < 1e-5, self.module.d2kdq2(torch.tensor(0.), dim = dim), 0))
        d2Wdy2 = factor * (termA_y + termB_y + termC_y + 1/hij**2 * torch.where(q_ij < 1e-5, self.module.d2kdq2(torch.tensor(0.), dim = dim), 0))

        d2Wdxy = self.module.C_d(dim) * hij**(-dim) * torch.where(q_ij > -1e-7, \
            ( x_ij[:,0] * x_ij[:,1]) / (hij * r_ij **2 + 1e-5 * hij) * (1 / hij * self.module.d2kdq2(q_ij, dim = dim) - 1 / (r_ij + 1e-3 * hij) * self.module.dkdq(q_ij, dim = dim)),0)
        d2Wdyx = d2Wdxy
            
        hessian[:,0,0] = torch.where(q_ij <= 1, d2Wdx2, 0)
        hessian[:,1,1] = torch.where(q_ij <= 1, d2Wdy2, 0)
        hessian[:,0,1] = torch.where(q_ij <= 1, d2Wdxy, 0)
        hessian[:,1,0] = torch.where(q_ij <= 1, d2Wdyx, 0)

        return hessian




def getKernel(kernel = 'Wendland2'):
    if kernel == 'Wendland2': return KernelWrapper(Wendland2)
    if kernel == 'Wendland4': return KernelWrapper(Wendland4)
    if kernel == 'Wendland6': return KernelWrapper(Wendland6)
    if kernel == 'CubicSpline': return KernelWrapper(CubicSpline)
    if kernel == 'QuarticSpline': return KernelWrapper(QuarticSpline)
    if kernel == 'QuinticSpline': return KernelWrapper(QuinticSpline)

    if kernel == 'Spiky': return KernelWrapper(Spiky)
    if kernel == 'ViscosityKernel': return KernelWrapper(ViscosityKernel)
    if kernel == 'CohesionKernel': return KernelWrapper(CohesionKernel)
    if kernel == 'AdhesionKernel': return KernelWrapper(AdhesionKernel)
    if kernel == 'Poly6': return KernelWrapper(Poly6)

# @torch.jit.script
# def getKernel(kernel : str = 'Wendland2'):
#     if kernel == 'Wendland2': return Wendland2.kernel
#     elif kernel == 'Wendland4': return Wendland4.kernel
#     elif kernel == 'Wendland6': return Wendland6.kernel
#     elif kernel == 'CubicSpline': return CubicSpline.kernel
#     elif kernel == 'QuarticSpline': return QuarticSpline.kernel
#     elif kernel == 'QuinticSpline': return QuinticSpline.kernel

# @torch.jit.script
# def getKernelGradient(kernel : str = 'Wendland2'):
#     if kernel == 'Wendland2': return Wendland2.kernelGradient
#     elif kernel == 'Wendland4': return Wendland4.kernelGradient
#     elif kernel == 'Wendland6': return Wendland6.kernelGradient
#     elif kernel == 'CubicSpline': return CubicSpline.kernelGradient
#     elif kernel == 'QuarticSpline': return QuarticSpline.kernelGradient
#     elif kernel == 'QuinticSpline': return QuinticSpline.kernelGradient

# @torch.jit.script
# def getKernelLaplacian(kernel : str = 'Wendland2'):
#     if kernel == 'Wendland2': return Wendland2.kernelLaplacian
#     elif kernel == 'Wendland4': return Wendland4.kernelLaplacian
#     elif kernel == 'Wendland6': return Wendland6.kernelLaplacian
#     elif kernel == 'CubicSpline': return CubicSpline.kernelLaplacian
#     elif kernel == 'QuarticSpline': return QuarticSpline.kernelLaplacian
#     elif kernel == 'QuinticSpline': return QuinticSpline.kernelLaplacian