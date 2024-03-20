import torch
import numpy as np
from .basis import evalBasisFunction
import scipy

MCache = None

def optimizeWeights2D(weights, basis, periodicity, nmc = 32 * 1024, targetIntegral = 1, windowFn = None, verbose = False):
    global MCache
    M = None
    numWeights = weights.shape[0] * weights.shape[1]    
    
    # print(weights.shape, numWeights)
    normalizedWeights = (weights - torch.sum(weights) / weights.numel())/torch.std(weights)
    if not MCache is None:
        cfg, M = MCache
        w,b,n,p,wfn = cfg
        if not(w == weights.shape and np.all(b == basis) and n == nmc and np.all(p ==periodicity) and wfn == windowFn):
            M = None
    # else:
        # print('no cache')
    if M is None:
        r = torch.sqrt(torch.rand(size=(nmc,1)).to(weights.device).type(torch.float32))
        theta = torch.rand(size=(nmc,1)).to(weights.device).type(torch.float32) *2 * np.pi

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        u = evalBasisFunction(weights.shape[0], x.T, which = basis[0], periodic = periodicity[0])[0,:].mT
        v = evalBasisFunction(weights.shape[1], y.T, which = basis[1], periodic = periodicity[1])[0,:].mT
        
    #     print('u', u.shape, u)
    #     print('v', v.shape, v)
        
        window = weights.new_ones(x.shape[0]) if windowFn is None else windowFn(torch.sqrt(x**2 + y**2))[:,0]
        
        
        nuv = torch.einsum('nu, nv -> nuv', u, v)
        nuv = nuv * window[:,None, None]

    #     print('nuv', nuv.shape, nuv)
        M = np.pi * torch.sum(nuv, dim = 0).flatten().detach().cpu().numpy() / nmc
#     print('M', M.shape, M)
        MCache = ((weights.shape, basis, nmc, periodicity, windowFn), M)

    
    w = normalizedWeights.flatten().detach().cpu().numpy()


    eps = 1e-2
    
    if 'chebyshev' in basis or 'fourier' in basis:        
        res = scipy.optimize.minimize(fun = lambda x: (M.dot(x) - targetIntegral)**2, \
                                      jac = lambda x: 2 * M * (M.dot(x) - targetIntegral), \
                                      hess = lambda x: 2. * np.outer(M,M), x0 = w, \
                                      method ='trust-constr', constraints = None,\
                                      options={'disp': False, 'maxiter':100})
    else:
        sumConstraint = scipy.optimize.NonlinearConstraint(fun = np.sum, lb = -eps, ub = eps)
        stdConstraint = scipy.optimize.NonlinearConstraint(fun = np.std, lb = 1 - eps, ub = 1 + eps)

        res = scipy.optimize.minimize(fun = lambda x: (M.dot(x) - targetIntegral)**2, \
                                      jac = lambda x: 2 * M * (M.dot(x) - targetIntegral), \
                                      hess = lambda x: 2. * np.outer(M,M), x0 = w, \
                                      method ='trust-constr', constraints = [sumConstraint, stdConstraint],\
                                      options={'disp': False, 'maxiter':100})
    result = torch.from_numpy(res.x.reshape(weights.shape)).type(torch.float32).to(weights.device)
    if verbose:
        print('result: ', res)
        print('initial weights:', normalizedWeights)
        print('result weights:',result)
        print('initial:', M.dot(w))
        print('integral:', M.dot(res.x))
        print('sumConstraint:', np.sum(res.x))
        print('stdConstraint:', np.std(res.x))
    return result, res.constr, res.fun, M.dot(w), M.dot(res.x)

