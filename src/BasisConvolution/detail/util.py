import torch
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Spacing for basis functions
@torch.jit.script
def getSpacing(n : int, periodic : bool = False):
    if n == 1:
        return 2.
    else:
        return 2. / n if periodic else 2./(n-1)
    
# Function that returns the distance between a given set of points and a set of basis function centers
# Caches the basis function center positions for computational efficiency
centroidCache = {False:{'cuda':{},'cpu':{}},True:{'cuda':{},'cpu':{}}}
def getDistancesRelCached(n, x, periodic = False):
    if n in centroidCache[periodic][x.device.type]:
        centroids = centroidCache[periodic][x.device.type][n]
        if periodic:
            spacing = getSpacing(n, True)
            offset = -1 + spacing / 2.
            ra = torch.unsqueeze(x,axis=0) - centroids
            rb = torch.unsqueeze(x,axis=0) - centroids - 2.
            rc = torch.unsqueeze(x,axis=0) - centroids + 2.
            return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        else:
            spacing = getSpacing(n, False)
            r = torch.unsqueeze(x,axis=0) - centroids
            return r  / spacing


    if periodic:
        spacing = getSpacing(n, True)
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],axis=1)
        centroidCache[periodic][x.device.type][n] = centroids

        ra = torch.unsqueeze(x,axis=0) - centroids
        rb = torch.unsqueeze(x,axis=0) - centroids - 2.
        rc = torch.unsqueeze(x,axis=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, axis = 1)
    centroidCache[periodic][x.device.type][n] = centroids
    r = torch.unsqueeze(x,axis=0) - centroids
    return r  / spacing
@torch.jit.script
def getDistancesRel(n : int, x : torch.Tensor, periodic : bool = False):
    if periodic:
        spacing = getSpacing(n, True)
        centroids = torch.unsqueeze(torch.linspace(-1.,1.,n+1, device = x.device)[:n],dim=1)

        ra = torch.unsqueeze(x,dim=0) - centroids
        rb = torch.unsqueeze(x,dim=0) - centroids - 2.
        rc = torch.unsqueeze(x,dim=0) - centroids + 2.
        return torch.minimum(torch.minimum(torch.abs(ra)/spacing, torch.abs(rb)/spacing), torch.abs(rc)/spacing)
        
    spacing = getSpacing(n, False)
    
    centroids = torch.linspace(-1.,1.,n, device = x.device) if n > 1 else torch.tensor([0.], device = x.device)
    centroids = torch.unsqueeze(centroids, dim = 1)
    r = torch.unsqueeze(x,dim=0) - centroids
    return r  / spacing

# Evaluate a set of radial basis functions with a variety of options
@torch.jit.script
def cpow(x : torch.Tensor, p : int):
    return torch.maximum(x, torch.zeros_like(x)) ** p


import itertools
import numbers
from typing import Any

import torch
from torch import Tensor


def repeat(src: Any, length: int) -> Any:
    if src is None:
        return None

    if isinstance(src, Tensor):
        if src.numel() == 1:
            return src.repeat(length)

        if src.numel() > length:
            return src[:length]

        if src.numel() < length:
            last_elem = src[-1].unsqueeze(0)
            padding = last_elem.repeat(length - src.numel())
            return torch.cat([src, padding])

        return src

    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))

    if (len(src) > length):
        return src[:length]

    if (len(src) < length):
        return src + list(itertools.repeat(src[-1], length - len(src)))

    return src
