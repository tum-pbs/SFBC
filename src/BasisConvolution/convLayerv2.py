# Copyright 2023 <COPYRIGHT HOLDER>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
# from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
# from torch_geometric.utils.repeat import repeat
import torch
# from torch_sparse import SparseTensor
from torch import Tensor, nn
from torch.nn import Parameter
from BasisConvolution.detail.util import repeat
# from BasisConvolution.detail.cutlass import cutlass

# import math
import numpy as np

from .detail.cutlassv2 import cutlass
from .detail.typing import zeros, is_list_of_strings
from typing import List, Optional, Union
from .detail.initialization import optimizeWeights2D
from .detail.basis import evalBasisFunction
from .detail.mapping import mapToSpherePreserving, mapToSpherical

from .detail.typing import Adj, OptPairTensor, OptTensor, Size

def buildMLP(layers, inputFeatures = 1, gain = 1/np.sqrt(34), useBias = False):
    modules = []
    if len(layers) > 1:
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(inputFeatures if i == 0 else layers[i-1],layers[i], bias = useBias))
            torch.nn.init.xavier_normal_(modules[-1].weight,gain)
            if useBias:
                torch.nn.init.zeros_(modules[-1].bias)
            # modules.append(nn.BatchNorm1d(layers[i]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[-2],layers[-1], bias = useBias))
    else:
        modules.append(nn.Linear(inputFeatures,layers[-1], bias = useBias))        
    torch.nn.init.xavier_normal_(modules[-1].weight,gain)
    if useBias:
        torch.nn.init.zeros_(modules[-1].bias)
    return nn.Sequential(*modules)


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def initializeWeights2D(basis, nx, n, weights, plot = False):
    device = 'cpu'
    # weights = torch.zeros(n, n)
    # if initialWeights == 'xavier_normal':
    #     weights = torch.nn.init.xavier_normal_(weights).view(n,n,1,1).to(device)
    # elif initialWeights == 'xavier_uniform':
    #     weights = torch.nn.init.xavier_uniform_(weights).view(n,n,1,1).to(device)
    # elif initialWeights == 'kaiming_normal':
    #     weights = torch.nn.init.kaiming_normal_(weights).view(n,n,1,1).to(device)
    # elif initialWeights == 'kaiming_uniform':
    #     weights = torch.nn.init.kaiming_uniform_(weights).view(n,n,1,1).to(device)
    # elif initialWeights == 'exponential':
    #     weights = torch.nn.init.xavier_normal_(weights).view(n,n,1,1).to(device)
    #     for i in range(n):
    #         for j in range(n):
    #             weights[i,j,:,:] = weights[i,j,:,:] * np.exp(-(i + j))
    
    xlin = torch.linspace(-1,1,nx, dtype = torch.float32, device = device)
    ylin = torch.linspace(-1,1,nx, dtype = torch.float32, device = device)
    X, Y = torch.meshgrid(xlin, ylin, indexing='ij')
    edge_attr = torch.stack([X.flatten(), Y.flatten()], dim = 1)
    features = torch.normal(0., 1., size = (edge_attr.shape[0], 1), device = device)
    i = torch.zeros(edge_attr.shape[0], dtype = torch.int64, device = device)
    j = torch.arange(edge_attr.shape[0], dtype = torch.int64, device = device)
    edge_index = torch.stack([i,j], dim = 0)
    edge_index_two = torch.stack([j,i], dim = 0)
    features = torch.ones(edge_attr.shape[0], 1, device = device)
    convolution = cutlass.apply
    conv = convolution(edge_index_two, features, features, edge_attr, None, weights, 
                    edge_attr.shape[0], 0,
                        [n,n] , [basis, basis], [False, False], 
                        32, 32)
    out = conv.reshape(X.shape).detach().cpu().numpy()
    if plot:
        fig, axis = plt.subplots(1, 1, figsize=(6, 5), squeeze = False)
        ax = axis[0,0]

        # ax.set_title(f'{base}: jitter: {augJ}, rotation: {augR}')

        # ax.set_title(configurations[im])
        out = conv.reshape(X.shape).detach().cpu().numpy()
        im = ax.pcolormesh(X.cpu().numpy(), Y.cpu().numpy(), out, cmap = 'Spectral_r', vmin = -np.max(np.abs(out)), vmax = np.max(np.abs(out)))
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        # ax.set_title('Kernel')
        circle = plt.Circle((0, 0), 1, color='r', fill = False)
        ax.add_artist(circle)
        ax.set_aspect('equal', adjustable='box')

    return (weights - np.mean(out)) / (np.std(out))

class BasisConv(torch.nn.Module):
    def __init__(
        self,
        inputFeatures: int,
        outputFeatures: int,
        dim: int = 2,

        basisTerms : Union[int, List[int]] = [4, 4],
        basisFunction : Union[int, List[int]] = 'linear',
        basisPeriodicity : Union[bool, List[bool]] = False,

        linearLayerActive: bool = False,
        linearLayerHiddenLayout: List[int] = [32, 32],
        linearLayerActivation: str = 'relu',

        biasActive: bool = False,
        feedThrough: bool = False,

        preActivation: Optional[str] = None,
        postActivation: Optional[str] = None,

        cutlassBatchSize = 16,

        cutlassNormalization = False,
        initializer = 'uniform',
        optimizeWeights = False,
        exponentialDecay = False,
        **kwargs
    ):
        super().__init__(**kwargs)      

        self.inputFeatures = inputFeatures
        self.outputFeatures = outputFeatures
        self.dim = dim
        
        # print('coordinate mapping', self.coordinateMapping)
        self.basisTerms       = basisTerms if isinstance(basisTerms, list) else repeat(basisTerms, dim)
        self.basisFunctions   = basisFunction if is_list_of_strings(basisFunction) else [basisFunction] * dim
        self.basisPeriodicity = basisPeriodicity if isinstance(basisPeriodicity, list) else repeat(basisPeriodicity, dim) 
        self.cutlassBatchSize = cutlassBatchSize

        self.linearLayerActive = linearLayerActive
        self.linearLayerHiddenLayout = linearLayerHiddenLayout
        self.linearLayerActivation = None if linearLayerActivation is None else getattr(nn.functional, linearLayerActivation)


        self.preActivation = None if preActivation is None else getattr(nn.functional, preActivation)
        self.postActivation = None if postActivation is None else getattr(nn.functional, postActivation)
        
        self.feedThrough = feedThrough
        self.biasActive = biasActive
        self.cutlassNormalization = cutlassNormalization

        if isinstance(inputFeatures, int):
            in_channels = (inputFeatures, inputFeatures)

        if self.biasActive:
            self.bias = Parameter(torch.zeros(outputFeatures))
        else:
            self.register_parameter('bias', None)

        self.weight = Parameter(torch.Tensor(*self.basisTerms, inputFeatures, outputFeatures))
        if initializer == 'uniform':
            torch.nn.init.uniform_(self.weight, -0.05, 0.05)
        elif initializer == 'normal':
            torch.nn.init.normal_(self.weight, 0, 0.05)
        elif initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.weight)
        elif initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.weight)
        elif initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.weight)
        elif initializer == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.weight)

        if exponentialDecay:
            with torch.no_grad():
                if self.basisFunctions[0] in ['chebyshev', 'fourier', 'gabor']:
                    for i in range(self.weight.shape[0]):
                        if len(self.basisFunctions) == 1:
                            self.weight[i] *= np.exp(-i)
                        if len(self.basisFunctions) == 2:
                            self.weight[i,:] *= np.exp(-i)
                        if len(self.basisFunctions) == 3:
                            self.weight[i,:,:] *= np.exp(-i)
                if len(self.basisFunctions) > 1 and self.basisFunctions[1] in ['chebyshev', 'fourier', 'gabor']:
                    for i in range(self.weight.shape[1]):
                        if len(self.basisFunctions) == 2:
                            self.weight[:,i] *= np.exp(-i)
                        if len(self.basisFunctions) == 3:
                            self.weight[:,i,:] *= np.exp(-i)
                if len(self.basisFunctions) > 2 and self.basisFunctions[2] in ['chebyshev', 'fourier', 'gabor']:
                    for i in range(self.weight.shape[2]):
                        self.weight[:,:,i] = self.weight[:,:,i] * np.exp(-i)
    

        if optimizeWeights and len(self.basisFunctions) == 2:
            with torch.no_grad():
                for fin in range(inputFeatures):
                    for fout in range(outputFeatures):
                        self.weight[:,:,fin,fout] = initializeWeights2D(self.basisFunctions[0], 256, self.basisTerms[0], self.weight[:,:,fin,fout].view(*self.basisTerms,1,1), plot = False).reshape(*self.basisTerms) * 2 / (outputFeatures + inputFeatures) / 4 * 2**(0.09 * outputFeatures)
                if 'fourier' in self.basisFunctions or 'ffourier' in self.basisFunctions:
                    self.weight[0,0,:,:] += 1 / 45 / inputFeatures
                if 'linear' in self.basisFunctions:
                    self.weight += 1 / 45 / inputFeatures
                # self.weight[0,0,:,:]
        # self.root_weight = linearLayer
        if linearLayerActive:
            self.linearLayer = buildMLP(self.linearLayerHiddenLayout + [outputFeatures], inputFeatures, 1, False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.linearLayerActive:
            self.linearLayer.reset_parameters()
        if self.biasActive:
            zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor, edge_weights : OptTensor = None) -> Tensor:
        x_i, x_j = x

        convolution = cutlass.apply

        if self.preActivation is not None:
            x_j = self.preActivation(x_j)

        out = convolution(edge_index, x_i, x_j, edge_attr, edge_weights, self.weight, 
                                  x_i.shape[0], 0,
                              self.basisTerms , self.basisFunctions, self.basisPeriodicity, 
                              self.cutlassBatchSize, self.cutlassBatchSize) 
        if self.linearLayerActive:
            out = out + self.linearLayer(x_j)
        if self.biasActive:
            out = out + self.bias
        if self.feedThrough:
            out = out + x_j
        if self.postActivation:
            out = self.postActivation(out)

        return out

