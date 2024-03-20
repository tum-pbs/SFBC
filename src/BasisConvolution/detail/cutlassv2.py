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
#

import torch
from torch.profiler import record_function
# import numpy as np
# from typing import Dict, Optional
from .basis import evalBasisFunction
from .scatter import scatter_sum

from typing import Dict, Optional, List
import numpy as np

@torch.jit.script
def sine(x: torch.Tensor, n : int, phase : float = 0):
    fx = torch.arange(n, device = x.device, dtype = x.dtype).unsqueeze(0).repeat(x.shape[0],1) + 1
    return torch.sin(fx * (x.unsqueeze(1)) * np.pi + phase)


@torch.jit.script
def convolutionOpBackwardFeatures(grad_output: torch.Tensor, edge_index : torch.Tensor, features_i : torch.Tensor, features_j : torch.Tensor, edge_attr : torch.Tensor, edge_weights : Optional[torch.Tensor], weight : torch.Tensor, 
                  dim_size : int, dim : int, size : List[int], rbfs : List[str], periodic : List[bool], forwardBatchSize : int, backwardBatchSize : int, normalized: bool = False):
    # with record_function("convolution op - Feature Lookup"): 
    #     x_j = features_j[edge_index[1]]
    # with record_function("convolution op - Weight Function Application"): 
    #     x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
    with record_function("convolution op - Batch Generation"): 
        indices = torch.arange(0,edge_attr.shape[0], device = features_j.device, dtype = torch.int64)         
        batches = torch.split(indices, forwardBatchSize * 1024)    
    with record_function("convolution op - Output Allocation"): 
        out = features_i.new_zeros((features_i.shape[0], weight.shape[-1])).type(features_i.dtype)
    transposedWeights = torch.transpose(weight, -2, -1)    
    featureGrad = torch.zeros_like(features_j)        
    for batch in batches:
        jj = edge_index[1,batch]
        x_j = features_j[jj]
        x_j = x_j if edge_weights is None else x_j * edge_weights[batch,None]
        with record_function("convolution op - Batch"): 
            with record_function("convolution op - Batch - DMCF Mapping"): 
                coords = [edge_attr[batch,i] for i in range(edge_attr.shape[1])]
                mask = coords[0] < 0
                if rbfs[0] == 'dmcf':
                    for i in range(edge_attr.shape[1]-1, -1, -1):
                        coords[i][mask] = -coords[i][mask]
                    coords[0] = 2 * coords[0] - 1
            with record_function("convolution op - Batch - Basis Function Eval"): 
#                 basisValues = [sine(coords[i], size[i], 1/3 * np.pi * i) for i in range(edge_attr.shape[1])]
                basisValues = [evalBasisFunction(size[i], coords[i], which=rbfs[i], periodic = periodic[i]).T for i in range(edge_attr.shape[1])]
            with record_function("convolution op - Batch - DMCF Back Mapping"): 
                if rbfs[0] == 'dmcf':
                    basisValues[0][mask] = -basisValues[0][mask]
            with record_function("convolution op - Batch - Convolution"): 
                gradValues = grad_output[edge_index[0, batch]]
                bM = basisValues[0]
                for i in range(1, edge_attr.shape[1]):
                    bM = torch.einsum('nu, nv -> nuv', bM, basisValues[i]).flatten(1)
#                 if normalized:
#                     normalizationFactor = torch.sum(bM, dim = 1)
#                 else:
#                     normalizationFactor = torch.ones(bM.shape[0]).to(bM.dtype).to(bM.device)

                res = torch.matmul(
                    (bM * edge_weights[batch].unsqueeze(1)) if edge_weights is not None else bM,
                    transposedWeights.flatten(0,-3).flatten(1)
                        ).reshape(-1, transposedWeights.shape[-2], transposedWeights.shape[-1])
                res = res.transpose(-2,-1)
                conv = torch.matmul(
                    res, 
                    ((gradValues / torch.sum(bM, dim = 1)[:,None]) if normalized else gradValues).unsqueeze(2))[:,:,0]
            with record_function("convolution op - Batch - Scatter"): 
                featureGrad += scatter_sum(conv, index = edge_index[1,batch], dim_size = features_j.shape[0], dim = dim)
    return featureGrad

@torch.jit.script
def convolutionOpBackwardWeight(grad_output: torch.Tensor, edge_index : torch.Tensor, features_i : torch.Tensor, features_j : torch.Tensor, edge_attr : torch.Tensor, edge_weights :Optional[torch.Tensor], weight : torch.Tensor, 
                  dim_size : int, dim : int, size : List[int], rbfs : List[str], periodic : List[bool], forwardBatchSize : int, backwardBatchSize : int, normalized: bool = False):
    # with record_function("convolution op - Feature Lookup"): 
        # x_j = features_j[edge_index[1]]
    # with record_function("convolution op - Weight F/unction Application"): 
        # x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
    with record_function("convolution op - Batch Generation"): 
        indices = torch.arange(0,edge_attr.shape[0], device = features_j.device, dtype = torch.int64)         
        batches = torch.split(indices, forwardBatchSize * 1024)    
        
    gradFeatures = torch.index_select(grad_output, 0, edge_index[0])   
    transposedWeights = torch.transpose(weight, -2, -1)        
    weightGrad = weight.new_zeros(weight.shape)                    
    for batch in batches:
        jj = edge_index[1,batch]
        x_j = features_j[jj]
        x_j = x_j if edge_weights is None else x_j * edge_weights[batch,None]
#         print(batch, batches)
        with record_function("convolution op - Batch"): 
            with record_function("convolution op - Batch - DMCF Mapping"): 
                coords = [edge_attr[batch,i] for i in range(edge_attr.shape[1])]
                mask = coords[0] < 0
                if rbfs[0] == 'dmcf':
                    for i in range(edge_attr.shape[1]-1, -1, -1):
                        coords[i][mask] = -coords[i][mask]
                    coords[0] = 2 * coords[0] - 1
            with record_function("convolution op - Batch - Basis Function Eval"): 
#                 basisValues = [sine(coords[i], size[i], 1/3 * np.pi * i) for i in range(edge_attr.shape[1])]
                basisValues = [evalBasisFunction(size[i], coords[i], which=rbfs[i], periodic = periodic[i]).T for i in range(edge_attr.shape[1])]
            with record_function("convolution op - Batch - DMCF Back Mapping"): 
                if rbfs[0] == 'dmcf':
                    basisValues[0][mask] = -basisValues[0][mask]
            with record_function("convolution op - Batch - Convolution"): 
                gradValues = gradFeatures[batch]
                bM = basisValues[0]
                for i in range(1, edge_attr.shape[1]):
                    bM = torch.einsum('nu, nv -> nuv', bM, basisValues[i]).flatten(1)
                u = basisValues[0]
#                 localGrad = torch.einsum('nu, ni, no -> uio', u, x_j[batch], (gradFeatures[batch] / torch.sum(bM, dim = 1)[:,None] ) if normalized else gradFeatures[batch])
#                 if normalized:
#                     normalizationFactor = torch.sum(bM, dim = 1)
#                 else:
#                     normalizationFactor = torch.ones(bM.shape[0]).to(bM.dtype).to(bM.device)
# #                 print(edge_)
# #                 print(bM.shape)
                io = torch.einsum('ni, no -> nio', x_j, (gradValues / torch.sum(bM, dim = 1)[:,None]) if normalized else gradValues)
# #                 print(io.shape)
# #                 print(weight.shape)
                localGrad = torch.einsum('nx, nio -> xio', bM, io).reshape(weight.shape)
                weightGrad += localGrad
    return weightGrad

@torch.jit.script
def convolutionOpBackward(grad_output: torch.Tensor, edge_index : torch.Tensor, features_i : torch.Tensor, features_j : torch.Tensor, edge_attr : torch.Tensor, edge_weights : Optional[torch.Tensor], weight : torch.Tensor, 
                  dim_size : int, dim : int, size : List[int], rbfs : List[str], periodic : List[bool], forwardBatchSize : int, backwardBatchSize : int, normalized: bool = False, weightGradient : bool = True, featureGradient : bool = True):
    # with record_function("convolution op - Feature Lookup"): 
        # x_j = features_j[edge_index[1]]
    # with record_function("convolution op - Weight Function Application"): 
        # x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
    with record_function("convolution op - Batch Generation"): 
        indices = torch.arange(0,edge_attr.shape[0], device = features_j.device, dtype = torch.int64)         
        batches = torch.split(indices, forwardBatchSize * 1024)    
    with record_function("convolution op - Output Allocation"): 
        out = features_i.new_zeros((features_i.shape[0], weight.shape[-1])).type(features_i.dtype)
    transposedWeights = torch.transpose(weight, -2, -1)         
    
    featureGrad = torch.zeros_like(features_j)
    weightGrad = weight.new_zeros(weight.shape)           
    gradFeatures = torch.index_select(grad_output, 0, edge_index[0])    

    for batch in batches:
        jj = edge_index[1,batch]
        x_j = features_j[jj]
        x_j = x_j if edge_weights is None else x_j * edge_weights[batch,None]
        with record_function("convolution op - Batch"): 
            with record_function("convolution op - Batch - DMCF Mapping"): 
                coords = [edge_attr[batch,i] for i in range(edge_attr.shape[1])]
                mask = coords[0] < 0
                if rbfs[0] == 'dmcf':
                    for i in range(edge_attr.shape[1]-1, -1, -1):
                        coords[i][mask] = -coords[i][mask]
                    coords[0] = 2 * coords[0] - 1
            with record_function("convolution op - Batch - Basis Function Eval"): 
#                 basisValues = [sine(coords[i], size[i], 1/3 * np.pi * i) for i in range(edge_attr.shape[1])]
                basisValues = [evalBasisFunction(size[i], coords[i], which=rbfs[i], periodic = periodic[i]).T for i in range(edge_attr.shape[1])]
            with record_function("convolution op - Batch - DMCF Back Mapping"): 
                if rbfs[0] == 'dmcf':
                    basisValues[0][mask] = -basisValues[0][mask]
            with record_function("convolution op - Batch - Convolution"): 
                gradValues = gradFeatures[batch]
                bM = basisValues[0]
                for i in range(1, edge_attr.shape[1]):
                    bM = torch.einsum('nu, nv -> nuv', bM, basisValues[i]).flatten(1)
#                 if normalized:
#                     normalizationFactor = torch.sum(bM, dim = 1)
#                 else:
#                     normalizationFactor = torch.ones(bM.shape[0]).to(bM.dtype).to(bM.device)

                io = torch.einsum('ni, no -> nio', x_j, (gradValues / torch.sum(bM, dim = 1)[:,None]) if normalized else gradValues)
                localGrad = torch.einsum('nx, nio -> xio', bM, io).reshape(weight.shape)
                weightGrad += localGrad

                res = torch.matmul(
                    (bM * edge_weights[batch].unsqueeze(1)) if edge_weights is not None else bM,
                    transposedWeights.flatten(0,-3).flatten(1)
                        ).reshape(-1, transposedWeights.shape[-2], transposedWeights.shape[-1])
                res = res.transpose(-2,-1)
                conv = torch.matmul(
                    res, 
                    ((gradValues / torch.sum(bM, dim = 1)[:,None]) if normalized else gradValues).unsqueeze(2))[:,:,0]
            with record_function("convolution op - Batch - Scatter"): 
                    featureGrad += scatter_sum(conv, index = edge_index[1,batch], dim_size = features_j.shape[0], dim = dim)
    return featureGrad, weightGrad

@torch.jit.script
def convolutionOp(edge_index : torch.Tensor, features_i : torch.Tensor, features_j : torch.Tensor, edge_attr : torch.Tensor, edge_weights : Optional[torch.Tensor], weight : torch.Tensor, 
                  dim_size : int, dim : int, size : List[int], rbfs : List[str], periodic : List[bool], forwardBatchSize : int, backwardBatchSize : int, normalized: bool = False):
    # with record_function("convolution op - Feature Lookup"): 
    #     x_j = features_j[edge_index[1]]
    # with record_function("convolution op - Weight Function Application"): 
    #     x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
    with record_function("convolution op - Batch Generation"): 
        indices = torch.arange(0,edge_attr.shape[0], device = features_j.device, dtype = torch.int64)         
        batches = torch.split(indices, forwardBatchSize * 1024)    
    with record_function("convolution op - Output Allocation"): 
        out = features_i.new_zeros((features_i.shape[0], weight.shape[-1])).type(features_i.dtype)
    for batch in batches:
        jj = edge_index[1,batch]
        x_j = features_j[jj]
        x_j = x_j if edge_weights is None else x_j * edge_weights[batch,None]
        with record_function("convolution op - Batch"): 
            with record_function("convolution op - Batch - DMCF Mapping"): 
                coords = [edge_attr[batch,i] for i in range(edge_attr.shape[1])]
                mask = coords[0] < 0
                if rbfs[0] == 'dmcf':
                    for i in range(edge_attr.shape[1]-1, -1, -1):
                        coords[i][mask] = -coords[i][mask]
                    coords[0] = 2 * coords[0] - 1
            with record_function("convolution op - Batch - Basis Function Eval"): 
#                 basisValues = [sine(coords[i], size[i], 1/3 * np.pi * i) for i in range(edge_attr.shape[1])]
                basisValues = [evalBasisFunction(size[i], coords[i], which=rbfs[i], periodic = periodic[i]).T for i in range(edge_attr.shape[1])]
            with record_function("convolution op - Batch - DMCF Back Mapping"): 
                if rbfs[0] == 'dmcf':
                    basisValues[0][mask] = -basisValues[0][mask]
            with record_function("convolution op - Batch - Convolution"): 
                bM = basisValues[0]
                for i in range(1, edge_attr.shape[1]):
                    bM = torch.einsum('nu, nv -> nuv', bM, basisValues[i]).flatten(1)
                if normalized:
                    bM = bM / torch.sum(bM, dim = 1).unsqueeze(1)
                res = torch.matmul(bM, weight.flatten(0,-3).flatten(1)).reshape(-1, weight.shape[-2], weight.shape[-1])
                res = res.reshape(-1, weight.shape[-2], weight.shape[-1]).transpose(-2,-1)
                conv = torch.matmul(res, x_j.unsqueeze(2))[:,:,0]
            with record_function("convolution op - Batch - Scatter"): 
                out += scatter_sum(conv, index = edge_index[0,batch], dim_size = dim_size, dim = dim)
    return out

class cutlass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_index : torch.Tensor, features_i : torch.Tensor, features_j : torch.Tensor, edge_attr : torch.Tensor, edge_weights : Optional[torch.Tensor], weight : torch.Tensor, 
                  dim_size : int, dim : int, size : List[int], rbfs : List[str], periodic : List[bool], forwardBatchSize : int, backwardBatchSize : int, normalized: bool = False):
        with record_function("cutlass forward step"): 
            ctx.save_for_backward(edge_index, features_i, features_j, edge_attr, edge_weights, weight)
#             print('fwd', normalized)
            ctx.dimensions = len(size)
            ctx.dim_size = dim_size
            ctx.dim = dim
            ctx.size = size
            ctx.rbfs = rbfs
            ctx.periodic = periodic
            ctx.forwardBatchSize = forwardBatchSize
            ctx.backwardBatchSize = backwardBatchSize
            ctx.normalized = normalized

            return convolutionOp(edge_index, features_i, features_j, edge_attr, edge_weights, weight, dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized)
    @staticmethod
    def backward(ctx, grad_output):
        with record_function("cutlass backward step"): 
            edge_index, features_i, features_j, edge_attr, edge_weights, weight = ctx.saved_tensors
            dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized = ctx.dim_size, ctx.dim, ctx.size, ctx.rbfs, ctx.periodic, ctx.forwardBatchSize, ctx.backwardBatchSize, ctx.normalized

#             print('bck', normalized)
            featureGrad = None
            weightGrad = None
            if ctx.needs_input_grad[2] and not ctx.needs_input_grad[5]:  
                with record_function("cutlass backward feature Gradient"): 
                    featureGrad = convolutionOpBackwardFeatures(grad_output, edge_index, features_i, features_j, edge_attr, edge_weights, weight, dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized)
            if not ctx.needs_input_grad[2] and ctx.needs_input_grad[5]:  
                with record_function("cutlass backward - weight Gradient"): 
                    weightGrad = convolutionOpBackwardWeight(grad_output, edge_index, features_i, features_j, edge_attr, edge_weights, weight, dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized)
            if ctx.needs_input_grad[2] and ctx.needs_input_grad[5]:  
                with record_function("cutlass backward - both"): 
                    featureGrad, weightGrad = convolutionOpBackward(grad_output, edge_index, features_i, features_j, edge_attr, edge_weights, weight, dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized)

            return None, None, featureGrad, None, None, weightGrad, None, None, None, None, None, None, None, None 