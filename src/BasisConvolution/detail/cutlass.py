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

# This function is used to compute a continuous convolution with an arbitrary radial basis function
# The naming is a reference to NVIDIA's cutlass library which can be used to perform similar tasks for normal
# convolutions and continuous convolutions with linear basii as done in the Open3D codebase used in the original
# continuous convolution paper by Benjamin Ummenhofer
class cutlass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_index, features_i, features_j, edge_attr, edge_weights, weight, 
                dim_size, dim, size, rbfs, periodic, forwardBatchSize, backwardBatchSize, normalized = False):
        with record_function("cutlass forward step"): 
            ctx.save_for_backward(edge_index, features_i, features_j, edge_attr, edge_weights, weight)
            ctx.dimensions = len(size)
            ctx.dim_size = dim_size
            ctx.dim = dim
            ctx.size = size
            ctx.rbfs = rbfs
            ctx.periodic = periodic
            ctx.forwardBatchSize = forwardBatchSize
            ctx.backwardBatchSize = backwardBatchSize
            ctx.normalized = normalized
            
            with record_function("cutlass forward presync"):
                torch.cuda.synchronize() 

            with record_function("cutlass forward batchprep"): 
                x_j = features_j[edge_index[1]]
                x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
                indices = torch.arange(0,edge_attr.shape[0]).to(features_j.device)            
                batches = torch.split(indices, ctx.forwardBatchSize * 1024)
            out = features_i.new_zeros((features_i.shape[0], weight.shape[-1])).type(features_i.dtype)

            for batch in batches:
                if ctx.dimensions == 1:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            x = edge_attr[batch,0]
                            if ctx.rbfs[0] == 'dmcf':
                                s = torch.ones_like(x)
                                s[x<0] = -1
                                x = torch.abs(x)
                            u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            if ctx.rbfs[0] == 'dmcf':
                                # print(s, u)
                                u = s[:,None] * u
                                # print(u)
                        if ctx.normalized:
                            normalizationFactor = u.sum(-1)
                            conv = torch.einsum('nu, uio,ni -> no',u,weight, x_j[batch]) * normalizationFactor[:,None]
                        else:
                            conv = torch.einsum('nu, uio,ni -> no',u,weight, x_j[batch])
                        del u
                        out += scatter_sum(conv, index = edge_index[0,batch], dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
                if ctx.dimensions == 2:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            x = edge_attr[batch,0]
                            y = edge_attr[batch,1]
                            if ctx.rbfs[0] == 'dmcf':
                                s = torch.ones_like(x)
                                s[x<0] = -1
                                y[x<0] = -y[x<0]
                                x[x<0] = -x[x<0]
                                x = x * 2.0 - 1.0
                                # y = y * 2.0 - 1.0

                            u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            v = evalBasisFunction(ctx.size[1], y, which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                            if ctx.rbfs[0] == 'dmcf':
                                u[s < 0] = - u[s < 0]

                        with record_function("cutlass forward einsum"): 
                            if ctx.normalized:
                                normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, x_j[batch]) * normalizationFactor[:,None]
                            else:
                                conv = torch.einsum('nu, nv, uvio,ni -> no',u,v,weight, x_j[batch])
                        del u,v
                        out += scatter_sum(conv, index = edge_index[0,batch], dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
                if ctx.dimensions == 3:
                    with record_function("cutlass forward batch"): 
                        with record_function("cutlass forward basis"): 
                            u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                            v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                            w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T
                        with record_function("cutlass forward einsum"): 
                            if ctx.normalized:
                                normalizationFactor = torch.einsum('nu,nv,nw -> nuvw',u, v, w).sum(-1).sum(-1).sum(-1)
                                conv = torch.einsum('nu, nv, nw, uvwio,ni -> no',u,v,w,weight, x_j[batch]) * normalizationFactor[:,None]
                            else:
                                conv = torch.einsum('nu, nv, nw, uvwio,ni -> no',u,v,w,weight, x_j[batch])
                        del u,v,w
                        out += scatter_sum(conv, index = edge_index[0,batch], dim_size = ctx.dim_size, dim = ctx.dim)
                        del conv
            with record_function("cutlass forward postsync"):
                torch.cuda.synchronize() 
            return out
    # needs manual gradients as the auto diff version requires excessive amounts of memory and is computationally slower
    # the mathematical details here will be explained at a later point in a more written out form
    @staticmethod
    def backward(ctx, grad_output):
        with record_function("cutlass backward step"): 
            edge_index, features_i, features_j, edge_attr, edge_weights, weight = ctx.saved_tensors
            
            featureGrad = None
            weightGrad = None
            
            with record_function("cutlass backward presync"):
                torch.cuda.synchronize() 
            with record_function("cutlass backward batching"): 
                x_j = torch.index_select(features_j, 0, edge_index[1])
                x_j = x_j if edge_weights is None else x_j * edge_weights[:,None]
                gradFeatures = torch.index_select(grad_output, 0, edge_index[0])
                indices = torch.arange(0,edge_attr.shape[0]).to(features_i.device)            
                batches = torch.split(indices, ctx.backwardBatchSize * 1024)
            
            if ctx.needs_input_grad[2] and not ctx.needs_input_grad[5]:  
                with record_function("cutlass backward feature grad"):                        
                    transposedWeights = torch.transpose(weight, -2, -1)     
                    convs = []
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    x = edge_attr[batch,0]
                                    if ctx.rbfs[0] == 'dmcf':
                                        s = torch.ones_like(x)
                                        s[x<0] = -1
                                        x = torch.abs(x)
                                    u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    if ctx.rbfs[0] == 'dmcf':
                                        u = s[:,None] * u
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    if ctx.normalized:
                                        if edge_weights is not None:
                                            normalizationFactor = u.sum(-1)
                                            convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]*normalizationFactor[:,None]))
                                        else:
                                            normalizationFactor = u.sum(-1)
                                            convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]*normalizationFactor[:,None]))
                                    else:       
                                        if edge_weights is not None: 
                                            convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                        else:
                                            convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]))
                                del u
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    x = edge_attr[batch,0]
                                    y = edge_attr[batch,1]
                                    if ctx.rbfs[0] == 'dmcf':
                                        s = torch.ones_like(x)
                                        s[x<0] = -1
                                        y[x<0] = -y[x<0]
                                        x[x<0] = -x[x<0]
                                        x = x * 2.0 - 1.0
                                        # y = y * 2.0 - 1.0

                                    u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], y, which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    if ctx.rbfs[0] == 'dmcf':
                                        u[s < 0] = - u[s < 0]

                                with record_function("cutlass backward feature grad einsum"):    
                                    if ctx.normalized:
                                        if edge_weights is not None:
                                            normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, n, uvio,ni -> no',u,v, edge_weights[batch], transposedWeights, gradFeatures[batch]* normalizationFactor[:,None]) )
                                        else:
                                            normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, uvio,ni -> no',u,v, transposedWeights, gradFeatures[batch]* normalizationFactor[:,None]))
                                    else:       
                                        if edge_weights is not None: 
                                            convs.append(torch.einsum('nu, nv, n, uvio,ni -> no',u,v, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                        else:
                                            convs.append(torch.einsum('nu, nv, uvio,ni -> no',u,v, transposedWeights, gradFeatures[batch]))
                                del u,v
                        if ctx.dimensions == 3:
                            with record_function("cutlass backward feature grad batch"):    
                                with record_function("cutlass backward feature grad basis"):    
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T
                                
                                with record_function("cutlass backward feature grad einsum"):    
                                    if ctx.normalized:
                                        if edge_weights is not None:
                                            normalizationFactor = torch.einsum('nu,nv,nw -> nuvw',u, v,w).sum(-1).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, nw, n, uvwio,ni -> no',u,v,w, edge_weights[batch], transposedWeights, gradFeatures[batch]* normalizationFactor[:,None]))
                                        else:
                                            normalizationFactor = torch.einsum('nu,nv,nw -> nuvw',u, v, w).sum(-1).sum(-1).sum(-1)
                                            convs.append(torch.einsum('nu, nv, nw, uvwio,ni -> no',u,v,w, transposedWeights, gradFeatures[batch]* normalizationFactor[:,None]))
                                    else:       
                                        if edge_weights is not None: 
                                            convs.append(torch.einsum('nu, nv, nw, n, uvwio,ni -> no',u,v,w, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                        else:
                                            convs.append(torch.einsum('nu, nv, nw, uvwio,ni -> no',u,v, w,transposedWeights, gradFeatures[batch]))
                                del u,v,w
                    with record_function("cutlass backward feature grad stacking"):   
                        out = torch.vstack(convs)
                    with record_function("cutlass backward feature grad aggregation"):   
                        featureGrad = scatter_sum(out, index = edge_index[1], dim_size = features_j.shape[0], dim = ctx.dim)       
            if ctx.needs_input_grad[5] and not ctx.needs_input_grad[2]:   
                with record_function("cutlass backward weight grad"):    
                    weightGrad = weight.new_zeros(weight.shape)                    
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    x = edge_attr[batch,0]
                                    if ctx.rbfs[0] == 'dmcf':
                                        s = torch.ones_like(x)
                                        s[x<0] = -1
                                        x = torch.abs(x)
                                    u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    if ctx.rbfs[0] == 'dmcf':
                                        u = s[:,None] * u
                                    # u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    if ctx.normalized:
                                        normalizationFactor = u.sum(-1)
                                        localGrad = torch.einsum('nu, ni, no -> uio', u, x_j[batch], gradFeatures[batch] * normalizationFactor[:,None])
                                    else:                                        
                                        localGrad = torch.einsum('nu, ni, no -> uio', u, x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"): 
                                    x = edge_attr[batch,0]
                                    y = edge_attr[batch,1]
                                    if ctx.rbfs[0] == 'dmcf':
                                        s = torch.ones_like(x)
                                        s[x<0] = -1
                                        y[x<0] = -y[x<0]
                                        x[x<0] = -x[x<0]
                                        x = x * 2.0 - 1.0
                                        # y = y * 2.0 - 1.0

                                    u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], y, which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    if ctx.rbfs[0] == 'dmcf':
                                        u[s < 0] = - u[s < 0]

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    if ctx.normalized:
                                        normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                        localGrad = torch.einsum('nu, nv, ni, no -> uvio', u, v, x_j[batch], gradFeatures[batch] * normalizationFactor[:,None])
                                    else:                                        
                                        localGrad = torch.einsum('nu, nv, ni, no -> uvio', u, v,x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u,v
                        if ctx.dimensions == 3:
                            with record_function("cutlass backward weight grad batch"):   
                                with record_function("cutlass backward weight grad batch basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T

                                with record_function("cutlass backward weight grad batch einsum"):   
                                    if ctx.normalized:
                                        normalizationFactor = torch.einsum('nu,nv ,nw-> nuvw',u, v,w).sum(-1).sum(-1).sum(-1)
                                        localGrad = torch.einsum('nu, nv, nw, ni, no -> uvwio', u, v, w, x_j[batch], gradFeatures[batch]* normalizationFactor[:,None])
                                    else:                                        
                                        localGrad = torch.einsum('nu, nv, nw, ni, no -> uvwio', u, v, w, x_j[batch], gradFeatures[batch])
                                    weightGrad += localGrad
                                del u,v,w
            if ctx.needs_input_grad[2] and ctx.needs_input_grad[5]:  
                with record_function("cutlass backward"):      
                    weightGrad = weight.new_zeros(weight.shape)                    
                    transposedWeights = torch.transpose(weight, -2, -1)        
                    convs = []
                    for batch in batches:
                        if ctx.dimensions == 1:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    x = edge_attr[batch,0]
                                    if ctx.rbfs[0] == 'dmcf':
                                        s = torch.ones_like(x)
                                        s[x<0] = -1
                                        x = torch.abs(x)
                                    u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    if ctx.rbfs[0] == 'dmcf':
                                        u = s[:,None] * u
                                    # u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                # if ctx.normalized:
                                    # normalizationFactor = u.sum(-1)
                                    # u = u * normalizationFactor[:,None]
                            with record_function("cutlass backward einsum features"):   

                                if ctx.normalized:
                                    if edge_weights is not None:
                                        normalizationFactor = u.sum(-1)
                                        convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch] * normalizationFactor[:,None]))
                                    else:
                                        normalizationFactor = u.sum(-1)
                                        convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]* normalizationFactor[:,None]))
                                else:       
                                    if edge_weights is not None: 
                                        convs.append(torch.einsum('nu, n, uio,ni -> no',u, edge_weights[batch], transposedWeights, gradFeatures[batch]))
                                    else:
                                        convs.append(torch.einsum('nu, uio,ni -> no',u, transposedWeights, gradFeatures[batch]))
                            with record_function("cutlass backward einsum grad"):   
                                if ctx.normalized:
                                    io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch]* normalizationFactor[:,None])
                                else:
                                    io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nu, nio -> uio', u, io)
                                weightGrad += localGrad
                        if ctx.dimensions == 2:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    x = edge_attr[batch,0]
                                    y = edge_attr[batch,1]
                                    if ctx.rbfs[0] == 'dmcf':
                                        s = torch.ones_like(x)
                                        s[x<0] = -1
                                        y[x<0] = -y[x<0]
                                        x[x<0] = -x[x<0]
                                        x = x * 2.0 - 1.0
                                        # y = y * 2.0 - 1.0

                                    u = evalBasisFunction(ctx.size[0], x, which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], y, which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    if ctx.rbfs[0] == 'dmcf':
                                        u[s < 0] = - u[s < 0]  

                            with record_function("cutlass backward einsum uvw"):   
                                if ctx.normalized:
                                    normalizationFactor = torch.einsum('nu,nv -> nuv',u, v).sum(-1).sum(-1)
                                    uvw = torch.einsum('nu, nv -> nuv', u, v)
                                else:
                                    uvw = torch.einsum('nu, nv -> nuv', u, v) 
                                del u,v
                            with record_function("cutlass backward einsum features"):   
                                if edge_weights is not None:
                                    convs.append(torch.einsum('nuv, n, uvio,ni -> no',uvw, edge_weights[batch], transposedWeights, gradFeatures[batch]* (normalizationFactor[:,None] if ctx.normalized else 1)))
                                else:
                                    convs.append(torch.einsum('nuv, uvio,ni -> no',uvw, transposedWeights, gradFeatures[batch]* (normalizationFactor[:,None] if ctx.normalized else 1)))
                            with record_function("cutlass backward einsum grad"):   
                                if ctx.normalized:
                                    io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch]* normalizationFactor[:,None])
                                else:
                                    io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nuv, nio -> uvio', uvw, io)
                                weightGrad += localGrad
                        if ctx.dimensions == 3:
                            with record_function("cutlass backward batch"):   
                                with record_function("cutlass backward basis"):   
                                    u = evalBasisFunction(ctx.size[0], edge_attr[batch,0], which=ctx.rbfs[0], periodic = ctx.periodic[0]).T
                                    v = evalBasisFunction(ctx.size[1], edge_attr[batch,1], which=ctx.rbfs[1], periodic = ctx.periodic[1]).T
                                    w = evalBasisFunction(ctx.size[2], edge_attr[batch,2], which=ctx.rbfs[2], periodic = ctx.periodic[2]).T
                            with record_function("cutlass backward einsum uvw"):   
                                if ctx.normalized:
                                    normalizationFactor = torch.einsum('nu,nv,nw -> nuvw',u, v,w).sum(-1).sum(-1).sum(-1)
                                    uvw = torch.einsum('nu, nv, nw -> nuvw', u, v, w) 
                                else:
                                    uvw = torch.einsum('nu, nv, nw -> nuvw', u, v, w) 
                                del u,v, w
                            with record_function("cutlass backward einsum features"):   
                                if edge_weights is not None:
                                    convs.append(torch.einsum('nuvw, n, uvwio,ni -> no',uvw, edge_weights[batch], transposedWeights, gradFeatures[batch]* (normalizationFactor[:,None] if ctx.normalized else 1)))
                                else:
                                    convs.append(torch.einsum('nuvw, uvwio,ni -> no',uvw, transposedWeights, gradFeatures[batch]* (normalizationFactor[:,None] if ctx.normalized else 1)))
                            with record_function("cutlass backward einsum grad"):   
                                if ctx.normalized:
                                    io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch]* normalizationFactor[:,None])
                                else:
                                    io = torch.einsum('ni, no -> nio', x_j[batch], gradFeatures[batch])
                                localGrad = torch.einsum('nuvw, nio -> uvwio', uvw, io)
                                weightGrad += localGrad
                    with record_function("cutlass backward stacking"):   
                        out = torch.vstack(convs)
                    with record_function("cutlass backward aggregation"):   
                        featureGrad = scatter_sum(out, index = edge_index[1], dim_size = features_j.shape[0], dim = ctx.dim) 
            with record_function("cutlass backward postsync"):
                torch.cuda.synchronize() 
            return None, None, featureGrad, None, None, weightGrad, None, None, None, None, None, None, None, None 
