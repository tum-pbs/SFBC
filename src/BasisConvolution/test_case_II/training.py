import torch
from torch.profiler import profile, record_function, ProfilerActivity
from .util import constructFluidFeatures
from BasisConvolution.detail.radius import radius
import portalocker
from BasisConvolution.detail.augment import augment
import numpy as np
from .datautils import loadFrame


def loadBatch(train_ds, bdata, featureFun, unroll = 1, frameDistance = 1, augmentAngle = False, augmentJitter = False, jitterAmount = 0.01, adjustForFrameDistance = True):
    with record_function("load batch - hdf5"): 
        fluidPositions = []
        boundaryPositions = []
        fluidFeatures = []
        boundaryFeatures = []
        fluidBatchIndices = []
        boundaryBatchIndices = []
        groundTruths = []
        fluidGravities = []
        attributeArray = []
        for i in range(unroll):
            groundTruths.append([])

        for i,b in enumerate(bdata):
            with record_function("load batch - hdf5[batch]"): 
        #         debugPrint(i)
        #         debugPrint(b)
                attributes, fluidPosition, boundaryPosition, fluidFeature, boundaryFeature, fluidGravity, groundTruth = loadData(train_ds, b, featureFun, unroll = unroll, frameDistance = frameDistance,\
                                augmentAngle = torch.rand(1)[0] if augmentAngle else 0., augmentJitter = jitterAmount if augmentJitter else 0., adjustForFrameDistance = adjustForFrameDistance)     
        #         debugPrint(groundTruth)
                fluidPositions.append(fluidPosition)
                attributeArray.append(attributes)
        #         debugPrint(fluidPositions)
                boundaryPositions.append(boundaryPosition)
                fluidFeatures.append(fluidFeature)
                boundaryFeatures.append(boundaryFeature)
                
                fluidGravities.append(fluidGravity)

                batchIndex = torch.ones(fluidPosition.shape[0]) * i
                fluidBatchIndices.append(batchIndex)

                batchIndex = torch.ones(boundaryPosition.shape[0]) * i
                boundaryBatchIndices.append(batchIndex)
                for u in range(unroll):
                    groundTruths[u].append(groundTruth[u])

        fluidPositions = torch.vstack(fluidPositions)
        boundaryPositions = torch.vstack(boundaryPositions)
        fluidFeatures = torch.vstack(fluidFeatures)
        boundaryFeatures = torch.vstack(boundaryFeatures)
        fluidGravities = torch.vstack(fluidGravities)
        fluidBatchIndices = torch.hstack(fluidBatchIndices)
        boundaryBatchIndices = torch.hstack(boundaryBatchIndices)
        for u in range(unroll):
            groundTruths[u] = torch.vstack(groundTruths[u])

        return fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidGravities, fluidBatchIndices, boundaryBatchIndices, groundTruths, attributeArray

def processBatch(model, device, li, e, unroll, train_ds, bdata, frameDistance, augmentAngle = False, augmentJitter = False, jitterAmount = 0.01, adjustForFrameDistance = True):
    with record_function("process batch"): 
        fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, fluidGravity, fluidBatches, boundaryBatches, groundTruths, attributes = \
            loadBatch(train_ds, bdata, constructFluidFeatures, unroll, frameDistance, augmentAngle = augmentAngle, augmentJitter = augmentJitter, jitterAmount = jitterAmount, adjustForFrameDistance = adjustForFrameDistance)    


        predictedPositions = fluidPositions.to(device)
        predictedVelocity = fluidFeatures[:,1:3].to(device)

        bLosses = []
        boundaryPositions = boundaryPositions.to(device)
        fluidFeatures = fluidFeatures.to(device)
        boundaryFeatures = boundaryFeatures.to(device)
        fluidBatches = fluidBatches.to(device)
        boundaryBatches = boundaryBatches.to(device)

        gravity = torch.zeros_like(predictedVelocity)
        gravity = fluidGravity[:,:2].to(device)
        
    #     gravity[:,1] = -9.81

        for u in range(unroll):
            with record_function("prcess batch[unroll]"): 
    #         loss, predictedPositions, predictedVelocity = runNetwork(fluidPositions.to(device), inputData['fluidVelocity'].to(device), attributes['dt'], frameDistance, gravity, fluidFeatures, boundaryPositions.to(device), boundaryFeatures.to(device), groundTruths[0], model, None, None, True)
                loss, predictedPositions, predictedVelocity = runNetwork(predictedPositions, predictedVelocity, attributes[0], frameDistance, gravity, fluidFeatures, boundaryPositions, boundaryFeatures, groundTruths[u], model, fluidBatches, boundaryBatches, li)

                batchedLoss = []
                for i in range(len(bdata)):
                    L = loss[fluidBatches == i]
                    Lterms = (torch.mean(L), torch.max(torch.abs(L)), torch.min(torch.abs(L)), torch.std(L))            
                    batchedLoss.append(torch.hstack(Lterms))
                batchedLoss = torch.vstack(batchedLoss).unsqueeze(0)
                bLosses.append(batchedLoss)

        bLosses = torch.vstack(bLosses)
        maxLosses = torch.max(bLosses[:,:,1], dim = 0)[0]
        minLosses = torch.min(bLosses[:,:,2], dim = 0)[0]
        meanLosses = torch.mean(bLosses[:,:,0], dim = 0)
        stdLosses = torch.mean(bLosses[:,:,3], dim = 0)


        del predictedPositions, predictedVelocity, boundaryPositions, fluidFeatures, boundaryFeatures, fluidBatches, boundaryBatches

        bLosses = bLosses.transpose(0,1)

        return bLosses, meanLosses, minLosses, maxLosses, stdLosses


def runNetwork(initialPosition, initialVelocity, attributes, frameDistance, gravity, fluidFeatures, boundaryPositions, boundaryFeatures, groundTruth, model,fluidBatches, boundaryBatches, li):
    # if verbose:
    #     print('running network with')
    #     print('initialPosition', initialPosition[:4])
    #     print('initialVelocity', initialVelocity[:4])
    #     print('dt', dt)
    #     print('frameDistance', frameDistance)        
    #     print('gravity', gravity[:4])
    #     print('fluidFeatures', fluidFeatures[:4])
    #     print('boundaryPositions', boundaryPositions[:4])
    #     print('boundaryFeatures', boundaryFeatures[:4])
    #     print('fluidBatches', fluidBatches)
    #     print('boundaryBatches', boundaryBatches)
    #     print('li', li)
# Heun's method:
    # vel2 = initialVelocity + frameDistance * attributes['dt'] * gravity
    # pos2 = initialPosition + frameDistance * attributes['dt'] * (initialVelocity + vel2) / 2
# semi implicit euler
    d = (frameDistance) * ((frameDistance) + 1) / 2
    vel2 = initialVelocity + frameDistance * attributes['dt'] * gravity
    pos2 = initialPosition + frameDistance * attributes['dt'] * initialVelocity + d * attributes['dt']**2 * gravity
        
    fluidFeatures = torch.hstack((fluidFeatures[:,0][:,None], vel2, fluidFeatures[:,3:]))
    # if verbose:
    #     print('calling network with' )
    #     print('d', d)
    #     print('vel2', vel2[:4])
    #     print('pos2', pos2[:4])
    #     print('fluidFeatures', fluidFeatures[:4])
    
    fi, fj = radius(pos2, pos2, attributes['support'], max_num_neighbors = 256, batch_x = fluidBatches, batch_y = fluidBatches)
    bf, bb = radius(boundaryPositions, pos2, attributes['support'], max_num_neighbors = 256, batch_x = boundaryBatches, batch_y = fluidBatches)
#     if self.centerIgnore:
#         nequals = fi != fj

    i, ni = torch.unique(fi, return_counts = True)
    b, nb = torch.unique(bf, return_counts = True)

#     self.ni = ni
#     self.nb = nb

    ni[i[b]] += nb
#     print('min: %g, mean: %g, max: %g' %( torch.min(ni), torch.mean(ni.type(torch.float32)), torch.max(ni)))
    boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
    boundaryEdgeLengths = (boundaryPositions[boundaryEdgeIndex[1]] - pos2[boundaryEdgeIndex[0]])/attributes['support']
    boundaryEdgeLengths = boundaryEdgeLengths.clamp(-1,1)
    fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
    fluidEdgeLengths = -(pos2[fluidEdgeIndex[1]] - pos2[fluidEdgeIndex[0]])/attributes['support']
    fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
    
    
    predictions = model(fluidFeatures, fi, fj, fluidEdgeLengths, boundaryFeatures, bf, bb, boundaryEdgeLengths)
#     predictions = model(pos2, boundaryPositions, fluidFeatures, boundaryFeatures, attributes, fluidBatches, boundaryBatches)

    predictedVelocity = (pos2 + predictions[:,:2] - initialPosition) / (frameDistance * attributes['dt'])
    predictedPositions = pos2 + predictions[:,:2]

    if li:
        loss =  model.li * computeLoss(predictedPositions, predictedVelocity, groundTruth.to(pos2.device), predictions)
    else:
        loss =   computeLoss(predictedPositions, predictedVelocity, groundTruth.to(pos2.device), predictions)

    return loss, predictedPositions, predictedVelocity
    
def computeLoss(predictedPosition, predictedVelocity, groundTruth, modelOutput):
#     debugPrint(modelOutput.shape)
#     debugPrint(groundTruth.shape)
#     return torch.sqrt((modelOutput - groundTruth[:,-1:].to(device))**2)
    # return torch.abs(modelOutput - groundTruth[:,-1:].to(modelOutput.device))
    # return torch.linalg.norm(groundTruth[:,2:4] - predictedVelocity, dim = 1) 
    # debugPrint(groundTruth.shape)
    # debugPrint(predictedPosition.shape)
    # debugPrint(predictedVelocity.shape)
    posLoss = torch.sqrt(torch.linalg.norm(groundTruth[:,:2] - predictedPosition, dim = 1))
#     if verbose:
#         print('computing Loss with')
#         print('predictedPositions', predictedPosition[:4])
#         print('predictedVelocity', predictedVelocity[:4])
#         print('groundTruth', groundTruth[:4])
#         print('modelOutput', modelOutput[:4])
#         print('resulting loss', posLoss[:4])
    return posLoss
    velLoss = torch.sqrt(torch.linalg.norm(groundTruth[:,2:4] - predictedVelocity, dim = 1))
    return posLoss + velLoss
    # return torch.sqrt(torch.linalg.norm(groundTruth[:,2:4] - modelOutput, dim = 1))

def loadData(dataset, index, featureFun, unroll = 1, frameDistance = 1, augmentAngle = 0., augmentJitter = 0., adjustForFrameDistance = True):
    with record_function("load data - hdf5"): 
        fileName, frameIndex, maxRollouts = dataset[index]

        attributes, inputData, groundTruthData = loadFrame(fileName, frameIndex, 1 + np.arange(unroll), frameDistance = frameDistance, adjustForFrameDistance = adjustForFrameDistance)
        # attributes['support'] = 4.5 * attributes['support']
        if augmentAngle != 0 or augmentJitter != 0:
            attributes, inputData, groundTruthData = augment(attributes, inputData, groundTruthData, augmentAngle, augmentJitter)
        fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures = featureFun(attributes, inputData)

        return attributes, fluidPositions, boundaryPositions, fluidFeatures, boundaryFeatures, inputData['fluidGravity'], groundTruthData
