from BasisConvolution.util.arguments import parser
import shlex
from torch.optim import Adam
import torch

from BasisConvolution.convNetv2 import BasisNetwork
from BasisConvolution.detail.windows import getWindowFunction
from BasisConvolution.detail.util import count_parameters


def buildModel(hyperParameterDict, verbose = False):

    fluidFeatureCount = hyperParameterDict['fluidFeatureCount']
    boundaryFeaturecount = hyperParameterDict['boundaryFeatureCount']
    layers = hyperParameterDict['layers']
    coordinateMapping = hyperParameterDict['coordinateMapping']
    windowFunction = getWindowFunction(hyperParameterDict['windowFunction'])

    rbfs = hyperParameterDict['rbfs']
    dims = hyperParameterDict['dims']

    cutlassBatchSize = hyperParameterDict['cutlassBatchSize']
    normalized = hyperParameterDict['normalized']
    outputBias = hyperParameterDict['outputBias']
    initializer = hyperParameterDict['initializer']
    optimizeWeights = hyperParameterDict['optimizeWeights']
    exponentialDecay = hyperParameterDict['exponentialDecay']
    inputEncoder = hyperParameterDict['inputEncoder'] if 'inputEncoder' in hyperParameterDict else None
    outputDecoder = hyperParameterDict['outputDecoder'] if 'outputDecoder' in hyperParameterDict else None
    edgeMLP = hyperParameterDict['edgeMLP'] if 'edgeMLP' in hyperParameterDict else None
    vertexMLP = hyperParameterDict['vertexMLP'] if 'vertexMLP' in hyperParameterDict else None

    if verbose:
        print(f'fluidFeatureCount: {fluidFeatureCount}')
        print(f'boundaryFeaturecount: {boundaryFeaturecount}')
        print(f'layers: {layers}')
        print(f'coordinateMapping: {coordinateMapping}')
        print(f'windowFunction: {windowFunction}')

        print(f'rbfs: {rbfs}')
        print(f'dims: {dims}')

        print(f'cutlassBatchSize: {cutlassBatchSize}')
        print(f'normalized: {normalized}')
        print(f'outputBias: {outputBias}')
        print(f'initializer: {initializer}')
        print(f'optimizeWeights: {optimizeWeights}')
        print(f'exponentialDecay: {exponentialDecay}')
        print(f'inputEncoder: {inputEncoder}')
        print(f'outputDecoder: {outputDecoder}')
        print(f'edgeMLP: {edgeMLP}')
        print(f'vertexMLP: {vertexMLP}')

    model = BasisNetwork(fluidFeatureCount, boundaryFeaturecount, layers = layers, coordinateMapping = coordinateMapping, windowFn = windowFunction, rbfs = rbfs, dims = dims, batchSize = cutlassBatchSize, normalized = normalized, outputBias = outputBias, initializer = initializer, optimizeWeights = optimizeWeights, exponentialDecay = exponentialDecay, inputEncoder = inputEncoder, outputDecoder = outputDecoder, edgeMLP = edgeMLP, vertexMLP = vertexMLP)

    model = model.to(hyperParameterDict['device'])

    parameterCount = count_parameters(model)
    hyperParameterDict['parameterCount'] = parameterCount


    optimizer = Adam(model.parameters(), lr=hyperParameterDict['initialLR'], weight_decay = hyperParameterDict['weight_decay'])
    # model = model.to(device)

    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = hyperParameterDict['gamma'])

    return model, optimizer, scheduler


def runInference(perennialState, config, model, batches = 1, verbose = False):
    boundaryFeatures = None
    bf = None
    bb = None
    boundaryEdgeLengths = None
    offsets = []

    if isinstance(perennialState, list):
        numStates = len(perennialState)
        offsets = [0]
        fluidFeatures = perennialState[0]['fluid']['features']
        fi, fj = perennialState[0]['fluid']['neighborhood']['indices']
        fluidEdgeLengths = perennialState[0]['fluid']['neighborhood']['distances'].view(-1,1) * perennialState[0]['fluid']['neighborhood']['vectors']
        fluidOffset = perennialState[0]['fluid']['numParticles']
        offsets.append(fluidOffset)

        if 'boundary' in perennialState[0] and perennialState[0]['boundary'] is not None:
            boundaryFeatures = perennialState[0]['boundary']['features']
            bb, bf = perennialState[0]['fluidToBoundaryNeighborhood']['indices']
            boundaryEdgeLengths = perennialState[0]['fluidToBoundaryNeighborhood']['distances'].view(-1,1) * perennialState[0]['fluidToBoundaryNeighborhood']['vectors']
            boundaryOffset = perennialState[0]['boundary']['numParticles']
        
        for i in range(1, numStates):
            fluidFeatures = torch.cat([fluidFeatures, perennialState[i]['fluid']['features']], dim = 0)
            fi = torch.cat([fi, perennialState[i]['fluid']['neighborhood']['indices'][0] + fluidOffset], dim = 0)
            fj = torch.cat([fj, perennialState[i]['fluid']['neighborhood']['indices'][1] + fluidOffset], dim = 0)
            fluidEdgeLengths = torch.cat([fluidEdgeLengths, perennialState[i]['fluid']['neighborhood']['distances'].view(-1,1) * perennialState[i]['fluid']['neighborhood']['vectors']], dim = 0)

            if 'boundary' in perennialState[0] and perennialState[0]['boundary'] is not None:
                boundaryFeatures = torch.cat([boundaryFeatures, perennialState[i]['boundary']['features']], dim = 0)
                bb = torch.cat([bb, perennialState[i]['fluidToBoundaryNeighborhood']['indices'][0] + boundaryOffset], dim = 0)
                bf = torch.cat([bf, perennialState[i]['fluidToBoundaryNeighborhood']['indices'][1] + fluidOffset], dim = 0)
                boundaryEdgeLengths = torch.cat([boundaryEdgeLengths, perennialState[i]['fluidToBoundaryNeighborhood']['distances'].view(-1,1) * perennialState[i]['fluidToBoundaryNeighborhood']['vectors']], dim = 0)
                boundaryOffset += perennialState[i]['boundary']['numParticles']
            fluidOffset += perennialState[i]['fluid']['numParticles']
            offsets.append(fluidOffset)

    else:
        fluidFeatures = perennialState['fluid']['features']
        fi, fj = perennialState['fluid']['neighborhood']['indices']
        fluidEdgeLengths = distances = perennialState['fluid']['neighborhood']['distances'].view(-1,1) * perennialState['fluid']['neighborhood']['vectors']
        
        if 'boundary' in perennialState and perennialState['boundary'] is not None:
            boundaryFeatures = perennialState['boundary']['features']
            bb, bf = perennialState['fluidToBoundaryNeighborhood']['indices']
            boundaryEdgeLengths = perennialState['fluidToBoundaryNeighborhood']['distances'].view(-1,1) * perennialState['fluidToBoundaryNeighborhood']['vectors']
        
    prediction = model(fluidFeatures, fi, fj, fluidEdgeLengths, 
                boundaryFeatures, bf, bb, boundaryEdgeLengths, verbose = verbose)
    
    if isinstance(perennialState, list):
        predictions = []
        for i in range(numStates):
            predictions.append(prediction[offsets[i]:offsets[i+1]])
        return predictions
    else:
        return prediction