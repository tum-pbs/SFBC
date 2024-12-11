import copy
import torch
import torch.nn as nn
import torch.utils
import torch.utils.checkpoint


    
# for activation in getActivationFunctions():
#     print(activation, getActivationLayer(activation), getActivationFunction(activation))


# from .detail.cutlass import cutlass
from .convLayerv3 import BasisConvLayer
# from datautils import *
# from plotting import *

# Use dark theme
# from tqdm.autonotebook import trange, tqdm
# import os

from .detail.mapping import mapToSpherePreserving, mapToSpherical, process
import torch.nn as nn
from .detail.mlp import buildMLPwDict, runMLP


from .detail.basis import evalBasisFunction
@torch.jit.script
def basisEncoderLayer(edgeLengths, basisTerms : int, basisFunction : str = 'ffourier', mode : str = 'cat'):
    bTerms = []
    for e in edgeLengths.T:
        bTerm = evalBasisFunction(basisTerms, e, basisFunction).mT
        bTerms.append(bTerm)
    if mode == 'cat':
        return torch.cat(bTerms, dim = 1)
    elif mode == 'sum':
        return torch.stack(bTerms, dim = 0).sum(dim = 0)
    elif mode == 'prod':
        return torch.stack(bTerms, dim = 0).prod(dim = 0)
    elif mode == 'outer':
        return torch.einsum('ij,ik->ijk', bTerms[0], bTerms[1]).reshape(-1, basisTerms * basisTerms)
    elif mode == 'i':
        return bTerms[0]
    elif mode == 'j':
        return bTerms[1]
    elif mode == 'k':
        return bTerms[2]
    else:
        raise ValueError(f'Unknown mode: {mode}')

def applyNorm(norm, batches, features):
    transposedFeatures = features.view(batches,-1, *features.shape[1:])
    transposedFeatures = transposedFeatures.permute(0,2,1)
    normOutput = norm(transposedFeatures)
    normOutput = normOutput.permute(0,2,1)
    normOutput = normOutput.view(-1, *normOutput.shape[2:])   
    return normOutput  

def buildSkipConnection(currentFeatures, nextFeatures, skipConnectionMode, skipLayerMode, skipConnectionProperties, verbose = False, layer = 1, layerCount = -1):
        if skipConnectionMode != 'none':
            if skipLayerMode == 'mlp':
                newDict = copy.copy(skipConnectionProperties)
                newDict['inputFeatures'] = currentFeatures
                newDict['output'] = nextFeatures
                if isinstance(skipConnectionProperties['bias'], str):
                    newDict['bias'] = True if skipConnectionProperties['bias'] == 'except-last' else False
                    if skipConnectionProperties['bias'] == 'except-last' and layer == layerCount:
                        newDict['bias'] = False
                mlp = buildMLPwDict(newDict)
                cfg = newDict
                if verbose:
                    print(f'Layer[{layer}]:\tLinear: {newDict["inputFeatures"]} -> {newDict["output"]} ({sum([p.numel() for p in mlp.parameters()])} parameters)')
                skipFeatureSize = nextFeatures
            elif skipLayerMode == 'linear':
                newDict = {
                    'inputFeatures': currentFeatures,
                    'output': nextFeatures,
                    'bias': False,
                    'gain': 1,
                    'norm': False,
                    'activation': 'none',
                    'preNorm': False,
                    'postNorm': False,
                    'noLinear': False,
                    'layout': []
                }
                mlp = buildMLPwDict(newDict)
                cfg = newDict
                if verbose:
                    print(f'Layer[{layer}]:\tLinear: {newDict["inputFeatures"]} -> {newDict["output"]} ({sum([p.numel() for p in mlp.parameters()])} parameters)')
                skipFeatureSize = nextFeatures
            else:
                skipFeatureSize = currentFeatures
                mlp = None
                cfg = None
        else:
            skipFeatureSize = 0
            mlp = None
            cfg = None
        return mlp is not None, mlp, cfg, skipFeatureSize

def buildVertexMLP(previousFeatures, currentFeatures, nextFeatures, vertexMLPProperties, verbose = False, layer = 1):        
    if vertexMLPProperties is not None:
        newDict = copy.copy(vertexMLPProperties)

        vMLPinputs = currentFeatures
        if vertexMLPProperties['vertexInput']:
            vMLPinputs += previousFeatures

        newDict['inputFeatures'] = vMLPinputs
        newDict['output'] = nextFeatures
        mlp = buildMLPwDict(newDict)
        if verbose:
            print(f'Layer[{layer}]:\tVertex MLP: {newDict["inputFeatures"]} -> {newDict["output"]} ({sum([p.numel() for p in mlp.parameters()])} parameters)')
        cfg = newDict

        return True, mlp, cfg
        # currentFeatures = self.features[0]
    else:
        return False, None, None
        # if self.vertexMLPmode == 'stack':
        #     currentFeatures = self.features[0] + (self.features[0] if boundaryFeatures != 0 else 0) + skipFeatureSize
        # else:
        #     currentFeatures = self.features[0]

class GraphNetwork(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures = 0, dim = 2, layers = [32,64,64,2], 
                 
                 activation = 'relu', coordinateMapping = 'cartesian', windowFn = None, 
                 
                 batchSize = 32, outputScaling = 1/128, 

                inputEncoder = None,
                inputEdgeEncoder = None,
                outputDecoder = None,
                basisEncoder = None,

                vertexMLP = None,

                edgeMLP = None,
                edgeMode = 'none', # 'none' | 'message' | 'mlp'

                skipLayerMLP = None,
                skipLayerMode = 'none', # 'none' | 'mlp' | 'linear'
                skipConnectionMode = 'stack', # 'nonde' | 'stack' | 'add' | 'cconv'

                convLayer = {
                    'basisFunction': 'linear',
                    'basisTerms': 4,
                    'basisPeriodicity': False,
                    'cutlassBatchSize': 512,
                    'biasActive': False,
                    'linearLayerActive': False,
                    'initializer': 'uniform',
                    'optimizeWeights': False,
                    'exponentialDecay': False,
                    'mode': 'conv'
                },
                messageMLP = None,

                outputBias = True,
                centerIgnore = False,

                normalization = 'none',
                activationOnNode = True,
                verbose = False):
        super().__init__()
        self.features = copy.copy(layers)

        self.messageProcessors = torch.nn.ModuleList()
        self.skipConnections = torch.nn.ModuleList()

        self.vertexMLPs = torch.nn.ModuleList()
        self.edgeMLPs = torch.nn.ModuleList()

        self.inputEncoder = None
        self.inputEdgeEncoder = None
        self.outputDecoder = None


        self.relu = getattr(nn.functional, activation)
        self.dim = dim
        self.hasBoundaryLayers = boundaryFeatures != 0
        self.coordinateMapping = coordinateMapping
        self.windowFn = windowFn
        self.outputScaling = outputScaling
        self.centerIgnore = centerIgnore

        self.messageMLPProperties = messageMLP
        self.messageLayerProperties = convLayer

        self.inputEncoderProperties = inputEncoder
        self.inputEdgeEncoderProperties = inputEdgeEncoder
        self.basisEncoderProperties = basisEncoder
        self.outputDecoderProperties = outputDecoder

        self.vertexMLPProperties = vertexMLP
        # self.vertexMLPmode = layerMode

        self.edgeMLPProperties = edgeMLP
        self.edgeMLPmode = edgeMode

        self.skipConnectionProperties = skipLayerMLP
        self.skipLayerMode = skipLayerMode
        self.skipConnectionMode = skipConnectionMode
        self.activationOnNode = activationOnNode


        self.normalization = normalization
        self.normalizationLayers = torch.nn.ModuleList()

        if self.messageMLPProperties is not None:
            if self.messageMLPProperties['activation'] == 'default':
                self.messageMLPProperties['activation'] = activation
            self.messageLayerProperties['mlpProperties'] = self.messageMLPProperties

        if 'dim' not in self.messageLayerProperties and self.inputEncoderProperties is None:
            self.messageLayerProperties['dim'] = dim

        self.vertexMLPDicts = []
        self.edgeMLPDicts = []
        self.skipConnectionDicts = []

        edge_dimensioniality = dim

        ### ----------------------------------------------------------------------------------- ###
        ### Build Input Vertex Encoder
        if self.inputEncoderProperties is not None:
            # Default to fluid features if not set both for input and output
            if 'inputFeatures' not in self.inputEncoderProperties:
                self.inputEncoderProperties['inputFeatures'] = fluidFeatures
            if 'output' not in self.inputEncoderProperties:
                self.inputEncoderProperties['output'] = fluidFeatures

            self.inputEncoder = buildMLPwDict(self.inputEncoderProperties)
            if self.inputEncoderProperties['output'] != fluidFeatures:
                if self.inputEncoderProperties['noLinear']:
                    raise ValueError(f'Input encoder must have a linear layer if shapes change: {self.inputEncoderProperties}')
            if verbose:
                print(f'Input Encoder: {self.inputEncoderProperties["inputFeatures"]} -> {self.inputEncoderProperties["output"]} features ({sum([p.numel() for p in self.inputEncoder.parameters()])} parameters)')

        # Fourier Features
        if self.basisEncoderProperties is not None:
            terms = self.basisEncoderProperties['basisTerms']
            mode = self.basisEncoderProperties['mode']
            edge_dimensioniality = 0
            if mode == 'cat':
                edge_dimensioniality = terms * dim
            elif mode == 'sum' or mode == 'prod':
                edge_dimensioniality = terms
            elif mode == 'outer':
                edge_dimensioniality = int(terms ** dim)
            elif mode == 'i' or mode == 'j' or mode == 'k':
                edge_dimensioniality = terms
            else:
                raise ValueError(f'Unknown mode: {mode}')

                ### Build Input Edge Encoder
        if self.inputEdgeEncoderProperties is not None:
            # Default to spatial features if not set both for input and output
            self.inputEdgeEncoderProperties['inputFeatures'] = edge_dimensioniality

            if 'output' not in self.inputEdgeEncoderProperties:
                self.inputEdgeEncoderProperties['output'] = dim
            self.inputEdgeEncoder = buildMLPwDict(self.inputEdgeEncoderProperties)
            # Update the dimensionality of the convolution layer
            # if 'dim' not in self.messageLayerProperties:
            edge_dimensioniality = self.inputEdgeEncoderProperties['output']
            if verbose:
                print(f'Input Edge Encoder: {self.inputEdgeEncoderProperties["inputFeatures"]} -> {self.inputEdgeEncoderProperties["output"]} features ({sum([p.numel() for p in self.inputEdgeEncoder.parameters()])} parameters)')

        self.messageLayerProperties['dim'] = edge_dimensioniality
        self.messageLayerProperties['edgeMode'] = edgeMode

        ### ----------------------------------------------------------------------------------- ###\
        ### Build Output Decoder
        if self.outputDecoderProperties is not None:
            if 'output' not in self.outputDecoderProperties:
                self.outputDecoderProperties['output'] = self.features[-1]
            if 'inputFeatures' not in self.outputDecoderProperties:
                self.outputDecoderProperties['inputFeatures'] = self.features[-1]
                self.features[-1] = self.outputDecoderProperties['inputFeatures']
            else:
                self.features[-1] = self.outputDecoderProperties['inputFeatures']

            self.outputDecoder = buildMLPwDict(self.outputDecoderProperties)
            if verbose:
                print(f'Output Decoder: {self.outputDecoderProperties["inputFeatures"]} -> {self.outputDecoderProperties["output"]} features ({sum([p.numel() for p in self.outputDecoder.parameters()])} parameters)')


        # ### ----------------------------------------------------------------------------------- ###
        # ### Single Layer Case
        # inputFeatures = fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output']
        # if len(self.features) == 1:
        #     outputFeatures = self.features[0] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures']
        #     if verbose:
        #         print(f'Running SINGLE Convolution {inputFeatures} -> {outputFeatures} features')

        #     self.convs.append(BasisConvLayer(inputFeatures=inputFeatures, outputFeatures=outputFeatures, **self.convLayerProperties))
        #     if verbose: 
        #         print(f'Layer[0]:\tFluid Convolution: {self.convs[0].inputFeatures} -> {self.convs[0].outputFeatures} features')
        #     if boundaryFeatures != 0:
        #         self.convs.append(BasisConvLayer(inputFeatures=boundaryFeatures, outputFeatures=outputFeatures, **self.convLayerProperties))
        #         if verbose:
        #             print(f'Layer[0]:\tBoundary Convolution: {self.convs[1].inputFeatures} -> {self.convs[1].outputFeatures} features')   

        #     if self.fcLayerMLPProperties is not None:
        #         newDict = copy.copy(fcLayerMLP)
        #         newDict['inputFeatures'] = fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output']
        #         newDict['output'] = self.features[0]
        #         self.fcs.append(buildMLPwDict(newDict))    
        #         self.fcLayerMLPDicts.append(newDict)
        #         if verbose:
        #             print(f'Layer[0]:\tLinear: {newDict}')
        #     if self.vertexMLPProperties is not None:
        #         # print(f'Layer[0]:\tVertex MLP: {self.vertexMLPProperties}')

        #         vMLPinputs = self.features[0] if boundaryFeatures == 0 else 2 * self.features[0]
        #         if self.fcLayerMLPProperties is not None:
        #             vMLPinputs += self.features[0]

        #         # print(f'vMLPinputs: {vMLPinputs} ({self.features[0]} | {boundaryFeatures} | {self.fcLayerMLPProperties})')

        #         newDict = copy.copy(self.vertexMLPProperties)
        #         newDict['inputFeatures'] = vMLPinputs
        #         newDict['output'] = outputFeatures
        #         self.vertexMLPs.append(buildMLPwDict(newDict))
        #         self.vertexMLPDicts.append(newDict)
        #         if verbose:
        #             print(f'Layer[0]:\tVertex MLP: {newDict}')
        #     if self.outputDecoder is not None:
        #         if self.normalization != 'none':
        #             if self.normalization == 'layer':
        #                 self.normalizationLayers.append(nn.LayerNorm(outputFeatures))
        #             elif 'group' in self.normalization:
        #                 split = self.normalization.split('_')
        #                 self.normalizationLayers.append(nn.GroupNorm(int(split[1]), outputFeatures))
        #         # self.normalizationLayers.append()

        #     return

        ### ----------------------------------------------------------------------------------- ###
        ### Multi Layer Case

        ### First Layer
        if verbose:
            print('---------------------------------------------------------------------------------------\nBuilding first layer')
        currentFeatures = fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output']

        if self.edgeMLPProperties is not None:
            newDict = copy.copy(self.edgeMLPProperties)   
            newDict['inputFeatures'] = edge_dimensioniality        
            if newDict['output'] == -1:
                newDict['output'] = edge_dimensioniality
            else:
                edge_dimensioniality = newDict['output']
            self.edgeMLPs.append(buildMLPwDict(newDict))
            if verbose:
                print(f'Layer[{1}]:\tEdge MLP: {newDict["inputFeatures"]} -> {newDict["output"]} ({sum([p.numel() for p in self.edgeMLPs[0].parameters()])} parameters)') 
            self.edgeMLPDicts.append(newDict)
        self.messageLayerProperties['dim'] = edge_dimensioniality
            
        self.messageProcessors.append(BasisConvLayer(inputFeatures=currentFeatures, outputFeatures= self.features[0], **self.messageLayerProperties))
        if verbose: 
            print(f'Layer[{1}]:\tFluid Convolution: {self.messageProcessors[0].inputFeatures} -> {self.messageProcessors[0].outputFeatures} features ({sum([p.numel() for p in self.messageProcessors[0].parameters()])} parameters) [dim = {edge_dimensioniality}]')
        if boundaryFeatures != 0:
            self.messageProcessors.append(BasisConvLayer(inputFeatures=boundaryFeatures, outputFeatures= self.features[0],**self.messageLayerProperties ))
            if verbose:
                print(f'Layer[{1}]:\tBoundary Convolution: {self.messageProcessors[1].inputFeatures} -> {self.messageProcessors[1].outputFeatures} features')

        if self.edgeMLPmode == 'message':
            edge_dimensioniality = self.features[0]
            if self.edgeMLPProperties is not None:
                raise ValueError(f'Edge MLPs are not supported in message mode')

        skipLayerActive, skipMLP, skipCfg, skipFeatureSize = buildSkipConnection(currentFeatures, self.features[0], self.skipConnectionMode, self.skipLayerMode, self.skipConnectionProperties, verbose = verbose, layer = 1)
        if self.skipConnectionMode == 'add' or not skipLayerActive:
            skipFeatureSize = 0
        if skipLayerActive:
            self.skipConnections.append(skipMLP)
            self.skipConnectionDicts.append(skipCfg)

        vMLPActive, vMLP, vMLPCfg = buildVertexMLP(currentFeatures, self.features[0] * (1 if boundaryFeatures == 0 else 2), self.features[0], self.vertexMLPProperties, verbose = verbose, layer = 1)
        if vMLPActive:
            self.vertexMLPs.append(vMLP)
            self.vertexMLPDicts.append(vMLPCfg)
                

        if self.normalization != 'none':    
            # inputFeatures = currentFeatures
            if self.normalization == 'layer':
                self.normalizationLayers.append(nn.LayerNorm(self.features[0] + skipFeatureSize))
            elif 'group' in self.normalization:
                split = self.normalization.split('_')
                self.normalizationLayers.append(nn.GroupNorm(int(split[1]), self.features[0] + skipFeatureSize))
        if verbose:
            print('---------------------------------------------------------------------------------------\nBuilding middle layers')

        ### Middle Layers
        for i, l in enumerate(self.features[1:-1]):
            if verbose:
                print(f'Layer[{i+2}]:\t{self.features[i]} -> {self.features[i+1]} features')
            if self.skipConnectionMode == 'stack' or (self.skipConnectionMode == 'cconv' and i == 0):
                currentFeatures = self.features[i] + (self.features[0] if boundaryFeatures != 0 and i == 0 else 0) + skipFeatureSize
            else:
                currentFeatures = self.features[i] + (self.features[0] if boundaryFeatures != 0 and i == 0 else 0)

            if verbose:
                print(f'Layer[{i+2}]: {currentFeatures} -> {self.features[i+1]} features')
            if self.edgeMLPProperties is not None:
                newDict = copy.copy(self.edgeMLPProperties)   
                newDict['inputFeatures'] = edge_dimensioniality
                self.edgeMLPs.append(buildMLPwDict(newDict))
                if verbose:
                    print(f'Layer[{i+2}]:\tEdge MLP: {newDict["inputFeatures"]} -> {newDict["output"]} ({sum([p.numel() for p in self.edgeMLPs[i].parameters()])} parameters)')
                self.edgeMLPDicts.append(newDict)
            ### Convolution
            self.messageLayerProperties['dim'] = edge_dimensioniality# dim if self.edgeMLPProperties is None else self.edgeMLPProperties['output']
            self.messageProcessors.append(BasisConvLayer(
                inputFeatures = currentFeatures, 
                outputFeatures = self.features[i+1],
                **self.messageLayerProperties))
            if verbose:
                print(f'Layer[{i+2}]:\tFluid Convolution: {self.messageProcessors[i+1].inputFeatures} -> {self.messageProcessors[i+1].outputFeatures} features ({sum([p.numel() for p in self.messageProcessors[i+1].parameters()])} parameters)')
            if self.edgeMLPmode == 'message':
                edge_dimensioniality = self.features[i+1]
            ### Fully Connected Layer

            skipLayerActive, skipMLP, skipCfg, skipFeatureSize = buildSkipConnection(currentFeatures, self.features[i+1], self.skipConnectionMode, self.skipLayerMode, self.skipConnectionProperties, verbose = verbose, layer = i+2, layerCount = len(self.features))
            if skipLayerActive:
                self.skipConnections.append(skipMLP)
                self.skipConnectionDicts.append(skipCfg)

            ### Vertex MLP
            vMLPActive, vMLP, vMLPCfg = buildVertexMLP(currentFeatures, self.features[i+1], self.features[i+1], self.vertexMLPProperties, verbose = verbose, layer = i+2)
            if vMLPActive:
                self.vertexMLPs.append(vMLP)
                self.vertexMLPDicts.append(vMLPCfg)
                
            if self.normalization != 'none':
                outputFeatures = self.features[i+1] if self.vertexMLPProperties is None else self.vertexMLPDicts[-1]['output']
                if self.normalization == 'layer':
                    self.normalizationLayers.append(nn.LayerNorm(outputFeatures + skipFeatureSize))
                elif 'group' in self.normalization:
                    split = self.normalization.split('_')
                    self.normalizationLayers.append(nn.GroupNorm(int(split[1]), outputFeatures + skipFeatureSize))
            

        if self.edgeMLPProperties is not None:
            newDict = copy.copy(self.edgeMLPProperties)   
            newDict['inputFeatures'] = edge_dimensioniality
            self.edgeMLPs.append(buildMLPwDict(newDict))
            if verbose:
                print(f'Layer[{len(layers)}]:\tEdge MLP: {newDict["inputFeatures"]} -> {newDict["output"]} ({sum([p.numel() for p in self.edgeMLPs[-1].parameters()])} parameters)')
            self.edgeMLPDicts.append(newDict)

        ### Last Layer        
        #   
        if len(layers) <= 2:
            if self.skipConnectionMode == 'stack' or self.skipConnectionMode == 'cconv' :
                currentFeatures = self.features[0] + (self.features[0] if boundaryFeatures != 0 else 0) + skipFeatureSize
            else:
                currentFeatures = self.features[0] + (self.features[0] if boundaryFeatures != 0 else 0)
        else:
            if self.skipConnectionMode == 'stack':
                currentFeatures = self.features[-2] + (self.features[0] if boundaryFeatures != 0 else 0) + skipFeatureSize
            else:
                currentFeatures = self.features[-2] + (self.features[0] if boundaryFeatures != 0 else 0)
            
        outputFeatures = self.features[-1] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures']
        if verbose:
            print(f'Layer[{len(layers)}]: {currentFeatures} -> {outputFeatures} features')
        ### Convolution
        self.messageProcessors.append(BasisConvLayer(inputFeatures = currentFeatures, outputFeatures = outputFeatures, **self.messageLayerProperties))
        if verbose:
            print(f'Layer[{len(layers)}]:\tFluid Convolution: {self.messageProcessors[-1].inputFeatures} -> {self.messageProcessors[-1].outputFeatures} features ({sum([p.numel() for p in self.messageProcessors[-1].parameters()])} parameters)')
        ### Fully Connected Layer
        skipLayerActive, skipMLP, skipCfg, skipFeatureSize = buildSkipConnection(currentFeatures, outputFeatures, self.skipConnectionMode, self.skipLayerMode, self.skipConnectionProperties, verbose = verbose, layer = len(layers), layerCount = len(self.features))
        if skipLayerActive:
            self.skipConnections.append(skipMLP)
            self.skipConnectionDicts.append(skipCfg)
        ### Vertex MLP
        vMLPActive, vMLP, vMLPCfg = buildVertexMLP(currentFeatures, outputFeatures, outputFeatures, self.vertexMLPProperties, verbose = verbose, layer = len(layers))
        if vMLPActive:
            self.vertexMLPs.append(vMLP)
            self.vertexMLPDicts.append(vMLPCfg)

        if self.normalization != 'none' and self.outputDecoder is not None:
            outputFeatures = self.features[-1] if self.vertexMLPProperties is None else self.vertexMLPDicts[-1]['output']
            if self.normalization == 'layer':
                self.normalizationLayers.append(nn.LayerNorm(outputFeatures + skipFeatureSize))
            elif 'group' in self.normalization:
                split = self.normalization.split('_')
                self.normalizationLayers.append(nn.GroupNorm(int(split[1]), outputFeatures + skipFeatureSize))

    def runEdgeMLP(self, edgeMLP, edgeMLPDict, edgeLengths, batches, numEdges, verbose = False, layer = 0, clamp = True, checkpoint = True):
        if edgeMLP is None:
            return edgeLengths
        if verbose:
            print(f'Layer[{layer}]:\tRunning Edge MLP {edgeMLPDict["inputFeatures"]} -> {edgeMLPDict["output"]} features')

        newEdgeLengths = []
        for b in range(batches):
            transposedEdges = edgeLengths[numEdges[b]:numEdges[b+1]].view(1,-1, *edgeLengths.shape[1:])
            if checkpoint:
                processedEdges = torch.utils.checkpoint.checkpoint(edgeMLP, transposedEdges, use_reentrant=False)
            else:
                processedEdges = edgeMLP(transposedEdges)
            if clamp:
                processedEdges = processedEdges.clamp(-1,1)
            newEdgeLengths.append(processedEdges.view(-1, *processedEdges.shape[2:]))
        result = torch.cat(newEdgeLengths, dim = 0)
        return result

    def forward(self, \
                fluidFeatures, \
                fluid_edge_index_i, fluid_edge_index_j, distances, boundaryFeatures = None, bf = None, bb = None, boundaryDistances = None, batches = 1, verbose = True):
        
        if verbose:
            print(f'---------------------------------------------------')
        ni, i, fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights = process(
            fluid_edge_index_i, fluid_edge_index_j, distances, self.centerIgnore, self.coordinateMapping, self.windowFn)
        
        numEdges = [0] + ni.view(batches, -1).sum(dim = 1).detach().cpu().numpy().tolist()
        if verbose:
            print(f'ni: {ni}, i: {i}, fluidEdgeIndex: {fluidEdgeIndex.shape}, fluidEdgeLengths: {fluidEdgeLengths.shape}, fluidEdgeWeights: {fluidEdgeWeights.shape}')
            print(f'numEdges: {numEdges}')

        if self.hasBoundaryLayers:
            nb, b, boundaryEdgeIndex, boundaryEdgeLengths, boundaryEdgeWeights = process(bf, bb, boundaryDistances, False, self.coordinateMapping, self.windowFn)
            self.nb = nb
            ni[i[b]] += nb
            if verbose:
                print(f'ni: {ni}, i: {i}, fluidEdgeIndex: {fluidEdgeIndex.shape}, fluidEdgeLengths: {fluidEdgeLengths.shape}, fluidEdgeWeights: {fluidEdgeWeights.shape}')
                print(f'nb: {nb}, b: {b}, boundaryEdgeIndex: {boundaryEdgeIndex.shape}, boundaryEdgeLengths: {boundaryEdgeLengths.shape}, boundaryEdgeWeights: {boundaryEdgeWeights.shape}')
        else:
            if verbose:
                print(f'ni: {ni}, i: {i}, fluidEdgeIndex: {fluidEdgeIndex.shape}, fluidEdgeLengths: {fluidEdgeLengths.shape}, fluidEdgeWeights: {fluidEdgeWeights.shape}')
        self.li = torch.exp(-1 / 16 * ni)
        # if len(self.rbfs) > 2:
            # self.li = torch.exp(-1 / 50 * ni)
        
        if self.inputEncoder is not None:
            if verbose:
                print(f'(pre encoder) fluidFeatures: {fluidFeatures.shape} -> {self.inputEncoderProperties["output"]} features')
            fluidFeatures = runMLP(self.inputEncoder, fluidFeatures, batches, verbose = False)
            if verbose:
                print(f'(post encoder) fluidFeatures: {fluidFeatures.shape}')
                
        if self.basisEncoderProperties is not None:
            if verbose:
                print(f'(pre basis encoder) fluidEdgeLengths: {fluidEdgeLengths.shape}')

            fluidEdgeLengths = basisEncoderLayer(fluidEdgeLengths, self.basisEncoderProperties['basisTerms'], self.basisEncoderProperties['basisFunction'], self.basisEncoderProperties['mode'])

            if verbose:
                print(f'(post basis encoder) fluidEdgeLengths: {fluidEdgeLengths.shape}')

        if self.inputEdgeEncoder is not None:
            if verbose:
                print(f'(pre edge encoder) fluidEdgeLengths: {fluidEdgeLengths.shape}')

            fluidEdgeLengths = self.runEdgeMLP(self.inputEdgeEncoder, self.inputEdgeEncoderProperties, fluidEdgeLengths, batches, numEdges, verbose = False, layer = 0, clamp = True)

            # newEdgeLengths = []
            # for b in range(batches):
            #     transposedEdges = fluidEdgeLengths[numEdges[b]:numEdges[b+1]].view(1,-1, *fluidEdgeLengths.shape[1:])
            #     processedEdges = self.inputEdgeEncoder(transposedEdges)
            #     processedEdges = processedEdges.clamp(-1,1)
            #     newEdgeLengths.append(processedEdges.view(-1, *processedEdges.shape[2:]))
            # fluidEdgeLengths = torch.cat(newEdgeLengths, dim = 0)

            if verbose:
                print(f'(post edge encoder) fluidEdgeLengths: {fluidEdgeLengths.shape}')

        if self.edgeMLPProperties is not None:
            fluidEdgeLengths = self.runEdgeMLP(self.edgeMLPs[0], self.edgeMLPDicts[0], fluidEdgeLengths, batches, numEdges, verbose = verbose, layer = 0, clamp = False)
            

        if verbose:
            print(f'Layer[0]:\tConvolution (FTF): {self.messageProcessors[0].inputFeatures} -> {self.messageProcessors[0].outputFeatures} features [edge_dim = {self.messageProcessors[0].dim}]')
        fluidConvolution, fluidMessages = (self.messageProcessors[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights, batches=  batches, verbose  = False))
        if self.hasBoundaryLayers:
            if verbose:
                print(f'Layer[0]:\tConvolution (BTF) {self.messageProcessors[1].inputFeatures} -> {self.messageProcessors[1].outputFeatures} features')
            boundaryConvolution, _ = (self.messageProcessors[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths, boundaryEdgeWeights, batches=  batches))
        else:
            boundaryConvolution = None

        

        # ### Single Layer Case
        # if len(self.features) == 1:                
        #     if self.hasBoundaryLayers:
        #         fluidConvolution += boundaryConvolution

        #     if self.fcLayerMLPProperties is not None:
        #         if self.vertexMLPProperties is not None:
        #             fluidConvolution = torch.hstack((linearOutput, fluidConvolution))
        #         else:
        #             fluidConvolution = linearOutput + fluidConvolution

        #     if self.vertexMLPProperties is not None:
        #         if verbose:
        #             print(f'Layer[0]:\tVertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features')
        #             fluidConvolution = runMLP(self.vertexMLPs[0], fluidConvolution, batches, verbose = False)

        #     if self.normalization != 'none':
        #         if verbose:
        #             print(f'Layer[0]:\tApplying Normalization {fluidConvolution.shape}')
        #         fluidConvolution = applyNorm(self.normalizationLayers[0], batches, fluidConvolution)

        #     if self.outputDecoder is not None:
        #         # if verbose:
        #             # print(f'(pre outputDecoder) fluidConvolution: {fluidConvolution.shape}')
        #         if verbose:
        #             print(f'Layer[0]:\tOutput Decoder {self.outputDecoderProperties["inputFeatures"]} -> {self.outputDecoderProperties["output"]} features')
        #         fluidConvolution = runMLP(self.outputDecoder, fluidConvolution, batches, verbose = False)          
        #     if verbose:
        #         print(f'Final: {fluidConvolution.shape} [min: {torch.min(fluidConvolution)}, max: {torch.max(fluidConvolution)}, mean: {torch.mean(fluidConvolution)}]')  
        #     return fluidConvolution 
        ### Multi Layer Case
        # if verbose:
        #     print(f'Layer[0]:\tStacking Features: {linearOutput.shape if linearOutput is not None else 0} | {fluidConvolution.shape} | {boundaryConvolution.shape if self.hasBoundaryLayers else 0}')
        # if self.hasBoundaryLayers:
        #     if self.fcLayerMLPProperties is not None:
        #         ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
        #     else:
        #         ans = torch.hstack((fluidConvolution, boundaryConvolution))
        # else:
        #     if self.fcLayerMLPProperties is not None:
        #         ans = torch.hstack((linearOutput, fluidConvolution))
        #     else:
        #         ans = fluidConvolution
        # print(self.edgeMLPmode)
        if self.edgeMLPmode == 'message':
            fluidEdgeLengths = fluidMessages
            
        convolutions = torch.hstack((fluidConvolution, boundaryConvolution)) if self.hasBoundaryLayers else fluidConvolution
        if self.vertexMLPProperties is not None:
            ans = torch.hstack((convolutions, fluidFeatures)) if self.vertexMLPProperties['vertexInput'] else convolutions
        else:
            ans = convolutions

        if verbose:
            print(f'Layer[0]: Convolution Shape: {convolutions.shape} | ans Shape: {ans.shape}')


        if verbose:
            print(f'Pre-Message Passing Done: {ans.shape}\n')
        
        # if self.edgeMLPProperties is not None:
            # fluidEdgeLengths = self.runEdgeMLP(self.edgeMLPs[0], self.edgeMLPDicts[0], fluidEdgeLengths, batches, numEdges, verbose = verbose, layer = 0, clamp = False)
        # print(self.vertexMLP)
        if self.vertexMLPProperties is not None:
            # print(f'Running Vertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features {ans.shape}')
            if verbose:
                print(f'Layer[0]:\tRunning Vertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features\n')
            transposedFeatures = ans.view(batches,-1, *ans.shape[1:])
            ans = torch.utils.checkpoint.checkpoint(self.vertexMLPs[0], transposedFeatures, use_reentrant = False)
            # ans = self.vertexMLPs[0](transposedFeatures)
            ans = ans.view(-1, *ans.shape[2:])

        skipLayerOutput = None
        if self.skipConnectionMode != 'none':
            if verbose:
                print(f'Layer[0]:\tLinear {self.skipConnectionDicts[0]["inputFeatures"]} -> {self.skipConnectionDicts[0]["output"]} features')
            skipLayerOutput = runMLP(self.skipConnections[0], fluidFeatures, batches, verbose = False)
            # print(ans.shape, skipLayerOutput.shape)
            if self.skipConnectionMode == 'stack' or self.skipConnectionMode == 'cconv':
                ans = torch.hstack((ans, skipLayerOutput))
            else:
                ans = ans + skipLayerOutput
        
        if self.normalization != 'none':
            if verbose:
                print(f'Layer[0]:\tApplying Normalization {ans.shape}')
            fluidConvolution = applyNorm(self.normalizationLayers[0], batches, ans)

        layers = len(self.messageProcessors)
        for i in range(1 if not self.hasBoundaryLayers else 2,layers):
            if verbose:
                print(f'Layer[{i}]:\tInput {ans.shape}')
            # print(f'Layer[{i}]:\tRelu: {ans.shape}')
            if self.activationOnNode:
                ansc = self.relu(ans)
            else:
                ansc = ans
                
            if self.edgeMLPProperties is not None:
                fluidEdgeLengths = self.runEdgeMLP(self.edgeMLPs[i], self.edgeMLPDicts[i], fluidEdgeLengths, batches, numEdges, verbose = verbose, layer = i, clamp = False)
                # if verbose:
                #     print(f'Layer[{i}]:\tRunning Edge MLP {self.edgeMLPDicts[i]["inputFeatures"]} -> {self.edgeMLPDicts[i]["output"]} features')
                # newEdgeLengths = []
                # for b in range(batches):
                #     transposedEdges = fluidEdgeLengths[numEdges[b]:numEdges[b+1]].view(1,-1, *fluidEdgeLengths.shape[1:])

                #     # processedEdges = self.edgeMLPs[i](transposedEdges)
                #     processedEdges = torch.utils.checkpoint.checkpoint(self.edgeMLPs[i], transposedEdges, use_reentrant = False)

                #     processedEdges = processedEdges.clamp(-1,1)
                #     newEdgeLengths.append(processedEdges.view(-1, *processedEdges.shape[2:]))
                # fluidEdgeLengths = torch.cat(newEdgeLengths, dim = 0)

                # fluidEdgeLengths = self.edgeMLPs[i](fluidEdgeLengths)
                # fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
                
            if verbose:
                # print(f'Layer[{i}]:\tResult for layer {i-1} [min: {torch.min(ansc)}, max: {torch.max(ansc)}, mean: {torch.mean(ansc)}] | [min: {torch.min(ans)}, max: {torch.max(ans)}, mean: {torch.mean(ans)}]')
                print(f'Layer[{i}]:\tRunning Convolution {self.messageProcessors[i].inputFeatures} -> {self.messageProcessors[i].outputFeatures} features [edge_dim = {self.messageProcessors[i].dim}]')
            # print(f'Layer[{i}]:\tConvolution: {self.convs[i].inputFeatures} -> {self.convs[i].outputFeatures} features')
            ansConv, ansMessages = self.messageProcessors[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights)
            
            if self.edgeMLPmode == 'message':
                fluidEdgeLengths = ansMessages
            if self.vertexMLPProperties is not None:
                ans = torch.hstack((ansConv, ansc)) if self.vertexMLPProperties['vertexInput'] else ansConv
            else:
                ans = ansConv

            if self.vertexMLPProperties is not None:
                # print(f'Running Vertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features {ans.shape}')
                if verbose:
                    print(f'Layer[0]:\tRunning Vertex MLP {self.vertexMLPDicts[i]["inputFeatures"]} -> {self.vertexMLPDicts[i]["output"]} features\n')
                ans = runMLP(self.vertexMLPs[i], ans, batches, verbose = False)                
                
            skipLayerOutput = None
            if self.skipConnectionMode != 'none':
                if verbose:
                    print(f'Layer[{i}]:\tLinear {self.skipConnectionDicts[i]["inputFeatures"]} -> {self.skipConnectionDicts[i]["output"]} features')
                skipLayerOutput = runMLP(self.skipConnections[i], ansc, batches, verbose = False)
                if self.skipConnectionMode == 'stack':
                    ans = torch.hstack((ans, skipLayerOutput))
                else:
                    ans = ans + skipLayerOutput
            else:
                skipLayerOutput = None


            if self.normalization != 'none':
                if verbose:
                    print(f'Layer[{i}]:\tApplying Normalization {ans.shape}')
                ans = applyNorm(self.normalizationLayers[i], batches, ans)
            if verbose:
                print(f'\n')

        if verbose:
            print(f'Done With Message Passing: {ans.shape}')
        if self.outputDecoder is not None:
            if verbose:
                print(f'Running Output Decoder {self.outputDecoderProperties["inputFeatures"]} -> {self.outputDecoderProperties["output"]} features')
            ans = self.outputDecoder(ans.view(batches,-1, *ans.shape[1:]))
            ans = ans.view(-1, *ans.shape[2:])
        if verbose:
            print(f'Final: {ans.shape} [min: {torch.min(ans)}, max: {torch.max(ans)}, mean: {torch.mean(ans)}]')
        if verbose:
            print(f'---------------------------------------------------')
        return ans * self.outputScaling #(ans * outputScaling) if self.dim == 2 else ans
    