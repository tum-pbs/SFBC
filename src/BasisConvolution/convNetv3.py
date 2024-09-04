import copy
import torch
import torch.nn as nn


    
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
from .detail.mlp import buildMLPwDict

def runMLP(mlp, features, batches, verbose = False):       
    if verbose:
        print(f'MLP {features.shape} -> {mlp[-1].out_features} features')
    transposedFeatures = features.view(batches,-1, *features.shape[1:])
    processedFeatures = mlp(transposedFeatures)
    # print(f'(post encoder) fluidFeatures: {fluidFeatures.shape}')
    processedFeatures = processedFeatures.view(-1, *processedFeatures.shape[2:])
    if verbose:
        print(f'\tFeatures: {processedFeatures.shape} [min: {torch.min(processedFeatures)}, max: {torch.max(processedFeatures)}, mean: {torch.mean(processedFeatures)}]')
    return processedFeatures


class GraphNetwork(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures = 0, dim = 2, layers = [32,64,64,2], activation = 'relu',
                coordinateMapping = 'cartesian', windowFn = None, batchSize = 32, outputScaling = 1/128, 
                vertexMLP = None,
                edgeMLP = None,
                inputEncoder = None,
                inputEdgeEncoder = None,
                outputDecoder = None,
                fcLayerMLP = None,
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
                outputBias = True,
                centerIgnore = False,
                verbose = False):
        super().__init__()
        self.features = copy.copy(layers)

        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
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

        self.inputEncoderProperties = inputEncoder
        self.inputEdgeEncoderProperties = inputEdgeEncoder
        self.outputDecoderProperties = outputDecoder
        self.vertexMLPProperties = vertexMLP
        self.edgeMLPProperties = edgeMLP
        self.fcLayerMLPProperties = fcLayerMLP
        self.convLayerProperties = convLayer
        if 'dim' not in self.convLayerProperties and self.inputEncoderProperties is None:
            self.convLayerProperties['dim'] = dim

        self.vertexMLPDicts = []
        self.edgeMLPDicts = []
        self.fcLayerMLPDicts = []
        self.fcLayerMLP = fcLayerMLP
        self.agglomerateViaMLP = False

        ### ----------------------------------------------------------------------------------- ###
        ### Build Input Vertex Encoder
        if self.inputEncoderProperties is not None:
            if 'inputFeatures' not in self.inputEncoderProperties:
                self.inputEncoderProperties['inputFeatures'] = fluidFeatures
            if 'output' not in self.inputEncoderProperties:
                self.inputEncoderProperties['output'] = fluidFeatures
            self.inputEncoder = buildMLPwDict(self.inputEncoderProperties)
            if self.inputEncoderProperties['output'] != fluidFeatures:
                if self.inputEncoderProperties['noLinear']:
                    raise ValueError(f'Input encoder must have a linear layer if shapes change: {self.inputEncoderProperties}')
            if verbose:
                print(f'Input Encoder: {self.inputEncoderProperties["inputFeatures"]} -> {self.inputEncoderProperties["output"]} features')

        ### Build Input Edge Encoder
        if self.inputEdgeEncoderProperties is not None:
            if 'inputFeatures' not in self.inputEdgeEncoderProperties:
                self.inputEdgeEncoderProperties['inputFeatures'] = dim
            if 'output' not in self.inputEdgeEncoderProperties:
                self.inputEdgeEncoderProperties['output'] = dim
            self.inputEdgeEncoder = buildMLPwDict(self.inputEdgeEncoderProperties)
            if 'dim' not in self.convLayerProperties:
                self.convLayerProperties['dim'] = self.inputEdgeEncoderProperties['output']
            if verbose:
                print(f'Input Edge Encoder: {self.inputEdgeEncoderProperties["inputFeatures"]} -> {self.inputEdgeEncoderProperties["output"]} features')

        ### ----------------------------------------------------------------------------------- ###\
        ### Build Output Decoder
        if self.outputDecoderProperties is not None:
            if 'output' not in self.outputDecoderProperties:
                self.outputDecoderProperties['output'] = self.features[-1]
            if 'inputFeatures' not in self.outputDecoderProperties:
                self.outputDecoderProperties['inputFeatures'] = self.features[-1]
                if self.fcLayerMLPProperties is not None and self.vertexMLPProperties is None:
                    self.outputDecoderProperties['inputFeatures'] += self.features[-1]

                self.features[-1] = self.outputDecoderProperties['inputFeatures']
            else:
                self.features[-1] = self.outputDecoderProperties['inputFeatures']

            self.outputDecoder = buildMLPwDict(self.outputDecoderProperties)
            if verbose:
                print(f'Output Decoder: {self.outputDecoderProperties["inputFeatures"]} -> {self.outputDecoderProperties["output"]} features')


        ### ----------------------------------------------------------------------------------- ###
        ### Single Layer Case
        inputFeatures = fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output']
        if len(self.features) == 1:
            outputFeatures = self.features[0] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures']
            if verbose:
                print(f'Running SINGLE Convolution {inputFeatures} -> {outputFeatures} features')

            self.convs.append(BasisConvLayer(inputFeatures=inputFeatures, outputFeatures=outputFeatures, **self.convLayerProperties))
            if verbose: 
                print(f'Layer[0]:\tFluid Convolution: {self.convs[0].inputFeatures} -> {self.convs[0].outputFeatures} features')
            if boundaryFeatures != 0:
                self.convs.append(BasisConvLayer(inputFeatures=boundaryFeatures, outputFeatures=outputFeatures, **self.convLayerProperties))
                if verbose:
                    print(f'Layer[0]:\tBoundary Convolution: {self.convs[1].inputFeatures} -> {self.convs[1].outputFeatures} features')   

            if self.fcLayerMLPProperties is not None:
                newDict = copy.copy(fcLayerMLP)
                newDict['inputFeatures'] = fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output']
                newDict['output'] = self.features[0]
                self.fcs.append(buildMLPwDict(newDict))    
                self.fcLayerMLPDicts.append(newDict)
                if verbose:
                    print(f'Layer[0]:\tLinear: {newDict}')
            if self.vertexMLPProperties is not None:
                newDict = copy.copy(self.vertexMLPProperties)
                newDict['inputFeatures'] = 2 * self.features[0] if boundaryFeatures != 0 else 1 * self.features[0]
                newDict['inputFeatures'] += self.features[0] if self.fcLayerMLPProperties is not None else 0
                newDict['output'] = outputFeatures
                self.vertexMLPs.append(buildMLPwDict(newDict))
                self.vertexMLPDicts.append(newDict)
                if verbose:
                    print(f'Layer[0]:\tVertex MLP: {newDict}')
            return

        ### ----------------------------------------------------------------------------------- ###
        ### Multi Layer Case

        ### First Layer
        self.convs.append(BasisConvLayer(inputFeatures=inputFeatures, outputFeatures= self.features[0], **self.convLayerProperties))
        if verbose: 
            print(f'Layer[{1}]:\tFluid Convolution: {self.convs[0].inputFeatures} -> {self.convs[0].outputFeatures} features')
        if boundaryFeatures != 0:
            self.convs.append(BasisConvLayer(inputFeatures=boundaryFeatures, outputFeatures= self.features[0],**self.convLayerProperties ))
            if verbose:
                print(f'Layer[{1}]:\tBoundary Convolution: {self.convs[1].inputFeatures} -> {self.convs[1].outputFeatures} features')

        if self.fcLayerMLPProperties is not None:
            newDict = copy.copy(fcLayerMLP)
            newDict['inputFeatures'] = fluidFeatures if self.inputEncoder is None else self.inputEncoderProperties['output']
            newDict['output'] = self.features[0]
            self.fcLayerMLPDicts.append(newDict)
            self.fcs.append(buildMLPwDict(newDict))    
            if verbose:
                print(f'Layer[{1}]:\tLinear: {newDict}')
        if self.vertexMLPProperties is not None:
            newDict = copy.copy(self.vertexMLPProperties)
            newDict['inputFeatures'] = 3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]
            newDict['output'] = newDict['inputFeatures'] if 'outputFeatures' not in newDict else newDict['outputFeatures']
            if verbose:
                print(f'Layer[{1}]:\tVertex MLP: {newDict}')
            self.vertexMLPs.append(buildMLPwDict(newDict))
            self.vertexMLPDicts.append(newDict)
        if self.edgeMLPProperties is not None:
            newDict = copy.copy(self.edgeMLPProperties)   
            if verbose:
                print(f'Layer[{1}]:\tEdge MLP: {newDict}')         
            self.edgeMLPs.append(buildMLPwDict(newDict))
            self.edgeMLPDicts.append(newDict)

        ### Middle Layers
        for i, l in enumerate(self.features[1:-1]):
            # if verbose:
                # print(f'Layer[{i+2}]:\t{self.features[i]} -> {self.features[i+1]} features')
            inputFeatures = (2 * self.features[0] if boundaryFeatures != 0 else 1 * self.features[0]) if i == 0 else self.features[i]
            if i == 0 and self.fcLayerMLPProperties is not None:
                inputFeatures += self.features[0]

            if self.vertexMLPProperties is not None:
                inputFeatures = newDict['output']
            if verbose:
                print(f'Layer[{i+2}]:\t{inputFeatures} -> {self.features[i+1]} features')
            ### Convolution
            self.convs.append(BasisConvLayer(
                inputFeatures = inputFeatures, 
                outputFeatures = self.features[i+1],
                **self.convLayerProperties))
            if verbose:
                print(f'Layer[{i+2}]:\tFluid Convolution: {self.convs[i+1].inputFeatures} -> {self.convs[i+1].outputFeatures} features')
            ### Fully Connected Layer
            if self.fcLayerMLPProperties is not None:
                newDict = copy.copy(fcLayerMLP)
                newDict['inputFeatures'] = (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]) if i == 0 else self.features[i]
                newDict['output'] =self.features[i+1]
                if verbose:
                    print(f'Layer[{i+2}]:\tLinear: {newDict}')
                self.fcs.append(buildMLPwDict(newDict))
                self.fcLayerMLPDicts.append(newDict)
            ### Vertex MLP
            if self.vertexMLPProperties is not None:
                newDict = copy.copy(self.vertexMLPProperties)
                agglomerateViaMLP = 'agglomerate' in newDict and newDict['agglomerate']
                self.agglomerateViaMLP = agglomerateViaMLP
                newDict['inputFeatures'] = self.features[i+1]
                newDict['output'] = self.features[i+1] 
                newDict['inputFeatures'] = self.features[i+1] + (self.features[i+1] + (self.features[i+1] if self.features[i+0] == self.features[i+1] and i > 0 else 0) if agglomerateViaMLP else 0) 
                if verbose:
                    print(f'Layer[{i+2}]:\tVertex MLP: {newDict}')
                self.vertexMLPs.append(buildMLPwDict(newDict))
                self.vertexMLPDicts.append(newDict)
            if self.edgeMLPProperties is not None:
                if verbose:
                    print(f'Layer[{i+2}]:\tEdge MLP: {self.edgeMLPProperties}')
                self.edgeMLPs.append(buildMLPwDict(self.edgeMLPProperties))
                self.edgeMLPDicts.append(self.edgeMLPProperties)


        ### Last Layer            
        inputFeatures = self.features[-2] if len(layers) > 2 else (2 * self.features[0] if boundaryFeatures != 0 else 1 * self.features[0]) + (self.features[0] if self.fcLayerMLPProperties is not None else 0)
        outputFeatures = self.features[-1] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures']
        if verbose:
            print(f'Layer[-1]:\t{inputFeatures} -> {outputFeatures} features')
        ### Convolution
        self.convs.append(BasisConvLayer(inputFeatures = inputFeatures, outputFeatures = outputFeatures, **self.convLayerProperties))
        if verbose:
            print(f'Layer[-1]:\tFluid Convolution: {self.convs[-1].inputFeatures} -> {self.convs[-1].outputFeatures} features')
        ### Fully Connected Layer
        if self.fcLayerMLPProperties is not None:
            newDict = copy.copy(fcLayerMLP)
            newDict['inputFeatures'] = self.features[-2] if len(layers) > 2 else (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0])
            newDict['output'] = self.features[-1] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures']
            newDict['bias'] = outputBias
            if verbose:
                print(f'Layer[-1]:\tLinear: {newDict}')
            self.fcs.append(buildMLPwDict(newDict))
            self.fcLayerMLPDicts.append(newDict)
        ### Vertex MLP
        if self.vertexMLPProperties is not None:
            newDict = copy.copy(self.vertexMLPProperties)
            inFeat = self.features[-1] if self.outputDecoder is None else self.outputDecoderProperties['inputFeatures']
            newDict['inputFeatures'] = inFeat
            newDict['output'] = inFeat 
            newDict['inputFeatures'] = inFeat + (inFeat + (self.features[i+2] if self.features[i+1] == self.features[i+2] and i > 0 else 0)if agglomerateViaMLP else 0) 
            if verbose:
                print(f'Layer[-1]:\tVertex MLP: {newDict}')
            self.vertexMLPs.append(buildMLPwDict(newDict))
            self.vertexMLPDicts.append(newDict)



    def forward(self, \
                fluidFeatures, \
                fluid_edge_index_i, fluid_edge_index_j, distances, boundaryFeatures = None, bf = None, bb = None, boundaryDistances = None, batches = 1, verbose = True):
        
        if verbose:
            print(f'---------------------------------------------------')
        ni, i, fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights = process(
            fluid_edge_index_i, fluid_edge_index_j, distances, self.centerIgnore, self.coordinateMapping, self.windowFn)
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
        if verbose:
            print(f'Layer[0]:\tConvolution (FTF): {self.convs[0].inputFeatures} -> {self.convs[0].outputFeatures} features')
        fluidConvolution = (self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights, batches=  batches, verbose  = False))
        if self.hasBoundaryLayers:
            if verbose:
                print(f'Layer[0]:\tConvolution (BTF) {self.convs[1].inputFeatures} -> {self.convs[1].outputFeatures} features')
            boundaryConvolution = (self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths, boundaryEdgeWeights, batches=  batches))
        if self.fcLayerMLPProperties is not None:
            if verbose:
                print(f'Layer[0]:\tLinear {self.fcLayerMLPDicts[0]["inputFeatures"]} -> {self.fcLayerMLPDicts[0]["output"]} features')
            transposedFeatures = fluidFeatures.view(batches,-1, *fluidFeatures.shape[1:])
            linearOutput = self.fcs[0](transposedFeatures)
            linearOutput = linearOutput.view(-1, *linearOutput.shape[2:]) 
        else:
            linearOutput = None

        ### Single Layer Case
        if len(self.features) == 1:                
            if self.hasBoundaryLayers:
                fluidConvolution += boundaryConvolution

            if self.fcLayerMLPProperties is not None:
                if self.vertexMLPProperties is not None:
                    fluidConvolution = torch.hstack((linearOutput, fluidConvolution))
                else:
                    fluidConvolution = linearOutput + fluidConvolution

            if self.vertexMLPProperties is not None:
                if verbose:
                    print(f'Layer[0]:\tVertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features')
                    fluidConvolution = runMLP(self.vertexMLPs[0], fluidConvolution, batches, verbose = False)

            if self.outputDecoder is not None:
                # if verbose:
                    # print(f'(pre outputDecoder) fluidConvolution: {fluidConvolution.shape}')
                if verbose:
                    print(f'Layer[0]:\tOutput Decoder {self.outputDecoderProperties["inputFeatures"]} -> {self.outputDecoderProperties["output"]} features')
                fluidConvolution = runMLP(self.outputDecoder, fluidConvolution, batches, verbose = False)          
            if verbose:
                print(f'Final: {fluidConvolution.shape} [min: {torch.min(fluidConvolution)}, max: {torch.max(fluidConvolution)}, mean: {torch.mean(fluidConvolution)}]')  
            return fluidConvolution 
        ### Multi Layer Case
        if verbose:
            print(f'Layer[0]:\tStacking Features: {linearOutput.shape if linearOutput is not None else 0} | {fluidConvolution.shape} | {boundaryConvolution.shape if self.hasBoundaryLayers else 0}')
        if self.hasBoundaryLayers:
            if self.fcLayerMLPProperties is not None:
                ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
            else:
                ans = torch.hstack((fluidConvolution, boundaryConvolution))
        else:
            if self.fcLayerMLPProperties is not None:
                ans = torch.hstack((linearOutput, fluidConvolution))
            else:
                ans = fluidConvolution

        if verbose:
            print(f'Pre-Message Passing Done: {ans.shape}\n')
        
        if self.edgeMLPProperties is not None:
            if verbose:
                print(f'Layer[0]:\tRunning Edge MLP {self.edgeMLPDicts["inputFeatures"]} -> {self.edgeMLPDicts["output"]} features')
            fluidEdgeLengths = self.edgeMLPs[0](fluidEdgeLengths)
            fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
        # print(self.vertexMLP)
        if self.vertexMLPProperties is not None:
            # print(f'Running Vertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features {ans.shape}')
            if verbose:
                print(f'Layer[0]:\tRunning Vertex MLP {self.vertexMLPDicts[0]["inputFeatures"]} -> {self.vertexMLPDicts[0]["output"]} features\n')
            transposedFeatures = ans.view(batches,-1, *ans.shape[1:])
            ans = self.vertexMLPs[0](transposedFeatures)
            ans = ans.view(-1, *ans.shape[2:])
        layers = len(self.convs)
        for i in range(1 if not self.hasBoundaryLayers else 2,layers):
            if verbose:
                print(f'Layer[{i}]:\tInput {ans.shape}')
            # print(f'Layer[{i}]:\tRelu: {ans.shape}')
            ansc = self.relu(ans)
            if verbose:
                # print(f'Layer[{i}]:\tResult for layer {i-1} [min: {torch.min(ansc)}, max: {torch.max(ansc)}, mean: {torch.mean(ansc)}] | [min: {torch.min(ans)}, max: {torch.max(ans)}, mean: {torch.mean(ans)}]')
                print(f'Layer[{i}]:\tRunning Convolution {self.convs[i].inputFeatures} -> {self.convs[i].outputFeatures} features')
            # print(f'Layer[{i}]:\tConvolution: {self.convs[i].inputFeatures} -> {self.convs[i].outputFeatures} features')
            ansConv = self.convs[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths, fluidEdgeWeights)

            # print(f'Layer[{i}]:\tLinear: {self.fcs[i - (1 if self.hasBoundaryLayers else 0)].in_features} -> {self.fcs[i - (1 if self.hasBoundaryLayers else 0)].out_features} features')
            if self.fcLayerMLPProperties is not None:
                if verbose:
                # print(f'Layer[{i}]:\t\tResult [min: {torch.min(ansConv)}, max: {torch.max(ansConv)}, mean: {torch.mean(ansConv)}]')
                    print(f'Layer[{i}]:\tRunning Linear {self.fcLayerMLPDicts[i - (1 if self.hasBoundaryLayers else 0)]["inputFeatures"]} -> {self.fcLayerMLPDicts[i - (1 if self.hasBoundaryLayers else 0)]["output"]} features') 
                transposedFeatures = ansc.view(batches,-1, *ansc.shape[1:])
                ansDense = self.fcs[i - (1 if self.hasBoundaryLayers else 0)](transposedFeatures)
                ansDense = ansDense.view(-1, *ansDense.shape[2:])
            else:
                ansDense = None
            # if verbose:
                # print(f'Layer[{i}]:\t\tResult [min: {torch.min(ansDense)}, max: {torch.max(ansDense)}, mean: {torch.mean(ansDense)}]')
            
            if self.agglomerateViaMLP == False:
                if verbose:
                    print(f'Layer[{i}]:\tSumming {ans.shape} | {ansConv.shape} | {ansDense.shape if ansDense is not None else 0}')
                if self.features[i- (2 if self.hasBoundaryLayers else 1)] == self.features[i-(1 if self.hasBoundaryLayers else 0)] and ans.shape == ansConv.shape:
                    ans = ansConv + ans
                else:
                    ans = ansConv
                if ansDense is not None:
                    ans = ans + ansDense
            else:
                if verbose:
                    print(f'Layer[{i}]:\tStacking {ans.shape} | {ansConv.shape} | {ansDense.shape if ansDense is not None else 0}')
                if self.features[i- (2 if self.hasBoundaryLayers else 1)] == self.features[i-(1 if self.hasBoundaryLayers else 0)] and ans.shape == ansConv.shape:
                    if ansDense is not None:
                        ans_ = torch.hstack((ansConv, ansDense, ans))
                    else:   
                        ans_ = torch.hstack((ansConv, ans))
                else:
                    if ansDense is not None:
                        ans_ = torch.hstack((ansConv, ansDense))
                    else:
                        ans_ = ansConv
            print(f'Layer[{i}]:\tans: {ans.shape if self.agglomerateViaMLP == False else ans_.shape}')

            if self.edgeMLPProperties is not None and i < layers - 1:
                if verbose:
                    print(f'Layer[{i}]:\tRunning Edge MLP {self.edgeMLP["inputFeatures"]} -> {self.edgeMLP["output"]} features')
                fluidEdgeLengths = self.edgeMLPs[i](fluidEdgeLengths)
                fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
            if self.vertexMLPProperties is not None:# and i < layers:# - 1:
                # print(f'Layer[{i}]:\tRunning Vertex MLP {self.vertexMLPDicts[i]["inputFeatures"]} -> {self.vertexMLPDicts[i]["output"]} features')

                if verbose:
                    print(f'Layer[{i}]:\tRunning Vertex MLP {self.vertexMLPDicts[i]["inputFeatures"]} -> {self.vertexMLPDicts[i]["output"]} features')
                if self.agglomerateViaMLP:
                    ans = ans_
                # print(ans.shape)
                transposedFeatures = ans.view(batches,-1, *ans.shape[1:])
                ans = self.vertexMLPs[i](transposedFeatures)
                ans = ans.view(-1, *ans.shape[2:])
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
    