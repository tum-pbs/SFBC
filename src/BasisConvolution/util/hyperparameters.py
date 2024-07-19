import torch
def parseEntry(config, namespace, variable, dictionary, target):
    if namespace in config:
        if variable in config[namespace]:
            dictionary[target] = config[namespace][variable]
    return dictionary[target]

def defaultHyperParameters():
    hyperParameterDict = {
        'basisTerms': 4,
        'coordinateMapping': 'cartesian',
        'basisFunctions': 'linear',
        'windowFunction': 'None',
        'liLoss': 'yes',
        'initialLR': 0.01,
        'finalLR': 0.0001,
        'lrStep': 10,
        'maxRollOut': 10,
        'epochs': 25,
        'frameDistance': 4,
        'iterations': 1000,
        'dataDistance': 1,
        'cutoff': 1800,
        'dataLimit': -1,
        'seed': 42,
        'minUnroll': 2,
        'maxUnroll': 10,
        'augmentAngle': False,
        'augmentJitter': False,
        'jitterAmount': 0.1,
        'networkSeed': 42,
        'network': 'default',
        'normalized': False,
        'adjustForFrameDistance': True,
        'cutlassBatchSize': 128,
        'weight_decay': 0,
        'input': '',
        'input': './',
        'output': '../../trainingData_TGV/randomFlows/',
        'outputBias': False,
        'loss': 'mse',
        'batchSize': 1,
        'optimizedWeights': False,
        'exponentialDecay': True,
        'initializer': 'uniform',
        'fluidFeatures': 'constant:1',
        'boundaryFeatures': 'attributes:n constant:1',
        'boundary': True,
        'groundTruth': 'compute[rho]:attribute:rho',
        'gtMode': 'abs',
        'verbose': False,
        'independent_dxdt': False,
        'unrollIncrement': 100,
        'networkType': 'w/o shift',
        'skipLastShift': False,
        'shiftLoss': False,
        'activation': 'relu',
        'dataIndex': '',
        'dxdtLossScaling': 1,
        'exportPath': 'experiments',
        'arch':'',
        'scaleShiftLoss': False,
        'zeroOffset': True,
        'device': 'cpu',
        'dtype': torch.float32,
        'inputEncoder': None,
        'outputDecoder': None,
        'edgeMLP': None,
        'vertexMLP': None
    }
    return hyperParameterDict

def parseArguments(args, hyperParameterDict):
    hyperParameterDict['basisTerms'] = args.basisTerms if hasattr(args, 'basisTerms') else hyperParameterDict['basisTerms']
    hyperParameterDict['coordinateMapping'] = args.coordinateMapping if hasattr(args, 'coordinateMapping') else hyperParameterDict['coordinateMapping']
    hyperParameterDict['basisFunctions'] = args.basisFunctions if hasattr(args, 'basisFunctions') else hyperParameterDict['basisFunctions']
    hyperParameterDict['windowFunction'] =  args.windowFunction if hasattr(args, 'windowFunction') else hyperParameterDict['windowFunction']
    hyperParameterDict['liLoss'] = ('yes' if args.li else 'no' ) if hasattr(args, 'li') else hyperParameterDict['liLoss']
    hyperParameterDict['initialLR'] = args.lr if hasattr(args, 'lr') else hyperParameterDict['initialLR']
    hyperParameterDict['finalLR'] = args.finalLR if hasattr(args, 'finalLR') else hyperParameterDict['finalLR']
    hyperParameterDict['lrStep'] = args.lrStep if hasattr(args, 'lrStep') else hyperParameterDict['lrStep']
    
    
    hyperParameterDict['maxRollOut'] = args.maxUnroll if hasattr(args, 'maxUnroll') else hyperParameterDict['maxUnroll']
    hyperParameterDict['epochs'] = args.epochs if hasattr(args, 'epochs') else hyperParameterDict['epochs']
    hyperParameterDict['frameDistance'] = args.frameDistance if hasattr(args, 'frameDistance') else hyperParameterDict['frameDistance']
    hyperParameterDict['iterations'] = args.iterations if hasattr(args, 'iterations') else hyperParameterDict['iterations']
    hyperParameterDict['dataDistance'] = args.dataDistance if hasattr(args, 'dataDistance') else hyperParameterDict['dataDistance']
    hyperParameterDict['cutoff'] =  args.cutoff if hasattr(args, 'cutoff') else hyperParameterDict['cutoff']
    hyperParameterDict['dataLimit'] =  args.dataLimit  if hasattr(args, 'dataLimit') else hyperParameterDict['dataLimit']
    hyperParameterDict['seed'] =  args.seed if hasattr(args, 'seed') else hyperParameterDict['seed']
    hyperParameterDict['minUnroll'] =  args.minUnroll if hasattr(args, 'minUnroll') else hyperParameterDict['minUnroll']
    hyperParameterDict['maxUnroll'] =  args.maxUnroll if hasattr(args, 'maxUnroll') else hyperParameterDict['maxUnroll']
    hyperParameterDict['augmentAngle'] =  args.augmentAngle if hasattr(args, 'augmentAngle') else hyperParameterDict['augmentAngle']
    hyperParameterDict['augmentJitter'] =  args.augmentJitter if hasattr(args, 'augmentJitter') else hyperParameterDict['augmentJitter']
    hyperParameterDict['jitterAmount'] =  args.jitterAmount if hasattr(args, 'jitterAmount') else hyperParameterDict['jitterAmount']
    hyperParameterDict['networkSeed'] =  args.networkseed if hasattr(args, 'networkseed') else hyperParameterDict['networkSeed']
    hyperParameterDict['network'] = args.network if hasattr(args, 'network') else hyperParameterDict['network']
    hyperParameterDict['normalized'] = args.normalized if hasattr(args, 'normalized') else hyperParameterDict['normalized']
    hyperParameterDict['adjustForFrameDistance'] = args.adjustForFrameDistance if hasattr(args, 'adjustForFrameDistance') else hyperParameterDict['adjustForFrameDistance']
    hyperParameterDict['cutlassBatchSize'] = args.cutlassBatchSize if hasattr(args, 'cutlassBatchSize') else hyperParameterDict['cutlassBatchSize']
    hyperParameterDict['normalized'] = args.normalized if hasattr(args, 'normalized') else hyperParameterDict['normalized']
    hyperParameterDict['weight_decay'] = args.weight_decay if hasattr(args, 'weight_decay') else hyperParameterDict['weight_decay']
    hyperParameterDict['zeroOffset'] = args.zeroOffset if hasattr(args, 'zeroOffset') else hyperParameterDict['zeroOffset']

    # hyperParameterDict['iterations'] = 10
    hyperParameterDict['outputBias'] = args.outputBias if hasattr(args, 'outputBias') else hyperParameterDict['outputBias']
    hyperParameterDict['loss'] = args.loss if hasattr(args, 'loss') else hyperParameterDict['loss']
    hyperParameterDict['batchSize'] = args.batchSize if hasattr(args, 'batchSize') else hyperParameterDict['batchSize']

    hyperParameterDict['optimizeWeights'] = args.optimizedWeights if hasattr(args, 'optimizedWeights') else hyperParameterDict['optimizedWeights']
    hyperParameterDict['exponentialDecay'] = args.exponentialDecay  if hasattr(args, 'exponentialDecay') else hyperParameterDict['exponentialDecay']
    hyperParameterDict['initializer'] = args.initializer if hasattr(args, 'initializer') else hyperParameterDict['initializer']

    hyperParameterDict['fluidFeatures'] = args.fluidFeatures if hasattr(args, 'fluidFeatures') else hyperParameterDict['fluidFeatures']
    hyperParameterDict['boundaryFeatures'] = args.boundaryFeatures if hasattr(args, 'boundaryFeatures') else hyperParameterDict['boundaryFeatures']
    hyperParameterDict['boundary'] = args.boundary if hasattr(args, 'boundary') else hyperParameterDict['boundary']
    
    hyperParameterDict['groundTruth'] = args.groundTruth if hasattr(args, 'groundTruth') else hyperParameterDict['groundTruth']

    hyperParameterDict['gtMode'] = args.gtMode if hasattr(args, 'gtMode') else hyperParameterDict['gtMode']
    hyperParameterDict['arch'] = args.arch if hasattr(args, 'arch') else hyperParameterDict['arch']

    hyperParameterDict['input'] = args.input if hasattr(args, 'input') else hyperParameterDict['input']
    hyperParameterDict['output'] = args.output if hasattr(args, 'output') else hyperParameterDict['output']

    hyperParameterDict['verbose'] = args.verbose if hasattr(args, 'verbose') else hyperParameterDict['verbose']
    hyperParameterDict['independent_dxdt'] = args.independent_dxdt if hasattr(args, 'independent_dxdt') else hyperParameterDict['independent_dxdt']
    hyperParameterDict['unrollIncrement'] = args.unrollIncrement if hasattr(args, 'unrollIncrement') else hyperParameterDict['unrollIncrement']
    hyperParameterDict['networkType'] = args.networkType if hasattr(args, 'networkType') else hyperParameterDict['networkType']
    hyperParameterDict['shiftLoss'] = args.shiftLoss if hasattr(args, 'shiftLoss') else hyperParameterDict['shiftLoss']
    hyperParameterDict['dataIndex'] = args.dataIndex if hasattr(args, 'dataIndex') else hyperParameterDict['dataIndex']
    hyperParameterDict['skipLastShift'] = args.skipLastShift if hasattr(args, 'skipLastShift') else hyperParameterDict['skipLastShift']
    hyperParameterDict['dxdtLossScaling'] = args.dxdtLossScaling if hasattr(args, 'dxdtLossScaling') else hyperParameterDict['dxdtLossScaling']
    hyperParameterDict['scaleShiftLoss'] = args.scaleShiftLoss if hasattr(args, 'scaleShiftLoss') else hyperParameterDict['scaleShiftLoss']
    hyperParameterDict['activation'] = args.activation if hasattr(args, 'activation') else hyperParameterDict['activation']
    hyperParameterDict['exportPath'] = args.exportPath if hasattr(args, 'exportPath') else hyperParameterDict['exportPath']

    hyperParameterDict['device'] = args.device if hasattr(args, 'device') else hyperParameterDict['device']
    # hyperParameterDict['dtype'] = torch.

    if hasattr(args, 'inputEncoder'):
        if args.inputEncoder == False:
            hyperParameterDict['inputEncoder'] = None
        elif args.inputEncoder == True and hyperParameterDict['inputEncoder'] is None:
            hyperParameterDict['inputEncoder'] = {
                'activation': 'celu',
                'gain': 1,
                'norm': True,
                'layout': [32],
                'preNorm': False,
                'postNorm': True,
                'noLinear': True,
                'channels': 1
            }
    if hasattr(args, 'outputDecoder'):
        if args.outputDecoder == False:
            hyperParameterDict['outputDecoder'] = None
        elif args.outputDecoder == True and hyperParameterDict['outputDecoder'] is None:
            hyperParameterDict['outputDecoder'] = {
                'activation': 'celu',
                'gain': 1,
                'norm': True,
                'layout': [32],
                'output': 1,
                'preNorm': False,
                'postNorm': True,
                'noLinear': True,
                'channels': 1
            }
    if hasattr(args, 'edgeMLP'):
        if args.edgeMLP == False:
            hyperParameterDict['edgeMLP'] = None
        elif args.edgeMLP == True and hyperParameterDict['edgeMLP'] is None:
            hyperParameterDict['edgeMLP'] = {
                'activation': 'celu',
                'gain': 1,
                'norm': True,
                'layout': [32],
                'preNorm': False,
                'postNorm': True,
                'noLinear': False,
                'channels': [8,2]
            }
    if hasattr(args, 'vertexMLP'):
        if args.vertexMLP == False:
            hyperParameterDict['vertexMLP'] = None
        elif args.vertexMLP == True and hyperParameterDict['vertexMLP'] is None:
            hyperParameterDict['vertexMLP'] = {
                'activation': 'celu',
                'gain': 1,
                'norm': True,
                'layout': [32],
                'preNorm': False,
                'postNorm': True,
                'noLinear': False,
                'channels': [8,1]
            }


    return hyperParameterDict

import tomli
def parseConfig(config, hyperParameterDict):
    with open(config, 'rb') as f:
        cfg = tomli.load(f) 
        parseEntry(cfg, 'training', 'epochs', hyperParameterDict, 'epochs')
        parseEntry(cfg, 'training', 'iterations', hyperParameterDict, 'iterations')
        parseEntry(cfg, 'training', 'minUnroll', hyperParameterDict, 'minUnroll')
        parseEntry(cfg, 'training', 'maxUnroll', hyperParameterDict, 'maxUnroll')

        parseEntry(cfg, 'augmentation', 'jitter', hyperParameterDict, 'augmentJitter') 
        parseEntry(cfg, 'augmentation', 'rotation', hyperParameterDict, 'augmentAngle')
        parseEntry(cfg, 'augmentation', 'jitterAmount', hyperParameterDict, 'jitterAmount')

        parseEntry(cfg, 'randomization', 'seed', hyperParameterDict, 'seed')
        parseEntry(cfg, 'randomization', 'networkSeed', hyperParameterDict, 'networkSeed')
        parseEntry(cfg, 'randomization', 'initializer', hyperParameterDict, 'initializer')
        parseEntry(cfg, 'randomization', 'exponentialDecay', hyperParameterDict, 'exponentialDecay')
        parseEntry(cfg, 'randomization', 'optimizeWeights', hyperParameterDict, 'optimizeWeights')

        parseEntry(cfg, 'network', 'coordinateMapping', hyperParameterDict, 'coordinateMapping')
        parseEntry(cfg, 'network', 'windowFunction', hyperParameterDict, 'windowFunction')
        if hyperParameterDict['windowFunction'] == 'None':
            hyperParameterDict['windowFunction'] = None
        parseEntry(cfg, 'network', 'activation', hyperParameterDict, 'activation')
        parseEntry(cfg, 'network', 'outputBias', hyperParameterDict, 'outputBias')
        parseEntry(cfg, 'network', 'arch', hyperParameterDict, 'arch')

        parseEntry(cfg, 'basis', 'r', hyperParameterDict, 'basisFunctions')
        parseEntry(cfg, 'basis', 'b', hyperParameterDict, 'basisTerms')

        parseEntry(cfg, 'optimizer', 'lr', hyperParameterDict, 'initialLR')
        parseEntry(cfg, 'optimizer', 'finalLR', hyperParameterDict, 'finalLR')
        parseEntry(cfg, 'optimizer', 'lrStep', hyperParameterDict, 'lrStep')
        # parseEntry(cfg, 'optimizer', 'weight_decay', hyperParameterDict, 'weight_decay')

        parseEntry(cfg, 'compute', 'cutlassBatchSize', hyperParameterDict, 'cutlassBatchSize')
        parseEntry(cfg, 'compute', 'device', hyperParameterDict, 'device')

        parseEntry(cfg, 'io', 'output', hyperParameterDict, 'output')
        parseEntry(cfg, 'io', 'input', hyperParameterDict, 'input')
        parseEntry(cfg, 'io', 'exportPath', hyperParameterDict, 'exportPath')

        parseEntry(cfg, 'dataset', 'frameDistance', hyperParameterDict, 'frameDistance')
        parseEntry(cfg, 'dataset', 'dataDistance', hyperParameterDict, 'dataDistance')
        parseEntry(cfg, 'dataset', 'cutoff', hyperParameterDict, 'cutoff')
        parseEntry(cfg, 'dataset', 'batchSize', hyperParameterDict, 'batchSize')
        parseEntry(cfg, 'dataset', 'dataLimit', hyperParameterDict, 'dataLimit')
        parseEntry(cfg, 'dataset', 'zeroOffset', hyperParameterDict, 'zeroOffset')

        parseEntry(cfg, 'loss', 'li', hyperParameterDict, 'liLoss')
        parseEntry(cfg, 'loss', 'loss', hyperParameterDict, 'loss')
        parseEntry(cfg, 'network', 'ff', hyperParameterDict, 'fluidFeatures')
        parseEntry(cfg, 'network', 'bf', hyperParameterDict, 'boundaryFeatures')
        parseEntry(cfg, 'network', 'gt', hyperParameterDict, 'groundTruth')
        parseEntry(cfg, 'network', 'boundary', hyperParameterDict, 'boundary')

        parseEntry(cfg, 'misc', 'verbose', hyperParameterDict, 'verbose')
        parseEntry(cfg, 'loss', 'independent_dxdt', hyperParameterDict, 'independent_dxdt')
        parseEntry(cfg, 'training', 'unrollIncrement', hyperParameterDict, 'unrollIncrement')

        parseEntry(cfg, 'shifting', 'networkType', hyperParameterDict, 'networkType')
        parseEntry(cfg, 'shifting', 'shiftLoss', hyperParameterDict, 'shiftLoss')
        parseEntry(cfg, 'shifting', 'scaleShiftLoss', hyperParameterDict, 'scaleShiftLoss')
        parseEntry(cfg, 'dataset', 'dataIndex', hyperParameterDict, 'dataIndex')
        parseEntry(cfg, 'shifting', 'skipLastShift', hyperParameterDict, 'skipLastShift')
        parseEntry(cfg, 'loss', 'dxdtLossScaling', hyperParameterDict, 'dxdtLossScaling')

        if 'inputEncoder' in cfg:
            hyperParameterDict['inputEncoder'] = cfg['inputEncoder']
        else:
            hyperParameterDict['inputEncoder'] = None
        if 'outputDecoder' in cfg:
            hyperParameterDict['outputDecoder'] = cfg['outputDecoder']
        else:
            hyperParameterDict['outputDecoder'] = None
        if 'edgeMLP' in cfg:
            hyperParameterDict['edgeMLP'] = cfg['edgeMLP']
        else:
            hyperParameterDict['edgeMLP'] = None
        if 'vertexMLP' in cfg:
            hyperParameterDict['vertexMLP'] = cfg['vertexMLP']
        else:
            hyperParameterDict['vertexMLP'] = None
    return hyperParameterDict

import datetime as datetime
import numpy as np
def parseHyperParameters(args, config = None):
    hyperParameterDict = defaultHyperParameters()

    if config is not None:
        hyperParameterDict = parseConfig(config, hyperParameterDict)
    
    hyperParameterDict = parseArguments(args, hyperParameterDict)

    
    

    # featureNames = [f for f in hyperParameterDict['features'].split(' ') if f] 
    # targetNames = [f for f in hyperParameterDict['targets'].split(' ') if f]

    # inputFeatures = assembleDummyFeatures(featureNames).shape[1]
    # outputFeatures = assembleDummyGroundTruth(targetNames).shape[1]
    
    # hyperParameterDict['arch'] =  hyperParameterDict['arch'] + ' ' + str(outputFeatures)

    # hyperParameterDict['fluidFeatures'] = inputFeatures
    # hyperParameterDict['boundaryFeatures'] = 0
    # hyperParameterDict['outputFeatures'] = outputFeatures
    # hyperParameterDict['featureNames'] = featureNames
    # hyperParameterDict['targetNames'] = targetNames

    # hyperParameterDict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # hyperParameterDict['networkPrefix'] = hyperParameterDict['network']
    # hyperParameterDict['exportString'] = '%s - n=[%2d,%2d] rbf=[%s,%s] map = %s window = %s d = %2d e = %2d arch %s distance = %2d - %s seed %s%s' % (
    #     hyperParameterDict['networkPrefix'], hyperParameterDict['basisTerms'], hyperParameterDict['basisFunctions'], hyperParameterDict['coordinateMapping'], 
    #     hyperParameterDict['windowFunction'], hyperParameterDict['frameDistance'], hyperParameterDict['epochs'], 
    #     hyperParameterDict['arch'], hyperParameterDict['frameDistance'], hyperParameterDict['timestamp'], hyperParameterDict['networkSeed'], hyperParameterDict['networkType'])
    # hyperParameterDict['shortLabel'] = '%8s [%14s] - %s -> [%8s, %8s] x [%2d, %2d] @ %2s, %s %s-> %s %s' % (
    #     hyperParameterDict['windowFunction'], hyperParameterDict['arch'], hyperParameterDict['coordinateMapping'], 
    #     hyperParameterDict['basisFunctions'], hyperParameterDict['basisTerms'],,hyperParameterDict['networkSeed'], hyperParameterDict['features'], ' Idp' if hyperParameterDict['independent_dxdt'] else '', hyperParameterDict['targets'], hyperParameterDict['networkType'])
    
    # hyperParameterDict['widths'] = hyperParameterDict['arch'].strip().split(' ')
    # hyperParameterDict['layers'] = [int(s) for s in hyperParameterDict['widths']]
        
    # setSeeds(hyperParameterDict['networkSeed'], verbose = hyperParameterDict['verbose'])

    return hyperParameterDict


import pandas as pd
import copy
def make_hash(o):

  """
  Makes a hash from a dictionary, list, tuple or set to any level, that contains
  only other hashable types (including any lists, tuples, sets, and
  dictionaries).
  """

  if isinstance(o, (set, tuple, list)):

    return tuple([make_hash(e) for e in o])    

  elif not isinstance(o, dict):

    return hash(o)

  new_o = copy.deepcopy(o)
  for k, v in new_o.items():
    new_o[k] = make_hash(v)

  return hash(tuple(frozenset(sorted(new_o.items()))))

# dataset = pd.concat([dataset, pd.DataFrame({**hyperParameterDict})], axis=1)

def toPandaDict(hyperParameterDict):
    # print('inputEncoder:', hyperParameterDict['inputEncoder'])
    config = {
        'timestamp': hyperParameterDict['timestamp'],

        'basisTerms': hyperParameterDict['basisTerms'],
        'basisFunctions': hyperParameterDict['basisFunctions'],

        'network': hyperParameterDict['network'],
        'outputBias': hyperParameterDict['outputBias'],
        'activation': hyperParameterDict['activation'],
        'networkSeed': hyperParameterDict['networkSeed'],
        'arch': hyperParameterDict['arch'],
        'widths': hyperParameterDict['widths'],
        'layers': hyperParameterDict['layers'],
        'seed': hyperParameterDict['seed'],

        'windowFunction': hyperParameterDict['windowFunction'],
        'coordinateMapping' : hyperParameterDict['coordinateMapping'],

        'trainingFiles': hyperParameterDict['trainingFiles'],
        'frameDistance': hyperParameterDict['frameDistance'],
        'dataDistance': hyperParameterDict['dataDistance'],
        'adjustForFrameDistance': hyperParameterDict['adjustForFrameDistance'],

        'initializer': hyperParameterDict['initializer'],



        'initialLR': hyperParameterDict['initialLR'],
        'finalLR': hyperParameterDict['finalLR'],
        'lrStep': hyperParameterDict['lrStep'],
        'LRgamma': hyperParameterDict['gamma'],

        'epochs': hyperParameterDict['epochs'],
        'iterations': hyperParameterDict['iterations'],
        'batchSize': hyperParameterDict['batchSize'],

        'fluidFeatures': hyperParameterDict['fluidFeatures'],
        'boundaryFeatures': hyperParameterDict['boundaryFeatures'],
        'groundTruth': hyperParameterDict['groundTruth'],

        'fluidFeatureCount': hyperParameterDict['fluidFeatureCount'],
        'boundaryFeatureCount': hyperParameterDict['boundaryFeatureCount'],
        'groundTruthCount': hyperParameterDict['groundTruthCount'],

        # 'features': hyperParameterDict['features'],
        # 'targets': hyperParameterDict['targets'],
        'loss': hyperParameterDict['loss'],

        'augmentJitter': hyperParameterDict['augmentJitter'],
        'augmentAngle': hyperParameterDict['augmentAngle'],
        'jitterAmount': hyperParameterDict['jitterAmount'],

        'minUnroll': hyperParameterDict['minUnroll'],
        'maxUnroll': hyperParameterDict['maxUnroll'],
        'maxRollOut': hyperParameterDict['maxRollOut'],

        'cutlassBatchSize': hyperParameterDict['cutlassBatchSize'],
        'li' : hyperParameterDict['liLoss'] if 'liLoss' in hyperParameterDict else None,

        'normalized': hyperParameterDict['normalized'],
        'optimizeWeights': hyperParameterDict['optimizeWeights'],
        'exponentialDecay': hyperParameterDict['exponentialDecay'],
        'independent_dxdt': hyperParameterDict['independent_dxdt'],
        'unrollIncrement': hyperParameterDict['unrollIncrement'],
        'networkType': hyperParameterDict['networkType'],
        'shiftLoss': hyperParameterDict['shiftLoss'],
        'dataIndex': hyperParameterDict['dataIndex'],
        'skipLastShift': hyperParameterDict['skipLastShift'],
        'dxdtLossScaling': hyperParameterDict['dxdtLossScaling'],
        'scaleShiftLoss': hyperParameterDict['scaleShiftLoss'] if 'scaleShiftLoss' in hyperParameterDict else False,

        'inputEncoder': True if hyperParameterDict['inputEncoder'] is not None else False,
        'outputDecoder': True if hyperParameterDict['outputDecoder'] is not None else False,

        'inputEncoder_activation': hyperParameterDict['inputEncoder']['activation'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_gain': hyperParameterDict['inputEncoder']['gain'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_norm': hyperParameterDict['inputEncoder']['norm'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_layout': hyperParameterDict['inputEncoder']['layout'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_output': hyperParameterDict['inputEncoder']['output'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_preNorm': hyperParameterDict['inputEncoder']['preNorm'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_postNorm': hyperParameterDict['inputEncoder']['postNorm'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_noLinear': hyperParameterDict['inputEncoder']['noLinear'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_channels': hyperParameterDict['inputEncoder']['channels'] if hyperParameterDict['inputEncoder'] is not None else None,
        'inputEncoder_inputFeatures': hyperParameterDict['inputEncoder']['inputFeatures'] if hyperParameterDict['inputEncoder'] is not None else None,

        'outputDecoder_activation': hyperParameterDict['outputDecoder']['activation'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_gain': hyperParameterDict['outputDecoder']['gain'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_norm': hyperParameterDict['outputDecoder']['norm'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_layout': hyperParameterDict['outputDecoder']['layout'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_output': hyperParameterDict['outputDecoder']['output'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_preNorm': hyperParameterDict['outputDecoder']['preNorm'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_postNorm': hyperParameterDict['outputDecoder']['postNorm'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_noLinear': hyperParameterDict['outputDecoder']['noLinear'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_channels': hyperParameterDict['outputDecoder']['channels'] if hyperParameterDict['outputDecoder'] is not None else None,
        'outputDecoder_inputFeatures': hyperParameterDict['outputDecoder']['inputFeatures'] if hyperParameterDict['outputDecoder'] is not None else None,

        'edgeMLP': True if hyperParameterDict['edgeMLP'] is not None else False,
        'vertexMLP': True if hyperParameterDict['vertexMLP'] is not None else False,
        'edgeMLP_activation': hyperParameterDict['edgeMLP']['activation'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_gain': hyperParameterDict['edgeMLP']['gain'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_norm': hyperParameterDict['edgeMLP']['norm'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_layout': hyperParameterDict['edgeMLP']['layout'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_output': hyperParameterDict['edgeMLP']['output'] if hyperParameterDict['edgeMLP'] and 'output'in hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_preNorm': hyperParameterDict['edgeMLP']['preNorm'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_postNorm': hyperParameterDict['edgeMLP']['postNorm'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_noLinear': hyperParameterDict['edgeMLP']['noLinear'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_channels': hyperParameterDict['edgeMLP']['channels'] if hyperParameterDict['edgeMLP'] is not None else None,
        'edgeMLP_inputFeatures': hyperParameterDict['edgeMLP']['inputFeatures'] if hyperParameterDict['edgeMLP'] and 'inputFeatures'in hyperParameterDict['edgeMLP'] is not None else None,

        'vertexMLP_activation': hyperParameterDict['vertexMLP']['activation'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_gain': hyperParameterDict['vertexMLP']['gain'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_norm': hyperParameterDict['vertexMLP']['norm'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_layout': hyperParameterDict['vertexMLP']['layout'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_output': hyperParameterDict['vertexMLP']['output'] if hyperParameterDict['vertexMLP'] and 'output'in hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_preNorm': hyperParameterDict['vertexMLP']['preNorm'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_postNorm': hyperParameterDict['vertexMLP']['postNorm'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_noLinear': hyperParameterDict['vertexMLP']['noLinear'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_channels': hyperParameterDict['vertexMLP']['channels'] if hyperParameterDict['vertexMLP'] is not None else None,
        'vertexMLP_inputFeatures': hyperParameterDict['vertexMLP']['inputFeatures'] if hyperParameterDict['vertexMLP'] and 'inputFeatures'in hyperParameterDict['vertexMLP']  is not None else None,

    }
    # for k in list(config.keys()):
        # print(k, type(config[k]), config[k])

    # display(config)
    hashedConfig = make_hash(config)
    config['hash'] = hashedConfig
    return config

import warnings
import random

def setSeeds(seed, verbose = False):
    if verbose:
        print('Setting all rng seeds to %d' % seed)


    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

from BasisConvolution.util.augment import loadAugmentedFrame
from datetime import datetime

def finalizeHyperParameters(hyperParameterDict, dataset):
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(0, dataset, hyperParameterDict)
    hyperParameterDict['trainingFiles'] = ' '.join(dataset.fileNames)
    boundaryFeatureCount = 0
    if hyperParameterDict['boundary']:
        if 'boundary' in currentState and currentState['boundary'] is not None:
            boundaryFeatures = currentState['boundary']['features']# getFeatures(hyperParameterDict['boundaryFeatures'].split(' '), currentState['boundary'], priorState['boundary'] if priorState is not None else None, config, currentState['time'] - priorState['time'] if priorState is not None else 0.0)
            boundaryFeatureCount = boundaryFeatures.shape[1]
            # pass
        else:
            hyperParameterDict['boundary'] = False
            warnings.warn('Boundary data requested but not available. Disabling boundary data.')
    else:
        if 'boundary' in currentState and currentState['boundary'] is not None:
            warnings.warn('Boundary data available but not requested. Ignoring boundary data!')
    
    fluidFeatures = currentState['fluid']['features'] #getFeatures(hyperParameterDict['fluidFeatures'].split(' '), currentState['fluid'], priorState['fluid'] if priorState is not None else None, config, currentState['time'] - priorState['time'] if priorState is not None else 0.0)
    fluidFeatureCount = fluidFeatures.shape[1]

    # print(currentState['fluid'])
    groundTruth = trajectoryStates[0]['fluid']['target'] # getFeatures(hyperParameterDict['groundTruth'].split(' '), trajectoryStates[0]['fluid'], currentState['fluid'], config, trajectoryStates[0]['time'] - currentState['time'])
    # print(groundTruth)
    groundTruthCount = groundTruth.shape[1]

    lrStep = hyperParameterDict['lrStep']
    initialLR = hyperParameterDict['initialLR']
    finalLR = hyperParameterDict['finalLR']

    totalIterations = hyperParameterDict['iterations'] * hyperParameterDict['epochs']
    hyperParameterDict['totalIterations'] = totalIterations 
    lrSteps = int(np.ceil((totalIterations - lrStep) / lrStep))
    gamma = np.power(finalLR / initialLR, 1/lrSteps)
    hyperParameterDict['gamma'] = gamma

    hyperParameterDict['fluidFeatureCount'] = fluidFeatureCount
    hyperParameterDict['boundaryFeatureCount'] = boundaryFeatureCount
    hyperParameterDict['groundTruthCount'] = groundTruthCount
    hyperParameterDict['dimension'] = currentState['fluid']['positions'].shape[1]

    hyperParameterDict['rbfs'] = hyperParameterDict['basisFunctions'].split(' ') if isinstance(hyperParameterDict['basisFunctions'], str) else hyperParameterDict['basisFunctions']
    if len(hyperParameterDict['rbfs']) == 1:
        hyperParameterDict['rbfs'] = hyperParameterDict['rbfs'] * hyperParameterDict['dimension']
    elif len(hyperParameterDict['rbfs']) != hyperParameterDict['dimension']:
        raise ValueError('Number of basis functions must match the dimensionality of the problem or be 1')
    
    hyperParameterDict['rbfs'] = [s.replace('_', ' ') for s in hyperParameterDict['rbfs']]

    hyperParameterDict['dims'] = hyperParameterDict['basisTerms'].split(' ') if isinstance(hyperParameterDict['basisTerms'], str) else (hyperParameterDict['basisTerms'] if isinstance(hyperParameterDict['basisTerms'], list) else [hyperParameterDict['basisTerms']])
    if len(hyperParameterDict['dims']) == 1:
        hyperParameterDict['dims'] = hyperParameterDict['dims'] * hyperParameterDict['dimension']
    elif len(hyperParameterDict['dims']) != hyperParameterDict['dimension']:
        raise ValueError('Number of basis terms must match the dimensionality of the problem or be 1')

    hyperParameterDict['dims'] = [int(d) for d in hyperParameterDict['dims']]

    if hyperParameterDict['dimension'] >= 1:
        hyperParameterDict['n'] = hyperParameterDict['dims'][0]
        hyperParameterDict['rbf_x'] = hyperParameterDict['rbfs'][0]
    if hyperParameterDict['dimension'] >= 2:    
        hyperParameterDict['m'] = hyperParameterDict['dims'][1]
        hyperParameterDict['rbf_y'] = hyperParameterDict['rbfs'][1]
    if hyperParameterDict['dimension'] >= 3:
        hyperParameterDict['l'] = hyperParameterDict['dims'][2]
        hyperParameterDict['rbf_z'] = hyperParameterDict['rbfs'][2]


    hyperParameterDict['arch'] =  hyperParameterDict['arch'] + ' ' + str(groundTruthCount)

    hyperParameterDict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hyperParameterDict['networkPrefix'] = hyperParameterDict['network']
    # hyperParameterDict['exportString'] = '%s - n=[%s] rbf=[%s] map = %s window = %s d = %2d e = %2d arch %s distance = %2d - %s seed %s%s' % (
    #     hyperParameterDict['networkPrefix'], hyperParameterDict['basisTerms'], hyperParameterDict['basisFunctions'], hyperParameterDict['coordinateMapping'], 
    #     hyperParameterDict['windowFunction'], hyperParameterDict['frameDistance'], hyperParameterDict['epochs'], 
    #     hyperParameterDict['arch'], hyperParameterDict['frameDistance'], hyperParameterDict['timestamp'], hyperParameterDict['networkSeed'], hyperParameterDict['networkType'])
    # hyperParameterDict['shortLabel'] = '%8s [%14s] - %s -> [%8s] x [%s] @ %2s, %s %s-> %s %s' % (
    #     hyperParameterDict['windowFunction'], hyperParameterDict['arch'], hyperParameterDict['coordinateMapping'], 
    #     hyperParameterDict['basisFunctions'], hyperParameterDict['basisTerms'],hyperParameterDict['networkSeed'], hyperParameterDict['features'], ' Idp' if hyperParameterDict['independent_dxdt'] else '', hyperParameterDict['targets'], hyperParameterDict['networkType'])
    
    hyperParameterDict['widths'] = hyperParameterDict['arch'].strip().split(' ')
    hyperParameterDict['layers'] = [int(s) for s in hyperParameterDict['widths']]


    hyperParameterDict['shortLabel'] = f'{hyperParameterDict["networkType"]:8s} [{hyperParameterDict["arch"]:14s}] -> [{hyperParameterDict["basisFunctions"]:8s}] x [{hyperParameterDict["basisTerms"]:2d}] @ {hyperParameterDict["coordinateMapping"]:4s}/{hyperParameterDict["windowFunction"]:4s}, {hyperParameterDict["fluidFeatures"]} -> {hyperParameterDict["groundTruth"]}'

    hyperParameterDict['progressLabel'] = f'{hyperParameterDict["networkType"]:8s} [{hyperParameterDict["arch"]:4s}] -> [{hyperParameterDict["basisFunctions"]:8s}] x [{hyperParameterDict["basisTerms"]:2d}] @ {hyperParameterDict["coordinateMapping"]:4s}/{hyperParameterDict["windowFunction"]:4s}'

    hyperParameterDict['exportLabel'] = f'{hyperParameterDict["timestamp"]} - {hyperParameterDict["networkSeed"]} - {hyperParameterDict["shortLabel"]}'.replace(":", ".").replace("/", "_")

    setSeeds(hyperParameterDict['networkSeed'], verbose = hyperParameterDict['verbose'])

    return hyperParameterDict