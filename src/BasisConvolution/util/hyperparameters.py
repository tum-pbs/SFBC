import torch
def parseEntry(config, namespace, variable, dictionary, target):
    if namespace in config:
        if variable in config[namespace]:
            dictionary[target] = config[namespace][variable]
    return dictionary[target]

def defaultHyperParameters():
    hyperParameterDict = {
        # 'basisTerms': 4,
        'coordinateMapping': 'cartesian',
        # 'basisFunctions': 'linear',
        'windowFunction': 'None',
        'liLoss': 'yes',
        'initialLR': 0.01,
        'finalLR': 0.0001,
        'lrStep': 10,
        'epochs': 25,
        'frameDistance': 4,
        'iterations': 1000,
        'dataDistance': 1,
        'cutoff': 1800,
        'dataLimit': -1,
        'seed': 42,
        'minUnroll': 2,
        'maxUnroll': 10,
        'historyLength': 0,
        'augmentAngle': False,
        'augmentJitter': False,
        'jitterAmount': 0.1,
        'networkSeed': 42,
        'network': 'default',
        'normalized': False,
        'adjustForFrameDistance': True,
        # 'cutlassBatchSize': 128,
        'weight_decay': 0,
        'input': '',
        'input': './',
        'output': 'training',
        # 'outputBias': False,
        'loss': 'mse',
        'batchSize': 1,
        # 'optimizedWeights': False,
        # 'exponentialDecay': True,
        # 'initializer': 'uniform',
        'fluidFeatures': 'constant:1',
        'boundaryFeatures': 'attributes:n constant:1',
        'boundary': True,
        'groundTruth': 'compute[rho]:attribute:rho',
        'gtMode': 'abs',
        'verbose': False,
        'independent_dxdt': False,
        'unrollIncrement': 100,
        'networkType': 'w/o shift',
        'normalization': 'none',
        'skipLastShift': False,
        'shiftLoss': False,
        'activation': 'silu',
        'dataIndex': '',
        'dxdtLossScaling': 2,
        'exportPath': 'experiments',
        'arch':'',
        'scaleShiftLoss': False,
        'zeroOffset': True,
        'normalizeDensity' : True,
        'device': 'cpu',
        'dtype': torch.float32,

        'optimizer': 'adamw',
        'momentum': 0.9,

        'inputEncoderActive': False,
        'inputEdgeEncoderActive': False,
        'inputBasisEncoderActive': False,
        'firstLayerMode': 'stack',
        'outputDecoderActive': False,
        'edgeMLPActive': False,
        'vertexMLPActive': False,
        'fcLayerMLPActive': True,
        
        'edgeMode': 'none',
        'skipLayerMode': 'mlp',
        'skipConnectionMode': 'cconv',
        'activationOnNode': True,

        'velocityNoise': False,
        'velocityNoiseMagnitude': 0.01,
        'velocityNoiseScaling': 'rel',

        'positionNoise': False,
        'positionNoiseMagnitude': 0.005,

        'unrollVelocityNoise': False,
        'unrollPositionNoise': False,

        'outputScaling': 1/128,


        'inputEncoder': {
                'activation': 'default',
                'gain': 1,
                'norm': True,
                'layout': [32,32],
                'preNorm': True,
                'postNorm': False,
                'noLinear': True,
                'channels': [1],
                'bias':True
            },
        'inputEdgeEncoder': {
                'activation': 'default',
                'gain': 1,
                'norm': True,
                'layout': [32,32],
                'preNorm': False,
                'postNorm': False,
                'noLinear': False,
                'channels': [-1, 16],
                'bias':True
            },
        'inputBasisEncoder': {
            'basisTerms': 5,
            'basisFunction': 'ffourier',
            'mode': 'cat'
        },
        'outputDecoder': {
                'activation': 'default',
                'gain': 1,
                'norm': True,
                'layout': [32,32],
                'preNorm': True,
                'postNorm': False,
                'noLinear': False,
                'channels': [-1,16,16],
                'bias':False
            },
        'edgeMLP': {
                'activation': 'default',
                'gain': 1,
                'norm': True,
                'layout': [32,32],
                'preNorm': False,
                'postNorm': False,
                'noLinear': False,
                'channels': [16,16],
                'bias':True
            },
        'vertexMLP': {
                'activation': 'default',
                'gain': 1,
                'norm': True,
                'layout': [48,48],
                'preNorm': True,
                'postNorm': True,
                'noLinear': False,
                'channels': [-1,8,8,-1],
                'bias':True,
                'vertexInput': True
            },
        'messageMLP': {
                'activation': 'default',
                'gain': 1,
                'norm': False,
                'layout': [48,48],
                'preNorm': False,
                'postNorm': True,
                'noLinear': False,
                'channels': [1],
                'bias':True,
            },
        'fcLayerMLP': {
                'activation': 'default',
                'gain': 1,
                'norm': True,
                'layout': [48,48],
                'preNorm': True,
                'postNorm': True,
                'noLinear': False,
                'channels': [-1,8,8,-1],
                'bias':True
            },
        'convLayer': {
            'basisFunction': 'linear',
            'basisTerms': 4,
            'cutlassBatchSize': 128,
            'initializer': 'uniform',
            'optimizeWeights': False,
            'exponentialDecay': True,
            'cutlassNormalization': False,
            'biasActive': False,
            'mode': 'conv',
            'vertexMode': 'ij',
            'edgeSkip': 'none'
        },
        'shiftCFL': 10,
        'shiftIters': 1,
        'shiftComputeDensity': True,
        'lossTerms': 'both',
        'integrationScheme': 'semiImplicitEuler',
        'numNeighbors': -1
    }
    return hyperParameterDict

def parseArguments(args, hyperParameterDict):
    hyperParameterDict['convLayer']['basisTerms'] = args.basisTerms if hasattr(args, 'basisTerms') else hyperParameterDict['convLayer']['basisTerms']
    hyperParameterDict['convLayer']['basisFunction'] = args.basisFunctions if hasattr(args, 'basisFunctions') else hyperParameterDict['convLayer']['basisFunction']
    hyperParameterDict['convLayer']['cutlassNormalization'] = args.normalized if hasattr(args, 'normalized') else hyperParameterDict['convLayer']['cutlassNormalization']
    hyperParameterDict['convLayer']['biasActive'] = args.biasActive if hasattr(args, 'biasActive') else hyperParameterDict['convLayer']['biasActive']
    hyperParameterDict['convLayer']['optimizeWeights'] = args.optimizedWeights if hasattr(args, 'optimizeWeights') else hyperParameterDict['convLayer']['optimizeWeights']
    hyperParameterDict['convLayer']['exponentialDecay'] = args.exponentialDecay  if hasattr(args, 'exponentialDecay') else hyperParameterDict['convLayer']['exponentialDecay']
    hyperParameterDict['convLayer']['initializer'] = args.initializer if hasattr(args, 'initializer') else hyperParameterDict['convLayer']['initializer']
    hyperParameterDict['convLayer']['cutlassBatchSize'] = args.cutlassBatchSize if hasattr(args, 'cutlassBatchSize') else hyperParameterDict['convLayer']['cutlassBatchSize']
    hyperParameterDict['convLayer']['mode'] = args.convMode if hasattr(args, 'convMode') else hyperParameterDict['convLayer']['mode']


    hyperParameterDict['coordinateMapping'] = args.coordinateMapping if hasattr(args, 'coordinateMapping') else hyperParameterDict['coordinateMapping']
    hyperParameterDict['windowFunction'] =  args.windowFunction if hasattr(args, 'windowFunction') else hyperParameterDict['windowFunction']
    hyperParameterDict['liLoss'] = ('yes' if args.li else 'no' ) if hasattr(args, 'li') else hyperParameterDict['liLoss']
    hyperParameterDict['initialLR'] = args.lr if hasattr(args, 'lr') else hyperParameterDict['initialLR']
    hyperParameterDict['finalLR'] = args.finalLR if hasattr(args, 'finalLR') else hyperParameterDict['finalLR']
    hyperParameterDict['lrStep'] = args.lrStep if hasattr(args, 'lrStep') else hyperParameterDict['lrStep']
    hyperParameterDict['lossTerms'] = args.lossTerms if hasattr(args, 'lossTerms') else hyperParameterDict['lossTerms']
    
    hyperParameterDict['epochs'] = args.epochs if hasattr(args, 'epochs') else hyperParameterDict['epochs']
    hyperParameterDict['frameDistance'] = args.frameDistance if hasattr(args, 'frameDistance') else hyperParameterDict['frameDistance']
    hyperParameterDict['iterations'] = args.iterations if hasattr(args, 'iterations') else hyperParameterDict['iterations']
    hyperParameterDict['dataDistance'] = args.dataDistance if hasattr(args, 'dataDistance') else hyperParameterDict['dataDistance']
    hyperParameterDict['cutoff'] =  args.cutoff if hasattr(args, 'cutoff') else hyperParameterDict['cutoff']
    hyperParameterDict['dataLimit'] =  args.dataLimit  if hasattr(args, 'dataLimit') else hyperParameterDict['dataLimit']
    hyperParameterDict['seed'] =  args.seed if hasattr(args, 'seed') else hyperParameterDict['seed']
    hyperParameterDict['minUnroll'] =  args.minUnroll if hasattr(args, 'minUnroll') else hyperParameterDict['minUnroll']
    hyperParameterDict['maxUnroll'] =  args.maxUnroll if hasattr(args, 'maxUnroll') else hyperParameterDict['maxUnroll']
    hyperParameterDict['historyLength'] =  args.historyLength if hasattr(args, 'historyLength') else hyperParameterDict['historyLength']

    hyperParameterDict['augmentAngle'] =  args.augmentAngle if hasattr(args, 'augmentAngle') else hyperParameterDict['augmentAngle']
    hyperParameterDict['augmentJitter'] =  args.augmentJitter if hasattr(args, 'augmentJitter') else hyperParameterDict['augmentJitter']
    hyperParameterDict['jitterAmount'] =  args.jitterAmount if hasattr(args, 'jitterAmount') else hyperParameterDict['jitterAmount']
    hyperParameterDict['networkSeed'] =  args.networkseed if hasattr(args, 'networkseed') else hyperParameterDict['networkSeed']
    hyperParameterDict['outputScaling'] = args.outputScaling if hasattr(args, 'outputScaling') else hyperParameterDict['outputScaling']
    hyperParameterDict['network'] = args.network if hasattr(args, 'network') else hyperParameterDict['network']

    hyperParameterDict['optimizer'] = args.optimizer if hasattr(args, 'optimizer') else hyperParameterDict['optimizer']
    hyperParameterDict['momentum'] = args.momentum if hasattr(args, 'momentum') else hyperParameterDict['momentum']
    
    hyperParameterDict['adjustForFrameDistance'] = args.adjustForFrameDistance if hasattr(args, 'adjustForFrameDistance') else hyperParameterDict['adjustForFrameDistance']

    hyperParameterDict['weight_decay'] = args.weight_decay if hasattr(args, 'weight_decay') else hyperParameterDict['weight_decay']
    hyperParameterDict['zeroOffset'] = args.zeroOffset if hasattr(args, 'zeroOffset') else hyperParameterDict['zeroOffset']
    hyperParameterDict['normalizeDensity'] = args.normalizeDensity if hasattr(args, 'normalizeDensity') else hyperParameterDict['normalizeDensity']

    # hyperParameterDict['iterations'] = 10
    hyperParameterDict['loss'] = args.loss if hasattr(args, 'loss') else hyperParameterDict['loss']
    hyperParameterDict['batchSize'] = args.batchSize if hasattr(args, 'batchSize') else hyperParameterDict['batchSize']

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
    hyperParameterDict['normalization'] = args.normalized if hasattr(args, 'normalized') else hyperParameterDict['normalization']

    hyperParameterDict['shiftLoss'] = args.shiftLoss if hasattr(args, 'shiftLoss') else hyperParameterDict['shiftLoss']
    hyperParameterDict['dataIndex'] = args.dataIndex if hasattr(args, 'dataIndex') else hyperParameterDict['dataIndex']
    hyperParameterDict['skipLastShift'] = args.skipLastShift if hasattr(args, 'skipLastShift') else hyperParameterDict['skipLastShift']
    hyperParameterDict['dxdtLossScaling'] = args.dxdtLossScaling if hasattr(args, 'dxdtLossScaling') else hyperParameterDict['dxdtLossScaling']
    hyperParameterDict['scaleShiftLoss'] = args.scaleShiftLoss if hasattr(args, 'scaleShiftLoss') else hyperParameterDict['scaleShiftLoss']
    hyperParameterDict['activation'] = args.activation if hasattr(args, 'activation') else hyperParameterDict['activation']
    hyperParameterDict['exportPath'] = args.exportPath if hasattr(args, 'exportPath') else hyperParameterDict['exportPath']
    hyperParameterDict['integrationScheme'] = args.integrationScheme if hasattr(args, 'integrationScheme') else hyperParameterDict['integrationScheme']
    hyperParameterDict['shiftCFL'] = args.shiftCFL if hasattr(args, 'shiftCFL') else hyperParameterDict['shiftCFL'] 
    hyperParameterDict['shiftIters'] = args.shiftIters if hasattr(args, 'shiftIters') else hyperParameterDict['shiftIters']
    hyperParameterDict['shiftComputeDensity'] = args.shiftComputeDensity if hasattr(args, 'shiftComputeDensity') else hyperParameterDict['shiftComputeDensity']

    hyperParameterDict['numNeighbors'] = args.numNeighbors if hasattr(args, 'numNeighbors') else hyperParameterDict['numNeighbors']
    hyperParameterDict['firstLayerMode'] = args.firstLayerMode if hasattr(args, 'firstLayerMode') else hyperParameterDict['firstLayerMode']
            
    hyperParameterDict['edgeMode'] = args.edgeMode if hasattr(args, 'edgeMode') else hyperParameterDict['edgeMode']
    hyperParameterDict['skipLayerMode'] = args.skipLayerMode if hasattr(args, 'skipLayerMode') else hyperParameterDict['skipLayerMode']
    hyperParameterDict['skipConnectionMode'] = args.skipConnectionMode if hasattr(args, 'skipConnectionMode') else hyperParameterDict['skipConnectionMode']
    hyperParameterDict['activationOnNode'] = args.activationOnNode if hasattr(args, 'activationOnNode') else hyperParameterDict['activationOnNode']


    hyperParameterDict['velocityNoise'] = args.velocityNoise if hasattr(args, 'velocityNoise') else hyperParameterDict['velocityNoise']
    hyperParameterDict['velocityNoiseMagnitude'] = args.velocityNoiseMagnitude if hasattr(args, 'velocityNoiseMagnitude') else hyperParameterDict['velocityNoiseMagnitude']
    hyperParameterDict['velocityNoiseScaling'] = args.velocityNoiseScaling if hasattr(args, 'velocityNoiseScaling') else hyperParameterDict['velocityNoiseScaling']

    hyperParameterDict['positionNoise'] = args.positionNoise if hasattr(args, 'positionNoise') else hyperParameterDict['positionNoise']
    hyperParameterDict['positionNoiseMagnitude'] = args.positionNoiseMagnitude if hasattr(args, 'positionNoiseMagnitude') else hyperParameterDict['positionNoiseMagnitude']

    hyperParameterDict['unrollVelocityNoise'] = args.unrollVelocityNoise if hasattr(args, 'unrollVelocityNoise') else hyperParameterDict['unrollVelocityNoise']
    hyperParameterDict['unrollPositionNoise'] = args.unrollPositionNoise if hasattr(args, 'unrollPositionNoise') else hyperParameterDict['unrollPositionNoise']


    hyperParameterDict['device'] = args.device if hasattr(args, 'device') else hyperParameterDict['device']
    # hyperParameterDict['dtype'] = torch.

    hyperParameterDict['progressLabel'] = ''
    if hasattr(args, 'commandIndex'):
        # Batch mode
        hyperParameterDict['progressLabel'] = f'[{args.gpu}/{args.gpus}|{args.commandIndex:3d}/{args.commandCount:3d}]' 

    if hasattr(args, 'inputEncoder'):
        if args.inputEncoder == False:
            hyperParameterDict['inputEncoderActive'] = False
        elif args.inputEncoder == True:
            hyperParameterDict['inputEncoderActive'] = True
    if hasattr(args, 'inputEdgeEncoder'):
        if args.inputEdgeEncoder == False:
            hyperParameterDict['inputEdgeEncoderActive'] = False
        elif args.inputEdgeEncoder == True:
            hyperParameterDict['inputEdgeEncoderActive'] = True
    if hasattr(args, 'inputBasisEncoder'):
        if args.inputBasisEncoder == False:
            hyperParameterDict['inputBasisEncoderActive'] = False
        elif args.inputBasisEncoder == True:
            hyperParameterDict['inputBasisEncoderActive'] = True
    if hasattr(args, 'outputDecoder'):
        if args.outputDecoder == False:
            hyperParameterDict['outputDecoderActive'] = False
        elif args.outputDecoder == True:
            hyperParameterDict['outputDecoderActive'] = True
    if hasattr(args, 'edgeMLP'):
        if args.edgeMLP == False:
            hyperParameterDict['edgeMLPActive'] = False
        elif args.edgeMLP == True:
            hyperParameterDict['edgeMLPActive'] = True
    if hasattr(args, 'vertexMLP'):
        if args.vertexMLP == False:
            hyperParameterDict['vertexMLPActive'] = False
        elif args.vertexMLP == True:
            hyperParameterDict['vertexMLPActive'] = True
    if hasattr(args, 'fcLayer'):
        if args.fcLayer == False:
            hyperParameterDict['fcLayerMLPActive'] = False
        elif args.fcLayer == True:
            hyperParameterDict['fcLayerMLPActive'] = True

    if hasattr(args, 'mlpLayers') and hasattr(args, 'mlpWidth'):
        hyperParameterDict['inputEncoder']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['inputEdgeEncoder']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['outputDecoder']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['edgeMLP']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['vertexMLP']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['fcLayerMLP']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['messageMLP']['layout'] = [args.mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['mlpLayersOverride'] = args.mlpLayers
        hyperParameterDict['mlpWidthOverride'] = args.mlpWidth
    elif hasattr(args, 'mlpWidth'):
        mlpLayers = 2
        hyperParameterDict['inputEncoder']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['inputEdgeEncoder']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['outputDecoder']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['edgeMLP']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['vertexMLP']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['fcLayerMLP']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['messageMLP']['layout'] = [args.mlpWidth for _ in range(mlpLayers)]
        hyperParameterDict['mlpLayersOverride'] = 2
        hyperParameterDict['mlpWidthOverride'] = args.mlpWidth
    elif hasattr(args, 'mlpLayers'):
        mlpWidth = 32
        hyperParameterDict['inputEncoder']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['inputEdgeEncoder']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['outputDecoder']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['edgeMLP']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['vertexMLP']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['fcLayerMLP']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['messageMLP']['layout'] = [mlpWidth for _ in range(args.mlpLayers)]
        hyperParameterDict['mlpLayersOverride'] = args.mlpLayers
        hyperParameterDict['mlpWidthOverride'] = 32


    return hyperParameterDict

import tomli
def parseConfig(config, hyperParameterDict):
    with open(config, 'rb') as f:
        cfg = tomli.load(f) 
        parseEntry(cfg, 'training', 'epochs', hyperParameterDict, 'epochs')
        parseEntry(cfg, 'training', 'iterations', hyperParameterDict, 'iterations')
        parseEntry(cfg, 'training', 'minUnroll', hyperParameterDict, 'minUnroll')
        parseEntry(cfg, 'training', 'maxUnroll', hyperParameterDict, 'maxUnroll')
        parseEntry(cfg, 'training', 'historyLength', hyperParameterDict, 'historyLength')

        parseEntry(cfg, 'augmentation', 'jitter', hyperParameterDict, 'augmentJitter') 
        parseEntry(cfg, 'augmentation', 'rotation', hyperParameterDict, 'augmentAngle')
        parseEntry(cfg, 'augmentation', 'jitterAmount', hyperParameterDict, 'jitterAmount')

        parseEntry(cfg, 'randomization', 'seed', hyperParameterDict, 'seed')
        parseEntry(cfg, 'randomization', 'networkSeed', hyperParameterDict, 'networkSeed')
        # parseEntry(cfg, 'randomization', 'initializer', hyperParameterDict, 'initializer')
        # parseEntry(cfg, 'randomization', 'exponentialDecay', hyperParameterDict, 'exponentialDecay')
        # parseEntry(cfg, 'randomization', 'optimizeWeights', hyperParameterDict, 'optimizeWeights')

        parseEntry(cfg, 'network', 'coordinateMapping', hyperParameterDict, 'coordinateMapping')
        parseEntry(cfg, 'network', 'windowFunction', hyperParameterDict, 'windowFunction')
        if hyperParameterDict['windowFunction'] == 'None':
            hyperParameterDict['windowFunction'] = None
        parseEntry(cfg, 'network', 'activation', hyperParameterDict, 'activation')
        # parseEntry(cfg, 'network', 'outputBias', hyperParameterDict, 'outputBias')
        parseEntry(cfg, 'network', 'arch', hyperParameterDict, 'arch')
        parseEntry(cfg, 'network', 'normalization', hyperParameterDict, 'normalization')

        # parseEntry(cfg, 'basis', 'r', hyperParameterDict, 'basisFunctions')
        # parseEntry(cfg, 'basis', 'b', hyperParameterDict, 'basisTerms')

        parseEntry(cfg, 'optimizer', 'lr', hyperParameterDict, 'initialLR')
        parseEntry(cfg, 'optimizer', 'finalLR', hyperParameterDict, 'finalLR')
        parseEntry(cfg, 'optimizer', 'lrStep', hyperParameterDict, 'lrStep')
        parseEntry(cfg, 'optimizer', 'momentum', hyperParameterDict, 'momentum')
        parseEntry(cfg, 'optimizer', 'optimizer', hyperParameterDict, 'optimizer')
        # parseEntry(cfg, 'optimizer', 'weight_decay', hyperParameterDict, 'weight_decay')

        # parseEntry(cfg, 'compute', 'cutlassBatchSize', hyperParameterDict, 'cutlassBatchSize')
        parseEntry(cfg, 'compute', 'device', hyperParameterDict, 'device')

        parseEntry(cfg, 'io', 'output', hyperParameterDict, 'output')
        parseEntry(cfg, 'io', 'input', hyperParameterDict, 'input')
        parseEntry(cfg, 'io', 'exportPath', hyperParameterDict, 'exportPath')
        parseEntry(cfg, 'io', 'numNeighbors', hyperParameterDict, 'numNeighbors')
        # parseEntry(cfg, 'io', 'normalizeDensity', hyperParameterDict, 'dataIndex')

        parseEntry(cfg, 'dataset', 'frameDistance', hyperParameterDict, 'frameDistance')
        parseEntry(cfg, 'dataset', 'dataDistance', hyperParameterDict, 'dataDistance')
        parseEntry(cfg, 'dataset', 'cutoff', hyperParameterDict, 'cutoff')
        parseEntry(cfg, 'dataset', 'batchSize', hyperParameterDict, 'batchSize')
        parseEntry(cfg, 'dataset', 'dataLimit', hyperParameterDict, 'dataLimit')
        parseEntry(cfg, 'dataset', 'zeroOffset', hyperParameterDict, 'zeroOffset')
        parseEntry(cfg, 'dataset', 'normalizeDensity', hyperParameterDict, 'normalizeDensity')

        parseEntry(cfg, 'loss', 'li', hyperParameterDict, 'liLoss')
        parseEntry(cfg, 'loss', 'loss', hyperParameterDict, 'loss')
        parseEntry(cfg, 'loss', 'lossTerms', hyperParameterDict, 'lossTerms')
        parseEntry(cfg, 'network', 'ff', hyperParameterDict, 'fluidFeatures')
        parseEntry(cfg, 'network', 'bf', hyperParameterDict, 'boundaryFeatures')
        parseEntry(cfg, 'network', 'gt', hyperParameterDict, 'groundTruth')
        parseEntry(cfg, 'network', 'boundary', hyperParameterDict, 'boundary')
        parseEntry(cfg, 'network', 'firstLayerMode', hyperParameterDict, 'firstLayerMode')
        parseEntry(cfg, 'network', 'edgeMode', hyperParameterDict, 'edgeMode')
        parseEntry(cfg, 'network', 'skipLayerMode', hyperParameterDict, 'skipLayerMode')
        parseEntry(cfg, 'network', 'skipConnectionMode', hyperParameterDict, 'skipConnectionMode')
        parseEntry(cfg, 'network', 'activationOnNode', hyperParameterDict, 'activationOnNode')
        parseEntry(cfg, 'network', 'outputScaling', hyperParameterDict, 'outputScaling')

        parseEntry(cfg, 'noise', 'velocity', hyperParameterDict, 'velocityNoise')
        parseEntry(cfg, 'noise', 'velocityMagnitude', hyperParameterDict, 'velocityNoiseMagnitude')
        parseEntry(cfg, 'noise', 'velocityScaling', hyperParameterDict, 'velocityNoiseScaling')
        parseEntry(cfg, 'noise', 'position', hyperParameterDict, 'positionNoise')
        parseEntry(cfg, 'noise', 'positionMagnitude', hyperParameterDict, 'positionNoiseMagnitude')
        parseEntry(cfg, 'noise', 'unrollVelocity', hyperParameterDict, 'unrollVelocityNoise')
        parseEntry(cfg, 'noise', 'unrollPosition', hyperParameterDict, 'unrollPositionNoise')

        parseEntry(cfg, 'misc', 'verbose', hyperParameterDict, 'verbose')
        parseEntry(cfg, 'loss', 'independent_dxdt', hyperParameterDict, 'independent_dxdt')
        parseEntry(cfg, 'training', 'unrollIncrement', hyperParameterDict, 'unrollIncrement')

        parseEntry(cfg, 'shifting', 'networkType', hyperParameterDict, 'networkType')
        parseEntry(cfg, 'shifting', 'shiftLoss', hyperParameterDict, 'shiftLoss')
        parseEntry(cfg, 'shifting', 'scaleShiftLoss', hyperParameterDict, 'scaleShiftLoss')
        parseEntry(cfg, 'shifting', 'integrationScheme', hyperParameterDict, 'integrationScheme')
        parseEntry(cfg, 'shifting', 'shiftIters', hyperParameterDict, 'shiftIters')
        parseEntry(cfg, 'shifting', 'shiftComputeDensity', hyperParameterDict, 'shiftComputeDensity')
        parseEntry(cfg, 'shifting', 'shiftCFL', hyperParameterDict, 'shiftCFL')

        parseEntry(cfg, 'dataset', 'dataIndex', hyperParameterDict, 'dataIndex')
        parseEntry(cfg, 'shifting', 'skipLastShift', hyperParameterDict, 'skipLastShift')
        parseEntry(cfg, 'loss', 'dxdtLossScaling', hyperParameterDict, 'dxdtLossScaling')

        if 'dataset' in cfg:
            if 'additionalData' in cfg['dataset']:
                hyperParameterDict['additionalData'] = cfg['dataset']['additionalData']

        dictList = ['inputEncoder', 'outputDecoder', 'edgeMLP', 'vertexMLP', 'fcLayerMLP', 'convLayer', 'inputBasisEncoder', 'inputEdgeEncoder', 'messageMLP']
        for d in dictList:
            if d in cfg:
                for key in cfg[d]:
                    if key in hyperParameterDict[d]:
                        hyperParameterDict[d][key] = cfg[d][key]
                    else:
                        if key != 'inputFeatures' and key != 'output':
                            raise ValueError('Key %s not found in %s' % (key, d))
                        hyperParameterDict[d][key] = cfg[d][key]
        for d in dictList:
            if d == 'inputBasisEncoder' or d == 'convLayer':
                continue
            # print(d)
            if hyperParameterDict[d]['activation'] == 'default':
                hyperParameterDict[d]['activation'] = hyperParameterDict['activation']

        # if 'inputEncoder' in cfg:
        #     hyperParameterDict['inputEncoder'] = cfg['inputEncoder']
        # if 'outputDecoder' in cfg:
        #     hyperParameterDict['outputDecoder'] = cfg['outputDecoder']
        # if 'edgeMLP' in cfg:
        #     hyperParameterDict['edgeMLP'] = cfg['edgeMLP']
        # if 'vertexMLP' in cfg:
        #     hyperParameterDict['vertexMLP'] = cfg['vertexMLP']
        # if 'vertexMLP' in cfg:
        #     hyperParameterDict['fcLayerMLP'] = cfg['fcLayerMLP']


        parseEntry(cfg, 'mlp', 'inputEncoder', hyperParameterDict, 'inputEncoderActive')
        parseEntry(cfg, 'mlp', 'inputEdgeEncoder', hyperParameterDict, 'inputEdgeEncoderActive')
        parseEntry(cfg, 'mlp', 'inputBasisEncoder', hyperParameterDict, 'inputBasisEncoderActive')
        parseEntry(cfg, 'mlp', 'outputDecoder', hyperParameterDict, 'outputDecoderActive')
        parseEntry(cfg, 'mlp', 'edgeMLP', hyperParameterDict, 'edgeMLPActive')
        parseEntry(cfg, 'mlp', 'vertexMLP', hyperParameterDict, 'vertexMLPActive')
        parseEntry(cfg, 'mlp', 'fcLayer', hyperParameterDict, 'fcLayerMLPActive')

    return hyperParameterDict

import datetime as datetime
import numpy as np
def parseHyperParameters(args, config = None):
    hyperParameterDict = defaultHyperParameters()

    if config is not None:
        if ' ' in config:
            configs = config.split(' ')
            for c in configs:
                hyperParameterDict = parseConfig(c, hyperParameterDict)
        else:
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

        'basisTerms': hyperParameterDict['convLayer']['basisTerms'],
        'basisFunctions': hyperParameterDict['convLayer']['basisFunction'],

        'network': hyperParameterDict['network'],
        'normalization': hyperParameterDict['normalization'],
        # 'outputBias': hyperParameterDict['outputBias'],
        'activation': hyperParameterDict['activation'],
        'networkSeed': hyperParameterDict['networkSeed'],
        'arch': hyperParameterDict['arch'],
        'widths': hyperParameterDict['widths'],
        'layers': hyperParameterDict['layers'],
        'seed': hyperParameterDict['seed'],
        'outputScaling': hyperParameterDict['outputScaling'],

        'lossTerms': hyperParameterDict['lossTerms'],

        'windowFunction': hyperParameterDict['windowFunction'] if hyperParameterDict['windowFunction'] is not None else 'None',
        'coordinateMapping' : hyperParameterDict['coordinateMapping'],

        'trainingFiles': hyperParameterDict['trainingFiles'],
        'frameDistance': hyperParameterDict['frameDistance'],
        'dataDistance': hyperParameterDict['dataDistance'],
        'adjustForFrameDistance': hyperParameterDict['adjustForFrameDistance'],

        # 'initializer': hyperParameterDict['initializer'],



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
        'historyLength': hyperParameterDict['historyLength'],

        'cutlassBatchSize': hyperParameterDict['cutlassBatchSize'],
        'li' : hyperParameterDict['liLoss'] if 'liLoss' in hyperParameterDict else None,

        # 'normalized': hyperParameterDict['normalized'],
        # 'optimizeWeights': hyperParameterDict['optimizeWeights'],
        # 'exponentialDecay': hyperParameterDict['exponentialDecay'],
        'independent_dxdt': hyperParameterDict['independent_dxdt'],
        'unrollIncrement': hyperParameterDict['unrollIncrement'],
        'networkType': hyperParameterDict['networkType'],
        'shiftLoss': hyperParameterDict['shiftLoss'],
        'dataIndex': hyperParameterDict['dataIndex'],
        'skipLastShift': hyperParameterDict['skipLastShift'],
        'dxdtLossScaling': hyperParameterDict['dxdtLossScaling'],
        'scaleShiftLoss': hyperParameterDict['scaleShiftLoss'] if 'scaleShiftLoss' in hyperParameterDict else False,
        'integrationScheme': hyperParameterDict['integrationScheme'],
        'inputEncoderActive': hyperParameterDict['inputEncoderActive'],
        'inputEdgeEncoderActive': hyperParameterDict['inputEdgeEncoderActive'],
        'inputBasisEncoderActive': hyperParameterDict['inputBasisEncoderActive'],
        'firstLayerMode': hyperParameterDict['firstLayerMode'],
        'edgeMode': hyperParameterDict['edgeMode'],

        'velocityNoise': hyperParameterDict['velocityNoise'],
        'velocityNoiseMagnitude': hyperParameterDict['velocityNoiseMagnitude'],
        'velocityNoiseScaling': hyperParameterDict['velocityNoiseScaling'],

        'positionNoise': hyperParameterDict['positionNoise'],
        'positionNoiseMagnitude': hyperParameterDict['positionNoiseMagnitude'],

        'unrollVelocityNoise': hyperParameterDict['unrollVelocityNoise'],
        'unrollPositionNoise': hyperParameterDict['unrollPositionNoise'],

        'skipLayerMode': hyperParameterDict['skipLayerMode'],
        'skipConnectionMode': hyperParameterDict['skipConnectionMode'],
        'activationOnNode': hyperParameterDict['activationOnNode'],
        
        'outputDecoderActive': hyperParameterDict['outputDecoderActive'],
        'edgeMLPActive': hyperParameterDict['edgeMLPActive'],
        'vertexMLPActive': hyperParameterDict['vertexMLPActive'],
        'fcLayerMLPActive': hyperParameterDict['fcLayerMLPActive']
    }
    dictList = ['inputEncoder', 'outputDecoder', 'edgeMLP', 'vertexMLP', 'fcLayerMLP', 'convLayer', 'inputEdgeEncoder', 'inputBasisEncoder', 'messageMLP']
    for d in dictList:
        if d in hyperParameterDict:
            for key in hyperParameterDict[d]:
                config[d + '.' + key] = hyperParameterDict[d][key]
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
    config, attributes, currentState, priorState, trajectoryStates = loadAugmentedFrame(0, dataset, hyperParameterDict, unrollLength=1)
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

    hyperParameterDict['fluidFeatureCount'] = fluidFeatureCount #* max(hyperParameterDict['historyLength'], 1)
    hyperParameterDict['boundaryFeatureCount'] = boundaryFeatureCount #* max(hyperParameterDict['historyLength'], 1)
    hyperParameterDict['groundTruthCount'] = groundTruthCount
    hyperParameterDict['dimension'] = currentState['fluid']['positions'].shape[1]

    # hyperParameterDict['rbfs'] = hyperParameterDict['convLayer']['basisFunction'].split(' ') if isinstance(hyperParameterDict['basisFunctions'], str) else hyperParameterDict['basisFunctions']
    # if len(hyperParameterDict['rbfs']) == 1:
    #     hyperParameterDict['rbfs'] = hyperParameterDict['rbfs'] * hyperParameterDict['dimension']
    # elif len(hyperParameterDict['rbfs']) != hyperParameterDict['dimension']:
    #     raise ValueError('Number of basis functions must match the dimensionality of the problem or be 1')
    
    # hyperParameterDict['rbfs'] = [s.replace('_', ' ') for s in hyperParameterDict['rbfs']]

    # hyperParameterDict['dims'] = hyperParameterDict['basisTerms'].split(' ') if isinstance(hyperParameterDict['basisTerms'], str) else (hyperParameterDict['basisTerms'] if isinstance(hyperParameterDict['basisTerms'], list) else [hyperParameterDict['basisTerms']])
    # if len(hyperParameterDict['dims']) == 1:
    #     hyperParameterDict['dims'] = hyperParameterDict['dims'] * hyperParameterDict['dimension']
    # elif len(hyperParameterDict['dims']) != hyperParameterDict['dimension']:
    #     raise ValueError('Number of basis terms must match the dimensionality of the problem or be 1')

    # hyperParameterDict['dims'] = [int(d) for d in hyperParameterDict['dims']]

    # if hyperParameterDict['dimension'] >= 1:
    #     hyperParameterDict['n'] = hyperParameterDict['dims'][0]
    #     hyperParameterDict['rbf_x'] = hyperParameterDict['rbfs'][0]
    # if hyperParameterDict['dimension'] >= 2:    
    #     hyperParameterDict['m'] = hyperParameterDict['dims'][1]
    #     hyperParameterDict['rbf_y'] = hyperParameterDict['rbfs'][1]
    # if hyperParameterDict['dimension'] >= 3:
    #     hyperParameterDict['l'] = hyperParameterDict['dims'][2]
    #     hyperParameterDict['rbf_z'] = hyperParameterDict['rbfs'][2]


    hyperParameterDict['arch'] =  hyperParameterDict['arch'] + ' ' + str(groundTruthCount)

    hyperParameterDict['timestamp'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
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

    hyperParameterDict['mlpLabel'] = f'[{"V" if hyperParameterDict["vertexMLPActive"] else " "}{"E" if hyperParameterDict["edgeMLPActive"] else " "}{"I" if hyperParameterDict["inputEncoderActive"] else " "}{"O" if hyperParameterDict["outputDecoderActive"] else " "}]'

    if 'mlpLayersOverride' in hyperParameterDict and 'mlpWidthOverride' in hyperParameterDict:
        layoutLabel = f'[{hyperParameterDict["mlpWidthOverride"]}x{hyperParameterDict["mlpLayersOverride"]}]'
        hyperParameterDict['mlpLabel'] = hyperParameterDict['mlpLabel'] + layoutLabel
        
    modeText = ""
    if hyperParameterDict['convLayer']['mode'] == 'conv':
        modeText = f'{hyperParameterDict["convLayer"]["mode"]:4s} [{hyperParameterDict["convLayer"]["basisFunction"]:8s} x {hyperParameterDict["convLayer"]["basisTerms"]:2d}]'
    else:
        modeText = f'{hyperParameterDict["convLayer"]["mode"]:4s}'
        if hyperParameterDict['inputBasisEncoderActive']:
            modeText += f' [{hyperParameterDict["inputBasisEncoder"]["basisFunction"]:8s} x {hyperParameterDict["inputBasisEncoder"]["basisTerms"]:2d}]'
        else:
            modeText += f' [{"" :13s}]'

    encoderText = f'[{"I" if hyperParameterDict["inputEncoderActive"] else " "}{"O" if hyperParameterDict["outputDecoderActive"] else " "}{"E" if hyperParameterDict["inputEdgeEncoderActive"] else " "}{"B" if hyperParameterDict["inputBasisEncoderActive"] and hyperParameterDict["convLayer"]["mode"] == "mlp" else " "}]'

    mlpText = f'[{"V" if hyperParameterDict["vertexMLPActive"] else " "}{"E" if hyperParameterDict["edgeMLPActive"] else " "}{"F" if hyperParameterDict["fcLayerMLPActive"] else " "}]'
    

    layers = [int(a) for a in hyperParameterDict['arch'].split(' ')]
    layerString = ''
    i = 0
    while i < len(layers):
        count = 1
        while i + 1 < len(layers) and layers[i] == layers[i + 1]:
            count += 1
            i += 1
        if count > 1:
            layerString += f'{layers[i]}x{count} '
        else:
            layerString += f'{layers[i]} '
        i += 1

    layerString = f'[{layerString.strip()}]'
    mappingString = f'[{hyperParameterDict["coordinateMapping"][:4]}/{hyperParameterDict["windowFunction"][:4] if hyperParameterDict["windowFunction"] is not None else "None"}]'

    mappingString = mappingString + f'[{hyperParameterDict["frameDistance"]:2d}x{hyperParameterDict["maxUnroll"]:2d}@{hyperParameterDict["historyLength"]:2d}]'

    normString = f'[{hyperParameterDict["activation"][:min(len(hyperParameterDict["activation"]),4)]:4s}/{hyperParameterDict["optimizer"]}/{hyperParameterDict["normalization"][:min(len(hyperParameterDict["normalization"]),5)]:5s}]'

    noiseText = f'[{"u" if hyperParameterDict["velocityNoise"] else " "}{"p" if hyperParameterDict["positionNoise"] else " "}{"v" if hyperParameterDict["unrollVelocityNoise"] else " "}{"p" if hyperParameterDict["unrollPositionNoise"] else " "}]'

    lossTerm = 'b' if hyperParameterDict['lossTerms'] == 'both' else 'x' if hyperParameterDict['lossTerms'] == 'position' else 'u' if hyperParameterDict['lossTerms'] == 'velocity' else '?'

    lossText = f'[{"S" if hyperParameterDict["shiftLoss"] else " "}{lossTerm}{int(hyperParameterDict["dxdtLossScaling"])}{"i" if hyperParameterDict["independent_dxdt"] else " "}]'

    progressLabel = modeText + encoderText + mlpText + layerString + mappingString + normString + f'[{hyperParameterDict["networkType"]}]' + noiseText + lossText

    # progressLabel = modeText + encoderText + mlpText + layerString + mappingString + normString + f'[{hyperParameterDict["networkType"]}]'

    shortLabel = progressLabel + f' - {hyperParameterDict["fluidFeatures"]} - {hyperParameterDict["groundTruth"]}'
    exportLabel = f'{shortLabel} - {hyperParameterDict["timestamp"]} - {hyperParameterDict["networkSeed"]}'.replace(":", ".").replace("/", "_")

    randomNumber = np.random.randint(0, 1000000)

    hyperParameterDict['progressLabel'] += progressLabel
    hyperParameterDict['shortLabel'] = shortLabel
    hyperParameterDict['exportLabel'] = exportLabel

    # hyperParameterDict['shortLabel'] = f'{hyperParameterDict["networkType"]:8s}{"+loss" if hyperParameterDict["shiftLoss"] else ""} [{hyperParameterDict["arch"]:14s}] - [{hyperParameterDict["convLayer"]["basisFunction"]:8s}] x [{hyperParameterDict["convLayer"]["basisTerms"]:2d}] @ {hyperParameterDict["coordinateMapping"]:4s}/{hyperParameterDict["windowFunction"] if hyperParameterDict["windowFunction"] is not None else "None":4s}, {hyperParameterDict["fluidFeatures"]} - {hyperParameterDict["groundTruth"]} {hyperParameterDict["mlpLabel"]}'

    # hyperParameterDict['progressLabel'] = f'{hyperParameterDict["networkType"]:8s}{"+loss" if hyperParameterDict["shiftLoss"] else ""} [{hyperParameterDict["arch"]:4s}] - [{hyperParameterDict["convLayer"]["basisFunction"]:8s}] x [{hyperParameterDict["convLayer"]["basisTerms"]:2d}] @ {hyperParameterDict["coordinateMapping"]:4s}/{hyperParameterDict["windowFunction"] if hyperParameterDict["windowFunction"] is not None else "None":4s} {hyperParameterDict["mlpLabel"]}'

    # hyperParameterDict['exportLabel'] = f'{hyperParameterDict["timestamp"]} - {hyperParameterDict["networkSeed"]} - {hyperParameterDict["shortLabel"]}'.replace(":", ".").replace("/", "_")

    setSeeds(hyperParameterDict['networkSeed'], verbose = hyperParameterDict['verbose'])

    return hyperParameterDict