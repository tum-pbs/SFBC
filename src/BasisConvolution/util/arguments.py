import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, default=argparse.SUPPRESS, help='Number of epochs [default = 25]')
parser.add_argument('-cmap','--coordinateMapping', type=str, default=argparse.SUPPRESS, help='Coordinate mapping [default = cartesian]')
parser.add_argument('-w','--windowFunction', type=str, default=argparse.SUPPRESS, help='Window function [default = poly6]')
parser.add_argument('-c','--cutoff', type=int, default=argparse.SUPPRESS, help='Cutoff distance [default = 1800]')
parser.add_argument('-b','--batch_size', type=int, default=argparse.SUPPRESS, help='Batch size [default = 1]')
parser.add_argument('-o','--output', type = str, default = argparse.SUPPRESS, help='Output directory [default = ""]')
parser.add_argument('--cutlassBatchSize', type=int, default=argparse.SUPPRESS, help='Cutlass batch size [default = 512]')
parser.add_argument('--lr', type=float, default=argparse.SUPPRESS, help='Learning rate [default = 0.01]')
parser.add_argument('--finalLR', type=float, default=argparse.SUPPRESS, help='Final learning rate [default = 0.0001]')
parser.add_argument('--lrStep', type=int, default=argparse.SUPPRESS, help='Learning rate step [default = 1000]')

# used for batch processing
parser.add_argument('--commandIndex', type=int, default=argparse.SUPPRESS, help='Command index [default = 0]')
parser.add_argument('--commandCount', type=int, default=argparse.SUPPRESS, help='Command [default = ""]')

# parser.add_argument('--lr_decay_factor', type=float, default=argparse.SUPPRESS, help='Learning rate decay factor [default = 0.9]')
# parser.add_argument('--lr_decay_step_size', type=int, default=argparse.SUPPRESS, help='Learning rate decay step size [default = 1]')
parser.add_argument('--weight_decay', type=float, default=argparse.SUPPRESS, help='Weight decay [default = 0]')
parser.add_argument('-r','--basisFunctions', type=str, default=argparse.SUPPRESS, help='RBF kernel X-Component [default = linear]')
parser.add_argument('-n','--basisTerms', type=int, default=argparse.SUPPRESS, help = "RBF Kernel X-Width [default = 4]")
parser.add_argument('--convMode', type=str, default=argparse.SUPPRESS, help='Convolution mode [default = "conv"]')
parser.add_argument('--normalizeDensity', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Normalize density [default = False]')

parser.add_argument('--seed', type=int, default=argparse.SUPPRESS, help='Seed [default = 42]')
parser.add_argument('--networkseed', type=int, default=argparse.SUPPRESS, help='Network seed [default = 42]')
parser.add_argument('-d','--frameDistance', type=int, default=argparse.SUPPRESS, help='Frame distance [default = 16]')
parser.add_argument('--dataDistance', type=int, default=argparse.SUPPRESS, help='Data distance [default = 1]')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('-f','--forwardLoss', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Forward loss [default = False]')
parser.add_argument('-v','--verbose', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Verbose [default = False]')
parser.add_argument('-l','--li', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='LI (Neighborhood scaled loss) [default = True]')
parser.add_argument('-a','--activation', type=str, default=argparse.SUPPRESS, help='Activation function [default = relu]')
parser.add_argument('-i','--input', type=str, default=argparse.SUPPRESS, help='Input directory [default = ""]')
# parser.add_argument('-o','--output', type=str, default=argparse.SUPPRESS
parser.add_argument('--arch', type=str, default=argparse.SUPPRESS, help='Architecture [default = ""]')
parser.add_argument('--dataLimit', type=int, default=argparse.SUPPRESS, help='Limit data [default = -1]')
parser.add_argument('--iterations', type=int, default=argparse.SUPPRESS, help='Iterations [default = 1000]')
parser.add_argument('-u', '--maxUnroll', type=int, default=argparse.SUPPRESS, help='Max unroll [default = 10]')
parser.add_argument('--minUnroll', type=int, default=argparse.SUPPRESS, help='Min unroll [default = 2]')
parser.add_argument('-augj', '--augmentJitter', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction)
parser.add_argument('-j', '--jitterAmount', type=float, default=argparse.SUPPRESS, help='Jitter amount [default = 0.1]')
parser.add_argument('-augr', '--augmentAngle', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction)
parser.add_argument('-adjust', '--adjustForFrameDistance', type = bool, default = True, action=argparse.BooleanOptionalAction)
parser.add_argument('-netArch', '--network', type=str, default=argparse.SUPPRESS, help='Network architecture [default = "default"]')
parser.add_argument('-norm', '--normalized', type=bool, default=argparse.SUPPRESS, action=argparse.BooleanOptionalAction)
parser.add_argument('--fluidFeatures', type=str, default=argparse.SUPPRESS, help='Features [default = "ones"]')
parser.add_argument('--boundaryFeatures', type=str, default=argparse.SUPPRESS, help='Features [default = "ones"]')


parser.add_argument('--optimizer', type=str, default=argparse.SUPPRESS, help='Optimizer [default = "adam"]')
parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS, help='Momentum [default = 0.9]')
parser.add_argument('--numNeighbors', type=float, default=argparse.SUPPRESS, help='Number of neighbors [default = 32]')

parser.add_argument('--groundTruth', type=str, default=argparse.SUPPRESS, help='Targets [default = "compute_density"]')
parser.add_argument('--gtMode', type=str, default=argparse.SUPPRESS, help='Ground Truth mode [default = "abs"]')
parser.add_argument('--biasActive', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Output bias [default = False]')
parser.add_argument('--optimizeWeights', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Optimized weights [default = False]')
parser.add_argument('--exponentialDecay', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Exponential decay [default = True]')
parser.add_argument('--initializer', type = str, default = argparse.SUPPRESS, help='Initializer [default = "uniform"]')
parser.add_argument('--loss', type = str, default = argparse.SUPPRESS, help='Loss function [default = "mse"]')
parser.add_argument('--historyLength', type = int, default = argparse.SUPPRESS, help='History length [default = 1]')

parser.add_argument('--unrollIncrement', type = int, default = argparse.SUPPRESS, help='Unroll increment [default = 100]')

parser.add_argument('--cfg', type = str, default = "", help='Config file [default = ""]')

parser.add_argument('--inputEncoder', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Input encoder [default = False]')
parser.add_argument('--inputEdgeEncoder', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Input encoder [default = False]')
parser.add_argument('--inputBasisEncoder', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Input encoder [default = False]')
parser.add_argument('--outputDecoder', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Input encoder [default = False]')
parser.add_argument('--edgeMLP', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Edge MLP [default = False]')
parser.add_argument('--vertexMLP', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Vertex MLP [default = False]')
parser.add_argument('--fcLayer', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Vertex MLP [default = False]')
parser.add_argument('--firstLayerMode', type = str, default = argparse.SUPPRESS, help='First layer mode [default = "stack"]')

parser.add_argument('--independent_dxdt', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='dxdt predicts velocity as well')
parser.add_argument('--networkType', type = str, default = argparse.SUPPRESS, help='Shifting the loop')
parser.add_argument('--normalization', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Shifting the loop')

parser.add_argument('--lossTerms', type = str, default = argparse.SUPPRESS, help='Loss terms')

parser.add_argument('--dataIndex', type = str, default = argparse.SUPPRESS, help='Data override')
# dxdtLossScaling
parser.add_argument('--dxdtLossScaling', type = float, default = argparse.SUPPRESS, help='Data override')
parser.add_argument('--shiftLoss', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Shifting the loop')
parser.add_argument('--skipLastShift', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Shifting the loop')
parser.add_argument('--scaleShiftLoss', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Shifting the loop')
parser.add_argument('--integrationScheme', type = str, default = argparse.SUPPRESS, help='Integration scheme')

parser.add_argument('--shiftCFL', type = float, default = argparse.SUPPRESS, help='Integration scheme')
parser.add_argument('--shiftIters', type = int, default = argparse.SUPPRESS, help='Integration scheme')
parser.add_argument('--shiftLossScaling', type = float, default = argparse.SUPPRESS, help='Integration scheme')

parser.add_argument('--exportPath', type = str, default = argparse.SUPPRESS, help='Export path')

parser.add_argument('--velocityNoise', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Velocity noise')
parser.add_argument('--velocityNoiseMagnitude', type = float, default = argparse.SUPPRESS, help='Velocity noise magnitude')
parser.add_argument('--velocityNoiseScaling', type = float, default = argparse.SUPPRESS, help='Velocity noise scaling')

parser.add_argument('--positionNoise', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Position noise')
parser.add_argument('--positionNoiseMagnitude', type = float, default = argparse.SUPPRESS, help='Position noise magnitude')

parser.add_argument('--unrollVelocityNoise', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Unroll velocity noise')
parser.add_argument('--unrollPositionNoise', type = bool, default = argparse.SUPPRESS, action=argparse.BooleanOptionalAction, help='Unroll position noise')

parser.add_argument('--mlpLayers', type = int, default = argparse.SUPPRESS, help='MLP layers')
parser.add_argument('--mlpWidth', type = int, default = argparse.SUPPRESS, help='MLP width')
parser.add_argument('--outputScaling', type = float, default = argparse.SUPPRESS, help='Output scaling')