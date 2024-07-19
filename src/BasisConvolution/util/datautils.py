import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))



import torch
import torch
import numpy as np
import h5py
from itertools import groupby


from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.profiler import profile, record_function, ProfilerActivity


# from BasisConvolution.test_case_II.datautils import splitFile

def getSamples(frames, maxRollOut = 8, chunked = False, trainValidationSplit = 0.8, limitRollOut = False):
    if chunked:
        validationSamples = int(frames * (1 - trainValidationSplit))
        validationSamples = validationSamples - (validationSamples % maxRollOut)
        trainingSamples = frames - validationSamples

        chunks = validationSamples // maxRollOut


    #     for i in range(32):
        marker = np.ones(frames)
        for j in range(chunks):
            while True:
                i = np.random.randint(maxRollOut, frames - maxRollOut)
                if np.any(marker[i:i+maxRollOut] == 0):
                    continue
                marker[i:i+maxRollOut] = 0
                break

        count_dups = [sum(1 for _ in group) for _, group in groupby(marker.tolist())]
        counter = np.zeros(frames, dtype=np.int32)
        cs = np.cumsum(count_dups)
        prev = 1
        k = 0
        for j in range(frames):
            if prev != marker[j]:
                k = k + 1
            counter[j] = np.clip(cs[k] - j,0, maxRollOut)
            if marker[j] == 0:
                counter[j] = -counter[j]
            prev = marker[j]

    #         markers.append(counter)

    #     markers = np.array(markers)
    else:
        validationSamples = int(frames * (1 - trainValidationSplit))
        trainingSamples = frames - validationSamples


    #     for i in range(32):
        marker = np.zeros(frames)
        marker[np.random.choice(frames, trainingSamples, replace = False)] = 1
    #         print(np.random.choice(frames, trainingSamples, replace = False))

        count_dups = [sum(1 for _ in group) for _, group in groupby(marker.tolist())]
        counter = np.zeros(frames, dtype=np.int32)
        cs = np.cumsum(count_dups)
        prev = marker[0]
        k = 0
        for j in range(frames):
            if prev != marker[j]:
                k = k + 1
            counter[j] = np.clip(cs[k] - j,0, maxRollOut)
            if marker[j] == 0:
                counter[j] = -counter[j]
            prev = marker[j]

    #         markers.append(counter)

    #     markers = np.array(markers)
    trainingFrames = np.arange(frames)[counter > 0]
    validationFrames = np.arange(frames)[counter < 0]
    
    if limitRollOut:
        maxIdx = counter.shape[0] - maxRollOut + 1
        c = counter[:maxIdx][np.abs(counter[:maxIdx]) != maxRollOut]
        c = c / np.abs(c) * 8
        counter[:maxIdx][np.abs(counter[:maxIdx]) != maxRollOut] = c
        
    
    return trainingFrames, validationFrames, counter


def splitFile(s, skip = 32, cutoff = 300, chunked = True, maxRollOut = 8, limitRollOut = False, distance = 1):
    inFile = h5py.File(s, 'r')
    frameCount = int(len(inFile['simulationExport'].keys()) -1) // distance # adjust for bptcls
    inFile.close()
    if cutoff > 0:
        frameCount = min(cutoff+skip, frameCount)
    if cutoff < 0:
        frameCount = frameCount + cutoff - 1
    # print(frameCount)
    # frameCount -= 100
    actualCount = frameCount - 1 - skip
    
    # print(frameCount, cutoff, actualCount)
    training, _, counter = getSamples(actualCount, maxRollOut = maxRollOut, chunked = chunked, trainValidationSplit = 1.)
    return s, training + skip, counter



def loadDataset(trainingFiles, limitData = 0, frameDistance = 16, maxUnroll = 10, adjustForFrameDistance = True, verbose = False):
    # basePath = os.path.expanduser(path)
    # trainingFiles = [basePath + f for f in os.listdir(basePath) if f.endswith('.hdf5')]

    training = []
    validation = []
    testing = []

    
    if limitData > 0:
        files = []
        for i in range(max(len(trainingFiles), limitData)):
            files.append(trainingFiles[i])
        simulationFiles = files
    # simulationFiles = [simulationFiles[0]]
    if verbose:
        print('Input files:')
        for i, c in enumerate(trainingFiles):
            print('\t', i ,c)

    training = []
    validation = []
    testing = []

    for s in trainingFiles:
        f, s, u = splitFile(s, split = False, cutoff = -frameDistance * maxUnroll, skip = frameDistance if adjustForFrameDistance else 0)
        training.append((f, (s,u)))
    # for s in tqdm(validationFiles):
    #     f, s, u = splitFile(s, split = False, cutoff = -4, skip = 0)
    #     validation.append((f, (s,u)))

    if verbose:
        print('Processed data into datasets:')
        debugPrint(training)
    return training, trainingFiles


from torch.utils.data import DataLoader


from torch.utils.data import Dataset
class datasetLoader(Dataset):
    def __init__(self, data):
        self.frameCounts = [indices[0].shape[0] for s, indices in data]
        self.fileNames = [s for s, indices in data]
        
        self.indices = [indices[0] for s, indices in data]
        self.counters = [indices[1] for s, indices in data]
        
#         print(frameCounts)
        
        
    def __len__(self):
#         print('len', np.sum(self.frameCounts))
        return np.sum(self.frameCounts)
    
    def __getitem__(self, idx):
#         print(idx , ' / ', np.sum(self.frameCounts))
        cs = np.cumsum(self.frameCounts)
        p = 0
        for i in range(cs.shape[0]):
#             print(p, idx, cs[i])
            if idx < cs[i] and idx >= p:
#                 print('Found index ', idx, 'in dataset ', i)
#                 print('Loading frame ', self.indices[i][idx - p], ' from dataset ', i, ' for ', idx, p)
                return self.fileNames[i], self.indices[i][idx - p], self.counters[i][idx-p]
        

                return (i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None
    
def getDataLoader(data, batch_size, shuffle = True, verbose = False):
    if verbose:
        print('Setting up data loaders')
    train_ds = datasetLoader(data)
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size = batch_size).batch_sampler
    return train_ds, train_dataloader

def parseDataset(hyperParameterDict, files):   
    training, trainingFiles = loadDataset(files, limitData = hyperParameterDict['dataLimit'], frameDistance = hyperParameterDict['frameDistance'], maxUnroll = hyperParameterDict['maxUnroll'], adjustForFrameDistance = hyperParameterDict['adjustForFrameDistance'], verbose = hyperParameterDict['verbose'])
    train_ds, train_dataloader = getDataLoader(training, hyperParameterDict['batch_size'], verbose = hyperParameterDict['verbose'])

    return training, train_ds, train_dataloader, iter(train_dataloader)

def isTemporalData(inFile):
    if 'simulationExport' in inFile:
        return True
    if 'simulationData' in inFile:
        if 'fluidPosition' in inFile['simulationData']:
            return True    
    return False

def getFrameCount(inFile):
    if 'simulationExport' in inFile:
        return int(len(inFile['simulationExport'].keys()) -1)
    if 'simulationData' in inFile:
        if 'fluidPosition' in inFile['simulationData']:
            return inFile['simulationData']['fluidPosition'].shape[0] - 1
        else:
            return int(len(inFile['simulationData'].keys()))
    raise ValueError('Could not parse file')

def getFrames(inFile):
    if 'simulationExport' in inFile:
        return [int(i) for i in inFile['simulationExport'].keys()], list(inFile['simulationExport'].keys())
    if 'simulationData' in inFile:
        if 'fluidPosition' in inFile['simulationData']:
            return np.arange(inFile['simulationData']['fluidPosition'].shape[0]).tolist(), np.arange(inFile['simulationData']['fluidPosition'].shape[0]).tolist()
        else:
            return [int(i) for i in inFile['simulationData'].keys()], list(inFile['simulationData'].keys())
    raise ValueError('Could not parse file')


import warnings
def getSamples(inFile, frameSpacing = 1, frameDistance = 1, maxRollout = 0, skip = 0, limit = 0):
    temporalData = isTemporalData(inFile)
    frameCount = getFrameCount(inFile)
    frames, samples = getFrames(inFile)

    if maxRollout > 0 and not temporalData:
        raise ValueError('Max rollout only supported for temporal data')
    if frameCount < maxRollout:
        raise ValueError('Max rollout larger than frame count')
    
    if limit > 0:
        if limit > len(frames):
            warnings.warn(f'Limit larger than frame count for {inFile.filename}')
        frames = frames[:min(len(frames), limit)]
        samples = samples[:min(len(samples), limit)]
    if skip > 0:
        if skip > len(frames):
            warnings.warn(f'Skip larger than frame count for {inFile.filename}')
        frames = frames[min(len(frames), skip):]
        samples = samples[min(len(samples), skip):]

    if not temporalData or maxRollout == 0:
        return frames[::frameSpacing], samples[::frameSpacing]
    else:
        lastPossible = len(frames) - 1 - maxRollout * frameDistance
        if lastPossible < 0:
            raise ValueError('Frame count too low for max rollout')
        return frames[:lastPossible:frameSpacing], samples[:lastPossible:frameSpacing]

def getStyle(inFile):
    try:
        if 'simulationExport' in inFile:
            if 'config' in inFile: # New format
                return 'newFormat'
            if 'config' not in inFile:
                if isTemporalData(inFile): # temporal old format data, test case II/III
                    return 'testcase_II'
                else:
                    raise ValueError('Unsupported Format for file')
        else:
            # This should be test case I with flat 1D data
            if isTemporalData(inFile):
                return 'testcase_I'
            else:
                return 'testcase_IV'
    except Exception as e:
        print('Unable to load frame (unknown format)')
        raise e

def parseFile(inFile, hyperParameterDict):
    frameDistance = hyperParameterDict['frameDistance'] if 'frameDistance' in hyperParameterDict else 1
    frameSpacing = hyperParameterDict['dataDistance'] if 'dataDistance' in hyperParameterDict else 1
    maxRollout = hyperParameterDict['maxRollOut'] if 'maxRollOut' in hyperParameterDict else 0

    temporalData = isTemporalData(inFile)
    skip = (1 if hyperParameterDict['zeroOffset'] and temporalData else 0) if 'zeroOffset' in hyperParameterDict else 0
    limit = hyperParameterDict['dataLimit'] if 'dataLimit' in hyperParameterDict else 0


    frames, samples = getSamples(inFile, frameSpacing = frameSpacing, frameDistance = frameDistance, maxRollout = maxRollout, skip = skip, limit = limit)

    data= {'fileName': inFile.filename, 'frames': frames, 'samples': samples, 'frameDistance': frameDistance, 'frameSpacing': frameSpacing, 'maxRollout': maxRollout, 'skip': skip, 'limit': limit, 'isTemporalData': temporalData, 'style': getStyle(inFile)}
    # print(data)

    return data