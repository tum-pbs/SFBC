import os 
# %run -i customLayer.py
# %matplotlib notebook
import time
import struct
# plotFrame(frame)
import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.profiler import profile, record_function, ProfilerActivity

class compressedFluidDataset(Dataset):
    def __init__(self, basePath, train = True, test = False, splits = [0.7, 0.9], split = True, cutoff = -1):
        basePath = os.path.expanduser(basePath)

        self.sceneFiles = [basePath + '/' + f for f in os.listdir(basePath) if f.endswith('.hdf5')]
        
        self.priorScene = -1
        
        self.frameCounts = []
        self.indices = []
        
        for s in self.sceneFiles:
            inFile = h5py.File(s, 'r')
            frameCount = int(len(inFile.keys()) -1) # adjust for bptcls
            inFile.close()
            
            if cutoff > 0:
                frameCount = min(frameCount, cutoff)
#             print(frameCount)
            if split:
                if train or (not train and not test):
                    trainCount = int(frameCount * splits[0])
                    splitCount = int(frameCount * splits[1])

                    indices = np.arange(0, splitCount)
                    perm = torch.randperm(splitCount, generator = torch.Generator().manual_seed(42))
                    indices = indices[perm.detach().numpy()]
                    # print(indices)
                    if train:
                        # print('train: ', indices[:trainCount])
                        self.indices.append(np.sort(indices[:trainCount]))
                        # print(self.indices)
                        self.frameCounts.append(indices[:trainCount].shape[0])
#                         print(self.indices)
                    else:
                        # print('valid: ', indices[trainCount:])
                        self.indices.append(np.sort(indices[trainCount:]))
                        # print(self.indices)
                        self.frameCounts.append(indices[trainCount:].shape[0])
                else:
                    splitCount = int(frameCount * splits[1])
                    indices = np.arange(splitCount, frameCount)
                    self.indices.append(indices)
                    self.frameCounts.append(indices.shape[0])

            else:
                indices = np.arange(0, frameCount)
                self.indices.append(indices)
                self.frameCounts.append(indices.shape[0])
#                 print(indices)
            
            
#             self.frameCounts.append(frameCount)
        self.frameCounts = np.array(self.frameCounts)
#         print('Prepared dataset from ', path, 'with scenes:\n', list(zip(self.scenes, self.frameCounts)))
#         print('radius %g, support %g, dt %g'%(self.radius, self.support, self.dt))


    def __len__(self):
#         print('len', np.sum(self.frameCounts))
        return np.sum(self.frameCounts)
    
    def prepData(self, sceneIdx, frameIdx):
        inFile = h5py.File(self.sceneFiles[sceneIdx])
        
        grp = inFile['%04d' % (1 + frameIdx)]
        
#         print(grp)
#         print(grp['position'])
        
        cached = {
            'position' : torch.from_numpy(grp['position'][:]),
            # 'features' : torch.from_numpy(grp['features'][:]),
            'outPosition' : torch.from_numpy(grp['outPosition'][:]),
            'velocity' : torch.from_numpy(grp['velocity'][:]),
            'area' : torch.from_numpy(grp['area'][:]),
            'density' : torch.from_numpy(grp['density'][:]),
            'ghostIndices' : torch.from_numpy(grp['ghostIndices'][:]),
            'finalPosition' : torch.from_numpy(grp['finalPosition'][:]),

            # 'dx' : torch.from_numpy(grp['dx'][:]),
            'support': inFile.attrs['support'],
            'dt': inFile.attrs['dt'],
            'radius': inFile.attrs['radius'],
                 }
        if('UID' in grp and not 'boundaryIntegral' in grp):
            cached = {                
                'position' : torch.from_numpy(grp['position'][:]),
                # 'features' : torch.from_numpy(grp['features'][:]),
                'outPosition' : torch.from_numpy(grp['outPosition'][:]),
                'velocity' : torch.from_numpy(grp['velocity'][:]),
                'area' : torch.from_numpy(grp['area'][:]),
                'density' : torch.from_numpy(grp['density'][:]),
                'ghostIndices' : torch.from_numpy(grp['ghostIndices'][:]),
            'finalPosition' : torch.from_numpy(grp['finalPosition'][:]),
                'UID' : torch.from_numpy(grp['UID'][:]),
                # 'dx' : torch.from_numpy(grp['dx'][:]),
                'support': inFile.attrs['support'],
                'dt': inFile.attrs['dt'],
                'radius': inFile.attrs['radius'],
                    }
        if('UID' in grp and 'boundaryIntegral' in grp):
            cached = {                
                'position' : torch.from_numpy(grp['position'][:]),
                # 'features' : torch.from_numpy(grp['features'][:]),
                'outPosition' : torch.from_numpy(grp['outPosition'][:]),
                'velocity' : torch.from_numpy(grp['velocity'][:]),
                'area' : torch.from_numpy(grp['area'][:]),
                'density' : torch.from_numpy(grp['density'][:]),
                'ghostIndices' : torch.from_numpy(grp['ghostIndices'][:]),
            'finalPosition' : torch.from_numpy(grp['finalPosition'][:]),
                'UID' : torch.from_numpy(grp['UID'][:]),
                'boundaryIntegral' : torch.from_numpy(grp['boundaryIntegral'][:]),
                'boundaryGradient' : torch.from_numpy(grp['boundaryGradient'][:]),
                'support': inFile.attrs['support'],
                'dt': inFile.attrs['dt'],
                'radius': inFile.attrs['radius'],
                    }
        inFile.close()
        return cached
    
    def __getitem__(self, idx):
#         print(idx , ' / ', np.sum(self.frameCounts))
        cs = np.cumsum(self.frameCounts)
        p = 0
        for i in range(cs.shape[0]):
#             print(p, idx, cs[i])
            if idx < cs[i] and idx >= p:
#                 print('Found index ', idx, 'in dataset ', i)
#                 print('Loading frame ', self.indices[i][idx - p], ' from dataset ', i, ' for ', idx, p)
                return self.prepData(i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None



dataCache = {}

def prepareData(batch, dataset, device):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with record_function("prepare Data"):
        if len(batch) == 1:
            if dataset in dataCache:
                ds = dataCache[dataset]
                if batch[0] in ds:
                    # print(' reading %d from cache' % batch[0])
                    positions, features, persistent_output, ghostIndices, batches, persistent_batches, gt, support, indices = ds[batch[0]]
                    return positions.to(device), features.to(device), persistent_output.to(device), ghostIndices.to(device), batches.to(device), persistent_batches.to(device), gt.to(device), support, indices


        positions = []
        features = []
        output = []
        ghostIndices = []
        gt = []
        batches = []

        persistent_positions = []
        persistent_features = []
        persistent_output = []
        persistent_ghosts = []
        persistent_gt = []
        persistent_batches = []

        transient_positions = []
        transient_features  = []
        transient_output = []
        transient_ghosts = []
        transient_gt = []
        transient_batches = []
        offsetCounter = 0
        support = 1.0

        bLoss = []
        bLosses = []
        indices = []
        for ib, b in enumerate(batch):
            data, actualIndex = dataset[b]
            indices.append(actualIndex)
            # print('features', data['features'])
            # print('position', data['position'][:,:2])
            # print('velocity', data['velocity'])
            # print('finalPosition', data['finalPosition'])
            dt = data['dt']
            groundTruth = (data['finalPosition'] - data['position'][:,:2]) / data['support'] * 50

            if 'boundaryIntegral' in data:
                featureVec = torch.hstack((data['area'].reshape(data['area'].shape[0],1), data['velocity'], data['boundaryIntegral'].reshape(data['area'].shape[0],1), data['boundaryGradient']))
            else:
                featureVec = torch.hstack((data['area'], data['velocity']))
            # featureVec[:,0] = data['radius'] * data['radius'] * np.pi            

            ghosts = data['ghostIndices']
            persistent_positions.append(data['position'][ghosts == -1])
            persistent_features.append(featureVec[ghosts == -1])
            persistent_output.append(data['outPosition'])
            persistent_ghosts.append(data['ghostIndices'][ghosts == -1])
            persistent_gt.append(groundTruth[ghosts == -1])
            persistent_batches.append(torch.ones(data['density'][ghosts == -1].shape[0]) *ib)

            transient_positions.append(data['position'][ghosts != -1])
            transient_features.append(featureVec[ghosts != -1])
            g = data['ghostIndices'][ghosts != -1]
            g += offsetCounter
            offsetCounter += ghosts[ghosts == -1].shape[0]
            transient_ghosts.append(g)
            transient_gt.append(groundTruth[ghosts != -1])
            transient_batches.append(torch.ones(data['density'][ghosts != -1].shape[0]) * ib)

            support = data['support']


        persistent_output = torch.cat(persistent_output)
        persistent_batches = torch.cat(persistent_batches)
        positions = torch.cat([torch.cat(persistent_positions), torch.cat(transient_positions)])
        features = torch.cat([torch.cat(persistent_features), torch.cat(transient_features)])
        # features[:,:] = 1
        ghostIndices = torch.cat([torch.cat(persistent_ghosts), torch.cat(transient_ghosts)])
        gt = torch.cat([torch.cat(persistent_gt), torch.cat(transient_gt)])
        batches = torch.cat([persistent_batches, torch.cat(transient_batches)])


        if len(batch) == 1:
            if dataset not in dataCache:
                dataCache[dataset] = {}
            dataCache[dataset][batch[0]] = [positions, features, persistent_output, ghostIndices, batches, persistent_batches, gt, support, indices]

        positions = positions.to(device)
        features = features.to(device)
        persistent_output = persistent_output.to(device)
        ghostIndices = ghostIndices.to(device)
        batches = batches.to(device)
        persistent_batches = persistent_batches.to(device)

        gt = gt.to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        

        return positions, features, persistent_output, ghostIndices, batches, persistent_batches, gt, support, indices