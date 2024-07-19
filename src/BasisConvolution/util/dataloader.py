import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py 
from BasisConvolution.util.datautils import parseFile
import os

class datasetLoader(Dataset):
    def __init__(self, data):
        self.frameCounts = [len(s['samples']) for s in data]
        self.fileNames = [s['fileName'] for s in data]
        
        self.indices = [s['samples'] for s in data]
        self.fileFormat = [s['style'] for s in data][0]
        self.data = data
        #self.counters = [indices[1] for s, indices in data]
        
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
                return self.fileNames[i], self.indices[i][idx - p], self.data[i], i, idx - p
        

                return (i, self.indices[i][idx - p]), (i, self.indices[i][idx-p])
#                 return torch.rand(10,1), 2
            p = cs[i]
        return None, None
    
def processFolder(hyperParameterDict, folder):
    folder = os.path.expanduser(folder)
    simulationFiles = sorted([folder + '/' + f for f in os.listdir(folder) if f.endswith('.hdf5')])

    data = []
    for s in simulationFiles:
        inFile = h5py.File(s, 'r')
        data.append(parseFile(inFile, hyperParameterDict))
        inFile.close()
    return data