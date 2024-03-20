# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
    
# sph related imports
# from BasisConvolution.oneDimensionalSPH.sph import *
# from BasisConvolution.oneDimensionalSPH.perlin import *
# neural network rlated imports
# from torch.optim import Adam
# from BasisConvolution.oneDimensionalSPH.rbfConv import *
# from torch_geometric.loader import DataLoader
# from BasisConvolution.oneDimensionalSPH.trainingHelper import *
# plotting/UI related imports
# from BasisConvolution.oneDimensionalSPH.plotting import *
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.style.use('dark_background')
cmap = mpl.colormaps['viridis']
# from tqdm.autonotebook import trange, tqdm
# from IPython.display import display, Latex
# from datetime import datetime
# from BasisConvolution.oneDimensionalSPH.rbfNet import *
# from BasisConvolution.convNet import RbfNet
from BasisConvolution.detail.util import count_parameters
# import h5py
# import matplotlib.colors as colors
import torch
import numpy as np

# import torch.nn as nn
# %matplotlib notebook

import math

import sklearn.metrics as metrics
import pandas
def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


import pandas
def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def getTestingLossFrame(modelState, testData, plot = False):
    label = '%s x %2d @ %s' % (modelState['basis'], modelState['n'], str(modelState['layers']))
    testPredictions = {}
    testGTs = {}
    testPositions = {}
    with torch.no_grad():
        for i, k in  enumerate(testData.keys()):
            gt = testData[k][9].reshape(len(testData[k][0]), testData[k][0][0].shape[0])
            prediction = modelState['model'](testData[k][5], testData[k][6], testData[k][7], testData[k][8]).reshape(len(testData[k][0]), testData[k][0][0].shape[0]).detach().cpu().numpy()
            testPredictions[k] = prediction
            testGTs[k] = gt.detach().cpu().numpy()
            testPositions[k] = testData[k][0]
    if plot:
        fig, axis = plt.subplots(len(testPredictions.keys()),3, figsize=(16,8), sharex = True, sharey = 'col', squeeze = False)

        for s, k in  enumerate(testData.keys()):
            norm = mpl.colors.Normalize(vmin=0, vmax=len(testPredictions[k]) - 1)
            for i, (xs, rhos) in enumerate(zip(testPositions[k], testGTs[k])):
                c = cmap(norm(i))
                axis[s,0].plot(xs.cpu().numpy(), rhos, ls = '-', c = c)
            for i, (xs, rhos) in enumerate(zip(testPositions[k], testPredictions[k])):
                c = cmap(norm(i))
                axis[s,1].plot(xs.cpu().numpy(), rhos, ls = '-', c = c)
            for i, (xs, pred, gt) in enumerate(zip(testPositions[k], testPredictions[k], testGTs[k])):
                c = cmap(norm(i))
                axis[s,2].plot(xs.cpu().numpy(), pred- gt, ls = '-', c = c)
        axis[0,0].set_title('GT')
        axis[0,1].set_title('Pred')
        axis[0,2].set_title('Loss')
        fig.tight_layout()
        # axis[0,0].plot()
    lossDict = []
    for s, k in  enumerate(testData.keys()):
        lossTerms = []
        for i, (xs, pred, gt) in enumerate(zip(testPositions[k], testPredictions[k], testGTs[k])):
            loss = (pred - gt)**2
            r2 = metrics.r2_score(gt, pred)
            l2 = metrics.mean_squared_error(gt, pred)

            maxSignal = np.max(np.abs(gt))
    #         mse = np.mean((pred - gt)**2)
            psnr = 20 * math.log10(maxSignal / np.sqrt(l2))
    #         print(20 * math.log10(maxSignal / np.sqrt(mse)))
    #         print(maxSignal, mse)
    #         male = np.mean(np.abs(np.log(np.abs(pred) / np.abs(gt))[gt != 0]))
    #         rmsle = np.sqrt(np.mean(np.log(np.abs(pred) / np.abs(gt))[gt != 0]**2)
            minVal = np.min(loss)
            maxVal = np.max(loss)
            std = np.std(loss)
            q1, q3 = np.percentile(loss, [25, 75])
    #         print('r2', r2, 'l2', l2, 'psnr', l2, 'min', minVal, 'max', maxVal, 'q1', q1, 'q3', q3, 'std', std)
            lossTerms.append({'label': label, 'seed':modelState['seed'], 'window':modelState['window'], 'file':k, 'basis':modelState['basis'], 'n':modelState['n'],'entry':str(i), 'params': count_parameters(modelState['model']), 'depth':len(modelState['layers']), 'width':np.max(modelState['layers']),'r2':r2,'l2':l2,'psnr':psnr, 'min':minVal, 'max':maxVal, 'q1':q1, 'q3':q3, 'std':std})
    #         print(r2, l2, psnr)
    #         print(male, rmsle)
    #         break
        lossDict += lossTerms
    #     break
    return pandas.DataFrame(data = lossDict)