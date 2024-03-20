# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# from tqdm import trange, tqdm
# import yaml
# import warnings
# warnings.filterwarnings(action='once')
# from datetime import datetime

# import torch
# from torch_geometric.nn import radius
# from torch_geometric.nn import SplineConv, fps, global_mean_pool, radius_graph, radius
# from torch_scatter import scatter

# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets

# from itertools import groupby
# import h5py

# from datautils import *
# from sphUtils import *


# def getLoss(prediction, config, state):
#     realParticles = state['ghostIndices'] == -1
    
#     expanded = prediction[state['UID'][state['ghostIndices'][state['ghostIndices'] != -1]]]
#     prediction = torch.vstack((prediction, expanded)) * config['support'] * config['dt']

#     ballisticPosition = state['fluidPosition'] + state['fluidVelocity'] * config['dt']

#     predictedPosition = ballisticPosition[realParticles] + prediction[realParticles]
#     actualPosition = state['positionAfterStep'][realParticles]

#     loss = torch.linalg.norm(predictedPosition - actualPosition, axis = 1)

#     return prediction, loss


# def getRealLoss(sceneIdx, frameIdx, config, state, data = 'fluidPosition'):
#     config_1, state_1 = loadFrame(sceneIdx, frameIdx)

#     realAdvanced = state_1['ghostIndices'] == -1
#     realAdvancedPositions = state_1[data][realAdvanced]

#     predicetedAdvanced = state['ghostIndices'] == -1
#     predictedAdvancedPositions = state['fluidPosition'][predicetedAdvanced]

#     vmax = torch.tensor(config['domain']['virtualMax']).to(config['device']).type(config['precision'])
#     vmin = torch.tensor(config['domain']['virtualMin']).to(config['device']).type(config['precision'])
#     offset = vmax - vmin
#     offsetx = torch.clone(offset)
#     offsety = torch.clone(offset)
#     offsetx[1] = 0
#     offsety[0] = 0

#     l_pp = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions + offsetx + offsety, axis = 1)
#     l_pc = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions + offsetx, axis = 1)
#     l_pn = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions + offsetx - offsety, axis = 1)

#     l_cp = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions + offsety, axis = 1)
#     l_cc = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions, axis = 1)
#     l_cn = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions - offsety, axis = 1)

#     l_np = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions - offsetx + offsety, axis = 1)
#     l_nc = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions - offsetx, axis = 1)
#     l_nn = torch.linalg.norm(realAdvancedPositions - predictedAdvancedPositions - offsetx - offsety, axis = 1)

#     loss_p = torch.min(torch.min(l_pp, l_pc), l_pn)
#     loss_c = torch.min(torch.min(l_cp, l_cc), l_cn)
#     loss_n = torch.min(torch.min(l_np, l_nc), l_nn)

#     loss = torch.min(torch.min(loss_p, loss_c), loss_n)
#     return loss

