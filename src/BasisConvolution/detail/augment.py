import torch
import numpy as np

def augment(attributes, inputData, groundTruthData, angle, jitter):    
    
#     angle = np.pi / 2
    rot = torch.tensor([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]], device = inputData['fluidPosition'].device, dtype = inputData['fluidPosition'].dtype)
    
    rotinP = torch.matmul(rot.unsqueeze(0).repeat(inputData['fluidPosition'].shape[0],1,1), inputData['fluidPosition'].unsqueeze(2))[:,:,0] 
#     print(rotinP.shape)
    if jitter > 0:
        noise = torch.normal(torch.zeros_like(inputData['fluidPosition']), torch.ones_like(inputData['fluidPosition']) * jitter * attributes['support'])
#         print(noise)
        rotinP = rotinP + noise
#     print(rotinP.shape)
    rotinVel = torch.matmul(rot.unsqueeze(0).repeat(inputData['fluidPosition'].shape[0],1,1), inputData['fluidVelocity'].unsqueeze(2))[:,:,0]
    rotinGrav = torch.matmul(rot.unsqueeze(0).repeat(inputData['fluidPosition'].shape[0],1,1), inputData['fluidGravity'].unsqueeze(2))[:,:,0]
    rotBoundaryP = torch.matmul(rot.unsqueeze(0).repeat(inputData['boundaryPosition'].shape[0],1,1), inputData['boundaryPosition'].unsqueeze(2))[:,:,0]
    rotBoundaryVel = torch.matmul(rot.unsqueeze(0).repeat(inputData['boundaryVelocity'].shape[0],1,1), inputData['boundaryVelocity'].unsqueeze(2))[:,:,0]
    rotBoundaryNormal = torch.matmul(rot.inverse().mT.unsqueeze(0).repeat(inputData['boundaryPosition'].shape[0],1,1), inputData['boundaryNormal'].unsqueeze(2))[:,:,0]
#     print(rotinP.shape)
    rotatedData = {'fluidPosition' : rotinP,
                  'fluidVelocity': rotinVel,
                  'fluidArea': inputData['fluidArea'],
                  'fluidDensity': inputData['fluidDensity'],
                  'fluidSupport': inputData['fluidSupport'],
                  'fluidGravity': rotinGrav,
                  'boundaryPosition': rotBoundaryP,
                  'boundaryNormal': rotBoundaryNormal,
                  'boundaryArea': inputData['boundaryArea'],
                  'boundaryVelocity': rotBoundaryVel}
#     print(rotatedData.keys())
    rotatedGT = []
    for i in range(len(groundTruthData)):
        gtP = torch.matmul(rot.unsqueeze(0).repeat(groundTruthData[i].shape[0],1,1), groundTruthData[i][:,0:2].unsqueeze(2))[:,:,0]
        gtV = torch.matmul(rot.unsqueeze(0).repeat(groundTruthData[i].shape[0],1,1), groundTruthData[i][:,2:4].unsqueeze(2))[:,:,0]
        gtD = groundTruthData[i][:,-1].unsqueeze(-1)
        
#         print(gtP.shape)
#         print(gtV.shape)
#         print(gtD.shape)
        rotated = torch.hstack((\
                gtP,\
                gtV,\
                gtD))
        rotatedGT.append(rotated)
#         print(rotatedData.shape)
#         print(groundTruthData[i])
    
#     print(rotatedData.keys())
    return attributes, rotatedData, rotatedGT
