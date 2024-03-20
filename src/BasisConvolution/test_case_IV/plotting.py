import torch

def plotSlices(fig, ax, cm, norm, func, z = -0.9, nPoints = 32):
    xx, yy = torch.meshgrid(torch.linspace(-1,z,nPoints), torch.linspace(-1,-z,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx,yy, torch.ones_like(xx) * z, facecolors = cm(norm(func(torch.stack((xx,yy,torch.ones_like(xx) * z), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(-1,z,nPoints), torch.linspace(-z,1,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx,yy, torch.ones_like(xx) * z, facecolors = cm(norm(func(torch.stack((xx,yy,torch.ones_like(xx) * z), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(z,1,nPoints), torch.linspace(-1,-z,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx,yy, torch.ones_like(xx) * z, facecolors = cm(norm(func(torch.stack((xx,yy,torch.ones_like(xx) * z), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(z,1,nPoints), torch.linspace(-z,1,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx,yy, torch.ones_like(xx) * z, facecolors = cm(norm(func(torch.stack((xx,yy,torch.ones_like(xx) * z), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)    
    
    xx, yy = torch.meshgrid(torch.linspace(-1,z,nPoints), torch.linspace(-1,z,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx, torch.ones_like(xx) * -z, yy, facecolors = cm(norm(func(torch.stack((xx,torch.ones_like(xx) * -z, yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(-1,z,nPoints), torch.linspace(z,1,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx, torch.ones_like(xx) * -z, yy, facecolors = cm(norm(func(torch.stack((xx,torch.ones_like(xx) * -z, yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(z,1,nPoints), torch.linspace(-1,z,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx, torch.ones_like(xx) * -z, yy, facecolors = cm(norm(func(torch.stack((xx,torch.ones_like(xx) * -z, yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(z,1,nPoints), torch.linspace(z,1,nPoints), indexing='ij')    
    sc = ax.plot_surface(xx, torch.ones_like(xx) * -z, yy, facecolors = cm(norm(func(torch.stack((xx,torch.ones_like(xx) * -z, yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)    
    
    xx, yy = torch.meshgrid(torch.linspace(-1,-z,nPoints), torch.linspace(-1,z,nPoints), indexing='ij')    
    sc = ax.plot_surface(torch.ones_like(xx) * z,xx,yy, facecolors = cm(norm(func(torch.stack((torch.ones_like(xx) * z,xx,yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(-1,-z,nPoints), torch.linspace(z,1,nPoints), indexing='ij')    
    sc = ax.plot_surface(torch.ones_like(xx) * z,xx,yy, facecolors = cm(norm(func(torch.stack((torch.ones_like(xx) * z,xx,yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(-z,1,nPoints), torch.linspace(-1,z,nPoints), indexing='ij')    
    sc = ax.plot_surface(torch.ones_like(xx) * z,xx,yy, facecolors = cm(norm(func(torch.stack((torch.ones_like(xx) * z,xx,yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    xx, yy = torch.meshgrid(torch.linspace(-z,1,nPoints), torch.linspace(z,1,nPoints), indexing='ij')    
    sc = ax.plot_surface(torch.ones_like(xx) * z,xx,yy, facecolors = cm(norm(func(torch.stack((torch.ones_like(xx) * z,xx,yy), axis = -1).flatten().reshape(-1,3)))).reshape(nPoints,nPoints,4), rstride=1, cstride=1, shade=False)
    return sc

# Code to plot 3D data
# def func(_planePositions):
#     planePositions = _planePositions.type(torch.float32)
#     fp, ff, rij, dist = periodicNeighborSearchXYZ(gridPositions.type(torch.float32) + jitter.type(torch.float32), planePositions, minDomain, maxDomain, support, True, True )
    
# #     ff, fp = radius(planePositions, gridPositions.type(torch.float32) + jitter.type(torch.float32), r = support, max_num_neighbors = 256, batch_x = None, batch_y = None)
# #     _, nf = torch.unique(fp, return_counts = True)
    
# #     rij = (gridPositions[ff] - planePositions[fp]) / support
# #     dist = torch.linalg.norm(rij, axis = -1)
#     rho = scatter_sum(vols[ff] * wendland(dist, support), fp, dim = 0, dim_size = planePositions.shape[0])
    
# #     return planePositions[:,0] * planePositions[:,1]
#     return rho


# x = generatedData['x']
# vols = generatedData['vols']

# fig = plt.figure(figsize=(12,5))
# ax = fig.add_subplot(1, 3, 1, projection='3d')
# sc = ax.scatter(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], c = generatedData['rho'], s = 32)
# ax.set_box_aspect([1,1,1]) 

# ax = fig.add_subplot(1, 3, 2, projection='3d')
# ax.quiver(generatedData['x'][:,0], generatedData['x'][:,1], generatedData['x'][:,2], generatedData['gradRhoSymmetric'][:,0], generatedData['gradRhoSymmetric'][:,1], generatedData['gradRhoSymmetric'][:,2], normalize = True, length = 0.1)
# ax.set_box_aspect([1,1,1]) 
# fig.colorbar(sc, ax=ax)


# norm = colors.Normalize(vmin = torch.min(generatedData['rho']), vmax = torch.max(generatedData['rho']))
# # fig = plt.figure(figsize=(8,5))
# ax = fig.add_subplot(1, 3, 3, projection='3d')
# nPoints  = 32
# sc = plotSlices(fig, ax, cm = cm,norm = norm, func = func, z = -0.5, nPoints = nPoints)
# fig.colorbar(mpl.cm.ScalarMappable(cmap = cm, norm = norm), ax=ax)
# # fig.tight_layout()

# fig.tight_layout()

# fig, axis = plt.subplots(3, 3, figsize=(12,6), sharex = False, sharey = False, squeeze = False)
# sns.kdeplot(data = generatedData['gradRhoNaive'][:,0].numpy(), ax = axis[0,0])
# sns.kdeplot(data = generatedData['gradRhoNaive'][:,1].numpy(), ax = axis[0,1])
# sns.kdeplot(data = generatedData['gradRhoNaive'][:,2].numpy(), ax = axis[0,2])

# sns.kdeplot(data = generatedData['gradRhoDifference'][:,0].numpy(), ax = axis[1,0])
# sns.kdeplot(data = generatedData['gradRhoDifference'][:,1].numpy(), ax = axis[1,1])
# sns.kdeplot(data = generatedData['gradRhoDifference'][:,2].numpy(), ax = axis[1,2])

# sns.kdeplot(data = generatedData['gradRhoSymmetric'][:,0].numpy(), ax = axis[2,0])
# sns.kdeplot(data = generatedData['gradRhoSymmetric'][:,1].numpy(), ax = axis[2,1])
# sns.kdeplot(data = generatedData['gradRhoSymmetric'][:,2].numpy(), ax = axis[2,2])

# fig.tight_layout()