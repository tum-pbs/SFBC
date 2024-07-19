import torch
import copy
from BasisConvolution.sph.neighborhood import neighborSearch
from BasisConvolution.sph.sphOps import sphOperationStates, sphOperation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib

def mapToGrid(visualizationState, quantity):
    return sphOperation(
        (None, visualizationState['fluid']['masses']), 
        (None, visualizationState['fluid']['densities']), 
        (quantity, quantity), 
        (visualizationState['gridNeighborhood']['indices'][0], visualizationState['gridNeighborhood']['indices'][1]), visualizationState['gridNeighborhood']['kernels'], visualizationState['gridNeighborhood']['gradients'], 
        visualizationState['gridNeighborhood']['distances'], visualizationState['gridNeighborhood']['vectors'],
        visualizationState['gridNeighborhood']['supports'],   
        visualizationState['grid'].shape[0], operation = 'interpolate')


def prepVisualizationState(perennialState, config, nGrid = 128, fluidNeighborhood = True, grid = True):
    visualizationState = copy.deepcopy(perennialState)
    if fluidNeighborhood:
        _, visualizationState['fluid']['neighborhood'] = neighborSearch(visualizationState['fluid'],visualizationState['fluid'], config)
        if 'densities' not in visualizationState['fluid']:
            visualizationState['fluid']['densities'] = sphOperationStates(visualizationState['fluid'], visualizationState['fluid'], quantities = None, operation = 'density', neighborhood = visualizationState['fluid']['neighborhood'])
    # visualizationState['fluid']['masses'] = perennialState['fluid']['masses']

    # visualizationState['fluid']['velocities'] = perennialState['fluid']['velocities']
    x = perennialState['fluid']['positions']

    periodicity = config['domain']['periodicity']
    minD = config['domain']['minExtent']
    maxD = config['domain']['maxExtent']

    if periodicity[0] and not periodicity[1]:
        visualizationState['fluid']['positions'] = torch.stack((torch.remainder(x[:,0] - minD[0], maxD[0] - minD[0]) + minD[0], x[:,1]), dim = 1)
    elif not periodicity[0] and periodicity[1]:
        visualizationState['fluid']['positions'] = torch.stack((x[:,0], torch.remainder(x[:,1] - minD[1], maxD[1] - minD[1]) + minD[1]), dim = 1)
    elif periodicity[0] and periodicity[1]:
        visualizationState['fluid']['positions'] = torch.remainder(x - minD, maxD - minD) + minD
    else:
        visualizationState['fluid']['positions'] = x  

    if 'boundary' in perennialState and perennialState['boundary'] is not None:
        x = perennialState['boundary']['positions']
        if periodicity[0] and not periodicity[1]:
            visualizationState['boundary']['positions'] = torch.stack((torch.remainder(x[:,0] - minD[0], maxD[0] - minD[0]) + minD[0], x[:,1]), dim = 1)
        elif not periodicity[0] and periodicity[1]:
            visualizationState['boundary']['positions'] = torch.stack((x[:,0], torch.remainder(x[:,1] - minD[1], maxD[1] - minD[1]) + minD[1]), dim = 1)
        elif periodicity[0] and periodicity[1]:
            visualizationState['boundary']['positions'] = torch.remainder(x - minD, maxD - minD) + minD
        else:
            visualizationState['boundary']['positions'] = x

    # nGrid = 128
    xGrid = torch.linspace(config['domain']['minExtent'][0], config['domain']['maxExtent'][0], nGrid, dtype = perennialState['fluid']['positions'].dtype, device = perennialState['fluid']['positions'].device)
    yGrid = torch.linspace(config['domain']['minExtent'][1], config['domain']['maxExtent'][1], nGrid, dtype = perennialState['fluid']['positions'].dtype, device = perennialState['fluid']['positions'].device)
    X, Y = torch.meshgrid(xGrid, yGrid, indexing = 'xy')
    P = torch.stack([X,Y], dim=-1).flatten(0,1)
    if grid:
        visualizationState['grid'] = P
        visualizationState['X'] = X
        visualizationState['Y'] = Y
        visualizationState['nGrid'] = nGrid

        gridState = {
            'positions': P,	
            'numParticles': P.shape[0],    
            'supports': P.new_ones(P.shape[0]) * config['particle']['support'],        
        }
        gridConfig = {
            'domain': config['domain'],
            'simulation': {
                'supportScheme': 'scatter'
            },
            'neighborhood':{
                'algorithm': 'compact',
                'scheme': 'compact',
                'verletScale': 1.0
            },
            'kernel': config['kernel']
        }
        gridConfig['simulation']['supportScheme'] = 'scatter'
        # printState(gridState)
        _, visualizationState['gridNeighborhood'] = neighborSearch(gridState, visualizationState['fluid'], gridConfig) #0, perennialState['fluid']['supports'], getKernel('Wendland2'), config['domain']['dim'], config['domain']['periodicity'], config['domain']['minExtent'], config['domain']['maxExtent'], mode = 'scatter', algorithm ='compact')
        # visualizationState['gridNeighborhood'] = {}
        # visualizationState['gridNeighborhood']['indices'] = (i, j)
        # visualizationState['gridNeighborhood']['distances'] = rij
        # visualizationState['gridNeighborhood']['vectors'] = xij
        # visualizationState['gridNeighborhood']['kernels'] = Wij
        # visualizationState['gridNeighborhood']['gradients'] = gradWij
        # visualizationState['gridNeighborhood']['supports'] = hij

    return visualizationState

import matplotlib.patches as patches
def setPlotBaseAttributes(axis, config):
    domainMin = config['domain']['minExtent'].detach().cpu().numpy()
    domainMax = config['domain']['maxExtent'].detach().cpu().numpy()
    axis.set_xlim(domainMin[0], domainMax[0])
    axis.set_ylim(domainMin[1], domainMax[1])
    square = patches.Rectangle((domainMin[0], domainMin[1]), domainMax[0] - domainMin[0], domainMax[1] - domainMin[1], linewidth=1, edgecolor='b', facecolor='none',ls='--')
    axis.add_patch(square)
    axis.set_aspect('equal')
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    # axis.set_xticklabels([])
    # axis.set_yticklabels([])


from typing import Union, Tuple
def visualizeParticleQuantity(fig, axis, config, visualizationState, quantity: Union[str, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], mapping = '.x', cbar = True, cmap = 'viridis', scaling = 'lin', s = 4, linthresh = 0.5, midPoint = 0, gridVisualization = False, which = 'fluid', plotBoth = True, operation = None, streamLines = False, title = None):  
    inputQuantity = None
    pos_x = None

    if isinstance(quantity, str):
        if which == 'fluid' or not config['boundary']['active']:
            inputQuantity = visualizationState['fluid'][quantity]
            pos_x = visualizationState['fluid']['positions']
        elif which == 'boundary':
            inputQuantity = visualizationState['boundary'][quantity]
            pos_x = visualizationState['boundary']['positions']
        else:
            inputQuantity = torch.cat([visualizationState['fluid'][quantity], visualizationState['boundary'][quantity]], dim = 0)
            pos_x = torch.cat([visualizationState['fluid']['positions'], visualizationState['boundary']['positions']], dim = 0)
    else:
        if isinstance(quantity, tuple):
            if which == 'fluid' or not config['boundary']['active']:
                inputQuantity = quantity[0]
                pos_x = visualizationState['fluid']['positions']
            elif which == 'boundary':
                inputQuantity = quantity[1]
                pos_x = visualizationState['boundary']['positions']
            else:
                inputQuantity = torch.cat([quantity[0], quantity[1]], dim = 0)
                pos_x = torch.cat([visualizationState['fluid']['positions'], visualizationState['boundary']['positions']], dim = 0)
        else:
            if which == 'fluid' or not config['boundary']['active']:
                if quantity.shape[0] != visualizationState['fluid']['numParticles']:
                    raise ValueError('Quantity does not have the same number of particles as the fluid')
                inputQuantity = quantity[:visualizationState['fluid']['numParticles']]
                pos_x = visualizationState['fluid']['positions']
            elif which == 'boundary':
                if quantity.shape[0] != visualizationState['boundary']['numParticles']:
                    raise ValueError('Quantity does not have the same number of particles as the boundary')
                inputQuantity = quantity[visualizationState['fluid']['numParticles']:]
                pos_x = visualizationState['boundary']['positions']
            else:
                if quantity.shape[0] != visualizationState['fluid']['numParticles'] + visualizationState['boundary']['numParticles']:
                    raise ValueError('Quantity does not have the same number of particles as the fluid and boundary combined')
                inputQuantity = quantity
                pos_x = torch.cat([visualizationState['fluid']['positions'], visualizationState['boundary']['positions']], dim = 0)

    setPlotBaseAttributes(axis, config)
    if title is not None:
        axis.set_title(title)

    if operation is not None:
        initialQuantity = inputQuantity.clone()
        if which == 'fluid' or not config['boundary']['active']:
            inputQuantity = sphOperationStates(visualizationState['fluid'], visualizationState['fluid'], (inputQuantity, inputQuantity), operation = operation, neighborhood = visualizationState['fluid']['neighborhood'])
        elif which == 'boundary':
            inputQuantity = sphOperationStates(visualizationState['boundary'], visualizationState['boundary'], (inputQuantity, inputQuantity), operation = operation, neighborhood = visualizationState['boundary']['neighborhood'])
        else:
            numFluid = visualizationState['fluid']['numParticles']
            inputQuantityF = sphOperationStates(visualizationState['fluid'], visualizationState['fluid'], (inputQuantity[:numFluid], inputQuantity[:numFluid]), operation = operation, neighborhood = visualizationState['fluid']['neighborhood'])
            inputQuantityB = sphOperationStates(visualizationState['boundary'], visualizationState['boundary'], (inputQuantity[numFluid:], inputQuantity[numFluid:]), operation = operation, neighborhood = visualizationState['boundary']['neighborhood'])
            inputQuantity = torch.cat([inputQuantityF, inputQuantityB], dim = 0)



    if len(inputQuantity.shape) == 2:
        # Non scalar quantity
        if mapping == '.x' or mapping == '[0]':
            quantity = inputQuantity[:,0]
        if mapping == '.y' or mapping == '[1]':
            quantity = inputQuantity[:,1]
        if mapping == '.z' or mapping == '[2]':
            quantity = inputQuantity[:,2]
        if mapping == '.w' or mapping == '[3]':
            quantity = inputQuantity[:,3]
        if mapping == 'Linf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = float('inf'))
        if mapping == 'L-inf':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = -float('inf'))
        if mapping == 'L0':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 0)
        if mapping == 'L1':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 1)
        if mapping == 'L2' or mapping == 'norm' or mapping == 'magnitude':
            quantity = torch.linalg.norm(inputQuantity, dim = -1, ord = 2)
        if mapping == 'theta':
            quantity = torch.atan2(inputQuantity[:,1], inputQuantity[:,0])
    else:
        quantity = inputQuantity

    # pos_x = visualizationState['fluid']['positions']

    minScale = torch.min(quantity)
    maxScale = torch.max(quantity)
    if 'sym' in scaling:
        minScale = - torch.max(torch.abs(quantity))
        maxScale =   torch.max(torch.abs(quantity))
        if 'log'in scaling:
            norm = matplotlib.colors.SymLogNorm(vmin = minScale, vmax = maxScale, linthresh = linthresh)
        else:
            minScale = - torch.max(torch.abs(quantity - midPoint))
            maxScale =   torch.max(torch.abs(quantity - midPoint))
            norm = matplotlib.colors.CenteredNorm(vcenter = midPoint, halfrange = maxScale)
    else:
        if 'log'in scaling:
            vmm = torch.min(torch.abs(quantity[quantity!= 0]))
            norm = matplotlib.colors.LogNorm(vmin = vmm, vmax = maxScale)
        else:
            norm = matplotlib.colors.Normalize(vmin = minScale, vmax = maxScale)
        
    scFluid = None
    scBoundary = None
    if not gridVisualization:
        if which == 'fluid' or not config['boundary']['active']:
            scFluid = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = quantity.detach().cpu().numpy(), cmap = cmap, norm = norm)
            if plotBoth and config['boundary']['active']:
                scBoundary = axis.scatter(visualizationState['boundary']['positions'][:,0].detach().cpu().numpy(), visualizationState['boundary']['positions'][:,1].detach().cpu().numpy(), s = s * 5, c = 'black', marker = 'x')
        elif which == 'boundary':
            scBoundary = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s * 5, c = quantity.detach().cpu().numpy(), cmap = cmap, norm = norm, marker = 'x')
            if plotBoth:
                scFluid = axis.scatter(visualizationState['fluid']['positions'][:,0].detach().cpu().numpy(), visualizationState['fluid']['positions'][:,1].detach().cpu().numpy(), s = s, c = 'black', cmap = cmap, norm = norm)
        else:
            scFluid = axis.scatter(pos_x[:visualizationState['fluid']['numParticles'],0].detach().cpu().numpy(), pos_x[:visualizationState['fluid']['numParticles'],1].detach().cpu().numpy(), s = s, c = quantity[:visualizationState['fluid']['numParticles']].detach().cpu().numpy(), cmap = cmap, norm = norm)
            scBoundary = axis.scatter(pos_x[visualizationState['fluid']['numParticles']:,0].detach().cpu().numpy(), pos_x[visualizationState['fluid']['numParticles']:,1].detach().cpu().numpy(), s = s * 5, c = quantity[visualizationState['fluid']['numParticles']:].detach().cpu().numpy(), cmap = cmap, norm = norm, marker = 'x')

    else:
        if which != 'fluid':
            raise ValueError('Grid visualization is only supported for fluid particles')
        gridDensity = mapToGrid(visualizationState, quantity)
        X = visualizationState['X']
        Y = visualizationState['Y']
        scFluid = axis.pcolormesh(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), gridDensity.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), cmap = cmap, norm = norm)

        if streamLines:
            if operation is not None and len(quantity.shape) != 2:
                inputQuantity = initialQuantity
            grid_ux = mapToGrid(visualizationState, inputQuantity[:,0])
            grid_uy = mapToGrid(visualizationState, inputQuantity[:,1])
            X = visualizationState['X']
            Y = visualizationState['Y']

            stream = axis.streamplot(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), grid_ux.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), grid_uy.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), color='k', linewidth=1, density=1, arrowstyle='->', arrowsize=0.5)

        
    if cbar:
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="4%", pad="1%")
        cb = fig.colorbar(scFluid if which == 'fluid' or which == 'all' else scBoundary, cax=cax1,orientation='vertical')
        cb.ax.tick_params(labelsize=8)
    # if periodicX:
    #     axis.axis('equal')
    #     axis.set_xlim(minDomain[0], maxDomain[0])
    #     axis.set_ylim(minDomain[1], maxDomain[1])
    # else:
    #     axis.set_aspect('equal', 'box')

    return {'plot': scFluid, 'boundaryPlot': scBoundary, 'cbar': cb if cbar else None, 'mapping': mapping, 'colormap': cmap, 'scale': scaling, 'size':4, 'mapToGrid': gridVisualization, 'midPoint' : midPoint, 'linthresh': linthresh, 'which': which, 'plotBoth': plotBoth, 'quantity': quantity, 'operation': operation, 'streamLines': streamLines, 'streamPlot': stream if streamLines and gridVisualization else None, 'axis': axis}



def updatePlot(plotState, visualizationState, quantity : Union[str, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):        
    # print(inputQuantity.shape)
    # setPlotBaseAttributes(axis, config)
    mapping = plotState['mapping']
    scaling = plotState['scale']
    midPoint = plotState['midPoint']
    linthresh = plotState['linthresh']
    s = plotState['size']
    gridVisualization = plotState['mapToGrid']

    scFluid = plotState['plot']
    scBoundary = plotState['boundaryPlot'] if 'boundaryPlot' in plotState else None

    inputQuantity = None
    pos_x = None
    which = plotState['which'] if 'which' in plotState else 'fluid'
    if isinstance(quantity, str):
        if which == 'fluid' or scBoundary is None:
            inputQuantity = visualizationState['fluid'][quantity]
            pos_x = visualizationState['fluid']['positions']
        elif which == 'boundary':
            inputQuantity = visualizationState['boundary'][quantity]
            pos_x = visualizationState['boundary']['positions']
        else:
            inputQuantity = torch.cat([visualizationState['fluid'][quantity], visualizationState['boundary'][quantity]], dim = 0)
            pos_x = torch.cat([visualizationState['fluid']['positions'], visualizationState['boundary']['positions']], dim = 0)
    else:
        if isinstance(quantity, tuple):
            if which == 'fluid' or scBoundary is None:
                inputQuantity = quantity[0]
                pos_x = visualizationState['fluid']['positions']
            elif which == 'boundary':
                inputQuantity = quantity[1]
                pos_x = visualizationState['boundary']['positions']
            else:
                inputQuantity = torch.cat([quantity[0], quantity[1]], dim = 0)
                pos_x = torch.cat([visualizationState['fluid']['positions'], visualizationState['boundary']['positions']], dim = 0)
        else:
            if which == 'fluid' or scBoundary is None:
                inputQuantity = quantity[:visualizationState['fluid']['numParticles']]
                if inputQuantity.shape[0] != visualizationState['fluid']['numParticles']:
                    raise ValueError('Quantity does not have the same number of particles as the fluid')
                inputQuantity = quantity[:visualizationState['fluid']['numParticles']]
                pos_x = visualizationState['fluid']['positions']
            elif which == 'boundary':
                inputQuantity = quantity[visualizationState['fluid']['numParticles']:]
                if inputQuantity.shape[0] != visualizationState['boundary']['numParticles']:
                    raise ValueError('Quantity does not have the same number of particles as the boundary')
                inputQuantity = quantity[visualizationState['fluid']['numParticles']:]
                pos_x = visualizationState['boundary']['positions']
            else:
                inputQuantity = quantity
                if inputQuantity.shape[0] != visualizationState['fluid']['numParticles'] + visualizationState['boundary']['numParticles']:
                    raise ValueError('Quantity does not have the same number of particles as the fluid and boundary combined')
                inputQuantity = quantity
                pos_x = torch.cat([visualizationState['fluid']['positions'], visualizationState['boundary']['positions']], dim = 0)
    # else:
    #     pos_x = visualizationState['fluid']['positions']

    operation = plotState['operation'] if 'operation' in plotState else None
    if operation is not None:
        initialQuantity = inputQuantity
        if which == 'fluid' or scBoundary is None:
            inputQuantity = sphOperationStates(visualizationState['fluid'], visualizationState['fluid'], (inputQuantity, inputQuantity), operation = operation, neighborhood = visualizationState['fluid']['neighborhood'])
        elif which == 'boundary':
            inputQuantity = sphOperationStates(visualizationState['boundary'], visualizationState['boundary'], (inputQuantity, inputQuantity), operation = operation, neighborhood = visualizationState['boundary']['neighborhood'])
        else:
            numFluid = visualizationState['fluid']['numParticles']
            inputQuantityF = sphOperationStates(visualizationState['fluid'], visualizationState['fluid'], (inputQuantity[:numFluid], inputQuantity[:numFluid]), operation = operation, neighborhood = visualizationState['fluid']['neighborhood'])
            inputQuantityB = sphOperationStates(visualizationState['boundary'], visualizationState['boundary'], (inputQuantity[numFluid:], inputQuantity[numFluid:]), operation = operation, neighborhood = visualizationState['boundary']['neighborhood'])
            inputQuantity = torch.cat([inputQuantityF, inputQuantityB], dim = 0)


    if len(inputQuantity.shape) == 2:
        # Non scalar quantity
        if mapping == '.x' or mapping == '[0]':
            quantityDevice = inputQuantity[:,0]
        if mapping == '.y' or mapping == '[1]':
            quantityDevice = inputQuantity[:,1]
        if mapping == '.z' or mapping == '[2]':
            quantityDevice = inputQuantity[:,2]
        if mapping == '.w' or mapping == '[3]':
            quantityDevice = inputQuantity[:,3]
        if mapping == 'Linf':
            quantityDevice = torch.linalg.norm(inputQuantity, dim = -1, ord = float('inf'))
        if mapping == 'L-inf':
            quantityDevice = torch.linalg.norm(inputQuantity, dim = -1, ord = -float('inf'))
        if mapping == 'L0':
            quantityDevice = torch.linalg.norm(inputQuantity, dim = -1, ord = 0)
        if mapping == 'L1':
            quantityDevice = torch.linalg.norm(inputQuantity, dim = -1, ord = 1)
        if mapping == 'L2' or mapping == 'norm' or mapping == 'magnitude':
            quantityDevice = torch.linalg.norm(inputQuantity, dim = -1, ord = 2)
        if mapping == 'theta':
            quantityDevice = torch.atan2(inputQuantity[:,1], inputQuantity[:,0])
    else:
        quantityDevice = inputQuantity

    # pos_x = visualizationState['fluid']['positions']
    qcpu = quantityDevice.detach().cpu()
    minScale = torch.min(qcpu)
    maxScale = torch.max(qcpu)
    if 'sym' in scaling:
        minScale = - torch.max(torch.abs(qcpu))
        maxScale =   torch.max(torch.abs(qcpu))
        if 'log'in scaling:
            norm = matplotlib.colors.SymLogNorm(vmin = minScale, vmax = maxScale, linthresh = linthresh)
        else:
            minScale = - torch.max(torch.abs(qcpu - midPoint))
            maxScale =   torch.max(torch.abs(qcpu - midPoint))
            norm = matplotlib.colors.CenteredNorm(vcenter = midPoint, halfrange = maxScale)
    else:
        if 'log'in scaling:
            vmm = torch.min(torch.abs(qcpu[qcpu!= 0]))
            norm = matplotlib.colors.LogNorm(vmin = vmm, vmax = maxScale)
        else:
            norm = matplotlib.colors.Normalize(vmin = minScale, vmax = maxScale)
        
    if not gridVisualization:
        # if 'quantity' in plotState:
        # print('Updating plot')
        # print(plotState['plot'])
        # print(which)
        scFluid = plotState['plot']
        scBoundary = plotState['boundaryPlot']
        if scFluid is not None:
            if which == 'fluid' or scBoundary is None:
                scFluid.set_offsets(pos_x.detach().cpu().numpy())
                scFluid.set_array(qcpu.numpy())
                scFluid.set_norm(norm)
            elif which == 'boundary':
                scFluid.set_offsets(visualizationState['fluid']['positions'].detach().cpu().numpy())
                # scFluid.set_array(qcpu.numpy())
                # scFluid.set_norm(norm)
            else:
                scFluid.set_offsets(pos_x[:visualizationState['fluid']['numParticles']].detach().cpu().numpy())
                scFluid.set_array(qcpu[:visualizationState['fluid']['numParticles']].numpy())
                scFluid.set_norm(norm)
        if scBoundary is not None:
            if which == 'fluid':
                scBoundary.set_offsets(visualizationState['boundary']['positions'].detach().cpu().numpy())
                # scBoundary.set_array(qcpu.numpy())
                # scBoundary.set_norm(norm)
            elif which == 'boundary':
                scBoundary.set_offsets(pos_x.detach().cpu().numpy())
                scBoundary.set_array(qcpu.numpy())
                scBoundary.set_norm(norm)
            else:         
                scBoundary = plotState['boundaryPlot']
                scBoundary.set_offsets(pos_x[visualizationState['fluid']['numParticles']:].detach().cpu().numpy())
                scBoundary.set_array(qcpu[visualizationState['fluid']['numParticles']:].numpy())
                scBoundary.set_norm(norm)

        # else:
        #     sc = plotState['plot']
        #     sc.set_offsets(pos_x.detach().cpu().numpy())
        #     sc.set_array(qcpu.numpy())
        #     sc.set_norm(norm)

        # scVelocity_x.set_clim(vmin = torch.abs(c).max().item() * -1, vmax = torch.abs(c).max().item())
        # cbarVelocity_x.update_normal(scVelocity_x)
        # sc = axis.scatter(pos_x[:,0].detach().cpu().numpy(), pos_x[:,1].detach().cpu().numpy(), s = s, c = quantity.detach().cpu().numpy(), cmap = cmap, norm = norm)
    else:
        sc = plotState['plot']
        gridDensity = mapToGrid(visualizationState, quantityDevice)
        sc.set_array(gridDensity.detach().cpu().numpy())
        sc.set_norm(norm)

        if plotState['streamLines']:
            if operation is not None and len(quantityDevice.shape) != 2:
                inputQuantity = initialQuantity
            # else:
                # inputQuantity = quantityDevice
            axis = plotState['axis']
            grid_ux = mapToGrid(visualizationState, inputQuantity[:,0])
            grid_uy = mapToGrid(visualizationState, inputQuantity[:,1])
            X = visualizationState['X']
            Y = visualizationState['Y']
            priorStream = plotState['streamPlot']
            # ax = axis
            # keep = lambda x: not isinstance(x, mpl.patches.FancyArrowPatch)
            # axis.patches = [patch for patch in axis.patches if keep(patch)]


            priorStream.lines.remove()  # Removes the stream lines
            priorStream.arrows.set_visible(False)  # Does nothing
            # priorStream.arrows.remove()  # Raises NotImplementedError
            for art in axis.get_children():
                if not isinstance(art, matplotlib.patches.FancyArrowPatch):
                    continue
                art.remove()        # Method 1

            plotState['streamPlot'] = axis.streamplot(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), grid_ux.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), grid_uy.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), color='k', linewidth=1, density=1, arrowstyle='->', arrowsize=0.5)
            # sc = axis.pcolormesh(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), gridDensity.reshape(visualizationState['nGrid'], visualizationState['nGrid']).detach().cpu().numpy(), cmap = cmap, norm = norm)
    