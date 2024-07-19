from BasisConvolution.sph.sphOps import sphOperationStates
import torch
import numpy as np
import re

def translateFeature(featureName):
    featureName = featureName.strip()#.lower()
    if featureName == 'x' or featureName == 'position' or featureName == 'pos':
        return 'positions'
    elif featureName == 'u' or featureName == 'velocity' or featureName == 'vel':
        return 'velocities'
    elif featureName == 'rho' or featureName == 'density':
        return 'densities'
    elif featureName == 'p' or featureName == 'pressure':
        return 'pressures'
    elif featureName == 'a' or featureName == 'area':
        return 'areas'
    elif featureName == 'm' or featureName == 'mass':
        return 'masses'
    elif featureName == 'n' or featureName == 'normal':
        return 'normals'
    
    return featureName

def getFeaturev2(featureName, currentState, priorState, which, config, dt, verbose = False, includeOther = True):
    refDtype = currentState[which]['positions'].dtype
    refDevice = currentState[which]['positions'].device
    nPtcls = currentState[which]['numParticles']
    featureName = featureName.strip().replace('=', '@').replace(':','@')

    arithmetic = ['+', '-', '*', '/']
    arithmetic = ['*', '/']
    if any([op in featureName for op in arithmetic]):
        if verbose:
            print('Processing arithmetic operation:', featureName)
        features = []
        # if '+' in featureName:
        #     features = featureName.split('+')
        # if '-' in featureName:
        #     features = featureName.split('-')
        if '*' in featureName:
            features = featureName.split('*')
        if '/' in featureName:
            features = featureName.split('/')

        if verbose:
            print('Input Operands: ', features)
        a = getFeaturev2(features[0], currentState, priorState, which, config, dt, verbose = verbose, includeOther=includeOther)
        b = getFeaturev2(features[1], currentState, priorState, which, config, dt, verbose = verbose, includeOther=includeOther)
        if a.dim() != b.dim():
            if a.dim() < b.dim():
                a = a.view(-1,1)
            else:
                b = b.view(-1,1)
        
        # if '+' in featureName:
        #     return a + b
        # if '-' in featureName:
        #     return a - b
        if '*' in featureName:
            return a * b
        if '/' in featureName:
            return a / b

    if featureName.startswith('compute'):
        if verbose:
            print('Processing compute operation:', featureName)

        regex = re.compile(r'compute\[(.*)\][:@=](.*)')
        operation, feature = regex.match(featureName).groups()
        operation = operation.strip()
        # feature = translateFeature(quantity)

        if verbose:
            print('Arguments:', operation, feature)

        if '.' in feature:
            op = feature.split('.')[1]
            feature = feature.split('.')[0]
        else:
            op = None
        if verbose:
            print('Looking up feature:', feature)
        if includeOther and ('boundary' in currentState and currentState['boundary'] is not None):
            if verbose:
                print('Looking up Fluid Attribute', feature)

            fluidAttribute = getFeaturev2(feature, currentState, priorState, 'fluid', config, dt, verbose = verbose, includeOther=includeOther)
            if verbose:
                print('Looking up Boundary Attribute', feature)
            boundaryAttribute = getFeaturev2(feature, currentState, priorState, 'boundary', config, dt, verbose = verbose, includeOther=includeOther)

            if verbose:
                print('Lookup done')
            if op is not None:
                if op == 'x':
                    fluidAttribute = fluidAttribute[:,0]
                    boundaryAttribute = boundaryAttribute[:,0]
                if op == 'y':
                    fluidAttribute = fluidAttribute[:,1]
                    boundaryAttribute = boundaryAttribute[:,1]
                if op == 'z':
                    fluidAttribute = fluidAttribute[:,2]
                    boundaryAttribute = boundaryAttribute[:,2]
                if op == 'L2':
                    fluidAttribute = torch.norm(fluidAttribute, dim = 1)
                    boundaryAttribute = torch.norm(boundaryAttribute, dim = 1)
                if op == 'L1':
                    fluidAttribute = torch.norm(fluidAttribute, dim = 1, p = 1)
                    boundaryAttribute = torch.norm(boundaryAttribute, dim = 1, p = 1)
                if op == 'Linf':
                    fluidAttribute = torch.norm(fluidAttribute, dim = 1, p = float('inf'))
                    boundaryAttribute = torch.norm(boundaryAttribute, dim = 1, p = float('inf'))
                else:
                    raise ValueError('Unknown attribute operation: ' + op)
                

            operationName = operation
            gradientMode = 'naive'
            if operation == 'grad':
                operationName = 'gradient'
            if operation == 'gradDiff':
                operationName = 'gradient'
                gradientMode = 'difference'
            if operation == 'gradSum':
                operationName = 'gradient'
                gradientMode = 'summation'
            if operation == 'gradSym':
                operationName = 'gradient'
                gradientMode = 'symmetric'
            if operation == 'rho':
                operationName = 'density'
            if operation == 'vorticity':
                operationName = 'curl'

            if verbose: 
                print('Operation:', operationName, 'Gradient Mode:', gradientMode)
            if which == 'fluid':
                if verbose:
                    print('Fluid To Fluid Operation')
                ftf = sphOperationStates(currentState['fluid'], currentState['fluid'], (fluidAttribute, fluidAttribute), operation = operationName, neighborhood = currentState['fluid']['neighborhood'], gradientMode = gradientMode)
                if verbose:
                    print('Boundary To Fluid Operation')
                btf = sphOperationStates(currentState['fluid'], currentState['boundary'], (fluidAttribute, boundaryAttribute), operation = operationName, neighborhood = currentState['boundaryToFluidNeighborhood'], gradientMode = gradientMode)
                if verbose:
                    print('Returning Fluid To Fluid + Boundary To Fluid')
                return ftf + btf
            else:
                if verbose:
                    print('Boundary To Boundary Operation')
                btb = sphOperationStates(currentState['boundary'], currentState['boundary'], (boundaryAttribute, boundaryAttribute), operation = operationName, neighborhood = currentState['boundary']['neighborhood'], gradientMode = gradientMode)
                if verbose:
                    print('Fluid To Boundary Operation')
                ftb = sphOperationStates(currentState['boundary'], currentState['fluid'], (boundaryAttribute, fluidAttribute), operation = operationName, neighborhood = currentState['fluidToBoundaryNeighborhood'], gradientMode = gradientMode)
                if verbose:
                    print('Returning Boundary To Boundary + Fluid To Boundary')
                return btb + ftb

                
        else:
            attribute = getFeaturev2(feature, currentState, priorState, which, config, dt, verbose = verbose, includeOther=includeOther)
            if op is not None:
                if op == 'x':
                    attribute = attribute[:,0]
                if op == 'y':
                    attribute = attribute[:,1]
                if op == 'z':
                    attribute = attribute[:,2]
                if op == 'L2':
                    attribute = torch.norm(attribute, dim = 1)
                if op == 'L1':
                    attribute = torch.norm(attribute, dim = 1, p = 1)
                if op == 'Linf':
                    attribute = torch.norm(attribute, dim = 1, p = float('inf'))
                else:
                    raise ValueError('Unknown attribute operation: ' + op)


            if operation == 'divergence':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'divergence', neighborhood = currentState[which]['neighborhood']).view(-1,1)
            elif operation == 'vorticity':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'curl', neighborhood = currentState[which]['neighborhood']).view(-1,1)
            elif operation == 'grad':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'gradient', neighborhood = currentState[which]['neighborhood'], gradientMode = 'naive')
            elif operation == 'gradDiff':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'gradient', neighborhood = currentState[which]['neighborhood'], gradientMode = 'difference')
            elif operation == 'gradSum':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'gradient', neighborhood = currentState[which]['neighborhood'], gradientMode = 'summation')
            elif operation == 'gradSym':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'gradient', neighborhood = currentState[which]['neighborhood'], gradientMode = 'symmetric')
            elif operation == 'laplacian':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'laplacian', neighborhood = currentState[which]['neighborhood']).view(-1,1)
            elif operation == 'rho':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'density', neighborhood = currentState[which]['neighborhood']).view(-1,1)  
            elif operation == 'interpolate':
                return sphOperationStates(currentState[which], currentState[which], (attribute, attribute), operation = 'interpolate', neighborhood = currentState[which]['neighborhood']).view(-1,1)      
            else:
                raise ValueError('Unknown operation: ' + operation)

    if featureName.startswith('diff'):
        if verbose:
                print('Processing difference operation:', featureName)
        feature = featureName.split('@', maxsplit = 1)[1]
            
        if '.' in feature:
            op = feature.split('.')[1]
            feature = feature.split('.')[0]
        else:
            op = None
        
        if verbose:
            print('Looking up feature:', feature)
        attributeCurrent = getFeaturev2(feature, currentState, priorState, which, config, dt, verbose = verbose, includeOther=includeOther)
        attributePrior = getFeaturev2(feature, priorState, None, which, config, dt, verbose = verbose, includeOther=includeOther)

        attribute = attributeCurrent - attributePrior

        if op is not None:
            if op == 'x':
                return attribute[:,0]
            if op == 'y':
                return attribute[:,1]
            if op == 'z':
                return attribute[:,2]
            if op == 'L2':
                return torch.norm(attribute, dim = 1)
            if op == 'L1':
                return torch.norm(attribute, dim = 1, p = 1)
            if op == 'Linf':
                return torch.norm(attribute, dim = 1, p = float('inf'))
            else:
                raise ValueError('Unknown attribute operation: ' + op)
        return attribute
    if featureName.startswith('dt'):
        if verbose:
            print('Processing time difference operation:', featureName)
        feature = featureName.split('@', maxsplit = 1)[1]
            
        if '.' in feature:
            op = feature.split('.')[1]
            feature = feature.split('.')[0]
        else:
            op = None
        if verbose:
            print('Looking up feature:', feature)
        attributeCurrent = getFeaturev2(feature, currentState, priorState, which, config, dt, verbose = verbose, includeOther=includeOther)
        attributePrior = getFeaturev2(feature, priorState, None, which, config, dt, verbose = verbose, includeOther=includeOther)

        attribute = (attributeCurrent - attributePrior) / dt

        if op is not None:
            if op == 'x':
                return attribute[:,0]
            if op == 'y':
                return attribute[:,1]
            if op == 'z':
                return attribute[:,2]
            if op == 'L2':
                return torch.norm(attribute, dim = 1)
            if op == 'L1':
                return torch.norm(attribute, dim = 1, p = 1)
            if op == 'Linf':
                return torch.norm(attribute, dim = 1, p = float('inf'))
            else:
                raise ValueError('Unknown attribute operation: ' + op)
        return attribute



    if featureName.startswith('attribute'):
        if verbose:
            print('Processing attribute operation:', featureName)
        feature = featureName.split('@', maxsplit = 1)[1]
            
        if '.' in feature:
            op = feature.split('.')[1]
            feature = feature.split('.')[0]
        else:
            op = None
        if verbose:
            print('Looking up feature:', feature)
        attribute = currentState[which][translateFeature(feature)]
        if op is not None:
            if op == 'x':
                return attribute[:,0]
            if op == 'y':
                return attribute[:,1]
            if op == 'z':
                return attribute[:,2]
            if op == 'L2':
                return torch.norm(attribute, dim = 1)
            if op == 'L1':
                return torch.norm(attribute, dim = 1, p = 1)
            if op == 'Linf':
                return torch.norm(attribute, dim = 1, p = float('inf'))
            else:
                raise ValueError('Unknown attribute operation: ' + op)
        return attribute
    if 'constant' in featureName:
        if verbose:
            print('Processing constant operation:', featureName)
        feature = featureName.split('@', maxsplit = 1)[1]

        constant = 0
        if feature == 'zero' or feature == 'zeroes' or feature == 'zeros':
            constant = 0
        elif feature == 'one' or feature == 'ones':
            constant = 1
        elif feature == 'pi':
            constant = np.pi
        elif feature == 'rho0':
            constant = config['fluid']['rho0']
        elif feature == 'cs':
            constant = config['fluid']['cs']
        elif feature == 'targetNeighbors':
            constant = config['kernel']['targetNeighbors']
        else:
            constant = float(feature)
        if verbose:
            print('Constant = ', constant)
        return torch.ones(nPtcls, dtype = refDtype, device = refDevice) * constant

    raise ValueError('Unknown feature: ' + featureName)
 

def getFeatures(featureNames, currentState, priorState, which, config, dt, includeOther = True, verbose = False):
    features = []
    for featureName in featureNames:
        if verbose:
            print('Processing feature (main loop):', featureName)
        feat = getFeaturev2(featureName, currentState, priorState, which, config, dt, includeOther = includeOther, verbose = verbose)
        if feat.dim() == 1:
            feat = feat.view(-1,1)
        features.append(feat)
    # print(features)
    return torch.cat(features, dim = 1)