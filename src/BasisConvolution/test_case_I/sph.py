# Copyright 2023 <COPYRIGHT HOLDER>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Math/parallelization library includes
import numpy as np
import torch

# Imports for neighborhood searches later on
from BasisConvolution.detail.scatter import scatter_sum

# Plotting includes
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from BasisConvolution.detail.radius import periodicNeighborSearch

# Wendland 2 kernel as per Dehnen and Aly 2012 https://arxiv.org/abs/1204.2471
C = 5/4
def kernel(q, support):
    return C * (1-q)**3 * (1 + 3 * q) / (support)

def kernelGradient(q, dist, support):
    return -dist * C * 12 * (q-1)**2 * q / (support ** 2)

# Ghost particle creation for periodic BC, these particles not actually used
# but only used as part of the neighborhood search
def createGhostParticles(particles, minDomain, maxDomain):
    ghostParticlesLeft = particles - (maxDomain - minDomain)
    ghostParticlesRight = particles + (maxDomain - minDomain)

    allParticles = torch.hstack((ghostParticlesLeft, particles, ghostParticlesRight))
    return allParticles

# Summation density formulation. Note that we ignore the rest density here as this term cancels out everywhere
# that we use it and it makes learning this term more straight forward.
def computeDensity(particles, particleArea, particleSupport, fluidRadialDistances, fluidNeighbors):
    pairWiseDensity = particleArea[fluidNeighbors[1]] * kernel(fluidRadialDistances, particleSupport)
    fluidDensity = scatter_sum(pairWiseDensity, fluidNeighbors[0], dim=0, dim_size = particles.shape[0])
    
    return fluidDensity

# symmetric pressure gradient based on DJ Price 2010 https://arxiv.org/pdf/1012.1885.pdf
def computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwisePressureForces = fluidAreas[j] * restDensity * \
            (fluidPressure[i] / (fluidDensity[i] * restDensity)**2 + fluidPressure[j] / (fluidDensity[j]* restDensity)**2) *\
            kernelGradient(fluidRadialDistances, fluidDistances, particleSupport)
    fluidPressureForces = scatter_sum(pairwisePressureForces, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0])
    
    return fluidPressureForces

# Laminar viscosity term based on DJ Price 2010 https://arxiv.org/pdf/1012.1885.pdf
def computeLaminarViscosity(i, j, ri, rj, Vi, Vj, distances, radialDistances, support, numParticles : int, eps : float, rhoi, rhoj, ui, uj, alpha : float, beta: float, c0 : float, restDensity : float):
    gradW = kernelGradient(radialDistances, distances, support)

    uij = ui[i] - uj[j]
    rij = ri[i] - rj[j]
    rij = -distances * radialDistances
    
    mu_nom = support * (uij * rij)
    mu_denom = torch.abs(rij) + 0.01 * support**2
    mu = mu_nom / mu_denom
    
    
    nom = - alpha * c0 * mu + beta * mu**2
    denom = (rhoi[i] + rhoj[j]) / 2
    termL = Vi[j] * nom / denom
    
    term = termL * gradW
    return scatter_sum(term, i, dim=0, dim_size = Vi.shape[0])
# Helper function that calls the laminar viscosity function
def computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensities, particleSupport, restDensity, alpha, beta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances):
    laminarViscosity = computeLaminarViscosity(fluidNeighbors[0], fluidNeighbors[1], \
                                                                                      fluidPositions, fluidPositions, fluidAreas, fluidAreas,\
                                                                                      fluidDistances, fluidRadialDistances,\
                                                                                      particleSupport, fluidDensities.shape[0], 1e-7,\
                                                                                      fluidDensities, fluidDensities,\
                                                                                      fluidVelocities,fluidVelocities,
                                                                                      alpha, beta, c0, restDensity)
    
    return laminarViscosity
# XSPH based numerical viscosity term based on Monaghan 2005 https://ui.adsabs.harvard.edu/link_gateway/2005RPPh...68.1703M/doi:10.1088/0034-4885/68/8/R01
def computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphConstant, fluidNeighbors, fluidRadialDistances):
    i = fluidNeighbors[0,:]
    j = fluidNeighbors[1,:]

    pairwiseXSPH = xsphConstant * fluidAreas[j] / ( fluidDensity[i] + fluidDensity[j]) * 2 \
                * (fluidVelocities[j] - fluidVelocities[i]) \
                * kernel(fluidRadialDistances, particleSupport) 
    
    xsphUpdate = scatter_sum(pairwiseXSPH, fluidNeighbors[0], dim=0, dim_size = fluidPositions.shape[0])
    
    return xsphUpdate

# Function to sample particles such that their density equals a desired PDF
def samplePDF(pdf, n = 2048, numParticles = 1024, plot = False, randomSampling = False):
    x = np.linspace(-1,1,n)
    if plot:
        fig, axis = plt.subplots(1, 1, figsize=(9,6), sharex = False, sharey = False, squeeze = False)

    n = 2048
    xs = np.linspace(-1,1,n)

    if plot:
        axis[0,0].plot(xs, pdf(xs))

    normalized_pdf = lambda x: pdf(x) / np.sum(pdf(np.linspace(-1,1,n)))
    if plot:
        axis[0,0].plot(xs, normalized_pdf(xs))
        axis[0,0].axhline(0,ls= '--', color = 'black')


    xs = np.linspace(-1,1,n)
    fxs = normalized_pdf(xs)
    sampled_cdf = np.cumsum(fxs) - fxs[0]
    sampled_cdf = sampled_cdf / sampled_cdf[-1] 
    inv_cdf = lambda x : np.interp(x, sampled_cdf, np.linspace(-1,1,n))

    samples = np.random.uniform(size = numParticles)
    if not randomSampling:
        samples = np.linspace(0,1,numParticles, endpoint=False)
    sampled = inv_cdf(samples)

    return sampled

# SPH simulation step, returns dudt, dxdt as well as current density and pressure
def computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphCoefficient, particleSupport, dt):
    # 1. Find neighborhoods of all particles:
    fluidNeighbors, fluidRadialDistances, fluidDistances = periodicNeighborSearch(fluidPositions, particleSupport, minDomain, maxDomain)
    # 2. Compute \rho using an SPH interpolation
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)
    # 3. Compute the pressure of each particle using an ideal gas EOS
    fluidPressure = (fluidDensity - 1.0) * kappa * restDensity
    # 4. Compute pressure forces and resulting acceleration
    fluidPressureForces = computePressureForces(fluidPositions, fluidDensity, fluidPressure, fluidAreas, particleSupport, restDensity, fluidNeighbors, fluidRadialDistances, fluidDistances)
    fluidAccel = fluidPressureForces # / (fluidAreas * restDensity)
    # 5. Compute the XSPH term and apply it to the particle velocities:    
    # xsphUpdate = computeXSPH(fluidPositions, fluidVelocities, fluidDensity, fluidAreas, particleSupport, xsphCoefficient, fluidNeighbors, fluidRadialDistances)
    # fluidAccel += xsphUpdate / dt
    # 6. Compute kinematic viscosity
    fluidAccel += computeDiffusion(fluidPositions, fluidVelocities, fluidAreas, fluidDensity, particleSupport, restDensity, diffusionAlpha, diffusionBeta, c0, fluidNeighbors, fluidRadialDistances, fluidDistances) # currently broken for some reason
    return fluidAccel, fluidVelocities, fluidDensity, fluidPressure

from .plotting import plotDensityField
def initSimulation(pdf, numParticles, minDomain, maxDomain, baseArea, particleSupport, dtype, device, plot = False):
    # sample the pdf using the inverse CFD, plotting shows the pdf
    sampled = samplePDF(pdf, plot = False, numParticles = numParticles)
    # sample positions according to the given pdf
    fluidPositions = ((torch.tensor(sampled)/2 +0.5)* (maxDomain - minDomain) + minDomain).type(dtype).to(device)
    # initially zero velocity everywhere
    fluidVelocities = torch.zeros(fluidPositions.shape[0]).type(dtype).to(device)
    # and all particles with identical masses
    fluidAreas = torch.ones_like(fluidPositions) * baseArea
    # simulationStates holds all timestep information
    simulationStates = []
    # plot initial density field to show starting conditions
    if plot:
        density = plotDensityField(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport)
    return fluidPositions, fluidAreas, fluidVelocities

def runSimulation(fluidPositions_, fluidAreas_, fluidVelocities_, timesteps, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt):
    fluidPositions = torch.clone(fluidPositions_)
    fluidAreas = torch.clone(fluidAreas_)
    fluidVelocities = torch.clone(fluidVelocities_)
    simulationStates = []
    # run the simulation using RK4
    for i in tqdm(range(timesteps)):
        # Compute state for substep 1
        v1 = torch.clone(fluidVelocities)
        # RK4 substep 1
        dudt_k1, dxdt_k1, fluidDensity, fluidPressure = computeUpdate(fluidPositions, fluidVelocities, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt)   
        # Compute state for substep 2
        x_k1 = fluidPositions + 0.5 * dt * dxdt_k1
        u_k1 = fluidVelocities + 0.5 * dt * dudt_k1    
        # RK4 substep 2
        dudt_k2, dxdt_k2, _, _ = computeUpdate(x_k1, u_k1, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, 0.5 * dt)    
        # Compute state for substep 2
        x_k2 = fluidPositions + 0.5 * dt * dxdt_k2
        u_k2 = fluidVelocities + 0.5 * dt * dudt_k2
        # RK4 substep 3
        dudt_k3, dxdt_k3, _, _ = computeUpdate(x_k2, u_k2, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport,  0.5 * dt)    
        # Compute state for substep 4    
        x_k3 = fluidPositions + dt * dxdt_k3
        u_k3 = fluidVelocities + dt * dudt_k3
        # RK4 substep 4
        dudt_k4, dxdt_k4, _, _ = computeUpdate(x_k3, u_k3, fluidAreas, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt)    
        # RK substeps done, store current simulation state for later processing/learning. density and pressure are based on substep 1 (i.e., the starting point for this timestep)
        simulationStates.append(torch.stack([fluidPositions, fluidVelocities, fluidDensity, fluidPressure, dt/6 * (dudt_k1 + 2* dudt_k2 + 2 * dudt_k3 + dudt_k4), dudt_k1, dudt_k2, dudt_k3, dudt_k4, dxdt_k1, dxdt_k2, dxdt_k3, dxdt_k4, fluidAreas]))
        # time integration using RK4 for velocity
    #     fluidVelocities = fluidVelocities + dt * dudt_k1 # semi implicit euler mode
        fluidVelocities = fluidVelocities + dt/6 * (dudt_k1 + 2* dudt_k2 + 2 * dudt_k3 + dudt_k4)
        fluidPositions = fluidPositions + dt * fluidVelocities
    # After the simulation has run we stack all the states into one large array for easier slicing and analysis
    simulationStates = torch.stack(simulationStates)
    return simulationStates
