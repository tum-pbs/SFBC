import os
import sys
module_path = os.path.abspath(os.path.join('../'))
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
# sph related imports
from BasisConvolution.test_case_I.sph import initSimulation, runSimulation
from BasisConvolution.test_case_I.perlin import generate1DPeriodicNoise
from BasisConvolution.test_case_I.io import export
# from BasisConvolution.oneDimensionalSPH.plotting import plotSimulationState, regularPlot
# neural network rlated imports
from torch.optim import Adam
# from rbfConv import *
# from torch_geometric.loader import DataLoader
# from trainingHelper import *
# plotting/UI related imports
import matplotlib as mpl
# plt.style.use('dark_background')
cmap = mpl.colormaps['viridis']
from tqdm.autonotebook import tqdm
from IPython.display import display, Latex
from datetime import datetime
import h5py
import torch
import numpy as np


numParticles = 2048 # Modify th number of desird particles here, this value is defined here to ensure that the linar sampling for some pdfs is accurate.
generator = 'perlin'
generatorSettings = {'r' : 0.75, 'freq' : 1, 'octaves' : 2, 'seed' : 1234, 'mag' : 0.25, 'offset': 2}
noise  = generatorSettings['offset'] + generate1DPeriodicNoise(numSamples = numParticles, r = generatorSettings['r'], freq = generatorSettings['freq'], octaves = generatorSettings['octaves'], plot = False, seed = generatorSettings['seed']) * generatorSettings['mag']
pdf = lambda x : np.interp(x, np.linspace(-1,1,numParticles), noise)

def pdf(x):
    x = np.array(x)
    out = np.array(np.ones_like(x))
    out[x > 0] = x[x>0] + 0
    out[x < 0] = x[x<0] + 2
    return out + 1

# simulation parameters
minDomain = -1 # minimum domain, leave at -1 for the most part
maxDomain = 1 # maximum domain, leave at 1 for the most part
# change base area to change initial starting density
baseArea = 2 / numParticles * 2
particleRadius = baseArea / 2.0
# change particle support to make simulation more/less smooth
particleSupport = particleRadius * 8.
# SPH parameters
xsphConstant = 0.0
diffusionAlpha = 1. # kinematic viscosity coefficient
diffusionBeta = 2.
kappa = 10 # EOS kappa term
restDensity = 1000 # EOS rest density term
dt = 1e-3 # fixed global timestep
c0 = 10 # speed of sound used in kinematic viscosity

timesteps = 2048 # timesteps to be simulated
# display(Latex(f'Estimated kinematic diffusion $\\mu\\approx\\frac{{1}}{{2(d+2)}}\\alpha c_s h = \\frac{{1}}{{2(1+2)}} \\cdot {diffusionAlpha} \\cdot {c0} \\cdot {particleSupport:4.2e} = {1/6 * diffusionAlpha * c0 * particleSupport:4.2e}$'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64 if device == 'cpu' else torch.float32
# print('Running on ', device, 'using type', dtype)

torch.manual_seed(10957120378)
seeds = torch.randint(0,2**30, (1,36)).flatten()

seeds = [141077155, 9771992, 521260098, 943800868, 879567972, 843734504, 302569625, 172735587, 549857803, 123272218, 734684074, 926567716, 927288476, 883194790, 497511951, 341412464, 86702325, 31736609, 773229209, 379424249, 577570728, 323111077, 581356742, 712262378, 884632587, 450979520, 15165885, 159763383, 882368412, 783832494, 137881214, 686823772, 433611252, 960745502, 759352143, 37078787]

print(seeds)


for seed in tqdm(seeds):
    generator = 'perlin'
    generatorSettings = {'r' : 0.75, 'freq' : 1, 'octaves' : 2, 'seed' : int(seed), 'mag' : 0.25, 'offset': 2}
    noise  = generatorSettings['offset'] + generate1DPeriodicNoise(numSamples = numParticles, r = generatorSettings['r'], freq = generatorSettings['freq'], octaves = generatorSettings['octaves'], plot = False, seed = generatorSettings['seed']) * generatorSettings['mag']
    pdf = lambda x : np.interp(x, np.linspace(-1,1,numParticles), noise)

    fluidPositions, fluidAreas, fluidVelocities = initSimulation(pdf, numParticles, minDomain, maxDomain, baseArea, particleSupport, dtype, device, plot = False)
    simulationStates = runSimulation(fluidPositions, fluidAreas, fluidVelocities, timesteps, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleSupport, dt)
    export(simulationStates, numParticles, timesteps, minDomain, maxDomain, kappa, restDensity, diffusionAlpha, diffusionBeta, c0, xsphConstant, particleRadius, baseArea, particleSupport, dt, generator, generatorSettings, folder = '../datasets/test_case_I',nameOverride = True)