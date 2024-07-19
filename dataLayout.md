## Test Case I:

### attrs:
- baseArea: Basic Area of a particle (i.e., length in 1D), i.e., $\Delta x$
- c0: Speed of sound, approximately $10\times u_max$
- diffusionAlpha: Velocity Diffusion constant for the Price artificial viscosity model (1.0)
- diffusionBeta: Velocity Diffusion constant for the Price artificial viscosity model (2.0)
- dt: Fixed Timestep used for the simulation
- generator: Scheme used to generate the data, perlin and simplex indicates random velocity fields.
- kappa: Stiffness coefficient for compressible EOS
- maxDomain: Maximum domain extent (1)
- minDomain: Minimum domain extent (-1)
- numParticles: Expected number of particles in a simulation
- particleRadius: Radius, i.e., half particle spacing $\Delta x$
- particleSupport: Support radius used during the simulation
- restDensity: Rest Density of the fluid, 1000 in most cases
- timesteps: Number of timesteps expected in the simulation
- xsphConstant: Legacy constant, not used during the simulation

### generatorSetting.attrs:
- freq: Base Frequency of the noise 
- mag: Maximum amplitude of the noise
- octaves: Number of octaves, i.e., powers of two of the base frequency
- offset: Constant used to shift the density noise up, needed to avoid zero density
- r: Implementation specific parameter
- seed: Seed used for generation (should be included in the file name)

### simulationData (no attrs):
- dudt: Tensor of shape [t,i] containing $\frac{d u}{d t}$
- dudt_k1, dxdt_k1: Substep information from the RK4 integrator for velocity and position updates
- dudt_k2, dxdt_k2: Substep information from the RK4 integrator for velocity and position updates
- dudt_k3, dxdt_k3: Substep information from the RK4 integrator for velocity and position updates
- dudt_k4, dxdt_k4: Substep information from the RK4 integrator for velocity and position updates
- fluidAreas: Tensor of shape [t,i] containing area of each particle (constant)
- fluidDensity: Tensor of shape [t,i] containing the current summation density of each particle
- fluidPosition: Tensor of shape [t,i] containing the current position of each particle
- fluidPressure: Tensor of shape [t,i] containing the current pressure of each particle
- fluidVelocities: Tensor of shape [t,i] containing the current velocity of each particle

## Test Case II/III:

### attrs:
attribute | example | description | legacy
---|---|---
EOSgamma                | 7.0       | Stiffness coefficient for the stiff Tait EOS | yes
alphaDiffusion          | 0.01      | Diffusion coefficient for the standard velocity  | yesdiffusion model
boundaryPressureTerm    | PBSPH     | Method to compute the presssure on the boundary particles if any | yes
boundaryScheme          | solid     | Scheme to sample boundary particles if any | yes
c0                      | 100       | Speed of sound for the EOS, should be $10\times u_max$ | yes
defaultKernel           | wendland2 | Kernel function used for the SPH simulation | yes
deltaDiffusion          | 0.1       | Diffiusion coefficient for the delta-SPH model | yes
densityDiffusionScheme  | MOG       | Density diffusion scheme model, default is MOlteni and coloGrassi | yes
densityScheme           | continuum | continuum indicates that density is integrated over time, summation means instantenous density | yes
device                  | cuda      | Device simulated on | yes
fixedDt                 | True      | Indicates if a fixed timestep was used | yes
floatprecision          | single    | Floating point precision used during simulation | yes
fluidGravity            | [0 0]     | Gravity direction and magnitude, if any | no
fluidPressureTerm       | TaitEOS   | Equation of State model used | yes
initialDt               | 0.001     | Timestep used during the first timestep | yes
integrationScheme       | RK4       | Temporal integration scheme (RK4 means Runge Kutta 4th Order) | yes
kinematicDiffusion      | 0.001     | Diffusion coefficient for actual viscosity | yes
packing                 | 0.398825  | Particle Packing $\Delta x$ | yes
radius                  | 0.017781  | Half the packing distance | no
restDensity             | 998       | Rest density of the fluid | no
shiftingEnabled         | False     | Indicates if Particle Shifting is used | yes
shiftingScheme          | deltaPlus | Particle Shifting Technique | yes
simulationScheme        | deltaSPH  | Simulation Scheme (either deltaSPH or DFSPH) | yes
spacing                 | 0.398825  | Parameter that is usually identical to the particle packing (might differ for incompressible SPH) | yes
staticBoundary          | True      | Indicates if the boundary particles, if any, are dynamic over time | yes
targetNeighbors         | 20        | Expected number of neighbors per SPH particle | no
velocityDiffusionScheme | deltaSPH  | Scheme used for the velocity diffusion  | yes

### boundaryInformation:
Note that if there are no boundary particles, this node may not exist
- boundaryArea: Array of shape [b], contains the area of each boundary particle (may be different for the Akinci boundary model)
- boundaryBodyAssociation: Array of shape [b]: Contains the solid body each particle is associated with
- boundaryNormals: Array of shape [b,d]: Normal associated with the boundary, points towards the outside of the body
- boundaryPosition: Array of shape [b,d]: Position of the particle in simulation space
- boundaryRestDensity: Array of shape [b]: Rest Density of the boundary particle
- boundarySupport: Array of shape [b]: Support radius of the boundary particle
- boundaryVelocity: Array of shape [b,d]: Velocity of the boundary particle (0 if static)

### simulationExport
Contrary to the 1D data, this group contains a subgroup per frame in the simulation export. The key of each group is the frame in %05d format.

### Frame Information:
attrs:
- dt: Current timestep
- time: Current time of the simulation (after integration)
- timestep: Current timestep count of the simulation (after integration)

optional attrs (in new format):
- CFLNumber: 
- averageCompression: 
- averageDensity: 
- dt_a: 
- dt_c: 
- dt_v: 
- maxCompression: 
- maxNeighborCount: 
- maxShift: 
- medianNeighborCount: 
- minCompression: 
- minNeighborCount: 

entries:
- UID: Array of shape [f]: Contains a unique identifier for each particle to track particles over time, necesssary if inlet/outlet conditions are used
- boundaryDensity: Array of shape [f]: Density contribution of the boundary to each fluid particle, _may not exist_
- finalPosition: Array of shape [f,d]: Position of each particle _after_ integration, _may not exist_
- finalVelocity: Array of shape [f,d]: Velocity of each particle _after_ integration, _may not exist_
- fluidAcceleration: Array of shape [f,d]: Acceleration of each particle, i.e., du/dt, _may not exist_
- fluidArea: Array of shape [f]: Area of each particle, should be constant, _may not exist_
- fluidDensity: Array of shape [f]: Current density of each particle, if continuum is used this quantity is integrated over time, otherwise it is recomputed every timestep
- fluidDpdt: Array of shape [f]: Change of density of a particle, i.e., drho/dt, _may not exist_
- fluidPosition: Array of shape [f,d]: Position of each particle _before_ integration
- fluidPressure: Array of shape [f]: Pressure of each particle based on EOS/pressure Solve, _may not exist_
- fluidSupport: Array of shape [f]: Support radius of each particle, should be constant, _may not exist_
- fluidVelocity: Array of shape [f,d]: Velocity of each particle _before_ integration
- fluidShiftAmount: Array of shape [f,d]: Particle Shifting amount, _may not exist_


## Test Case IV:

### attrs:
- dx: 
- jitterAmount: 
- jitterMean: 
- maxDomain: 
- minDomain: 
- numNeighbors: 
- nx: 
- ny: 
- nz: 
- simplexFrequency: 
- simplexScale: 
- support: 
- volume: 

simulationData:
Contains a list of random seeds as subgroups, each subgroup consists of:
attrs:
- frequency: 
- seed: 

and groups:
- gradRhoDifference: 
- gradRhoNaive: 
- gradRhoSymmetric: 
- jitter: 
- ni: 
- rho: 
- vols: 
- x: 