# Symmetric Fourier Basis Convolutions for Learning Lagrangian Fluid Simulations

Authors: Rene Winchenbach, Nils Thuerey

Accepted at: International Conference on Learning Representation (ICLR) 2024 - Vienna (as a Poster)

Repository: https://github.com/tum-pbs/SFBC
ArXiV Paper: https://arxiv.org/abs/2403.16680

## Supported Basis Functions:

As basis Functions our code base supports:

- Radial Basis Functions ('__gaussian__', '__multiquadric__', '__inverse_quadric__', '__inverse_multiquadric__', '__polyharmonic__', '__bump__')
- Interpolation Schemes ('__linear__' [This is the approach used by the CConv paper by Ummenhofer et al], '__square__' [Nearest Neighbor Interpolation])
- B-Spline Schemes ('__cubic_spline__' [This is the SplineConv basis from Fey et al], '__quartic_spline__', '__quintic_spline__')
- SPH Kernels ('__wendland2__', '__wendland4__', '__wendland6__', '__poly6__', '__spiky__')

All of these can be either called as they are ('__rbf x__'), normalized to be partitions of unity by widening their shape parameter ('__abf x__'), or normalized by dividing by the shape function ('__ubf x__'). To replicate the ressults of Fey et al using 'abf cubic_spline' is necessary, for example.

We also offer:

- Fourier Terms ('__ffourier__' [Our primary approach], '__fourier__' [which drops the first antisymmetric term]), each with suffixes ' even' and ' odd' to use even and odd symmetric terms
- Chebyshev Terms ('__chebyshev__', '__chebyshev2__')
- An antisymmetric enforcement term ('__dmcf__' [This is the approach by Prantl et al])

## Network Setup:

The code provides two primary classes _BasisConv_ and _BasisNetwork_. The former is an individual basis convolution layer and the second is the network setup used for our publication.

### Convolution Layer

The _BasisConv_ class has the following arguments (src/BasisConvolution/convLayerv2.py):

```py
BasisConv(inputFeatures: int, outputFeatures: int, dim: int = 2,
        basisTerms = [4, 4], basisFunction = 'linear', basisPeriodicity = False,
        linearLayerActive = False, linearLayerHiddenLayout = [32, 32], linearLayerActivation = 'relu',
        biasActive= False, feedThrough = False,
        preActivation None, postActivation = None, cutlassBatchSize = 16,
        cutlassNormalization = False, initializer = 'uniform', optimizeWeights = False, exponentialDecay = False
    )
```

<!-- The arguments are:

__inputFeatures__ : [_int_]: The number of features given to the network for which the convoluution is performed
__outputFeatures__ : [_int_]: The number of features of the messages created (and agglomorated) by the network
__dim__ : [_int_]: Spatial Dimensionality of the data
__basisTerms__ : [_Union[int, List[int]]_] = [4, 4]: 
__basisFunction__ : [_Union[int, List[int]]_] = 'linear': 
__basisPeriodicity__ : [_Union[bool, List[bool]]_] = False: 

__linearLayerActive__: [_bool_] = False: 
__linearLayerHiddenLayout__: [_List[int]_] = [32, 32]: 
__linearLayerActivation__: [_str_] = 'relu': 

__biasActive__: bool = False: 
__feedThrough__: bool = False: 

__preActivation__: Optional[str] = None: 
__postActivation__: Optional[str] = None: 

__cutlassBatchSize__ = 16: 

__cutlassNormalization__ = False: 
__initializer__ = 'uniform': 
__optimizeWeights__ = False: 
__exponentialDecay__ = False:  -->

### Convolution Network


The _BasisNetwork_ class has the following arguments (src/BasisConvolution/convNetv2.py):

```py
BasisNetwork(self, fluidFeatures, boundaryFeatures = 0, layers = [32,64,64,2], 
    denseLayer = True, 
    activation = 'relu', coordinateMapping = 'cartesian', 
    dims = [8],  rbfs = ['linear', 'linear'], windowFn = None, 
    batchSize = 32, ignoreCenter = True, 
    normalized = False, outputScaling = 1/128, 
    layerMLP = False, MLPLayout = [32,32], 
    convBias = False, outputBias = True, 
    initializer = 'uniform', optimizeWeights = False, exponentialDecay = True, 
    inputEncoder = None, outputDecoder = None, edgeMLP = None, vertexMLP = None):
```

For more information, see the respective source files.

### Inference

The primary forward function of the network can be called as
```py
model(fluidFeatures, fi, fj, fluidEdgeLengths, 
        boundaryFeatures, bf, bb, boundaryEdgeLenghts)
```

For this call fluidFeatures are per-vertex features for the primary point cloud and boundaryFeatures are per-vertex features for the secondary point cloud. [fi, fj] are the adjacency matrix of the primary point cloud in COO format and [bf, bb] are the adjacency matrix from the secondary (bb) to the primary (bf) point cloud. fluidEdgeLengths and boundaryEdgeLengths are the relative distances between nodes normalized by the node support radius (i.e., in the range of $[-1,1]^d$). The boundary information can be 'None' if not used. For convenience we also have a _runInference_ function (util/network) that takes as argument the simulation state, config and the model.

## Training example

As an example for training, see notebooks/exampleTraining. This notebook contains a simple training script that learns the SPH density kernel function for any of the four included datasets in a small ablation study. Here's an example result of the training for test case II with 5 different basis functions (Fourier, Fourier even terms only, Fourier odd terms only, Linear and Chebyshev) with 3 different basis term counts (2,4,8):

![alt text](https://github.com/tum-pbs/SFBC/raw/main/figures/train.png)

You can also find an example of this ablation study on [Google Colab](https://colab.research.google.com/drive/1p0NChJwexFaNUtRKvEEtkDHf9AsONVCI?usp=sharing)


## Datasets:

This paper included four datasets in its evalutions. You can find a tool to visualize the datasets under notebooks/datasetVisualizer. Summary information:

Test Case | Scenario | Size | Link
---|---|---|---
I | compressible 1D | 7.9GByte | [https://huggingface.co/datasets/Wi-Re/SFBC_dataset_I](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_I)
II | WCSPH 2D | 45 GByte | [https://huggingface.co/datasets/Wi-Re/SFBC_dataset_II](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_II)
III | IISPH 2D | 2.1 GByte | [https://huggingface.co/datasets/Wi-Re/SFBC_dataset_III](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_III)
IV | 3D Toy | 1.2 GByte | [https://huggingface.co/datasets/Wi-Re/SFBC_dataset_IV](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_IV)

### Test Case I:

This test case was a pseudo-compressible 1D SPH simulation with random initial conditions. The dataset comprises 36 files with 2048 timesteps and 2048 particles each. Example:

![alt text](https://github.com/tum-pbs/SFBC/raw/main/figures/image.png)

You can find the dataset [here](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_I) (size approximately 7.9 GByte), the dataset is also a submodule in this repo under datasets/SFBC_dataset_I.

### Test Case II:

This test case was a weakly-compressible 2D SPH simulation with random initial conditions and enclosed by a rigid boundary. The dataset comprises 36 simulations for training and 16 for testing each with 4096 timesteps and 4096 particles. Example:

![alt text](https://github.com/tum-pbs/SFBC/raw/main/figures/image-1.png)

You can find the dataset [here](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_II) (size approximately 45 GByte), the dataset is also a submodule in this repo under datasets/SFBC_dataset_II.

### Test Case III:

This test case was an incompressible 2D SPH simulation where to randomly sized blobs of liquid collide in free-space. The dataset comprises 64 simulations for training and 4 for testing each with 128 timesteps and 4096 particles. Example:

![alt text](https://github.com/tum-pbs/SFBC/raw/main/figures/image-2.png)

You can find the dataset [here](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_III) (size approximately 2.1 GByte), the dataset is also a submodule in this repo under datasets/SFBC_dataset_III.

### Test Case IV:

The last test case is a toy-problem to evaluate SPH kernel learning in a 3D setting. For this setup we sampled 4096 particles in a $[-1,1]^3$ domain with random (including negative) massses and additional jitter on the particle possitions. The test set contains a setup with no jitter (1024 seeds), low jitter (1024 seeds), medium jitter (1 seed) and high jitter (1 seed). Example (low jitter): 

![alt text](https://github.com/tum-pbs/SFBC/raw/main/figures/image-3.png)

You can find the dataset [here](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_IV) (size approximately 1.2 GByte), the dataset is also a submodule in this repo under datasets/SFBC_dataset_IV.

## Requirements

To setup a conda environment for this code base simply run:
```bash
conda create -n torch_sfbc python=3.11 -y
conda activate torch_sfbc
conda install -c anaconda ipykernel -y
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cudnn -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install tqdm seaborn pandas matplotlib numpy tomli msgpack msgpack-numpy portalocker h5py zstandard ipykernel ipympl 
pip install scipy scikit-image scikit-learn
```

If you would like to use a faster and less memory intensive neighbor search to build your networks, especially for larger simulations, consider using our [torchCompactRadius](https://pypi.org/project/torchCompactRadius/) package (`pip install torchCompactRadius`) which uses a C++/Cuda implementation of a compact hash map based neigbor search. Note that this module performs a just in time compilation on its first use for new configurations and may take some time (it will time out your colab instance as a free user).

--- 
This work was supported by the __DFG Individual Research Grant TH 2034/1-2__. 