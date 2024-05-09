# Symmetric Fourier Basis Convolutions for Learning Lagrangian Fluid Simulations

Authors: Rene Winchenbach, Nils Thuerey

Accepted at: International Conference on Learning Representation (ICLR) 2024 - Vienna (as a Poster)

If you have any questions on the paper itself or Continuous Convolutions in general, feel free to reach out to me via rene.winchenbach@tum.de

Frequently asked questions about the paper will appear here as they are asked.

Q: Can the method handle oversampled or undersampled data during inference related to training?
A: No. The internal formulation of the Graph Convolution does not account for the volume of the contributing nodes, which works for our purposes as we are only dealing with uniformly sized particles and thus the volume is a constant term that can be ignored. However, if you wanted to do this you would need to change the Graph Convolution formulation to include volume.

Q: What about scaling to other resolutions?
A: Our method cannot handle this as there is some general uncertainty about particle resolutions as (a) scalar quantities in our SPH data scale with $h^{-d}$ and gradients with $h^{-d-1}$ and thus learning resolution varying quantities would require including this support much more tightly into the training. While this could potentially work, we did not investigate this for now.

Q: What about adaptive resolutions?
A: Similar to the prior question, this could conceptually be added to the network but makes the training significantly more complicated as there are many potential constellations of relative sizes and distributions a particle can see. This would make the dataset generation significantly harder and was beyond our scope.

Q: What is the resolution to Fourier Neural Operators/Do you use FFTs?
A: No. Our method works in normal coordinate space and not in a global frequency space. Instead our method works on a local limited convolution where we use the Fourier Terms to represent a filter function. 

Q: What about other appications?
A: We did try our method on some general Pattern Recognitiion tasks but including these results was beyond our scope. Our Codebase can be applied to general graph tasks similar to how pyTorch geometric works.

Q: What about larger simulations?
A: Our method can readily do this due to its local Graph Convolution architecture. Accordingly, we can simply expand the simulation domain to be larger. Furthermore, the memory consumption during inference is relatively small so it would be possible to use multiple orders of magnitude more particles during inference if so desired.

Q: How do you encode boundaries?
A: Periodic boundaries are modeled by connecting particles across the periodic boundary with appropriate modular distances. Rigid boundaries are included with rigid particles with a seperate CConv on the first network layer.



Repository: https://github.com/tum-pbs/SFBC
ArXiV Paper: https://arxiv.org/abs/2403.16680