conda create -n torch_sfbc python=3.11 -y
conda activate torch_sfbc
conda install -c anaconda ipykernel -y
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit cudnn -y
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install tqdm seaborn pandas matplotlib numpy tomli msgpack msgpack-numpy portalocker h5py zstandard ipykernel ipympl 
pip install scipy scikit-image scikit-learn
<!-- pip install numba -->