# DeepMerge: a unified framework for diagonal integration of multi-batch multi-modal single-cell omics data

DeepMerge is a method for batch correcting and integrating multimodal single-cell omics data using a multi-task deep learning framework. DeepMerge not only performs batch correction and integration across data modalities but also generates normalised and corrected data matrices that can be readily utilised for downstream analyses such as identifying differentially expressed genes, ADTs, and/or cis-regulatory elements (CREs) from multiple modalities of the batch corrected and integrated dataset.

<img width=100% src="https://github.com/liuchunlei0430/Matilda/blob/main/img/main.jpg"/>


## Installation
DeepMerge is developed using PyTorch 1.9.1. We recommend using conda enviroment to install and run DeepMerge. We assume conda is installed. You can use the provided environment or install the environment by yourself accoring to your hardware settings. Note the following installation code snippets were tested on a Ubuntu system (v20.04) with NVIDIA GeForce 3090 GPU. The installation process needs about 15 minutes.

### Installation using provided environment
Step 1: Create and activate the conda environment for DeepMerge using our provided file
```
conda env create -f environment_deepmerge.yaml
conda activate environment_deepmerge
```

Step 2:
The following python packages are required for running DeepMerge: h5py, numpy, pandas, tqdm, scipy, scanpy. They can be installed in the conda environment as below:
```
pip install h5py
pip install numpy
pip install pandas
pip install tqdm
pip install scipy
pip install scanpy
```

Step 3:
Otain DeepMerge by clonning the github repository:
```
git clone https://github.com/liuchunlei0430/DeepMerge.git
```

### Installation by youself

Step 1:
Create and activate the conda environment for DeepMerge
```
conda create -n environment_deepmerge python=3.8
conda activate environment_deepmerge
```

Step 2:
Check the environment including GPU settings and the highest CUDA version allowed by the GPU.
```
nvidia-smi
```

Step 3:
Install pytorch and cuda version based on your GPU settings.
```
# Example code for installing CUDA 11.3
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Step 4:
The following python packages are required for running DeepMerge: h5py, numpy, pandas, tqdm, scipy, scanpy. They can be installed in the conda environment as below:
```
pip install h5py
pip install numpy
pip install pandas
pip install tqdm
pip install scipy
pip install scanpy
```

Step 5:
Otain DeepMerge by clonning the github repository:
```
git clone https://github.com/liuchunlei0430/DeepMerge.git
```
