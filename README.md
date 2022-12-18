# DeepMerge: a unified framework for diagonal integration of multi-batch multimodal single-cell omics data

DeepMerge is a method for batch correcting and integrating multimodal single-cell omics data using a multi-task deep learning framework. DeepMerge not only performs batch correction and integration across data modalities but also generates normalised and corrected data matrices that can be readily utilised for downstream analyses such as identifying differentially expressed genes, ADTs, and/or cis-regulatory elements (CREs) from multiple modalities of the batch corrected and integrated dataset. By applying DeepMerge to a large collection of datasets generated from various biotechnological platforms, we demonstrate its utility for integrative analyses of multi-batch multimodal single-cell omics datasets.


<img width=100% src="https://github.com/liuchunlei0430/DeepMerge/blob/main/img/main.png"/>


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

## Preparing intput for Matilda
DeepMerge’s main function takes expression data (e.g., RNA, ADT, ATAC) in `.h5` format and cell type labels in `.csv` format. DeepMerge expects raw count data for RNA and ADT modalities. For ATAC modality, DeepMerge expects the raw count data.

An example for creating .h5 file from expression matrix in the R environment is as below:
```
write_h5 <- function(exprs_list, h5file_list) {  
  for (i in seq_along(exprs_list)) {
    h5createFile(h5file_list[i])
    h5createGroup(h5file_list[i], "matrix")
    writeHDF5Array(t((exprs_list[[i]])), h5file_list[i], name = "matrix/data")
    h5write(rownames(exprs_list[[i]]), h5file_list[i], name = "matrix/features")
    h5write(colnames(exprs_list[[i]]), h5file_list[i], name = "matrix/barcodes")
  }  
}
write_h5(exprs_list = list(rna = train_rna, h5file_list = "/DeepMerge/data/TEA-seq/train_rna.h5")
```

### Example dataset

As an example, the processed CITE-seq dataset by RAMA et al. (GSMxxxx) is provided for the example run, which is saved in `./DeepMerge/data/CITEseq/Rama/`.
Users can prepare the example dataset as input for DeepMerge or use their own datasets.
Training and testing on demo dataset will cost no more than 1 minute with GPU.

## Running DeepMerge with the example dataset
### Training the DeepMerge model (see Arguments section for more details).
```
cd DeepMerge
cd main
python main.py --rna [path_RNA] --adt [path_ADT] --atac [path_ATAC] --cty [path_cty] --batch [path_batch]
# Example run
python main.py --rna ../data/CITEseq/Rama/rna.h5 --adt ../data/CITEseq/Rama/adt.h5 --cty ../data/CITEseq/Rama/cty.csv --batch ../data/CITEseq/Rama/batch.csv
```

### Argument
Training dataset information
+ `--rna`: path to training data RNA modality.
+ `--adt`: path to training data ADT modality (can be null if ATAC is provided).
+ `--atac`: path to training data ATAC modality (can be null if ADT is provided). Note ATAC data should be summarised to the gene level as "gene activity score".
+ `--cty`: path to the labels of training data.

Training and model config
+ `--batch_size`: Batch size (set as 64 by default)
+ `--epochs`: Number of epochs.
+ `--lr`: Learning rate.
+ `--z_dim`: Dimension of latent space.
+ `--hidden_rna`: Dimension of RNA branch.
+ `--hidden_adt`: Dimension of ADT branch.
+ `--hidden_atac`: Dimension of ATAC branch.

Other config
+ `--seed`: The random seed for training.
+ `--augmentation`: Whether to augment simulated data.

Note: after training, the model will be saved in `./Matilda/trained_model/`.


## Reference
[1] Ramaswamy, A. et al. Immune dysregulation and autoreactivity correlate with disease severity in
SARS-CoV-2-associated multisystem inflammatory syndrome in children. Immunity 54, 1083–
1095.e7 (2021).

[2] Ma, A., McDermaid, A., Xu, J., Chang, Y. & Ma, Q. Integrative Methods and Practical Challenges
for Single-Cell Multi-omics. Trends Biotechnol. 38, 1007–1022 (2020).

[3] Swanson, E. et al. Simultaneous trimodal single-cell measurement of transcripts, epitopes, and
chromatin accessibility using TEA-seq. Elife 10, (2021).

## License

This project is covered under the Apache 2.0 License.
