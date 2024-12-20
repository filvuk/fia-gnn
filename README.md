# Predicting Lewis Acidity: Machine Learning the Fluoride Ion Affinity of _p_-Block Atom-based Molecules

This repository corresponds to  
L. M. Sigmund, S. Sowndarya S. V., A. Albers, P. Erdmann, R. S. Paton, L. Greb, _Angew. Chem. Int. Ed._ **2024**, _63_, e202401084, [DOI: 10.1002/anie.202401084](https://doi.org/10.1002/anie.202401084).  
The FIA49k dataset and all other data files can be downloaded from the corresponding [figshare project](https://figshare.com/projects/FIA-GNN/187050) and should be placed in the "data" folder of this repository to run the provided Jupyter notebooks and Python scripts.  

## Use FIA-GNN via the Web App

The FIA-GNN models can be used free of charge at [this homepage](https://www.grebgroup.de/fia-gnn/).  

## Use FIA-GNN within a Jupyter Notebook

The ```deploy_fia_gnn.ipynb``` notebook demonstrates how to use the trained models of FIA-GNN as published.  

## Train FIA-GNN

If model training is intended, run the ```run_preprocessing.py``` script. This will create the preprocessed graph inputs.

```bash
python run_preprocessing.py 
```

Preprocessing of the data is done simultaneously for FIA<sub>gas</sub> and FIA<sub>solv</sub>. For the preprocessing to be successful, the provided data file must contain a column named "la_smiles" with the SMILES strings of the Lewis acids, as well as columns named "fia_gas-DSDBLYP" and "fia_solv-DSDBLYP", with the respective FIA values for gas and solution phase.  
  
Next, the model can be trained with the ```train_fia_gnn.py``` script. Training is either done for FIA<sub>gas</sub> or FIA<sub>solv</sub>.

```bash
python train_fia_gnn.py 
```
  
A respective conda environment for training can be obtained with

```bash
cd fia_gnn/
conda env create -f tf2_gpu_fia_gnn.yml -n fia_gnn_tf_env
conda activate fia_gnn_tf_env
```
