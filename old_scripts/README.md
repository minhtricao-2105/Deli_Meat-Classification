# Hyperspectral Data Classification

Framework for using classification models defined in the [scikit-learn](https://scikit-learn.org/) python library with hyperspectral data.

## Requirements

This code has been tested with python 2.7 and python 3.6 (note that the MLP and Gaussian process classifiers are not available with python 2.7).

Python packages installed by `python3.6 -m pip install <package_name>`
- numpy
- scipy
- matplotlib
- scikit-learn
- pickle
- opencv-python

## Data

Data is stored as .csv files. Each file stores the reflectance values of pixels for the whole dataset. Each row in the .csv file has the following format:

| Class          | Training/Testing (0/1) | u    | v    | Reflectance values of 146 bands ... |
| -------------- | ---------------------- | ---- | ---- | ----------------------------------- |
| ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

The first four columns contain the metadata for each pixel: the class label, train/test membership (0 if pixel is for training or 1 for testing) and finally the image pixel coordinates. The remaining columns are the reflectance values for the 146 wavelengths. Classification assumes that all data is normalised using a white and dark reference material or calibrated with the modelled light source.

Different datasets within a common root directory can be used to change the training and testing sources. For example, the `deli_meats_4` dataset has data collected separately on a *flat* and *curved* surface. Each of these different datasets can be used interchangably for training and testing.

## Evaluation

Classification with various models as defined in [scikit-learn](https://scikit-learn.org/) is performed with the [evaluate.py](https://code.research.uts.edu.au/uts-cas/awi/fleece_processing/-/blob/master/scripts/hyperspectral-classification/evaluate.py) script. The script is run with the following arguments:

- `--root` directory where all common datasets are stored
- `--dataset` name of the dataset to process
- `--train` training source if the dataset has different variations
- `--test` testing source if the dataset has different variations; if not specified, test samples are taken from the *train* source \[default: ''\]
- `--classifier` name of the classifier; default is *all* which means all available classifiers are analysed \[default: *all*\]
- `--preprocess-type` type of preprocessing to apply; either *minmax* for [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) or *standard* for [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) \[default: *standard*\]
- `--pca` integer to specify the number of components to reduce the features to using [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html); *-1* if no dimensionality reduction is required, *0* if dimensions are to be reduced to a minimum explanation variance \[default: -1\]
- `--down-sample` toggle (0/1) if training data should be downsampled to a maximum of 500 elements \[default: 0\]
- `--use-validation` toggle (0/1) if testing data should be the smaller validation set (used for optimising model parameters) \[default: 0\]
- `--save` toggle (0/1) if classification results should be written to a [pickle](https://docs.python.org/3/library/pickle.html) file (this can be loaded to re-print and plot results) \[default: 0\]

Note that the parameters for each classifier should be specified in the file [classifier_parameters.json](https://code.research.uts.edu.au/uts-cas/awi/fleece_processing/-/blob/master/scripts/hyperspectral-classification/classifier_parameters.json). If this file is not found, default parameters will be used.

Parameters for each dataset are specified in [dataset_parameters.json](https://code.research.uts.edu.au/uts-cas/awi/fleece_processing/-/blob/master/scripts/hyperspectral-classification/dataset_parameters.json). These are mainly for visualisation.

An example for running this is as follows:

```
python3 evaluate.py --root <path_to_data> --dataset deli_meats_4 --train flat --test curved --classifier svm_rbf --preprocess-type standard --pca -1 --down-sample 0 --use-validation 0 --save 0
```

Running this script will print the classification performance of all models in the terminal. Additionally, ROC curves will be plotted. If RGB images of the testing scenario are provided, all testing pixels will be coloured green or red if they are classified correct or incorrect, respectively. If `--save` is *1*, the results will be written to a pickle file in the current directory.

## Visualisation

The results can be quickly visualised by loading a saved pickle file with the [plot_from_pkl.py](https://code.research.uts.edu.au/uts-cas/awi/fleece_processing/-/blob/master/scripts/hyperspectral-classification/plot_from_pkl.py) script. The script is run with the following arguments:

- `--file` pickle file of stored results
- `--dataset` name of the dataset to process

## Parameter Optimisation

Find the best model hyperparameters with [optimise_parameters.py](https://code.research.uts.edu.au/uts-cas/awi/fleece_processing/-/blob/master/scripts/hyperspectral-classification/optimise_parameters.py) to generate a `.json` file similar to [classifier_parameters.json](https://code.research.uts.edu.au/uts-cas/awi/fleece_processing/-/blob/master/scripts/hyperspectral-classification/classifier_parameters.json). The script is run with the following arguments:

- `--root` directory where all common datasets are stored
- `--dataset` name of the dataset to use
- `--train` training source if the dataset has different variations
- `--test` testing source if the dataset has different variations; if not specified, test samples are taken from the *train* source \[default: ''\]
- `--classifier` name of the classifier; default is *all* which means all available classifiers are analysed \[default: *all*\]
- `--preprocess-type` type of preprocessing to apply; either *minmax* for [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) or *standard* for [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) \[default: *standard*\]
- `--pca` integer to specify the number of components to reduce the features to using [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html); *-1* if no dimensionality reduction is required, *0* if dimensions are to be reduced to a minimum explanation variance \[default: -1\]
- `--save` toggle (0/1) if classification results should be written to a [pickle](https://docs.python.org/3/library/pickle.html) file (this can be loaded to re-print and plot results) \[default: 0\]

Running this script will save a new file `opt_params.json` in the current directory.
