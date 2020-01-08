#!/bin/bash

yes | pip install --upgrade pip
yes | conda update -n base conda

# Base
yes | conda install python=3.6
yes | conda install ipython
yes | conda install -c conda-forge ipdb
yes | conda install -c anaconda cython

# DICOM
yes | conda install -c conda-forge pydicom
yes | conda install -c conda-forge gdcm
yes | pip install mudicom

# Utilities
yes | conda install -c anaconda requests 
yes | conda install -c anaconda dill 
yes | conda install h5py
yes | conda install pyyaml

# Data Science Libraries
yes | pip install tensorflow-gpu==2.0
# yes | pip install tensorflow==2.0
yes | conda install -c anaconda scipy
yes | conda install -c anaconda pandas
yes | conda install -c anaconda scikit-image
yes | conda install -c anaconda scikit-learn

# Database
yes | conda install -c anaconda pymongo
yes | pip install redis 
