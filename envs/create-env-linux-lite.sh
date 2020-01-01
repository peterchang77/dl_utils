#!/bin/bash

yes | conda install python=3.5.0
yes | conda install ipython=5.0.0
yes | conda install -c conda-forge ipdb=0.10.1
yes | conda install -c anaconda cython
yes | conda install -c anaconda dill 

# DICOM
yes | conda install -c conda-forge pydicom
yes | conda install -c conda-forge gdcm
yes | pip install mudicom

# Utilities
yes | conda install -c anaconda requests 
yes | conda install h5py

# Data Science Libraries
# yes | pip install tensorflow-gpu==1.9
yes | pip install tensorflow==1.9
yes | conda install -c anaconda seaborn 
yes | conda install -c anaconda scipy
yes | conda install -c anaconda pandas
yes | conda install -c anaconda scikit-image

# Database
yes | conda install -c anaconda pymongo
yes | pip install redis 
