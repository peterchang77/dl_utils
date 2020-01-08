#!/bin/bash

yes | pip install --upgrade pip
yes | conda update -n base conda

# Base
yes | conda install python=3.6
yes | conda install -c conda-forge jupyterlab
yes | conda install -c conda-forge jupyterhub
yes | conda install ipython
yes | conda install -c conda-forge ipdb
yes | conda install -c anaconda cython

# Imaging 
yes | conda install -c conda-forge pydicom
yes | conda install -c conda-forge gdcm
yes | pip install mudicom
yes | conda install -c conda-forge nibabel
yes | conda install -c anaconda pillow
yes | conda install -c simpleitk simpleitk
yes | pip install opencv-python
yes | pip install openslide-python

# Utilities
yes | conda install -c anaconda requests 
yes | conda install -c anaconda dill 
yes | conda install h5py
yes | conda install pyyaml

# Data Science Libraries
# yes | pip install tensorflow-gpu==2.0
yes | pip install tensorflow==2.0
yes | conda install -c anaconda scipy
yes | conda install -c anaconda seaborn 
yes | conda install -c anaconda pandas
yes | conda install -c anaconda xlrd 
yes | conda install -c anaconda scikit-image
yes | conda install -c anaconda scikit-learn

# Flask
yes | conda install -c anaconda flask
yes | conda install -c conda-forge flask-socketio

# Database
yes | conda install -c anaconda pymongo
yes | pip install redis 
