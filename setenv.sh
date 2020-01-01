#!/bin/bash

# --- Set default values
LIB=${1:-lite}

# --- Set environ variables
export DL_UTILS_ROOT=$PWD 

# --- Set PYTHONPATH 
export PYTHONPATH=$PYTHONPATH:$PWD/$LIB
