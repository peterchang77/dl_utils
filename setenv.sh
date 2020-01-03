#!/bin/bash

# ============================================================
# SET ENVIRONMENT VARIABLES 
# ============================================================
# 
# This bash script will attempt to look for all dl_* repos
# that are available in the parent directory of the current
# folder and add all matches to $PYTHONPATH.
# 
# IMPORTANT: if the remaining dl_* repos are in a different 
# location on your machine, please manually add here.
# 
# ============================================================
# USAGE
# ============================================================
# 
# $ source ./setenv.sh [lite/full]
# 
# By default, the lite version of the dl_utils libary is used.
# 
# ============================================================

# --- (0) Add DL_UTILS_ROOT, PATH
export DL_UTILS_ROOT=$PWD
export PATH=$PATH:$DL_UTILS_ROOT/bins

# --- (1) Add current dl_utils libary path
LIB=${1:-lite}
echo "Adding to PYTHONPATH: $PWD/$LIB"
export PYTHONPATH=$PYTHONPATH:$PWD/$LIB

# --- (2) Add all other dl_* paths in parent directory
for DL_DIR in ../dl_*
do
    DL_DIR=$(readlink -f $DL_DIR)
    if [ "$DL_DIR" != "$PWD" ]
    then
        echo "Adding to PYTHONPATH: $DL_DIR"
        export PYTHONPATH=$PYTHONPATH:$DL_DIR
    fi
done

