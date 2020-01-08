#!/bin/bash

# ===============================================================
# OVERVIEW
# ===============================================================
#
# Source this file in order to set the proper ENV variables for
# this module. Note that this script will attempt to look for 
# all "dl_*" subdirectories in the current parent folder with a
# corresponding ./setenv.sh file and source those as well.
# 
# ===============================================================
# USAGE
# ===============================================================
# 
# $ source ./setenv.sh
# 
# ===============================================================

echo "Setting ENV variables for module: DL_UTILS"

# --- Set ENV for current module
export DL_UTILS_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$PWD

# --- Source any other scripts for other modules
for SCRIPTS in ../dl_*; do
    if [ $SCRIPTS != "../dl_utils" ]; then 
        cd $SCRIPTS
        source ./setenv.sh
    fi
done

cd ../dl_utils
