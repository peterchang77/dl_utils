# Overview

This Python module contains general purpose utilities common to many data science tasks, and is a dependency for all other `dl_*` repositories such as `dl_train` (algorithm training) and `dl_tools` (web-based tools). 

# Configuration

1. Clone the repository 
2. Install dependencies
3. Set shell environment variables 

## Code 

First, clone this repository:

```bash
$ git clone https://github.com/peterchang77/dl_utils
``` 

Note: if you are planning to use other `dl_*` packages, it is recommended to keep all such repositories in the same parent folder. For example:

```
|--- parent/
   |--- dl_utils
   |--- dl_train
   |--- dl_tools
   ...
```

## Dependencies

The **recommended** approach is to perform all code development within Docker containers. A series of prebuilt Docker images are configured with all the necessary Python dependencies as well as any OS configurations. They can be downloaded using any one of the following commands:

```bash
$ sudo docker pull peterchang77/gpu-full:latest
$ sudo docker pull peterchang77/gpu-lite:latest
$ sudo docker pull peterchang77/cpu-full:latest
$ sudo docker pull peterchang77/cpu-lite:latest
```

Use the `full` version for a comprehensive suite of data science package and tools, compared to the `lite` version which contains only the essential dependencies. Use `gpu` if you are own a NVIDIA GPU-enabled device, otherwise use `cpu`. 

For more information about these Docker images, as well as further details for installation of the Docker runtime and running these containers interactively, refer to the following repository: https://github.com/peterchang77/install.

### Python Virtual Environment

As an alternative to Docker images, a series of `*.yml` files has been exported with the necessary Python dependencies and versions, available in the `/envs/` subdirectory of this repository. The four permutations reflect the `full`, `lite`, `gpu` and `cpu` versions of the libraries as described above. The Conda `*.yml` files can be installed using any one of the following commands:

```bash
conda env create -f env-linux-gpu-full.yml
conda env create -f env-linux-cpu-full.yml
conda env create -f env-linux-gpu-lite.yml
conda env create -f env-linux-gpu-lite.yml
```

### Bash Script

If you need to manually install all Python dependencies, a series of Bash scripts have been provided to recreate the development environment. Note, this method does not carefully control for specific library versions and is **not** recommended unless absolutely necessary.

## Environment Variables

After installing both code and dependencies, subsequent usage of the `dl_utils` library (and any other `dl_*` repository) requires setting several shell environment variables as well appending the current repository root in `$PYTHONPATH`. This can be done by sourcing the `setenv.sh` script provided in the root directory of this repository:

```bash
source ./setenv.sh
```

All other `dl_*` repositories contain an identically named `./setenv.sh` file that can be sourced to set the necessary environment variables for that additional repository. 

**NOTE:** For convenience, the specific `./setenv.sh` file in this repository e.g. `dl_utils/setenv.sh` will search for all other such scripts in all other directories named `dl_*` and source those as well. Thus if you are working with multiple other `dl_*` repositories, simply source this single `./setenv.sh` file to automatically trigger the remaining files. 

# Modules

## I/O

This `io` module contains a general purpose wrapper for loading various file formats. It can be easily extended for custom file formats. 

**Usage**

```python
from dl_utils import io
data, meta = io.load(...)
```

The method will return two objects, `data` and `meta`, regardless of file format. The `data` object is a 4D Numpy array in format Z x Y x X x C. The `meta` object is a Python dictionary that contains, if available:

* `meta['affine']`: a 4x4 affine transform matrix
* `meta['header']`: DICOM headers

Depending on other file formats, other additional key-value pairs may be populated.

## General

The `general` module contains several common utility functions. The most popular is a series of formatted print commands that prepend a timestamp on the output:

* `printd(...)`: standard print + timestamp
* `printr(...)`: use the `end='\r'` flag to rewrite the same line
* `printp(...)`: print progress bar with message

**Usage**
```python
from dl_utils.general import printd, printr, printp

# --- Single line output
printd('Action initiated.')
printd('Action complete.')

# --- Series of single line outputs
for i in range(100):
    printr('Iteration number: {}'.format(i))

# --- Series of outputs with progress bar
for i in range(100):
    printp('Iteration number: {}'.format(i), i / 99)
```

## Database

The `db` module contains code for table-based data management backed via `*.csv` files. Each row represents a single case (e.g. study, patient, etc) and each column represents either a filename (`db.fname`) or derived metadata (`db.header`). Advanced functionality includes support for complex mapping operations and multiprocessed distribution of tasks on a dataset.

## Display

The `display` module contains code for visualization of medical imaging data using the `matplotlib` library.

**Usage**

```python
from dl_utils.display import imshow
imshow(...)
```

# Tutorials
