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

The **recommended** approach is to perform all code development within Docker containers. A series of prebuilt Docker images are configured with all the necessary Python dependencies as well as any OS installations. Then can pulled using the following commands:

```bash
$ sudo docker pull peterchang77/gpu-full:latest
$ sudo docker pull peterchang77/gpu-lite:latest
$ sudo docker pull peterchang77/cpu-full:latest
$ sudo docker pull peterchang77/cpu-lite:latest
```

Use the `full` version for a comprehensive suite of data science package and tools, compared to the `lite` version which contains only the essential dependencies. Use `gpu` if you are own a NVIDIA GPU-enabled device, otherwise `cpu`. 

For more information about these Docker images, as well as further details for installation of the Docker runtime, refer to the following repository: https://github.com/peterchang77/install.

## Environment Variables

# Modules

# Tutorials
