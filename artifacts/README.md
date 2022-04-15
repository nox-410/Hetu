# SD-PIPE

## Installation
1. Clone the repository.

2. Prepare the environment. We use Anaconda to manage packages. The following command create the conda environment to be used:`conda env create -f environment.yml`. The environment requires Cuda toolkit version 10.2.

3. We use CMake to compile Hetu. Please copy the example configuration for compilation by `cp cmake/config.example.cmake cmake/config.cmake`. If your system is using another version of NCCL/cudnn/MPI, you should manually change the path defined in this config file.

```bash
# compile
# make all
mkdir build && cd build && cmake ..
make -j
```

4. Run `conda activate pipe` and prepare PYTHONPATH by executing the command `source hetu.exp` at the root folder.

5. Run python and try `import hetu` to check whether the compilation is successful.

## Usage

