# Installing Gpufit
Significant speed gains can be acquired by utilizing GPU acceleration for computationally intensive portions of this software. This has been achieved by using a [fork](https://github.com/ironictoo/Gpufit) of the [gpufit](https://github.com/gpufit/Gpufit) library, which so far has implemented T1 mapping and various curve fitting models such as Patlak on NVIDIA GPUs (via CUDA).

It is highly recommended to enable GPU acceleration so that you may enjoy a huge decrease in runtime with marginally greater or equal accuracy.

## Installing via pre-compiled binaries (WIP)
1. Go to the [release page](https://github.com/ironictoo/Gpufit/releases) and install an appropriate archive for your system.

2. 

## Building from source
_Requires CMake_
1. Clone [our fork of gpufit](https://github.com/ironictoo/Gpufit)

`git clone https://github.com/ironictoo/Gpufit.git`

2. Open [CMake](https://cmake.org/download/) and set the source folder and build folder. Check the models under `GPUFIT` that you wish to use or test. You may hover your cursor over most options to see a brief explanation. 

Enabling too many models, particularly multiple Dixon models, may cause a memory issue.

We recommend checking the `USE_CUBLAS` option if cuBLAS is installed. It is also perfectly fine to leave it unchecked.

3. Click `Configure` and `Generate`. Hopefully there are no errors.

4. Open a terminal in your build folder and compile with `make`.

Gpufit should then be ready for use. The program will attempt to detect if you have an NVIDIA GPU and the Gpufit MATLAB functions.

If you wish to use the CPU, you may set `force_cpu = 1` in `dce/dce_preferences.txt`.