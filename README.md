# **GPUCell**
GPUCell is a GPU-accelerated program that simulates Cellular scale blood flow using the smoothed dissipative particle dynamics(SDPD) method.
GPUCell(https://github.com/gidar2020/GPUCell.git) was developed by Ye's group to simulate cellular behavior, which includes cell aggregation and cell deformation.

# **Code**
## main.cuh
## main.cu
The main loop of the program.
## Particle.cuh
## Particle.cu
CPU functions implementation , including reading and output data, and setting the initial state of the particle.
## ParticleGPU.cuh
## ParticleGPU.cu
GPU kernel functions implementation, including SDPD force, cell deformation, aggregation force, etc.
## ConfigDomain.cuh
## ConfigDomain.cu
GPU kernel functions implementation, including construct of neighbor list of particles. 
## TypesDef.cuh
Some self-defined structures used in programs.
## erro.cuh
A marco function to detect CUDA runtime erros.
## makefile
