source=main.cu

RUN:=./sdpdgpu

CC=/data/apps/cuda/11.4/bin/nvcc

$(RUN):$(source) Particle.o  ParticleGPU.o ConfigDomain.o
	$(CC) --device-debug --debug $(source) Particle.o  ParticleGPU.o ConfigDomain.o -o $(RUN)
	
Particle.o:Particle.cu
	$(CC) --device-debug --debug --device-c Particle.cu -o Particle.o
	

ParticleGPU.o:ParticleGPU.cu
	$(CC) --device-debug --debug --device-c ParticleGPU.cu -o ParticleGPU.o
	
ConfigDomain.o:ConfigDomain.cu
	$(CC) --device-debug --debug --device-c ConfigDomain.cu -o ConfigDomain.o
	
.PHONY:clean
clean:
	-rm -rf $(RUN)
	-rm -rf *.o
