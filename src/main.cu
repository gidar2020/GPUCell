// Program GPUSDPD:GPU program for cell through blood vessels using Smoothed Disspative Paticle Dynamics
//Assumptions: incompressible fluid

// Standard function library
#include <iostream> 
#include <fstream> 
#include <iomanip> 
#include <cstring>  
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <cuda_runtime.h>
//Self-defined function head files
#include "main.cuh"
#include "Particle.cuh"
#include "erro.cuh"

using namespace std;

//basic parameters
long int StepCount = -1;//The value is set to -1, and 1 is added in the InitPar() function
long int StepData = 0;

int main(int argc, char* argv[])
{
	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));
	CHECK(cudaEventRecord(start));
	cudaEventQuery(start);
	cout << endl << "******************Begin of Simulation******************" << endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Particle Par; ////define the all the particles
	Par.InitPar();//initial the particle system
	while (StepCount <= StepLimit + StepEquil)
	{
		if (StepCount % StepDisplay == 0)//show the current step per StepDisplay steps
			cout << "StepCurr/StepLimit = " << StepCount - StepEquil << "/" << StepLimit << endl;
		if (StepCount >= StepEquil && (StepCount - StepEquil) % StepOutput == 0)
			Par.OutputPar();//output the particle system
		Par.DrivePar();//drive the particle system
		Par.UpdatePar();//update the particle system
		StepCount++;
	}
	Par.DeletePar();//delete the particle system to release the allocated memory
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	cout << endl << "******************End of Simulation!******************" << endl;
	CHECK(cudaEventRecord(stop));
	CHECK(cudaEventSynchronize(stop));
	float elapsed_time;
	CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
	cout << endl << "*********The Total running time of the program is " << elapsed_time << " ms*********" << endl << endl;
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(stop));

	system("pause");
	return 0;
}

char * itoa(long int number)
{
	char * str = new char[42];
	if (str)
		sprintf(str, "%d", number);
	return str;
}