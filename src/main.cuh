//main.cuh: define global constants, variables and functions

#ifndef PRECOMMAIN
#define PRECOMMAIN

//Global constants
const int NDIM = 3;
const float TimeStep = 1.0e-3;
const float Mass = 1.0;//particle mass, we assume that all particles have the same unit mass
const float Temp = 1.0;//particle temperature, we assume that all particles have the same unit temperature
const float CutRadius = 1.0;//must be 1
const float BBox[2 * NDIM] = { 0, 20, -10, 10, -10, 10};//enclosed the channel;
const long int StepLimit = 10000;//the total computational steps;
const long int StepEquil = 0;// 10000;//the steps for arriving the steady state before starting computation
const long int StepDisplay = 1;//every StepDisplay, we display the current step on running window
const long int StepOutput = 100;// 10;//every StepOutput, we output the results


//extern global variables
extern long int StepCount;//record the simulation step
extern long int StepData;//record the step of outputting data
//extern int DriveOn;//show whether the system is driven

//global function
char * itoa(long int number);//change a number to a character

#endif