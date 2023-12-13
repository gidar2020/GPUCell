#ifndef _ConfigDomain_
#define _ConfigDomain_

//#include"system.cuh"

//#define DIVSIZE 256
#define DIVSIZE 128

namespace cudiv
{
	void PreConList(int n, int cutradius, float3* d_totalcoor, float* d_bboxsph, float* d_invwid, int* d_ncube, int3* d_cc, int* d_cubepart, int* d_sortpart, int* d_oldpos);//Prepare construct the cube list, that is , find the position of particles in the cube 
	void Sort(int n, int *keys, int *values);//Sort particles by cube
	void CalBeginEndCube(int n, int SizeCube, int* d_cubepart, int2* d_beginendcube);//Calculate the starting and ending particles for each cube
	void SortBasicArrays(int n, int* d_sortpart, int* d_oldpos, float3* d_totalcoor, int3* d_cc, float4* d_velrhop, float3* d_velhalf, float3* d_totalforce, float* d_press);//Reorders basic arrays according to SortPart
	void SortArraysWithCells(int n, int* d_sortpart, int* d_oldpos, int* d_curpos, float3* d_totalcoor, int3* d_cc, float4* d_velrhop, float3* d_velhalf, float3* d_totalforce, float* d_press, int3* d_nthrobound);

	void Printf(int n, int * d_curtpart, int *d_sortpart, float3* d_TotalCoor);
}
#endif
