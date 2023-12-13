//Particle.chu: implement the program consisting of wall ,fluid and cell membrane particle

//Standard function library 
#include <iostream> 
#include <fstream> 
#include <iomanip> 
#include <cstring>  
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <ctime>
#include <cuda_runtime.h>
//Self-defined function head files
#include "main.cuh"
#include "Particle.cuh"
#include "erro.cuh"
#include "TypesDef.cuh"
#include "ConfigDomain.cuh"
#include "ParticleGPU.cuh"

using namespace std;
void Particle::InitPar()
{
	LoadParInfo();
	SetPar();
}

void Particle::LoadParInfo()
{
	tfloat3* SDPDCoor = new tfloat3[2000000];//2000000 is the maximum number of wall + fluid particles
	tfloat3* CellCoor = new tfloat3[4000000];//20000 is the maximum number of membrane particles of each cell, 200 is the maximum cell number

	LoadFBPar(SDPDCoor);
	LoadSDPDPara();
	LoadCellStruct(CellCoor);
	LoadCellPara();
	AllocCpuMemory();
	CoorMerge(SDPDCoor, CellCoor);

	delete[] SDPDCoor;
	delete[] CellCoor;
}

void Particle::LoadFBPar(tfloat3* SDPDCoor)
{
	int i, k;
	char * filename = "SDPDParCoor.txt";
	ifstream data;
	char totalline[200];//read one line
	char str[20];//read each data
	data.open(filename, ios::in);
	if (data)
	{
		data.getline(totalline, 200);//read the first line, i.e. the instruction line
		data.getline(totalline, 200);//read number of wall1  wall2 ,fluid and particles
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		nBound1Par = atoi(str);
		for (i = 0; i < 20; i++)
			str[i] = totalline[i + 20];
		nBound2Par = atoi(str);
		for (i = 0; i < 20; i++)
			str[i] = totalline[i + 40];
		nFluidPar = atoi(str);

		nB12Par = nBound1Par + nBound2Par;
		nFB12Par = nB12Par + nFluidPar;

		/////////////////////////read the wall1 particles, just provide the repulsive force./////////////////////////////////////////////////	
		data.getline(totalline, 200);//read the instruction line
		for (i = 0; i < nBound1Par; i++)
		{
			data.getline(totalline, 200);//read the coordinates
			for (k = 0; k < 20; k++)
				str[k] = totalline[k];
			SDPDCoor[i].x = atof(str);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k + 20];
			SDPDCoor[i].y = atof(str);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k + 40];
			SDPDCoor[i].z = atof(str);
		}
		/////////////////////////read the wall2 particles/////////////////////////////////////////////////			
		data.getline(totalline, 200);//read the instruction line
		for (i = nBound1Par; i < nB12Par; i++)
		{
			data.getline(totalline, 200);//read the coordinates
			for (k = 0; k < 20; k++)
				str[k] = totalline[k];
			SDPDCoor[i].x = atof(str);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k + 20];
			SDPDCoor[i].y = atof(str);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k + 40];
			SDPDCoor[i].z = atof(str);
		}
		/////////////////////////read the fluid particles/////////////////////////////////////////////////
		data.getline(totalline, 200);//read the instruction line
		for (i = nB12Par; i < nFB12Par; i++)
		{
			data.getline(totalline, 200);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k];
			SDPDCoor[i].x = atof(str);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k + 20];
			SDPDCoor[i].y = atof(str);
			for (k = 0; k < 20; k++)
				str[k] = totalline[k + 40];
			SDPDCoor[i].z = atof(str);
		}
	} //if file is successfully opened
	else //fail to open the file
	{
		cout << "Error: Failure to open \"SDPDParCoor.txt\" !" << endl;
		exit(0);
	}
	data.close();
}

void Particle::LoadSDPDPara()
{
	memset(&consts, 0, sizeof(Structconst));
	int i;
	char * filename = "SDPDPhyPara.txt";
	ifstream data;
	char totalline[200];//read one line
	char str[20];//read each data
	data.open(filename, ios::in);
	if (data)
	{
		data.getline(totalline, 200);//read the first line, i.e. the instruction line
		data.getline(totalline, 200);//read number of particles and structures
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		consts.SoundSpeed1 = atof(str);
		data.getline(totalline, 200);//read number of particles and structures
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		consts.SoundSpeed2 = atof(str);
		data.getline(totalline, 200);
		data.getline(totalline, 200);
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		consts.ShearVis = atof(str);
		data.getline(totalline, 200);
		data.getline(totalline, 200);
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		consts.BulkVis = atof(str);
		data.getline(totalline, 200);
		data.getline(totalline, 200);
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		DenInit = atof(str);
		data.getline(totalline, 200);
		data.getline(totalline, 200);
		for (i = 0; i < 20; i++)
			str[i] = totalline[i];
		consts.ConstExtForce = atof(str)*Mass;
	}//if file is successfully opened
	else //fail to open the file
	{
		cout << "Error: Failure to open \"SDPDPhyPara.txt\" !" << endl;
		exit(0);
	}
	data.close();

}

void Particle::LoadCellStruct(tfloat3* CellCoor)
{
	int i, j, k;
	int s = 0, m = 0, n = 0, a = 0, b = 0;
	char filename[20] = "CELLParStruct.txt";
	ifstream data;
	char totalline[200];//read one line
	char str[20];//read each data
	consts.nCellPar = 0;
	data.open(filename, ios::in);
	if (data)
	{
		data.getline(totalline, 200);
		data.getline(totalline, 200);
		for (k = 0; k < 20; k++)
			str[k] = totalline[k];
		nCell = atoi(str);
		if (nCell > 0)
		{
			Type = new int[nCell];
			nPar = new int[nCell];
			nTri = new int[nCell];
			nEdge = new int[nCell];
			BeginEndTri = new tint2[nCell];
			BeginEndCoor = new tint2[nCell];
			nEdgeInfo = new int[nCell];
			Tri = new tint3[4000000];//This arry are allocated a large amount of space
			Edge = new tint6[4000000];//This arry are allocated a large amount of space
			CellTab = new int[4000000];//This arry are allocated a large amount of space
			for (i = 0; i < nCell; i++)
			{
				//number of IB particles and triangles
				data.getline(totalline, 200);
				data.getline(totalline, 200);
				for (k = 0; k < 20; k++)
					str[k] = totalline[k];
				Type[i] = atoi(str);
				for (k = 0; k < 20; k++)
					str[k] = totalline[k + 20];
				nPar[i] = atoi(str);
				for (k = 0; k < 20; k++)
					str[k] = totalline[k + 40];
				nTri[i] = atoi(str);
				nEdge[i] = nTri[i] * 3 / 2;
				for (k = 0; k < 20; k++)
					str[k] = totalline[k + 60];
				nEdgeInfo[i] = atoi(str);
				//Store the numbers of the first and last particles on each cell
				BeginEndCoor[i].x = s;
				BeginEndCoor[i].y = s + nPar[i] - 1;

				//coordinates of Cell particles		
				data.getline(totalline, 200);
				for (j = 0; j < nPar[i]; j++)
				{
					data.getline(totalline, 200);
					for (k = 0; k < 20; k++)
						str[k] = totalline[k];
					CellCoor[s].x = atof(str);
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 20];
					CellCoor[s].y = atof(str);
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 40];
					CellCoor[s].z = atof(str);

					CellTab[s] = i;//each particle has a label that marks which cell it belongs to
					s++;//The final value of s is the sum of the number of particles on all the cells
				}
				//triangular structure				
				BeginEndTri[i].x = m;
				BeginEndTri[i].y = m + nTri[i] - 1;
				data.getline(totalline, 200);
				for (j = 0; j < nTri[i]; j++)
				{
					data.getline(totalline, 200);
					for (k = 0; k < 20; k++)
						str[k] = totalline[k];
					Tri[m].x = atof(str) + b;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 20];
					Tri[m].y = atof(str) + b;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 40];
					Tri[m].z = atof(str) + b;
					m++;//The final m is the total number of triangles on all the cells
				}
				//all edges of triangles				
				data.getline(totalline, 200);
				for (j = 0; j < nEdge[i]; j++)
				{
					data.getline(totalline, 200);
					for (k = 0; k < 20; k++)
						str[k] = totalline[k];
					//Edge[i*nEdge[i] + j].x = atof(str);
					Edge[n].x = atof(str) + b;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 20];
					Edge[n].y = atof(str) + b;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 40];
					Edge[n].z = atof(str) + b;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 60];
					Edge[n].k = atof(str) + b;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 80];
					Edge[n].v = atof(str) + a;
					for (k = 0; k < 20; k++)
						str[k] = totalline[k + 100];
					Edge[n].w = atof(str) + a;

					n++;//The final n is the number of sides of the triangle on all the cells
				}
				a += nTri[i];
				b += nPar[i];
			}//for (i = 0; i < nCell; i++)

			nAllTri = m;
			nAllEdge = n;
			consts.nCellPar = s;
		}//if (nCell > 0)			
	}
	else //fail to open the file
	{
		cout << "Error: Particle::LoadCellStruct()!!" << endl;
		exit(0);
	}
	data.close();
}

void Particle::LoadCellPara()
{
	if (nCell > 0)
	{
		int i, j, k, m;
		char * filename = "CELLPhyPara.txt";
		ifstream data;
		char totalline[200];//read one line
		char str[20];//read each data
		data.open(filename, ios::in);
		if (data)
		{
			data.getline(totalline, 200);//read the first line, i.e. the instruction line
			data.getline(totalline, 200);//read number of particles and structures
			for (i = 0; i < 20; i++)
				str[i] = totalline[i];
			nType = atoi(str);
			if (nType > 0)
			{
				ShearMod = new float[nType];
				BendingMod = new float[nType];
				ShearMod_Para = new float[nAllEdge];
				BendingMod_Para = new float[nAllEdge];
				HGarea = new float[nType];
				HLarea = new float[nType];
				HLarea_Para = new float[nAllTri];
				HGarea_Para = new float[nAllTri];
				HVol = new float[nType];
				HVol_Para = new float[nAllTri];
				data.getline(totalline, 200);
				for (j = 0; j < nType; j++)
				{
					data.getline(totalline, 200);
					for (i = 0; i < 20; i++)
						str[i] = totalline[i];
					ShearMod[j] = atof(str);
					//cout << ShearMod[j] << endl;
				}
				data.getline(totalline, 200);
				for (j = 0; j < nType; j++)
				{
					data.getline(totalline, 200);
					for (i = 0; i < 20; i++)
						str[i] = totalline[i];
					BendingMod[j] = atof(str);
				}
				data.getline(totalline, 200);
				for (j = 0; j < nType; j++)
				{
					data.getline(totalline, 200);
					for (i = 0; i < 20; i++)
						str[i] = totalline[i];
					HGarea[j] = atof(str);
				}
				data.getline(totalline, 200);
				for (j = 0; j < nType; j++)
				{
					data.getline(totalline, 200);
					for (i = 0; i < 20; i++)
						str[i] = totalline[i];
					HLarea[j] = atof(str);
				}
				data.getline(totalline, 200);
				for (j = 0; j < nType; j++)
				{
					data.getline(totalline, 200);
					for (i = 0; i < 20; i++)
						str[i] = totalline[i];
					HVol[j] = atof(str);
				}
				data.getline(totalline, 200);
				data.getline(totalline, 200);
				for (i = 0; i < 20; i++)
					str[i] = totalline[i];
				consts.MaxCut = atof(str);
				data.getline(totalline, 200);
				data.getline(totalline, 200);
				for (i = 0; i < 20; i++)
					str[i] = totalline[i];
				consts.zeroLen = atof(str);
				data.getline(totalline, 200);
				data.getline(totalline, 200);
				for (i = 0; i < 20; i++)
					str[i] = totalline[i];
				consts.scaFac = atof(str);
				data.getline(totalline, 200);
				data.getline(totalline, 200);
				for (i = 0; i < 20; i++)
					str[i] = totalline[i];
				consts.surEnergy = atof(str);
			}
		}//if file is successfully opened
		
		else //fail to open the file
		{
			cout << "Error: Particle::LoadCellPara()!!" << endl;
			exit(0);
		}
		data.close();

		k = 0;
		m = 0;

		for (i = 0; i < nCell; i++)
		{
			for (j = 0; j < nEdge[i]; j++)
			{
				//The shear modulus and bending modulus are assigned to each edge
				ShearMod_Para[k] = ShearMod[Type[i] - 1];
				BendingMod_Para[k] = BendingMod[Type[i] - 1];
				k++;
			}
			for (j = 0; j < nTri[i]; j++)
			{
				//The area and volume constraints are assigned to each triangle
				HLarea_Para[m] = HLarea[Type[i] - 1];
				HGarea_Para[m] = HGarea[Type[i] - 1];
				HVol_Para[m] = HVol[Type[i] - 1];
				m++;
			}
		}
	}//if (nCell > 0)
}

void Particle::AllocCpuMemory()
{
	int i;
	consts.nBound2Start = nBound1Par;
	consts.nFluidStart = consts.nBound2Start + nBound2Par;
	consts.nCellStart = consts.nFluidStart + nFluidPar;
	consts.nTotalPar = consts.nCellStart + consts.nCellPar;
	consts.VVsigma = 0.65;
	consts.iSeed = 1.0;

	for (i = 0; i < NDIM * 2; i++)
		BBoxSDPD[i] = BBox[i];

	TotalCoor = new tfloat3[consts.nTotalPar];
	TotalForce = new tfloat3[consts.nTotalPar];
	Press = new float[consts.nTotalPar];
	nThroBound = new tint3[consts.nTotalPar];
	Velrhop = new tfloat4[consts.nTotalPar];
	VelHalf = new tfloat3[consts.nTotalPar];
	SortPart = new int[consts.nTotalPar];
	h_SDPDForce = new tfloat3[consts.nTotalPar];
	if (nCell > 0)
	{
		h_DefForce = new tfloat3[consts.nCellPar];
		h_AggForce = new tfloat3[consts.nCellPar];
		h_Area = new float[nCell];
		h_Vol = new  float[nCell];
		TriNorDir = new tfloat3[nAllTri];
		TriCenter = new tfloat3[nAllTri];
	}

}

void Particle::CoorMerge(tfloat3* SDPDCoor, tfloat3* CellCoor)
{
	int i;
	for (i = 0; i < nFB12Par; i++)
		TotalCoor[i] = SDPDCoor[i];
	for (i = nFB12Par; i < consts.nTotalPar; i++)
		TotalCoor[i] = CellCoor[i - nFB12Par];
}

void Particle::SetPar()
{
	SetParticle();
	ConfigDomain();

	cupar::CalGeoPara(StepCount, nAllTri, nCell, d_BBoxSDPD, d_CurPos, d_Tri, d_BeginEndTri, d_nThroBound, d_TotalCoor, d_TriNorDir, d_TriArea, d_RefTriArea, d_TriCenter,
		d_RefArea_Para, d_RefVol_Para, d_Area_Para, d_Vol_Para, d_Area, d_Vol, d_ParArea);

	cupar::SetMechPara(nCell, nAllEdge, d_Edge, d_CurPos, d_RefEdgeLen, d_MaxRefEdgeLen, d_BendingMod_Para, d_ShearMod_Para, d_HBending, d_PowerIndex, d_PerLenWLC, d_HPow,
		d_RefTriAngle, d_TriNorDir, d_TriCenter, d_TotalCoor);

	StepCount += 1;
}

void Particle::SetParticle()
{
	int i;
	tfloat3 VelSum;
	iSeed = 1;
	//Rand = RandG(iSeed);
	/////////////////set their values/////////////////////////////////////////////////////////
	VelSum.x = 0.0;
	VelSum.y = 0.0;
	VelSum.z = 0.0;
	for (i = 0; i < consts.nTotalPar; i++)
	{
		Velrhop[i].x = 0.0;
		Velrhop[i].y = 0.0;
		Velrhop[i].z = 0.0;

		//the velocity of whole system, to ensure that the whole system is stationary
		VelSum.x += Velrhop[i].x;
		VelSum.y += Velrhop[i].y;
		VelSum.z += Velrhop[i].z;
	}
	for (i = 0; i < consts.nTotalPar; i++)
	{
		Velrhop[i].x -= 1.0 / nFB12Par * VelSum.x;//ensure that the whole system is stationary	
		Velrhop[i].y -= 1.0 / nFB12Par * VelSum.y;
		Velrhop[i].z -= 1.0 / nFB12Par * VelSum.z;


		VelHalf[i].x = Velrhop[i].x;
		VelHalf[i].y = Velrhop[i].y;
		VelHalf[i].z = Velrhop[i].z;


		nThroBound[i].x = 0;
		nThroBound[i].y = 0;
		nThroBound[i].z = 0;

		TotalForce[i].x = 0;
		TotalForce[i].y = 0;
		TotalForce[i].z = 0;

		h_SDPDForce[i].x = 0;
		h_SDPDForce[i].y = 0;
		h_SDPDForce[i].z = 0;

		Velrhop[i].w = DenInit;

		if (i >= consts.nCellStart)
		{
			h_DefForce[i - consts.nCellStart].x = 0;
			h_DefForce[i - consts.nCellStart].y = 0;
			h_DefForce[i - consts.nCellStart].z = 0;

			h_AggForce[i - consts.nCellStart].x = 0;
			h_AggForce[i - consts.nCellStart].y = 0;
			h_AggForce[i - consts.nCellStart].z = 0;
		}

	}

	
}

void Particle::ConfigDomain()
{
	int i = 0;
	nCube[0] = floor((BBoxSDPD[1] - BBoxSDPD[0]) / CutRadius);//create the computational cubes, which should contain all the particles, such as fluid particles, boundary particle 1 and 2.
	nCube[1] = floor((BBoxSDPD[3] - BBoxSDPD[2] + 2.0 * CutRadius) / CutRadius);
	nCube[2] = floor((BBoxSDPD[5] - BBoxSDPD[4] + 2.0 * CutRadius) / CutRadius);
	OuterRegion[0] = BBoxSDPD[1] - BBoxSDPD[0];
	OuterRegion[1] = BBoxSDPD[3] - BBoxSDPD[2] + 2.0 * CutRadius;
	OuterRegion[2] = BBoxSDPD[5] - BBoxSDPD[4] + 2.0 * CutRadius;
	invWid[0] = (float)(nCube[0] / OuterRegion[0]);
	invWid[1] = (float)(nCube[1] / OuterRegion[1]);
	invWid[2] = (float)(nCube[2] / OuterRegion[2]);
	SizeCube = nCube[0] * nCube[1] * nCube[2];

	for (i = 0; i < consts.nTotalPar; i++)
	{
		SortPart[i] = i;
	}
		
	AllocGpuMemory();
	ParticlesDataUp();
	ConstantDataUp();
	ConCubeList();
}

void Particle::AllocGpuMemory()
{
	CHECK(cudaMalloc((void**)&d_TotalCoor, sizeof(float3)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_TotalForce, sizeof(float3)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_nCube, sizeof(int) * NDIM));
	CHECK(cudaMalloc((void**)&d_OuterRegion, sizeof(float) * NDIM));
	CHECK(cudaMalloc((void**)&d_invWid, sizeof(float) * NDIM));
	CHECK(cudaMalloc((void**)&d_CC, sizeof(int3)* consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_CubePart, sizeof(int)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_SortPart, sizeof(int)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_OldPos, sizeof(int)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_BeginEndCube, sizeof(int2)*SizeCube));
	CHECK(cudaMalloc((void**)&d_Velrhop, sizeof(float4)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_VelHalf, sizeof(float3)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_Press, sizeof(float)*consts.nTotalPar));
	CHECK(cudaMalloc((void**)&d_BBoxSDPD, sizeof(float) * 2 * NDIM));
	CHECK(cudaMalloc((void**)&d_nThroBound, sizeof(int3)*consts.nTotalPar));

	if (nCell > 0)
	{
		CHECK(cudaMalloc((void**)&d_CellTab, sizeof(int)*consts.nCellPar));
		CHECK(cudaMalloc((void**)&d_CurPos, sizeof(int)*consts.nTotalPar));
		CHECK(cudaMalloc((void**)&d_Edge, sizeof(int6)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_Tri, sizeof(int3)*nAllTri));
		CHECK(cudaMalloc((void**)&d_TriNorDir, sizeof(float3)*nAllTri));
		CHECK(cudaMalloc((void**)&d_TriCenter, sizeof(float3)*nAllTri));
		CHECK(cudaMalloc((void**)&d_TriArea, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_RefTriArea, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_RefTriAngle, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_RefEdgeLen, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_BendingMod_Para, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_ShearMod_Para, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_PowerIndex, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_HLarea_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_HGarea_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_HVol_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_RefArea_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_RefVol_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_Area_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_Vol_Para, sizeof(float)*nAllTri));
		CHECK(cudaMalloc((void**)&d_PerLenWLC, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_BeginEndTri, sizeof(int2)*nCell));
		CHECK(cudaMalloc((void**)&d_Area, sizeof(float)*nCell));
		CHECK(cudaMalloc((void**)&d_Vol, sizeof(float)*nCell));
		CHECK(cudaMalloc((void**)&d_HPow, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_HBending, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_MaxRefEdgeLen, sizeof(float)*nAllEdge));
		CHECK(cudaMalloc((void**)&d_ParArea, sizeof(float)*consts.nCellPar));
		CHECK(cudaMalloc((void**)&d_DefForce, sizeof(float3)*consts.nCellPar));
		CHECK(cudaMalloc((void**)&d_AggForce, sizeof(float3)*consts.nCellPar));
	}

}

void Particle::ParticlesDataUp()
{
	CHECK(cudaMemcpy(d_TotalCoor, TotalCoor, sizeof(float3)* consts.nTotalPar, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_TotalForce, TotalForce, sizeof(float3)* consts.nTotalPar, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_nCube, nCube, sizeof(int) * NDIM, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_OuterRegion, OuterRegion, sizeof(float) * NDIM, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_invWid, invWid, sizeof(float) * NDIM, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Velrhop, Velrhop, sizeof(float4)*consts.nTotalPar, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_VelHalf, VelHalf, sizeof(float3)*consts.nTotalPar, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_SortPart, SortPart, sizeof(int)*consts.nTotalPar, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_BBoxSDPD, BBoxSDPD, sizeof(float) * 2 * NDIM, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_nThroBound, nThroBound, sizeof(int3)* consts.nTotalPar, cudaMemcpyHostToDevice));

	if (nCell>0)
	{
		CHECK(cudaMemcpy(d_CellTab, CellTab, sizeof(int)*consts.nCellPar, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_BendingMod_Para, BendingMod_Para, sizeof(float)*nAllEdge, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_ShearMod_Para, ShearMod_Para, sizeof(float)*nAllEdge, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_HLarea_Para, HLarea_Para, sizeof(float)*nAllTri, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_HGarea_Para, HGarea_Para, sizeof(float)*nAllTri, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_HVol_Para, HVol_Para, sizeof(float)*nAllTri, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Tri, Tri, sizeof(int3)*nAllTri, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_Edge, Edge, sizeof(int6)*nAllEdge, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_BeginEndTri, BeginEndTri, sizeof(int2)*nCell, cudaMemcpyHostToDevice));
	}
}

void Particle::ConstantDataUp()
{
	consts.CutRadius = CutRadius;
	consts.Mass = Mass;
	consts.DenInit = DenInit;
	consts.TimeStep = TimeStep;
	consts.Temp = Temp;

	cupar::ConstsDataUp(&consts);

}

void Particle::ConCubeList()
{
	cudiv::PreConList(consts.nTotalPar, consts.CutRadius, d_TotalCoor, d_BBoxSDPD, d_invWid, d_nCube, d_CC, d_CubePart, d_SortPart, d_OldPos);
	cudiv::Sort(consts.nTotalPar, d_CubePart, d_SortPart);
	
	
	cudiv::CalBeginEndCube(consts.nTotalPar, SizeCube, d_CubePart, d_BeginEndCube);

	if (nCell>0)
		cudiv::SortArraysWithCells(consts.nTotalPar, d_SortPart, d_OldPos, d_CurPos, d_TotalCoor, d_CC, d_Velrhop, d_VelHalf, d_TotalForce, d_Press, d_nThroBound);//数组nThroBound[]后面也得跟着排
	else
		cudiv::SortBasicArrays(consts.nTotalPar, d_SortPart, d_OldPos, d_TotalCoor, d_CC, d_Velrhop, d_VelHalf, d_TotalForce, d_Press);
}


void Particle::OutputPar()
{	
	ParticlesDataReturn();
	OutputCoor(StepData);
	OutputField(StepData);
	OutPutCellCoor(StepData);
	OutPutCellForce(StepData);
	OutPutCellGeo(StepData);
	OutPutTriNorDir(StepData);
	StepData++;
}

void Particle::ParticlesDataReturn()
{
	CHECK(cudaMemcpy(TotalCoor, d_TotalCoor, sizeof(float3)* consts.nTotalPar, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Press, d_Press, sizeof(float)*consts.nTotalPar, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(Velrhop, d_Velrhop, sizeof(float4)*consts.nTotalPar, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(SortPart, d_SortPart, sizeof(int)*consts.nTotalPar, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(TotalForce, d_TotalForce, sizeof(float3)*consts.nTotalPar, cudaMemcpyDeviceToHost));

	if (nCell > 0)
	{
		CHECK(cudaMemcpy(h_Area, d_Area, sizeof(float)*nCell, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_Vol, d_Vol, sizeof(float)*nCell, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_DefForce, d_DefForce, sizeof(float3)*consts.nCellPar, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(h_AggForce, d_AggForce, sizeof(float3)*consts.nCellPar, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(TriNorDir, d_TriNorDir, sizeof(tfloat3)*nAllTri, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(TriCenter, d_TriCenter, sizeof(tfloat3)*nAllTri, cudaMemcpyDeviceToHost));
	}
}

void Particle::OutputCoor(long int StepData)
{
	int i;
	char * filename = NULL;
	char * string_num;
	char * filetype;
	char s[50] = "Result/FluidCoor/";
	filetype = ".plt";
	string_num = itoa(StepData);
	filetype = strcat(string_num, filetype);
	filename = strcat(s, filetype);
	ofstream tfile(filename);
	tfile << setiosflags(ios::left) << setw(20) << "Title=\"Particle Coordinate\"" << endl;
	tfile << setiosflags(ios::left) << setw(20) << "Variables=\"x\",\"y\",\"z\",\"SP\"" << endl;
	tfile.precision(5);
	tfile.setf(ios::scientific, ios::floatfield);

	if (nBound1Par > 0)
	{
		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the type-1 virtual boundary particles
		for (i = 0; i < consts.nTotalPar; i++)
		{
			if (SortPart[i] >= 0 && SortPart[i] < nBound1Par)
			{
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
				tfile << endl;
			}
		}
	}

	if (nBound2Par > 0)
	{
		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the type-2 virtual boundary particles
		for (i = 0; i < consts.nTotalPar; i++)
		{
			if (SortPart[i] >= nBound1Par && SortPart[i] < nB12Par)
			{
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
				tfile << endl;
			}
		}
	}

	if (nFluidPar > 0)
	{
		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the fluid particles
		for (i = 0; i < consts.nTotalPar; i++)
		{
			if (SortPart[i] >= nB12Par && SortPart[i] < nFB12Par)
			{
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
				tfile << endl;
			}
		}
	}
	if (consts.nCellPar>0)
	{
		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the Cell particles
		for (i = 0; i < consts.nTotalPar; i++)
		{
			if (SortPart[i] >= nFB12Par && SortPart[i] < consts.nTotalPar)
			{
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
				tfile << endl;
			}
		}


	}
	tfile.close();
}

//void Particle::OutputField(long int StepData)
//{
//	int i;
//	char * filename = NULL;
//	char * string_num;
//	char * filetype;
//	char s[50] = "Result/FluidField/";
//	filetype = (char*)".plt";
//	//filetype = ".plt";
//	string_num = itoa(StepData);
//	filetype = strcat(string_num, filetype);
//	filename = strcat(s, filetype);
//	ofstream tfile(filename);
//	tfile << setiosflags(ios::left) << setw(20) << "Title=\"Particle Fluid\"" << endl;
//	tfile << setiosflags(ios::left) << setw(20) << "Variables=\"x\",\"y\",\"z\",\"Den\",\"Press\",\"Vx\",\"Vy\",\"Vz\",\"SP\"" << endl;
//	tfile.precision(5);
//	tfile.setf(ios::scientific, ios::floatfield);
//
//	if (nBound1Par > 0)
//	{
//		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the type-1 virtual boundary particles
//		for (i = 0; i < consts.nTotalPar; i++)
//		{
//			if (SortPart[i] >= 0 && SortPart[i] < nBound1Par)
//			{
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].w;
//				tfile << setiosflags(ios::left) << setw(20) << Press[i];
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].z;
//
//				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
//				tfile << endl;
//			}
//		}
//	}
//
//	if (nBound2Par > 0)
//	{
//		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the type-2 virtual boundary particles
//		for (i = 0; i < consts.nTotalPar; i++)
//		{
//			if (SortPart[i] >= nBound1Par && SortPart[i] < nB12Par)
//			{
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].w;
//				tfile << setiosflags(ios::left) << setw(20) << Press[i];
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].z;
//
//				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
//				tfile << endl;
//			}
//		}
//	}
//
//	if (nFluidPar > 0)
//	{
//		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the fluid particles
//		for (i = 0; i < consts.nTotalPar; i++)
//		{
//			if (SortPart[i] >= nB12Par && SortPart[i] < nFB12Par)
//			{
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].w;
//				tfile << setiosflags(ios::left) << setw(20) << Press[i];
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].z;
//
//				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
//				tfile << endl;
//			}
//		}
//	}
//	if (consts.nCellPar > 0)
//	{
//		tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;//output the Cell particles
//		for (i = 0; i < consts.nTotalPar; i++)
//		{
//			if (SortPart[i] >= nFB12Par && SortPart[i] < consts.nTotalPar)
//			{
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].w;
//				tfile << setiosflags(ios::left) << setw(20) << Press[i];
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].x;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].y;
//				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].z;
//
//				tfile << setiosflags(ios::left) << setw(20) << SortPart[i];
//				tfile << endl;
//			}
//		}
//
//
//	}
//	tfile.close();
//	
//}

void Particle::OutputField(long int StepData)
{
	int i;
	char * filename = NULL;
	char * string_num;
	char * filetype;
	char s[50] = "Result/FluidField/";
	filetype = (char*)".plt";
	//filetype = ".plt";
	string_num = itoa(StepData);
	filetype = strcat(string_num, filetype);
	filename = strcat(s, filetype);
	ofstream tfile(filename);
	tfile << setiosflags(ios::left) << setw(20) << "Title=\"Particle Fluid\"" << endl;
	tfile << setiosflags(ios::left) << setw(20) << "Variables=\"x\",\"y\",\"z\",\"Den\",\"Press\",\"Vx\",\"Vy\",\"Vz\"" << endl;
	tfile.precision(5);
	tfile.setf(ios::scientific, ios::floatfield);

	if (nFB12Par > 0)
	{
		for (i = 0; i < consts.nTotalPar; i++)
		{
			if (SortPart[i] >= 0 && SortPart[i] < nFB12Par)
			{
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].x;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].y;
				tfile << setiosflags(ios::left) << setw(20) << TotalCoor[i].z;
				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].w;
				tfile << setiosflags(ios::left) << setw(20) << Press[i];
				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].x;
				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].y;
				tfile << setiosflags(ios::left) << setw(20) << Velrhop[i].z;

				tfile << endl;
			}
		}
	}
	tfile.close();
}

void Particle::OutPutCellCoor(long int StepData)
{
	if (nCell > 0)
	{
		int i, j, k, m;
		int* Pos;
		Pos = new int[consts.nCellPar];
		char * filename = NULL;
		char * string_num;
		char * filetype;
		filetype = ".plt";
		string_num = itoa(StepData);
		filetype = strcat(string_num, filetype);
		char s[50] = "Result/CellCoor/";
		filename = strcat(s, filetype);
		ofstream tfile(filename);
		tfile << setiosflags(ios::left) << setw(20) << "Title=\"Particle Coordinate\"" << endl;
		tfile << setiosflags(ios::left) << setw(20) << "Variables=\"x\",\"y\",\"z\"" << endl;
		tfile.precision(5);
		tfile.setf(ios::scientific, ios::floatfield);

		j = 0, k = 0, m = 1;
		for (i = 0; i < nCell; i++)
		{
			tfile << "Zone N=" << nPar[i] << ", E=" << nTri[i] << ", F=FEPOINT, ET=TRIANGLE" << endl;
			for (j = 0; j < consts.nTotalPar; j++)
			{
				k = SortPart[j] - consts.nCellStart;
				int a = BeginEndCoor[i].x;
				int b = BeginEndCoor[i].y;
				/*if (i == 0)
					cout << a << "  " << b << endl;a=0 b=2557*/
				if (k >= a && k <= b)
				{
					Pos[k] = m;
					//m++;

					tfile << setiosflags(ios::left) << setw(20) << TotalCoor[j].x;
					tfile << setiosflags(ios::left) << setw(20) << TotalCoor[j].y;
					tfile << setiosflags(ios::left) << setw(20) << TotalCoor[j].z;
					tfile << endl;
					m++;
				}

			}
			for (j = 0; j < nAllTri; j++)
			{
				if (j >= BeginEndTri[i].x&& j <= BeginEndTri[i].y)
				{
					tfile << setiosflags(ios::left) << setw(20) << Pos[Tri[j].x - 1];
					tfile << setiosflags(ios::left) << setw(20) << Pos[Tri[j].y - 1];
					tfile << setiosflags(ios::left) << setw(20) << Pos[Tri[j].z - 1];
					tfile << endl;
				}
			}
			m = 1;
		}
		tfile.close();

		delete[]Pos;
	}
}

void Particle::OutPutCellForce(long int StepData)
{
	if (nCell > 0)
	{
		int i, j, k, m;
		int* Pos = new int[consts.nCellPar];
		tfloat3*TDefForce = new tfloat3[nCell];
		tfloat3*TAggForce = new tfloat3[nCell];
		char * filename = NULL;
		char * string_num;
		char * filetype;
		filetype = ".plt";
		string_num = itoa(StepData);
		filetype = strcat(string_num, filetype);
		char s[50] = "Result/CellForce/";
		filename = strcat(s, filetype);
		ofstream tfile(filename);
		tfile << setiosflags(ios::left) << setw(20) << "Title=\"Particle Force\"" << endl;
		tfile << setiosflags(ios::left) << setw(20) << "Variables=\"x\",\"y\",\"z\", \"DFx\",\"DFy\",\"DFz\",\"AFx\",\"AFy\",\"AFz\"" << endl;
		tfile.precision(5);
		tfile.setf(ios::scientific, ios::floatfield);


		j = 0, k = 0, m = 1;
		for (i = 0; i < nCell; i++)
		{
			TDefForce[i].x = 0.0;
			TDefForce[i].y = 0.0;
			TDefForce[i].z = 0.0;
			TAggForce[i].x = 0.0;
			TAggForce[i].y = 0.0;
			TAggForce[i].z = 0.0;

			tfile << "Zone N=" << nPar[i] << ", E=" << nTri[i] << ", F=FEPOINT, ET=TRIANGLE" << endl;
			for (j = 0; j < consts.nTotalPar; j++)
			{
				k = SortPart[j] - consts.nCellStart;
				int a = BeginEndCoor[i].x;
				int b = BeginEndCoor[i].y;
				if (k >= a && k <= b)
				{
					Pos[k] = m;

					tfile << setiosflags(ios::left) << setw(20) << TotalCoor[j].x;
					tfile << setiosflags(ios::left) << setw(20) << TotalCoor[j].y;
					tfile << setiosflags(ios::left) << setw(20) << TotalCoor[j].z;

					tfile << setiosflags(ios::left) << setw(20) << h_DefForce[k].x;
					tfile << setiosflags(ios::left) << setw(20) << h_DefForce[k].y;
					tfile << setiosflags(ios::left) << setw(20) << h_DefForce[k].z;


					tfile << setiosflags(ios::left) << setw(20) << h_AggForce[k].x;
					tfile << setiosflags(ios::left) << setw(20) << h_AggForce[k].y;
					tfile << setiosflags(ios::left) << setw(20) << h_AggForce[k].z;

					tfile << endl;
					m++;

					TDefForce[i].x += h_DefForce[k].x;
					TDefForce[i].y += h_DefForce[k].y;
					TDefForce[i].z += h_DefForce[k].z;

					TAggForce[i].x += h_AggForce[k].x;
					TAggForce[i].y += h_AggForce[k].y;
					TAggForce[i].z += h_AggForce[k].z;

				}

			}
			TDefForce[i].x /= nPar[i];
			TDefForce[i].y /= nPar[i];
			TDefForce[i].z /= nPar[i];
			TAggForce[i].x /= nPar[i];
			TAggForce[i].y /= nPar[i];
			TAggForce[i].z /= nPar[i];


			for (j = 0; j < nAllTri; j++)
			{
				if (j >= BeginEndTri[i].x&& j <=BeginEndTri[i].y)
				{
					tfile << setiosflags(ios::left) << setw(20) << Pos[Tri[j].x - 1];
					tfile << setiosflags(ios::left) << setw(20) << Pos[Tri[j].y - 1];
					tfile << setiosflags(ios::left) << setw(20) << Pos[Tri[j].z - 1];
					tfile << endl;
				}
			}
			m = 1;
		}
		tfile.close();
		delete[]Pos;

		/////////////////////////Output the force acting on each IB/////////////////////////////////////////////////
		char * filenameT = NULL;
		char * string_numT;
		char * filetypeT;
		char sT[50];
		fstream tfileT;
		for (i = 0; i < nCell; i++)
		{
			filetypeT = ".plt";
			strcpy(sT, "Result/CellForce/TF_Cell");//Every cell has this.PLT file,such as TF_Cell0.plt，TF_Cell1.plt
			string_numT = itoa(i);//converts a number to a string
			filetypeT = strcat(string_numT, filetypeT);//concatenate two char strings
			filenameT = strcat(sT, filetypeT);
			if (StepCount == 0)
			{
				ofstream tfileInitial(filenameT);
				tfileInitial.precision(5);
				tfileInitial.setf(ios::scientific, ios::floatfield);
				tfileInitial << setiosflags(ios::left) << setw(15) << StepData * TimeStep*StepOutput;
				tfileInitial << setiosflags(ios::left) << setw(15) << TDefForce[i].x;
				tfileInitial << setiosflags(ios::left) << setw(15) << TDefForce[i].y;
				tfileInitial << setiosflags(ios::left) << setw(15) << TDefForce[i].z;
				tfileInitial << setiosflags(ios::left) << setw(15) << TAggForce[i].x;
				tfileInitial << setiosflags(ios::left) << setw(15) << TAggForce[i].y;
				tfileInitial << setiosflags(ios::left) << setw(15) << TAggForce[i].z;

				tfileInitial << setiosflags(ios::left) << setw(15) << endl;
				tfileInitial.close();
			}
			else
			{
				tfileT.open(filenameT, ios::app | ios::out);//open file using append model
				if (!tfileT)
				{
					cout << "Error: Particle::OutputForce!!" << endl;
					exit(0);
				}
				tfileT.precision(5);
				tfileT.setf(ios::scientific, ios::floatfield);
				tfileT << setiosflags(ios::left) << setw(15) << StepData * TimeStep*StepOutput;
				tfileT << setiosflags(ios::left) << setw(15) << TDefForce[i].x;
				tfileT << setiosflags(ios::left) << setw(15) << TDefForce[i].y;
				tfileT << setiosflags(ios::left) << setw(15) << TDefForce[i].z;
				tfileT << setiosflags(ios::left) << setw(15) << TAggForce[i].x;
				tfileT << setiosflags(ios::left) << setw(15) << TAggForce[i].y;
				tfileT << setiosflags(ios::left) << setw(15) << TAggForce[i].z;

				tfileT << setiosflags(ios::left) << setw(15) << endl;
				tfileT.close();
			}
		}
		delete[] TDefForce;
		delete[] TAggForce;

	}
}

void Particle::OutPutCellGeo(long int StepData)
{
	if (nCell > 0)
	{
		int i, j, k;
		int* Pos = new int[consts.nCellPar];
		//Pos = new int[consts.nCellPar];
		tfloat3*CellVel = new tfloat3[nCell];
		float *MagCellVel = new float[nCell];
		char * filename = NULL;
		char * string_num;
		char * filetype;
		char s[50];
		fstream tfile;

		j = 0, k = 0;
		for (i = 0; i < nCell; i++)//the moving velocity of each IB
		{
			CellVel[i].x = 0;
			CellVel[i].y = 0;
			CellVel[i].z = 0;

			for (j = 0; j < consts.nTotalPar; j++)
			{
				k = SortPart[j] - consts.nCellStart;
				int a = BeginEndCoor[i].x;
				int b = BeginEndCoor[i].y;
				//cout << a << "  " << b << endl; 0 2557
				if (k >= a && k <= b)
				{
					CellVel[i].x += Velrhop[j].x;
					CellVel[i].y += Velrhop[j].y;
					CellVel[i].z += Velrhop[j].z;
				}
			}
			CellVel[i].x /= nPar[i];
			CellVel[i].y /= nPar[i];
			CellVel[i].z /= nPar[i];

			MagCellVel[i] = 0.0;
			MagCellVel[i] = CellVel[i].x*CellVel[i].x + CellVel[i].y*CellVel[i].y + CellVel[i].z* CellVel[i].z;
			MagCellVel[i] = sqrt(MagCellVel[i]);
		}

		for (i = 0; i < nCell; i++)//the moving velocity of each IB
		{
			filetype = ".plt";
			strcpy(s, "Result/CellGeo/Geo_Cell");
			string_num = itoa(i);
			filetype = strcat(string_num, filetype);
			filename = strcat(s, filetype);

			if (StepCount == 0)
			{
				ofstream tfileInitial(filename);
				tfileInitial.precision(5);
				tfileInitial.setf(ios::scientific, ios::floatfield);
				tfileInitial << setiosflags(ios::left) << setw(15) << StepData * TimeStep*StepOutput;
				tfileInitial << setiosflags(ios::left) << setw(15) << MagCellVel[i];
				tfileInitial << setiosflags(ios::left) << setw(15) << h_Area[i];
				tfileInitial << setiosflags(ios::left) << setw(15) << h_Vol[i];
				tfileInitial << setiosflags(ios::left) << setw(15) << endl;
				tfileInitial.close();
			}
			
			else
			{
				tfile.open(filename, ios::app | ios::out);//open file using append model
				if (!tfile)
				{
					cout << "Error: CIB::OutputGeo!!" << endl;
					exit(0);
				}
				tfile.precision(5);
				tfile.setf(ios::scientific, ios::floatfield);
				tfile << setiosflags(ios::left) << setw(15) << StepData * TimeStep*StepOutput;
				tfile << setiosflags(ios::left) << setw(15) << MagCellVel[i];
				tfile << setiosflags(ios::left) << setw(15) << h_Area[i];
				tfile << setiosflags(ios::left) << setw(15) << h_Vol[i];
				tfile << setiosflags(ios::left) << setw(15) << endl;
				tfile.close();
			}
		}
		delete[] CellVel;
		delete[] MagCellVel;
		delete[]Pos;
	}
}

void Particle::OutPutTriNorDir(long int StepData)
{
	if (nCell > 0)
	{
		int i, j, k, m;
		int* Pos;
		Pos = new int[consts.nCellPar];
		char * filename = NULL;
		char * string_num;
		char * filetype;
		filetype = ".plt";
		string_num = itoa(StepData);
		filetype = strcat(string_num, filetype);
		char s[50] = "Result/CellNorDir/";
		filename = strcat(s, filetype);
		ofstream tfile(filename);
		tfile << setiosflags(ios::left) << setw(20) << "Title=\"Triangle information\"" << endl;
		tfile << setiosflags(ios::left) << setw(20) << "Variables=\"Cenx\",\"Ceny\",\"Cenz\",\"Norx\",\"Nory\",\"Norz\",\"ID\"" << endl;
		//tfile << setiosflags(ios::left) << setw(20) << "Variables=\"x\",\"y\",\"z\"" << endl;
		tfile.precision(5);
		tfile.setf(ios::scientific, ios::floatfield);

		j = 0, k = 0, m = 1;

		for (i = 0; i < nCell; i++)
		{
			tfile << setiosflags(ios::left) << setw(20) << "Zone" << endl;
			for (j = 0; j < nAllTri; j++)
			{
				if (j >= BeginEndTri[i].x&& j < (BeginEndTri[i].y + 1))
				{
					tfile << setiosflags(ios::left) << setw(20) << TriCenter[j].x;
					tfile << setiosflags(ios::left) << setw(20) << TriCenter[j].y;
					tfile << setiosflags(ios::left) << setw(20) << TriCenter[j].z;

					tfile << setiosflags(ios::left) << setw(20) << TriNorDir[j].x;
					tfile << setiosflags(ios::left) << setw(20) << TriNorDir[j].y;
					tfile << setiosflags(ios::left) << setw(20) << TriNorDir[j].z;



					tfile << setiosflags(ios::left) << setw(20) << j;

					tfile << endl;
				}
			}
			m = 1;
		}
		tfile.close();

		delete[]Pos;
	}
}

void Particle::DrivePar()
{
	if (StepCount >= StepEquil)//drive the system;
		DriveOn = 1;//if DriveOn==1, we should set the velocity of virtual particle, and the external force
	else
		DriveOn = 0;
}

void Particle::UpdatePar()
{
	UpdateCoor();
	AppPeriodicBC();
	ConCubeList();
	CalDen();
	PredictVel();
	CalVel();
	CalEOS();
	CalSDPDForce();
	CalRepForce();
	SetExtForce(DriveOn);
	cupar::CalGeoPara(StepCount, nAllTri, nCell, d_BBoxSDPD, d_CurPos, d_Tri, d_BeginEndTri, d_nThroBound, d_TotalCoor, d_TriNorDir, d_TriArea, d_RefTriArea, d_TriCenter,
		d_RefArea_Para, d_RefVol_Para, d_Area_Para, d_Vol_Para, d_Area, d_Vol, d_ParArea);
	CalDefForce();
	CalAggForce();
	CorrectVel();
	CalVel();

}

void Particle::UpdateCoor()
{
	cupar::UpdateFluidCoor(consts.nTotalPar, d_SortPart, d_TotalForce, d_TotalCoor, d_Velrhop);
}

void Particle::AppPeriodicBC()
{
	cupar::AppFluidPeriodicBC(consts.nTotalPar, d_SortPart, d_TotalCoor, d_nThroBound, d_BBoxSDPD);
}

void Particle::CalDen()
{
	cupar::CalFluidDen(consts.nTotalPar, d_TotalCoor, d_Velrhop, d_CC, d_nCube, d_OuterRegion, d_SortPart, d_BeginEndCube);
}

void Particle::PredictVel()
{
	cupar::PredictFluidVel(consts.nTotalPar, d_SortPart, d_Velrhop, d_VelHalf, d_TotalForce);
}

void Particle::CalVel()
{
	cupar::CalBound2Vel(consts.nTotalPar, d_TotalCoor, d_Velrhop, d_CC, d_nCube, d_OuterRegion, d_SortPart, d_BeginEndCube);
}

void Particle::CalEOS()
{
	cupar::EOS(consts.nTotalPar, d_SortPart, d_Velrhop, d_Press);
}

void Particle::CalSDPDForce()
{
	cupar::CalFluidSDPDForce(consts.nTotalPar, h_SDPDForce, d_TotalCoor, d_Velrhop, d_CC, d_nCube, d_SortPart, d_OuterRegion, d_BeginEndCube, d_TotalForce, d_Press);
}

void Particle::CalRepForce()
{
	cupar::CalFluidRepForce(consts.nTotalPar, d_TotalCoor, d_CC, d_nCube, d_SortPart, d_OuterRegion, d_BeginEndCube, d_TotalForce);
}

void Particle::SetExtForce(int DriveOn)
{
	cupar::SetFluidExtForce(DriveOn, consts.nTotalPar, d_SortPart, d_TotalForce);
}

void Particle::CorrectVel()
{
	//cupar::CorrectFliudVel(consts.nTotalPar, d_SortPart, d_TotalForce, d_VelHalf, d_Velrhop);
	cupar::CorrectFliudVel(consts.nTotalPar, d_SortPart, d_TotalForce, d_VelHalf, d_Velrhop, d_TotalCoor, d_CC, d_nCube, d_OuterRegion, d_BeginEndCube);
	//cupar::XSPHCorrect(consts.nTotalPar, d_SortPart, d_TotalForce, d_VelHalf, d_Velrhop);
}

void Particle::CalDefForce()
{
	cupar::CalCellDefForce(nCell, nAllEdge, nAllTri, consts.nTotalPar, consts.nCellPar, d_BBoxSDPD, d_Edge, d_Tri, d_TotalCoor, h_DefForce,  d_DefForce ,d_nThroBound, d_MaxRefEdgeLen, d_PowerIndex, d_PerLenWLC, d_HPow, d_TriNorDir, d_TriCenter, d_RefTriAngle, d_HBending, d_TriArea, d_RefTriArea, d_HLarea_Para, d_HGarea_Para, d_Area_Para, d_RefArea_Para, d_HVol_Para, d_Vol_Para, d_RefVol_Para, d_TotalForce, d_SortPart, d_CurPos);
}

void Particle::CalAggForce()
{
	cupar::CalCellAggForce(nCell, consts.nTotalPar, consts.nCellPar, d_TotalCoor, d_AggForce, d_CC, d_nCube, d_SortPart, d_CellTab, d_OuterRegion, d_BeginEndCube, d_TotalForce, d_ParArea);
}


void Particle::DeletePar()
{
	delete[] TotalCoor;
	delete[] Velrhop;
	delete[] VelHalf;
	delete[] SortPart;
	delete[] nThroBound;
	delete[]h_SDPDForce;

	CHECK(cudaFree(d_TotalCoor));
	CHECK(cudaFree(d_TotalForce));
	CHECK(cudaFree(d_Velrhop));
	CHECK(cudaFree(d_VelHalf));
	CHECK(cudaFree(d_SortPart));
	CHECK(cudaFree(d_CubePart));
	CHECK(cudaFree(d_nCube));
	CHECK(cudaFree(d_OuterRegion));
	CHECK(cudaFree(d_invWid));
	CHECK(cudaFree(d_CC));
	CHECK(cudaFree(d_BeginEndCube));
	CHECK(cudaFree(d_Press));
	CHECK(cudaFree(d_BBoxSDPD));
	CHECK(cudaFree(d_OldPos));
	CHECK(cudaFree(d_nThroBound));


	if (nCell > 0)
	{
		delete[] CellTab;
		delete[] Type;
		delete[] nPar;
		delete[] nTri;
		delete[] nEdge;
		delete[] BeginEndTri;
		delete[] BeginEndCoor;
		delete[] nEdgeInfo;
		delete[] ShearMod;
		delete[] ShearMod_Para;
		delete[] BendingMod;
		delete[] BendingMod_Para;
		delete[] HGarea;
		delete[] HGarea_Para;
		delete[] HLarea;
		delete[] HLarea_Para;
		delete[] HVol;
		delete[] HVol_Para;
		delete[] h_DefForce;
		delete[] h_AggForce;
		delete[] h_Area;
		delete[] h_Vol;
		delete[] TriNorDir;
		delete[] TriCenter;


		CHECK(cudaFree(d_CellTab));
		CHECK(cudaFree(d_CurPos));
		CHECK(cudaFree(d_BeginEndTri));
		CHECK(cudaFree(d_BendingMod_Para));
		CHECK(cudaFree(d_ShearMod_Para));
		CHECK(cudaFree(d_HGarea_Para));
		CHECK(cudaFree(d_HLarea_Para));
		CHECK(cudaFree(d_HVol_Para));
		CHECK(cudaFree(d_Edge));
		CHECK(cudaFree(d_Tri));
		CHECK(cudaFree(d_Area));
		CHECK(cudaFree(d_Vol));
		CHECK(cudaFree(d_TriNorDir));
		CHECK(cudaFree(d_TriCenter));
		CHECK(cudaFree(d_TriArea));
		CHECK(cudaFree(d_RefTriArea));
		CHECK(cudaFree(d_RefTriAngle));
		CHECK(cudaFree(d_RefEdgeLen));
		CHECK(cudaFree(d_RefArea_Para));
		CHECK(cudaFree(d_RefVol_Para));
		CHECK(cudaFree(d_Area_Para));
		CHECK(cudaFree(d_Vol_Para));
		CHECK(cudaFree(d_MaxRefEdgeLen));
		CHECK(cudaFree(d_PerLenWLC));
		CHECK(cudaFree(d_PowerIndex));
		CHECK(cudaFree(d_HPow));
		CHECK(cudaFree(d_HBending));
		CHECK(cudaFree(d_ParArea));
		CHECK(cudaFree(d_DefForce));
		CHECK(cudaFree(d_AggForce));
	}

}

float Particle::RandG(int &idum)
{
	float s, x, y, fac;
	static int iset = 0;
	static float gset;
	if (iset == 0)
	{
		do
		{
			x = 2.0*RandN(idum) - 1.0;//uniform random number in [-1,1]
			y = 2.0*RandN(idum) - 1.0;
			s = x * x + y * y;
		} while (s >= 1 || s == 0);
		fac = sqrt(-2.0*log(s) / s);
		gset = y * fac;
		iset = 1;
		return x * fac;
	}
	else
	{
		iset = 0;
		return gset;
	}
}

float Particle::RandN(int &idum) //& is reference transfer. generate the uniform random number on [0 1], and the average value is 0.5
{
	const unsigned long IM1 = 2147483563, IM2 = 2147483399;
	const unsigned long IA1 = 40014, IA2 = 40692, IQ1 = 53668, IQ2 = 52774;
	const unsigned long IR1 = 12211, IR2 = 3791, NTAB = 32, IMM1 = IM1 - 1;
	const unsigned long NDIV = 1 + IMM1 / NTAB;
	const float EPS = 3.0e-16, AM = 1.0 / IM1, RNMX = (1.0 - EPS);
	static int iy = 0, idum2 = 314159269;
	static int iv[NTAB];
	int j, k;
	float tempvalue;
	if (idum <= 0)
	{
		if (idum == 0)
			idum = 1;
		else
			idum = -idum;
		idum2 = idum;
		for (j = NTAB + 7; j >= 0; j--)
		{
			k = idum / IQ1;
			idum = IA1 * (idum - k * IQ1) - k * IR1;
			if (idum < 0)
				idum += IM1;
			if (j < NTAB)
				iv[j] = idum;
		}
		iy = iv[0];
	}
	k = idum / IQ1;
	idum = IA1 * (idum - k * IQ1) - k * IR1;
	if (idum < 0)
		idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
	if (idum2 < 0)
		idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = idum;
	if (iy < 1)
		iy += IMM1;
	if ((tempvalue = AM * iy) > RNMX)
		return RNMX;
	else
		return tempvalue;
}
