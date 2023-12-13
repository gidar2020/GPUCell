//TypesDef.cuh: Some self-defined structures used in programs

#ifndef _TypesDef_
#define _TypesDef_

typedef struct {

	int x, y;

}tint2;

typedef struct {

	int x, y, z;

}tint3;

typedef struct {

	int x, y, z, k, v, w;

}tint6;

typedef struct {

	int x, y, z, k, v, w;

}int6;


typedef struct {

	float x, y, z, w, v;

}tfloat5;

typedef struct {

	float x, y, z, w;

}tfloat4;

typedef struct {

	float x, y, z;

}tfloat3;

typedef struct {

	//int nSDPDStart;
	unsigned nFluidStart;
	unsigned nBound2Start;
	unsigned nCellStart;
	unsigned nTotalPar;
	unsigned nCellPar;

	long int iSeed;//random seed

	float CutRadius;
	float Mass;
	float SoundSpeed1;//FiuidPar sound speed
	float SoundSpeed2;//CellPar sound speed
	float DenInit;
	float VVsigma;
	float ShearVis;//shear viscosity
	float BulkVis;//bulk viscosity
	float ConstExtForce;//the external force for Poiseuille flow

	float TimeStep;

	float surEnergy;//surface energy
	float MaxCut;//maximum cut-off radius of cell-cell interaction. Above this value, the cell-cell interaction is not considered. 
	float zeroLen;//at this value, the interaction force is zero
	float scaFac;//scaling factor 
	float Temp;//particle temperature, we assume that all particles have the same unit temperature

}Structconst;//Structure of constant required for Particle interaction 


#endif