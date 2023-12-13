//Particle.chu: implement the program consisting of wall ,fluid and cell membrane particle

#ifndef PARTICLES
#define PARTICLES

#include "TypesDef.cuh"

class Particle
{
public:
	//particle parameters on CPU
	Structconst consts;
	int nFluidPar, nBound1Par, nBound2Par, nB12Par, nFB2Par, nFB12Par;//the number of each type of SDPD particles, including bound1, bound2 and fluid	
	tfloat3* TotalCoor;//all particle coordinates, including fluid, wall, cell particle 
	tfloat3* TotalForce;//the sum of the forces on each particle
	tfloat4* Velrhop;//particle velocity and rhop
	tfloat3* VelHalf;//particle velocities at the half time step, used in velocity-Verlet algorithm	
	tfloat3* h_SDPDForce;
	float* Press;//particle pressure
	int* SortPart; //Variables generated to prepare for particle sorting	
	int nCube[NDIM];//the system domain (including the real and virtual boundary domains) is divided into many cubes in order to save computational costs. Only the particles in the same and neighboring cubes may have interaction
	float BBoxSDPD[2 * NDIM];
	float OuterRegion[NDIM], invWid[NDIM];//invWid: how many cubes per unit region
	int SizeCube;//number of cube

	int nCell, nAllTri, nAllEdge;//nCell: the number of cells; nCellPar: all the cell particles	
	int* CellTab;//the number of the cell in which the particle is located
	int* Type;//the type of cells, like RBC, platelets 
	int* nPar, *nTri, *nEdge;//nPar:number of particles per cell ; nTri:nPar:number of triangle per cell 
	int* nEdgeInfo;//the number of triangular edge information
	tint2* BeginEndTri;//Store the numbers of the first and last triangles on each cell
	tint2* BeginEndCoor;//Store the numbers of the first and last particles on each cell
	tint3* nThroBound;//Times of Cell particles through the boundary				   
	tint3* Tri;//[i].a (a=x,y,z),i->i^th triangle, x,y,z save vertex
	tint6* Edge;//[i].a (a=x,y,z,k,v,w),i->i^th edge ,include '6' information. For example: Edge(i, :)=[1 2 3 4 5 6], meaning that the 5-th and 6-th triangles have the same edge with particles 1 and 2. The rest particles of 5-th and 6-th are 3 and 4, respectively	 
	tfloat3* h_DefForce;//Deforamtion Force on all particles for output image
	tfloat3* h_AggForce;//Aggregation Force on all particles for output image
	float* h_Vol;//The volume of each cell
	float* h_Area;//The area of each cell 
	tfloat3* TriNorDir;//Normal direction of triangular on cell
	tfloat3* TriCenter;//Center of triangular on cell


	//physical parameters
	//float SoundSpeed;//sound speed
	//float ShearVis;//shear viscosity
	//float BulkVis;//bulk viscosity
	float DenInit;//initial particle density
	//float ConstExtForce;//the external force for Poiseuille flow
	float VVsigma;//an empirical parameter in Velocity-Verlet algorithm
	//int nBound1Start, nBound2Start, nFluidStart, nTotalPar;//nTotalPar: the number of all types of particles, including bound1, bound2 , fluid particles and Nodes

	int nType;//number of cell type	 
	float* ShearMod;//shear modulus
	float* BendingMod;//bending modulus
	float* ShearMod_Para;//shear modulus for parallel
	float* BendingMod_Para;//bending modulus for parallel
	float* HGarea;//the constraint constant for global area or cell, strong constraint
	float* HLarea;//the constraint constant for local area or each triangle, weak constraint 
	float* HLarea_Para;//the constraint constant for global area or cell, strong constraint for parallel
	float* HGarea_Para; //the constraint constant for local area or each triangle, weak constraint for parallel
	float* HVol;//the constraint constant for volume, strong constraint
	float* HVol_Para;//the constraint constant for volume, strong constraint for parallel
	//float MaxCut;//maximum cut-off radius of cell-cell interaction. Above this value, the cell-cell interaction is not considered. 
	//float zeroLen;//at this value, the interaction force is zero
	//float scaFac;//scaling factor 
	//float surEnergy;//surface energy
	float Rand;
	int iSeed;//random seeds 

	//particle parameters on GPU	
	float* d_OuterRegion, *d_invWid;
	int* d_nCube;//variables on GPU of nCube[NDIM]
	float3* d_TotalCoor;//all particle coordinates, including fluid, wall, cell particle 
	float4 *d_Velrhop;//particle velocity and rhop for GPU
	float3* d_VelHalf;//particle velocities at the half time step for GPU 
	float3* d_TotalForce;//各个粒子所受力总和
	float* d_Press;//particle pressure for GPU
	int3* d_CC;//particle located which cube on the x, y, z directions, respectively
	int* d_CubePart;//Store the 1D cube position of the particles on the device
	int* d_SortPart;//Variables generated to prepare for particle sorting
	int* d_OldPos;//Temporary  variables generated to prepare for particle sorting
	int2* d_BeginEndCube;//Store the starting and ending particles for each cube
	float* d_BBoxSDPD;//the boundary box of SDPD particles used for GPU.

	int* d_CurPos;//Particle position after sorting
	int* d_CellTab;//the number of the cell in which the particle is located
	int2* d_BeginEndTri;//Store the numbers of the first and last triangles on each cell
	int3* d_nThroBound;//Times of Cell particles through the boundary	
	int3* d_Tri;//[i].a (a=x,y,z),i->i^th triangle, x,y,z save vertex
	int6* d_Edge;//[i].a (a=x,y,z,k,v,w),i->i^th edge ,include '6' information. For example: Edge(i, :)=[1 2 3 4 5 6], meaning that the 5-th and 6-th triangles have the same edge with particles 1 and 2. The rest particles of 5-th and 6-th are 3 and 4, respectively	 
	float3* d_TriNorDir;//Normal direction of triangular on cell
	float* d_TriArea;//Area of triangular on cell
	float* d_RefTriArea;//Reference Area of triangular on cell
	float3* d_TriCenter;//Center of triangular on cell
	float* d_RefTriAngle;//Dihedral reference size of adjacent triangular on cell
	float* d_RefEdgeLen;//Reference Length of each edge of the triangle on cell
	float* d_MaxRefEdgeLen;//Maximum reference Length of each edge of the triangle on cell
	float* d_BendingMod_Para;//bending modulus for parallel
	float* d_ShearMod_Para;//shear modulus for parallel
	float* d_PerLenWLC;//Persistence length of each edge of a triangle on cell
	float* d_HPow;//the power function force for WLC
	float* d_HLarea_Para;//the constraint constant for global area or cell, strong constraint for parallel
	float* d_HGarea_Para; //the constraint constant for local area or each triangle, weak constraint for parallel
	float* d_HVol_Para;//the constraint constant for volume, strong constraint for parallel
	float* d_Vol;//The volume of each cell
	float* d_Area;//The area of each cell
	float* d_Area_Para;//The area of each cell for parallel. For programming purpose, we have triangles on each cell corresponding to one cell area. For example: d_Area_Para[3000]=20.0 ,meaning that the 3000th of all triangles in cells with an area of 20.0
	float* d_RefArea_Para;//The reference area of each cell for parallel. Other explanations are the same as variable a
	float* d_Vol_Para;//The volume of each cell for parallel.  Other explanations are the same as variable a
	float* d_RefVol_Para;//The reference volume of each cell for parallel.  Other explanations are the same as variable a
	float* d_PowerIndex;//the power index in the power law
	float* d_HBending;//the bending coefficient
	float* d_ParArea;//The number of the triangle where each cell particle is located, used for cell aggregation
	float3* d_DefForce;//Deforamtion Force on all particles
	float3* d_AggForce;//Aggregation Force on all particles 



	void InitPar();//initial the particle system
	void LoadParInfo();//load particle information 
	void LoadFBPar(tfloat3* SDPDCoor);//load fluid and wall particle position 
	void LoadSDPDPara();//load SDPD physical parameter
	void LoadCellStruct(tfloat3* CellCoor);//load cell membrane particle and structure
	void LoadCellPara();//load SDPD physical parameter
	void AllocCpuMemory();//allocate CPU memory
	void CoorMerge(tfloat3* SDPDCoor, tfloat3* CellCoor);//merge SDPD particles with cell particle coordinates
	void SetPar();
	void SetParticle();
	void ConfigDomain();
	void AllocGpuMemory();
	void ParticlesDataUp();
	void ConstantDataUp();
	void ConCubeList();

	void OutputPar();
	void ParticlesDataReturn();//returns data from the GPU to the CPU
	void OutputCoor(long int StepData);//output coordinates
	void OutputField(long int StepData);//output coordinates, velocity, density, force, and so on, for the purpose of obtaining the field format of data
	void OutPutCellCoor(long int StepData);//output cell particle coordinates
	void OutPutCellForce(long int StepData);//output cell particle force
	void OutPutCellGeo(long int StepData);//output cell geo
	void OutPutTriNorDir(long int StepData);//output center of gravity and normal direction of triangle on cell

	void DrivePar();
	int DriveOn;

	void UpdatePar();
	void UpdateCoor();
	void AppPeriodicBC();
	void CalDen();//calculate the fluid (include cell particles) density  
	void PredictVel();//predict the fluid velocity
	void CalVel();//calculate the velocity of bound 2 particles by the fluid velocity
	void CalEOS();//equation of state, i.e., get the pressure from the density
	void CalSDPDForce();//calculate the SDPD force
	void CalRepForce();//calculate the repulsive force
	void SetExtForce(int DriveOn);//set the external force
	void CorrectVel();//correct the fluid velocity
	void CalDefForce();//calculate the force on cells due to deformation
	void CalAggForce();//calculate the force on cells due to Aggregation

	void DeletePar();

	float RandN(int &idum);
	float RandG(int &idum);

};

#endif