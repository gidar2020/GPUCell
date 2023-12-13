//ParticleGPU.cu: Declares functions and CUDA kernels for the Particle Interaction and System Update.

#ifndef _PARTICLEGPU_
#define _PARTICLEGPU_

#define SDPDSIZE 128

namespace cupar
{
	void ConstsDataUp(const Structconst *consts);//put constants commonly used in particle interactions into constant memory
	void UpdateFluidCoor(int n, int* d_sortpart, float3* d_Totalforce, float3* d_totalcoor, float4* d_velrhop);//update coordinates of fluid particles and cell membrane particle s
	void AppFluidPeriodicBC(int n, int* d_sortpart, float3* d_totalcoor, int3* d_nthrobound, float* d_bboxsdpd);//apply (fluid) periodic boundary conditions
	void CalFluidDen(int n, float3* d_totalcoor, float4* d_velrhop, int3* d_cc, int* d_ncube, float* d_outerregion, int* d_sortpart, int2* d_beginendcube);//calculate fluid density,note that the bound2 density is assumed to be constant
	void PredictFluidVel(int n, int* d_sortpart, float4* d_velrhop, float3* d_velhalf, float3* d_Totalforce);//Predict Fluid Velocity
	void CalBound2Vel(int n, float3* d_totalcoor, float4* d_velrhop, int3* d_cc, int* d_ncube, float* d_outerregion, int* d_sortpart, int2* d_beginendcube);//calculate the velocity of bound 2 particles	
	void EOS(int n, int* d_sortpart, float4* d_velrhop, float* d_press);//equation of state, i.e., get the pressure from the density
	void CalFluidSDPDForce(int n, tfloat3* d_sdpdforce, float3* d_totalcoor, float4* d_velrhop, int3* d_cc, int* d_ncube, int* d_sortpart, float* d_outerreigon, int2* d_beginendcube, float3* d_totalforce, float* d_press);//calculate the SDPDforce of the fluid and cell particle
	void CalFluidRepForce(int n, float3* d_totalcoor, int3* d_cc, int* d_ncube, int* d_sortpart, float* d_outerreigon, int2* d_beginendcube, float3* d_totalforce);//calculate the repulsive force of the fluid particle
	void SetFluidExtForce(int DriveOn, int n, int* d_sortpart, float3* d_totalforce);//set the external force on the fluid
	//void CorrectFliudVel(int n, int* d_sortpart, float3* d_Totalforce, float3* d_velhalf, float4* d_velrhop);//correct fluid velocity
	void CorrectFliudVel(int n, int* d_sortpart, float3* d_totalforce, float3* d_velhalf, float4* d_velrhop, float3* d_totalcoor, int3*d_cc, int* d_ncube, float* d_outerreigon, int2* d_beginendcube);



	void CalGeoPara(long stepcount, int ntri, int ncell, float* d_bbox, int* d_curpos, int3* d_tri, int2* d_beginendtri, int3* d_nthrobound, float3* d_totalcoor, float3* d_trinordir, float* d_triarea,
		float* d_reftriarea, float3* d_tricenter, float* d_refarea_para, float* d_refvol_para, float* d_area_para, float* d_vol_para, float* d_area, float* d_vol, float* d_pararea);

	void SetMechPara(int ncell, int nedge, int6* d_edge, int* d_curpos, float* d_refedgelen, float* d_maxrefedgelen, float* d_bendmodpara, float* d_shearmodpara, float* d_hbending,
		float* d_powerindex, float* d_perlenWLC, float* d_hpow, float* d_reftriangle, float3* d_trinordir, float3* d_tricente, float3* d_coor);
	void CalCellDefForce(int ncell, unsigned nedge, unsigned ntri, unsigned ntotalpar, unsigned ncellpar, float* d_bbox, int6* d_edge, int3* d_tri, float3* d_coor, tfloat3* defforce, float3* d_defforce, int3* d_nthrobound,
		float* d_maxrefedgelen, float* d_powerindex, float* d_perlenWLC, float* d_hpow, float3* d_trinordir, float3* d_tricenter,
		float* d_reftriangle, float* d_hbending, float* d_triarea, float* d_reftriarea, float* d_hlarea_para, float* d_hgarea_para, float* d_area_para,
		float* d_refarea_para, float* d_hvol_para, float* d_vol_para, float* d_refvol_para, float3* d_totalforce, int* d_sortpart, int* d_curpos);
	void CalCellAggForce(unsigned ncell, unsigned n, unsigned ncellpar, float3* d_totalcoor, float3* d_aggforce, int3* d_cc, int* d_ncube, int* d_sortpart,
		int* d_celltab, float* d_outerreigon, int2* d_beginendcube, float3* d_totalforce, float* d_pararea);

}

#endif