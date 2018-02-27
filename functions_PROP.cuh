////////////////////////////////////////////////////////////////////////
#define NU0_HB	1.0			//Herschel-Bulkey model parameter
#define TAU0_HB	18.24		//Herschel-Bulkey model parameter (for lava flow)
#define K0_HB		1.90		//Herschel-Bulkey model parameter (for lava flow)
#define	N0_HB		0.53		//Herschel-Bulkey model parameter (for lava flow)

////////////////////////////////////////////////////////////////////////
__host__ __device__ Real interp1(Real *x_data,Real *y_data,Real tx)
{
	//Real y1,y2,y3,y4;
	Real y;

	if(tx<=x_data[1]){
		y=y_data[0]+(y_data[1]-y_data[0])/(x_data[1]-x_data[0])*(tx-x_data[0]);
	}else if((tx>x_data[1])&(tx<=x_data[2])){
		y=y_data[1]+(y_data[2]-y_data[1])/(x_data[2]-x_data[1])*(tx-x_data[1]);
	}else if((tx>x_data[2])&(tx<=x_data[3])){
		y=y_data[2]+(y_data[3]-y_data[2])/(x_data[3]-x_data[2])*(tx-x_data[2]);
	}else if(tx>x_data[3]){
		y=y_data[3]+(y_data[4]-y_data[3])/(x_data[4]-x_data[3])*(tx-x_data[3]);
	}
	/*
	y1=y_data[0]+(y_data[1]-y_data[0])/(x_data[1]-x_data[0])*(tx-x_data[0]);
	y2=y_data[1]+(y_data[2]-y_data[1])/(x_data[2]-x_data[1])*(tx-x_data[1]);
	y3=y_data[2]+(y_data[3]-y_data[2])/(x_data[3]-x_data[2])*(tx-x_data[2]);
	y4=y_data[3]+(y_data[4]-y_data[3])/(x_data[4]-x_data[3])*(tx-x_data[3]);

	y=y1*(tx<=x_data[1])+y2*((tx>x_data[1])&(tx<=x_data[2]))+y3*((tx>x_data[2])&(tx<=x_data[3]))+y4*(tx>x_data[3]);
	//*/
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real interp2(Real *x_data, Real *y_data, int size, Real x)
{

	Real y;
	int end_idx = size - 1;

	if (x_data[end_idx] < x)
	{
		y = y_data[end_idx] + (y_data[end_idx] - y_data[end_idx - 1]) / (x_data[end_idx] - x_data[end_idx - 1]) * (x - x_data[end_idx]);
	}
	else if (x <= x_data[0])
	{
		y = y_data[0] + (y_data[1] - y_data[0]) / (x_data[1] - x_data[0]) * (x - x_data[0]);
	}
	else
	{
		for (int i = 0; i < size; i++)
		{
			if ((x_data[i] < x) & (x <= x_data[i + 1]))
			{
				y = y_data[i] + (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i]) * (x - x_data[i]);
				break;
			}
		}
	}

	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real htoT(Real tenthalpy,uint_t tp_type)
{
	// corium data
	Real x_data1[5]={-554932,215789,757894,905263,1268421};
	Real y_data1[5]={280,1537,2300,2450,2650};

	// concrete data (calceous concrete) - (Sevon, 2008, CCI experiment)
	//Real x_data2[6] = { 0, 47000, 172020, 845000, 1706800, 3548000 };
	//Real y_data2[6] = { 298.15, 347.15, 399.15, 873.15, 1100.2, 2573.2 };

	// concrete data (siliceous concrete) - (Sevon, 2008, CCI experiment)
	Real x_data2[4] = { 0, 41000, 132000, 2474500 };
	Real y_data2[4] = { 298.15, 347.15, 399.15, 2273.15 };

	// stainless steel data
	Real x_data3[5] = { 47838, 151494, 262962, 380730, 503454 };
	Real y_data3[5] = { 373.15, 573.15, 773.15, 973.15, 1169.15 };

	// water data
	Real x_data4[2] = { 0, 418700};
	Real y_data4[2] = { 273.15, 373.15};

	Real y;

	tp_type = abs(tp_type);

	switch(tp_type){
		case CORIUM:
			y=interp1(x_data1,y_data1,tenthalpy);
			break;
		case CONCRETE:
			y=interp2(x_data2,y_data2,4,tenthalpy);
			break;
		case MCCI_CORIUM:
			y = interp2(x_data1, y_data1, 5, tenthalpy);
			break;
		case IVR_CORIUM:
			y=interp1(x_data1,y_data1,tenthalpy);
			break;
		case IVR_METAL:
			y=interp1(x_data1,y_data1,tenthalpy);
			break;
		case IVR_VESSEL:
			y = interp1(x_data3, y_data3, tenthalpy);
			break;
		// case BOUNDARY:
		// 	y = interp1(x_data3, y_data3, tenthalpy);
		// 	break;
		default:
			y=interp2(x_data4,y_data4,2,tenthalpy);
			break;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real Ttoh(Real temp,uint_t p_type)
{
	// corium data
	Real y_data1[5]={-554932,215789,757894,905263,1268421};
	Real x_data1[5]={280,1537,2300,2450,2650};
	// concrete data (calceous concrete) - (Sevon, 2008, CCI experiment)
	//Real y_data2[6] = { 0, 47000, 172020, 845000, 1706800, 3548000 };
	//Real x_data2[6] = { 298.15, 347.15, 399.15, 873.15, 1100.2,  2573.2 };

	// concrete data (siliceous concrete) - (Sevon, 2008, CCI experiment)
	Real y_data2[4] = { 0, 41000, 132000, 2474500 };
	Real x_data2[4] = { 298.15, 347.15, 399.15, 2273.15 };

	// stainless steel data
	Real y_data3[5] = { 47838, 151494, 262962, 380730, 503454 };
	Real x_data3[5] = { 373.15, 573.15, 773.15, 973.15, 1169.15 };

	// water data
	Real y_data4[2] = { 0, 418700};
	Real x_data4[2] = { 273.15, 373.15};

	Real y;

	p_type = abs(p_type);

	switch(p_type){
		case CORIUM:
			y=interp1(x_data1,y_data1,temp);		
			break;
		case CONCRETE:
			y=interp2(x_data2,y_data2,4,temp);		
			break;
		case MCCI_CORIUM:
			//y = interp1(x_data1, y_data1, temp);
			y = interp2(x_data1,y_data1,5,temp);
			break;
		case IVR_CORIUM:
			y = interp1(x_data1, y_data1, temp);
			break;
		case IVR_METAL:
			y=interp1(x_data1,y_data1,temp);
			break;
		case IVR_VESSEL:
			y = interp1(x_data3, y_data3, temp);
			break;
		// case BOUNDARY:
		// 	y = interp1(x_data3, y_data3, temp);
		// 	break;
		default:
			y=interp2(x_data4,y_data4,2,temp);
			break;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real viscosity(Real temp,uint_t p_type)
{
	// corium data
	Real x_data1[5]={300,1000,1500,2000,2500};
	Real y_data1[5]={50,12,5,0.1,0.01};

	// concrete data
	Real x_data2[5]={300,1000,1500,2000,2500};
	Real y_data2[5]={50,12,5,0.1,0.01};

	// water data
	Real x_data3[8] = { 273.16, 283.15, 293.15, 303.15, 323.15, 343.15, 363.15, 373.15 };
	Real y_data3[8] = { 0.0017914, 0.001306, 0.0010016, 0.0007972, 0.0005465, 0.0004035, 0.0003142, 0.0002816 };

	Real vis;

	p_type = abs(p_type);

	switch(p_type){
		case CORIUM:
			vis=interp1(x_data1,y_data1,temp);
			vis=fmin(vis,100.0);
			break;
		case CONCRETE:
			vis=interp1(x_data2,y_data2,temp);
			vis=fmin(vis,100.0);
			break;
		case MCCI_CORIUM:
			vis = interp1(x_data1, y_data1, temp);
			vis = fmin(vis, 100.0);
			break;
		case IVR_CORIUM:
			vis=interp1(x_data1,y_data1,temp);
			vis=fmin(vis,100.0);
			break;
		case IVR_METAL:
			vis=interp1(x_data1,y_data1,temp);
			vis=fmin(vis,100.0);
			break;
		case IVR_VESSEL:
			vis = interp1(x_data1, y_data1, temp);
			vis = fmin(vis, 100.0);
			break;
		// case BOUNDARY:
		// 	vis=interp1(x_data1,y_data1,temp);
		// 	vis=fmin(vis,100.0);
		// 	break;
		default:
			vis = interp2(x_data3, y_data3, 8, temp);
			//vis=0.001;	// water viscosity
			break;
	}
	return vis;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real conductivity(Real temp,uint_t p_type)
{
	//// IVR_CORIUM data (��¥)
	Real x_data1[5] = { 300, 1000, 1500, 2000, 2500 };
	Real y_data1[5] = { 0.2, 0.2, 1, 2, 3.25 };


	//// concrete data (calceous concrete)
	//Real x_data2[4] = { 300, 373, 374, 375 };
	//Real y_data2[4] = { 2.0, 2.0, 1.1, 1.1 };

	//// concrete data (siliceous concrete)
	//Real x_data2[4] = { 300, 1073, 1074,1075};
	//Real y_data2[4] = { 2.5, 2.5, 1.3, 1.3 };

	Real cond;

	p_type = abs(p_type);

	switch(p_type){
		case CORIUM:
			cond=1.65*1000;
			break;
		case CONCRETE:
			cond = 1.2 * 500;
			break;
		case MCCI_CORIUM:
			cond = 3.0 * 500;
			break;
		case IVR_CORIUM:
			cond = interp1(x_data1, y_data1, temp) * 50.0;
			break;
		case IVR_METAL:
			cond = 36.0 * 50;
			break;
		case IVR_VESSEL:
			//cond = 24.1;
			cond = 24.1 * 50;
			break;
		// case BOUNDARY:
		// 	//cond = 24.1;
		// 	cond = 24.1 * 50;
		// 	break;
		default:
			cond=0.58;		//water conductivity
			break;
	}
	return cond;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real sigma(Real temp,uint_t p_type)
{
	Real y;

	p_type = abs(p_type);

	switch(p_type){
		case CORIUM:
			y=71.97*1e-3;
			break;
		case CONCRETE:
			y=71.97*1e-3;
			break;
		case MCCI_CORIUM:
			y = 71.97 * 1e-3;
			break;
		case IVR_CORIUM:
			y = 71.97 * 1e-3;
			break;
		case IVR_METAL:
			y = 71.97 * 1e-3;
			break;
		case IVR_VESSEL:
			y = 71.97 * 1e-3;
			break;
		default:
			y=0.072;		// water surface tension
			break;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real diffusion_coefficient(Real temp,uint_t p_type)
{
	Real y;

	p_type = abs(p_type);

	switch(p_type){
		case CORIUM:
			y=0;
			break;
		case CONCRETE:
			y=0;
			break;
		case MCCI_CORIUM:
			y=0;
			break;
		case IVR_CORIUM:
			y=0;
			break;
		case IVR_METAL:
			y=0;
			break;
		case IVR_VESSEL:
			y=0;
			break;
		default:
			y=1.38e-6;  				//0.58*0.01/4187.	// diffusivity for modeling salt-finger by Monaghan's 
			//y=1.43e-7;				// physical diffusivity * 0.1 for modeling salt-finger by Monaghan's 
			break;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
/*
__host__ __device__ Real soundspeed(uint_t p_type)
{
	Real y;

	y = 200;

	return y;
}
*/
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real reference_density(uint_t tp_type,Real ttemp,Real tconcn)
{
	//water density - temp data
	Real x_data1[9] = { 273.25, 288.15, 298.15, 308.15, 313.15, 328.15, 343.15, 363.15, 373.15 };
	Real y_data1[9] = { 999.85, 999.1, 997.05, 994.03, 992.22, 985.69, 977.76, 965.31, 958.35 };

	Real y;

	tp_type = abs(tp_type);

	switch(tp_type){
		case CORIUM:
			y=5890.0;
			break;
		case CONCRETE:
			y=2000.0;
			break;
		case MCCI_CORIUM:
			y = 6000.;
			break;
		case IVR_CORIUM:
			y=5890.0;
			break;
		case IVR_METAL:
			//y=5000.0;
			y=3000.0;		//temporary
			break;
		case IVR_VESSEL:
			//y=7020;
			y=5890.0;		//temporary
			break;
		// case BOUNDARY:
		// 	y=5890.0;
		// 	break;
		default:
			y = interp2(x_data1, y_data1, 9, ttemp);
			//y=1000.0;
			break;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real reference_density2(uint_t p_type,Real temp,Real m,Real h, int d)
{
	Real y;
	Real vol,s;
	s=h/1.5;
	vol=pow(s,d);
	y = m/vol;
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__ Real thermal_expansion(Real temp,uint_t p_type)
{
	Real y;

	p_type = abs(p_type);

	switch(p_type){
		case CORIUM:
			y=3.81e-4;
			break;
		case CONCRETE:
			y=3.81e-4;
			break;
		case MCCI_CORIUM:
			y=3.81e-4;
			break;
		case IVR_CORIUM:
			y=3.81e-4 * 2;
			break;
		case IVR_METAL:
			y=3.81e-4 * 2;
			break;
		case IVR_VESSEL:
			y = 3.81e-4 * 2;
			break;
		// case BOUNDARY:
		// 	y=3.81e-4 * 2;
		// 	break;
		default:
			y=2.1e-4;		// water thermal expansion coefficient(beta) at 293.15K 
			break;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_EOS(int_t nop,Real tgamma,Real tsoundspeed,Real trho0_eos, part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real rhoi,ci,tB,rho0;
	Real tpressure;
	Real rho_refi=Pa11[i].rho_ref;
	uint_t p_typei=Pa11[i].p_type;

	rhoi=Pa11[i].rho;
	ci=tsoundspeed;
	rho0=fmax(1000,trho0_eos);	//minimum rho0... need to discuss later
	tB = ci*ci*rho0/tgamma;

	p_typei = abs(p_typei);

	switch(p_typei){
		case CORIUM:
			tpressure=tB*(pow(rhoi/rho_refi,tgamma)-1.0);
			break;
		case CONCRETE:
			tpressure=tB*(pow(rhoi/rho_refi,tgamma)-1.0);
			break;
		case BOUNDARY:
			tpressure=tB*(pow(rhoi/rho_refi,tgamma)-1.0);
			break;
		case IVR_CORIUM:
			tpressure=tB*(pow(rhoi/rho_refi,tgamma)-1.0);
			break;
		case IVR_METAL:
			tpressure=tB*(pow(rhoi/rho_refi,tgamma)-1.0);
			break;
		default:
			tpressure=tB*(pow(rhoi/rho_refi,tgamma)-1.0);
			break;
	}
	Pa11[i].pres=tpressure;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real DEVICE_clc_heat_source(Real temp,Real lbl_surf, uint_t p_type)
{
	Real y=0.0;
	Real C0=5.67e-8;
	Real eps=0.8;

	if(lbl_surf>0.5){
		y=-C0*eps*temp*temp*temp*temp*10;
		//y=0.;
	}
	return y;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real DEVICE_clc_heat_generation(Real temp, uint_t p_type)
{
	Real qs=0.0;

	p_type = abs(p_type);

	if(p_type==IVR_CORIUM){
		qs=6700000.0;			// volumetric heat generation rate [W/m^3]
	}

	if(p_type==MCCI_CORIUM){
		qs=9.6e6;				// volumetric heat generation rate [W/m^3]
	}

	return qs;
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real DEVICE_clc_boiling_h(Real temp, Real lbl_surf, uint_t p_type)
{
	Real y = 0.;
	Real T_sat = 373.15;

	p_type = abs(p_type);

	if ((lbl_surf > 0.5) & (p_type == IVR_VESSEL))
	{
		y = -2.54e05 * (8.20e-02*(temp - T_sat)) * (8.20e-02*(temp - T_sat)) * (8.20e-02*(temp - T_sat));
	}

	return y;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_find_psedo_max(int_t nop,part11*Pa11,part2*Pa2,Real*prove)
{
/*
	__shared__ Real cache1[1024];

	cache1[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;

	int id=nop/1024*i;
	cache1[id]=Pa11[i].ux*Pa11[i].ux+Pa11[i].uy*Pa11[i].uy+Pa11[i].uz*Pa11[i].uz;

	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cache1[cache_idx]=fmax(cachex[cache_idx],cachex[cache_idx+s]);
		}
		__syncthreads();
	}
	if(cache_idx==0){
		&prove=cachex[0];
	}
	*/
}
////////////////////////////////////////////////////////////////////////
__host__ __device__  Real K_to_eta(Real tK_stiff)
{
	// corium data
	Real x_data1[5]={0,1000,5000,25000,100000};
	Real y_data1[5]={1.0,1.8,2.3,2.5,2.5};

	Real y=1;

	y=interp1(x_data1,y_data1,tK_stiff);

	return y;
}
