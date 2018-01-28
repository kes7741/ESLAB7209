/*
using namespace std;

__global__ void KERNEL_clc_concn_diffusion(uint_t *number_of_neighbors_,uint_t*pnb_,Real *m_,Real *rho_,Real *dconcn_,Real *concn_,Real *temp_,Real *h_,Real *dist_,Real *x_,Real *y_,Real *z_,Real *dwx_,Real *dwy_,Real *dwz_,int_t pnb_size,uint_t *p_type_);
__global__ void KERNEL_clc_predictor_concn(Real*dconcn_,Real *concn_,Real *concn0_,const Real dt);
__global__ void KERNEL_clc_precor_update_concn(Real*dconcn_,Real *concn_,Real *concn0_,const Real dt);
__global__ void KERNEL_clc_euler_update_concn(Real*dconcn_,Real *concn_,Real *concn0_,const Real dt);


////////////////////////////////////////////////////////////////////////
void Cuda_Particle_Array::calculate_concn_diffusion()
{
	KERNEL_clc_concn_diffusion<<<number_of_particles,thread_size>>>(number_of_neighbors,pnb,m,rho,dconcn,concn,temp,h,dist,x,y,z,dwx,dwy,dwz,pnb_size,p_type);
}
////////////////////////////////////////////////////////////////////////
void Cuda_Particle_Array::predictor_concn(const Real dt)
{
	KERNEL_clc_predictor_concn<<<number_of_particles,1>>>(dconcn,concn,concn0,dt);
}
//*/
////////////////////////////////////////////////////////////////////////
//Functions for concentration diffusion model
__global__ void KERNEL_clc_concn_diffusion(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	__shared__ Real cache[256];
	cache[threadIdx.x]=0;
	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	int_t j;																					// neighbor particle index-j
	uint_t non;

	Real mi,rhoi,rhoj;
	Real diffi,diffj;
	Real tempi,tempj;
	uint_t p_typei,p_typej;
	Real concni,concnj; //dconcn
	Real tdist,hi;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwx,tdwy,tdwz;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdist=Pa2[tid].dist;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;

		p_typei=Pa11[i].p_type;
		tempi=Pa11[i].temp;
		mi=Pa11[i].m;
		rhoi=Pa11[i].rho;
		hi=Pa11[i].h;
		xi=Pa11[i].x;
		yi=Pa11[i].y;
		zi=Pa11[i].z;

		p_typej=Pa11[j].p_type;
		tempj=Pa11[j].temp;
		rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		concni=Pa12[i].concn;																// concentration of particle i
		concnj=Pa12[j].concn;																// concentration on particle j

		diffi=diffusion_coefficient(tempi,p_typei);				// diffusion coefficient of particle i
		diffj=diffusion_coefficient(tempj,p_typej);			// diffusion coefficient of particle j

		cache[cache_idx]=mi*(diffi*rhoi+diffj*rhoj)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/(rhoi*rhoj*(tdist*tdist+0.01*hi*hi))*(concni-concnj)*(p_typei==p_typej);
	}
	__syncthreads();

	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa12[i].dconcn=cache[0];

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;																					// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mi,rhoi,rhoj;
	Real diffi,diffj;
	Real tempi,tempj;
	uint_t p_typei,p_typej;
	Real concni,concnj; //dconcn
	Real tdist,hi;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwx,tdwy,tdwz;
	Real tmp_Result=0.0;

	concni=Pa12[i].concn;																// concentration of particle i

	non=Pa11[i].number_of_neighbors;
	p_typei=Pa11[i].p_type;
	tempi=Pa11[i].temp;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	hi=Pa11[i].h;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	diffi=diffusion_coefficient(tempi,p_typei);				// diffusion coefficient of particle i
	// calculate diffusion rate
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdist=Pa2[tid].dist;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;

		p_typej=Pa11[j].p_type;
		tempj=Pa11[j].temp;
		rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		concnj=Pa12[j].concn;																// concentration on particle j
		diffj=diffusion_coefficient(tempj,p_typej);			// diffusion coefficient of particle j
		// Diffusion rate
		tmp_Result+=mi*(diffi*rhoi+diffj*rhoj)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/(rhoi*rhoj*(tdist*tdist+0.01*hi*hi))*(concni-concnj)*(p_typei==p_typej);
	}
	//save values to particle array
	Pa12[i].dconcn=tmp_Result;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_predictor_concn(int_t nop,Real tdt,part12*Pa12)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tconcn0,tconcnp,tdconcn_dt0;

	// predict density
	tconcn0=Pa12[i].concn0;												// Inital concentration
	tdconcn_dt0=Pa12[i].dconcn;										// Initial time derivative of concentration
	tconcnp=tconcn0+tdconcn_dt0*(tdt*0.5);			// Predict concentration (dconcn_dt0 : time derivatve of density of before time step)
	Pa12[i].concn=tconcnp;												// Update particle data by predicted concentration
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_precor_update_concn(int_t nop,Real tdt,part12*Pa12)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//Real tconcn;
	Real tconcn0,tconcnc,tdconcn_dt;						// concentration,time derivative of concentration ('0' : initial value / 'c' : corrected value for Predictor-Corrector time stepping scheme)

	// update concentration 
	tconcn0=Pa12[i].concn0;												// initial concentration
	tdconcn_dt=Pa12[i].dconcn;										// time derivative of concentration
	tconcnc=tconcn0+tdconcn_dt*(tdt*0.5);				// correct concentration
	Pa12[i].concn=tconcnc;

	// update concentration
	Pa12[i].concn0=2*tconcnc-tconcn0;
	Pa12[i].concn=2*tconcnc-tconcn0;
}
/*
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_euler_update_concn(Real*dconcn_,Real *concn_,Real *concn0_,const Real dt)
{
	//
}
//*/
////////////////////////////////////////////////////////////////////////
void update_properties_concn(int_t*vii,Real*vif,part12*Pa12)
{
	dim3 b,t;
	t.x=256;
	b.x=(number_of_particles-1)/t.x+1;

	switch(time_type){
		case Euler:
			// Eulerian time integration function
			//
			break;
		case Pre_Cor:
			// Predictor-Corrector time integration function
			KERNEL_clc_precor_update_concn<<<b,t>>>(number_of_particles,dt,Pa12);
			break;
		default:
			break;
	}
}
