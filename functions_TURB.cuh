////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_strain_rate(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachex[256];
	 __shared__ Real cachey[256];
	 __shared__ Real cachez[256];

	cachex[threadIdx.x]=0;
	cachey[threadIdx.x]=0;
	cachez[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;

	Real mj,rhoi,rhoj;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real uij2;
	Real Sa;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		// dist
		tdist=Pa2[tid].dist;
		// calculate strain rate
		if(tdist>0){
			j=Pa2[tid].pnb;
			// kernel (by esk)???
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;
			// density
			rhoi=Pa11[i].rho;
			// position
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			// velocity
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			// position
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			// velocity
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;

			uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);

			cachex[cache_idx]=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(xi-xj)*tdwx*uij2;
			cachey[cache_idx]=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(yi-yj)*tdwy*uij2;
			cachez[cache_idx]=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(zi-zj)*tdwz*uij2;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachex[cache_idx]+=cachex[cache_idx+s];
			cachey[cache_idx]+=cachey[cache_idx+s];
			cachez[cache_idx]+=cachez[cache_idx+s];
		}
		__syncthreads();
	}
	//if(cache_idx==0) Pa12[i].SR=0;
	if(cache_idx==0)
	{
		Sa=-0.5*(cachex[0]+cachey[0]+cachez[0]);
		Sa=fmax(1e-20,Sa);
		Pa12[i].SR=sqrtf(Sa);
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj;

	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real uij2;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	// density
	rhoi=Pa11[i].rho;
	// position
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	// velocity
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel (by esk)???
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// dist
		tdist=Pa2[tid].dist;

		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// position
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;

		uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);

		// calculate strain rate
		if(tdist>0){
			tmpx+=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(xi-xj)*tdwx*uij2;
			tmpy+=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(yi-yj)*tdwy*uij2;
			tmpz+=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(zi-zj)*tdwz*uij2;
		}
	}

	// save values to particle array
	Pa12[i].SR=-0.5*(tmpx+tmpy+tmpz);
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_strain_rate2(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachex[256];
	 __shared__ Real cachey[256];
	 __shared__ Real cachez[256];

	cachex[threadIdx.x]=0;
	cachey[threadIdx.x]=0;
	cachez[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;

	Real mj,rhoi,rhoj;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real uij2;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		// dist
		tdist=Pa2[tid].dist;
		// calculate strain rate
		if(tdist>0){
			j=Pa2[tid].pnb;
			// kernel (by esk)???
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;
			// density
			rhoi=Pa11[i].rho;
			// position
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			// velocity
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			// position
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			// velocity
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;

			uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);

			cachex[cache_idx]=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(xi-xj)*tdwx*uij2;
			cachey[cache_idx]=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(yi-yj)*tdwy*uij2;
			cachez[cache_idx]=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(zi-zj)*tdwz*uij2;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachex[cache_idx]+=cachex[cache_idx+s];
			cachey[cache_idx]+=cachey[cache_idx+s];
			cachez[cache_idx]+=cachez[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0) Pa12[i].SR=-0.5*(cachex[0]+cachey[0]+cachez[0]);

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj;

	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real uij2;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	// density
	rhoi=Pa11[i].rho;
	// position
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	// velocity
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel (by esk)???
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// dist
		tdist=Pa2[tid].dist;

		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// position
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;

		uij2=(uxi-uxj)*(uxi-uxj)+(uyi-uyj)*(uyi-uyj)+(uzi-uzj)*(uzi-uzj);

		// calculate strain rate
		if(tdist>0){
			tmpx+=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(xi-xj)*tdwx*uij2;
			tmpy+=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(yi-yj)*tdwy*uij2;
			tmpz+=mj*(rhoi+rhoj)/(rhoi*rhoj*tdist*tdist)*(zi-zj)*tdwz*uij2;
		}
	}

	// save values to particle array
	Pa12[i].SR=-0.5*(tmpx+tmpy+tmpz);
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_turb_viscosity(int_t nop,part11*Pa11,part12*Pa12)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//Real vis_t_;
	if(Pa12[i].e_turb>0){
		//vis_t_=C_mu*Pa11[i].rho*Pa12[i].k_turb*Pa12[i].k_turb/Pa12[i].e_turb;
		//Pa12[i].vis_t=vis_t_;
		Pa12[i].vis_t=C_mu*Pa11[i].rho*Pa12[i].k_turb*Pa12[i].k_turb/Pa12[i].e_turb;
	}else{
		Pa12[i].vis_t=0;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_HB_viscosity(int_t nop,part11*Pa11,part12*Pa12)
{
	// calculation of Herschel-Bulkley Viscosity (Visco-plastic)
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	Real SR=Pa12[i].SR;	//strain-rate
	Real vis_a;
	if(i>=nop) return;

	//check equation
	vis_a=TAU0_HB/SR+K0_HB*powf(SR,N0_HB-1);
	vis_a=fmin(NU0_HB,vis_a);

	Pa12[i].vis_t=vis_a;
	//Pa12[i].vis_t=0;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_klm_turb(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachex[256];
	 __shared__ Real cachey[256];
	 __shared__ Real cachez[256];

	cachex[threadIdx.x]=0;
	cachey[threadIdx.x]=0;
	cachez[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t ptypei,ptypej;
	Real mj,rhoj,visi,visj;
	Real tempi,tempj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real tPi,tPi2,tSR;
	Real vis_ti,vis_tj;
	Real k_turbi,k_turbj,e_turbi;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		// dist
		tdist=Pa2[tid].dist;
		// calculate strain rate
		if(tdist>0){
			j=Pa2[tid].pnb;
			// kernel (by esk)???
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;
			// density
			tempi=Pa11[i].temp;
			ptypei=Pa11[i].p_type;
			// position
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			// strain rate: 2S:S
			tSR=Pa12[i].SR;
			// turbulence viscosity
			vis_ti=Pa12[i].vis_t;
			// turbulence kinetic energy
			k_turbi=Pa12[i].k_turb;
			// turbulence dissipation
			e_turbi=Pa12[i].e_turb;
			ptypej=Pa11[j].p_type;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			tempj=Pa11[j].temp;
			// position
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			// turbulence viscosity
			vis_tj=Pa12[j].vis_t;
			// turbulence kinetic energy
			k_turbj=Pa12[j].k_turb;

			// viscosity
			visi=viscosity(tempi,ptypei);
			visj=viscosity(tempj,ptypej);

			// turbulence production rate: Pi
			if(e_turbi==0){
				tPi=0;
			}else{
				tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
				tPi2=0.3*k_turbi*sqrtf(tSR);
				tPi=fmin(tPi,tPi2);
			}
			cachex[cache_idx]=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*(xi-xj)*tdwx;
			cachey[cache_idx]=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*(yi-yj)*tdwy;
			cachez[cache_idx]=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*(zi-zj)*tdwz;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachex[cache_idx]+=cachex[cache_idx+s];
			cachey[cache_idx]+=cachey[cache_idx+s];
			cachez[cache_idx]+=cachez[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0){
		Pa12[i].dk_turb=tPi-e_turbi+(cachex[0]+cachey[0]+cachez[0]);
		Pa12[i].e_turb=powf(C_mu,0.75)*powf(k_turbi,1.5)/Lm;		// (by esk)xxx => fast math
	}
	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	//Real uij2,rhoi,e_turbj;
	uint_t ptypei,ptypej;
	Real mj,rhoj,visi,visj;
	Real tempi,tempj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real tPi,tPi2,tSR;
	Real vis_ti,vis_tj;
	Real k_turbi,k_turbj,e_turbi;

	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	// density
	//rhoi=Pa11[i].rho;
	tempi=Pa11[i].temp;
	ptypei=Pa11[i].p_type;
	// position
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	// strain rate: 2S:S
	tSR=Pa12[i].SR;
	// turbulence viscosity
	vis_ti=Pa12[i].vis_t;
	// turbulence kinetic energy
	k_turbi=Pa12[i].k_turb;
	// turbulence dissipation
	e_turbi=Pa12[i].e_turb;

	visi=viscosity(tempi,ptypei);

	tmpx=tmpy=tmpz=0.0;
	// calculate contribution of j particle on density variation (drho)
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel (by esk)???
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// dist
		tdist=Pa2[tid].dist;

		ptypej=Pa11[j].p_type;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		tempj=Pa11[j].temp;
		// position
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		// turbulence viscosity
		vis_tj=Pa12[j].vis_t;
		// turbulence kinetic energy
		k_turbj=Pa12[j].k_turb;
		// turbulence dissipation
		//e_turbj=Pa12[j].e_turb;

		// viscosity
		visj=viscosity(tempj,ptypej);

		// turbulence production rate: Pi
		if(e_turbi==0){
			//Pi=C_mu*k_turbi*k_turbi/(e_turbi+1e-10)*tSR;
			tPi=0;
		}else{
			tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
			tPi2=0.3*k_turbi*sqrtf(tSR);
			tPi=fmin(tPi,tPi2);
		}

		// calculate strain rate
		if(tdist>0){
			tmpx+=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*(xi-xj)*tdwx;
			tmpy+=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*(yi-yj)*tdwy;
			tmpz+=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*(zi-zj)*tdwz;
		}
	}
	// save values to particle array
	Pa12[i].dk_turb=tPi-e_turbi+(tmpx+tmpy+tmpz);
	Pa12[i].e_turb=powf(C_mu,0.75)*powf(k_turbi,1.5)/Lm;		// (by esk)xxx => fast math
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_turbulence_viscous_force(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachex[256];
	 __shared__ Real cachey[256];
	 __shared__ Real cachez[256];

	cachex[threadIdx.x]=0;
	cachey[threadIdx.x]=0;
	cachez[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;

	uint_t ptypei,ptypej;
	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real tempi,tempj;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real C_v;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		tdist=Pa2[tid].dist;
		if(tdist>0){
			j=Pa2[tid].pnb;
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;

			rhoi=Pa11[i].rho;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			tempi=Pa11[i].temp;
			ptypei=Pa11[i].p_type;

			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;
			tempj=Pa11[j].temp;
			ptypej=Pa11[j].p_type;

			visj=viscosity(tempj,ptypej)+Pa12[j].vis_t;
			visi=viscosity(tempi,ptypei)+Pa12[i].vis_t;

			C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
			cachex[cache_idx]=C_v*(uxi-uxj);
			cachey[cache_idx]=C_v*(uyi-uyj);
			cachez[cache_idx]=C_v*(uzi-uzj);
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachex[cache_idx]+=cachex[cache_idx+s];
			cachey[cache_idx]+=cachey[cache_idx+s];
			cachez[cache_idx]+=cachez[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0){
		Pa11[i].ftotalx+=cachex[0];
		Pa11[i].ftotaly+=cachey[0];
		Pa11[i].ftotalz+=cachez[0];
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;
	uint_t ptypei,ptypej;

	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real tempi,tempj;

	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real C_v;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	tempi=Pa11[i].temp;
	ptypei=Pa11[i].p_type;
	visi=viscosity(tempi,ptypei)+Pa12[i].vis_t;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		tdist=Pa2[tid].dist;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;
		tempj=Pa11[j].temp;
		ptypej=Pa11[j].p_type;

		visj=viscosity(tempj,ptypej)+Pa12[j].vis_t;
		if(tdist>0){
			//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
			C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;

			tmpx+=C_v*(uxi-uxj);
			tmpy+=C_v*(uyi-uyj);
			tmpz+=C_v*(uzi-uzj);
		}
	}
	// save values
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_ke_turb(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachek[256];
	 __shared__ Real cachee[256];

	cachek[threadIdx.x]=0;
	cachee[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t ptypei,ptypej;
	Real mj,rhoj,visi,visj;
	Real tempi,tempj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwx,tdwy,tdwz;
	Real tdist;
	Real tPi,tPi2,tSR;
	Real vis_ti,vis_tj;
	Real k_turbi,k_turbj,e_turbi,e_turbj;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		// dist
		tdist=Pa2[tid].dist;
		// calculate strain rate
		if(tdist>0){
			j=Pa2[tid].pnb;
			// kernel (by esk)???
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;

			ptypei=Pa11[i].p_type;
			// density
			tempi=Pa11[i].temp;
			// position
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			// strain rate: 2S:S
			tSR=Pa12[i].SR;
			// turbulence viscosity
			vis_ti=Pa12[i].vis_t;
			// turbulence kinetic energy
			k_turbi=Pa12[i].k_turb;
			// turbulence dissipation
			e_turbi=Pa12[i].e_turb;

			ptypej=Pa11[j].p_type;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			tempj=Pa11[j].temp;
			// position
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			// turbulence viscosity
			vis_tj=Pa12[j].vis_t;
			// turbulence kinetic energy
			k_turbj=Pa12[j].k_turb;
			// turbulence dissipation
			e_turbj=Pa12[j].e_turb;

			// viscosity
			visi=viscosity(tempi,ptypei);
			visj=viscosity(tempj,ptypej);

			if(e_turbi==0){
				tPi=0;
			}else{
				tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
				tPi2=0.3*k_turbi*sqrtf(tSR);
				tPi=fmin(tPi,tPi2);
			}

			cachek[cache_idx]=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
			cachee[cache_idx]=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_e)*(e_turbi-e_turbj)/(tdist*tdist+0.0000001)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachek[cache_idx]+=cachek[cache_idx+s];
			cachee[cache_idx]+=cachee[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0){
		Pa12[i].dk_turb=tPi-e_turbi+cachek[0];
		Pa12[i].de_turb=e_turbi/(k_turbi+1e-10)*(C_e1*tPi-C_e2*e_turbi)+cachee[0];
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;
	uint_t ptypei,ptypej;

	//Real rhoi,uij2;
	Real mj,rhoj,visi,visj;
	Real tempi,tempj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwx,tdwy,tdwz;
	Real tdist;

	Real tPi,tPi2,tSR;
	Real vis_ti,vis_tj;
	Real k_turbi,k_turbj,e_turbi,e_turbj;

	Real tmpk,tmpe;

	non=Pa11[i].number_of_neighbors;
	ptypei=Pa11[i].p_type;
	// density
	//rhoi=Pa11[i].rho;
	tempi=Pa11[i].temp;
	// position
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	// strain rate: 2S:S
	tSR=Pa12[i].SR;
	// turbulence viscosity
	vis_ti=Pa12[i].vis_t;
	// turbulence kinetic energy
	k_turbi=Pa12[i].k_turb;
	// turbulence dissipation
	e_turbi=Pa12[i].e_turb;

	// viscosity
	visi=viscosity(tempi,ptypei);

	tmpk=tmpe=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel (by esk)???
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// dist
		tdist=Pa2[tid].dist;

		ptypej=Pa11[j].p_type;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		tempj=Pa11[j].temp;
		// position
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;
		// turbulence viscosity
		vis_tj=Pa12[j].vis_t;
		// turbulence kinetic energy
		k_turbj=Pa12[j].k_turb;
		// turbulence dissipation
		e_turbj=Pa12[j].e_turb;

		// viscosity
		visj=viscosity(tempj,ptypej);

		if(e_turbi==0){
			//Pi=C_mu*k_turbi*k_turbi/(e_turbi+1e-10)*tSR;
			tPi=0;
		}else{
			tPi=C_mu*k_turbi*k_turbi/(e_turbi)*tSR;
			tPi2=0.3*k_turbi*sqrtf(tSR);
			tPi=fmin(tPi,tPi2);
		}
		// calculate strain rate
		if(tdist>0){
			tmpk+=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_k)*(k_turbi-k_turbj)/(tdist*tdist+0.0000001)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
			tmpe+=mj/rhoj*(visi+visj+(vis_ti+vis_tj)/sigma_e)*(e_turbi-e_turbj)/(tdist*tdist+0.0000001)*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz);
		}
	}
	// save values to particle array
	Pa12[i].dk_turb=tPi-e_turbi+tmpk;
	Pa12[i].de_turb=e_turbi/(k_turbi+1e-10)*(C_e1*tPi-C_e2*e_turbi)+tmpe;
	//Pa12[i].e_turb=powf(C_mu,0.75)*powf(k_turbi,1.5)/Lm;		// (by esk)xxx => fast math
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_update_turbulence(int_t nop,Real tdt,part11*Pa11,part12*Pa12)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t p_type_i=Pa11[i].p_type;

	Pa12[i].k_turb+=Pa12[i].dk_turb*tdt*(p_type_i>0);
	Pa12[i].e_turb+=Pa12[i].de_turb*tdt*(p_type_i>0);

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_SPS_strain_tensor(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachexx[256];
	 __shared__ Real cachexy[256];
	 __shared__ Real cachexz[256];
	 __shared__ Real cacheyy[256];
	 __shared__ Real cacheyz[256];
	 __shared__ Real cachezz[256];

	cachexx[threadIdx.x]=0;
	cachexy[threadIdx.x]=0;
	cachexz[threadIdx.x]=0;
	cacheyy[threadIdx.x]=0;
	cacheyz[threadIdx.x]=0;
	cachezz[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	Real S;
	Real mj,rhoj;
	Real uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz;
	Real tdist,th;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		// dist
		tdist=Pa2[tid].dist;
		// calculate strain rate
		// check please (by esk)
		if(tdist>0){
			j=Pa2[tid].pnb;
			// kernel (by esk)???
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;
			th=Pa11[i].h*L_SPS;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;

			cachexx[cache_idx]=-(mj/rhoj)*(uxj)*tdwx;
			cachexy[cache_idx]=-0.5*(mj/rhoj)*(uxj*tdwy+uyj*tdwx);
			cachexz[cache_idx]=-0.5*(mj/rhoj)*(uxj*tdwz+uzj*tdwx);
			cacheyy[cache_idx]=-(mj/rhoj)*(uyj)*tdwy;
			cacheyz[cache_idx]=-0.5*(mj/rhoj)*(uyj*tdwz+uzj*tdwy);
			cachezz[cache_idx]=-(mj/rhoj)*(uzj)*tdwz;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachexx[cache_idx]+=cachexx[cache_idx+s];
			cachexy[cache_idx]+=cachexy[cache_idx+s];
			cachexz[cache_idx]+=cachexz[cache_idx+s];
			cacheyy[cache_idx]+=cacheyy[cache_idx+s];
			cacheyz[cache_idx]+=cacheyz[cache_idx+s];
			cachezz[cache_idx]+=cachezz[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0){
		Pa12[i].Sxx=cachexx[0];
		Pa12[i].Sxy=cachexy[0];
		Pa12[i].Sxz=cachexz[0];
		Pa12[i].Syy=cacheyy[0];
		Pa12[i].Syz=cacheyz[0];
		Pa12[i].Szz=cachezz[0];
		S=sqrtf((2*cachexx[0]*cachexx[0]+4*cachexy[0]*cachexy[0]+4*cachexz[0]*cachexz[0]
						+2*cacheyy[0]*cacheyy[0]+4*cacheyz[0]*cacheyz[0]+2*cachezz[0]*cachezz[0]));
		Pa12[i].vis_t=(Cs_SPS*th)*(Cs_SPS*th)*S;
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real S;
	Real mj,rhoj;
	//Real xi,yi,zi,xj,yj,zj,uxi,uyi,uzi,rhoi;
	Real uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz;
	Real tdist,th;
	Real tmpxx,tmpxy,tmpxz;
	Real tmpyy,tmpyz,tmpzz;

	non=Pa11[i].number_of_neighbors;
	// density
	//rhoi=Pa11[i].rho;
	// position
	//xi=Pa11[i].x;
	//yi=Pa11[i].y;
	//zi=Pa11[i].z;
	// velocity
	//uxi=Pa11[i].ux;
	//uyi=Pa11[i].uy;
	//uzi=Pa11[i].uz;
	// kernel distance
	th=Pa11[i].h*L_SPS;

	tmpxx=tmpxy=tmpxz=0.0;
	tmpyy=tmpyz=tmpzz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel (by esk)???
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// dist
		tdist=Pa2[tid].dist;

		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// position
		//xj=Pa11[j].x;
		//yj=Pa11[j].y;
		//zj=Pa11[j].z;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;

		// calculate strain rate
		// check please (by esk)
		if(tdist>0){
			tmpxx+=-(mj/rhoj)*(uxj)*tdwx;
			tmpxy+=-0.5*(mj/rhoj)*(uxj*tdwy+uyj*tdwx);
			tmpxz+=-0.5*(mj/rhoj)*(uxj*tdwz+uzj*tdwx);
			tmpyy+=-(mj/rhoj)*(uyj)*tdwy;
			tmpyz+=-0.5*(mj/rhoj)*(uyj*tdwz+uzj*tdwy);
			tmpzz+=-(mj/rhoj)*(uzj)*tdwz;
		}
	}

	// save values to particle array
	Pa12[i].Sxx=tmpxx;
	Pa12[i].Sxy=tmpxy;
	Pa12[i].Sxz=tmpxz;
	Pa12[i].Syy=tmpyy;
	Pa12[i].Syz=tmpyz;
	Pa12[i].Szz=tmpzz;

	S=sqrtf((2*tmpxx*tmpxx+4*tmpxy*tmpxy+4*tmpxz*tmpxz+2*tmpyy*tmpyy+4*tmpyz*tmpyz+2*tmpzz*tmpzz));
	Pa12[i].vis_t=(Cs_SPS*th)*(Cs_SPS*th)*S;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_SPS_stress_tensor(int_t nop,part11*Pa11,part12*Pa12)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tvis_t;
	Real trho,th;

	Real tSxx,tSxy,tSxz,tSyy,tSyz,tSzz; //tSzx,tSzy,tSyx
	Real tau_xx,tau_xy,tau_xz,tau_yy,tau_yz,tau_zz;

	trho=Pa11[i].rho;
	th=Pa11[i].h*L_SPS;

	tvis_t=Pa12[i].vis_t;
	tSxx=Pa12[i].Sxx;
	tSxy=Pa12[i].Sxy;
	tSxz=Pa12[i].Sxz;
	//tSyx=tSxy;
	tSyy=Pa12[i].Syy;
	tSyz=Pa12[i].Syz;
	//tSzx=tSxz;
	//tSzy=tSyz;
	tSzz=Pa12[i].Szz;

	// please check equations (by esk)!!!
	tau_xx=trho*(2*tvis_t*tSxx-2/3*(tSxx+tSyy+tSzz))-2/3*trho*CI_SPS*th*th;
	tau_xy=trho*(2*tvis_t*tSxy);
	tau_xz=trho*(2*tvis_t*tSxz);
	tau_yy=trho*(2*tvis_t*tSyy-2/3*(tSxx+tSyy+tSzz))-2/3*trho*CI_SPS*th*th;
	tau_yz=trho*(2*tvis_t*tSyz);
	tau_zz=trho*(2*tvis_t*tSzz-2/3*(tSxx+tSyy+tSzz))-2/3*trho*CI_SPS*th*th;
	/*
	tau_xx=0; tau_xy=0; tau_xz=0;
	tau_yy=0; tau_yz=0; tau_zz=0;
	//*/
	Pa12[i].Sxx=tau_xx;
	Pa12[i].Sxy=tau_xy;
	Pa12[i].Sxz=tau_xz;
	Pa12[i].Syy=tau_yy;
	Pa12[i].Syz=tau_yz;
	Pa12[i].Szz=tau_zz;
}
////////////////////////////////////////////////////////////////////////
/*
// viscous force calculation
__global__ void KERNEL_clc_SPS_viscous_force(Real *fvx,Real *fvy,Real *fvz,Real *Sxx_,Real *Sxy_,Real *Sxz_,Real *Syy_,Real *Syz_,Real *Szz_,uint_t *number_of_neighbors_,uint_t *pnb_,Real *m_,Real *rho_,Real *x_,Real *y_,Real *z_,Real *ux_,Real *uy_,Real *uz_,Real *temp_,Real *dwx_,Real *dwy_,Real *dwz_,Real *dist_,int_t pnb_size,const int_t dim,uint_t *p_type_)
{
	__shared__ Real cache_x[1000];
	__shared__ Real cache_y[1000];
	__shared__ Real cache_z[1000];

	cache_x[threadIdx.x]=0;
	cache_y[threadIdx.x]=0;
	cache_z[threadIdx.x]=0;

	//uint_t i=blockIdx.x+blockIdx.y*gridDim.x;
	uint_t i=blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{

	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real tempi,tempj;
	uint_t ptypei,ptypej;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real dwx,dwy,dwz,dist;
	Real C_v;

	uint_t number_of_neighbors;
	uint_t tid=threadIdx.x+blockIdx.x*pnb_size;
	uint_t j;

	Real tau_xx_i,tau_xy_i,tau_xz_i,tau_yx_i,tau_yy_i,tau_yz_i,tau_zx_i,tau_zy_i,tau_zz_i;
	Real tau_xx_j,tau_xy_j,tau_xz_j,tau_yx_j,tau_yy_j,tau_yz_j,tau_zx_j,tau_zy_j,tau_zz_j;

	number_of_neighbors=number_of_neighbors_[i];

	int_t cache_idx=threadIdx.x;

	// calculate viscous force element from particle i and j
	if (cache_idx < number_of_neighbors)
	{
		j=pnb_[tid];

		mj=m_[j];

		rhoi=rho_[i];
		rhoj=rho_[j];

		xi=x_[i];
		yi=y_[i];
		zi=z_[i];

		xj=x_[j];
		yj=y_[j];
		zj=z_[j];

		uxi=ux_[i];
		uyi=uy_[i];
		uzi=uz_[i];

		uxj=ux_[j];
		uyj=uy_[j];
		uzj=uz_[j];

		tempi=temp_[i];
		tempj=temp_[j];

		ptypei=p_type_[i];
		ptypej=p_type_[j];

		visi=viscosity(tempi,ptypei);
		visj=viscosity(tempj,ptypej);

		tau_xx_i=Sxx_[i];
		tau_xy_i=Sxy_[i];
		tau_xz_i=Sxz_[i];

		tau_yx_i=tau_xy_i;
		tau_yy_i=Syy_[i];
		tau_yz_i=Syz_[i];

		tau_zx_i=tau_xz_i;
		tau_zy_i=tau_yz_i;
		tau_zz_i=Szz_[i];

		tau_xx_j=Sxx_[j];
		tau_xy_j=Sxy_[j];
		tau_xz_j=Sxz_[j];

		tau_yx_j=tau_xy_j;
		tau_yy_j=Syy_[j];
		tau_yz_j=Syz_[j];

		tau_zx_j=tau_xz_j;
		tau_zy_j=tau_yz_j;
		tau_zz_j=Szz_[j];

		dwx=dwx_[tid];
		dwy=dwy_[tid];
		dwz=dwz_[tid];

		dist=dist_[tid];

		if (dist>0)
		{
			//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
			C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*dwx+(yi-yj)*dwy+(zi-zj)*dwz)/dist/dist;

			cache_x[cache_idx]=C_v*(uxi-uxj)+mj*((tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*dwx)+(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*dwy)+(tau_xz_i/(rhoi*rhoi)+tau_xz_j/(rhoj*rhoj)*dwz));
			cache_y[cache_idx]=C_v*(uyi-uyj)+mj*((tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*dwx)+(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*dwy)+(tau_yz_i/(rhoi*rhoi)+tau_yz_j/(rhoj*rhoj)*dwz));
			cache_z[cache_idx]=C_v*(uzi-uzj)+mj*((tau_zx_i/(rhoi*rhoi)+tau_zx_j/(rhoj*rhoj)*dwx)+(tau_zy_i/(rhoi*rhoi)+tau_zy_j/(rhoj*rhoj)*dwy)+(tau_zz_i/(rhoi*rhoi)+tau_zz_j/(rhoj*rhoj)*dwz));
		}
		else
		{
			cache_x[cache_idx]=0;
			cache_y[cache_idx]=0;
			cache_z[cache_idx]=0;
		}
	}
	else
	{
		cache_x[cache_idx]=0;
		cache_y[cache_idx]=0;
		cache_z[cache_idx]=0;
	}

	__syncthreads();

	// reduction
	for (uint_t s=blockDim.x/2; s>0; s >>= 1)
	{
		if (cache_idx < s)
		{
			cache_x[cache_idx] += cache_x[cache_idx+s];
			cache_y[cache_idx] += cache_y[cache_idx+s];
			cache_z[cache_idx] += cache_z[cache_idx+s];
		}

		__syncthreads();
	}


	// save values
	if (cache_idx==0)
	{
		fvx[i]=cache_x[0];
		fvy[i]=cache_y[0];
		fvz[i]=cache_z[0];
	}

	//}

}
//*/
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_SPS_viscous_force(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
{
	 __shared__ Real cachex[256];
	 __shared__ Real cachey[256];
	 __shared__ Real cachez[256];

	cachex[threadIdx.x]=0;
	cachey[threadIdx.x]=0;
	cachez[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;

	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real tempi,tempj;
	uint_t ptypei,ptypej;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real C_v;
	Real tau_xx_i,tau_xy_i,tau_xz_i,tau_yx_i,tau_yy_i,tau_yz_i,tau_zx_i,tau_zy_i,tau_zz_i;
	Real tau_xx_j,tau_xy_j,tau_xz_j,tau_yx_j,tau_yy_j,tau_yz_j,tau_zx_j,tau_zy_j,tau_zz_j;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		tdist=Pa2[tid].dist;
		if(tdist>0){
			j=Pa2[tid].pnb;
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;

			rhoi=Pa11[i].rho;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			tempi=Pa11[i].temp;
			ptypei=Pa11[i].p_type;
			tau_xx_i=Pa12[i].Sxx;
			tau_xy_i=Pa12[i].Sxy;
			tau_xz_i=Pa12[i].Sxz;
			tau_yx_i=tau_xy_i;
			tau_yy_i=Pa12[i].Syy;
			tau_yz_i=Pa12[i].Syz;
			tau_zx_i=tau_xz_i;
			tau_zy_i=tau_yz_i;
			tau_zz_i=Pa12[i].Szz;

			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;
			tempj=Pa11[j].temp;
			ptypej=Pa11[j].p_type;
			tau_xx_j=Pa12[j].Sxx;
			tau_xy_j=Pa12[j].Sxy;
			tau_xz_j=Pa12[j].Sxz;
			tau_yx_j=tau_xy_j;
			tau_yy_j=Pa12[j].Syy;
			tau_yz_j=Pa12[j].Syz;
			tau_zx_j=tau_xz_j;
			tau_zy_j=tau_yz_j;
			tau_zz_j=Pa12[j].Szz;

			visi=viscosity(tempi,ptypei);
			visj=viscosity(tempj,ptypej);

			C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
			cachex[cache_idx]=C_v*(uxi-uxj)+mj*((tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx)+(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy)+(tau_xz_i/(rhoi*rhoi)+tau_xz_j/(rhoj*rhoj)*tdwz));
			cachey[cache_idx]=C_v*(uyi-uyj)+mj*((tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx)+(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy)+(tau_yz_i/(rhoi*rhoi)+tau_yz_j/(rhoj*rhoj)*tdwz));
			cachez[cache_idx]=C_v*(uzi-uzj)+mj*((tau_zx_i/(rhoi*rhoi)+tau_zx_j/(rhoj*rhoj)*tdwx)+(tau_zy_i/(rhoi*rhoi)+tau_zy_j/(rhoj*rhoj)*tdwy)+(tau_zz_i/(rhoi*rhoi)+tau_zz_j/(rhoj*rhoj)*tdwz));
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachex[cache_idx]+=cachex[cache_idx+s];
			cachey[cache_idx]+=cachey[cache_idx+s];
			cachez[cache_idx]+=cachez[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0){
		Pa11[i].ftotalx+=cachex[0];
		Pa11[i].ftotaly+=cachey[0];
		Pa11[i].ftotalz+=cachez[0];
	}
	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj;
	Real visi,visj;
	Real tempi,tempj;
	uint_t ptypei,ptypej;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real C_v;
	Real tmpx,tmpy,tmpz;

	Real tau_xx_i,tau_xy_i,tau_xz_i,tau_yx_i,tau_yy_i,tau_yz_i,tau_zx_i,tau_zy_i,tau_zz_i;
	Real tau_xx_j,tau_xy_j,tau_xz_j,tau_yx_j,tau_yy_j,tau_yz_j,tau_zx_j,tau_zy_j,tau_zz_j;

	non=Pa11[i].number_of_neighbors;

	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	tempi=Pa11[i].temp;
	ptypei=Pa11[i].p_type;

	tau_xx_i=Pa12[i].Sxx;
	tau_xy_i=Pa12[i].Sxy;
	tau_xz_i=Pa12[i].Sxz;
	tau_yx_i=tau_xy_i;
	tau_yy_i=Pa12[i].Syy;
	tau_yz_i=Pa12[i].Syz;
	tau_zx_i=tau_xz_i;
	tau_zy_i=tau_yz_i;
	tau_zz_i=Pa12[i].Szz;

	visi=viscosity(tempi,ptypei);
	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		tdist=Pa2[tid].dist;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;
		tempj=Pa11[j].temp;
		ptypej=Pa11[j].p_type;
		tau_xx_j=Pa12[j].Sxx;
		tau_xy_j=Pa12[j].Sxy;
		tau_xz_j=Pa12[j].Sxz;
		tau_yx_j=tau_xy_j;
		tau_yy_j=Pa12[j].Syy;
		tau_yz_j=Pa12[j].Syz;
		tau_zx_j=tau_xz_j;
		tau_zy_j=tau_yz_j;
		tau_zz_j=Pa12[j].Szz;

		visj=viscosity(tempj,ptypej);

		if(tdist>0){
			//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
			C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;
			tmpx+=C_v*(uxi-uxj)+mj*((tau_xx_i/(rhoi*rhoi)+tau_xx_j/(rhoj*rhoj)*tdwx)+(tau_xy_i/(rhoi*rhoi)+tau_xy_j/(rhoj*rhoj)*tdwy)+(tau_xz_i/(rhoi*rhoi)+tau_xz_j/(rhoj*rhoj)*tdwz));
			tmpy+=C_v*(uyi-uyj)+mj*((tau_yx_i/(rhoi*rhoi)+tau_yx_j/(rhoj*rhoj)*tdwx)+(tau_yy_i/(rhoi*rhoi)+tau_yy_j/(rhoj*rhoj)*tdwy)+(tau_yz_i/(rhoi*rhoi)+tau_yz_j/(rhoj*rhoj)*tdwz));
			tmpz+=C_v*(uzi-uzj)+mj*((tau_zx_i/(rhoi*rhoi)+tau_zx_j/(rhoj*rhoj)*tdwx)+(tau_zy_i/(rhoi*rhoi)+tau_zy_j/(rhoj*rhoj)*tdwy)+(tau_zz_i/(rhoi*rhoi)+tau_zz_j/(rhoj*rhoj)*tdwz));
		}
	}
	// save values
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
	//*/
}
