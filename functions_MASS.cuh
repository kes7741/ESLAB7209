////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	__shared__ Real cache[256];
	cache[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	Real mj,twij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		mj=Pa11[j].m;
		cache[cache_idx]=mj*twij;
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].rho=cache[0]/Pa11[i].flt_s;

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,twij,tmp_Result;
	non=Pa11[i].number_of_neighbors;
	tmp_Result=0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		mj=Pa11[j].m;
		tmp_Result+=mj*twij;
	}
	Pa11[i].rho=tmp_Result/Pa11[i].flt_s;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_mass_sum_norm(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	__shared__ Real cache[256];
	cache[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	Real mj,twij;
	Real rho_ref_i,rho_ref_j;

	non=Pa11[i].number_of_neighbors;
	rho_ref_i=Pa11[i].rho_ref;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		mj=Pa11[j].m;
		rho_ref_j=Pa11[j].rho_ref;
		cache[cache_idx]=(mj/rho_ref_j)*twij;
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].rho=rho_ref_i*cache[0]/Pa11[i].flt_s;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_reference_density(int_t nop,int_t*k_vii,part11*Pa11,part12*Pa12)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//Real trhoA;
	uint_t tp_type;
	Real ttemp;
	Real m,h,stoh;
	Real tconcn;
	int d=k_vii[1];

	tp_type=Pa11[i].p_type;
	m=Pa11[i].m;
	h=Pa11[i].h;
	stoh=Pa11[i].stoh;
	ttemp=Pa11[i].temp;
	tconcn=Pa12[i].concn;

	//Pa11[i].rho_ref=reference_density(tp_type,ttemp,tconcn);
	//Pa11[i].rho_ref=reference_density2(tp_type,ttemp,m,h,d);
	Pa11[i].rho_ref=reference_density3(tp_type,ttemp,m,h,stoh,d);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_update_reference_mass(int_t nop,int_t*k_vii,part11*Pa11,part12*Pa12)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//Real trhoA;
	uint_t tp_type;
	Real ttemp;
	Real m,h,stoh;
	Real tconcn;
	Real s,vol;				// space(s) and volume(vol)
	int d=k_vii[1];		// dimension

	tp_type=Pa11[i].p_type;
	m=Pa11[i].m;
	h=Pa11[i].h;
	stoh=Pa11[i].stoh;
	ttemp=Pa11[i].temp;
	tconcn=Pa12[i].concn;

	if (tp_type == CORIUM)
	{
		s=h/stoh;
		vol=pow(s,d);
		m=(rho0A*vol*tconcn)+(rho0B*vol*(1-tconcn));
	}

	if (tp_type == SALT_WATER)
	{
		s=h/stoh;
		vol=pow(s,d);
		m=(1100*vol*tconcn)+(1000*vol*(1-tconcn));
	}

	//Pa11[i].rho_ref=reference_density(tp_type,ttemp,tconcn);
	Pa11[i].m=m;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_density_renormalization(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	__shared__ Real cachen[256];
	__shared__ Real cached[256];

	cachen[threadIdx.x]=0;
	cached[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t p_type_i,p_type_j;
	Real mj,rhoj,twij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		p_type_i=Pa11[i].p_type;
		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;

		cachen[cache_idx]=mj*twij*(p_type_i==p_type_j);
		cached[cache_idx]=mj*twij*(p_type_i==p_type_j)/rhoj;
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachen[cache_idx]+=cachen[cache_idx+s];
			cached[cache_idx]+=cached[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].rho=cachen[0]/cached[0];

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	uint_t p_type_i,p_type_j;
	Real mj,rhoj,twij;
	Real tmpn,tmpd;

	non=Pa11[i].number_of_neighbors;
	p_type_i=Pa11[i].p_type;
	tmpn=tmpd=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;

		tmpn+=mj*twij*(p_type_i==p_type_j);
		tmpd+=mj*twij*(p_type_i==p_type_j)/rhoj;
	}
	// save values to particle array
	Pa11[i].rho=tmpn/tmpd;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_density_renormalization_norm(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	__shared__ Real cachen[256];
	__shared__ Real cached[256];

	cachen[threadIdx.x]=0;
	cached[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	//uint_t p_type_i,p_type_j;
	Real mj,rhoj,twij;
	Real rho_ref_i,rho_ref_j;

	non=Pa11[i].number_of_neighbors;
	rho_ref_i=Pa11[i].rho_ref;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		//p_type_i=Pa11[i].p_type;
		//p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		rho_ref_j=Pa11[j].rho_ref;

		cachen[cache_idx]=rho_ref_i*(mj/rho_ref_j)*twij;
		cached[cache_idx]=(mj/rhoj)*twij;
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachen[cache_idx]+=cachen[cache_idx+s];
			cached[cache_idx]+=cached[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].rho=cachen[0]/cached[0];

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	uint_t p_type_i,p_type_j;
	Real mj,rhoj,twij;
	Real tmpn,tmpd;

	non=Pa11[i].number_of_neighbors;
	p_type_i=Pa11[i].p_type;
	tmpn=tmpd=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;

		tmpn+=mj*twij*(p_type_i==p_type_j);
		tmpd+=mj*twij*(p_type_i==p_type_j)/rhoj;
	}
	// save values to particle array
	Pa11[i].rho=tmpn/tmpd;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
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
	Real mj,rhoi,rhoj,tmprho;
	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		// kernel
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;

		// velocity
		uxi=Pa11[i].ux;
		uyi=Pa11[i].uy;
		uzi=Pa11[i].uz;
		rhoi=Pa11[i].rho;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;
		// calculate rho increment
		tmprho=rhoi/rhoj;
		cachex[cache_idx]=(uxi-uxj)*mj*(tmprho)*tdwx;
		cachey[cache_idx]=(uyi-uyj)*mj*(tmprho)*tdwy;
		cachez[cache_idx]=(uzi-uzj)*mj*(tmprho)*tdwz;
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
	if(cache_idx==0) Pa11[i].drho=cachex[0]+cachey[0]+cachez[0];

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj,tmprho;

	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	// velocity
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	rhoi=Pa11[i].rho;
	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;
		// calculate rho increment
		tmprho=rhoi/rhoj;
		tmpx+=(uxi-uxj)*mj*(tmprho)*tdwx;
		tmpy+=(uyi-uyj)*mj*(tmprho)*tdwy;
		tmpz+=(uzi-uzj)*mj*(tmprho)*tdwz;
	}
	Pa11[i].drho=tmpx+tmpy+tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_continuity_norm(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
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
	Real mj,rhoi,rhoj,tmprho;
	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		// kernel
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;

		// velocity
		uxi=Pa11[i].ux;
		uyi=Pa11[i].uy;
		uzi=Pa11[i].uz;
		rhoi=Pa11[i].rho;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;
		// calculate rho increment
		tmprho=rhoi/rhoj;
		cachex[cache_idx]=(uxi-uxj)*mj*(tmprho)*tdwx;
		cachey[cache_idx]=(uyi-uyj)*mj*(tmprho)*tdwy;
		cachez[cache_idx]=(uzi-uzj)*mj*(tmprho)*tdwz;
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
	if(cache_idx==0) Pa11[i].drho=cachex[0]+cachey[0]+cachez[0] + (rhoi/Pa11[i].rho_ref)*DIFF_DENSITY*Pa12[i].dconcn;

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj,tmprho;

	Real uxi,uxj,uyi,uyj,uzi,uzj;
	Real tdwx,tdwy,tdwz;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	// velocity
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	rhoi=Pa11[i].rho;
	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// velocity
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;
		// calculate rho increment
		tmprho=rhoi/rhoj;
		tmpx+=(uxi-uxj)*mj*(tmprho)*tdwx;
		tmpy+=(uyi-uyj)*mj*(tmprho)*tdwy;
		tmpz+=(uzi-uzj)*mj*(tmprho)*tdwz;
	}
	Pa11[i].drho=tmpx+tmpy+tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_density_diffusion_molteni(int_t nop,int_t pnbs,Real tsoundspeed,part11*Pa11,part2*Pa2)
{
	__shared__ Real cache[256];
	cache[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t ptypei;
	Real mj,rhoi,rhoj;
	Real hi,ci;
	Real tdist,tdwij;
	Real phi_ij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		tdist=Pa2[tid].dist;
		// calculate rho increment
		if(tdist>0){
			// calculate contribution of j particle on density variation (drho)
			j=Pa2[tid].pnb;
			// kernel & distance
			tdwij=Pa2[tid].dwij;

			ptypei=Pa11[i].p_type;
			// kernel distance
			hi=Pa11[i].h;
			// density
			rhoi=Pa11[i].rho;
			non=Pa11[i].number_of_neighbors;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			phi_ij=rhoj-rhoi;
			// speed of sound
			ci=tsoundspeed;
			cache[cache_idx]=-2*delta*hi*ci*(mj/rhoj)*phi_ij*tdwij/tdist;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].rho=cache[0];

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;
	uint_t ptypei=Pa11[i].p_type;

	Real mj,rhoi,rhoj;
	Real hi,ci;
	//Real xi,xj,yi,yj,zi,zj,drhoi;
	Real tdist,tdwij;
	Real phi_ij,tmp_Result;

	// kernel distance
	hi=Pa11[i].h;
	// density
	rhoi=Pa11[i].rho;
	non=Pa11[i].number_of_neighbors;

	// speed of sound
	ci=soundspeed(ptypei);
	tmp_Result=0.0;
	// calculate contribution of j particle on density variation (drho)
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel & distance
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		phi_ij=rhoj-rhoi;
		// calculate rho increment
		if(tdist>0){
			tmp_Result+=-2*delta*hi*ci*(mj/rhoj)*phi_ij*tdwij/tdist;
		}
	}
	Pa11[i].drho=tmp_Result;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_grad_density(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
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
	Real tdw_cx,tdw_cy,tdw_cz;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		// calculate contribution of j particle on density variation (drho)
		j=Pa2[tid].pnb;
		// kernel & distance
		tdw_cx=Pa2[tid].dw_cx;
		tdw_cy=Pa2[tid].dw_cy;
		tdw_cz=Pa2[tid].dw_cz;

		rhoi=Pa11[i].rho;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// calculate rho increment
		cachex[cache_idx]=-(rhoj-rhoi)*(mj/rhoj)*tdw_cx;
		cachey[cache_idx]=-(rhoj-rhoi)*(mj/rhoj)*tdw_cy;
		cachez[cache_idx]=-(rhoj-rhoi)*(mj/rhoj)*tdw_cz;
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
		Pa11[i].grad_rhox+=cachex[0];
		Pa11[i].grad_rhoy+=cachey[0];
		Pa11[i].grad_rhoz+=cachez[0];
	}
	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj;
	Real tdw_cx,tdw_cy,tdw_cz;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	// density
	rhoi=Pa11[i].rho;
	tmpx=tmpy=tmpz=0.0;
	// calculate contribution of j particle on density variation (drho)
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel & distance
		tdw_cx=Pa2[tid].dw_cx;
		tdw_cy=Pa2[tid].dw_cy;
		tdw_cz=Pa2[tid].dw_cz;

		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// calculate rho increment
		tmpx+=-(rhoj-rhoi)*(mj/rhoj)*tdw_cx;
		tmpy+=-(rhoj-rhoi)*(mj/rhoj)*tdw_cy;
		tmpz+=-(rhoj-rhoi)*(mj/rhoj)*tdw_cz;
	}
	// save values to particle array
	Pa11[i].grad_rhox=tmpx;
	Pa11[i].grad_rhoy=tmpy;
	Pa11[i].grad_rhoz=tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_density_diffusion_antuono(int_t nop,int_t pnbs,Real tsoundspeed,part11*Pa11,part2*Pa2)
{
	__shared__ Real cache[256];
	cache[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t ptypei;

	Real mj,rhoi,rhoj;
	Real grad_rhoxi,grad_rhoxj,grad_rhoyi,grad_rhoyj,grad_rhozi,grad_rhozj;
	Real hi,ci;
	Real xi,xj,yi,yj,zi,zj;
	Real tdist,tdwij;
	Real phi_ij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		tdist=Pa2[tid].dist;
		// calculate rho increment
		if(tdist>0){
			// calculate contribution of j particle on density variation (drho)
			j=Pa2[tid].pnb;
			// kernel & distance
			tdwij=Pa2[tid].dwij;

			ptypei=Pa11[i].p_type;
			// kernel distance
			hi=Pa11[i].h;
			// density
			rhoi=Pa11[i].rho;
			// position
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			// normalized density gradient
			grad_rhoxi=Pa11[i].grad_rhox;
			grad_rhoyi=Pa11[i].grad_rhoy;
			grad_rhozi=Pa11[i].grad_rhoz;
			// position
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			// mass
			mj=Pa11[j].m;
			// density
			rhoj=Pa11[j].rho;
			// normalized density gradient
			grad_rhoxj=Pa11[j].grad_rhox;
			grad_rhoyj=Pa11[j].grad_rhoy;
			grad_rhozj=Pa11[j].grad_rhoz;

			// speed of sound
			ci=tsoundspeed;
			phi_ij=(rhoj-rhoi)-0.5*((grad_rhoxi+grad_rhoxj)*(xj-xi)+(grad_rhoyi+grad_rhoyj)*(yj-yi)+(grad_rhozi+grad_rhozj)*(zj-zi));
			cache[cache_idx]=-2*delta*hi*ci*(mj/rhoj)*phi_ij*tdwij/tdist;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].drho=cache[0];


	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;
	uint_t ptypei=Pa11[i].p_type;

	Real mj,rhoi,rhoj;
	Real grad_rhoxi,grad_rhoxj,grad_rhoyi,grad_rhoyj,grad_rhozi,grad_rhozj;
	Real hi,ci; //,drhoi
	Real xi,xj,yi,yj,zi,zj;
	Real tdist,tdwij;
	Real phi_ij,tmp_Result;

	non=Pa11[i].number_of_neighbors;
	// kernel distance
	hi=Pa11[i].h;
	// density
	rhoi=Pa11[i].rho;
	// position
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	// normalized density gradient
	grad_rhoxi=Pa11[i].grad_rhox;
	grad_rhoyi=Pa11[i].grad_rhoy;
	grad_rhozi=Pa11[i].grad_rhoz;

	// speed of sound
	ci=soundspeed(ptypei);
	tmp_Result=0.0;
	// calculate contribution of j particle on density variation (drho)
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		// kernel & distance
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;
		// position
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;
		// mass
		mj=Pa11[j].m;
		// density
		rhoj=Pa11[j].rho;
		// normalized density gradient
		grad_rhoxj=Pa11[j].grad_rhox;
		grad_rhoyj=Pa11[j].grad_rhoy;
		grad_rhozj=Pa11[j].grad_rhoz;

		phi_ij=(rhoj-rhoi)-0.5*((grad_rhoxi+grad_rhoxj)*(xj-xi)+(grad_rhoyi+grad_rhoyj)*(yj-yi)+(grad_rhozi+grad_rhozj)*(zj-zi));
		// calculate rho increment
		if(tdist>0){
			tmp_Result+=-2*delta*hi*ci*(mj/rhoj)*phi_ij*tdwij/tdist;
		}
	}
	// save values to particle array
	Pa11[i].drho=tmp_Result;
	//*/
}
