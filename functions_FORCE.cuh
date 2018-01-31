////////////////////////////////////////////////////////////////////////
// pressure forece calculation
__global__ void KERNEL_clc_pressure_force(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
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
	Real pi,pj;
	Real tdwx,tdwy,tdwz;
	Real C_p;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;

		rhoi=Pa11[i].rho;
		pi=Pa11[i].pres;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		pj=Pa11[j].pres;

		C_p=-mj*(pi+pj)/(rhoi*rhoj);

		cachex[cache_idx]=C_p*tdwx;
		cachey[cache_idx]=C_p*tdwy;
		cachez[cache_idx]=C_p*tdwz;
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
		Pa11[i].ftotalx=cachex[0];
		Pa11[i].ftotaly=cachey[0];
		Pa11[i].ftotalz=cachez[0];
	}
	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real mj,rhoi,rhoj;
	Real pi,pj;
	Real tdwx,tdwy,tdwz;
	Real C_p;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	rhoi=Pa11[i].rho;
	pi=Pa11[i].pres;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		pj=Pa11[j].pres;

		C_p=-mj*(pi+pj)/(rhoi*rhoj);

		tmpx+=C_p*tdwx;
		tmpy+=C_p*tdwy;
		tmpz+=C_p*tdwz;
	}
	// save values
	Pa11[i].ftotalx=tmpx;
	Pa11[i].ftotaly=tmpy;
	Pa11[i].ftotalz=tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
// viscous force calculation
__global__ void KERNEL_add_viscous_force(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
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
	Real ptypei,ptypej;
	Real tempi,tempj;
	Real visi,visj;
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

			ptypei=Pa11[i].p_type;
			rhoi=Pa11[i].rho;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			tempi=Pa11[i].temp;

			ptypej=Pa11[j].p_type;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;
			tempj=Pa11[j].temp;

			visi=viscosity(tempi,ptypei);
			visj=viscosity(tempj,ptypej);
			//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
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

	Real mj,rhoi,rhoj;
	Real ptypei,ptypej;
	Real tempi,tempj;
	Real visi,visj;
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

		visj=viscosity(tempj,ptypej);
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
// viscous force calculation
__global__ void KERNEL_add_HB_viscous_force(int_t nop,int_t pnbs,part11*Pa11,part12*Pa12,part2*Pa2)
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
	Real ptypei,ptypej;
	Real tempi,tempj;
	Real visi,visj;
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

			ptypei=Pa11[i].p_type;
			rhoi=Pa11[i].rho;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			tempi=Pa11[i].temp;

			ptypej=Pa11[j].p_type;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;
			tempj=Pa11[j].temp;

			visi=Pa12[i].vis_t;
			visj=Pa12[j].vis_t;
			//C_v=2*(mj/rhoj)*(visj/rhoj)*dwij/dist;
			C_v=4*(mj/(rhoi*rhoj))*((visi*visj)/(visi+visj+1e-20))*((xi-xj)*tdwx+(yi-yj)*tdwy+(zi-zj)*tdwz)/tdist/tdist;

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

	Real mj,rhoi,rhoj;
	Real ptypei,ptypej;
	Real tempi,tempj;
	Real visi,visj;
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

		visj=viscosity(tempj,ptypej);
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
__global__ void KERNEL_add_artificial_viscous_force(int_t nop,int_t pnbs,Real tsoundspeed,part11*Pa11,part2*Pa2)
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
	Real mi,rhoi,rhoj,rho_ij;
	Real hi,hj,h_ij;
	Real ci,cj,c_ij;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real uij_xij,phi_ij,P_ij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdist=Pa2[tid].dist;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwx;
		tdwz=Pa2[tid].dwz;

		if(tdist>0){
			ptypei=Pa11[i].p_type;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			uxi=Pa11[i].ux;
			uyi=Pa11[i].uy;
			uzi=Pa11[i].uz;
			mi=Pa11[i].m;
			rhoi=Pa11[i].rho;
			hi=Pa11[i].h;
			ptypej=Pa11[j].p_type;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;
			rhoj=Pa11[j].rho;
			hj=Pa11[j].h;

			ci=tsoundspeed;
			cj=tsoundspeed;

			rho_ij=(rhoi+rhoj)*0.5;
			h_ij=(hi+hj)*0.5;
			c_ij=(ci+cj)*0.5;

			uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
			phi_ij=h_ij*uij_xij/(tdist*tdist+0.01*h_ij*h_ij);
			if(uij_xij<0) P_ij=mi*(-Alpha*c_ij*phi_ij+Beta*phi_ij*phi_ij)/rho_ij;
			else P_ij=0;

			cachex[cache_idx]=-(P_ij)*tdwx;
			cachey[cache_idx]=-(P_ij)*tdwy;
			cachez[cache_idx]=-(P_ij)*tdwz;
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
	Real mi,rhoi,rhoj,rho_ij; //mj,
	Real hi,hj,h_ij;
	Real ci,cj,c_ij;
	Real xi,yi,zi,xj,yj,zj;
	Real uxi,uyi,uzi,uxj,uyj,uzj;
	Real tdwx,tdwy,tdwz,tdist;
	Real uij_xij,phi_ij,P_ij;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	ptypei=Pa11[i].p_type;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	hi=Pa11[i].h;

	ci=soundspeed(ptypei);

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwx;
			tdwz=Pa2[tid].dwz;

			ptypej=Pa11[j].p_type;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			uxj=Pa11[j].ux;
			uyj=Pa11[j].uy;
			uzj=Pa11[j].uz;
			//mj=Pa[j].m;
			rhoj=Pa11[j].rho;
			hj=Pa11[j].h;

			cj=soundspeed(ptypej);

			rho_ij=(rhoi+rhoj)*0.5;
			h_ij=(hi+hj)*0.5;
			c_ij=(ci+cj)*0.5;

			uij_xij=(uxi-uxj)*(xi-xj)+(uyi-uyj)*(yi-yj)+(uzi-uzj)*(zi-zj);
			phi_ij=h_ij*uij_xij/(tdist*tdist+0.01*h_ij*h_ij);
			if(uij_xij<0) P_ij=mi*(-Alpha*c_ij*phi_ij+Beta*phi_ij*phi_ij)/rho_ij;
			else P_ij=0;

			tmpx+=-(P_ij)*tdwx;
			tmpy+=-(P_ij)*tdwy;
			tmpz+=-(P_ij)*tdwz;
		}
	}
	// save values
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_gravity_force(int_t nop,const int_t tdim,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	switch(tdim){
		case 3:
			Pa11[i].ftotalz+=-Gravitational_CONST;				// z-directional gravitational force
			break;
		case 2:
			Pa11[i].ftotaly+=-Gravitational_CONST;				// y-directional gravitational force
			break;
		default:
			break;
	}
}
////////////////////////////////////////////////////////////////////////
// calcuate color field for two-phase flow surface tension model (2017.05.08 jyb)
__global__ void KERNEL_clc_color_field(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	__shared__ Real cachen[256];
	__shared__ Real cached[256];

	cachen[threadIdx.x]=0;
	cached[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	int_t ptypei;
	Real mj,rhoj,twij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		ptypei=Pa11[i].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		cachen[cache_idx]=mj*twij*(ptypei==Pa11[j].p_type)/rhoj;
		cached[cache_idx]=mj*twij/rhoj;
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
	if(cache_idx==0) Pa11[i].cc=cachen[0]/cached[0];

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;
	int_t ptypei;
	Real mj,rhoj,twij; //mi,rhoi
	Real tmpnum,tmpden;

	//mi=Pa11[i].m;
	//rhoi=Pa11[i].rho;
	non=Pa11[i].number_of_neighbors;
	ptypei=Pa11[i].p_type;

	tmpnum=tmpden=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;

		tmpnum+=mj*twij*(ptypei==Pa11[j].p_type)/rhoj;
		tmpden+=mj*twij/rhoj;
	}
	// save values
	Pa11[i].cc=tmpnum/tmpden;
	//*/
}
////////////////////////////////////////////////////////////////////////
// calcuate normal gradient vector for two-phase flow surface tension model (2017.04.20 jyb)
__global__ void KERNEL_clc_normal_gradient_c(int_t nop,int_t pnbs,part11*Pa11,part13*Pa13,part2*Pa2)
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
	int_t ptypei;
	Real mj,rhoj,cci,ccj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwij,tdist;
	Real C_s,tmpnmg;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			ptypei=Pa11[i].p_type;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			cci=Pa11[i].cc;

			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			ccj=Pa11[j].cc;

			//(fluid_1_i*fluid_1_j+fluid_2_i*fluid_2_j+boundary_i*boundary_j)
			C_s=(mj/rhoj)*(ccj-cci)*(ptypei==Pa11[j].p_type)*tdwij/tdist;

			cachex[cache_idx]=C_s*(xj-xi);
			cachey[cache_idx]=C_s*(yj-yi);
			cachez[cache_idx]=C_s*(zj-zi);
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
		Pa13[i].nx_c=cachex[0];
		Pa13[i].ny_c=cachey[0];
		Pa13[i].nz_c=cachez[0];
		tmpnmg=sqrt(cachex[0]*cachex[0]+cachey[0]*cachey[0]+cachez[0]*cachez[0]);
		Pa13[i].nmag_c=tmpnmg;
		if(tmpnmg<NORMAL_THRESHOLD){
			Pa13[i].nx_c=0;
			Pa13[i].ny_c=0;
			Pa13[i].nz_c=0;
			Pa13[i].nmag_c=1e-20;

		}
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	int_t ptypei;
	Real mj,rhoj,cci,ccj; //mi,rhoi
	Real xi,yi,zi,xj,yj,zj;
	Real tdwij,tdist;
	Real C_s;
	Real tmpnx,tmpny,tmpnz,tmpnmg;

	non=Pa11[i].number_of_neighbors;
	//mi=Pa11[i].m;
	//rhoi=Pa11[i].rho;
	cci=Pa11[i].cc;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	ptypei=Pa11[i].p_type;
	tmpnx=tmpny=tmpnz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		ccj=Pa11[j].cc;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		if(tdist>0){
			//(fluid_1_i*fluid_1_j+fluid_2_i*fluid_2_j+boundary_i*boundary_j)
			C_s=(mj/rhoj)*(ccj-cci)*(ptypei==Pa11[j].p_type)*tdwij/tdist;

			tmpnx+=C_s*(xj-xi);
			tmpny+=C_s*(yj-yi);
			tmpnz+=C_s*(zj-zi);
		}
	}
	// save values
	Pa13[i].nx_c=tmpnx;
	Pa13[i].ny_c=tmpny;
	Pa13[i].nz_c=tmpnz;
	tmpnmg=sqrt(tmpnx*tmpnx+tmpny*tmpny+tmpnz*tmpnz);
	Pa13[i].nmag_c=tmpnmg;
	if(tmpnmg<NORMAL_THRESHOLD){
		Pa13[i].nx_c=0;
		Pa13[i].ny_c=0;
		Pa13[i].nz_c=0;
		Pa13[i].nmag_c=1e-20;
	}
	//*/
}
////////////////////////////////////////////////////////////////////////
// calcuate normal gradient vector for two-phase flow surface tension model (2017.04.20 jyb)
__global__ void KERNEL_clc_normal_gradient(int_t nop,int_t pnbs,part11*Pa11,part13*Pa13,part2*Pa2)
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
	int_t ptypei;
	Real mi,mj,rhoi,rhoj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwij,tdist;
	Real C_s,tmpnmg;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			ptypei=Pa11[i].p_type;
			mi=Pa11[i].m;
			rhoi=Pa11[i].rho;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;

			C_s=(ptypei!=Pa11[j].p_type);

			cachex[cache_idx]=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(xj-xi)/tdist;
			cachey[cache_idx]=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(yj-yi)/tdist;
			cachez[cache_idx]=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(zj-zi)/tdist;
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
		Pa13[i].nx=cachex[0];
		Pa13[i].ny=cachey[0];
		Pa13[i].nz=cachez[0];
		tmpnmg=sqrt(cachex[0]*cachex[0]+cachey[0]*cachey[0]+cachez[0]*cachez[0]);
		Pa13[i].nmag=tmpnmg;

		if(tmpnmg<NORMAL_THRESHOLD){
			Pa13[i].nx=0;
			Pa13[i].ny=0;
			Pa13[i].nz=0;
			Pa13[i].nmag=1e-20;
		}
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	int_t ptypei;
	Real mi,mj,rhoi,rhoj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwij,tdist;
	Real C_s;
	Real tmpnx,tmpny,tmpnz,tmpnmg;

	non=Pa11[i].number_of_neighbors;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	ptypei=Pa11[i].p_type;

	tmpnx=tmpny=tmpnz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		if(tdist>0){
			C_s=(ptypei!=Pa11[j].p_type);

			tmpnx+=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(xj-xi)/tdist;
			tmpny+=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(yj-yi)/tdist;
			tmpnz+=C_s*((mi/rhoi)*(mi/rhoi)+(mj/rhoj)*(mj/rhoj))*(rhoi/(rhoi+rhoj))*(rhoi/mi)*tdwij*(zj-zi)/tdist;
		}
	}

	// save values
	Pa13[i].nx=tmpnx;
	Pa13[i].ny=tmpny;
	Pa13[i].nz=tmpnz;
	tmpnmg=sqrt(tmpnx*tmpnx+tmpny*tmpny+tmpnz*tmpnz);
	Pa13[i].nmag=tmpnmg;

	if(tmpnmg<NORMAL_THRESHOLD){
		Pa13[i].nx=0;
		Pa13[i].ny=0;
		Pa13[i].nz=0;
		Pa13[i].nmag=1e-20;
	}
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_surface_tension(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part13*Pa13,part2*Pa2)
{
	__shared__ Real cachen[256];
	__shared__ Real cached[256];

	cachen[threadIdx.x]=0;
	cached[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	int_t ptypei,ptypej;
	Real sigmai,mj,rhoi,rhoj,hi,tempi;
	Real xi,yi,zi,xj,yj,zj;
	Real nxi,nyi,nzi,nmagi;
	Real nx_ci,ny_ci,nz_ci,nx_cj,ny_cj,nz_cj,nmag_ci,nmag_cj,curvi;
	Real tdwij,tdist;
	Real Phi_s;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			ptypei=Pa11[i].p_type;
			tempi=Pa11[i].temp;
			rhoi=Pa11[i].rho;
			hi=Pa11[i].h;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			nxi=Pa13[i].nx;
			nyi=Pa13[i].ny;
			nzi=Pa13[i].nz;
			nmagi=Pa13[i].nmag;
			nx_ci=Pa13[i].nx_c;
			ny_ci=Pa13[i].ny_c;
			nz_ci=Pa13[i].nz_c;
			nmag_ci=Pa13[i].nmag_c;

			ptypej=Pa11[j].p_type;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			nx_cj=Pa13[j].nx_c;
			ny_cj=Pa13[j].ny_c;
			nz_cj=Pa13[j].nz_c;
			nmag_cj=Pa13[j].nmag_c;

			sigmai=sigma(tempi,ptypei);
			Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

			cachen[cache_idx]=tdim*(mj/rhoj)*(((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi)+((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi)+((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi))*tdwij/tdist;
			cached[cache_idx]=(mj/rhoj)*tdist*abs(tdwij);
		}
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
	if(cache_idx==0){
		if ((nmagi>0.1/hi)&(cachen[0]>0)) Pa13[i].curv=cachen[0]/cached[0];
		else Pa13[i].curv=0;

		curvi=Pa13[i].curv;
		Pa11[i].ftotalx+=sigmai*curvi*nxi/rhoi;
		Pa11[i].ftotaly+=sigmai*curvi*nyi/rhoi;
		Pa11[i].ftotalz+=sigmai*curvi*nzi/rhoi;
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	int_t ptypei,ptypej;
	Real sigmai,mj,rhoi,rhoj,hi,tempi; //mi,
	Real xi,yi,zi,xj,yj,zj;
	Real nxi,nyi,nzi,nmagi;
	Real nx_ci,ny_ci,nz_ci,nx_cj,ny_cj,nz_cj,nmag_ci,nmag_cj,curvi;
	Real tdwij,tdist;
	Real Phi_s;
	Real tmpnum,tmpden;

	non=Pa11[i].number_of_neighbors;
	ptypei=Pa11[i].p_type;
	tempi=Pa11[i].temp;
	//mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	hi=Pa11[i].h;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	nxi=Pa13[i].nx;
	nyi=Pa13[i].ny;
	nzi=Pa13[i].nz;
	nmagi=Pa13[i].nmag;
	nx_ci=Pa13[i].nx_c;
	ny_ci=Pa13[i].ny_c;
	nz_ci=Pa13[i].nz_c;
	nmag_ci=Pa13[i].nmag_c;

	sigmai=sigma(tempi,ptypei);

	tmpnum=tmpden=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		ptypej=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		nx_cj=Pa13[j].nx_c;
		ny_cj=Pa13[j].ny_c;
		nz_cj=Pa13[j].nz_c;
		nmag_cj=Pa13[j].nmag_c;

		if(tdist>0){
			// (fluid_1_i*fluid_1_j+fluid_2_i*fluid_2_j+boundary_i*boundary_j)
			Phi_s=-(ptypei!= ptypej)+(ptypei==ptypej);

			tmpnum+=tdim*(mj/rhoj)*(((nx_ci/nmag_ci)-Phi_s*(nx_cj/nmag_cj))*(xj-xi)+((ny_ci/nmag_ci)-Phi_s*(ny_cj/nmag_cj))*(yj-yi)+((nz_ci/nmag_ci)-Phi_s*(nz_cj/nmag_cj))*(zj-zi))*tdwij/tdist;
			tmpden+=(mj/rhoj)*tdist*abs(tdwij);
		}
	}
	// save values
	if ((nmagi>0.1/hi)&(tmpden>0)) Pa13[i].curv=tmpnum/tmpden;
	else Pa13[i].curv=0;

	curvi=Pa13[i].curv;

	Pa11[i].ftotalx+=sigmai*curvi*nxi/rhoi;
	Pa11[i].ftotaly+=sigmai*curvi*nyi/rhoi;
	Pa11[i].ftotalz+=sigmai*curvi*nzi/rhoi;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_interface_sharpness(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
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
	int_t ptypei,ptypej;
	Real mi,mj,rhoi,rhoj,pi,pj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwij,tdist;
	Real C_i;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			ptypei=Pa11[i].p_type;
			mi=Pa11[i].m;
			rhoi=Pa11[i].rho;
			pi=Pa11[i].pres;
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;

			ptypej=Pa11[j].p_type;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			pj=Pa11[j].pres;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;

			C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
			// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
			cachex[cache_idx]=C_i*(xj-xi);
			cachey[cache_idx]=C_i*(yj-yi);
			cachez[cache_idx]=C_i*(zj-zi);
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

	int_t ptypei,ptypej;
	Real mi,mj,rhoi,rhoj,pi,pj;
	Real xi,yi,zi,xj,yj,zj;
	Real tdwij,tdist;
	Real C_i;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	ptypei=Pa11[i].p_type;
	mi=Pa11[i].m;
	rhoi=Pa11[i].rho;
	pi=Pa11[i].pres;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		ptypej=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		pj=Pa11[j].pres;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		if(tdist>0){
			C_i=0.08/mi*(abs(pi)*(mi/rhoi)*(mi/rhoi)+abs(pj)*(mj/rhoj)*(mj/rhoj)*((ptypei!= BOUNDARY)&&(ptypei!=MOVING)&&(ptypej!=BOUNDARY)&&(ptypej!=MOVING)&&(ptypei!=ptypej)))*tdwij/tdist;
			// apply interface sharpness force just for the fluid particles (2017.06.22 jyb)
			tmpx+=C_i*(xj-xi);
			tmpy+=C_i*(yj-yi);
			tmpz+=C_i*(zj-zi);
		}
	}
	// save values
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_add_boundary_force(int_t nop,int_t pnbs,Real tC_repulsive,part11*Pa11,part2*Pa2)
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
	uint_t p_type_i,p_type_j;
	Real xi,yi,zi,xj,yj,zj;
	Real mi,mj;
	Real twij,tdist;
	Real fb_ij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		tdist=Pa2[tid].dist;

		p_type_i=Pa11[i].p_type;
		mi=Pa11[i].m;
		xi=Pa11[i].x;
		yi=Pa11[i].y;
		zi=Pa11[i].z;

		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		if((p_type_i==FLUID)&(p_type_j!=FLUID)){
			fb_ij=tC_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
			cachex[cache_idx]=fb_ij*(xi-xj);
			cachey[cache_idx]=fb_ij*(yi-yj);
			cachez[cache_idx]=fb_ij*(zi-zj);
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

	uint_t p_type_i,p_type_j;
	Real xi,yi,zi,xj,yj,zj;
	Real mi,mj;
	Real twij,tdist;
	Real fb_ij;
	Real tmpx,tmpy,tmpz;

	non=Pa11[i].number_of_neighbors;
	p_type_i=Pa11[i].p_type;
	mi=Pa11[i].m;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmpx=tmpy=tmpz=0.0;

	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		tdist=Pa2[tid].dist;

		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		if((p_type_i==FLUID)&(p_type_j!=FLUID)){
			fb_ij=tC_repulsive/(tdist+1e-10)/(tdist+1e-10)*twij*(2*mj/(mi+mj));
			tmpx+=fb_ij*(xi-xj);
			tmpy+=fb_ij*(yi-yj);
			tmpz+=fb_ij*(zi-zj);
		}
	}
	// save values
	Pa11[i].ftotalx+=tmpx;
	Pa11[i].ftotaly+=tmpy;
	Pa11[i].ftotalz+=tmpz;
	//*/
}
////////////////////////////////////////////////////////////////////////
// natural convection force (boussinesq approximation)
__global__ void KERNEL_add_boussinesq_force(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)
{
	__shared__ Real cachen[256];
	__shared__ Real cached[256];

	cachen[threadIdx.x]=0.0;
	cached[threadIdx.x]=1.0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t p_type_i,p_type_j;
	Real tempi,tempj,betai;			// beta : thermal expansion coefficient
	Real mj,rhoj,twij;
	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		p_type_i=Pa11[i].p_type;
		tempi=Pa11[i].temp;
		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		tempj=Pa11[j].temp;

		betai=thermal_expansion(tempi,p_type_i);

		if((p_type_i!=BOUNDARY)&(p_type_i!=MOVING)){
			cachen[cache_idx]=mj*(tempj-tempi)*twij*(p_type_i==p_type_j)/rhoj;
			cached[cache_idx]=mj*twij*(p_type_i==p_type_j)/rhoj;
		}
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
	if(cache_idx==0){
		switch(tdim){
			case 3:
				Pa11[i].ftotalz+=-betai*Gravitational_CONST*(cachen[0]/cached[0]);				// z-directional gravitational force
				break;
			case 2:
				Pa11[i].ftotaly+=-betai*Gravitational_CONST*(cachen[0]/cached[0]);				// y-directional gravitational force
				break;
			default:
				break;
		}
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	uint_t p_type_i,p_type_j;
	//Real xi,yi,zi,xj,yj,zj;
	Real tempi,tempj,betai;			// beta : thermal expansion coefficient
	Real mj,rhoj,twij;
	Real tmpnum,tmpden;

	non=Pa11[i].number_of_neighbors;
	p_type_i=Pa11[i].p_type;
	//xi=Pa11[i].x;
	//yi=Pa11[i].y;
	//zi=Pa11[i].z;
	tempi=Pa11[i].temp;
	betai=thermal_expansion(tempi,p_type_i);
	tmpnum=0.0;
	tmpden=1.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		p_type_j=Pa11[j].p_type;
		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		//xj=Pa11[j].x;
		//yj=Pa11[j].y;
		//zj=Pa11[j].z;
		tempj=Pa11[j].temp;

		if((p_type_i!=BOUNDARY)&(p_type_i!=MOVING)){
			tmpnum+=mj*(tempj-tempi)*twij*(p_type_i==p_type_j)/rhoj;
			tmpden+=mj*twij*(p_type_i==p_type_j)/rhoj;
		}
	}

	// save values
	switch(tdim){
		case 3:
			Pa11[i].ftotalz+=-betai*Gravitational_CONST*(tmpnum/tmpden);				// z-directional gravitational force
			break;
		case 2:
			Pa11[i].ftotaly+=-betai*Gravitational_CONST*(tmpnum/tmpden);				// y-directional gravitational force
			break;
		default:
			break;
	}
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_ftotal(int_t nop,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmpx,tmpy,tmpz;
	tmpx=Pa11[i].ftotalx;
	tmpy=Pa11[i].ftotaly;
	tmpz=Pa11[i].ftotalz;

	Pa11[i].ftotal=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
}
////////////////////////////////////////////////////////////////////////
void calculate_force(int_t*vii,Real*vif,part11*Pa11,part12*Pa12,part13*Pa13,part2*Pa2){
	dim3 b,t;
	t.x=256;
	b.x=(number_of_particles-1)/t.x+1;

	//int_t smsize=sizeof(Real)*thread_size;

	if(fp_solve==1){
		// pressure force calculation function
		KERNEL_clc_pressure_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
		cudaDeviceSynchronize();
		//KERNEL_clc_pressure_force<<<number_of_particles,thread_size>>>(fpx,fpy,fpz,number_of_neighbors,pnb,m,rho,p,dwx,dwy,dwz,pnb_size,dim);
		//KERNEL_clc_pressure_force_sun<<<number_of_particles,thread_size>>>(fpx,fpy,fpz,number_of_neighbors,pnb,m,rho,x,y,z,p,dwij,flt_s,dist,pnb_size,dim);
	}
	if(fv_solve==1){
		// viscous force calculation function
		switch(turbulence_model){
			case Laminar:
				//KERNEL_clc_viscous_force<<<number_of_particles,thread_size>>>(fvx,fvy,fvz,number_of_neighbors,pnb,m,rho,x,y,z,ux,uy,uz,temp,dwx,dwy,dwz,dist,pnb_size,dim,p_type);
				KERNEL_add_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
				cudaDeviceSynchronize();
				break;
			case K_LM:
				//KERNEL_clc_turbulence_viscous_force<<<number_of_particles,thread_size>>>(fvx,fvy,fvz,number_of_neighbors,pnb,m,rho,x,y,z,ux,uy,uz,temp,vis_t,dwx,dwy,dwz,dist,pnb_size,dim,p_type);
				KERNEL_add_turbulence_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa12,Pa2);
				cudaDeviceSynchronize();
				break;
			case K_E:
				//KERNEL_clc_turbulence_viscous_force<<<number_of_particles,thread_size>>>(fvx,fvy,fvz,number_of_neighbors,pnb,m,rho,x,y,z,ux,uy,uz,temp,vis_t,dwx,dwy,dwz,dist,pnb_size,dim,p_type);
				KERNEL_add_turbulence_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa12,Pa2);
				cudaDeviceSynchronize();
				break;
			case SPS:
				//KERNEL_clc_SPS_viscous_force<<<number_of_particles,thread_size>>>(fvx,fvy,fvz,Sxx,Sxy,Sxz,Syy,Syz,Szz,number_of_neighbors,pnb,m,rho,x,y,z,ux,uy,uz,temp,dwx,dwy,dwz,dist,pnb_size,dim,p_type);
				KERNEL_add_SPS_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa12,Pa2);
				cudaDeviceSynchronize();
				break;
			case HB:
				//printf("add_viscous_force()\n");
				KERNEL_add_HB_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa12,Pa2);
				cudaDeviceSynchronize();
				break;
			default:
				//KERNEL_clc_viscous_force<<<number_of_particles,thread_size>>>(fvx,fvy,fvz,number_of_neighbors,pnb,m,rho,x,y,z,ux,uy,uz,temp,dwx,dwy,dwz,dist,pnb_size,dim,p_type);
				KERNEL_add_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
				cudaDeviceSynchronize();
				break;
		}
	}
	if(fva_solve==1){
		// artificial viscous force calculation function
		//KERNEL_clc_artificial_viscous_force<<<number_of_particles,thread_size>>>(fvax,fvay,fvaz,number_of_neighbors,pnb,m,h,rho,x,y,z,ux,uy,uz,temp,dwx,dwy,dwz,dist,pnb_size,dim,p_type);
		KERNEL_add_artificial_viscous_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,soundspeed,Pa11,Pa2);
		cudaDeviceSynchronize();
	}
	if(fg_solve==1){
		// gravitational force calculation function
		//KERNEL_clc_gravity_force<<<number_of_particles,1>>>(fgx,fgy,fgz,dim);
		KERNEL_add_gravity_force<<<b,t>>>(number_of_particles,dim,Pa11);
		cudaDeviceSynchronize();
	}
	if(fs_solve==1){
		// surface tension force calculation function (2017.04.20 jyb)
		KERNEL_clc_color_field<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
		cudaDeviceSynchronize();
		KERNEL_clc_normal_gradient_c<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa13,Pa2);
		cudaDeviceSynchronize();
		KERNEL_clc_normal_gradient<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa13,Pa2);
		cudaDeviceSynchronize();
		//KERNEL_clc_surface_tension<<<number_of_particles,thread_size>>>(fsx,fsy,fsz,number_of_neighbors,pnb,m,rho,h,x,y,z,nx,ny,nz,nmag,nx_c,ny_c,nz_c,nmag_c,curv,temp,dwij,dist,pnb_size,dim,p_type);
		KERNEL_add_surface_tension<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,dim,Pa11,Pa13,Pa2);
		cudaDeviceSynchronize();
	}
	if(interface_solve==1){
		// interface sharpness force calculation function (2017.05.02 jyb)
		//KERNEL_clc_interface_sharpness<<<number_of_particles,thread_size>>>(p_type,number_of_neighbors,pnb,m,rho,p,x,y,z,fix,fiy,fiz,dwij,dist,pnb_size,dim);
		KERNEL_add_interface_sharpness<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
		cudaDeviceSynchronize();
	}
	if(fb_solve==1){
		// boundary force calculation function
		//KERNEL_clc_boundary_force<<<number_of_particles,thread_size>>>(fbx,fby,fbz,number_of_neighbors,pnb,m,rho,x,y,z,wij,p_type,dist,pnb_size,C_repulsive);
		KERNEL_add_boundary_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,c_repulsive,Pa11,Pa2);
		cudaDeviceSynchronize();
	}
	if(boussinesq_solve==1){
		// natural convection (boussinesq approximation) force calculation function
		KERNEL_add_boussinesq_force<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
		cudaDeviceSynchronize();
	}
	// sum up forces
	//KERNEL_clc_sum_force<<<number_of_particles,1>>>(ftotal,ftotalx,ftotaly,ftotalz,fpx,fpy,fpz,
	//	fvx,fvy,fvz,fvax,fvay,fvaz,fgx,fgy,fgz,fsx,fsy,fsz,fix,fiy,fiz,fbx,fby,fbz);
	KERNEL_clc_ftotal<<<b,t>>>(number_of_particles,Pa11);
	cudaDeviceSynchronize();
}
