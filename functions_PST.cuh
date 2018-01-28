////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_normal(int_t nop,int_t pnbs,Real tnd_ref,part11*Pa11,part13*Pa13,part2*Pa2)
{
	__shared__ Real cachex[256];
	__shared__ Real cachey[256];
	__shared__ Real cachez[256];
	__shared__ Real cachek[256];

	cachex[threadIdx.x]=0;
	cachey[threadIdx.x]=0;
	cachez[threadIdx.x]=0;
	cachek[threadIdx.x]=0;

	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	Real xi,yi,zi,xj,yj,zj;
	Real twij,tdist;
	Real tnx,tny,tnz,tn_mag;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		tdist=Pa2[tid].dist;

		xi=Pa11[i].x;
		yi=Pa11[i].y;
		zi=Pa11[i].z;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		cachex[cache_idx]=(xi-xj)/(tdist+1e-10)/non;
		cachey[cache_idx]=(yi-yj)/(tdist+1e-10)/non;
		cachez[cache_idx]=(zi-zj)/(tdist+1e-10)/non;
		cachek[cache_idx]=twij;
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s){
			cachex[cache_idx]+=cachex[cache_idx+s];
			cachey[cache_idx]+=cachey[cache_idx+s];
			cachez[cache_idx]+=cachez[cache_idx+s];
			cachek[cache_idx]+=cachek[cache_idx+s];
		}
		__syncthreads();
	}
	if(cache_idx==0){
		tnx=cachex[0];
		tny=cachey[0];
		tnz=cachez[0];
		tn_mag=sqrt(tnx*tnx+tny*tny+tnz+tnz);

		Pa13[i].nx_s=tnx/tn_mag;				
		Pa13[i].ny_s=tny/tn_mag;
		Pa13[i].nz_s=tnz/tn_mag;
		Pa13[i].lbl_surf=(tn_mag>0.3)|(non<10);
		Pa13[i].num_density=cachek[0]/tnd_ref;
	}
	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real xi,yi,zi,xj,yj,zj;
	Real twij,tdist;
	//Real mj,rhoj,hj,tdwx,tdwy,tdwz;
	//Real trho,flts,th;
	Real tmpx,tmpy,tmpz,tmpk;
	Real nx,ny,nz,n_mag;

	//int_t p_typej;
	non=Pa11[i].number_of_neighbors;

	//th=Pa11[i].h;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;
	//flts=Pa11[i].flt_s;

	tmpx=tmpy=tmpz=tmpk=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		tdist=Pa2[tid].dist;
		//tdwx=Pa2[tid].dwx;
		//tdwy=Pa2[tid].dwy;
		//tdwz=Pa2[tid].dwz;

		//mj=Pa11[j].m;
		//rhoj=Pa11[j].rho;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		//p_typej=p_type_[j];
		// surface normal vector: nx_s,ny_s,nz_s
		// method 1
		//cache_x[cache_idx]=(xi-xj)/number_of_neighbors;
		//cache_y[cache_idx]=(yi-yj)/number_of_neighbors;
		//cache_z[cache_idx]=(zi-zj)/number_of_neighbors;

		// method 2
		tmpx+=(xi-xj)/(tdist+1e-10)/non;
		tmpy+=(yi-yj)/(tdist+1e-10)/non;
		tmpz+=(zi-zj)/(tdist+1e-10)/non;

		// method 3
		//cache_x[cache_idx]=h*mj/rhoj*dwx;
		//cache_y[cache_idx]=h*mj/rhoj*dwy;
		//cache_z[cache_idx]=h*mj/rhoj*dwz;

		// number density
		tmpk+=twij;
	}

	//nx=cache_x[0]/(number_of_neighbors);
	//ny=cache_y[0]/(number_of_neighbors);
	//nz=cache_z[0]/(number_of_neighbors);
	nx=tmpx;
	ny=tmpy;
	nz=tmpz;
	//num_density_[i]=sqrt((nx*nx+ny*ny+nz*nz));
	n_mag=sqrt(nx*nx+ny*ny+nz+nz);

	//surface normal vecotr
	Pa13[i].nx_s=nx/n_mag;				
	Pa13[i].ny_s=ny/n_mag;
	Pa13[i].nz_s=nz/n_mag;

	//surface detection: label of surface particle (surface=1,non-surface=0)
	Pa13[i].lbl_surf=(n_mag>0.3)|(non<10);
	//lbl_surf[i]=n_mag;

	//number density
	//num_density_[i]=number_of_neighbors_[i]/cache_k[0]/tnd_ref;	// normalized
	Pa13[i].num_density=tmpk/tnd_ref;
	// num_density_[i]=tnd_ref;
	// num_density_[i]=number_of_neighbors_[i]/cache_k[0];
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_surface_detect(int_t nop,int_t pnbs,part11*Pa11,part13*Pa13,part2*Pa2)
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
	Real xi,yi,zi,xj,yj,zj;
	Real tdist,tnx,tny,tnz,tn_mag;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tdist=Pa2[tid].dist;

		xi=Pa11[i].x;
		yi=Pa11[i].y;
		zi=Pa11[i].z;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		cachex[cache_idx]=(xi-xj)/(tdist+1e-10)/non;
		cachey[cache_idx]=(yi-yj)/(tdist+1e-10)/non;
		cachez[cache_idx]=(zi-zj)/(tdist+1e-10)/non;
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
		tnx=cachex[0];
		tny=cachey[0];
		tnz=cachez[0];
		tn_mag=sqrt(tnx*tnx+tny*tny+tnz+tnz);
		//surface normal vecotr
		Pa13[i].nx_s=tnx/tn_mag;
		Pa13[i].ny_s=tny/tn_mag;
		Pa13[i].nz_s=tnz/tn_mag;
		//surface detection: label of surface particle (surface=1,non-surface=0)
		Pa13[i].lbl_surf=(tn_mag>0.3)|(non<10);
	}
	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real xi,yi,zi,xj,yj,zj;
	Real tdist,nx,ny,nz,n_mag;
	Real tmpx,tmpy,tmpz;

	//int_t p_typej;
	//Real mj,rhoj,hj;
	//Real twij,tdwx,tdwy,tdwz,th,trho,flts;
	//th=Pa11[i].h;
	//flts=Pa11[i].flt_s;


	non=Pa11[i].number_of_neighbors;

	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdist=Pa2[tid].dist;

		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		//mj=Pa11[j].m;
		//rhoj=Pa11[j].rho;
		//twij=Pa2[tid].wij;
		//tdwx=Pa2[tid].dwx;
		//tdwy=Pa2[tid].dwy;
		//tdwz=Pa2[tid].dwz;
		//p_typej=Pa11[j].p_type;

		// surface normal vector: nx_s,ny_s,nz_s
		// method 1
		//tmpx+=(xi-xj)/non;
		//tmpy+=(yi-yj)/non;
		//tmpz+=(zi-zj)/non;

		// method 2
		tmpx+=(xi-xj)/(tdist+1e-10)/non;
		tmpy+=(yi-yj)/(tdist+1e-10)/non;
		tmpz+=(zi-zj)/(tdist+1e-10)/non;

		// method 3
		//tmpx+=h*mj/rhoj*dwx;
		//tmpy+=h*mj/rhoj*dwy;
		//tmpz+=h*mj/rhoj*dwz;
	}

	//nx=tmpx/(non);
	//ny=tmpy/(non);
	//nz=tmpz/(non);

	nx=tmpx;
	ny=tmpy;
	nz=tmpz;

	//num_density_[i]=sqrt((nx*nx+ny*ny+nz*nz));
	n_mag=sqrt(nx*nx+ny*ny+nz+nz);

	//surface normal vecotr
	Pa13[i].nx_s=nx/n_mag;
	Pa13[i].ny_s=ny/n_mag;
	Pa13[i].nz_s=nz/n_mag;

	//surface detection: label of surface particle (surface=1,non-surface=0)
	Pa13[i].lbl_surf=(n_mag>0.3)|(non<10);
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_particle_shifting_lind(int_t nop,int_t pnbs,part11*Pa11,part13*Pa13,part2*Pa2)
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

	int_t p_type_i;
	Real xi,yi,zi;
	Real dx,dy,dz;
	Real mj,rhoj,hj;
	Real twij,tdwx,tdwy,tdwz,tdist,w_dx_i,ww;
	Real dr_square;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		tdist=Pa2[tid].dist;

		if(tdist>0){
			j=Pa2[tid].pnb;
			twij=Pa2[tid].wij;
			tdwx=Pa2[tid].dwx;
			tdwy=Pa2[tid].dwy;
			tdwz=Pa2[tid].dwz;

			p_type_i=Pa11[i].p_type;
			xi=Pa11[i].x0;
			yi=Pa11[i].y0;
			zi=Pa11[i].z0;
			w_dx_i=Pa11[i].w_dx;

			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;
			hj=Pa11[j].h;

			ww=twij/w_dx_i;
			cachex[cache_idx]=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
			cachey[cache_idx]=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
			cachez[cache_idx]=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwz;
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
		dr_square=cachex[0]*cachex[0]+cachey[0]*cachey[0]+cachez[0]*cachez[0];

		if ((Pa13[i].lbl_surf<0.5)&(dr_square<0.01*Pa11[i].h*Pa11[i].h)){		// interior
			dx=cachex[0]*(p_type_i>0);
			dy=cachey[0]*(p_type_i>0);
			dz=cachez[0]*(p_type_i>0);

			Pa11[i].x0=xi+dx;
			Pa11[i].y0=yi+dy;
			Pa11[i].z0=zi+dz;
		}
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	int_t p_type_i;
	Real tmpx,tmpy,tmpz;
	Real xi,yi,zi;//,xj,yj,zj;
	Real dx,dy,dz;
	Real mj,rhoj,hj;
	//Real uxi,uyi,uzi,uxj,uyj,uzj,rho;
	Real twij,tdwx,tdwy,tdwz,tdist,w_dx_i,ww;
	Real dr_square;
	//Real R,C;
	//Real ci,cj;

	non=Pa11[i].number_of_neighbors;
	p_type_i=Pa11[i].p_type;
	xi=Pa11[i].x0;
	yi=Pa11[i].y0;
	zi=Pa11[i].z0;
	
	//uxi=Pa11[i].ux0;
	//uyi=Pa11[i].uy0;
	//uzi=Pa11[i].uz0;

	//ci=Pa11[i].num_density;
	w_dx_i=Pa11[i].w_dx;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;
		tdwx=Pa2[tid].dwx;
		tdwy=Pa2[tid].dwy;
		tdwz=Pa2[tid].dwz;
		tdist=Pa2[tid].dist;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;
		hj=Pa11[j].h;

		//xj=Pa11[j].x0;
		//yj=Pa11[j].y0;
		//zj=Pa11[j].z0;
		//uxj=Pa11[j].ux0;
		//uyj=Pa11[j].uy0;
		//uzj=Pa11[j].uz0;

		//cj=Pa13[j].num_density;

		if(tdist>0){
			ww=twij/w_dx_i;

			//cache_x[cache_idx]=-0.5*hj*hj*mj/rhoj*(cj-ci)*dwx;
			//cache_y[cache_idx]=-0.5*hj*hj*mj/rhoj*(cj-ci)*dwy;
			//cache_z[cache_idx]=-0.5*hj*hj*mj/rhoj*(cj-ci)*dwz;
			//cj=fmin(cj,1.0);

			tmpx+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwx;
			tmpy+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwy;
			tmpz+=-0.02*hj*hj*mj/rhoj*(0.2*(ww*ww*ww*ww))*tdwz;

			//cache_x[cache_idx]=-0.5*hj*hj*mj/rhoj*cj*dwx;
			//cache_y[cache_idx]=-0.5*hj*hj*mj/rhoj*cj*dwy;
			//cache_z[cache_idx]=-0.5*hj*hj*mj/rhoj*cj*dwz;
		}
	}
	// shift position
	dr_square=tmpx*tmpx+tmpy*tmpy+tmpz*tmpz;

	//if ((flt_s_[i]>0.3)&(dr_square<0.04*h_[i]*h_[i]))
	if ((Pa13[i].lbl_surf<0.5)&(dr_square<0.01*Pa11[i].h*Pa11[i].h)){		// interior
		dx=tmpx*(p_type_i>0);
		dy=tmpy*(p_type_i>0);
		dz=tmpz*(p_type_i>0);

		Pa11[i].x0=xi+dx;
		Pa11[i].y0=yi+dy;
		Pa11[i].z0=zi+dz;
	}
	//*/
}
////////////////////////////////////////////////////////////////////////
// Gaussian kernel
__global__ void KERNEL_clc_gaussian_w_dx(part11*Pa11,const int_t tdim,int_t nop)			// Gaussian Kernel function
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_R,tmp_h;
	Real tmp_A;

	tmp_R=2/3 ;			// dx/h=(2/3*h)/h
	tmp_h=Pa11[i].h;

	switch(tdim){
		case 1:
			tmp_A=1.0/(pow(PI,0.5)*tmp_h);
			break;
		case 2:
			tmp_A=1.0/(PI*pow(tmp_h,2));
			break;
		case 3:
			tmp_A=1.0/(pow(PI,1.5)*pow(tmp_h,3));
			break;
		default:
			tmp_A=1.0/(pow(PI,1.5)*pow(tmp_h,3));
			break;
	}
	Pa11[i].w_dx=tmp_A*exp(-pow(tmp_R,2));
}
////////////////////////////////////////////////////////////////////////
// Quintic kernel
__global__ void KERNEL_clc_quintic_w_dx(part11*Pa11,const int_t tdim,int_t nop)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_R,tmp_h;
	Real tmp_A;

	tmp_R=2/3 ;			// dx/h=(2/3*h)/h
	tmp_h=Pa11[i].h;

	switch(tdim){
		case 1:
			tmp_A=1.0;
			break;
		case 2:
			tmp_A=7.0/(478.0*PI*pow(tmp_h,2));
			break;
		case 3:
			tmp_A=3.0/(359.0*PI*pow(tmp_h,3));
			break;
		default:
			tmp_A=3.0/(359.0*PI*pow(tmp_h,3));
			break;
	}
	Pa11[i].w_dx=tmp_A*(pow(3.0-tmp_R,5)-6.0*pow(2.0-tmp_R,5)+15.0*pow(1.0-tmp_R,5));
}
////////////////////////////////////////////////////////////////////////
// Quartic kernel
__global__ void KERNEL_clc_quartic_w_dx(part11*Pa11,const int_t tdim,int_t nop)				// Gaussian Kernel function
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_R,tmp_h;
	Real tmp_A;

	tmp_R=2/3 ;			// dx/h=(2/3*h)/h
	tmp_h=Pa11[i].h;

	switch(tdim){
		case 1:
			tmp_A=1.0/tmp_h;
			break;
		case 2:
			tmp_A=15.0/(7.0*PI*pow(tmp_h,2));
			break;
		case 3:
			tmp_A=315.0/(208.0*PI*pow(tmp_h,3));
			break;
		default:
			tmp_A=315.0/(208.0*PI*pow(tmp_h,3));
			break;
	}
	Pa11[i].w_dx=(tmp_R<2)*tmp_A*(2.0/3.0-9.0/8.0*pow(tmp_R,2)+19.0/24.0*pow(tmp_R,3)-5.0/32.0*pow(tmp_R,4));
}
////////////////////////////////////////////////////////////////////////
// Wendland2 kernel
__global__ void KERNEL_clc_wendland2_w_dx(part11*Pa11,const int_t tdim,int_t nop)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_R,tmp_h;
	Real tmp_C;

	tmp_R=1/3;			// dx/2h=(2/3*h)/h*0.5
	tmp_h=Pa11[i].h;

	switch(tdim){
		case 1:
			tmp_C=1.25/(2*tmp_h);																// 5./(4*(2h))
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+3*tmp_R);								// equation of Wendland 2 kernel function
			break;
		case 2:
			tmp_C=2.228169203286535/(4*tmp_h*tmp_h);											// 7./(pi *(2h)^2)  
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);							// equation of Wendland 2 kernel function
			break;
		case 3:
			tmp_C=3.342253804929802/(8*tmp_h*tmp_h*tmp_h);										// 21./(2*pi *(2h)^3)
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);							// equation of Wendland 2 kernel function
			break;
		default:
			tmp_C=3.342253804929802/(8*tmp_h*tmp_h*tmp_h);										// dim 3
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);							// equation of Wendland 2 kernel function
			break;
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland4 kernel
__global__ void KERNEL_clc_wendland4_w_dx(part11*Pa11,const int_t tdim,int_t nop)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_R,tmp_h;
	Real tmp_C;

	tmp_R=1/3;			// dx/2h=(2/3*h)/h*0.5
	tmp_h=Pa11[i].h;

	switch(tdim){
		case 1:
			tmp_C=1.5/(2*tmp_h);																																											// 3./ (2*(2h))
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+5*tmp_R+8*tmp_R*tmp_R);		// equation of Wendland 4 kernel function
			break;
		case 2:
			tmp_C=2.864788975654116/(4*tmp_h*tmp_h);																																																// 9./(pi*(2tmp_h)^2)
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);			// equation of Wendland 4 kernel function
			break;
		case 3:
			tmp_C=4.923856051905513/(8*tmp_h*tmp_h*tmp_h);																																												// 495./(32*pi*(2tmp_h)^3)
			Pa11[i].w_dx=(tmp_R<1)*tmp_C* (1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);		// equation of Wendland 4 kernel function
			break;
		default:
			tmp_C=4.923856051905513/(8*tmp_h*tmp_h*tmp_h);																																												// dim3
			Pa11[i].w_dx=(tmp_R<1)*tmp_C* (1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);		// equation of Wendland 4 kernel function
			break;
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland6 kernel
__global__ void KERNEL_clc_wendland6_w_dx(part11*Pa11,const int_t tdim,int_t nop)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_R,tmp_h;
	Real tmp_C;

	tmp_R=1/3;			// dx/2h=(2/3*h)/h*0.5
	tmp_h=Pa11[i].h;

	switch(tdim){
		case 1:
			tmp_C=1.71875/(2*tmp_h);																																																															// 55./(32*(2tmp_h))
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+7*tmp_R+19*tmp_R*tmp_R+21*tmp_R*tmp_R*tmp_R);			// equation of Wendland 6 kernel function
			break;
		case 2:
			tmp_C=3.546881588905096/(4*tmp_h*tmp_h);																																																											// 78./(7*pi*(2tmp_h)^2)
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);		// equation of Wendland 6 kernel function
			break;
		case 3:
			tmp_C=6.788953041263660/(8*tmp_h*tmp_h*tmp_h);																																																							// 1365./(64*pi*(2tmp_h)^3)
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);	// equation of Wendland 6 kernel function
			break;
		default:
			tmp_C=6.788953041263660/(8*tmp_h*tmp_h*tmp_h);																																																							// dim3
			Pa11[i].w_dx=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);	// equation of Wendland 6 kernel function
			break;
	}
}
////////////////////////////////////////////////////////////////////////
void calculate_w_dx(int_t*vii,part11*Pa11)
{
	dim3 b,t;
	t.x=256;
	b.x=(number_of_particles-1)/t.x+1;

	// Calculate kernel value for initial spacing
	switch(kernel_type){
		case Gaussian:
			KERNEL_clc_gaussian_w_dx<<<b,t>>>(Pa11,dim,number_of_particles);
			cudaDeviceSynchronize();
			break;
		case Quintic:
			KERNEL_clc_quintic_w_dx<<<b,t>>>(Pa11,dim,number_of_particles);
			cudaDeviceSynchronize();
			break;
		case Quartic:
			KERNEL_clc_quartic_w_dx<<<b,t>>>(Pa11,dim,number_of_particles);
			cudaDeviceSynchronize();
			break;
		case Wendland2:
			KERNEL_clc_wendland2_w_dx<<<b,t>>>(Pa11,dim,number_of_particles);
			cudaDeviceSynchronize();
			break;
		case Wendland4:
			KERNEL_clc_wendland4_w_dx<<<b,t>>>(Pa11,dim,number_of_particles);
			cudaDeviceSynchronize();
			break;
		case Wendland6:
			KERNEL_clc_wendland6_w_dx<<<b,t>>>(Pa11,dim,number_of_particles);
			cudaDeviceSynchronize();
			break;
		default:
			break;
	}
}
