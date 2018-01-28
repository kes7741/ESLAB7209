////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_dist(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;													// particle i index
	if(i>=nop) return;

	uint_t jj=ii%pnbs;
	uint_t tid=i*pnbs+jj;											// neighbor index
	uint_t j;												// particle j index

	uint_t non=Pa11[i].number_of_neighbors;
	Real dist_ij;			// distance between particle ip and jp
	Real tmpx,tmpy,tmpz;

	if(jj<non){
		j=Pa2[tid].pnb;
		// distance
		tmpx=(Pa11[i].x-Pa11[j].x);
		tmpy=(Pa11[i].y-Pa11[j].y);
		tmpz=(Pa11[i].z-Pa11[j].z);

		dist_ij=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
		// insert distance between ip and jp
		Pa2[tid].dist=dist_ij;
	}
}
////////////////////////////////////////////////////////////////////////
// Gaussian kernel
__global__ void KERNEL_clc_gaussian(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)			// Gaussian Kernel function
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;

	uint_t non;

	Real tmp_R,tmp_h;
	Real tmp_A;

	non=Pa11[i].number_of_neighbors;
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

	if(j<non){
		tmp_R=Pa2[tid].dist/tmp_h;
		Pa2[tid].wij=tmp_A*exp(-pow(tmp_R,2));
		Pa2[tid].dwij=(1.0/tmp_h)*tmp_A*(-2.0)*tmp_R*exp(-pow(tmp_R,2));
	}
}
////////////////////////////////////////////////////////////////////////
// Quintic kernel
__global__ void KERNEL_clc_quintic(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)				// Gaussian Kernel function
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;

	uint_t non;

	Real tmp_R,tmp_h;
	Real tmp_A;

	non=Pa11[i].number_of_neighbors;

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

	if(j<non){
		tmp_R=Pa2[tid].dist/tmp_h;
		if(tmp_R<1){
			Pa2[tid].wij=tmp_A*(pow(3.0-tmp_R,5)-6.0*pow(2.0-tmp_R,5)+15.0*pow(1.0-tmp_R,5));		// equation of quintic kernel function
			Pa2[tid].dwij=(1.0/tmp_h)*tmp_A*(pow(3.0-tmp_R,5)+30.0*pow(2.0-tmp_R,4)-75.0*pow(1.0-tmp_R,4));
		}else if(1<=tmp_R<2){
			Pa2[tid].wij=tmp_A*(pow(3.0-tmp_R,5)-6.0*pow(2.0-tmp_R,5));
			Pa2[tid].dwij=(1.0/tmp_h)*tmp_A*(-5.0*pow(3.0-tmp_R,4)+30.0*pow(2.0-tmp_R,4));
		}else if(2<=tmp_R<3){
			Pa2[tid].wij=tmp_A*(pow(3.0-tmp_R,5));
			Pa2[tid].dwij=(1.0/tmp_h)*tmp_A*(-5.0*pow(3.0-tmp_R,4));
		}else{
			Pa2[tid].wij=0;
			Pa2[tid].dwij=0;
		}
	}
}
////////////////////////////////////////////////////////////////////////
// Quartic kernel
__global__ void KERNEL_clc_quartic(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)				// Gaussian Kernel function
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;

	uint_t non;

	Real tmp_R,tmp_h;
	Real tmp_A;

	non=Pa11[i].number_of_neighbors;

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
	if(j<non){
		tmp_R=Pa2[tid].dist/tmp_h;
		Pa2[tid].wij=(tmp_R<2)*tmp_A*(2.0/3.0-9.0/8.0*pow(tmp_R,2)+19.0/24.0*pow(tmp_R,3)-5.0/32.0*pow(tmp_R,4));
		Pa2[tid].dwij=(tmp_R<2)*(1.0/tmp_h)*tmp_A*(-9.0/8.0*2*tmp_R+19.0/24.0*3.0*pow(tmp_R,2)-5.0/32.0*4.0*pow(tmp_R,3));
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland2 kernel
__global__ void KERNEL_clc_wendland2(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)				// Gaussian Kernel function
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;

	uint_t non;

	Real tmp_R,tmp_h;
	Real tmp_C;

	non=Pa11[i].number_of_neighbors;

	if(j<non){
		tmp_h=Pa11[i].h;
		tmp_R=Pa2[tid].dist/tmp_h*0.5;
		switch(tdim){
			case 1:
				tmp_C=1.25/(2*tmp_h);																																					// 5.0/(4*(2h))
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+3*tmp_R);												// equation of Wendland 2 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-12*tmp_R*(1-tmp_R)*(1-tmp_R));
				break;
			case 2:
				tmp_C=2.228169203286535/(4*tmp_h*tmp_h);																											// 7.0/(pi*(2h)^2)  
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);							// equation of Wendland 2 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-20*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R));
				break;
			case 3:
				tmp_C=3.342253804929802/(8*tmp_h*tmp_h*tmp_h);																								// 21.0/(2*pi*(2th)^3)
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);							// equation of Wendland 2 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-20*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R));
				break;
			default:
				tmp_C=3.342253804929802/(8*tmp_h*tmp_h*tmp_h);																								// dim 3
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R);							// equation of Wendland 2 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-20*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R));
				break;
		}
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland4 kernel
__global__ void KERNEL_clc_wendland4(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)				// Gaussian Kernel function
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;

	uint_t non;

	Real tmp_R,tmp_h;
	Real tmp_C;

	non=Pa11[i].number_of_neighbors;

	if(j<non){
		tmp_h=Pa11[i].h;
		tmp_R=Pa2[tid].dist/tmp_h*0.5;
		switch(tdim){
			case 1:
				tmp_C=1.5/(2*tmp_h);																																												// 3.0/(2*(2h))
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+5*tmp_R+8*tmp_R*tmp_R);		// equation of Wendland 4 kernel function
				Pa2[tid].dwij=(tmp_R<1)*tmp_C*(-14*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+4*tmp_R));
				break;
			case 2:
				tmp_C=2.864788975654116/(4*tmp_h*tmp_h);																																																	// 9.0/(pi*(2h)^2)
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);			// equation of Wendland 4 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-18.666666666666668*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(5*tmp_R+1));
				break;
			case 3:
				tmp_C=4.923856051905513/(8*tmp_h*tmp_h*tmp_h);																																														// 495.0/(32*pi*(2h)^3)
				Pa2[tid].wij=(tmp_R<1)*tmp_C* (1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);			// equation of Wendland 4 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-18.666666666666668*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(5*tmp_R+1));
				break;
			default:
				tmp_C=4.923856051905513/(8*tmp_h*tmp_h*tmp_h);																																														// dim3
				Pa2[tid].wij=(tmp_R<1)*tmp_C* (1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+6*tmp_R+11.666666666666666*tmp_R*tmp_R);			// equation of Wendland 4 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-18.666666666666668*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(5*tmp_R+1));
				break;
		}
	}
}
////////////////////////////////////////////////////////////////////////
// Wendland6 kernel
__global__ void KERNEL_clc_wendland6(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part2*Pa2)				// Gaussian Kernel function
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;

	uint_t non;

	Real tmp_R,tmp_h;
	Real tmp_C;

	non=Pa11[i].number_of_neighbors;

	if(j<non){
		tmp_h=Pa11[i].h;
		tmp_R=Pa2[tid].dist/tmp_h*0.5;
		switch(tdim){
			case 1:
				tmp_C=1.71875/(2*tmp_h);																																																																// 55.0/(32*(2h))
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+7*tmp_R+19*tmp_R*tmp_R+21*tmp_R*tmp_R*tmp_R);			// equation of Wendland 6 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-6*tmp_R*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(35*tmp_R*tmp_R+18*tmp_R+3));
				break;
			case 2:
				tmp_C=3.546881588905096/(4*tmp_h*tmp_h);																																																												// 78.0/(7*pi*(2h)^2)
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);		// equation of Wendland 6 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-22*tmp_R *(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(16*tmp_R*tmp_R+7*tmp_R+1));
				break;
			case 3:
				tmp_C=6.788953041263660/(8*tmp_h*tmp_h*tmp_h);																																																								// 1365.0/(64*pi*(2h)^3)
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);	// equation of Wendland 6 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-22*tmp_R *(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(16*tmp_R*tmp_R+7*tmp_R+1));
				break;
			default:
				tmp_C=6.788953041263660/(8*tmp_h*tmp_h*tmp_h);																																																								// dim3
				Pa2[tid].wij=(tmp_R<1)*tmp_C*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1+8*tmp_R+25*tmp_R*tmp_R+32*tmp_R*tmp_R*tmp_R);	// equation of Wendland 6 kernel function
				Pa2[tid].dwij=(tmp_R<1)*(1/(2*tmp_h))*tmp_C*(-22*tmp_R *(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(1-tmp_R)*(16*tmp_R*tmp_R+7*tmp_R+1));
				break;
		}
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_dwxyz(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t j=ii%pnbs;
	uint_t tid=i*pnbs+j;
	uint_t jj;
	uint_t non;

	Real tmp_dwij,tmp_dist;

	non=Pa11[i].number_of_neighbors;

	if(j<non){
		jj=Pa2[tid].pnb;
		tmp_dwij=Pa2[tid].dwij;
		tmp_dist=Pa2[tid].dist;
		if(tmp_dist>0){
			Pa2[tid].dwx=tmp_dwij*(Pa11[i].x-Pa11[jj].x)/tmp_dist;
			Pa2[tid].dwy=tmp_dwij*(Pa11[i].y-Pa11[jj].y)/tmp_dist;
			Pa2[tid].dwz=tmp_dwij*(Pa11[i].z-Pa11[jj].z)/tmp_dist;
		}else{
			Pa2[tid].dwx=0;
			Pa2[tid].dwy=0;
			Pa2[tid].dwz=0;
		}
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_filter(int_t nop,int_t pnbs,part11*Pa11,part2*Pa2)
{
	__shared__ Real cache[256];
	cache[threadIdx.x]=0;
	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	Real mj,rhoj,tmp_wij;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		tmp_wij=Pa2[tid].wij;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho;

		cache[cache_idx]=mj/rhoj*tmp_wij;
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa11[i].flt_s=cache[0];
	/*
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	uint_t non,jj;
	Real mj,rhoj,tmp_wij;
	Real tmp_Result=0.0;

	non=Pa11[i].number_of_neighbors;

	int_t j,tid;
	int_t ii=i*pnbs;

	// reduction
	for(j=0;j<non;j++){
		tid=ii+j;
		jj=Pa2[tid].pnb;
		tmp_wij=Pa2[tid].wij;

		mj=Pa11[jj].m;
		rhoj=Pa11[jj].rho;

		tmp_Result+=mj/rhoj*tmp_wij;
	}
	Pa11[i].flt_s=tmp_Result;
	//*/
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_gradient_correction001(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part13*Pa13,part2*Pa2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real xi,yi,zi,xj,yj,zj;
	Real mj,rhoj,tdwij,tdist,tmpcmd;

	Real tmpxx,tmpyy,tmpzz,tmpxy,tmpyz,tmpzx;

	non=Pa11[i].number_of_neighbors;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
	// reduction
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;

			if(tdim>=1){
				tmpxx+=-mj*tdwij*(xi-xj)*(xi-xj)/(rhoj*tdist);
			}
			if(tdim>=2){
				tmpxy+=-mj*tdwij*(yi-yj)*(xi-xj)/(rhoj*tdist);
				tmpyy+=-mj*tdwij*(yi-yj)*(yi-yj)/(rhoj*tdist);
			}
			if(tdim>=3){
				tmpzx+=-mj*tdwij*(xi-xj)*(zi-zj)/(rhoj*tdist);
				tmpyz+=-mj*tdwij*(yi-yj)*(zi-zj)/(rhoj*tdist);
				tmpzz+=-mj*tdwij*(zi-zj)*(zi-zj)/(rhoj*tdist);
			}
		}
	}
	// save values to particle array
	//Pa13[i].cm_xx=tmpxx;
	//Pa13[i].cm_yy=tmpyy;
	//Pa13[i].cm_zz=tmpzz;
	//Pa13[i].cm_xy=tmpxy;
	//Pa13[i].cm_yz=tmpyz;
	//Pa13[i].cm_zx=tmpzx;
	///////////////////////////////////////////////
	switch(tdim){
		case 1:
			Pa13[i].cm_d=tmpxx;
			if(abs(tmpxx)>0) Pa13[i].inv_cm_xx=1/tmpxx;
			break;
		case 2:
			tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
			Pa13[i].cm_d=tmpcmd;
			if(abs(tmpcmd)>0){
				Pa13[i].inv_cm_xx=tmpyy/tmpcmd;
				Pa13[i].inv_cm_xy=-tmpxy/tmpcmd;
				Pa13[i].inv_cm_yy=tmpxx/tmpcmd;
			}else{
				Pa13[i].inv_cm_xx=1;
				Pa13[i].inv_cm_xy=0;
				Pa13[i].inv_cm_yy=1;
			}
			break;
		case 3:
			tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz)-tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx)+tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);
			Pa13[i].cm_d=tmpcmd;
			if(abs(tmpcmd)>0){
				Pa13[i].inv_cm_xx=(tmpyy*tmpzz-tmpyz*tmpyz)/tmpcmd;
				Pa13[i].inv_cm_xy=(tmpzx*tmpyz-tmpxy*tmpzz)/tmpcmd;
				Pa13[i].inv_cm_zx=(tmpxy*tmpyz-tmpzx*tmpyy)/tmpcmd;
				Pa13[i].inv_cm_yy=(tmpxx*tmpzz-tmpzx*tmpzx)/tmpcmd;
				Pa13[i].inv_cm_yz=(tmpzx*tmpxy-tmpxx*tmpyz)/tmpcmd;
				Pa13[i].inv_cm_zz=(tmpxx*tmpyy-tmpxy*tmpxy)/tmpcmd;
			}
			break;
		default:
			tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
			Pa13[i].cm_d=tmpcmd;
			if(abs(tmpcmd)>0){
				Pa13[i].inv_cm_xx=tmpyy/tmpcmd;
				Pa13[i].inv_cm_xy=-tmpxy/tmpcmd;
				Pa13[i].inv_cm_yy=tmpxx/tmpcmd;
			}
			break;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_gradient_correction002(int_t nop,int_t pnbs,part11*Pa11,part13*Pa13,part2*Pa2)
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t jj=ii%pnbs;
	uint_t tid=i*pnbs+jj;

	Real xi,yi,zi,xj,yj,zj;
	Real flt_si,tdist,tdwij;
	// kernel gradient correction
	uint_t j=Pa2[tid].pnb;
	tdist=Pa2[tid].dist;
	tdwij=Pa2[tid].dwij;

	Pa2[tid].dwx=0;
	Pa2[tid].dwy=0;
	Pa2[tid].dwz=0;

	//flt_si=Pa11[i].flt_s;
	//if(flt_si<200.8){
	//}
			if(tdist>0){
			xi=Pa11[i].x;
			yi=Pa11[i].y;
			zi=Pa11[i].z;
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;

			Pa2[tid].dwx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
			Pa2[tid].dwy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
			Pa2[tid].dwz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
		}

}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_copy_dWij_to_dWij_cor(int_t nop,int_t pnbs,part2*Pa2)
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t jj=ii%pnbs;
	uint_t tid=i*pnbs+jj;

	Pa2[tid].dw_cx=Pa2[tid].dwx;
	Pa2[tid].dw_cy=Pa2[tid].dwy;
	Pa2[tid].dw_cz=Pa2[tid].dwz;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_gradient_correction_density001(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part13*Pa13,part2*Pa2)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;
	uint_t non,tid;
	int_t ii=i*pnbs;

	//Real dw_cx,dw_cy,dw_cz,rhoi;
	Real xi,yi,zi,xj,yj,zj;
	Real mj,rhoj,tdwij,tdist,tmpcmd;
	Real tmpxx,tmpyy,tmpzz;
	Real tmpxy,tmpyz,tmpzx;

	non=Pa11[i].number_of_neighbors;
	xi=Pa11[i].x;
	yi=Pa11[i].y;
	zi=Pa11[i].z;

	tmpxx=tmpyy=tmpzz=0;
	tmpxy=tmpyz=tmpzx=0;
	// reduction
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		tdwij=Pa2[tid].dwij;
		tdist=Pa2[tid].dist;

		if(tdist>0){
			xj=Pa11[j].x;
			yj=Pa11[j].y;
			zj=Pa11[j].z;
			mj=Pa11[j].m;
			rhoj=Pa11[j].rho;

			if(tdim>=1){
				tmpxx+=-mj*tdwij*(xi-xj)*(xi-xj)/(rhoj*tdist);
			}
			if(tdim>=2){
				tmpxy+=-mj*tdwij*(yi-yj)*(xi-xj)/(rhoj*tdist);
				tmpyy+=-mj*tdwij*(yi-yj)*(yi-yj)/(rhoj*tdist);
			}
			if(tdim>=3){
				tmpzx+=-mj*tdwij*(xi-xj)*(zi-zj)/(rhoj*tdist);
				tmpyz+=-mj*tdwij*(yi-yj)*(zi-zj)/(rhoj*tdist);
				tmpzz+=-mj*tdwij*(zi-zj)*(zi-zj)/(rhoj*tdist);
			}
		}
	}
	// save values to particle array
	//Pa13[i].cm_xx=tmpxx;
	//Pa13[i].cm_yy=tmpyy;
	//Pa13[i].cm_zz=tmpzz;
	//Pa13[i].cm_xy=tmpxy;
	//Pa13[i].cm_yz=tmpyz;
	//Pa13[i].cm_zx=tmpzx;

	switch(tdim){
		case 1:
			Pa13[i].cm_d=tmpxx;
			if(abs(tmpxx)>0) Pa13[i].inv_cm_xx=1/tmpxx;
			break;
		case 2:
			tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
			Pa13[i].cm_d=tmpcmd;
			if(abs(tmpcmd)>0){
				Pa13[i].inv_cm_xx=tmpyy/tmpcmd;
				Pa13[i].inv_cm_xy=-tmpxy/tmpcmd;
				Pa13[i].inv_cm_yy=tmpxx/tmpcmd;
			}
			break;
		case 3:
			tmpcmd=tmpxx*(tmpyy*tmpzz-tmpyz*tmpyz)-tmpxy*(tmpxy*tmpzz-tmpyz*tmpzx)+tmpzx*(tmpxy*tmpyz-tmpyy*tmpzx);
			Pa13[i].cm_d=tmpcmd;
			if(abs(tmpcmd)>0){
				Pa13[i].inv_cm_xx=(tmpyy*tmpzz-tmpyz*tmpyz)/tmpcmd;
				Pa13[i].inv_cm_xy=(tmpzx*tmpyz-tmpxy*tmpzz)/tmpcmd;
				Pa13[i].inv_cm_zx=(tmpxy*tmpyz-tmpzx*tmpyy)/tmpcmd;
				Pa13[i].inv_cm_yy=(tmpxx*tmpzz-tmpzx*tmpzx)/tmpcmd;
				Pa13[i].inv_cm_yz=(tmpzx*tmpxy-tmpxx*tmpyz)/tmpcmd;
				Pa13[i].inv_cm_zz=(tmpxx*tmpyy-tmpxy*tmpxy)/tmpcmd;
			}
			break;
		default:
			tmpcmd=tmpxx*tmpyy-tmpxy*tmpxy;
			Pa13[i].cm_d=tmpcmd;
			if(abs(tmpcmd)>0){
				Pa13[i].inv_cm_xx=tmpyy/tmpcmd;
				Pa13[i].inv_cm_xy=-tmpxy/tmpcmd;
				Pa13[i].inv_cm_yy=tmpxx/tmpcmd;
			}
			break;
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_gradient_correction_density002(int_t nop,int_t pnbs,part11*Pa11,part13*Pa13,part2*Pa2)
{
	uint_t ii=threadIdx.x+blockIdx.x*blockDim.x;
	if(ii>=nop*pnbs) return;

	uint_t i=ii/pnbs;
	if(i>=nop) return;

	uint_t jj=ii%pnbs;
	uint_t tid=i*pnbs+jj;

	Real xi,yi,zi,xj,yj,zj;
	Real flt_si,tdist,tdwij;
	// kernel gradient correction
	uint_t j=Pa2[tid].pnb;
	tdist=Pa2[tid].dist;
	tdwij=Pa2[tid].dwij;

	Pa2[tid].dw_cx=0;
	Pa2[tid].dw_cy=0;
	Pa2[tid].dw_cz=0;
	//flt_si=Pa11[i].flt_s;
	//if(flt_si<200.8){
	//}
	if(tdist>0){
		xi=Pa11[i].x;
		yi=Pa11[i].y;
		zi=Pa11[i].z;
		xj=Pa11[j].x;
		yj=Pa11[j].y;
		zj=Pa11[j].z;

		Pa2[tid].dw_cx=((Pa13[i].inv_cm_xx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_xy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zx*tdwij*(zi-zj)/tdist));
		Pa2[tid].dw_cy=((Pa13[i].inv_cm_xy*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yy*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(zi-zj)/tdist));
		Pa2[tid].dw_cz=((Pa13[i].inv_cm_zx*tdwij*(xi-xj)/tdist)+(Pa13[i].inv_cm_yz*tdwij*(yi-yj)/tdist)+(Pa13[i].inv_cm_zz*tdwij*(zi-zj)/tdist));
	}
}
////////////////////////////////////////////////////////////////////////
void calculate_kernel(int_t*vii,int_t*k_vii,part11*Pa11,part2*Pa2)
{
	dim3 b,t;
	t.x=256;
	b.x=(number_of_particles*pnb_size-1)/t.x+1;

	switch(kernel_type){
		case Gaussian:
			KERNEL_clc_gaussian<<<b,t>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
			cudaDeviceSynchronize();
			break;
		case Quintic:
			KERNEL_clc_quintic<<<b,t>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
			cudaDeviceSynchronize();
			break;
		case Quartic:
			KERNEL_clc_quartic<<<b,t>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
			cudaDeviceSynchronize();
			break;
		case Wendland2:
			KERNEL_clc_wendland2<<<b,t>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
			cudaDeviceSynchronize();
			break;
		case Wendland4:
			KERNEL_clc_wendland4<<<b,t>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
			cudaDeviceSynchronize();
			break;
		case Wendland6:
			KERNEL_clc_wendland6<<<b,t>>>(number_of_particles,pnb_size,dim,Pa11,Pa2);
			cudaDeviceSynchronize();
			break;
		default:
			break;
	}
	KERNEL_clc_dwxyz<<<b,t>>>(number_of_particles,pnb_size,Pa11,Pa2);
	cudaDeviceSynchronize();
}
