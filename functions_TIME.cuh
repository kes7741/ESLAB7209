
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_predictor(int_t nop,Real tdt,Real ttime,part11*Pa11)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tx0,ty0,tz0,txp,typ,tzp;
	Real tux0,tuy0,tuz0,tuxp,tuyp,tuzp;
	Real tdux_dt0,tduy_dt0,tduz_dt0;
	//Real temp0,tempp,dtemp_dt0;
	//Real rho0,rhop,drho_dt0;

	int_t p_type_i=Pa11[i].p_type;

	if(p_type_i==MOVING)
	{
		Pa11[i].ux=-0.06*PI*sinf(ttime*PI);					// Update particle data by predicted x-directional velocity
		Pa11[i].uy=0;					// Update particle data by predicted y-directional velocity
		Pa11[i].uz=0;					// Update particle data by predicted z-directional velocity
	}
	else
	{
		tx0=Pa11[i].x0;					// initial x-directional position
		ty0=Pa11[i].y0;					// initial y-directional position
		tz0=Pa11[i].z0;					// initial z-directional position

		tux0=Pa11[i].ux0*(p_type_i>0);					// initial x-directional velocity
		tuy0=Pa11[i].uy0*(p_type_i>0);					// initial y-directional velocity
		tuz0=Pa11[i].uz0*(p_type_i>0);					// initial z-directional velocity

		tdux_dt0=Pa11[i].ftotalx*(p_type_i>0);			// initial x-directional acceleration
		tduy_dt0=Pa11[i].ftotaly*(p_type_i>0);			// initial y-directional acceleration
		tduz_dt0=Pa11[i].ftotalz*(p_type_i>0);			// initial z-directional acceleration

		txp=tx0+tux0*(tdt*0.5);				// Predict x-directional position (ux0 : velocity of before time step)
		typ=ty0+tuy0*(tdt*0.5);				// Predict y-directional position (uy0 : velocity of before time step)
		tzp=tz0+tuz0*(tdt*0.5);				// Predict z-directional position (ux0 : velocity of before time step)

		tuxp=tux0+tdux_dt0*(tdt*0.5);		// Predict x-directional velocity (dux_dt0 : acceleration of before time step)
		tuyp=tuy0+tduy_dt0*(tdt*0.5);		// Predict y-directional velocity (duy_dt0 : acceleration of before time step)
		tuzp=tuz0+tduz_dt0*(tdt*0.5);		// Predict z-directional velocity (duz_dt0 : acceleration of before time step)

		Pa11[i].x=txp;					// Update particle data by predicted x-directional position
		Pa11[i].y=typ;					// Update particle data by predicted y-directional position
		Pa11[i].z=tzp;					// Update particle data by predicted z-directional position

		Pa11[i].ux=tuxp;					// Update particle data by predicted x-directional velocity
		Pa11[i].uy=tuyp;					// Update particle data by predicted y-directional velocity
		Pa11[i].uz=tuzp;					// Update particle data by predicted z-directional velocity
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_predictor_enthalpy(int_t nop,Real tdt,part12*Pa12)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tenthalpy0,tenthalpyp,tdenthalpy;

	// predict density
	tenthalpy0=Pa12[i].enthalpy0;												// Inital density
	tdenthalpy=Pa12[i].denthalpy;												// Initial time derivative of density
	tenthalpyp=tenthalpy0+tdenthalpy*(tdt*0.5);					// Predict density (drho_dt0 : time derivatve of density of before time step)
	Pa12[i].enthalpy=tenthalpyp;												// Update particle data by predicted density
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_predictor_continuity(int_t nop,Real tdt,part11*Pa11)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real trho0,trhop,tdrho_dt0;

	// predict density
	trho0=Pa11[i].rho0;											// Inital density
	tdrho_dt0=Pa11[i].drho;									// Initial time derivative of density
	trhop=trho0+tdrho_dt0*(tdt*0.5);			// Predict density (drho_dt0 : time derivatve of density of before time step)
	Pa11[i].rho=trhop;											// Update particle data by predicted density
}
////////////////////////////////////////////////////////////////////////
// Eulerian time integration function
__global__ void KERNEL_clc_euler_update(int_t nop,const Real tdt,const Real tu_limit,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tux,tuy,tuz;												// velocity
	Real tx,ty,tz;													// position
	Real dux_dt,duy_dt,duz_dt;							// accleration (time derivative of velocity)
	Real t_dt;
	//Real temp,dtemp_dt;										// temperature,time derivative of temperatre ('0' : initial value )

	t_dt=tdt;

	int_t p_type_i=Pa11[i].p_type;

	tx=Pa11[i].x;															// x-directional initial position
	ty=Pa11[i].y;															// y-directional initial position
	tz=Pa11[i].z;															// z-directional initial position

	tux=Pa11[i].ux*(p_type_i>0);							// x-directinoal initial velocity
	tuy=Pa11[i].uy*(p_type_i>0);							// y-directinoal initial velocity
	tuz=Pa11[i].uz*(p_type_i>0);							// z-directional initial velocity

	dux_dt=Pa11[i].ftotalx*(p_type_i>0);			// x-directional acceleration
	duy_dt=Pa11[i].ftotaly*(p_type_i>0);			// y-directional acceleration
	duz_dt=Pa11[i].ftotalz*(p_type_i>0);			// z-directional acceleration

	tx+=tux*t_dt;														// calculate x-directional position
	ty+=tuy*t_dt;														// calculate y-directional position
	tz+=tuz*t_dt;														// calculate z-directional position

	tux+=dux_dt*t_dt;												// calculate x-directional velocity
	tuy+=duy_dt*t_dt;												// calculate y-directional velocity
	tuz+=duz_dt*t_dt;												// calculate z-directional velocity

	Pa11[i].x=tx;															// update x-directional position
	Pa11[i].y=ty;															// update y-directional position
	Pa11[i].z=tz;															// update z-directional position

	Pa11[i].x0=tx;														// update x-directional position
	Pa11[i].y0=ty;														// update y-directional position
	Pa11[i].z0=tz;														// update z-directional position

	if((tux*tux+tuy*tuy+tuz*tuz)<tu_limit*tu_limit){
		Pa11[i].ux=tux;													// update x-directional velocity
		Pa11[i].uy=tuy;													// update y-directional velocity
		Pa11[i].uz=tuz;													// update z-directional velocity

		Pa11[i].ux0=tux;												// update x-directional velocity
		Pa11[i].uy0=tuy;												// update y-directional velocity
		Pa11[i].uz0=tuz;												// update z-directional velocity
	}

	// update temperature
	/*
	temp=temp_[i];
	dtemp_dt=dtemp_[i];
	temp+=dtemp_dt*dt;
	temp_[i]=temp;
	temp0_[i]=temp;
	*/
}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_clc_precor_update(int_t nop,const Real tdt,const Real tu_limit,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tux,tuy,tuz;											// velocity
	//Real tx,ty,tz;												// position
	Real dux_dt,duy_dt,duz_dt;						// accleration (time derivative of velocity)

	Real tux0,tuy0,tuz0,uxc,uyc,uzc;			// velocity ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	Real tx0,ty0,tz0,xc,yc,zc;						// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	//Real temp0,tempc,dtemp;								// temperature,time derivative of temperatre ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)

	int_t p_type_i=Pa11[i].p_type;

	if(p_type_i==MOVING){
		tx0=Pa11[i].x0;					// x-directional initial position
		ty0=Pa11[i].y0;					// x-directional initial position
		tz0=Pa11[i].z0;					// x-directional initial position
		tux=Pa11[i].ux;
		tuy=Pa11[i].uy;
		tuz=Pa11[i].uz;

		xc=tx0+tux*(tdt);			// correct x-directional position
		yc=ty0+tuy*(tdt);			// correct Y-directional position
		zc=tz0+tuz*(tdt);			// correct Z-directional position

		Pa11[i].x0=xc;					// update x-directional position
		Pa11[i].y0=yc;					// update y-directional position
		Pa11[i].z0=zc;					// update z-directional position
		Pa11[i].x=xc;						// update x-directional position
		Pa11[i].y=yc;						// update y-directional position
		Pa11[i].z=zc;						// update z-directional position
		Pa11[i].ux0=tux;				// update x-directional velocity
		Pa11[i].uy0=tuy;				// update y-directional velocity
		Pa11[i].uz0=tuz;				// update z-directional velocity
		Pa11[i].ux=tux;					// update x-directional velocity
		Pa11[i].uy=tuy;					// update y-directional velocity
		Pa11[i].uz=tuz;					// update z-directional velocity
	}else{
		tx0=Pa11[i].x0;					// x-directional initial position
		ty0=Pa11[i].y0;					// x-directional initial position
		tz0=Pa11[i].z0;					// x-directional initial position

		tux0=Pa11[i].ux0*(p_type_i>0);						// x-directional initial velocity
		tuy0=Pa11[i].uy0*(p_type_i>0);						// y-directional initial velocity
		tuz0=Pa11[i].uz0*(p_type_i>0);						// z-directional initial velocity

		dux_dt=Pa11[i].ftotalx*(p_type_i>0);			// x-directional acceleration
		duy_dt=Pa11[i].ftotaly*(p_type_i>0);			// y-directional acceleration
		duz_dt=Pa11[i].ftotalz*(p_type_i>0);			// z-directional acceleration

		uxc=tux0+dux_dt*(tdt);									// correct x-directional velocity
		uyc=tuy0+duy_dt*(tdt);									// correct y-directional velocity
		uzc=tuz0+duz_dt*(tdt);									// correct z-directional velocity

		if((uxc*uxc+uyc*uyc+uzc*uzc)>=tu_limit*tu_limit){
			uxc=tux0;
			uyc=tuy0;
			uzc=tuz0;
		}
		xc=tx0+uxc*(tdt);			// correct x-directional position
		yc=ty0+uyc*(tdt);			// correct Y-directional position
		zc=tz0+uzc*(tdt);			// correct Z-directional position

		Pa11[i].x0=xc;					// update x-directional position
		Pa11[i].y0=yc;					// update y-directional position
		Pa11[i].z0=zc;					// update z-directional position
		Pa11[i].x=xc;						// update x-directional position
		Pa11[i].y=yc;						// update y-directional position
		Pa11[i].z=zc;						// update z-directional position
		Pa11[i].ux0=uxc;				// update x-directional velocity
		Pa11[i].uy0=uyc;				// update y-directional velocity
		Pa11[i].uz0=uzc;				// update z-directional velocity
		Pa11[i].ux=uxc;					// update x-directional velocity
		Pa11[i].uy=uyc;					// update y-directional velocity
		Pa11[i].uz=uzc;					// update z-directional velocity
	}
	//umag_[i]=sqrt(uxc*uxc+uyc*uyc+uzc*uzc);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_precor_update_vel(int_t nop,const Real tdt,const Real tu_limit,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//Real tux,tuy,tuz;												// velocity
	Real dux_dt,duy_dt,duz_dt;							// accleration (time derivative of velocity)
	Real tux0,tuy0,tuz0,uxc,uyc,uzc;				// velocity ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)

	int_t p_type_i=Pa11[i].p_type;

	if(p_type_i==MOVING)
	{
	}
	else
	{
		tux0=Pa11[i].ux0*(p_type_i>0);							// x-directional initial velocity
		tuy0=Pa11[i].uy0*(p_type_i>0);							// y-directional initial velocity
		tuz0=Pa11[i].uz0*(p_type_i>0);							// z-directional initial velocity

		dux_dt=Pa11[i].ftotalx*(p_type_i>0);			// x-directional acceleration
		duy_dt=Pa11[i].ftotaly*(p_type_i>0);			// y-directional acceleration
		duz_dt=Pa11[i].ftotalz*(p_type_i>0);			// z-directional acceleration

		uxc=tux0+dux_dt*(tdt);			// correct x-directional velocity
		uyc=tuy0+duy_dt*(tdt);			// correct y-directional velocity
		uzc=tuz0+duz_dt*(tdt);			// correct z-directional velocity

		if((uxc*uxc+uyc*uyc+uzc*uzc)>=tu_limit*tu_limit){
			uxc=tux0;
			uyc=tuy0;
			uzc=tuz0;
		}

		Pa11[i].ux0=uxc;			// update x-directional velocity
		Pa11[i].uy0=uyc;			// update y-directional velocity
		Pa11[i].uz0=uzc;			// update z-directional velocity

		Pa11[i].ux=uxc;			// update x-directional velocity
		Pa11[i].uy=uyc;			// update y-directional velocity
		Pa11[i].uz=uzc;			// update z-directional velocity
	}
	//umag_[i]=sqrt(uxc*uxc+uyc*uyc+uzc*uzc);
}
////////////////////////////////////////////////////////////////////////
// corrector step for Predictor-Corrector time integration
__global__ void KERNEL_clc_precor_update_xsph(int_t nop,int_t pnbs,Real tdt,Real ttime,Real tC_xsph,part11*Pa11,part2*Pa2)
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
	Real tux,tuy,tuz;																	// velocity
	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	Real mj,rhoj,flt_si;
	Real uxi,uyi,uzi,uxj,uyj,uzj,twij;

	Real x_boundary=0.1*cosf(PI*ttime)+3.79;

	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		p_type_i=Pa11[i].p_type;
		flt_si=Pa11[i].flt_s;
		uxi=Pa11[i].ux;
		uyi=Pa11[i].uy;
		uzi=Pa11[i].uz;
		tx0=Pa11[i].x0;					// x-directional initial position
		ty0=Pa11[i].y0;					// x-directional initial position
		tz0=Pa11[i].z0;					// x-directional initial position
		tux=uxi;
		tuy=uyi;
		tuz=uzi;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho0;
		//p_type_j=p_type_[j];
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;

		cachex[cache_idx]=tC_xsph*mj/rhoj*(uxi-uxj)*twij/flt_si;
		cachey[cache_idx]=tC_xsph*mj/rhoj*(uyi-uyj)*twij/flt_si;
		cachez[cache_idx]=tC_xsph*mj/rhoj*(uzi-uzj)*twij/flt_si;
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
		if(p_type_i==MOVING){
			xc=tx0+tux*(tdt);			// correct x-directional position
			yc=ty0+tuy*(tdt);			// correct Y-directional position
			zc=tz0+tuz*(tdt);			// correct Z-directional position
		}else{
			xc=tx0+(tux-cachex[0])*(tdt)*(p_type_i==FLUID);			// correct x-directional position
			yc=ty0+(tuy-cachey[0])*(tdt)*(p_type_i==FLUID);			// correct Y-directional position
			zc=tz0+(tuz-cachez[0])*(tdt)*(p_type_i==FLUID);			// correct Z-directional position
			if(xc>x_boundary){
				xc=2*x_boundary-xc;
				Pa11[i].ux=-tux;
				Pa11[i].ux0=-tux;
			}
		}
		Pa11[i].x0=xc;			// update x-directional position
		Pa11[i].y0=yc;			// update y-directional position
		Pa11[i].z0=zc;			// update z-directional position
		Pa11[i].x=xc;				// update x-directional position
		Pa11[i].y=yc;				// update y-directional position
		Pa11[i].z=zc;				// update z-directional position
	}

	/*
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	int_t j,jj;						// neighbor particle index-j
	uint_t non,tid;
	int_t ii=i*pnbs;

	Real tux,tuy,tuz;																	// velocity
	Real tx0,ty0,tz0,xc,yc,zc;												// position ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	Real mj,rhoj,flt_si;
	Real uxi,uyi,uzi,uxj,uyj,uzj,twij;
	Real tmpx,tmpy,tmpz;

	Real x_boundary=0.1*cosf(PI*ttime)+3.79;

	non=Pa11[i].number_of_neighbors;
	int_t p_type_i=Pa11[i].p_type;
	//int_t p_type_j;

	flt_si=Pa11[i].flt_s;
	uxi=Pa11[i].ux;
	uyi=Pa11[i].uy;
	uzi=Pa11[i].uz;
	tx0=Pa11[i].x0;					// x-directional initial position
	ty0=Pa11[i].y0;					// x-directional initial position
	tz0=Pa11[i].z0;					// x-directional initial position
	tux=uxi;
	tuy=uyi;
	tuz=uzi;

	tmpx=tmpy=tmpz=0.0;
	for(jj=0;jj<non;jj++){
		tid=ii+jj;
		j=Pa2[tid].pnb;
		twij=Pa2[tid].wij;

		mj=Pa11[j].m;
		rhoj=Pa11[j].rho0;
		//p_type_j=p_type_[j];
		uxj=Pa11[j].ux;
		uyj=Pa11[j].uy;
		uzj=Pa11[j].uz;

		tmpx+=tC_xsph*mj/rhoj*(uxi-uxj)*twij/flt_si;
		tmpy+=tC_xsph*mj/rhoj*(uyi-uyj)*twij/flt_si;
		tmpz+=tC_xsph*mj/rhoj*(uzi-uzj)*twij/flt_si;
		//cache_flts[cache_idx]=mj/rhoj*twij;
	}

	// save values
	if(p_type_i==MOVING){
		xc=tx0+tux*(tdt);			// correct x-directional position
		yc=ty0+tuy*(tdt);			// correct Y-directional position
		zc=tz0+tuz*(tdt);			// correct Z-directional position
	}else{
		xc=tx0+(tux-tmpx)*(tdt)*(p_type_i==FLUID);			// correct x-directional position
		yc=ty0+(tuy-tmpy)*(tdt)*(p_type_i==FLUID);			// correct Y-directional position
		zc=tz0+(tuz-tmpz)*(tdt)*(p_type_i==FLUID);			// correct Z-directional position
		if(xc>x_boundary){
			xc=2*x_boundary-xc;
			Pa11[i].ux=-tux;
			Pa11[i].ux0=-tux;
		}
	}

	Pa11[i].x0=xc;			// update x-directional position
	Pa11[i].y0=yc;			// update y-directional position
	Pa11[i].z0=zc;			// update z-directional position
	Pa11[i].x=xc;				// update x-directional position
	Pa11[i].y=yc;				// update y-directional position
	Pa11[i].z=zc;				// update z-directional position
	//*/
}
////////////////////////////////////////////////////////////////////////
// enthalpy corrector step for Predictor-Corrector time integration
__global__ void KERNEL_clc_precor_update_enthalpy(int_t nop,Real tdt,part12*Pa12)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//Real tenthalpy;
	Real tenthalpy0,tenthalpyc,tdenthalpy;
	// density,time derivative of density ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)

	//int_t p_type_i=p_type_[i];

	// update density
	tenthalpy0=Pa12[i].enthalpy0;
	tdenthalpy=Pa12[i].denthalpy;

	tenthalpyc=tenthalpy0+tdenthalpy*(tdt*0.5);

	Pa12[i].enthalpy0=2*tenthalpyc-tenthalpy0;
	Pa12[i].enthalpy=2*tenthalpyc-tenthalpy0;
	//if(i==1455) printf("temp %f %f\n",tenthalpy0,tdenthalpy);
}
////////////////////////////////////////////////////////////////////////
// predictor_corrector_update_continuity function  (jyb,2017.06.22)-p_type
__global__ void KERNEL_clc_precor_update_continuity(int_t nop,Real tdt,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	//int_t p_type_i=Pa11[i].p_type;
	// density,time derivative of density ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)
	/* //update density
	rho0=Pa11[i].rho0;									// initial density
	drho_dt=Pa11[i].drho;								// time derivative of density
	rhoc=rho0+drho_dt*(tdt*0.5);			// correct density
	Pa11[i].rho=rhoc;
	//*/
	Pa11[i].rho=Pa11[i].rho0+Pa11[i].drho*(tdt*0.5);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_update_density(int_t nop,part11*Pa11)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_r=2*Pa11[i].rho-Pa11[i].rho0;
	// update density
	Pa11[i].rho0=tmp_r;
	Pa11[i].rho=tmp_r;
}
////////////////////////////////////////////////////////////////////////
/*
// Volume based predictor_corrector_update_continuity function  (jyb,2017.05.01)-available for both single-phase & two-phase
__global__ void KERNEL_clc_precor_update_continuity_volume(uint_t *p_type_,uint_t *number_of_neighbors_,Real *m_,Real *rho_,Real *rho0_,Real *drho_,
	uint_t *pnb_,Real *wij_,const Real dt,int_t pnb_size,const int_t dim,const int_t simulation_type,const int_t freq_mass_sum,const int_t count)
{
	//uint_t i=blockIdx.x+blockIdx.y*gridDim.x;
	uint_t i=blockIdx.x;

	//if (i < cNUM_PARTICLES[0])
	//{
	Real rho;
	Real rho0,rhoc,drho_dt;					// density,time derivative of density ('0' : initial value/'c' : corrected value for Predictor-Corrector time stepping scheme)

	Real mi,mj,wij,rhoi,rhoj;
	int_t p_type_j;


	uint_t number_of_neighbors;
	uint_t tid=threadIdx.x+blockIdx.x*pnb_size;
	uint_t j;

	int_t p_type_i=p_type_[i];

	// update density
	rho0=rho0_[i];					// initial density
	drho_dt=drho_[i];				// time derivative of density
	rhoc=rho0+drho_dt*(dt*0.5);			// correct density
	rho_[i]=rhoc;

	// Density Filtering Step for Continuity Equation (Mass Summation)
	if ((count%freq_mass_sum)==0)
	{
		__shared__ Real cache_num[1000];
		__shared__ Real cache_den[1000];

		cache_num[threadIdx.x]=0;
		cache_den[threadIdx.x]=0;

		number_of_neighbors=number_of_neighbors_[i];
		int_t cache_idx=threadIdx.x;

		// calculate contribution of j particles on density variation (drho)
		if (cache_idx < number_of_neighbors)
		{
			j=pnb_[tid];

			mi=m_[i];
			mj=m_[j];

			rhoi=rho_[j];
			rhoj=rho_[j];
			wij=wij_[tid];


			cache_num[cache_idx]=mi*wij;
			cache_den[cache_idx]=mj*wij/rhoj;


		}
		else
		{
			cache_num[cache_idx]=0;
			cache_den[cache_idx]=0;
		}

		__syncthreads();


		// reduction
		for (uint_t s=blockDim.x/2; s>0; s>>= 1)
		{
			if (cache_idx < s)
			{
				cache_num[cache_idx]+=cache_num[cache_idx+s];
				cache_den[cache_idx]+=cache_den[cache_idx+s];
			}
			__syncthreads();
		}

		// save values
		if (cache_idx==0)
		{
			rho_[i]=cache_num[0]/cache_den[0];
		}

		rhoc=rho_[i];
	}


	// update density
	rho0_[i]=2*rhoc-rho0;
	//}
}
//*/
// update properties for two-phase flow (jyb,2017.04.19)
////////////////////////////////////////////////////////////////////////
void update_properties(int_t*vii,Real*vif,part11*Pa11,part2*Pa2)
{
	dim3 b,t;
	t.x=256;
	b.x=(number_of_particles-1)/t.x+1;

	//int_t smsize=sizeof(Real)*thread_size;

	switch(time_type){
		case Euler:
			// Eulerian time integration function
			KERNEL_clc_euler_update<<<b,t>>>(number_of_particles,dt,u_limit,Pa11);
			cudaDeviceSynchronize();
			if(rho_type==Continuity){				
				if(count%freq_mass_sum==0){
					KERNEL_clc_density_renormalization_norm<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
					cudaDeviceSynchronize();
				}else{
					KERNEL_clc_precor_update_continuity<<<b,t>>>(number_of_particles,dt,Pa11);
					cudaDeviceSynchronize();
				}
				KERNEL_clc_update_density<<<b,t>>>(number_of_particles,Pa11);
				cudaDeviceSynchronize();
			}
			break;
		case Pre_Cor:
			// Predictor-Corrector time integration function
			if(xsph_solve==1){
				KERNEL_clc_precor_update_vel<<<b,t>>>(number_of_particles,dt,u_limit,Pa11);
				cudaDeviceSynchronize();
				KERNEL_clc_precor_update_xsph<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,dt,time,c_xsph,Pa11,Pa2);
				cudaDeviceSynchronize();
			}else{
				KERNEL_clc_precor_update<<<b,t>>>(number_of_particles,dt,u_limit,Pa11);
				cudaDeviceSynchronize();
			}
			if(rho_type==Continuity){
				KERNEL_clc_precor_update_continuity<<<b,t>>>(number_of_particles,dt,Pa11);
				cudaDeviceSynchronize();
				if(count%freq_mass_sum==0){
					KERNEL_clc_density_renormalization_norm<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,Pa11,Pa2);
					cudaDeviceSynchronize();
				}
				KERNEL_clc_update_density<<<b,t>>>(number_of_particles,Pa11);
				cudaDeviceSynchronize();
				//KERNEL_clc_precor_update_continuity_volume<<<number_of_particles,thread_size>>>(p_type,number_of_neighbors,m,rho,rho0,drho,pnb,wij,dt,pnb_size,dim,simulation_type,freq_mass_sum,count);
			}
			break;
		default:
			break;
	}
}
////////////////////////////////////////////////////////////////////////
void update_properties_enthalpy(int_t*vii,Real*vif,part12*Pa12)
{
	dim3 b,t;
	t.x=256;
	b.x=(number_of_particles-1)/t.x+1;

	switch(time_type){
		case Euler:
			// Eulerian time integration function
			KERNEL_clc_precor_update_enthalpy<<<b,t>>>(number_of_particles,dt,Pa12);
			cudaDeviceSynchronize();
			break;
		case Pre_Cor:
			// Predictor-Corrector time integration function
			KERNEL_clc_precor_update_enthalpy<<<b,t>>>(number_of_particles,dt,Pa12);
			cudaDeviceSynchronize();
			break;
		default:
			break;
	}
}
////////////////////////////////////////////////////////////////////////
