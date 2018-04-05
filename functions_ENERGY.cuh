// calculation of enthalpy to temperature
__global__ void KERNEL_clc_TemptoEnthalpy(int_t nop,part11*Pa11,part12*Pa12)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tmp_enthalpy;  //tmp_enthalpy0
	Real tmp_temp;
	uint_t tmp_pt=Pa11[i].p_type;
	tmp_temp=Pa11[i].temp;

	//enthalpy = 0.0001112 * (temp*temp*temp) - 0.3768 * (temp*temp) + 982.1 * temp - 828300.0;
	tmp_enthalpy = Ttoh(tmp_temp, tmp_pt);
	Pa12[i].enthalpy = tmp_enthalpy;
	Pa12[i].enthalpy0 = tmp_enthalpy;
}
////////////////////////////////////////////////////////////////////////
// calculation of enthalpy to temperature
__global__ void KERNEL_clc_EnthalpytoTemp(int_t nop,part11*Pa11,part12*Pa12)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real tenthalpy=Pa12[i].enthalpy;
	int_t tp_type=Pa11[i].p_type;

	Pa11[i].temp = htoT(tenthalpy, tp_type);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_conduction(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part12*Pa12,part13*Pa13,part2*Pa2)
{
	__shared__ Real cache[256];

	cache[threadIdx.x]=0;
	uint_t i=blockIdx.x;
	int_t cache_idx=threadIdx.x;
	uint_t tid=threadIdx.x+blockIdx.x*pnbs;

	uint_t non,j;
	uint_t ptypei,ptypej;
	Real rhoi,rhoj,mi,mj;
	Real kci,kcj,tempi,tempj;
	Real qwi,qsi,ai,voli;
	Real tdist,tdwij;
	Real hi,eta;
	Real sum_hf,sum_hs,sum_con_H; //sum_H
	Real lbl_surf_i;
	qsi = 0.;
	qwi = 0.;


	non=Pa11[i].number_of_neighbors;

	if(cache_idx<non){
		tdist=Pa2[tid].dist;																					// distance between i and j particles (dist)

		if(tdist>0.){
			j=Pa2[tid].pnb;																							// neighbor particle index: j
			tdwij=Pa2[tid].dwij;																					// kernel derivative function between i and j particles (dWij)

			ptypei=Pa11[i].p_type;
			mi=Pa11[i].m;																										// i particle mass
			rhoi=Pa11[i].rho;																								// i particle density
			tempi=Pa11[i].temp;																							// i particle temperature
			hi=Pa11[i].h;
			mj=Pa11[j].m;																									// j particle mass
			ptypej=Pa11[j].p_type;
			rhoj=Pa11[j].rho;																							// j particle density
			tempj=Pa11[j].temp;																						// j particle temperature
			lbl_surf_i=Pa13[i].lbl_surf;

			eta=0.001*hi;																										// eta=0.01*h (for smoothing the singularity when dist is close to zero)

			kci = conductivity(tempi, ptypei);		// i particle conductivity
			kcj = conductivity(tempj, ptypej);		// j particle conductivity

			//qwi = DEVICE_clc_heat_source(tempi, lbl_surf_i, ptypei);						// radiation heat transfer heat flux [W/m^2]
			//qwi += DEVICE_clc_boiling_h(tempi, lbl_surf_i, ptypei);							// heat flux of boiling heat transfer [W/m^2]
			// qsi = DEVICE_clc_heat_generation(tempi, ptypei);								// Volumetric heat genration rate [W/m^3]

			// if (tdim == 1)	ai = 1;												// heat flux area of particle i (dim: 1)
			// if (tdim == 2)	ai = (2. / 3.) * hi;								// heat flux area of particle i (dim: 2)
			// if (tdim == 3)	ai = (2. / 3.) * hi * (2. / 3.) * hi;				// heat flux area of particle i (dim: 3)

			// if (tdim == 1)	voli = (2. / 3.) * hi;													// volume of particle i (dim: 1)
			// if (tdim == 2)	voli = (2. / 3.) * hi * (2. / 3.) * hi;									// volume of particle i (dim: 2)
			// if (tdim == 3)	voli = (2. / 3.) * hi * (2. / 3.) * hi * (2. / 3.) * hi;				// volume of particle i (dim: 3)

			//sum_hf=qwi*ai/mi;																							// heat flux boundary equation
			// sum_hs=qsi*voli/mi;
			//sum_hs = 10000 *(ptypei==MCCI_CORIUM);

			sum_con_H=((4.0*mj*kcj*kci)/(rhoi*rhoj*(kci+kcj+0.0001)))*(tempi-tempj)*tdwij/((tdist)+eta*eta);		// Heat Conduction Equation
			cache[cache_idx]=sum_con_H;
		}
	}
	__syncthreads();
	uint_t s;
	for(s=blockDim.x*0.5;s>0;s>>=1){
		if(cache_idx<s) cache[cache_idx]+=cache[cache_idx+s];
		__syncthreads();
	}
	if(cache_idx==0) Pa12[i].denthalpy=(cache[0]);  //Pa12[i].denthalpy=(cache[0]+sum_hf+sum_hs)*((Real)(1.0-Pa12[i].ct_boundary));
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_clc_heat_source_sink_term(int_t nop,int_t pnbs,int_t tdim,part11*Pa11,part12*Pa12,part13*Pa13)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real radiation_flux, boil_flux, heat_source;
	Real tmp_h, area;
	Real tmp_rho_ref;
	uint_t ptypei = Pa11[i].p_type;

	radiation_flux=0.;
	boil_flux=0.;
	heat_source=0.;

	ptypei = abs(ptypei);
	Real tempi=Pa11[i].temp;
	Real lbl_surf_i=Pa13[i].lbl_surf;

	tmp_rho_ref=Pa11[i].rho_ref;
	tmp_h = Pa11[i].h;
	area = 1.5/(tmp_h*tmp_rho_ref);

	//radiation heat transfer
	//radiation_flux=DEVICE_clc_heat_source(tempi, lbl_surf_i, ptypei)*area;						// radiation heat transfer heat flux [W/m^2]
	//boil_flux= DEVICE_clc_boiling_h(tempi, lbl_surf_i, ptypei)*area;							// heat flux of boiling heat transfer [W/m^2]
	heat_source=DEVICE_clc_heat_generation(tempi, ptypei)/tmp_rho_ref;								// Volumetric heat genration rate [W/m^3]

	Pa12[i].denthalpy=Pa12[i].denthalpy+radiation_flux+boil_flux+heat_source;
}
