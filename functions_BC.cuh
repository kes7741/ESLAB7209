////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_switch_p_type_MCCI_CCI(int_t nop,part11*Pa11,part12*Pa12)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	Real temp=Pa11[i].temp;
	//Real concn=Pa12[i].concn;
	uint_t p_type=Pa11[i].p_type;

	if (p_type == CONCRETE_SOL)
	{
		if (temp > 1523.15)  // cci 1 siliceous concrete ablation temperature
		{
			Pa11[i].p_type = CONCRETE;
			//m_[i] = 0.6;
			//rho = reference_density(MCCI_CORIUM, temp, concn);
			//rho_ref_[i] = rho;
			//rho0_[i] = rho;
			//rho_[i] = rho;
		}
	}

}

/*
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_switch_p_type_CS_VUL(uint_t *p_type_, Real *y_, Real *ftotalx_, Real *ftotaly_, Real *ftotalz_)
{
	//uint_t i = blockIdx.x + gridDim.x * blockIdx.y;
	uint_t i = blockIdx.x;

	if (p_type_[i] == DUMMY_IN)
	{
		ftotalx_[i] = 0;
		ftotaly_[i] = 0;
		ftotalz_[i] = 0;

		if (y_[i] < Y_IN)
		{
			p_type_[i] = CORIUM;
		}
	}

}
////////////////////////////////////////////////////////////////////////
*/
