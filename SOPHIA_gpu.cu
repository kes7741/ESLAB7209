//---------------------------------------------------------------------------------------------------
// SOPHIA_gpu Version 2.0: Smoothed Particle Hydrodynamics code In Advanced nuclear safety
// Developed by Eung Soo Kim, Young Beom Jo, So Hyun Park, Hae Yoon Choi in 2017
// ENERGy SYSTEM LABORATORY, NUCLEAR ENGINEERING DEPARTIMENT, SEOUL NATIONAL UNIVERSITY, SOUTH KOREA
//---------------------------------------------------------------------------------------------------
// Optimized by Dong Hak Lee, Yong Woo Sim in 2018 (2018.01.08)
// Copyright 2018(C) CoColink Inc.
//---------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <string>
#include <algorithm>
#include <math.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "Cuda_Error.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include "Variable_Type.cuh"
#include "Parameters.cuh"

#include "class_Neighbor_Cell.cuh"
#include "class_Cuda_Particle_Array.cuh"

#include "function_init.cuh"
#include "functions_NNPS.cuh"
#include "functions_PROP.cuh"
#include "functions_KNL.cuh"
#include "functions_MASS.cuh"
#include "functions_TURB.cuh"
#include "functions_FORCE.cuh"
#include "functions_ENERGY.cuh"
#include "functions_TIME.cuh"
#include "functions_OUTPUT.cuh"
#include "functions_DIFF.cuh"
#include "functions_PST.cuh"
#include "functions_BC.cuh"

//#include "functions_PCISPH.cuh"
//#include "functions_DFSPH.cuh"

#include "WCSPH.cuh"
//#include "PCISPH.cuh"
//#include "DFSPH.cuh"


////////////////////////////////////////////////////////////////////////
int_t main()
{
	int_t vii[vii_size];
	Real vif[vif_size];
	char fn[64];
	strcpy(fn,"./input/solv.txt");

	read_solv_input(vii,vif,fn);

	switch(solver_type){
		case Wcsph:
			WCSPH(vii,vif);
			break;
		case Pcisph:
			//PCISPH(vii,vif);
			break;
		case Dfsph:
			//DFSPH(vii,vif);
			break;
		default:
			WCSPH(vii,vif);
	}

	return 0;
}
