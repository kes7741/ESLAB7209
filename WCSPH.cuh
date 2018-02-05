void WCSPH(int_t*vii,Real*vif)
{
	//-------------------------------------------------------------------------------------------------
	// ##. GPU device properties
	//-------------------------------------------------------------------------------------------------
	struct cudaDeviceProp prop;
	{
		int_t gcount,i;
		cudaGetDeviceCount(&gcount);

		for(i=0;i<gcount;i++){
			cudaGetDeviceProperties(&prop,i);
			printf("### GPU DEVICE PROPERTIES.................................\n\n");
			printf("	Name: %s\n",prop.name);
			printf("	Compute capability: %d.%d\n",prop.major,prop.minor);
			printf("	Clock rate: %d\n",prop.clockRate);
			printf("	Total global memory: %ld\n",prop.totalGlobalMem);
			printf("	Total constant memory: %d\n",prop.memPitch);
			printf("	Multiprocessor count: %d\n",prop.multiProcessorCount);
			printf("	Shared mem per block: %d\n",prop.sharedMemPerBlock);
			printf("	Registers per block: %d\n",prop.regsPerBlock);
			printf("	Threads in warp: %d\n",prop.warpSize);
			printf("	Max threads per block: %d\n",prop.maxThreadsPerBlock);
			printf("	Max thread dimensions: %d,%d,%d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
			printf("	Max grid dimensions: %d,%d,%d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
			printf("...........................................................\n\n");
		}
	}
	cudaSetDevice(1);		//device set-up

	// print ------------------------------------------------------------------------------------------
	printf(" ------------------------------------------------------------\n");
	printf(" SOPHIA_gpu v.1.0 \n");
	printf(" Developed by E.S. Kim,Y.B. Cho,S.H. Park\n");
	printf(" 2017. 02. 20 \n");
	printf("------------------------------------------------------------\n\n");
	//-------------------------------------------------------------------------------------------------

	//-------------------------------------------------------------------------------------------------
	// ##. READ SOLV & DECLARE VARIABLES
	//-------------------------------------------------------------------------------------------------

	char INPUT_FILE_NAME[128];
	char OUTPUT_FILE_NAME[128];
	strcpy(INPUT_FILE_NAME,"./input/input.txt");								// input file name and address
	strcpy(OUTPUT_FILE_NAME,"./output/output.txt");							// output file name and address

	// solution variables
	nb_cell_number=3;												// 3: 3x3,5: 5x5
	count=floor(time/dt+0.5);							// starting number of count

	//Real dt0=dt;
	Real cell_reduction_factor=1.1;
	Real search_incr_factor=1.1;				// coefficient for cell and search range (esk)

	//number_of_particles=gpu_count_particle_numbers(INPUT_FILE_NAME);	// calculating number of particles (from input file)
	number_of_particles=gpu_count_particle_numbers2(INPUT_FILE_NAME);	// calculating number of particles (from input file)
	//number_of_boundaries=gpu_count_boundary_numbers(INPUT_FILE_NAME);	// calculating number of bondary particles (from input file)

	//int_t particle_block_size=ceil(sqrt(number_of_particles));				// block size for 2-dimensional block
	//dim3 particle_blocks(particle_block_size,particle_block_size);				// 2-dimensional block

	if (nb_cell_type==1){
		cell_reduction_factor=0.7;
		search_incr_factor=1.2;
		nb_cell_number=5;
	}
	// estimate_thread_size
	int_t n_order;
	n_order=log(pnb_size)/log(2)+1;
	thread_size=pow(2,n_order);

	//int_t smsize=sizeof(Real)*thread_size;

	//-------------------------------------------------------------------------------------------------
	// ##. GENERATE PARTICLES
	//-------------------------------------------------------------------------------------------------

	// declare
	part11*host_particle_array11;
	part12*host_particle_array12;
	part13*host_particle_array13;

	part11*particle_array11;
	part12*particle_array12;
	part13*particle_array13;

	part11*particle_array_temp11;
	part12*particle_array_temp12;
	part13*particle_array_temp13;

	part2*host_particle_array2;
	part2*particle_array2;
	part2*particle_array_temp2;

	/*
	// read Csolver input file (in cpu,gpu)
	host_particle_array.read_solv(solv);
	particle_array.read_solv(solv);
	//*/

	// memory allocation & initialize
	//host_allocate(vii,vif,host_particle_array,host_particle_array2);
	int_t malloc_size=number_of_particles+10;
	int_t malloc_size2=number_of_particles*pnb_size+10;

	host_particle_array11=(part11*)malloc(malloc_size*sizeof(part11));
	host_particle_array12=(part12*)malloc(malloc_size*sizeof(part12));
	host_particle_array13=(part13*)malloc(malloc_size*sizeof(part13));
	host_particle_array2=(part2*)malloc(malloc_size2*sizeof(part2));	// o(^_^)o



	nd_ref=0;
	memset(host_particle_array11,0,sizeof(part11)*malloc_size);
	memset(host_particle_array12,0,sizeof(part12)*malloc_size);
	memset(host_particle_array13,0,sizeof(part13)*malloc_size);
	memset(host_particle_array2,0,sizeof(part2)*malloc_size2);

	// initialize to zeros (memset is not working....) -> if variable type is structure, memset is working.
	int_t i,j,k;
	for(i=0;i<malloc_size;i++){
		host_particle_array12[i].k_turb=1e-9;			// need to think later (by esk)
		host_particle_array12[i].e_turb=1e-12;			// need to think later (by esk)
		host_particle_array11[i].w_dx=1.0;
	}
	//device_allocate(vii,vif,particle_array,particle_array2);
	cudaMalloc((void**)&particle_array11,malloc_size*sizeof(part11));
	cudaMalloc((void**)&particle_array12,malloc_size*sizeof(part12));
	cudaMalloc((void**)&particle_array13,malloc_size*sizeof(part13));
	cudaMalloc((void**)&particle_array2,malloc_size2*sizeof(part2));

	if (flag_z_index==1){
		//device_allocate(vii,vif,particle_array_temp,particle_array_temp2);
		cudaMalloc((void**)&particle_array_temp11,malloc_size*sizeof(part11));
		cudaMalloc((void**)&particle_array_temp12,malloc_size*sizeof(part12));
		cudaMalloc((void**)&particle_array_temp13,malloc_size*sizeof(part13));
		cudaMalloc((void**)&particle_array_temp2,malloc_size2*sizeof(part2));
	}

	// read input file (in gpu)
	//read_input(vii,host_particle_array11,host_particle_array12);
	//read_input2(vii,host_particle_array11,host_particle_array12);
	read_input3(vii,host_particle_array11,host_particle_array12);
	//if(flag_z_index==1) particle_array_temp.read_solv(solv);// not used


	//host_particle_array.calculate_nd_ref();
	Real h_tmp=host_particle_array11[0].h;
	switch(dim){
		case 1:
			nd_ref=1.0/(2.0/3.0*h_tmp);
			break;
		case 2:
			nd_ref=1.0/((2.0/3.0*h_tmp)*(2.0/3.0*h_tmp));
			//nd_ref=3.141592*(2.*h_tmp)*(2.*h_tmp);
			break;
		case 3:
			nd_ref=1.0/((2.0/3.0*h_tmp)*(2.0/3.0*h_tmp)*(2.0/3.0*h_tmp));
			break;
		default:
			break;
	}
	//-- // copy host to device (particle_array)
	//-- // particle_array.copy_from_host(host_particle_array);

	// set up proves
		// set up proves
	Real*max_ux,*max_ft;
	cudaMalloc((void**)&max_ux,malloc_size*sizeof(Real));
	cudaMalloc((void**)&max_ft,malloc_size*sizeof(Real));
	cudaMemset(max_ux,0,malloc_size*sizeof(Real));
	cudaMemset(max_ft,0,malloc_size*sizeof(Real));
	/*
	thrust::device_ptr<Real> thrust_umag_ptr=thrust::device_pointer_cast(particle_array.ux);
	thrust::device_ptr<Real> thrust_ftotal_ptr=thrust::device_pointer_cast(particle_array.ftotal);
	//*/
	// proving variables:max_umag,max_ftotal
	Real max_umag,max_ftotal;
	Real dt_CFL, V_MAX, K_stiff, eta;		// variables for timestep control
	Real h0=host_particle_array11[0].h;	//initial kernel distance

	//-------------------------------------------------------------------------------------------------
	// ##. GENERATE CELLS
	//-------------------------------------------------------------------------------------------------
	// cell variables______
	//int_t NI,NJ,NK;																				// cell size: NI,NJ,NK
	int_t number_of_cells;																// number of cells
	Real KERNEL_DISTANCE;																	// kernel distance: h
	neighbor_cell nb_cell_list;														// kernel distance: h
	KERNEL_DISTANCE=host_particle_array11[0].h;							// kernel distance: h
	//Real*host_cell_range,*cell_range;											// cell range
	// memory allocation
	//host_cell_range=(Real*)malloc(6*sizeof(Real));
	//cudaMalloc((void **)&cell_range,6*sizeof(Real));

	// estimate simulation range
	/*
	x_min=host_particle_array.find_xmin(Xmargin_m);	host_cell_range[0]=x_min;
	x_max=host_particle_array.find_xmax(Xmargin_p);	host_cell_range[1]=x_max;
	y_min=host_particle_array.find_ymin(Ymargin_m);	host_cell_range[2]=y_min;
	y_max=host_particle_array.find_ymax(Ymargin_p);	host_cell_range[3]=y_max;
	z_min=host_particle_array.find_zmin(Zmargin_m);	host_cell_range[4]=z_min;
	z_max=host_particle_array.find_zmax(Zmargin_p);	host_cell_range[5]=z_max;
	//*/
	find_minmax(vii,vif,host_particle_array11);
	//host_cell_range[0]=x_min;	host_cell_range[1]=x_max;
	//host_cell_range[2]=y_min;	host_cell_range[3]=y_max;
	//host_cell_range[4]=z_min;	host_cell_range[5]=z_max;

	// estimate number of cells
	/*
	NI=host_particle_array.calculate_NI(cell_reduction_factor*kappa*KERNEL_DISTANCE);
	NJ=host_particle_array.calculate_NJ(cell_reduction_factor*kappa*KERNEL_DISTANCE);
	NK=host_particle_array.calculate_NK(cell_reduction_factor*kappa*KERNEL_DISTANCE);
	//*/
	Real tmp_h0=cell_reduction_factor*kappa*KERNEL_DISTANCE;
	NI=ceil((x_max-x_min)/tmp_h0);
	if(NI==0) NI=1;
	NJ=ceil((y_max-y_min)/tmp_h0);
	if(NJ==0) NJ=1;
	NK=ceil((z_max-z_min)/tmp_h0);
	if(NK==0) NK=1;
	number_of_cells=NI*NJ*NK;

	// copy particle_array from host to device
	int_t*k_vii;
	Real*k_vif;
	cudaMalloc((void**)&k_vii,sizeof(int_t)*vii_size);
	cudaMalloc((void**)&k_vif,sizeof(Real)*vif_size);
	cudaMemcpy(k_vii,vii,vii_size*sizeof(int_t),cudaMemcpyHostToDevice);
	cudaMemcpy(k_vif,vif,vif_size*sizeof(Real),cudaMemcpyHostToDevice);

	cudaMemcpy(particle_array11,host_particle_array11,malloc_size*sizeof(part11),cudaMemcpyHostToDevice);
	cudaMemcpy(particle_array12,host_particle_array12,malloc_size*sizeof(part12),cudaMemcpyHostToDevice);
	cudaMemcpy(particle_array13,host_particle_array13,malloc_size*sizeof(part13),cudaMemcpyHostToDevice);
	cudaMemcpy(particle_array2,host_particle_array2,malloc_size2*sizeof(part2),cudaMemcpyHostToDevice);

	//particle_array.copy_from_host(host_particle_array);
	if(flag_z_index==1){
		cudaMemcpy(particle_array_temp11,host_particle_array11,malloc_size*sizeof(part11),cudaMemcpyHostToDevice);
		cudaMemcpy(particle_array_temp12,host_particle_array12,malloc_size*sizeof(part12),cudaMemcpyHostToDevice);
		cudaMemcpy(particle_array_temp13,host_particle_array13,malloc_size*sizeof(part13),cudaMemcpyHostToDevice);
		cudaMemcpy(particle_array_temp2,host_particle_array2,malloc_size2*sizeof(part2),cudaMemcpyHostToDevice);
		//particle_array_temp.copy_from_host(host_particle_array);
	}

	//-------------------------------------------------------------------------------------------------
	// ##. NEIGHBOR SEARCH
	//-------------------------------------------------------------------------------------------------
	// variable
	//Cell_Index_Container Cell_Container(number_of_particles,NI,NJ,NK);
	// allocate cell container
	//Cell_Container.allocate();
	int_t*gpu_ncell,*gpu_cum_ncell,*gpu_start_idx,*g_array_ic_to_iz;
	int_t*map_cell_index,*map_z_index; //,*host_ncell,*host_cum_ncell,*host_start_idx;
	printf("Cell container has been created.\n");
	cudaMalloc((void **)&gpu_ncell,number_of_cells*sizeof(int_t));
	cudaMalloc((void **)&gpu_cum_ncell,number_of_cells*sizeof(int_t));
	cudaMalloc((void **)&gpu_start_idx,number_of_cells*sizeof(int_t));
	cudaMalloc((void **)&g_array_ic_to_iz,number_of_cells*sizeof(int_t));

	map_cell_index=(int_t*)malloc(number_of_cells*sizeof(int_t));
	map_z_index=(int_t*)malloc(number_of_cells*sizeof(int_t));
	//host_ncell=(int_t*)malloc(number_of_cells*sizeof(int_t));
	//host_cum_ncell=(int_t*)malloc(number_of_cells*sizeof(int_t));
	//host_start_idx=(int_t*)malloc(number_of_cells*sizeof(int_t));
	printf("Cell container has been initialized.\n");

	//Cell_Container.initialize(solv);
	int_t tmp_i,tmp_j;
	for(i=0;i<number_of_cells;i++) map_z_index[i]=i;

	int_t nb_cell_range=nb_cell_number;
	int_t half_cell_range=nb_cell_range*0.5;
	nb_cell_list.n=pow(nb_cell_range,dim);

	for(i=0;i<150;i++){
		for(j=0;j<3;j++){
			nb_cell_list.ijk[i][j]=0;
		}
	}
	switch(dim){
		case 1:
			for(i=0;i<nb_cell_range;i++) nb_cell_list.ijk[i][0]=i-half_cell_range;
			break;
		case 2:
			for(i=0;i<nb_cell_range;i++){
				tmp_i=i*nb_cell_range;
				for(j=0;j<nb_cell_range;j++){
					nb_cell_list.ijk[tmp_i+j][0]=i-half_cell_range;
					nb_cell_list.ijk[tmp_i+j][1]=j-half_cell_range;
				}
			}
			break;
		case 3:
			for(i=0;i<nb_cell_range;i++){
				tmp_i=i*nb_cell_range*nb_cell_range;
				for(j=0;j<nb_cell_range;j++){
					tmp_j=j*nb_cell_range;
					for(k=0;k<nb_cell_range;k++){
						nb_cell_list.ijk[tmp_i+tmp_j+k][0]=i-half_cell_range;
						nb_cell_list.ijk[tmp_i+tmp_j+k][1]=j-half_cell_range;
						nb_cell_list.ijk[tmp_i+tmp_j+k][2]=k-half_cell_range;
					}
				}
			}
			break;
		default:
			break;
	}

	cudaMemcpy(g_array_ic_to_iz,map_z_index,number_of_cells*sizeof(int_t),cudaMemcpyHostToDevice);
	//
	//Cell_Container.reset_cell_indexing_arrays();
	//KERNEL_initialize_cell<<<number_of_cells,1>>>(gpu_start_idx, gpu_ncell, number_of_cells);
	cudaMemset(gpu_start_idx,0,number_of_cells*sizeof(int_t));
	cudaMemset(gpu_ncell,0,number_of_cells*sizeof(int_t));

	// z-indexing
	if (flag_z_index==1) make_z_index_map(map_z_index,map_cell_index,number_of_cells,NI,NJ,NK,dim);
	//Cell_Container.apply_z_index_map();
	if(flag_z_index==1){
		cudaMemcpy(g_array_ic_to_iz,map_z_index,number_of_cells*sizeof(int_t),cudaMemcpyHostToDevice);
		printf("z index map has been applied!!\n");
	}else{
		printf("z index map has not beed applied. Check z-index flag!!\n");
	}

	dim3 b,t;
	dim3 b1,t1;
	t.x=256;
	b.x=(number_of_particles-1)/t.x+1;

	t1.x=256;
	b1.x=(number_of_cells-1)/t.x+1;

	// assingn particles to cell (build PID,CID)
	//particle_array.assign_to_cell(Cell_Container.PID,Cell_Container.CID);
	KERNEL_insert_particle_to_cell_002<<<b,t>>>(k_vii,k_vif,particle_array11,particle_array13);
	cudaDeviceSynchronize();

	// bin count (build ncell)
	//Cell_Container.count_bin();
	KERNEL_number_occurances<<<b,t>>>(number_of_particles,particle_array13,gpu_ncell);
	cudaDeviceSynchronize();

	// find cumulative sum (build cum_ncell)
	thrust::inclusive_scan(thrust::device,gpu_ncell,gpu_ncell+number_of_cells,gpu_cum_ncell);
	cudaDeviceSynchronize();

	// find start_idx (build start_idx)
	//Cell_Container.find_start_idx();
	KERNEL_find_start_idx<<<b1,t1>>>(gpu_start_idx,gpu_cum_ncell,gpu_ncell,number_of_cells);
	cudaDeviceSynchronize();

	// sort (build sorted_CID,sorted_PID)
	//Cell_Container.count_sort();
	KERNEL_count_sort<<<b,t>>>(particle_array13,gpu_cum_ncell,number_of_particles);
	cudaDeviceSynchronize();

	// search particles
	//particle_array.search_neighbors(Cell_Container.nb_cell_list,Cell_Container.sorted_PID,Cell_Container.gpu_ncell,Cell_Container.gpu_start_idx,Cell_Container.g_array_ic_to_iz,search_incr_factor*kappa);
	//particle_array.search_neighbors(&Cell_Container);
	//pnb,dist part2
	// number_of_neighbors,I_cell,J_cell,K_cell part

	KERNEL_search_neighbor_atomic<<<number_of_particles,nb_cell_list.n>>>(k_vii,search_incr_factor*kappa,particle_array11,particle_array13,particle_array2,nb_cell_list,gpu_ncell,gpu_start_idx,g_array_ic_to_iz);
	cudaDeviceSynchronize();

	//-------------------------------------------------------------------------------------------------
	// ##. ESTIMATE KERNEL & FILTER (INITIAL)
	//-------------------------------------------------------------------------------------------------

	//particle_array.calculate_kernel();
	calculate_kernel(vii,k_vii,particle_array11,particle_array2);
	if(pst_solve==1) calculate_w_dx(vii,particle_array11);

	//particle_array.calculate_filter(); !!check
	KERNEL_clc_filter<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
	cudaDeviceSynchronize();

	if(con_solve==1){
		//particle_array.calculate_TemptoEnthalpy();
		KERNEL_clc_TemptoEnthalpy<<<b,t>>>(number_of_particles,particle_array11,particle_array12);
		cudaDeviceSynchronize();
	}

	//-------------------------------------------------------------------------------------------------
	// ##. PRINT INITIAL STATUS
	//-------------------------------------------------------------------------------------------------
	// print out status____________
	printf("-----------------------------------------------------------\n");
	printf("Input Summary: \n");
	printf("-----------------------------------------------------------\n");
	printf("	Total number of particles=%d\n",number_of_particles);
	printf("	pnb_size=%d\n",pnb_size);
	//printf("	thread size=%d\n\n",thread_size);
	printf("	NI=%d,	NJ=%d,	NK=%d\n",NI,NJ,NK);
	printf("-----------------------------------------------------------\n\n");
	//______________________________

	// Input Check ____________
	printf("-----------------------------------------------------------\n");
	printf("Input Check: \n");
	printf("-----------------------------------------------------------\n");

	// check number of particles
	if(number_of_particles>NUM_PART){
		printf("The number of particles exceeds the maximum particle setup (=%d)\n",NUM_PART);
	}else{
		printf("Number of particles are appropriate.\n");
	}

	// check number of neighbors
	if(pnb_size>NB_SIZE){
		printf("The neighbor particle numbers exceeds the maximum setup (=%d)\n",NB_SIZE);
	}else{
		printf("Number of neighbors are appropriate.\n");
	}

	if(flag_z_index==1){printf("z-indexing is ON.\n");}
	else{printf("z-indexing is OFF.\n");}

	//host_particle_array.save_plot_boundary_vtk();
	//save_plot_boundary_vtk(vii,host_particle_array11);
	//save_plot_fluid_vtk2(vii,vif,host_particle_array11,host_particle_array12);


	printf("save boundary particle output.\n");
	printf("-----------------------------------------------------------\n\n");
	//______________________________

	//system("pause");

	// print out status____________
	{
		printf("\n");
		printf("-----------------------------\n");
		printf("Start Simultion!!\n");
		printf("-----------------------------\n");
		printf("\n");
	}

	// check results
	if(false){
		printf("dim=%d\n",dim);
		printf("check results\n\n");
		/*
		int_t*host_ncell;
		host_ncell=(int_t*)malloc(number_of_cells*sizeof(int_t));
		//*/
		//host_particle_array.copy_from_device(particle_array);
		cudaMemcpy(host_particle_array11,particle_array11,malloc_size*sizeof(part11),cudaMemcpyDeviceToHost);
		cudaMemcpy(host_particle_array12,particle_array12,malloc_size*sizeof(part12),cudaMemcpyDeviceToHost);
		cudaMemcpy(host_particle_array13,particle_array13,malloc_size*sizeof(part13),cudaMemcpyDeviceToHost);
		cudaMemcpy(host_particle_array2,particle_array2,malloc_size2*sizeof(part2),cudaMemcpyDeviceToHost);
		particle_check_results(100,number_of_particles,host_particle_array11,host_particle_array13);

		if(true){
			char tmp_name[256];
			strcpy(tmp_name,"./result_check/check_pnb_dm.txt");
			save_pnb(tmp_name,20,number_of_particles,pnb_size,host_particle_array2);

			strcpy(tmp_name,"./result_check/check_dist_dm.txt");
			save_dist(tmp_name,20,number_of_particles,pnb_size,host_particle_array2);

			strcpy(tmp_name,"./result_check/wij_dm.txt");
			save_Wij(tmp_name,20,number_of_particles,pnb_size,host_particle_array2);

			strcpy(tmp_name,"./result_check/dwij_dm.txt");
			save_dWij(tmp_name,20,number_of_particles,pnb_size,host_particle_array2);
		}
		//system("pause");
	}

	//-------------------------------------------------------------------------------------------------
	// ##. CODE MAIN LOOP
	//-------------------------------------------------------------------------------------------------
	dim3 b2,t2;
	t2.x=256;
	b2.x=(number_of_particles*pnb_size-1)/t.x+1;

	// code main	// Exception handling
	//try{
	while(time<time_end){
		//-------------------------------------------------------------------------------------------------
		// ##. PREDICTOR (Optional)
		//-------------------------------------------------------------------------------------------------

		if(time_type==Pre_Cor){
			//particle_array.predictor(dt,time);
			KERNEL_clc_predictor<<<b,t>>>(number_of_particles,dt,time,particle_array11);
			cudaDeviceSynchronize();
			if(rho_type==Continuity){
				KERNEL_clc_predictor_continuity<<<b,t>>>(number_of_particles,dt,particle_array11);
				cudaDeviceSynchronize();
			}
			if(con_solve==1){
				//particle_array.predictor_enthalpy(dt);
				KERNEL_clc_predictor_enthalpy<<<b,t>>>(number_of_particles,dt,particle_array12);
				cudaDeviceSynchronize();
			}
			if(concn_solve==1){
				//particle_array.predictor_concn(dt);
				KERNEL_clc_predictor_concn<<<b,t>>>(number_of_particles,dt,particle_array12);
				cudaDeviceSynchronize();
			}
		}
		//particle_array.update_reference_density();
		KERNEL_clc_reference_density<<<b,t>>>(number_of_particles,k_vii,particle_array11,particle_array12);
		cudaDeviceSynchronize();
		if(con_solve==1){
			//particle_array.calculate_EnthalpytoTemp();
			KERNEL_clc_EnthalpytoTemp<<<b,t>>>(number_of_particles,particle_array11,particle_array12);
			cudaDeviceSynchronize();
		}
		//-------------------------------------------------------------------------------------------------
		// ##. NEIGHBOR SEARCH
		//-------------------------------------------------------------------------------------------------

		// prepare cells
		if((count%freq_cell)==0){
			// re-indexing
			if(flag_z_index==1){
				//particle_array.reindex_to(&particle_array_temp,Cell_Container.sorted_PID);
				//particle_array_temp.reindex_to(&particle_array,Cell_Container.PID);
				KERNEL_reindex_by_pid<<<b,t>>>(number_of_particles,particle_array11,particle_array12,particle_array_temp11,particle_array_temp12,particle_array13);
				cudaDeviceSynchronize();
			}
			// reset cells (ncell,start_idx,...)
			//Cell_Container.reset_cell_indexing_arrays();
			cudaMemset(gpu_start_idx,0,number_of_cells*sizeof(int_t));
			cudaMemset(gpu_ncell,0,number_of_cells*sizeof(int_t));
			// assign particles to cells
			//particle_array.assign_to_cell(&Cell_Container);
			KERNEL_insert_particle_to_cell_002<<<b,t>>>(k_vii,k_vif,particle_array11,particle_array13);
			cudaDeviceSynchronize();
			// convert cell index to z-index
			if (flag_z_index==1){
				//Cell_Container.convert_CID_to_ZID();
				KERNEL_convert_cell_to_z<<<b,t>>>(number_of_particles,particle_array13,g_array_ic_to_iz);
				cudaDeviceSynchronize();
			}
			// bin count (build ncell)
			//Cell_Container.count_bin();
			KERNEL_number_occurances<<<b,t>>>(number_of_particles,particle_array13,gpu_ncell);
			cudaDeviceSynchronize();
			// find cumulative sum (build cum_ncell)
			//thrust::inclusive_scan(thrust::device,Cell_Container.gpu_ncell,Cell_Container.gpu_ncell+number_of_cells,Cell_Container.gpu_cum_ncell);
			thrust::inclusive_scan(thrust::device,gpu_ncell,gpu_ncell+number_of_cells,gpu_cum_ncell);
			cudaDeviceSynchronize();
			// find start_idx (build start_idx)
			//Cell_Container.find_start_idx();
			KERNEL_find_start_idx<<<b1,t1>>>(gpu_start_idx,gpu_cum_ncell,gpu_ncell,number_of_cells);
			cudaDeviceSynchronize();
			// sort (build sorted_CID,sorted_PID)
			//Cell_Container.count_sort();
			KERNEL_count_sort<<<b,t>>>(particle_array13,gpu_cum_ncell,number_of_particles);
			cudaDeviceSynchronize();
		}

		// search neighbors
		if((count%freq_cell)==0){
			//particle_array.search_neighbors(Cell_Container.nb_cell_list,Cell_Container.sorted_PID,Cell_Container.gpu_ncell,Cell_Container.gpu_start_idx,Cell_Container.g_array_ic_to_iz,search_incr_factor*kappa);
			//particle_array.search_neighbors(&Cell_Container);
			KERNEL_search_neighbor_atomic<<<number_of_particles,nb_cell_list.n>>>(k_vii,search_incr_factor*kappa,particle_array11,particle_array13,particle_array2,nb_cell_list,gpu_ncell,gpu_start_idx,g_array_ic_to_iz);
			cudaDeviceSynchronize();
		}else{
			//particle_array.calculate_dist();
			KERNEL_clc_dist<<<b2,t2>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
			cudaDeviceSynchronize();
		}


		//------------------------------------------------EOS-------------------------------------------------
		// ##. KERNEL UPDATE
		//-------------------------------------------------------------------------------------------------
		//particle_array.calculate_kernel();
			calculate_kernel(vii,k_vii,particle_array11,particle_array2);
		//particle_array.calculate_gradient_correction();
		if(kgc_solve==1){
			KERNEL_clc_gradient_correction001<<<b,t>>>(number_of_particles,pnb_size,dim,particle_array11,particle_array13,particle_array2);
			cudaDeviceSynchronize();
			KERNEL_clc_gradient_correction002<<<b2,t2>>>(number_of_particles,pnb_size,particle_array11,particle_array13,particle_array2);
			cudaDeviceSynchronize();
			if(delSPH_solve==1){
				KERNEL_copy_dWij_to_dWij_cor<<<b2,t2>>>(number_of_particles,pnb_size,particle_array2);
				cudaDeviceSynchronize();
			}
		}else if(delSPH_solve==1){
			KERNEL_clc_gradient_correction_density001<<<b,t>>>(number_of_particles,pnb_size,dim,particle_array11,particle_array13,particle_array2);
			cudaDeviceSynchronize();
			KERNEL_clc_gradient_correction_density002<<<b2,t2>>>(number_of_particles,pnb_size,particle_array11,particle_array13,particle_array2);
			cudaDeviceSynchronize();
		}

		//-------------------------------------------------------------------------------------------------
		// ##. DENSITY UPDATE
		//-------------------------------------------------------------------------------------------------
		// estimate density
		//particle_array.calculate_density();
		switch(rho_type){
			case(Mass_Sum) :
				// density estimation using mass summation method : calcuate density by mass summation scheme
				//KERNEL_clc_mass_sum<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
				KERNEL_clc_mass_sum_norm<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
				cudaDeviceSynchronize();
				break;
			case(Continuity) :
				// density estimation using continuity eq method : calculate time derivative of density
				//KERNEL_clc_continuity<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
				KERNEL_clc_continuity_norm<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
				cudaDeviceSynchronize();
				break;
			default:
				// density estimation using mass summation method
				//KERNEL_clc_mass_sum<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
				KERNEL_clc_mass_sum_norm<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
				cudaDeviceSynchronize();
				break;
		}

		if(delSPH_solve==1){
			//particle_array.calculate_density_diffusion();
			switch(delSPH_model){
				case 1:
					// Molteni delta-SPH Model
					KERNEL_clc_density_diffusion_molteni<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,soundspeed,particle_array11,particle_array2);
					cudaDeviceSynchronize();
					break;
				case 2:
					// Antuono delta-SPH Model
					KERNEL_clc_grad_density<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_clc_density_diffusion_antuono<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,soundspeed,particle_array11,particle_array2);
					cudaDeviceSynchronize();
					break;
				default:
					break;
				}
		}

		//-------------------------------------------------------------------------------------------------
		// ##. FILTER UPDATE
		//-------------------------------------------------------------------------------------------------
		// filter estimation (update)
		if((count%freq_cell)==0){
			KERNEL_clc_filter<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array2);
			cudaDeviceSynchronize();
		}
		// surface detection
		//particle_array.detect_surface();
		KERNEL_clc_surface_detect<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array13,particle_array2);
		cudaDeviceSynchronize();


		//-------------------------------------------------------------------------------------------------
		// ##. PRESSURE UPDATE
		//-------------------------------------------------------------------------------------------------
		// calculate pressure
		//particle_array.calculate_pressure();

		KERNEL_EOS<<<b,t>>>(number_of_particles,gamma,soundspeed,rho0_eos,particle_array11);
		cudaDeviceSynchronize();

		//-------------------------------------------------------------------------------------------------
		// ##. PARTICLE INTERACTIONS
		//-------------------------------------------------------------------------------------------------
		// calculate forces
		//particle_array.calculate_force();
		calculate_force(vii,vif,particle_array11,particle_array12,particle_array13,particle_array2);


		// calculate conduction
		if(con_solve==1){
			//---original//particle_array.calculate_heat_source();
			//particle_array.calculate_conduction();
			KERNEL_clc_conduction<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,dim,particle_array11,particle_array12,particle_array13,particle_array2);
			KERNEL_clc_heat_source_sink_term<<<b,t>>>(number_of_particles,pnb_size,dim,particle_array11,particle_array12,particle_array13);
			cudaDeviceSynchronize();
		}
		// calculate diffusion
		if(concn_solve==1){
			//particle_array.calculate_concn_diffusion();
			//KERNEL_clc_concn_diffusion<<<b,t>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
			KERNEL_clc_concn_diffusion<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
			cudaDeviceSynchronize();
		}


		//-------------------------------------------------------------------------------------------------
		// ##. PARTICLE TYPE SWTICH
		//-------------------------------------------------------------------------------------------------
		// use if necessary...
		//particle_array.switch_p_type();
		KERNEL_switch_p_type_MCCI_CCI<<<b,t>>>(number_of_particles,particle_array11,particle_array12);


		//-------------------------------------------------------------------------------------------------
		// ##. TIME INTEGRATION: UPDATE PROPERTY
		//-------------------------------------------------------------------------------------------------
		// update properties
		//particle_array.update_properties(dt,time,count);
		update_properties(vii,vif,particle_array11,particle_array2);

		if(con_solve==1){
			//particle_array.update_properties_enthalpy(dt,count);
			update_properties_enthalpy(vii,vif,particle_array12);
		}

		if(concn_solve==1){
			//particle_array.update_properties_concn(dt,count);
			update_properties_concn(vii,vif,particle_array12);
		}


		// turbulence model: (by esk)!!!
		if(fv_solve==1){
			switch(turbulence_model){
				case K_LM:
					/*
					particle_array.calculate_strain_rate();
					particle_array.calculate_dk_by_klm();
					particle_array.calculate_turbulence_viscosity();
					particle_array.update_turbulence_parameters(dt);
					//*/
					KERNEL_clc_strain_rate<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_clc_klm_turb<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_turb_viscosity<<<b,t>>>(number_of_particles,particle_array11,particle_array12);
					cudaDeviceSynchronize();
					KERNEL_update_turbulence<<<b,t>>>(number_of_particles,dt,particle_array11,particle_array12);
					cudaDeviceSynchronize();
					break;
				case K_E:
					/*
					particle_array.calculate_strain_rate();
					particle_array.calculate_dk_by_ke();
					particle_array.calculate_turbulence_viscosity();
					particle_array.update_turbulence_parameters(dt);
					//*/
					KERNEL_clc_strain_rate<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_clc_ke_turb<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_turb_viscosity<<<b,t>>>(number_of_particles,particle_array11,particle_array12);
					cudaDeviceSynchronize();
					KERNEL_update_turbulence<<<b,t>>>(number_of_particles,dt,particle_array11,particle_array12);
					cudaDeviceSynchronize();
				case SPS:
					//particle_array.calculate_SPS_strain_tensor();
					//particle_array.calculate_SPS_stress_tensor();
					KERNEL_clc_SPS_strain_tensor<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_clc_SPS_stress_tensor<<<b,t>>>(number_of_particles,particle_array11,particle_array12);
					cudaDeviceSynchronize();
					break;
				case HB:
					KERNEL_clc_strain_rate<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array12,particle_array2);
					cudaDeviceSynchronize();
					KERNEL_HB_viscosity<<<b,t>>>(number_of_particles,particle_array11,particle_array12);
					cudaDeviceSynchronize();
					break;
				default:
					break;
			}
		}
		// particle shifting
		if(pst_solve==1){
			//---original//particle_array.calculate_number_density();
			//particle_array.calculate_surface_normal();
			//particle_array.update_particle_shifting(dt);
			KERNEL_clc_surface_normal<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,nd_ref,particle_array11,particle_array13,particle_array2);
			cudaDeviceSynchronize();
			KERNEL_clc_particle_shifting_lind<<<number_of_particles,thread_size>>>(number_of_particles,pnb_size,particle_array11,particle_array13,particle_array2);
			cudaDeviceSynchronize();
		}

		//-------------------------------------------------------------------------------------------------
		// ##. TIME STEP CONTROL & UPDATE
		//-------------------------------------------------------------------------------------------------

		// time-step update
		time+=dt;
		++count;

		//* ------------ estimate new timestep (Goswami & Pajarola(2011))
		if((count%freq_cell)==0){		//timestep is updated every 10 steps
			kernel_copy_max<<<b,t>>>(number_of_particles,particle_array11,max_ux,max_ft);
			cudaDeviceSynchronize();
			max_umag=*(thrust::max_element(thrust::device_ptr<Real>(max_ux),thrust::device_ptr<Real>(max_ux+number_of_particles)));
			max_ftotal=*(thrust::max_element(thrust::device_ptr<Real>(max_ft),thrust::device_ptr<Real>(max_ft+number_of_particles)));

			//CFL timestep limit
			dt_CFL=0.4*h0/soundspeed;												//CFL limit
			V_MAX=max_umag+sqrtf(h0*max_ftotal);						//V_MAX
			K_stiff=soundspeed*soundspeed*rho0_eos/gamma;		//K stiffness update_turbulence_parameters
			eta=K_to_eta(K_stiff);

			//timestep-control
			if(flag_timestep_update==1)
			{
				if(V_MAX<0.1*soundspeed)
				{
					dt=eta*dt_CFL;
				}
				else
				{
					dt=dt_CFL;
				}
			}
		}
		//*/

		//-------------------------------------------------------------------------------------------------
		// ##. Print Output Files
		//-------------------------------------------------------------------------------------------------
		if(((count-1)%freq_output)==0){
			// copy device to host
			//host_particle_array.copy_from_device(particle_array);
			cudaMemcpy(host_particle_array11,particle_array11,malloc_size*sizeof(part11),cudaMemcpyDeviceToHost);
			cudaMemcpy(host_particle_array12,particle_array12,malloc_size*sizeof(part12),cudaMemcpyDeviceToHost);
			cudaMemcpy(host_particle_array13,particle_array13,malloc_size*sizeof(part13),cudaMemcpyDeviceToHost);
			cudaMemcpy(host_particle_array2,particle_array2,malloc_size2*sizeof(part2),cudaMemcpyDeviceToHost);
			// save *.txt
			//host_particle_array.save_plot_xyz(time-dt,count-1);
			// save *.vtk files
			//host_particle_array.save_plot_fluid_vtk(time-dt,count-1);
			//save_plot_fluid_vtk(vii,vif,host_particle_array11);
			save_plot_fluid_vtk2(vii,vif,host_particle_array11,host_particle_array12);

			printf("time=%f [sec]\tcount=%d [step]\n",time-dt,count-1);
			printf("dt_CFL=%f [sec]\tdt=%f [sec]\n",dt_CFL,dt);
			printf("max_umag=%f\tV_MAX=%f\t\tmax_ftotal=%f\n\n",max_umag,V_MAX,max_ftotal);

		}
	}

	//-------------------------------------------------------------------------------------------------
	// ##. Save Restart File
	//-------------------------------------------------------------------------------------------------
	// Save restart files
	if(false){
		// Save Restart File
		save_restart(vii,vif,host_particle_array11,host_particle_array12,host_particle_array13);
	}

	/*
	}
	catch (const char* ex)	// Exception handling
	{
		cout << "exception handling: " << ex << endl;
		cout << "time=" << time << " [sec]" << endl;
		cout << "saving restart files..." << endl << endl;

		// Save Restart Files
	}
	//*/

	//-------------------------------------------------------------------------------------------------
	// ##. Memory Release
	//-------------------------------------------------------------------------------------------------
	free(host_particle_array11);
	free(host_particle_array12);
	free(host_particle_array13);
	free(host_particle_array2);	// o(^_^)o
	//free(host_cell_range);
	free(map_cell_index);
	free(map_z_index);
	//free(host_ncell);
	//free(host_cum_ncell);
	//free(host_start_idx);

	cudaFree(particle_array11);
	cudaFree(particle_array12);
	cudaFree(particle_array13);
	cudaFree(particle_array2);
	cudaFree(particle_array_temp11);
	cudaFree(particle_array_temp12);
	cudaFree(particle_array_temp13);
	cudaFree(particle_array_temp2);
	//cudaFree(cell_range);
	cudaFree(k_vii);
	cudaFree(k_vif);
	cudaFree(gpu_ncell);
	cudaFree(gpu_cum_ncell);
	cudaFree(gpu_start_idx);
	cudaFree(g_array_ic_to_iz);

	//cout << endl << "Memory Leakage (Yes(1):NO(0)): " << _CrtDumpMemoryLeaks() << endl << endl;		// checking memory leaks
	//system("pause");
}
