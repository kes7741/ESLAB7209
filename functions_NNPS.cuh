////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_insert_particle_to_cell_002(int_t*vii,Real*vif,part11*Pa11,part13*Pa13)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=number_of_particles) return;

	int_t icell,jcell,kcell;

	// calculate I,J,K in cell
	if((x_max==x_min)){icell=0;}
	else{icell=fmin(floor((Pa11[idx].x-x_min)/(x_max-x_min)*NI),NI-1);}

	if((y_max==y_min)){jcell=0;}
	else{jcell=fmin(floor((Pa11[idx].y-y_min)/(y_max-y_min)*NJ),NJ-1);}

	if((z_max==z_min)){kcell=0;}
	else{kcell=fmin(floor((Pa11[idx].z-z_min)/(z_max-z_min)*NK),NK-1);}

	// out-of-range handling
	if(icell<0) icell=0;
	if(jcell<0) jcell=0;
	if(kcell<0) kcell=0;

	// save I,J,K
	Pa13[idx].I_cell=icell;
	Pa13[idx].J_cell=jcell;
	Pa13[idx].K_cell=kcell;

	// calculate cell index from I,J,K
	Pa13[idx].PID=idx;
	//CID[idx]=gpu_IJK_to_cell_idx2(icell,jcell,kcell,NI,NJ,NK);
	Pa13[idx].CID=(icell+NI*jcell+NI*NJ*kcell);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_search_neighbor_atomic(int_t*vii,Real tmp_kappa,part11*Pa11,part13*Pa13,part2*Pa2,neighbor_cell ncl,int_t*ncell,int_t*start_idx,int_t*ZID)
{
	// The number of blocks should be the same as "number of particles".
	// The number of threads should be the same as "number of neighbor cells"

	__shared__ int idx_g;			// global index (shared in block) for pnbv

	int_t ip = blockIdx.x;
	int_t ip_nc = threadIdx.x;											// neighbor cell index, thread idx: One thread is assigned to find neighbors in one neighbor cell. 

	int ip_pnbv = blockIdx.x * pnb_size;		// pnbv starting index in the global particle_array class for the given particle ID(idx): particle->pnbv

	//__shared__  uint_t cache1[150];
	//cache1[threadIdx.x]=0;
	//int_t ip=blockIdx.x+blockIdx.y*gridDim.x;					// current particle id
	//if (ip<cNUM_PARTICLES[0])
	//{
	//int_t ip_pnbv=ip*pnb_size;		// pnbv starting index in the global particle_array class for the given particle ID(idx): particle->pnbv

	//int_t pid_in_cell[40];				// array of neighbor particle indices in the cell
	//Real dist_in_cell[40];				// array of dist in the cell

	int_t idx_t;
	int_t np;									// number of particles in the cell
	//int_t nbp=0;							// number of neighbeor particles
	int_t ii;
	int_t I=Pa13[ip].I_cell+ncl.ijk[ip_nc][0];		// neiighbor cell I (different for each thread)
	int_t J=Pa13[ip].J_cell+ncl.ijk[ip_nc][1];		// neighbor cell J (different for each thread)
	int_t K=Pa13[ip].K_cell+ncl.ijk[ip_nc][2];		// neighbor cell K (different for each thread)

	int_t cell_idx;		// cell ID in 1D array

	int_t jp;				// neighbor particle id				
	Real tmp_dist;			// distance between particle ip and jp

	Real search_range=tmp_kappa*Pa11[ip].h;	// search range

	// initialize idx_g=-1. It is performed in the first thread (threadID=0). 
	if (ip_nc==0) idx_g=-1;
	__syncthreads();



	// estimate number of particles for each neighbor cell
	if((I>=0&&I<NI)&&(J>=0&&J<NJ)&&(K>=0&&K<NK)){ 		// It is done when the cell is within the proper range.
		// convert (I,J,K) to cell_idx for each thread
		cell_idx=(I+NI*J+NI*NJ*K);
		cell_idx=ZID[cell_idx];			// convert cell index to z index

		// number of particles in the cell
		np=ncell[cell_idx];

		// count number of neighbor particles within search range
		for(ii=0;ii<np;ii++){
			jp=Pa13[start_idx[cell_idx]+ii].sorted_PID;	// neighbor particle index
			// distance
			tmp_dist=sqrt((Pa11[ip].x-Pa11[jp].x)*(Pa11[ip].x-Pa11[jp].x)+(Pa11[ip].y-Pa11[jp].y)*(Pa11[ip].y-Pa11[jp].y)+(Pa11[ip].z-Pa11[jp].z)*(Pa11[ip].z-Pa11[jp].z));
		 	// If the distance is less than the search range,the particle index jp is inserted into the pnbv.
			if(tmp_dist<search_range){
				// add idx_g in shared memory and save it in the thread local memory. Each thread will have the same global index but different local index.
				//pid_in_cell[nbp]=jp;			// insert particle index
				//dist_in_cell[nbp]=dist;		// insert distance between ip and jp
				//nbp++;
				idx_t=atomicAdd(&idx_g,1)+1;

				Pa2[ip_pnbv+idx_t].pnb=jp;		// insert particle index
				Pa2[ip_pnbv+idx_t].dist=tmp_dist;		// insert distance between ip and jp
			}
		}
	}
	//cache1[threadIdx.x]=nbp;			// save the particle numbers in the cell

	__syncthreads();

	// caculate cumulative sum of cache1 (start_inx for pnv save)
	/*
	if (threadIdx.x==0)
	{
	for (int_t i=1; i <= nb_cell_list.n; i++)
	{
	//cache2[i]=cache2[i-1]+cache1[i-1];
	cache1[i]=cache1[i-1]+cache1[i];
	}
	number_of_neighbors[ip]=cache1[nb_cell_list.n-1];
	}

	__syncthreads();

	// save particle list and distance in the particle array
	uint_t idx_s=cache1[threadIdx.x]-nbp;

	for (int_t i=0; i<nbp; i++)
	{
	pnb[ip_pnbv+idx_s+i]=pid_in_cell[i];
	dist_[ip_pnbv+idx_s+i]=dist_in_cell[i];
	}
	//*/
	if (ip_nc==0) Pa11[ip].number_of_neighbors=idx_g+1;
}

////////////////////////////////////////////////////////////////////////
int partitionbykey(int*input,int*output,int p,int r)
{
	int pivot=input[r];

	while(p<r){
		while(input[p]<pivot) p++;
		while(input[r]>pivot)	r--;
		if(input[p]==input[r]){p++;}
		else if(p<r){
			int tmp=input[p];
			input[p]=input[r];
			input[r]=tmp;
			tmp=output[p];
			output[p]=output[r];
			output[r]=tmp;
		}
	}
	return r;
}
////////////////////////////////////////////////////////////////////////
void quicksortbykey(int*input,int*output,int p,int r)
{
	if(p<r){
		int j=partitionbykey(input,output,p,r);
		quicksortbykey(input,output,p,j-1);
		quicksortbykey(input,output,j+1,r);
	}
}
////////////////////////////////////////////////////////////////////////
uint64_t morton2d(uint64_t x,uint64_t y)
{
	uint64_t z=0;

	x=(x|(x<<16))&0x0000FFFF0000FFFF;
	x=(x|(x<<8))&0x00FF00FF00FF00FF;
	x=(x|(x<<4))&0x0F0F0F0F0F0F0F0F;
	x=(x|(x<<2))&0x3333333333333333;
	x=(x|(x<<1))&0x5555555555555555;
	y=(y|(y<<16))&0x0000FFFF0000FFFF;
	y=(y|(y<<8))&0x00FF00FF00FF00FF;
	y=(y|(y<<4))&0x0F0F0F0F0F0F0F0F;
	y=(y|(y<<2))&0x3333333333333333;
	y=(y|(y<<1))&0x5555555555555555;

	z=x|(y<<1);

	return z;
}
////////////////////////////////////////////////////////////////////////
// find morton 3d curve index
uint64_t morton3d(unsigned int a,unsigned int b,unsigned int c)
{
	uint64_t answer=0;

	uint64_t x=a&0x1fffff;// we only look at the first 21 bits
	x=(x|x<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	x=(x|x<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	x=(x|x<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	x=(x|x<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	x=(x|x<<2)&0x1249249249249249;

	uint64_t y=b&0x1fffff;// we only look at the first 21 bits
	y=(y|y<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	y=(y|y<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	y=(y|y<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	y=(y|y<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	y=(y|y<<2)&0x1249249249249249;

	uint64_t z=c&0x1fffff;// we only look at the first 21 bits
	z=(z|z<<32)&0x1f00000000ffff; // shift left 32 bits,OR with self,and 00011111000000000000000000000000000000001111111111111111
	z=(z|z<<16)&0x1f0000ff0000ff; // shift left 32 bits,OR with self,and 00011111000000000000000011111111000000000000000011111111
	z=(z|z<<8)&0x100f00f00f00f00f;// shift left 32 bits,OR with self,and 0001000000001111000000001111000000001111000000001111000000000000
	z=(z|z<<4)&0x10c30c30c30c30c3;// shift left 32 bits,OR with self,and 0001000011000011000011000011000011000011000011000011000100000000
	z=(z|z<<2)&0x1249249249249249;

	answer|=x|y<<1|z<<2;
	return answer;
}
////////////////////////////////////////////////////////////////////////
void cell_check_results(uint_t count,int_t nop,part13*Pa13)
{
	char FileName_xyz[256];
	sprintf(FileName_xyz,"./result_check/check_result_PID%d\n.txt",count);
	FILE*outFile_xyz;
	/*
	ofstream outFile_temp;
	ofstream outFile_p;
	ofstream outFile_rho;
	//*/
	outFile_xyz=fopen(FileName_xyz,"w");
	fprintf(outFile_xyz,"PID\tCID\tsorted PID\tsorted CID\n");

	int_t i;
	for(i=0;i<nop;i++){
		fprintf(outFile_xyz,"%d\t%d\t%d\t%d\n",Pa13[i].PID,Pa13[i].CID,Pa13[i].sorted_PID,Pa13[i].sorted_CID);
	}
	fclose(outFile_xyz);
}
////////////////////////////////////////////////////////////////////////
void make_z_index_map(int_t*map_z_index,int_t*map_cell_index,int_t nc,int_t ni,int_t nj,int_t nk,int_t tdim)
{
	int_t i,j,k;
	int_t cell_idx,z_index;
	int_t noc=nc;
	int_t tni,tnj,tnk;
	tni=ni; tnj=nj; tnk=nk;
	// make z_index_map vs. cell_index_map
	if(tdim==3){
		//clc_z_index_map_3d();
		for(k=0;k<tnk;k++){
			for(j=0;j<tnj;j++){
				for(i=0;i<tni;i++){
					//cell_idx=clc_IJK_to_cell_idx(i,j,k,tni,tnj,tnk);
					cell_idx=i+tni*j+tni*tnj*k;
					z_index=morton3d(i,j,k);

					map_cell_index[cell_idx]=cell_idx;
					map_z_index[cell_idx]=z_index;
				}
			}
		}
		// sort map_cell_index by map_z_index
		quicksortbykey(map_z_index,map_cell_index,0,noc-1);
		// modify map_z_index (to no vacancy)
		for(i=0;i<noc;i++) map_z_index[i]=i;
		// sort map_z_index by map_cell_index
		quicksortbykey(map_cell_index,map_z_index,0,noc-1);
	}else if(tdim==2){
		//clc_z_index_map_2d();
		for(j=0;j<tnj;j++){
			for(i=0;i<tni;i++){
				//cell_idx=clc_IJK_to_cell_idx(i,j,0,NI,NJ,NK);
				cell_idx=i+tni*j;
				z_index=morton2d(i,j);
				map_cell_index[cell_idx]=cell_idx;
				map_z_index[cell_idx]=z_index;
			}
		}
		// sort map_cell_index by map_z_index
		quicksortbykey(map_z_index,map_cell_index,0,noc-1);
		// modify map_z_index (to no vacancy)
		for(i=0;i<noc;i++) map_z_index[i]=i;
		// sort map_z_index by map_cell_index
		quicksortbykey(map_cell_index,map_z_index,0,noc-1);
	}else if(tdim==1){
		printf("no z_index\n");
	}
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_number_occurances(int_t array_length,part13*Pa13,int_t*count_array)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=array_length) return;

	atomicAdd(&count_array[Pa13[idx].CID],1);
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_find_start_idx(int_t*start_idx,int_t*gpu_cum_sum,int_t*gpu_ncell,int_t noc)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=noc) return;

	start_idx[idx]=gpu_cum_sum[idx]-gpu_ncell[idx];

	//start_idx[idx]=1;
	//gpu_cum_sum[idx]=3;
	//gpu_ncell[idx]=2;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_count_sort(part13*Pa13,int_t*cum_ncell,int_t nop)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=nop) return;

	int_t new_idx=atomicAdd(&cum_ncell[Pa13[idx].CID],-1)-1;

	Pa13[new_idx].sorted_CID=Pa13[idx].CID;
	Pa13[new_idx].sorted_PID=Pa13[idx].PID;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_convert_cell_to_z(int_t nop,part13*Pa13,int_t*gctz)
{
	int_t idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=nop) return;

	int_t tci=Pa13[idx].CID;

	//CID[idx]=g_array_ic_to_iz[cell_index];		// replace CID with z-index
	atomicExch(&Pa13[idx].CID,gctz[tci]);
}
////////////////////////////////////////////////////////////////////////