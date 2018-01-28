////////////////////////////////////////////////////////////////////////
__global__ void kernel_copy_max(int_t nop,part11*Pa11,Real*mux,Real*mft)
{
	uint_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

	mux[i]=Pa11[i].ux;
	mft[i]=Pa11[i].ftotal;
}
////////////////////////////////////////////////////////////////////////
void particle_check_results(uint_t tcount,int_t nop,part11*Pa11,part13*Pa13)
{
	char FileName_xyz[256];
	sprintf(FileName_xyz,"./result_check/check_result_dm%d\n.txt",tcount);
	FILE*outFile_xyz;

	/*
	ofstream outFile_xyz;
	ofstream outFile_temp;
	ofstream outFile_p;
	ofstream outFile_rho;

	outFile_xyz << "x0   " << "\t" << "y0   " << "\t" << "z0   " << "\t" <<
		"x   " << "\t" << "y   " << "\t" << "z   " << "\t" <<
		"Ic   " << "\t" << "Jc   " << "\t" << "Kc   " << "\t" <<
		"ux0   " << "\t" << "uy0   " << "\t" << "uz0   " << "\t" <<
		"ux   " << "\t" << "uy   " << "\t" << "uz   " << "\t" <<
	//"fx " << "\t" << "fy " << "\t" << "fz " << "\t" <<
		"n " << "\t" << "flt_s " << "\t" << "rho " << "\t" << "p " << endl;
	//*/
	int_t i;
	outFile_xyz=fopen(FileName_xyz,"w");
	fprintf(outFile_xyz,"x0   \ty0   \tz0   \tx   \ty   \tz   \tIc   \tJc   \tKc   \tux0   \tuy0   \tuz0   \tux   \tuy   \tuz   \tn \tflt_s \trho \tp \n");

	for(i=0;i<nop;i++){
		fprintf(outFile_xyz,"%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\n",
												Pa11[i].x0,Pa11[i].y0,Pa11[i].z0,Pa11[i].x,Pa11[i].y,Pa11[i].z,Pa13[i].I_cell,Pa13[i].J_cell,Pa13[i].K_cell,
												Pa11[i].ux0,Pa11[i].uy0,Pa11[i].uz0,Pa11[i].ux,Pa11[i].uy,Pa11[i].uz,Pa11[i].number_of_neighbors,Pa11[i].flt_s,
												Pa11[i].rho,Pa11[i].pres);
		/*
		outFile_xyz << x0[i] << "\t" << y0[i] << "\t" << z0[i] << "\t" <<
		x[i] << "\t" << y[i] << "\t" << z[i] << "\t" <<
		I_cell[i] << "\t" << J_cell[i] << "\t" << K_cell[i] << "\t" <<
		ux0[i] << "\t" << uy0[i] << "\t" << uz0[i] << "\t" <<
		ux[i] << "\t" << uy[i] << "\t" << uz[i] << "\t" <<
		//fpx[i] << "\t" << fpy[i] << "\t" << fpz[i] << "\t" <<
		number_of_neighbors[i] << "\t" << flt_s[i] << "\t" << rho[i] << "\t" << p[i] << endl;
		//*/
	}
	fclose(outFile_xyz);
}
////////////////////////////////////////////////////////////////////////
void save_pnb(char*FileName,int_t nn,int_t nop,int_t npbs,part2*Pa2)
{
	FILE*outFile;

	outFile=fopen(FileName,"w");

	int_t i,j,idx;
	int_t tnpbs=npbs;

	for(i=0;i<nop;i++){
		for(j=0;j<nn;j++){
			idx=i*tnpbs+j;
			fprintf(outFile,"%d\t",Pa2[idx].pnb);
		}
		fprintf(outFile,"\n");
	}
	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////
void save_dist(char*FileName,int_t nn,int_t nop,int_t npbs,part2*Pa2)
{
	FILE*outFile;

	outFile=fopen(FileName,"w");

	int_t i,j,idx;
	int_t tnpbs=npbs;

	for(i=0;i<nop;i++){
		for(j=0;j<nn;j++){
			idx=i*tnpbs+j;
			fprintf(outFile,"%d\t",Pa2[idx].dist);
		}
		fprintf(outFile,"\n");
	}
	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////
void save_Wij(char*FileName,int_t nn,int_t nop,int_t npbs,part2*Pa2)
{
	FILE*outFile;

	outFile=fopen(FileName,"w");

	int_t i,j,idx;
	int_t tnpbs=npbs;

	for(i=0;i<nop;i++){
		for(j=0;j<nn;j++){
			idx=i*tnpbs+j;
			fprintf(outFile,"%d\t",Pa2[idx].wij);
		}
		fprintf(outFile,"\n");
	}
	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////
void save_dWij(char*FileName,int_t nn,int_t nop,int_t npbs,part2*Pa2)
{
	FILE*outFile;

	outFile=fopen(FileName,"w");

	int_t i,j,idx;
	int_t tnpbs=npbs;

	for(i=0;i<nop;i++){
		for(j=0;j<nn;j++){
			idx=i*tnpbs+j;
			fprintf(outFile,"%d\t",Pa2[idx].dwij);
		}
		fprintf(outFile,"\n");
	}
	fclose(outFile);
}
////////////////////////////////////////////////////////////////////////
void save_plot_boundary_vtk(int_t*vii,part11*Pa11){
	// number_of_boundaries - number of fluid particles
	int_t i;
	char FileName_vtk[256];
	//Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	strcpy(FileName_vtk,"./plotdata/boundary.vtk");
	FILE*outFile_vtk;

	//If the file already exists, its contents are discarded and create the new one.
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");							// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");					// header: description of file, it never exceeds 256 characters
	fprintf(outFile_vtk,"ASCII\n");																		// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");												// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",number_of_boundaries);	// define particles position as POINTS

	// print out (x,y,z) coordinates of particles
	for(i=0;i<number_of_particles;i++){
		if(Pa11[i].p_type==0){
			fprintf(outFile_vtk,"%f\t%f\t%f\n",Pa11[i].x0,Pa11[i].y0,Pa11[i].z0);
		}
	}
	// define point coordinate as a particle position
	fprintf(outFile_vtk,"VERTICES\t%d\t%d\n",number_of_boundaries,2*number_of_boundaries);
	for(i=0;i<number_of_particles;i++){
		if(Pa11[i].p_type==0){
			// 1: POINT �� data ���� ������ 1�� / i: POINT�� index (������ numbering�� ���� �ǹ�)
			fprintf(outFile_vtk,"1\t%d\n",i);
		}
	}

	//Write data -------------------------------------------------------------------------
	//declare each point has its own data
	fprintf(outFile_vtk,"POINT_DATA\t%d\n",number_of_boundaries);

	/*
	outFile_vtk << "SCALARS" << "\t" << "index" << "\t" << "int" << endl;						 assign the index of particles at POINTS
	outFile_vtk << "LOOKUP_TABLE" << "\t" << "default" << endl;									 default
	for (int i=number_of_boundaries;i<number_of_particles;i++)
	{	outFile_vtk << i << endl;	}

	outFile_vtk << "FIELD FieldData" << "\t" << 1 << endl;										 all the data of particles are FieldData except 'index' ( 3: declare number of property data )


	outFile_vtk << "pressure" << "\t" << 1 << "\t" << number_of_boundaries << "\t" << "float" << endl;			 print out density
	for (int_t i=0;i<number_of_particles;i++)
	{
		if (p_type[i]==0)
		{
			outFile_vtk << p[i] << endl;
		}
	}

	outFile_vtk << "stiffness" << "\t" << 1 << "\t" << number_of_boundaries << "\t" << "float" << endl;		 print out pressure
	for (int i=0;i<number_of_particles;i++)
	{
		if (p_type[i]==0)
		{
			outFile_vtk << stiffness[i] << endl;
		}
	}


	outFile_vtk << "dw_c" << "\t" << 3 << "\t" << number_of_boundaries << " float" << endl;			 print out velocity
	for (int_t i=0;i<number_of_particles;i++)
	{
		if (p_type[i]==0)
		{
			outFile_vtk << dw_cx[i] << "\t" << dw_cy[i] << "\t" << dw_cz[i] << "\t" << endl;
		}
	}
	//*/
	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
void save_plot_fluid_vtk(int_t*vii,Real*vif,part11*Pa11)
{
	int_t i,nop;//,nob;
	nop=number_of_particles;
	//nob=number_of_boundaries;
	//int_t Nparticle=nop-nob;						// number of fluid particles
	int_t Nparticle=nop;									// number of fluid particles


	// Filename: It should be series of frame numbers(nameXXX.vtk) for the sake of auto-reading in PARAVIEW.
	char FileName_vtk[256];
	sprintf(FileName_vtk,"./plotdata/fluid_%dstp.vtk",count-1);
	// If the file already exists, its contents are discarded and create the new one.
	FILE*outFile_vtk;
	outFile_vtk=fopen(FileName_vtk,"w");

	fprintf(outFile_vtk,"# vtk DataFile Version 3.0\n");					// version & identifier: it must be shown.(ver 1.0/2.0/3.0)
	fprintf(outFile_vtk,"Print out results in vtk format\n");			// header: description of file, it never exceeds 256 characters
	fprintf(outFile_vtk,"ASCII\n");																// format of data (ACSII / BINARY)
	fprintf(outFile_vtk,"DATASET POLYDATA\n");										// define DATASET format: 'POLYDATA' is proper to represent SPH particles

	//Define SPH particles---------------------------------------------------------------
	fprintf(outFile_vtk,"POINTS\t%d\tfloat\n",Nparticle);					// define particles position as POINTS
	// print out (x,y,z) coordinates of particles
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"%f\t%f\t%f\t\n",Pa11[i].x0,Pa11[i].y0,Pa11[i].z0);
	}
	// define point coordinate as a particle position
	fprintf(outFile_vtk,"VERTICES\t%d\t%d\n",Nparticle,2*Nparticle);
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"1\t%d\n",i);
	}

	//Write data -------------------------------------------------------------------------
	// declare each point has its own data
	fprintf(outFile_vtk,"POINT_DATA\t%d\n",Nparticle);

	//outFile_vtk << "SCALARS" << "\t" << "index" << "\t" << "int" << endl;						// assign the index of particles at POINTS
	//outFile_vtk << "LOOKUP_TABLE" << "\t" << "default" << endl;									// default
	//for (int i=number_of_boundaries;i<number_of_particles;i++)
	//{	outFile_vtk << i << endl;	}

	// all the data of particles are FieldData except 'index' ( 3: declare number of property data )
	fprintf(outFile_vtk,"FIELD FieldData\t5\n");

	//outFile_vtk << "lbl_surf" << "\t" << 1 << "\t" << Nparticle << "\t" << "float" << endl;			// print out density
	//for (int_t i=0;i<number_of_particles;i++)
	//{
	//	if (p_type[i] > 0)
	//	{
	//		outFile_vtk << lbl_surf[i] << endl;
	//	}
	//}
	// temperature
	fprintf(outFile_vtk,"temperature\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"%f\n",Pa11[i].temp);
	}
	// pressure
	fprintf(outFile_vtk,"pressure\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"%f\n",Pa11[i].pres);
	}
	// mass
	fprintf(outFile_vtk,"p_type\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"%d\n",Pa11[i].p_type);
	}
	// print out density
	fprintf(outFile_vtk,"density_norm\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"%f\n",Pa11[i].rho/Pa11[i].rho_ref);
	}
	// volume
	fprintf(outFile_vtk,"rho_ref\t1\t%d\tfloat\n",Nparticle);
	for(i=0;i<nop;i++){
			fprintf(outFile_vtk,"%f\n",Pa11[i].rho_ref);
	}

	fclose(outFile_vtk);
}
////////////////////////////////////////////////////////////////////////
