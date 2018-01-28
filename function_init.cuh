// variable define
//
#define		solver_type							vii[0]		// solver type: WCSPH/PCISPH/DFSPH
#define		dim											vii[1]		// dimension
#define		pnb_size								vii[2]		// maximum number of neighbors
#define		kernel_type							vii[3]		// kernel type
#define		flt_type								vii[4]		// filter type
#define		rho_type								vii[5]		// density calcuation type
#define		time_type								vii[6]		// time stepping type
#define		fluid_type							vii[7]		// fluid type
#define		material_type						vii[8]		// material type(Water / Metal)
#define		simulation_type					vii[9]		// simulation type(single_phase / two_phase)
#define		freq_cell								vii[10]		// cell initialization frequency	
#define		flag_z_index						vii[11]		// flag for z-indexing
#define		flag_timestep_update		vii[12]		// flag for varying timestep update
#define		nb_cell_type						vii[13]		// neighbor cell type
#define		freq_filt								vii[14]		// filtering frequency
#define		freq_mass_sum						vii[15]		// mass summation frequency
#define		freq_temp								vii[16]		// temperature filtering frequency
#define		freq_output							vii[17]		// output frequency
#define		fp_solve								vii[18]		// solve pressure force ?
#define		fv_solve								vii[19]		// sovle viscous force ?
#define		fva_solve								vii[20]		// solve artificial viscous force ?
#define		fg_solve								vii[21]		// solve gravity force ?
#define		fs_solve								vii[22]		// solve surface tension force ?
#define		fb_solve								vii[23]		// solve boundary force ?
#define		con_solve								vii[24]		// solve conduction?
#define		boussinesq_solve				vii[25]		// solve boussinesq approximation based natural convection?
#define		interface_solve					vii[26]		// solve interface sharpness force?
#define		surf_model							vii[27]		// surface tension model
#define		xsph_solve							vii[28]		// solve xsph ? 
#define		kgc_solve								vii[29]		// solve kernel gradient correction ?
#define		delSPH_solve						vii[30]		// solve delta-SPH ?
#define		delSPH_model						vii[31]		// delta SPH model
#define		pst_solve								vii[32]		// solve particle shifting ?
#define		turbulence_model				vii[33]		// turbulence model	(by esk)
#define		concn_solve							vii[34]		// concentration diffusion model(PSH)
//
//psh:: ISPH input
#define		minIteration						vii[35]		// minimum number of PCISPH iteration
#define		maxIteration						vii[36]		// maximum number of PCISPH iteration
//
//solution variables
#define		nb_cell_number					vii[37]
#define		count										vii[38]
#define		number_of_particles			vii[39]
#define		number_of_boundaries		vii[40]
#define		thread_size							vii[41]
#define		NI											vii[42]
#define		NJ											vii[43]
#define		NK											vii[44]
//
#define		kappa										vif[0]		// k in k*h
#define		p_ref										vif[1]		// reference pressure(for EOS)
#define		gamma										vif[2]		// gamma
#define		dt											vif[3]		// time-step(s)
#define		time										vif[4]		// time(s)
#define		time_end								vif[5]
// margin for simulation range
#define		Xmargin_m								vif[6]
#define		Xmargin_p								vif[7]
#define		Ymargin_m								vif[8]
#define		Ymargin_p								vif[9]
#define		Zmargin_m								vif[10]
#define		Zmargin_p								vif[11]
//
#define		u_limit									vif[12]
#define		c_xsph									vif[13]		// coefficient for XSPH
#define		c_repulsive							vif[14]		// coefficient for repulsive boundary force
//
//psh:: ISPH input
#define		drho_th									vif[15]		// density convergence criterion
#define		dp_th										vif[16]		// pressure convergence criterion
#define		p_relaxation						vif[17]		// relaxation factor for PCISPH pressure 
//
//solution variables
#define		x_min										vif[18]
#define		x_max										vif[19]
#define		y_min										vif[20]
#define		y_max										vif[21]
#define		z_min										vif[22]
#define		z_max										vif[23]
#define		nd_ref									vif[24]
#define		soundspeed								vif[25]
////////////////////////////////////////////////////////////////////////
void read_solv_input(int_t*vii,Real*vif,const char*FileName)
{
	solver_type=Wcsph;

	fp_solve=0;
	fv_solve=0;
	fva_solve=0;
	fg_solve=0;
	fs_solve=0;
	fb_solve=0;
	con_solve=0;
	boussinesq_solve=0;
	interface_solve=0;
	xsph_solve=0;
	kgc_solve=0;
	delSPH_solve=0;
	pst_solve=0;
	turbulence_model=0;
	concn_solve=0;
	c_xsph=0.0;
	c_repulsive=0.0;

	char inputString[1000];

	FILE*fd;
	//inFile.open("../Result/output.txt");
	fd=fopen(FileName,"r");

	int end;

	while(1){
		end=fscanf(fd,"%s",&inputString);		// reading one data from cc
		if(strcmp(inputString,"solver_type")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"WCSPH")==0) solver_type=Wcsph;
			if(strcmp(inputString,"PCISPH")==0) solver_type=Pcisph;
			if(strcmp(inputString,"DFSPH")==0) solver_type=Dfsph;
		}
		if(strcmp(inputString,"dimension(1/2/3):")==0){
			fscanf(fd,"%s",&inputString);
			dim=atoi(inputString);
		}
		if(strcmp(inputString,"max_number_of_neighbors:")==0){
			fscanf(fd,"%s",&inputString);
			pnb_size=atoi(inputString);
		}
		if(strcmp(inputString,"kernel")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Quartic")==0) kernel_type=Quartic;
			if(strcmp(inputString,"Gaussian")==0)kernel_type=Gaussian;
			if(strcmp(inputString,"Quintic")==0) kernel_type=Quintic;
			if(strcmp(inputString,"Wendland2")==0) kernel_type=Wendland2;
			if(strcmp(inputString,"Wendland4")==0) kernel_type=Wendland4;
			if(strcmp(inputString,"Wendland6")==0) kernel_type=Wendland6;
		}
		if(strcmp(inputString,"filter")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Shepard")==0) flt_type=Shepard;
			if(strcmp(inputString,"MLS")==0) flt_type=MLS;
		}
		if(strcmp(inputString,"density")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Direct")==0) rho_type=Mass_Sum;
			if(strcmp(inputString,"Continuity")==0) rho_type=Continuity;
		}
		if(strcmp(inputString,"time")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Euler")==0) time_type=Euler;
			if(strcmp(inputString,"Predictor_Corrector")==0) time_type=Pre_Cor;
		}
		if(strcmp(inputString,"fluid")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Liquid")==0) fluid_type=Liquid;
			if(strcmp(inputString,"Gas")==0) fluid_type=Gas;
		}
		if(strcmp(inputString,"simulation")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Single_Phase")==0) simulation_type=Single_Phase;
			if(strcmp(inputString,"Two_Phase")==0) simulation_type=Two_Phase;
		}
		if(strcmp(inputString,"material")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Water")==0) material_type=Water;
			if(strcmp(inputString,"Metal")==0) material_type=Metal;
		}
		if(strcmp(inputString,"pressure-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fp_solve=1;
			if(strcmp(inputString,"NO")==0) fp_solve=0;
		}
		if(strcmp(inputString,"viscous-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fv_solve=1;
			if(strcmp(inputString,"NO")==0) fv_solve=0;
		}
		if(strcmp(inputString,"turbulence-model")==0){		//(by esk)
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Laminar")==0) turbulence_model=0;
			if(strcmp(inputString,"k-lm")==0) turbulence_model=1;
			if(strcmp(inputString,"k-e")==0) turbulence_model=2;
			if(strcmp(inputString,"SPS")==0) turbulence_model=3;
		}
		if(strcmp(inputString,"artificial-viscous-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fva_solve=1;
			if(strcmp(inputString,"NO")==0) fva_solve=0;
		}
		if(strcmp(inputString,"gravitational-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fg_solve=1;
			if(strcmp(inputString,"NO")==0) fg_solve=0;
		}
		if(strcmp(inputString,"surface-tension-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fs_solve=1;
			if(strcmp(inputString,"NO")==0) fs_solve=0;
		}
		if(strcmp(inputString,"surface-tension-model")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Potential")==0) surf_model=1;
			if(strcmp(inputString,"Curvature")==0) surf_model=2;
		}
		if(strcmp(inputString,"interface-sharpness-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) interface_solve=1;
			if(strcmp(inputString,"NO")==0) interface_solve=0;
		}
		if(strcmp(inputString,"boundary-force")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) fb_solve=1;
			if(strcmp(inputString,"NO")==0) fb_solve=0;
		}
		if(strcmp(inputString,"Conduction(YES/NO):")==0){
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) con_solve=1;
			if(strcmp(inputString,"NO")==0) con_solve=0;
		}
		if(strcmp(inputString,"Boussinesq-natural-convection")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) boussinesq_solve=1;
			if(strcmp(inputString,"NO")==0) boussinesq_solve=0;
		}
		if(strcmp(inputString,"Concentration-diffusion(YES/NO):")==0){
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) concn_solve=1;
			if(strcmp(inputString,"NO")==0) concn_solve=0;
		}
		if(strcmp(inputString,"XSPH")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) xsph_solve=1;
			if(strcmp(inputString,"NO")==0) xsph_solve=0;
		}
		if(strcmp(inputString,"kernel-gradient-correction")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) kgc_solve=1;
			if(strcmp(inputString,"NO")==0) kgc_solve=0;
		}
		if(strcmp(inputString,"delta-SPH")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) delSPH_solve=1;
			if(strcmp(inputString,"NO")==0) delSPH_solve=0;
		}
		if(strcmp(inputString,"delta-SPH-model")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"Molteni")==0) delSPH_model=1;
			if(strcmp(inputString,"Antuono")==0) delSPH_model=2;
		}
		if(strcmp(inputString,"particle-shifting")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) pst_solve=1;
			if(strcmp(inputString,"NO")==0) pst_solve=0;
		}
		if(strcmp(inputString,"reference-pressure")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			p_ref=atof(inputString);
		}
		if(strcmp(inputString,"sound-speed:")==0){
			fscanf(fd,"%s",&inputString);
			soundspeed=atof(inputString);
		}		
		if(strcmp(inputString,"gamma")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			gamma=atof(inputString);
		}
		if(strcmp(inputString,"kappa:")==0){
			fscanf(fd,"%s",&inputString);
			kappa=atof(inputString);
		}
		if(strcmp(inputString,"XSPH-coefficient:")==0){
			fscanf(fd,"%s",&inputString);
			c_xsph=atof(inputString);
		}
		if(strcmp(inputString,"Boundary-coefficient:")==0){
			fscanf(fd,"%s",&inputString);
			c_repulsive=atof(inputString);
		}
		if(strcmp(inputString,"minimum-iteration:")==0){
			fscanf(fd,"%s",&inputString);
			minIteration=atoi(inputString);
		}
		if(strcmp(inputString,"maximum-iteration:")==0){
			fscanf(fd,"%s",&inputString);
			maxIteration=atoi(inputString);
		}
		if(strcmp(inputString,"pressure-convergence")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			dp_th=atof(inputString);
		}
		if(strcmp(inputString,"density-convergence")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			drho_th=atof(inputString);
		}
		if(strcmp(inputString,"pressure-relaxation")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			p_relaxation=atof(inputString);
		}
		if(strcmp(inputString,"time-step")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			dt=atof(inputString);
		}
		if(strcmp(inputString,"start-time")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			time=atof(inputString);
		}
		if(strcmp(inputString,"end-time")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			time_end=atof(inputString);
		}
		if(strcmp(inputString,"cell-initialization")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_cell=atoi(inputString);
		}
		if(strcmp(inputString,"z-indexing")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) flag_z_index=1;
			if(strcmp(inputString,"NO")==0) flag_z_index=0;
		}
		if(strcmp(inputString,"neighbor")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"3X3")==0) nb_cell_type=0;
			if(strcmp(inputString,"5X5")==0) nb_cell_type=1;
		}
		if(strcmp(inputString,"timestep")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			if(strcmp(inputString,"YES")==0) flag_timestep_update=1;
			if(strcmp(inputString,"NO")==0) flag_timestep_update=0;
		}
		if(strcmp(inputString,"filtering")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_filt=atoi(inputString);
		}
		if(strcmp(inputString,"density-renormalization")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_mass_sum=atoi(inputString);
		}
		if(strcmp(inputString,"temperature-filtering")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_temp=atoi(inputString);
		}
		if(strcmp(inputString,"plot-output")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			freq_output=atoi(inputString);
		}
		if(strcmp(inputString,"velocity-limit")==0){
			fscanf(fd,"%s",&inputString);
			fscanf(fd,"%s",&inputString);
			u_limit=atof(inputString);
		}
		if(strcmp(inputString,"Xmargin(-):")==0){
			fscanf(fd,"%s",&inputString);
			Xmargin_m=atof(inputString);
		}
		if(strcmp(inputString,"Xmargin(+):")==0){
			fscanf(fd,"%s",&inputString);
			Xmargin_p=atof(inputString);
		}
		if(strcmp(inputString,"Ymargin(-):")==0){
			fscanf(fd,"%s",&inputString);
			Ymargin_m=atof(inputString);
		}
		if(strcmp(inputString,"Ymargin(+):")==0){
			fscanf(fd,"%s",&inputString);
			Ymargin_p=atof(inputString);
		}
		if(strcmp(inputString,"Zmargin(-):")==0){
			fscanf(fd,"%s",&inputString);
			Zmargin_m=atof(inputString);
		}
		if(strcmp(inputString,"Zmargin(+):")==0){
			fscanf(fd,"%s",&inputString);
			Zmargin_p=atof(inputString);
		}
		if(end==-1) break;
	}
	fclose(fd);
}
////////////////////////////////////////////////////////////////////////
// function calculating number of particles from input file
int_t gpu_count_particle_numbers(const char*FileName)
{
	int_t idx=0;
	int_t tmp,end;

	FILE*inFile;
	//inFile.open("../Result/output.txt");
	inFile=fopen(FileName,"r");

	while(1){
		end=fscanf(inFile,"%d\n",&tmp);
		if(end==-1) break;
		idx+=1;
	}
	//	cout << "number_of_particles = " << idx << endl;
	fclose(inFile);
	return idx;
}
////////////////////////////////////////////////////////////////////////
// function calculating number of particles from input file
int_t gpu_count_particle_numbers2(const char*FileName)
{
	int_t idx=0;
	int_t nop;
    char buffer[1000];


	FILE*inFile;

	inFile=fopen(FileName,"r");
	while(fgets(buffer,1000,inFile)!=NULL) {
		idx+=1;
	}
	fclose(inFile);
  	
  	nop=idx-1;
	
	return nop;
}
////////////////////////////////////////////////////////////////////////
// function calculating number of boundary particles from input file
int_t gpu_count_boundary_numbers(const char*FileName)
{
	FILE*fd;
	fd=fopen("./input/p_type.txt","r");
	int_t end;
	int_t nb = 0;		// number of boundary particles

	int tmp;

	// calculation of number of boundary particles
	while(1){
		end=fscanf(fd,"%d\n",&tmp);
		if(end==-1) break;
		else if(tmp==0) nb+=1;
	}
	nb=nb;				// actual number of bondary particles

	return nb;
}
/////////////////////////////////////////////////////////////
#define inp_x   1
#define inp_y   2
#define inp_z   3
#define inp_ux  4
#define inp_uy  5
#define inp_uz  6
#define inp_m   7
#define inp_ptype 8
#define inp_h 9
#define inp_temp  10
#define inp_pres  11
#define inp_rho 12
#define inp_rhoref  13
#define inp_ftotal  14
#define inp_concn 15
#define inp_cc 16
#define inp_vist  17
#define inp_ct_boundary 18
#define inp_hf_boundary 19
#define inp_lbl_surf  20
#define inp_drho  21
#define inp_denthalpy 22
#define inp_dconcn  23
#define inp_dk  24
#define inp_de  25
////////////////////////////////////////////////////////////////////////
void read_input2(int_t*vii,part11*Pa11,part12*Pa12)
{
  #define buffer_size 1000

  char *FileName="./input/input.txt";
  char buffer[buffer_size];

	int idx=0;
  int i,j,tmp,end;
  int nod,nov,nop;  //number of data, number of variables, number of partices

  FILE*inFile;


  // count number of data_________________
  inFile=fopen(FileName,"r");
	while(1){
		end=fscanf(inFile,"%s\n",buffer);
    //printf("%s",buffer);
		if(end==-1) break;
		idx+=1;
	}
	fclose(inFile);
  nod=idx;
  //printf("\nnod=%d\n",nod);


  // count number of particles_________________
  idx=0;
  inFile=fopen(FileName,"r");
	while(fgets(buffer,buffer_size,inFile)!=NULL)
  {
		idx+=1;
	}
	fclose(inFile);
  nop=idx-1;
  //printf("nop=%d\n",nop);


  // number of variables_________________
  nov=nod/(nop+1);
  //printf("nov=%d\n",nov);


  // memory variable number_________________
  int lbl_var[100];

  inFile=fopen(FileName,"r");

  for(i=0;i<nov;i++)
  {
    fscanf(inFile,"%s\n",buffer);

    lbl_var[i]=atoi(buffer);

  }

  //for(i=0;i<nov;i++)
  //{
   // printf("%d\n",lbl_var[i]);
  //}

  fclose(inFile);


  // read data_________________
  //float A[100],B[100];
  //int C[100];

  inFile=fopen(FileName,"r");

  fgets(buffer,buffer_size,inFile);

  for(i=0;i<nop;i++)
  {
    for(j=0;j<nov;j++)
    {
      fscanf(inFile,"%s\n",buffer);

      switch(lbl_var[j])
      {
        case inp_x:
          //A[i]=atof(buffer);
          Pa11[i].x=atof(buffer);
          Pa11[i].x0=atof(buffer);
          break;
        case inp_y:
          Pa11[i].y=atof(buffer);
          Pa11[i].y0=atof(buffer);
          break;
        case inp_z:
          Pa11[i].z=atof(buffer);          
          Pa11[i].z0=atof(buffer);          
          break;
        case inp_ux:
          //C[i]=atof(buffer);
          Pa11[i].ux=atof(buffer);    
          Pa11[i].ux0=atof(buffer);    
          break;
        case inp_uy:
          Pa11[i].uy=atof(buffer);        
          Pa11[i].uy0=atof(buffer);        
          break;
        case inp_uz:
          Pa11[i].uz=atof(buffer);
          Pa11[i].uz0=atof(buffer);
          break;
        case inp_m:
          Pa11[i].m=atof(buffer);
          break;
        case inp_ptype:
          //C[i]=atoi(buffer);
          Pa11[i].p_type=atoi(buffer);
          break;
        case inp_h:
          Pa11[i].h=atof(buffer);
          break;
        case inp_temp:
          Pa11[i].temp=atof(buffer);
          break;
        case inp_pres:
          //B[i]=atof(buffer);
          Pa11[i].pres=atof(buffer);
          break;
        case inp_rho:
          Pa11[i].rho=atof(buffer);
          Pa11[i].rho0=atof(buffer);
          break;
        case inp_ftotal:
          Pa11[i].ftotal=atof(buffer);
          break;
        case inp_concn:
          Pa12[i].concn=atof(buffer);
          Pa12[i].concn0=atof(buffer);
          break;
        case inp_vist:
          Pa12[i].vis_t=atof(buffer);
          break;
        case inp_ct_boundary:
          Pa12[i].ct_boundary=atoi(buffer);
          break;
        case inp_lbl_surf:
          //Pa13[i].lbl_surf=atoi(buffer);
          break;
        case inp_drho:
          Pa11[i].drho=atof(buffer);
          break;
        case inp_denthalpy:
          Pa12[i].denthalpy=atof(buffer);
          break;
        case inp_dconcn:
          Pa12[i].dconcn=atof(buffer);
          break;
        case inp_dk:
          Pa12[i].dk_turb=atof(buffer);
          break;
        case inp_de:
          Pa12[i].de_turb=atof(buffer);
          break;
        default:
          printf("undefined variable name");
        break;
      }
    }
  }

  fclose(inFile);

  printf("Input Files have been sucessfully read!!\n");
}
////////////////////////////////////////////////////////////////////////
void read_input3(int_t*vii,part11*Pa11,part12*Pa12)
{
  #define buffer_size 1000

  char *FileName="./input/input.txt";
  char buffer[buffer_size];
  char *tok;    //token of string

  int i,tmp;
  int nov,nop;   //number of data, number of variables, number of partices
  int j, end;
  int lbl_var[100];
  nov = 0; 
  nop = 0;
 
  FILE*inFile;
  inFile=fopen(FileName,"r");

  // count number of variables_________________  
  fgets(buffer, buffer_size - 1, inFile);		// read first line 
  tok = strtok(buffer, "\t");					// line segmentation
  while(tok != NULL){
	  tmp = atoi(tok);
	  lbl_var[nov] = tmp;
	  nov++;									// count number of segments(variables)
	  tok = strtok(NULL,"\t");
  }
 
  // read data_________________
  while(1){
	for(j=0;j<nov;j++){
	  end = fscanf(inFile,"%s\n",buffer);
	  if (end == -1) break;
	  switch(lbl_var[j]){
		case inp_x:
		  //A[i]=atof(buffer);
		  Pa11[nop].x=atof(buffer);
		  Pa11[nop].x0=atof(buffer);
		  break;
		case inp_y:
		  Pa11[nop].y=atof(buffer);
		  Pa11[nop].y0=atof(buffer);
		  break;
		case inp_z:
		  Pa11[nop].z=atof(buffer);          
		  Pa11[nop].z0=atof(buffer);          
		  break;
		case inp_ux:
		  //C[i]=atof(buffer);
		  Pa11[nop].ux=atof(buffer);    
		  Pa11[nop].ux0=atof(buffer);    
		  break;
		case inp_uy:
		  Pa11[nop].uy=atof(buffer);        
		  Pa11[nop].uy0=atof(buffer);        
		  break;
		case inp_uz:
		  Pa11[nop].uz=atof(buffer);
		  Pa11[nop].uz0=atof(buffer);
		  break;
		case inp_m:
		  Pa11[nop].m=atof(buffer);
		  break;
		case inp_ptype:
		  //C[i]=atoi(buffer);
		  Pa11[nop].p_type=atoi(buffer);
		  break;
		case inp_h:
		  Pa11[nop].h=atof(buffer);
		  break;
		case inp_temp:
		  Pa11[nop].temp=atof(buffer);
		  break;
		case inp_pres:
		  //B[i]=atof(buffer);
		  Pa11[nop].pres=atof(buffer);
		  break;
		case inp_rho:
		  Pa11[nop].rho=atof(buffer);
		  Pa11[nop].rho0=atof(buffer);
		  break;
		case inp_ftotal:
		  Pa11[nop].ftotal=atof(buffer);
		  break;
		case inp_concn:
		  Pa12[nop].concn=atof(buffer);
		  Pa12[nop].concn0=atof(buffer);
		  break;
		case inp_vist:
		  Pa12[nop].vis_t=atof(buffer);
		  break;
		case inp_ct_boundary:
		  Pa12[nop].ct_boundary=atoi(buffer);
		  break;
		case inp_lbl_surf:
		  //Pa13[i].lbl_surf=atoi(buffer);
		  break;
		case inp_drho:
		  Pa11[nop].drho=atof(buffer);
		  break;
		case inp_denthalpy:
		  Pa12[nop].denthalpy=atof(buffer);
		  break;
		case inp_dconcn:
		  Pa12[nop].dconcn=atof(buffer);
		  break;
		case inp_dk:
		  Pa12[nop].dk_turb=atof(buffer);
		  break;
		case inp_de:
		  Pa12[nop].de_turb=atof(buffer);
		  break;
		default:
		  printf("undefined variable name");
		  break;
	  }
	}
	if(end==-1) break;
	nop++;
  }
  fclose(inFile);
  //number_of_particles = nop;
  printf("Input Files have been sucessfully read!!\n");
}
////////////////////////////////////////////////////////////////////////
void find_minmax(int_t*vii,Real*vif,part11*Pa11)
{
	int_t i;
	int_t end=number_of_particles-1;

	Real min_x=Pa11[0].x;	Real max_x=Pa11[0].x;
	Real min_y=Pa11[0].y;	Real max_y=Pa11[0].y;
	Real min_z=Pa11[0].z;	Real max_z=Pa11[0].z;
	//
	Real tmp_x,tmp_y,tmp_z;

	for(i=0;i<end;i++){
		tmp_x=Pa11[i].x;
		tmp_y=Pa11[i].y;
		tmp_z=Pa11[i].z;

		if(tmp_x<min_x) min_x=tmp_x;
		if(tmp_x>max_x) max_x=tmp_x;
		if(tmp_y<min_y) min_y=tmp_y;
		if(tmp_y>max_y) max_y=tmp_y;
		if(tmp_z<min_z) min_z=tmp_z;
		if(tmp_z>max_z) max_z=tmp_z;
	}

	min_x-=(max_x-min_x)*Xmargin_m;
	max_x+=(max_x-min_x)*Xmargin_p;
	min_y-=(max_y-min_y)*Ymargin_m;
	max_y+=(max_y-min_y)*Ymargin_p;
	min_z-=(max_z-min_z)*Zmargin_m;
	max_z+=(max_z-min_z)*Zmargin_p;

	x_min=min_x;
	x_max=max_x;
	y_min=min_y;
	y_max=max_y;
	z_min=min_z;
	z_max=max_z;
}
////////////////////////////////////////////////////////////////////////
__global__ void KERNEL_reindex_by_pid(int_t nop,part11*Pa11,part12*Pa12,part11*Pa_tmp11,part12*Pa_tmp12,part13*Pa13)
{
	int_t i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i>=nop) return;

 	// index of particle id in PID array
	int_t i_in=Pa13[i].sorted_PID;

	Pa_tmp11[i]=Pa11[i_in];
	Pa_tmp12[i]=Pa12[i_in];
	/*
	x_o[i]=x_i[i_in];
	y_o[i]=y_i[i_in];
	z_o[i]=z_i[i_in];

	x0_o[i]=x0_i[i_in];
	y0_o[i]=y0_i[i_in];
	z0_o[i]=z0_i[i_in];

	m_o[i]=m_i[i_in];

	ux_o[i]=ux_i[i_in];
	uy_o[i]=uy_i[i_in];
	uz_o[i]=uz_i[i_in];

	ux0_o[i]=ux0_i[i_in];
	uy0_o[i]=uy0_i[i_in];
	uz0_o[i]=uz0_i[i_in];

	p_o[i]=p_i[i_in];
	rho_o[i]=rho_i[i_in];
	rho0_o[i]=rho0_i[i_in];
	//rho_ref_o[i]=rho_ref_i[i_in];

	grad_rhox_o[i]=grad_rhox_i[i_in];
	grad_rhoy_o[i]=grad_rhoy_i[i_in];
	grad_rhoz_o[i]=grad_rhoz_i[i_in];

	//vis_o[i]=vis_i[i_in];
	//sigma_o[i]=sigma_i[i_in];
	//kc_o[i]=kc_i[i_in];
	//cp_o[i]=cp_i[i_in];

	temp_o[i]=temp_i[i_in];
	//temp0_o[i]=temp0_i[i_in];
	//dtemp_o[i]=dtemp_i[i_in];

	h_o[i]=h_i[i_in];
	//qw_o[i]=qw_i[i_in];

	cc_o[i]=cc_i[i_in];
	//c_o[i]=c_i[i_in];
	curv_o[i]=curv_i[i_in];

	flt_s_o[i]=flt_s_i[i_in];
	number_of_neighbors_o[i]=number_of_neighbors_i[i_in];

	drho_o[i]=drho_i[i_in];

	p_type_o[i]=p_type_i[i_in];
	hf_boundary_o[i]=hf_boundary_i[i_in];
	ct_boundary_o[i]=ct_boundary_i[i_in];

	ftotalx_o[i]=ftotalx_i[i_in];
	ftotaly_o[i]=ftotaly_i[i_in];
	ftotalz_o[i]=ftotalz_i[i_in];
	//*/
}
