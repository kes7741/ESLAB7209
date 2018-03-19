// Cparticle Class Declaration
// Cparticle class contains particle information.
#ifndef fmax
#define fmax(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef fmin
#define fmin(a,b) (((a)<(b))?(a):(b))
#endif
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_11{
	uint_t number_of_neighbors;							// number of neighbor particles
	uint_t p_type;													// particle type: FLUID or BOUNDARY

	Real x,y,z;															// (Predicted) positions [m] ( Predictor_Corrector : Predicted position / Euler : Real Position )
	Real x0,y0,z0;													// Initial positions [m]
	Real m;																	// mass [kg]
	Real ux,uy,uz;													// (Predicted) velocity [m/s] ( Predictor_Corrector : Predicted velocity / Euler : Real Velocity )
	Real ux0,uy0,uz0;												// Initial velocity [m/s]

	Real cc;																// color code
	Real flt_s;															// Shepard filter
	Real w_dx;															// w(dx) for particle shifting
	Real h;																	// kernel distance
	Real stoh;															// initial particle space													
	Real temp;															// temperature [K]
	Real pres;															// pressure [Pa]
	Real rho;																// density [kg/m3]	( Predictor_Corrector : Predicted density / Euler : Real density )
	Real rho0;															// Initial density [kg/m3]
	Real drho;															// Time Derivative of density [kg/m3 s]
	Real rho_ref;														//
	Real grad_rhox,grad_rhoy,grad_rhoz;			// density gradient
	/*
	//Real fpx,fpy,fpz;											// pressure force [m/s2] ----------------- [����]ISPH ������ �Ҵ� �� !!!!
	//Real fvx,fvy,fvz;											// viscous force [m/s2]
	//Real fvax,fvay,fvaz;									// artificial viscous force [m/s2]
	//Real fgx,fgy,fgz;											// gravitational force [m/s2]
	//Real fsx,fsy,fsz;											// surface tension force [m/s2]
	//Real fix,fiy,fiz;											// interface force [m/s2]					[2017.05.02 jyb]
	//Real fbx,fby,fbz;											// boundary force [m/s2]
	//*/
	Real ftotalx,ftotaly,ftotalz;						// total force [m/s2]
	Real ftotal;

	Real p001;															// extra data
}part11;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_12{
	uint_t hf_boundary;											// heat flux particle index for heat transfer (heat flux particle : 1 / else : 0)
	uint_t ct_boundary;											// const temperatrue particle index (const temp particle :1 / else : 0)

	// psh: concentration diffusion
	Real dconcn;														// concentration time derivative
	Real concn,concn0;											// concentration

	Real enthalpy,enthalpy0;								// enthalpy [J/kg]
	Real denthalpy;

	// psh:: ISPH info
	Real fpx,fpy,fpz;												// pressure force [m/s2]
	Real x_adv,y_adv,z_adv;									// predicted position by advection forces
	Real rho_err;														// difference between predicted density and reference density
	Real stiffness;													// stiffness parameter
	Real drho0;															// Error compensation: divergence error

	// turbulence (by esk)
	Real SR;																// strain rate (2S:S)
	Real k_turb,e_turb;											// turbulence kinetic energy,dissipation rate --> check unit
	Real dk_turb,de_turb;										// turbulence kinetic energy,dissipation rate --> check unit
	Real vis_t;															// turbulence viscosity
	Real Sxx,Sxy,Sxz,Syy,Syz,Szz;						// strain tensor... for SPH model
}part12;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_13{
	Real curv;
	Real num_density;												// particle number density [1/m3]
	//Real nd_ref;

	Real nx,ny,nz;													// color code gradient for surface tension force (2016.09.02 jyb)
	Real nmag;															// 2017.04.20 jyb
	Real nx_c,ny_c,nz_c;										// color code gradient for surface tension force (2016.09.02 jyb)
	Real nmag_c;														// 2017.04.20 jyb

	Real nx_s,ny_s,nz_s;										// surface normal vector
	Real lbl_surf;													// surface label

	//Real cm_xx,cm_yy,cm_zz;
	//Real cm_xy,cm_yz,cm_zx;
	Real cm_d;
	Real inv_cm_xx,inv_cm_yy,inv_cm_zz;
	Real inv_cm_xy,inv_cm_yz,inv_cm_zx;

	uint_t I_cell,J_cell,K_cell;						// Cell number containing the particle

	// Cell_Index_Container  ##. NEIGHBOR SEARCH
	int_t PID;
	int_t CID;
	int_t sorted_PID;
	int_t sorted_CID;
}part13;
////////////////////////////////////////////////////////////////////////
typedef struct particles_array_2{
	uint_t pnb;//pnbv;											// neighbor list(pnb) & Verlet list(pnbv)
	Real dist;															// distance list [m]
	Real wij,dwij,dwij_cor;									// kernel(wij) & kernel derivative(dwij)
	Real dwx,dwy,dwz;
	Real dw_cx,dw_cy,dw_cz;
}part2;
////////////////////////////////////////////////////////////////////////
