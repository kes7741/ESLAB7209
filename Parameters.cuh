
#define 	vii_size		64
#define 	vif_size		32

//#define Real float
//#define int_t int
//#define uint_t unsigned int

#define Wcsph 0
#define Pcisph 1
#define Dfsph 2

#define FLUID	1
#define BOUNDARY	0
#define MOVING		9
#define CORIUM		50

#define CONCRETE	60
#define CONCRETE_SOL	-60
#define MCCI_CORIUM	65

#define IVR_METAL	70
#define IVR_CORIUM	75
#define IVR_VESSEL	80
#define IVR_VESSEL_SOL	-80

#define DUMMY_IN	100
#define Y_IN		1

#define Liquid 0
#define Gas 1
#define Solid 2

#define Single_Phase	1
#define Two_Phase	2

#define WRONG_INDEX	1e8			// Limitation of WRONG_INDEX: 2.2e9 (signed integer)

#define Gaussian	0
#define Quintic		1
#define Quartic		2
#define Wendland2	3
#define Wendland4	4
#define Wendland6	5

#define Mass_Sum	0			// Mass_Summation Method
#define Continuity	1			// Continuity Equation Method

#define Shepard		0			// Shepard Filter
#define MLS			1			// Moving Least Square(MLS) Filter

#define Euler		0			// Euler Explicit Time Stepping
#define Pre_Cor		1			// Predictor-Corrector Time Stepping

#define Potential	1			// Surface tension force based on inter-particle potential energy (Single / Two phase)
#define Curvature	2			// Surface tension force based on surface curvature ( Two phase )

#define PI	3.141592653

#define Gravitational_CONST 9.8		// gravitational constant [m/s2]

#define Alpha		0.005			// coefficient alpha of artificial viscosity
#define Beta		0.005			// coefficient beta of artificial viscosity

#define NORMAL_THRESHOLD 0.001			// normal threshold for surface tension force

#define PNBV_TO_PNB		1			// pnbv size / pnb size


#define NUM_PART		600		//
#define NB_SIZE			200

#define	Water	0
#define	Metal	1

#define epsilon			1e-6		// denominator
#define delta			0.1			// delta-SPH coefficient
#define K_repulsive		0.0001		// constant for repulsive boundary force

#define C_mu		0.09
#define C_e1		1.44
#define C_e2		1.92
#define sigma_k		1.0
#define sigma_e		1.3
#define kappa_t		0.41
#define Cs_SPS		0.12			// check please (by esk)
#define CI_SPS		0.00066
#define L_SPS		0.01			// scale of length scale (for test)

#define Lm			0.01
#define Maximum_turbulence_viscosity	0.1
#define Laminar		0
#define K_LM		1
#define K_E			2
#define SPS			3

#define DIFF_DENSITY	0
