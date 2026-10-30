/*
	c++ implementation of openCOSMO-RS including multiple segment descriptors
	@author: Simon Mueller, 2022
*/


#pragma once
#define NOMINMAX
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <cmath>
#include <unordered_map>
#include <numeric>
#include <memory>
#include <deque>
#include <algorithm>
#include <mutex>
#include <limits>


// always include this as at least SSE3 is required
#include <immintrin.h>

// Constants
#define PI 3.14159265358979323846
#define R_GAS_CONSTANT 8.31446261815324
#define N_AVOGADRO 6.022140857e+23
#define MAX_CONCENTRATION_DIFF_FROM_ZERO 0.000001
#define VMOL_V 6.022140857e-7
#define E_C 1.602176634e-19							// Elementary charge in C
#define EPSILON0_Fm 8.8541878188e-12				// Vaccum permittivity in F/m
#define kB 1.380649e-23								// Boltzmann constant in J/K
#define DEBYE 3.33564095e-30						// 1 Debye in Coulomb*meter

constexpr std::array<double, 103> ioniz_potential_ev = { {
		/*  0 */ NAN,
		/*  1 */ 13.598,
		/*  2 */ 24.587,
		/*  3 */  5.392,
		/*  4 */  9.323,
		/*  5 */  8.298,
		/*  6 */ 11.260,
		/*  7 */ 14.534,
		/*  8 */ 13.618,
		/*  9 */ 17.423,
		/* 10 */ 21.565,
		/* 11 */  5.139,
		/* 12 */  7.646,
		/* 13 */  5.986,
		/* 14 */  8.152,
		/* 15 */ 10.487,
		/* 16 */ 10.360,
		/* 17 */ 12.968,
		/* 18 */ 15.760,
		/* 19 */  4.341,
		/* 20 */  6.113,
		/* 21 */  6.561,
		/* 22 */  6.828,
		/* 23 */  6.746,
		/* 24 */  6.767,
		/* 25 */  7.434,
		/* 26 */  7.902,
		/* 27 */  7.881,
		/* 28 */  7.640,
		/* 29 */  7.726,
		/* 30 */  9.394,
		/* 31 */  5.999,
		/* 32 */  7.900,
		/* 33 */  9.789,
		/* 34 */  9.752,
		/* 35 */ 11.814,
		/* 36 */ 14.000,
		/* 37 */  4.177,
		/* 38 */  5.695,
		/* 39 */  6.217,
		/* 40 */  6.634,
		/* 41 */  6.759,
		/* 42 */  7.092,
		/* 43 */  7.280,
		/* 44 */  7.361,
		/* 45 */  7.459,
		/* 46 */  8.337,
		/* 47 */  7.576,
		/* 48 */  8.994,
		/* 49 */  5.786,
		/* 50 */  7.344,
		/* 51 */  8.608,
		/* 52 */  9.010,
		/* 53 */ 10.451,
		/* 54 */ 12.130,
		/* 55 */  3.894,
		/* 56 */  5.212,
		/* 57 */  5.577,
		/* 58 */  5.539,
		/* 59 */  5.473,
		/* 60 */  5.525,
		/* 61 */  5.582,
		/* 62 */  5.644,
		/* 63 */  5.670,
		/* 64 */  6.150,
		/* 65 */  5.864,
		/* 66 */  5.939,
		/* 67 */  6.022,
		/* 68 */  6.108,
		/* 69 */  6.184,
		/* 70 */  6.254,
		/* 71 */  5.426,
		/* 72 */  6.825,
		/* 73 */  7.550,
		/* 74 */  7.864,
		/* 75 */  7.834,
		/* 76 */  8.438,
		/* 77 */  8.967,
		/* 78 */  8.959,
		/* 79 */  9.226,
		/* 80 */ 10.438,
		/* 81 */  6.108,
		/* 82 */  7.417,
		/* 83 */  7.286,
		/* 84 */  8.417,
		/* 85 */  9.318,
		/* 86 */ 10.749,
		/* 87 */  4.073,
		/* 88 */  5.279,
		/* 89 */  5.170,
		/* 90 */  6.307,
		/* 91 */  5.890,
		/* 92 */  6.194,
		/* 93 */  6.266,
		/* 94 */  6.027,
		/* 95 */  5.974,
		/* 96 */  5.992,
		/* 97 */  6.198,
		/* 98 */  6.300,
		/* 99 */  6.420,
		/*100 */  6.500,
		/*101 */  6.580,
		/*102 */  6.650
	}
};

constexpr std::array<double, 104> atomic_mass_u = { {
	NAN,
	1.008,    // H
	4.003,    // He
	6.94,     // Li
	9.012,    // Be
	10.81,    // B
	12.011,   // C
	14.007,   // N
	15.999,   // O
	18.998,   // F
	20.180,   // Ne
	22.990,   // Na
	24.305,   // Mg
	26.982,   // Al
	28.085,   // Si
	30.974,   // P
	32.06,    // S
	35.45,    // Cl
	39.948,   // Ar
	39.098,   // K
	40.078,   // Ca
	44.956,   // Sc
	47.867,   // Ti
	50.942,   // V
	51.996,   // Cr
	54.938,   // Mn
	55.845,   // Fe
	58.933,   // Co
	58.693,   // Ni
	63.546,   // Cu
	65.38,    // Zn
	69.723,   // Ga
	72.630,   // Ge
	74.922,   // As
	78.971,   // Se
	79.904,   // Br
	83.798,   // Kr
	85.468,   // Rb
	87.62,    // Sr
	88.906,   // Y
	91.224,   // Zr
	92.906,   // Nb
	95.95,    // Mo
	98.0,     // Tc
	101.07,   // Ru
	102.906,  // Rh
	106.42,   // Pd
	107.868,  // Ag
	112.414,  // Cd
	114.818,  // In
	118.710,  // Sn
	121.760,  // Sb
	127.60,   // Te
	126.904,  // I
	131.293,  // Xe
	132.905,  // Cs
	137.327,  // Ba
	138.905,  // La
	140.116,  // Ce
	140.908,  // Pr
	144.242,  // Nd
	145.0,    // Pm
	150.36,   // Sm
	151.964,  // Eu
	157.25,   // Gd
	158.925,  // Tb
	162.500,  // Dy
	164.930,  // Ho
	167.259,  // Er
	168.934,  // Tm
	173.045,  // Yb
	174.967,  // Lu
	178.486,  // Hf
	180.948,  // Ta
	183.84,   // W
	186.207,  // Re
	190.23,   // Os
	192.217,  // Ir
	195.085,  // Pt
	196.967,  // Au
	200.592,  // Hg
	204.38,   // Tl
	207.2,    // Pb
	208.980,  // Bi
	209.0,    // Po
	210.0,    // At
	222.0,    // Rn
	223.0,    // Fr
	226.0,    // Ra
	227.0,    // Ac
	232.038,  // Th
	231.036,  // Pa
	238.029,  // U
	237.0,    // Np
	244.0,    // Pu
	243.0,    // Am
	247.0,    // Cm
	247.0,    // Bk
	251.0,    // Cf
	252.0,    // Es
	257.0,    // Fm
	258.0,    // Md
	259.0,    // No
	262.0     // Lr
} };

/* Structures and classes */

template <typename T>
void apply_vector_permutation_in_place(std::vector<T>& vec, const std::vector<int>& p) {
	std::vector<bool> done(vec.size());
	for (int i = 0; i < vec.size(); ++i) {
		if (done[i])
		{
			continue;
		}
		done[i] = true;
		int prev_j = i;
		int j = p[i];
		while (i != j) {
			std::swap(vec[prev_j], vec[j]);
			done[j] = true;
			prev_j = j;
			j = p[j];
		}
	}
}

struct parameters {

	parameters()
		: phi_param(NULL, 0, 0),
		permittivity_model_0_parameters(NULL, 0, 0)
	{ }

	/* General switches */
	int sw_phi = 0;					// Whether or not to use the EOS
	int sw_electrolyte = 0;			// Wheter or not to use openCOSMO-RS-Phi-ES
	int sw_Born_term = 1;			// Whether or not to include the Born term (only used when sw_electrolyte == 1)
	int sw_permittivity_mix_model = 0;
	int sw_PT_Flash = 0;
	int sw_checkAndCalcBinaryLLE = 0;
	int sw_lnGamma = 0;
	int sw_partitionCoeff = 0;
	int sw_isotherm = 0;
	int sw_excess = 0;
	int sw_Pbub = 0;
	int sw_Pdew = 0;
	int sw_Tbub = 0;
	int sw_Tdew = 0;
	int sw_criticalPoint = 0;
	int sw_criticalPoint_specifiedx = 0;
	int sw_azeotropicPoint = 0;
	int sw_misfit = 2;              /* switch: "0" to use contribution without accounting for sigma correlation.
										   "1" to account for sigma correlation in misfit contribution
										   "2" to account for sigma correlation in misfit contribution except for ions */

	int sw_useSegmentReferenceStateForInteractionMatrix = 0;		/* switch: "0" Set segment reference state to pure segment (COSMO-RS default) (segment reference state cancels if molecular refstate calculated)
																	   "1" Set segment reference state to conductor (segment reference state cancels if molecular refstate calculated) */

	int sw_combTerm = 1;            /* switch:  "0" No combinatorial term
											"1" to use the combinatorial term by Staverman-Guggenheim
											"2" to use the combinatorial term by Klamt (2003)
											"3" to use modified Staverman-Guggenheim combinatorial term with exponential scaling */

	int sw_atomicNumber = 1; // whether or not to use the atomic number as descriptor

	int sw_differentiateHydrogens = 0;
	int sw_SR_polarizabilities = 0;
	int sw_differentiateMoleculeGroups = 0;

	std::string sw_COSMOfiles_type = "ORCA_COSMO_TZVPD"; // Type of COSMOfile used, this is used to know which function to use to load the sigma profile. e.g. Turbomole

	int sw_calculateContactStatisticsAndAdditionalProperties = 0;		/* switch:  "0" Do not calculate contact statistics
																				"1" Calculate contact statistics
																				"2" Calculate contact statistics, partial molar properties and average surface energies*/
	int numberOfPartialInteractionMatrices;

	int sw_alwaysReloadSigmaProfiles = 0;

	int sw_reloadConcentrations = 0;
	int sw_reloadReferenceConcentrations = 0;

	int sw_alwaysCalculateSizeRelatedParameters = 0;	/* switch: "0" combinatorial term and segment fraction is only calculated once at first execution
															   "1" combinatorial term and segment fraction  is calculated on every iteration */

	int sw_skip_COSMOSPACE_errors = 0;	/* switch: "0" if COSMOSPACE does not converge, it stops execution showing DEBUG information
											   "1" if COSMOSPACE does not converge, execution continues setting the objective function very high */

	int sw_dGsolv_calculation_strict = 1; // 0Allows calculation of solvation free energies also for atoms that have not been parameterized, but gives a warning
										  // 1: Allows calculation of solvation free energies if all parameters are available

	std::string sw_LR_model = "ME-PDH";

	/* COSMO-RS MODEL PARAMETERS */
	/* General parameters */
	double Aeff;                    /* area of effective thermodynamic contact  [Angstrom^2] */
	double alpha;                   /* misfit energy prefactor (alpha')  [J*Angstrom^2/(mol*e^2)] */
	double CHB;                     /* hydrogen bond prefactor [J*Angstrom^2/(mol*e^2)] */
	double CHBT;                    /* hydrogen bond temperature parameter */
	double SigmaHB;                 /* sigma threshhold for hydrogen bond [e/Angstrom^2] */
	double Rav;                     /* Averaging Radius for sigma averaging [Angstrom] */
	double RavCorr;                 /* Averaging Radius to determine sigma correlation [Angstrom]*/
	double fCorr;                   /* Correction factor for the introduction of sigma correlation to the misfit energy */
	double comb_SG_A_std;           /* Standard area for Staverman-Guggenheim combinatorial contribution [Angstrom^2] */
	double comb_SG_z_coord;         /* Coordination number for Staverman Guggenheim contribution */
	double comb_modSG_exp;			/* Exponent parameter for modified Staverman-Guggenheim contribution with exponential scaling [-] */
	double comb_SGG_lambda;         /* Parameter for the mod. Staverman-Guggenheim by Grensemann published in Grensemann & Gmehling (2005) especially developed for COSMO-RS */
	double comb_SGG_beta;         	/* Parameter for the mod. Staverman-Guggenheim by Grensemann published in Grensemann & Gmehling (2005) especially developed for COSMO-RS */

	double comb_lambda0;			/* lambda paramters used in the calculation of the combinatorial contribuation from Klamt or Frank & Hannebauer */
	double comb_lambda1;
	double comb_lambda2;
	double hole_area;

	/* Parameters used for the dispersion term */
	double E_F_corr;
	double m_vdW;

	/* Parameters used for electrolyte systems (openCOSMO-RS-ES) */
	double salt_diam_sc_M;
	double salt_diam_sc_P;

	double msf_cP_org;
	double msf_aP_org;
	double msf_aM_org;
	double msf_cM_org;

	double fHB_cP_org;
	double fHB_aP_XH;
	double fHB_A_org;
	double fHB_cM_org;
	double fHB_aM_org;
	double C1;
	double DCO;

	double fHB_cP_H2O;
	double fHB_aO_H2O;
	double fHB_aP_H2O;
	double fHB_cM_H2O;
	double fHB_aM_H2O;

	double msf_cP_H2O;
	double msf_cM_H2O;
	double msf_aP_H2O;
	double msf_aM_H2O;
	double C0;
	double DCW;

	double msf_cP_aP;
	double msf_cP_aM;
	double msf_cM_aP;
	double msf_cM_aM;

	double fHB_cM_aM;
	double fHB_cM_aP;
	double fHB_cP_aM;
	double fHB_cP_aP;
	double fHB_iP_iP;

	double D1;
	double E1;
	double D2;
	double E2;
	double C2;

	// PDH term
	double omega0;
	double omega1;

	/* Parameters used for solvation energy calculation */
	double dGsolv_eta;
	double dGsolv_omega_ring;
	std::vector<double> dGsolv_tau = std::vector<double>(118, 0.0);
	std::vector<double> dGsolv_E_gas;
	std::vector<int> dGsolv_numberOfAtomsInRing;
	std::vector<double> dGsolv_molarVolume;

	/* Radii used for cosmo segment scaling for monoatomic ions */
	std::vector<double> R_i = std::vector<double>(118, 0.0);                       /* new radii, element specific (r_i[AN]) from the input file*/
	std::vector<double> R_i_COSMO = std::vector<double>(118, 0.0);                 /* old radii from COSMO-file, element specific (r_i[AN]); */

	/* Experimental parameters */
	/* These are used to test new modifications to COSMO-RS. Parameters of succesfull modifications should
	*  later be implemented with standard parameters */
	std::unordered_map<std::string, double> exp_param;


	/* COSMO-RS-PHI MODEL PARAMETERS*/
	/* Pure Component Parameters of the EoS*/
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> phi_param;

	/* Parameters for the permittivity model used within the openCOSMO-RS-Phi-ES framework */
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> permittivity_model_0_parameters;

	/* Hyperparameters / external values for the calculation of mixture properties */
	bool useGivenInitialK;
	bool noTracing;
	std::vector<double> temperatures_mixing;
	std::vector<double> pressures_mixing;
	std::vector<std::vector<double>> concentration_x_phase_mixing;
	std::vector<std::vector<double>> concentration_y_phase_mixing;


	std::vector<int> HBClassElmnt = std::vector<int>(300, 0);

	double sigmaMin = -0.15;
	double sigmaMax = 0.15;
	double sigmaStep = 0.001;

	std::vector<double> ChargeRaster;

};

struct segmentTypeCollection {

private:
	std::vector<double> SegmentTypeAreasRowTemplate;

	std::vector<int> get_permutation_vector()
	{
		std::vector<int> p(SegmentTypeGroup.size());
		std::iota(p.begin(), p.end(), 0);
		std::sort(p.begin(), p.end(),
			[&](int i, int j) {

				if (SegmentTypeMoleculeIndex[i] != SegmentTypeMoleculeIndex[j]) { return SegmentTypeMoleculeIndex[i] < SegmentTypeMoleculeIndex[j]; }

				if (SegmentTypeGroup[i] != SegmentTypeGroup[j]) { return SegmentTypeGroup[i] < SegmentTypeGroup[j]; }

				// if both segments belong to an monoatomic ion, first sort by atomic number
				if (SegmentTypeGroup[i] == 3 || SegmentTypeGroup[i] == 5) {
					if (SegmentTypeAtomicNumber[i] != SegmentTypeAtomicNumber[j]) { return SegmentTypeAtomicNumber[i] < SegmentTypeAtomicNumber[j]; }
				}

				if (SegmentTypeSigma[i] != SegmentTypeSigma[j]) { return SegmentTypeSigma[i] < SegmentTypeSigma[j]; }

				if (SegmentTypeSigmaCorr[i] != SegmentTypeSigmaCorr[j]) { return SegmentTypeSigmaCorr[i] < SegmentTypeSigmaCorr[j]; }

				if (SegmentTypeHBtype[i] != SegmentTypeHBtype[j]) { return SegmentTypeHBtype[i] < SegmentTypeHBtype[j]; }
				if (SegmentTypeAtomicPolariz[i] != SegmentTypeAtomicPolariz[j]) { return SegmentTypeAtomicPolariz[i] < SegmentTypeAtomicPolariz[j]; }

				if (SegmentTypeAtomicNumber[i] != SegmentTypeAtomicNumber[j]) { return SegmentTypeAtomicNumber[i] < SegmentTypeAtomicNumber[j]; }

				return i < j;

			});
		return p;
	}


public:

	int lowerBoundIndexForGroup[8] = { 0 };
	int upperBoundIndexForGroup[8] = { 0 };
	int numberOfSegmentsForGroup[8] = { 0 };

	segmentTypeCollection() {
		segmentTypeCollection(1);
	}

	segmentTypeCollection(int numberOfMolecules) {

		for (int i = 0; i < numberOfMolecules; i++) {
			SegmentTypeAreasRowTemplate.push_back(0.0);
		}
		SegmentTypeAreasRowTemplate.shrink_to_fit();
	}

	std::vector<std::vector<double>> SegmentTypeAreas;
	std::vector<unsigned short> SegmentTypeGroup;
	std::vector<double> SegmentTypeSigma;
	std::vector<double> SegmentTypeSigmaCorr;
	std::vector<unsigned short> SegmentTypeHBtype;
	std::vector<unsigned short> SegmentTypeAtomicNumber;
	std::vector<int> SegmentTypeMoleculeIndex;
	std::vector<double> SegmentTypeAtomicPolariz;

	void clear() {

		SegmentTypeAreas.clear();
		SegmentTypeGroup.clear();
		SegmentTypeSigma.clear();
		SegmentTypeSigmaCorr.clear();
		SegmentTypeHBtype.clear();
		SegmentTypeAtomicNumber.clear();
		SegmentTypeAtomicPolariz.clear();
		SegmentTypeMoleculeIndex.clear();
	}

	void reserve(int numberOfSegmentsTypes) {

		SegmentTypeAreas.reserve(numberOfSegmentsTypes);
		SegmentTypeGroup.reserve(numberOfSegmentsTypes);
		SegmentTypeSigma.reserve(numberOfSegmentsTypes);
		SegmentTypeSigmaCorr.reserve(numberOfSegmentsTypes);
		SegmentTypeHBtype.reserve(numberOfSegmentsTypes);
		SegmentTypeAtomicNumber.reserve(numberOfSegmentsTypes);
		SegmentTypeAtomicPolariz.reserve(numberOfSegmentsTypes);
		SegmentTypeMoleculeIndex.reserve(numberOfSegmentsTypes);
	}

	void shrink_to_fit() {

		for (int i = 0; i < SegmentTypeAreas.size(); i++) {
			SegmentTypeAreas[i].shrink_to_fit();
		}

		SegmentTypeAreas.shrink_to_fit();
		SegmentTypeSigma.shrink_to_fit();
		SegmentTypeSigmaCorr.shrink_to_fit();
		SegmentTypeHBtype.shrink_to_fit();
		SegmentTypeAtomicNumber.shrink_to_fit();
		SegmentTypeAtomicPolariz.shrink_to_fit();
		SegmentTypeMoleculeIndex.shrink_to_fit();
	}

	size_t size() {
		return SegmentTypeHBtype.size();
	}

	void add(unsigned short ind_molecule, unsigned short group, double Sigma, double SigmaCorr, unsigned short HBtype, unsigned short atomicNumber, double Area, int moleculeIndex, double atomicPolariz) {

		if (Area == 0) {
			return;
		}

		int index = -1;
		for (int i = 0; i < SegmentTypeHBtype.size(); i++) {
			if (SegmentTypeGroup[i] == group &&
				SegmentTypeHBtype[i] == HBtype &&
				SegmentTypeSigma[i] == Sigma &&
				SegmentTypeSigmaCorr[i] == SigmaCorr &&
				SegmentTypeAtomicPolariz[i] == atomicPolariz &&
				SegmentTypeAtomicNumber[i] == atomicNumber &&
				SegmentTypeMoleculeIndex[i] == moleculeIndex) {
				index = i;
				break;
			}
		}

		if (index == -1) {
			SegmentTypeGroup.push_back(group);
			SegmentTypeHBtype.push_back(HBtype);
			SegmentTypeSigma.push_back(Sigma);
			SegmentTypeSigmaCorr.push_back(SigmaCorr);
			SegmentTypeAtomicNumber.push_back(atomicNumber);
			SegmentTypeAtomicPolariz.push_back(atomicPolariz);
			SegmentTypeMoleculeIndex.push_back(moleculeIndex);

			index = (int)SegmentTypeHBtype.size() - 1;
			SegmentTypeAreas.push_back(SegmentTypeAreasRowTemplate);
		}

		SegmentTypeAreas[index][ind_molecule] += Area;
	}

	void sort() {
		std::vector<int> p = get_permutation_vector();

		apply_vector_permutation_in_place(SegmentTypeGroup, p);
		apply_vector_permutation_in_place(SegmentTypeSigma, p);
		apply_vector_permutation_in_place(SegmentTypeSigmaCorr, p);
		apply_vector_permutation_in_place(SegmentTypeHBtype, p);
		apply_vector_permutation_in_place(SegmentTypeAtomicNumber, p);
		apply_vector_permutation_in_place(SegmentTypeAtomicPolariz, p);
		apply_vector_permutation_in_place(SegmentTypeAreas, p);
		apply_vector_permutation_in_place(SegmentTypeMoleculeIndex, p);

		int temporaryindex = -1;
		int i;
		for (i = 0; i < size(); i++) {

			if (temporaryindex != SegmentTypeGroup[i]) {

				if (temporaryindex != -1) {
					upperBoundIndexForGroup[temporaryindex] = i;
				}
				temporaryindex = SegmentTypeGroup[i];
				lowerBoundIndexForGroup[temporaryindex] = i;
			}

		}
		upperBoundIndexForGroup[temporaryindex] = i;

		for (i = 0; i < 7; i++) {
			numberOfSegmentsForGroup[i] = std::max(upperBoundIndexForGroup[i] - lowerBoundIndexForGroup[i], 0);
		}
	}
};

struct molecule {
	/* segment properties */
	segmentTypeCollection segments;

	molecule() {
		segments = segmentTypeCollection(1);
	}

	std::string name;
	std::string qmMethod;

	int index;
	double Area;
	double Volume;
	double epsilonInfinityTotalEnergy;
	double molarVolumeAt25C;
	double molarMass;

	//double critical_T;
	//double critical_P;

	signed char moleculeCharge;
	unsigned short moleculeGroup;

	// Possible groups
	/*
	0 monoatomic neutral
	1 polyatomic neutral
	2 water
	3 monoatomic cation
	4 polyatomic cation
	5 monoatomic anion
	6 polyatomic anion
	*/

	// Atoms
	Eigen::MatrixXd atomPositions;
	Eigen::VectorXd atomRadii;
	Eigen::MatrixXd atomPolarizabilityTensors;
	Eigen::VectorXi atomAtomicNumbers;

	// Segment information directly from the input file
	Eigen::MatrixXd segmentPositions;
	Eigen::VectorXi segmentAtomIndices;
	Eigen::VectorXi segmentAtomicNumber;
	Eigen::VectorXd segmentAtomicPolariz;
	Eigen::VectorXi segmentHydrogenBondingType;
	Eigen::VectorXd segmentAreas;
	Eigen::VectorXd segmentSigmas;

	void clear_unneeded_matrices(bool keepDataNeededForReloadingSigmaProfile = false) {
		atomPositions.resize(0, 0);
		atomRadii.resize(0);
		// atomAtomicNumbers.resize(0); this is currently needed for monoatomic ions

		segmentAtomIndices.resize(0);

		if (keepDataNeededForReloadingSigmaProfile == false) {
			segmentAtomicNumber.resize(0);
			segmentAtomicPolariz.resize(0);
			segmentPositions.resize(0, 0);
			segmentHydrogenBondingType.resize(0);
			segmentAreas.resize(0);
			segmentSigmas.resize(0);
		}

	}
};

struct calculation {

	int number = -1;
	std::string calculationType = "";

	segmentTypeCollection segments;

	calculation(int numberOfMolecules) : \
		contactStatistics(NULL, 0, 0, 0), \
		averageSurfaceEnergies(NULL, 0, 0, 0, 0), \
		partialMolarEnergies(NULL, 0, 0, 0), \
		lnGammaCombinatorial(NULL, 0, 0), \
		lnGammaResidual(NULL, 0, 0), \
		lnGammaLR(NULL, 0, 0), \
		lnGammaTotal(NULL, 0, 0), \
		lnGammaSRLRTotal(NULL, 0, 0), \
		lnGammaBorn(NULL, 0, 0), \
		lnGammaElectrostatic(NULL, 0, 0), \
		MIACInMoleFractionScale(NULL, 0, 0), \
		MIACInMolalityScale(NULL, 0, 0), \
		lnPhiRepulsive(NULL, 0, 0), \
		lnPhiAttractive(NULL, 0, 0), \
		lnPhiElectrostatic(NULL, 0, 0), \
		lnPhiBorn(NULL, 0, 0), \
		lnPhiTotal(NULL, 0, 0), \
		originalConcentrations(NULL, 0, 0), \
		calculatedPartitionCoefficients(NULL, 0, 0), \
		molarVolume(NULL, 0), \
		repulsivePressure(NULL, 0), \
		attractivePressure(NULL, 0), \
		BornPressure(NULL, 0), \
		electrostaticPressure(NULL, 0), \
		totalPressure(NULL, 0), \
		targetPressure(NULL, 0), \
		hE(NULL, 0), \
		gE(NULL, 0), \
		cpE(NULL, 0), \
		hM(NULL, 0), \
		gM(NULL, 0), \
		cpM(NULL, 0), \
		targetTemperature(NULL, 0), \
		criticalT(NULL, 0), \
		criticalP(NULL, 0), \
		criticalx(NULL, 0, 0), \
		dGsolv(NULL, 0, 0), \
		ion_diam_mono(NULL), \
		ion_diam_poly_sph(NULL), \
		salt_diam(NULL)
	{
		segments = segmentTypeCollection(numberOfMolecules);
		moleculeIndices = std::vector<int>(numberOfMolecules);
		moleculeStochCoeffs = std::vector<int>(numberOfMolecules);
	}

	std::vector<std::shared_ptr<molecule>> components;
	std::vector<int> moleculeIndices;
	std::vector<int> moleculeStochCoeffs;


	std::vector<std::vector<double>> concentrations;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> originalConcentrations;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> calculatedPartitionCoefficients;
	MatrixCalcType segmentConcentrations;
	MatrixCalcType PartialsegmentConcentrationsPartialv;
	MatrixCalcType segmentGammas;
	MatrixCalcType segmentGammasPDETemperature; // added
	MatrixCalcType segmentGammasPDEVolume; // added
	MatrixCalcType segmentGammasSecondPDETemperature; // added
	std::vector<std::vector<int>> referenceStateCalculationIndices;
	std::vector<unsigned short> referenceStateType;

	Eigen::MatrixXd PhiDash_pxi;
	Eigen::MatrixXd ThetaDash_pxi;

	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaCombinatorial;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaResidual;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaTotal;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaLR;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaSRLRTotal;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaBorn;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnGammaElectrostatic;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MIACInMoleFractionScale;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MIACInMolalityScale;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiRepulsive;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiAttractive;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiElectrostatic;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiBorn;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> lnPhiTotal;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> gE; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> hE; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> cpE; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> gM; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> hM; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> cpM; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> targetTemperature; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> criticalT; // added
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> criticalP; // added
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> criticalx; // added
	VectorCalcType molarResidualEnthalpy; // added
	VectorCalcType residualHeatCapacityAtConstantPressure;
	MatrixCalcType lnPhiTotalPDETAtConstantPx; // added
	MatrixCalcType lnGammaResidualPDETatConstantvx; // added
	MatrixCalcType lnGammaResidualSecondPDETatConstantvx; // added
	MatrixCalcType lnGammaResidualPDEvatConstantTx; // added
	bool recalculateConcentrationsForPhi = false;

	Eigen::VectorXd componentCavityVolume;
	Eigen::VectorXd componentHoleVolume;
	Eigen::VectorXd componentCharges;
	Eigen::VectorXd componentBornDiameter;

	Eigen::VectorXd mixtureCavityVolume;
	Eigen::VectorXd mixtureHoleVolume;

	Eigen::MatrixXd PartitionCoefficients; // added
	Eigen::MatrixXi IndicesForPureComponentCalculations; // added
	std::vector<double> uniqueTemperatures; // added
	std::vector<double> uniquePressures; // added
	Eigen::MatrixXd SaturationPressures; // added
	Eigen::MatrixXd SaturationTemperatures; // added
	Eigen::MatrixXd overallConcentrations; // added
	int helperValue;

	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> molarVolume;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> repulsivePressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> attractivePressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> BornPressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> electrostaticPressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> targetPressure;
	Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> totalPressure;
	Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> dGsolv;

	double ion_diam_mono = std::numeric_limits<double>::quiet_NaN();
	double ion_diam_poly_sph;
	double salt_diam;

	Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> contactStatistics;
	Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>> averageSurfaceEnergies;
	Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> partialMolarEnergies;


	// the following matrices are to save the computations temporarily if needed
	// for internal use. Access is through the respective maps above.
	Eigen::MatrixXd concentrations_data;
	Eigen::MatrixXd lnGammaCombinatorial_data;
	Eigen::MatrixXd lnGammaResidual_data;
	Eigen::MatrixXd lnGammaTotal_data;
	Eigen::MatrixXd dGsolv_data;
	Eigen::VectorXd molarVolume_data;
	Eigen::VectorXd repulsivePressure_data;
	Eigen::VectorXd attractivePressure_data;
	Eigen::VectorXd BornPressure_data;
	Eigen::VectorXd electrostaticPressure_data;
	Eigen::VectorXd totalPressure_data;
	Eigen::VectorXd targetPressure_data;
	Eigen::VectorXd targetTemperature_data;
	Eigen::MatrixXd lnPhiTotal_data;
	Eigen::MatrixXd lnPhiRepulsive_data;
	Eigen::MatrixXd lnPhiAttractive_data;
	Eigen::MatrixXd lnPhiElectrostatic_data;
	Eigen::MatrixXd lnPhiBorn_data;
	Eigen::VectorXd criticalP_data;
	Eigen::VectorXd criticalT_data;
	Eigen::MatrixXd criticalx_data;

	Eigen::Tensor<double, 3, Eigen::RowMajor> contactStatistics_data;
	Eigen::Tensor<double, 4, Eigen::RowMajor> averageSurfaceEnergies_data;
	Eigen::Tensor<double, 3, Eigen::RowMajor> partialMolarEnergies_data;

	//holds first and last segment index that are needed for calculations with the COSMOSPACE equation allowing to save calculations
	std::vector<int> lowerBoundIndexForCOSMOSPACECalculation;
	std::vector<int> upperBoundIndexForCOSMOSPACECalculation;

	std::vector<double> temperatures;
	std::vector<double> pressures; // added

	std::vector<std::vector<int>> TauConcentrationIndices;
	std::vector<double> TauTemperatures;
	std::vector<MatrixCalcType>Taus;

	// liquid-root volumes of the previous call
	struct OptRootHint { int kind; int component; double T; double P; std::vector<double> x; double v; };
	std::vector<OptRootHint> optRootHints;

	std::vector<int> actualConcentrationIndices;

	size_t originalNumberOfCalculations;

	void invalidateTaus() {

		if (Taus.size() != 0) {
			for (int i = 0; i < TauTemperatures.size(); i++) {
				Taus[i] = MatrixCalcType::Zero(0, 0);
			}
		}
	}

	int addOrFindTauIndexForConditions(double temperature) {

		int index = -1;

		for (int i = 0; i < TauTemperatures.size(); i++) {
			if (TauTemperatures[i] == temperature) {

				index = i;
				break;
			}
		}

		if (index == -1) {
			TauTemperatures.push_back(temperature);
			Taus.push_back(MatrixCalcType::Zero(0, 0));
			TauConcentrationIndices.push_back(std::vector<int>());

			index = int(TauTemperatures.size()) - 1;
		}

		return index;
	}

	int addOrFindArrayIndexForConcentration(std::vector<double> concentration, double temperature, double pressure = -1)
	{
		int index = -1;
		int nConditions = 1; // only temperature
		for (int i = 0; i < concentrations.size(); i++) {
			int nEqual = 0;
			for (int j = 0; j < concentration.size(); j++) {
				if (concentrations[i][j] == concentration[j]) {
					nEqual++;
				}
			}

			// temperature
			if (temperatures[i] == temperature) {
				nEqual++;
			}

			// if concentrations and conditions the same, we found the guy
			if (nEqual == concentration.size() + nConditions) {
				index = i;
				break;
			}
		}

		if (index == -1) {
			concentrations.push_back(std::move(concentration));
			temperatures.push_back(temperature);

			// When calculating excess propoerties with openCOSMO-RS-Phi also pressure must be considered as fugacity coefficient = f(T,P,x)
			if (pressure != -1) {
				pressures.push_back(pressure);
			}

			index = int(concentrations.size()) - 1;
		}

		return index;
	}

	template <typename SubcalculationContainer>
	int addOrFindArrayIndexInSubcalculations(const SubcalculationContainer& subcalculations, std::vector<std::shared_ptr<molecule>> molecules, double temperature = -1, double pressure = -1)
	{
		int index = -1;
		int nConditions = 0;
		if (temperature != -1) {
			nConditions += 1;
		}
		if (pressure != -1) {
			nConditions += 1;
		}
		for (int i = 0; i < subcalculations.size(); i++) {
			int nEqual = 0;
			for (int j = 0; j < molecules.size(); j++) {
				if (subcalculations[i].components[j] == molecules[j]) {
					nEqual++;
				}
			}

			// temperature
			if (temperature != -1) {
				if (subcalculations[i].temperatures[0] == temperature) {
					nEqual++;
				}
			}

			// pressure if also part of checked conditions
			if (pressure != -1) {
				if (subcalculations[i].targetPressure(0) == pressure) {
					nEqual++;
				}
			}

			// if concentrations and conditions the same, we found the guy
			if (nEqual == molecules.size() + nConditions) {
				index = i;
				break;
			}
		}

		return index;
	}


	void shrink_to_fit() {

		for (int i = 0; i < concentrations.size(); i++) {
			concentrations[i].shrink_to_fit();
		}

		for (int i = 0; i < TauConcentrationIndices.size(); i++) {
			TauConcentrationIndices[i].shrink_to_fit();
		}

		TauTemperatures.shrink_to_fit();

		actualConcentrationIndices.shrink_to_fit();
		TauConcentrationIndices.shrink_to_fit();

		concentrations.shrink_to_fit();

		for (int i = 0; i < referenceStateCalculationIndices.size(); i++) {
			referenceStateCalculationIndices[i].shrink_to_fit();
		}
		referenceStateCalculationIndices.shrink_to_fit();
		referenceStateType.shrink_to_fit();

		lowerBoundIndexForCOSMOSPACECalculation.shrink_to_fit();
		upperBoundIndexForCOSMOSPACECalculation.shrink_to_fit();

		temperatures.shrink_to_fit();
	}
};

/* this class is needed to catch exceptions in the OPENMP threads and rethrow them after the parallel section */
class threadException {
	std::exception_ptr Ptr;
	std::mutex         Lock;
public:
	threadException() : Ptr(nullptr) {}

	void rethrow() {
		if (this->Ptr) std::rethrow_exception(this->Ptr);
	}
	void capture_exception() {
		std::unique_lock<std::mutex> guard(this->Lock);
		this->Ptr = std::current_exception();
	}

	template <typename Function, typename... Parameters>
	void run(Function f, Parameters... params)
	{
		if (!this->Ptr) {
			try
			{
				f(params...);
			}
			catch (...)
			{
				capture_exception();
			}
		}
	}
};