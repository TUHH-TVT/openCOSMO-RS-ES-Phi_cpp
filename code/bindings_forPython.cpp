/*
	c++ implementation of openCOSMO-RS including multiple segment descriptors
	@author: Simon Mueller, 2022
*/

#define USE_DOUBLE

//#define PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

#include "general.hpp"
#include "core_functions.hpp"

void displayOnPython(std::string message) {
	py::print(message, "end"_a = "");
}

void displayTimeOnPython(std::string message, unsigned long durationInMicroseconds) {
	displayOnPython(message + ": " + std::to_string(durationInMicroseconds) + " microseconds\n");
}


namespace {

	std::vector<std::vector<double>>
		list_to_vector_of_vectors(const py::handle& obj)
	{
		if (!py::isinstance<py::list>(obj))
			throw std::runtime_error("Expected a list of lists");

		py::list outer = py::cast<py::list>(obj);

		std::vector<std::vector<double>> result;
		result.reserve(outer.size());

		for (const py::handle& item : outer) {

			if (!py::isinstance<py::list>(item))
				throw std::runtime_error("Expected inner elements to be lists");

			py::list inner = py::cast<py::list>(item);

			std::vector<double> vec;
			vec.reserve(inner.size());

			for (const py::handle& value : inner) {
				vec.push_back(py::cast<double>(value));
			}

			result.push_back(std::move(vec));
		}

		return result;
	}

}


void initializeOnPython() {
	display = displayOnPython;
	displayTime = displayTimeOnPython;

	initialize(param);
}

void loadParametersOnPython(py::dict parameters) {

	py::gil_scoped_acquire acquire;

	param.Aeff = parameters["Aeff"].cast<double>();
	param.alpha = parameters["alpha"].cast<double>();

	param.CHB = parameters["CHB"].cast<double>();
	param.CHBT = parameters["CHBT"].cast<double>();
	param.SigmaHB = parameters["SigmaHB"].cast<double>();

	param.Rav = parameters["Rav"].cast<double>();

	if (param.sw_SR_polarizabilities > 0) {
		param.E_F_corr = parameters["E_F_corr"].cast<double>();
		param.m_vdW = parameters["m_vdW"].cast<double>();
	}

	if (param.sw_misfit > 0) {
		param.fCorr = parameters["fCorr"].cast<double>();
		param.RavCorr = parameters["RavCorr"].cast<double>();
	}

	if (param.sw_combTerm == 1 || param.sw_combTerm == 3) {
		param.comb_SG_A_std = parameters["comb_SG_A_std"].cast<double>();
		param.comb_SG_z_coord = parameters["comb_SG_z_coord"].cast<double>();

		if (param.sw_combTerm == 3)
			param.comb_modSG_exp = parameters["comb_modSG_exp"].cast<double>();
	}

	if (param.sw_combTerm == 2 || param.sw_combTerm == 5) {
		param.comb_lambda0 = parameters["comb_lambda0"].cast<double>();
		param.comb_lambda1 = parameters["comb_lambda1"].cast<double>();
		param.comb_lambda2 = parameters["comb_lambda2"].cast<double>();
	}

	if (param.sw_combTerm == 4) {
		param.comb_SGG_lambda = parameters["comb_SGG_lambda"].cast<double>();
		param.comb_SGG_beta = parameters["comb_SGG_beta"].cast<double>();
	}

	if (parameters.contains("dGsolv_eta")) {
		param.dGsolv_eta = parameters["dGsolv_eta"].cast<double>();
		param.dGsolv_omega_ring = parameters["dGsolv_omega_ring"].cast<double>();

		py::dict dGsolv_tau = parameters["dGsolv_tau"];

		for (auto item : dGsolv_tau) {
			std::string key = item.first.cast<std::string>();
			param.dGsolv_tau[std::stoi(key)] = item.second.cast<double>();
		}
	}

	if (param.sw_phi == 1) {
		param.hole_area = parameters["hole_area"].cast<double>();
	}

	if (parameters.contains("radii")) {
		py::dict radii = parameters["radii"];

		for (auto item : radii) {
			std::string key = item.first.cast<std::string>();
			param.R_i[std::stoi(key)] = item.second.cast<double>();
		}
	}

	// parameters for openCOSMO-RS-ES-Phi
	if (param.sw_electrolyte == 1) {
		param.salt_diam_sc_M = exp(parameters["salt_diam_sc_M"].cast<double>());
		param.salt_diam_sc_P = exp(parameters["salt_diam_sc_P"].cast<double>());

		param.msf_cP_org = exp(parameters["msf_cP_org"].cast<double>());
		param.msf_aP_org = exp(parameters["msf_aP_org"].cast<double>());
		param.msf_aM_org = exp(parameters["msf_aM_org"].cast<double>());
		param.msf_cM_org = exp(parameters["msf_cM_org"].cast<double>());

		param.fHB_cP_org = exp(parameters["fHB_cP_org"].cast<double>());
		param.fHB_aP_XH = exp(parameters["fHB_aP_XH"].cast<double>());
		param.fHB_A_org = exp(parameters["fHB_A_org"].cast<double>());
		param.fHB_cM_org = exp(parameters["fHB_cM_org"].cast<double>());
		param.fHB_aM_org = exp(parameters["fHB_aM_org"].cast<double>());
		param.C1 = exp(parameters["ln_C1"].cast<double>());
		param.DCO = exp(parameters["DCO"].cast<double>());

		param.fHB_cP_H2O = exp(parameters["fHB_cP_H2O"].cast<double>());
		param.fHB_aO_H2O = exp(parameters["fHB_aO_H2O"].cast<double>());
		param.fHB_aP_H2O = exp(parameters["fHB_aP_H2O"].cast<double>());
		param.fHB_cM_H2O = exp(parameters["fHB_cM_H2O"].cast<double>());
		param.fHB_aM_H2O = exp(parameters["fHB_aM_H2O"].cast<double>());

		param.msf_cP_H2O = exp(parameters["msf_cP_H2O"].cast<double>());
		param.msf_cM_H2O = exp(parameters["msf_cM_H2O"].cast<double>());
		param.msf_aP_H2O = exp(parameters["msf_aP_H2O"].cast<double>());
		param.msf_aM_H2O = exp(parameters["msf_aM_H2O"].cast<double>());
		param.C0 = exp(parameters["ln_C0"].cast<double>());
		param.DCW = exp(parameters["DCW"].cast<double>());

		param.msf_cP_aP = exp(parameters["msf_cP_aP"].cast<double>());
		param.msf_cP_aM = exp(parameters["msf_cP_aM"].cast<double>());
		param.msf_cM_aP = exp(parameters["msf_cM_aP"].cast<double>());
		param.msf_cM_aM = exp(parameters["msf_cM_aM"].cast<double>());

		param.fHB_cM_aM = exp(parameters["fHB_cM_aM"].cast<double>());
		param.fHB_cM_aP = exp(parameters["fHB_cM_aP"].cast<double>());
		param.fHB_cP_aM = exp(parameters["fHB_cP_aM"].cast<double>());
		param.fHB_cP_aP = exp(parameters["fHB_cP_aP"].cast<double>());
		param.fHB_iP_iP = exp(parameters["fHB_iP_iP"].cast<double>());

		param.D1 = exp(parameters["ln_D1"].cast<double>());
		param.E1 = exp(parameters["ln_E1"].cast<double>());
		param.D2 = exp(parameters["ln_D2"].cast<double>());
		param.E2 = exp(parameters["ln_E2"].cast<double>());
		param.C2 = exp(parameters["ln_C2"].cast<double>());

		param.omega0 = parameters["omega0"].cast<double>();
		param.omega1 = parameters["omega1"].cast<double>();

	}


	// experimental parameters for prototyping
	if (parameters.contains("exp")) {
		py::dict exp = parameters["exp"];
		for (auto item : exp)
			param.exp_param[item.first.cast<std::string>()] = item.second.cast<double>();
	}

	// hyperparameters for algorithms
	if (param.sw_phi == 1) {

		param.useGivenInitialK = false;
		param.noTracing = false;

		if (param.sw_PT_Flash == 1) {
			if (parameters.contains("parameters_PT_Flash")) {
				py::dict PT_Flash = parameters["parameters_PT_Flash"];
				if (PT_Flash.contains("useGivenInitialK")) {
					param.useGivenInitialK = PT_Flash["useGivenInitialK"].cast<bool>();
				}
				if (PT_Flash.contains("useGivenInitialK")) {
					param.noTracing = PT_Flash["noTracing"].cast<bool>();
				}
			}
		}

		if (param.sw_excess == 1) {
			if (parameters.contains("parameters_mixing")) {
				py::dict mixing = parameters["parameters_mixing"];

				param.temperatures_mixing = {};
				param.pressures_mixing = {};
				param.concentration_x_phase_mixing = {};
				param.concentration_y_phase_mixing = {};
				if (mixing.contains("temperatures")) {
					param.temperatures_mixing =
						mixing["temperatures"].cast<std::vector<double>>();
				}

				if (mixing.contains("pressures")) {
					param.pressures_mixing =
						mixing["pressures"].cast<std::vector<double>>();
				}

				if (mixing.contains("concentration_x_phase")) {
					param.concentration_x_phase_mixing =
						list_to_vector_of_vectors(mixing["concentration_x_phase"]);
				}

				if (mixing.contains("concentration_y_phase")) {
					param.concentration_y_phase_mixing =
						list_to_vector_of_vectors(mixing["concentration_y_phase"]);
				}

				const size_t n = param.temperatures_mixing.size();

				auto require_same_size = [&](const auto& v, const char* name) {
					if (!v.empty() && v.size() != n) {
						throw std::runtime_error(
							std::string("parameters_mixing: '") + name +
							"' must have the same length as 'temperatures'");
					}
				};

				require_same_size(param.pressures_mixing, "pressures");
				require_same_size(param.concentration_x_phase_mixing, "concentration_x_phase");
				require_same_size(param.concentration_y_phase_mixing, "concentration_y_phase");
			}
		}
	}

}

void loadMoleculesOnPython(py::dict options, py::dict parameters, py::list componentPaths) {

	py::gil_scoped_acquire acquire;

	// if uninitialized
	if (n_ex == -1) {
		initializeOnPython();
	}

	n_ex += 1;

	if (n_ex != 1) {
		throw std::runtime_error("loadMolecules should only be executed once after calling initiate.");
	}

	// options
	param.sw_phi = options["sw_phi"].cast<int>();
	param.sw_electrolyte = options["sw_electrolyte"].cast<int>();
	param.sw_Born_term = options["sw_Born_term"].cast<int>();
	if (param.sw_electrolyte == 1) {
		//param.sw_LR_model = options["sw_LR_model"].cast<int>();
		param.sw_permittivity_mix_model = options["sw_permittivity_mix_model"].cast<int>();

		if (param.sw_permittivity_mix_model == 0) {
			if (parameters.contains("permittivity_model_0_parameters")) {
				py::array_t<double> tempArray = py::array_t<double>(parameters["permittivity_model_0_parameters"]);
				if (tempArray.size() > 0) {

					new (&param.permittivity_model_0_parameters) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
						componentPaths.size(),
						3);
				}
			}
			else {
				throw std::runtime_error("Pure Component Parameters for the mixture permittivity model of Rueben et al. not provided although you specified to use it.");
			}
		}
	}
	param.sw_PT_Flash = options["sw_PT_Flash"].cast<int>();
	param.sw_checkAndCalcBinaryLLE = options["sw_checkAndCalcBinaryLLE"].cast<int>();
	param.sw_lnGamma = options["sw_lnGamma"].cast<int>();
	param.sw_partitionCoeff = options["sw_partitionCoeff"].cast<int>();
	param.sw_excess = options["sw_excess"].cast<int>();
	param.sw_isotherm = options["sw_isotherm"].cast<int>();
	param.sw_Pbub = options["sw_Pbub"].cast<int>();
	param.sw_Pdew = options["sw_Pdew"].cast<int>();
	param.sw_Tbub = options["sw_Tbub"].cast<int>();
	param.sw_Tdew = options["sw_Tdew"].cast<int>();
	param.sw_criticalPoint = options["sw_criticalPoint"].cast<int>();
	param.sw_azeotropicPoint = options["sw_azeotropicPoint"].cast<int>();
	param.sw_criticalPoint_specifiedx = options["sw_criticalPoint_specifiedx"].cast<int>();
	param.sw_combTerm = options["sw_SR_combTerm"].cast<int>();
	param.sw_alwaysCalculateSizeRelatedParameters = options["sw_SR_alwaysCalculateSizeRelatedParameters"].cast<int>();
	param.sw_alwaysReloadSigmaProfiles = options["sw_SR_alwaysReloadSigmaProfiles"].cast<int>();
	param.sw_useSegmentReferenceStateForInteractionMatrix = options["sw_SR_useSegmentReferenceStateForInteractionMatrix"].cast<int>();

	param.sw_calculateContactStatisticsAndAdditionalProperties = options["sw_SR_calculateContactStatisticsAndAdditionalProperties"].cast<int>();

	param.sw_differentiateHydrogens = options["sw_SR_differentiateHydrogens"].cast<int>();
	param.sw_differentiateMoleculeGroups = options["sw_SR_differentiateMoleculeGroups"].cast<int>();
	param.sw_COSMOfiles_type = options["sw_SR_COSMOfiles_type"].cast<std::string>();
	param.sw_SR_polarizabilities = options["sw_SR_polarizabilities"].cast<int>();

	if (param.sw_calculateContactStatisticsAndAdditionalProperties != 0) {
		py::list partialInteractionMatrices = options["sw_SR_partialInteractionMatrices"];
		param.numberOfPartialInteractionMatrices = int(partialInteractionMatrices.size());
	}
	else {
		param.numberOfPartialInteractionMatrices = 0;
	}

	param.sw_atomicNumber = options["sw_SR_atomicNumber"].cast<int>();
	param.sw_misfit = options["sw_SR_misfit"].cast<int>();

	if (param.sw_misfit < 0 && param.sw_misfit > 2) {
		throw std::runtime_error("sw_SR_misfit should have one of the following values: [0, 1, 2].");
	}
	param.sw_skip_COSMOSPACE_errors = options["sw_skip_COSMOSPACE_errors"].cast<int>();

	// parameters
	loadParametersOnPython(parameters);

	for (auto componentPath : componentPaths) {
		molecule newMolecule = loadNewMolecule(param, componentPath.cast<std::string>());
		newMolecule.index = int(molecules.size());
		molecules.push_back(std::make_shared<molecule>(newMolecule));
	}

	if (param.sw_phi == 1) {

		molecule holeMolecule = createHoleMolecule(param);
		molecules.push_back(std::make_shared<molecule>(holeMolecule));

		if (parameters.contains("pure_component_parameters")) {
			py::array_t<double> tempArray = py::array_t<double>(parameters["pure_component_parameters"]);
			if (tempArray.size() > 0) {

				new (&param.phi_param) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
					componentPaths.size(),
					4);
			}
		}
		else {
			throw std::runtime_error("pure_component_parameters were not provided for the EOS.");
		}
	}

	
	if (molecules.size() == 1) {
		throw std::runtime_error("Please load at least one molecule.");
	}

}

void loadCalculationsOnPython(py::list calculationsOnPython, bool reload = false) {

	py::gil_scoped_acquire acquire;

	n_ex += 1;

	if (reload) {
		initialize(param, false, false, true, false);
	}
	else {
		if (n_ex != 2) {
			throw std::runtime_error("loadCalculations should only be executed once after calling loadMolecules.");
		}
	}

	// load the calculations
	// calculationsOnPython list of dictionaries
	// calculationsList[i]["components"] = np.array([0 2])
	// calculationsList[i]["temperatures"] = np.array([298.15])
	// calculationsList[i]["concentrations"] = np.array([0.2 0.8; 0.5 0.5; 0.1 0.9])
	// calculationsList[i]["reference_state_types"] = np.array([0 2 3])  0: PureComponents | 1: PureComponentsOnlyNeutral | 2: ReferenceMixture (also for InfiniteDilution) | 3: COSMO
	// calculationsList[i]["reference_state_concentrations"] = np.array([[] [] [0 1]])  # np.array of concentrations where referenceStateType == 2

	const size_t numCalcs = calculationsOnPython.size();

	if (numCalcs == 0) {
		throw std::runtime_error("Please specify at least one calculation.");
	}

	// important as otherwise the data behind Eigen::Maps
	// can be lost when the vector is resized.
	calculations.reserve(numCalcs);

	for (int i = 0; i < numCalcs; i++) {

		py::dict calculationDict = calculationsOnPython[i];
		int numberOfComponentsToLoadSegmentsOf;

		// array of component indices
		py::list componentList = calculationDict["component_indices"];
		int numberOfComponents = int(componentList.size());

		// array of stochiometric coefficients
		py::list StochCoeffs;
		if (param.sw_electrolyte == 1) {
			StochCoeffs = calculationDict["component_stochiometric_coefficients"];
		}

		numberOfComponentsToLoadSegmentsOf = numberOfComponents;
		calculation newCalculation = calculation(numberOfComponentsToLoadSegmentsOf);

		if (param.sw_phi == 1) {
			numberOfComponentsToLoadSegmentsOf = numberOfComponents + 1; // add component for hole
			newCalculation = calculation(numberOfComponentsToLoadSegmentsOf);
			newCalculation.calculationType = calculationDict["type"].cast<std::string>();
		}

		for (int j = 0; j < numberOfComponentsToLoadSegmentsOf; j++) {
			int moleculeIndex;
			if (j < numberOfComponents)
				moleculeIndex = componentList[j].cast<int>();
			else
				moleculeIndex = int(molecules.size()) - 1;

			std::shared_ptr<molecule> thisMolecule = molecules[moleculeIndex];


			for (int k = 0; k < thisMolecule->segments.size(); k++) {
				newCalculation.segments.add((unsigned short)j, thisMolecule->segments.SegmentTypeGroup[k],
					thisMolecule->segments.SegmentTypeSigma[k],
					thisMolecule->segments.SegmentTypeSigmaCorr[k],
					thisMolecule->segments.SegmentTypeHBtype[k],
					thisMolecule->segments.SegmentTypeAtomicNumber[k],
					thisMolecule->segments.SegmentTypeAreas[k][0],
					thisMolecule->index,
					thisMolecule->segments.SegmentTypeAtomicPolariz[k]);
			}

			newCalculation.components.push_back(thisMolecule);
			newCalculation.moleculeIndices[j] = moleculeIndex;

			if (param.sw_electrolyte == 1) {
				if (j < numberOfComponents)
					newCalculation.moleculeStochCoeffs[j] = StochCoeffs[j].cast<int>();
				else
					newCalculation.moleculeStochCoeffs[j] = 0;
			}
		}

		newCalculation.segments.sort();
		newCalculation.segments.shrink_to_fit();

		// concentrations and temperatures
		auto temperatures = py::array_t<double>(calculationDict["temperatures"]).unchecked<1>();
		auto concentrations = py::array_t<double>(calculationDict["concentrations"]).unchecked<2>();

		for (int j = 0; j < (size_t)concentrations.shape(0); j++) {

			std::vector<double> rowConcentration;

			double tempSumOfConcentrations = 0.0;
			for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {
				double val = 0.0;
				if (k < numberOfComponents)
					val = (double)concentrations(j, k);
				tempSumOfConcentrations += val;
				rowConcentration.push_back(val);
			}

			if (abs(1.0 - tempSumOfConcentrations) > MAX_CONCENTRATION_DIFF_FROM_ZERO && (param.sw_phi == 1 != int(numberOfComponents) > 1)) {
				throw std::runtime_error("For calculation number " + std::to_string(i) + ", the concentrations do not add up to unity. residual concentration: " + std::to_string(abs(1.0f - tempSumOfConcentrations)));
			}

			double temperature = (double)temperatures(j);

			newCalculation.temperatures.push_back(temperature);
			newCalculation.concentrations.push_back(rowConcentration);
		}

		auto referenceStateConcentrations = py::array_t<double>(calculationDict["reference_state_concentrations"]).unchecked<2>();

		if (referenceStateConcentrations.shape(0) != newCalculation.concentrations.size()) {
			throw std::runtime_error("concentrations and referenceStateConcentrations of calculation number " + std::to_string(i) + " have different sizes.\n");
		}

		newCalculation.originalNumberOfCalculations = (unsigned short)newCalculation.concentrations.size();

		// reference states
		auto referenceStateTypes = py::array_t<int>(calculationDict["reference_state_types"]).unchecked<1>();
		for (int j = 0; j < (size_t)referenceStateTypes.shape(0); j++) {


			int referenceStateType = referenceStateTypes(j);

			newCalculation.referenceStateType.push_back((unsigned short)referenceStateType);

			double tempSumOfConcentrations = 0;
			if (calculationDict.contains("reference_state_concentrations")) {
				for (int k = 0; k < (size_t)referenceStateConcentrations.shape(1); k++) {
					tempSumOfConcentrations += (double)referenceStateConcentrations(j, k);
				}
			}


			if (referenceStateType == 0) { // Pure component

				if (tempSumOfConcentrations != 0) {
					throw std::runtime_error("A reference state concentration was specified for a calculation with reference state PureComponents, this does not make sense.");
				}

				std::vector<int> thisReferenceStateCalculationIndices;
				// full padded width, hole column included
				for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {

					std::vector<double> referenceStateConcentration;
					for (int m = 0; m < numberOfComponentsToLoadSegmentsOf; m++) {
						referenceStateConcentration.push_back(k == m ? 1.0f : 0.0f);
					}

					double temperature = (double)temperatures(j);
					int referenceStateCalculationIndex = (int)newCalculation.addOrFindArrayIndexForConcentration(referenceStateConcentration, temperature);
					thisReferenceStateCalculationIndices.push_back(referenceStateCalculationIndex);

				}
				newCalculation.referenceStateCalculationIndices.push_back(thisReferenceStateCalculationIndices);
			}
			else if (referenceStateType == 1) { // untested: Pure component only neutral

				if (tempSumOfConcentrations != 0) {
					throw std::runtime_error("A reference state concentration was specified for a calculation with reference state PureComponentsOnlyNeutral, this does not make sense.");
				}

				std::vector<int> thisReferenceStateCalculationIndices;
				for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {

					if (newCalculation.components[k]->moleculeCharge == 0) {

						std::vector<double> referenceStateConcentration;
						for (int m = 0; m < numberOfComponentsToLoadSegmentsOf; m++) {
							referenceStateConcentration.push_back(k == m ? 1.0f : 0.0f);
						}

						double temperature = (double)temperatures(j);
						int referenceStateCalculationIndex = (int)newCalculation.addOrFindArrayIndexForConcentration(referenceStateConcentration, temperature);
						thisReferenceStateCalculationIndices.push_back(referenceStateCalculationIndex);
					}
					else {
						thisReferenceStateCalculationIndices.push_back(-1);
					}

				}
				newCalculation.referenceStateCalculationIndices.push_back(thisReferenceStateCalculationIndices);
			}
			else if (referenceStateType == 2) { // Reference mixture

				std::vector<double> referenceStateConcentration;
				for (int m = 0; m < numberOfComponents; m++) {
					referenceStateConcentration.push_back((double)referenceStateConcentrations(j, m));
				}
				while ((int)referenceStateConcentration.size() < numberOfComponentsToLoadSegmentsOf) {
					referenceStateConcentration.push_back(0.0);
				}

				if (referenceStateConcentrations.shape(1) != (size_t)numberOfComponents) {
					throw std::runtime_error("A reference state concentration was specified with the wrong amount of concentrations.");
				}

				if (abs(1.0f - tempSumOfConcentrations) > MAX_CONCENTRATION_DIFF_FROM_ZERO) {
					throw std::runtime_error("For calculation number " + std::to_string(i) + ", the reference concentrations do not add up to unity. residual concentration: " + std::to_string(abs(1.0f - tempSumOfConcentrations)));
				}

				double temperature = (double)temperatures(j);
				int referenceStateCalculationIndex = (int)newCalculation.addOrFindArrayIndexForConcentration(referenceStateConcentration, temperature);

				std::vector<int> thisReferenceStateCalculationIndices;
				for (int m = 0; m < numberOfComponentsToLoadSegmentsOf; m++) {
					thisReferenceStateCalculationIndices.push_back(referenceStateCalculationIndex);
				}
				newCalculation.referenceStateCalculationIndices.push_back(thisReferenceStateCalculationIndices);
			}
			else if (referenceStateType == 3 || referenceStateType == 4 || referenceStateType == 5) { // 3: COSMO 4: COSMO for solvation energy 5: ideal gas

				if (tempSumOfConcentrations != 0) {
					throw std::runtime_error("A reference state concentration was specified for a calculation with reference state COSMO, this does not make sense.");
				}

				std::vector<int> thisReferenceStateCalculationIndices;
				if (referenceStateType == 5) {
					for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {

						std::vector<double> referenceStateConcentration(numberOfComponentsToLoadSegmentsOf, 0.0f);
						referenceStateConcentration[numberOfComponentsToLoadSegmentsOf - 1] = 1.0f;

						double temperature = (double)temperatures(j);
						int referenceStateCalculationIndex = (int)newCalculation.addOrFindArrayIndexForConcentration(referenceStateConcentration, temperature);
						thisReferenceStateCalculationIndices.push_back(referenceStateCalculationIndex);
					}
				}
				else {
					thisReferenceStateCalculationIndices = std::vector<int>(numberOfComponents, -1);
				}
				newCalculation.referenceStateCalculationIndices.push_back(thisReferenceStateCalculationIndices);
			}
			else {
				throw std::runtime_error("An unknown reference state type was given.");
			}
		}

		// directly bind to python numpy arrays
		// for this to work correctly the sizes of the n-dimensional numpy arrays and the type
		// must be the same in python and c++ (double) with same storage order: row major.

		py::array_t<double> tempArray = py::array_t<double>(calculationDict["concentrations"]);
		new (&newCalculation.originalConcentrations) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
			int(newCalculation.originalNumberOfCalculations),
			numberOfComponents);

		if (param.sw_partitionCoeff == 1 || param.useGivenInitialK == true) {
			if (calculationDict.contains("partition_coefficients") == false) {
				throw std::runtime_error("You chose to provide initial K values, but no partition_coefficients entry found in calculationDict.");
			}
			py::array_t<double> tempArray = py::array_t<double>(calculationDict["partition_coefficients"]);
			new (&newCalculation.calculatedPartitionCoefficients) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponents);
		};

		tempArray = py::array_t<double>(calculationDict["ln_gamma_x_SR_residual_calc"]);
		new (&newCalculation.lnGammaResidual) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
			int(newCalculation.originalNumberOfCalculations),
			numberOfComponentsToLoadSegmentsOf);

		tempArray = py::array_t<double>(calculationDict["ln_gamma_x_SR_combinatorial_calc"]);
		new (&newCalculation.lnGammaCombinatorial) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
			int(newCalculation.originalNumberOfCalculations),
			numberOfComponentsToLoadSegmentsOf);

		tempArray = py::array_t<double>(calculationDict["ln_gamma_x_SR_calc"]);
		new (&newCalculation.lnGammaTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
			int(newCalculation.originalNumberOfCalculations),
			numberOfComponentsToLoadSegmentsOf);

		if (param.sw_electrolyte == 1) {
			tempArray = py::array_t<double>(calculationDict["ln_gamma_x_LR_calc"]);
			new (&newCalculation.lnGammaLR) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponentsToLoadSegmentsOf);

			tempArray = py::array_t<double>(calculationDict["ln_gamma_x_Born_calc"]);
			new (&newCalculation.lnGammaBorn) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponentsToLoadSegmentsOf);

			tempArray = py::array_t<double>(calculationDict["ln_gamma_x_Electrostatic_calc"]);
			new (&newCalculation.lnGammaElectrostatic) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponentsToLoadSegmentsOf);

			tempArray = py::array_t<double>(calculationDict["ln_gamma_x_SRLR_calc"]);
			new (&newCalculation.lnGammaSRLRTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponentsToLoadSegmentsOf);

			tempArray = py::array_t<double>(calculationDict["MIAC_x_calc"]);
			new (&newCalculation.MIACInMoleFractionScale) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponentsToLoadSegmentsOf);

			tempArray = py::array_t<double>(calculationDict["MIAC_m_calc"]);
			new (&newCalculation.MIACInMolalityScale) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponentsToLoadSegmentsOf);
		}


		if (calculationDict.contains("dGsolv")) {
			tempArray = py::array_t<double>(calculationDict["dGsolv"]);
			new (&newCalculation.dGsolv) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				1);
		}

		if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {

			tempArray = py::array_t<double>(calculationDict["contact_statistics"]);
			new (&newCalculation.contactStatistics) Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponents,
				numberOfComponents);

			tempArray = py::array_t<double>(calculationDict["average_surface_energies"]);
			new (&newCalculation.averageSurfaceEnergies) Eigen::TensorMap<Eigen::Tensor<double, 4, Eigen::RowMajor>>(tempArray.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				int(param.numberOfPartialInteractionMatrices) + 1, // +1 because A_int is the first one
				numberOfComponents,
				numberOfComponents);

			if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {

				tempArray = py::array_t<double>(calculationDict["partial_molar_energies"]);
				new (&newCalculation.partialMolarEnergies) Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>>(tempArray.mutable_data(),
					int(newCalculation.originalNumberOfCalculations),
					int(param.numberOfPartialInteractionMatrices) + 1, // +1 because A_int is the first one
					numberOfComponents);
			}

		}

		newCalculation.number = (int)i;
		finishCalculationInitiation(newCalculation, param);

		if (param.sw_electrolyte == 1) {
			// Calculation inputs relevant to use of openCOSMO-RS-ES or openCOMO-RS-ES-Phi
			if (calculationDict.contains("salt_diam")) {
				newCalculation.salt_diam = calculationDict["salt_diam"].cast<double>();
			}
			if (calculationDict.contains("ion_diam_mono")) {
				newCalculation.ion_diam_mono = calculationDict["ion_diam_mono"].cast<double>();
			}
		}


		if (param.sw_phi == 1) {

			py::array_t<double> tempArray2 = py::array_t<double>(calculationDict["repulsive_pressure"]);
			new (&newCalculation.repulsivePressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations));

			tempArray2 = py::array_t<double>(calculationDict["attractive_pressure"]);
			new (&newCalculation.attractivePressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations));

			if (param.sw_electrolyte == 1) {
				tempArray2 = py::array_t<double>(calculationDict["Born_pressure"]);
				new (&newCalculation.BornPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["electrostatic_pressure"]);
				new (&newCalculation.electrostaticPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));
			}

			tempArray2 = py::array_t<double>(calculationDict["total_pressure"]);
			new (&newCalculation.totalPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations));

			tempArray2 = py::array_t<double>(calculationDict["target_pressure"]);
			new (&newCalculation.targetPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations));

			tempArray2 = py::array_t<double>(calculationDict["ln_phi"]);
			new (&newCalculation.lnPhiTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponents);

			tempArray2 = py::array_t<double>(calculationDict["ln_phi_repulsive"]);
			new (&newCalculation.lnPhiRepulsive) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponents);

			tempArray2 = py::array_t<double>(calculationDict["ln_phi_attractive"]);
			new (&newCalculation.lnPhiAttractive) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations),
				numberOfComponents);

			if (param.sw_electrolyte == 1) {
				tempArray2 = py::array_t<double>(calculationDict["ln_phi_electrostatic"]);
				new (&newCalculation.lnPhiElectrostatic) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations),
					numberOfComponents);

				tempArray2 = py::array_t<double>(calculationDict["ln_phi_Born"]);
				new (&newCalculation.lnPhiBorn) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations),
					numberOfComponents);
			}

			tempArray2 = py::array_t<double>(calculationDict["molar_volumes"]);
			new (&newCalculation.molarVolume) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
				int(newCalculation.originalNumberOfCalculations));

			if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
				tempArray2 = py::array_t<double>(calculationDict["temperatures"]);
				new (&newCalculation.targetTemperature) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));
			}

			if (param.sw_excess == 1) {
				tempArray2 = py::array_t<double>(calculationDict["gE"]);
				new (&newCalculation.gE) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["hE"]);
				new (&newCalculation.hE) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["cpE"]);
				new (&newCalculation.cpE) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["gM"]);
				new (&newCalculation.gM) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["hM"]);
				new (&newCalculation.hM) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["cpM"]);
				new (&newCalculation.cpM) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

			}

			if (param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1 || param.sw_criticalPoint_specifiedx == 1) {
				tempArray2 = py::array_t<double>(calculationDict["critical_P"]);
				new (&newCalculation.criticalP) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray2 = py::array_t<double>(calculationDict["critical_T"]);
				new (&newCalculation.criticalT) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(tempArray2.mutable_data(),
					int(newCalculation.originalNumberOfCalculations));

				tempArray = py::array_t<double>(calculationDict["critical_x"]);
				new (&newCalculation.criticalx) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tempArray.mutable_data(),
					int(newCalculation.originalNumberOfCalculations),
					numberOfComponents);
			}

			newCalculation.molarResidualEnthalpy = VectorCalcType::Zero(newCalculation.originalNumberOfCalculations);
			newCalculation.residualHeatCapacityAtConstantPressure = VectorCalcType::Zero(newCalculation.originalNumberOfCalculations);
			newCalculation.lnPhiTotalPDETAtConstantPx = MatrixCalcType::Zero(newCalculation.originalNumberOfCalculations, newCalculation.components.size());
			newCalculation.lnGammaResidualPDETatConstantvx = MatrixCalcType::Zero(int(newCalculation.originalNumberOfCalculations), numberOfComponentsToLoadSegmentsOf);
			newCalculation.lnGammaResidualSecondPDETatConstantvx = MatrixCalcType::Zero(int(newCalculation.originalNumberOfCalculations), numberOfComponentsToLoadSegmentsOf);
			newCalculation.lnGammaResidualPDEvatConstantTx = MatrixCalcType::Zero(int(newCalculation.originalNumberOfCalculations), numberOfComponentsToLoadSegmentsOf);

			//if (numberOfComponents > 1) {
			//	auto concentrations = py::array_t<double>(calculationDict["concentrations"]).unchecked<2>();
			//	newCalculation.overallConcentrations = Eigen::MatrixXd(int(newCalculation.originalNumberOfCalculations), numberOfComponents);
			//	for (int j = 0; j < int(newCalculation.originalNumberOfCalculations); j++) {
			//		for (int k = 0; k < numberOfComponents; k++) {
			//			newCalculation.overallConcentrations(j, k) = (double)concentrations(j, k);
			//		}
			//	}
			//}

			//if (numberOfComponents > 1) {
			//		auto concentrations = py::array_t<double>(calculationDict["concentrations"]).unchecked<2>();
			//		for (int j = 0; j < int(newCalculation.originalNumberOfCalculations); j++) {
			//			for (int k = 0; k < numberOfComponents; k++) {
			//				newCalculation.originalConcentrations(j, k) = (double)concentrations(j, k);
			//			}
			//		}
			//	}

			newCalculation.recalculateConcentrationsForPhi = true;

		}



		calculations.push_back(newCalculation);

		// added
		if (param.sw_phi == 1 && numberOfComponents > 1) {

			if (calculations[i].calculationType == "VL" || param.sw_excess == 1) {

				if (param.sw_Pdew == 1 || param.sw_Pbub == 1 || param.sw_PT_Flash == 1 || param.sw_criticalPoint == 1 || param.sw_criticalPoint_specifiedx == 1 || param.sw_azeotropicPoint == 1 || param.sw_excess == 1) {
					calculations[i].uniqueTemperatures = calculations[i].temperatures;
					std::sort(calculations[i].uniqueTemperatures.begin(), calculations[i].uniqueTemperatures.end());
					calculations[i].uniqueTemperatures.erase(std::unique(calculations[i].uniqueTemperatures.begin(), calculations[i].uniqueTemperatures.end()), calculations[i].uniqueTemperatures.end());
					calculations[i].IndicesForPureComponentCalculations = Eigen::MatrixXi(int(calculations[i].uniqueTemperatures.size()), numberOfComponents);
				}

				if (param.sw_excess == 1) {
					calculations[i].IndicesForPureComponentCalculations = Eigen::MatrixXi(int(newCalculation.originalNumberOfCalculations), numberOfComponents);
				}

				if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
					for (int k = 0; k < int(newCalculation.originalNumberOfCalculations); k++) {
						calculations[i].uniquePressures.push_back(calculations[i].targetPressure(k));
					}
					std::sort(calculations[i].uniquePressures.begin(), calculations[i].uniquePressures.end());
					calculations[i].uniquePressures.erase(std::unique(calculations[i].uniquePressures.begin(), calculations[i].uniquePressures.end()), calculations[i].uniquePressures.end());
					calculations[i].IndicesForPureComponentCalculations = Eigen::MatrixXi(int(calculations[i].uniquePressures.size()), numberOfComponents);
				}

				for (int j = 0; j < numberOfComponents; j++) {

					// ions are non-volatile: no pure-component state (-1)
					if (calculations[i].components[j]->moleculeCharge != 0) {
						for (int k = 0; k < calculations[i].IndicesForPureComponentCalculations.rows(); k++) {
							calculations[i].IndicesForPureComponentCalculations(k, j) = -1;
						}
						continue;
					}

					std::vector<std::shared_ptr<molecule>> actualMolecules;
					actualMolecules.push_back(calculations[i].components[j]);
					actualMolecules.push_back(calculations[i].components[int(numberOfComponentsToLoadSegmentsOf) - 1]);
					int subcalculationIndex = -2;

					if (param.sw_Pdew == 1 || param.sw_Pbub == 1 || param.sw_PT_Flash == 1 || param.sw_criticalPoint == 1 || param.sw_criticalPoint_specifiedx == 1 || param.sw_azeotropicPoint == 1) {
						for (int k = 0; k < calculations[i].uniqueTemperatures.size(); k++) {
							std::vector<double> temperatureVector;
							temperatureVector.push_back(calculations[i].uniqueTemperatures[k]);
							subcalculationIndex = calculations[i].addOrFindArrayIndexInSubcalculations(subcalculations, actualMolecules, calculations[i].uniqueTemperatures[k]);
							if (subcalculationIndex == -1) {
								subcalculations.push_back(loadSubcalculation(actualMolecules, temperatureVector, { {1.0, 0.0} }));
								subcalculationIndex = int(subcalculations.size()) - 1;
							}
							calculations[i].IndicesForPureComponentCalculations(k, j) = subcalculationIndex;
						}
					}
					else if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
						for (int k = 0; k < calculations[i].uniquePressures.size(); k++) {
							std::vector<double> pressureVector;
							pressureVector.push_back(calculations[i].uniquePressures[k]);
							subcalculationIndex = calculations[i].addOrFindArrayIndexInSubcalculations(subcalculations, actualMolecules, -1, calculations[i].uniquePressures[k]);
							if (subcalculationIndex == -1) {
								subcalculations.push_back(loadSubcalculation(actualMolecules, { 100.0 }, { {1.0, 0.0} }, pressureVector)); // set T = 100 K as initial value for later pure component Tsat search; may be changed
								subcalculationIndex = int(subcalculations.size()) - 1;
							}
							calculations[i].IndicesForPureComponentCalculations(k, j) = subcalculationIndex;
						}
					}
					else {
						for (int k = 0; k < int(newCalculation.originalNumberOfCalculations); k++) {
							std::vector<double> temperatureVector;
							std::vector<double> pressureVector;
							temperatureVector.push_back(temperatures(k));
							pressureVector.push_back(calculations[i].targetPressure(k));
							subcalculationIndex = calculations[i].addOrFindArrayIndexInSubcalculations(subcalculations, actualMolecules, double(temperatures(k)), calculations[i].targetPressure(k));
							if (subcalculationIndex == -1) {
								subcalculations.push_back(loadSubcalculation(actualMolecules, temperatureVector, { {1.0, 0.0} }, pressureVector));
								subcalculationIndex = int(subcalculations.size()) - 1;
							}
							calculations[i].IndicesForPureComponentCalculations(k, j) = subcalculationIndex;
						}
						// necessary because the data behind the maps are lost
						for (int k = 0; k < int(subcalculations.size()); k++) {
							subcalculations[k].originalConcentrations = subcalculations[k].concentrations_data;
						}
					}
				}
			}
		}
	}
}

py::list calculateOnPython(py::dict parameters, py::list calculationsOnPython, bool reloadConcentrations = false, bool reloadReferenceConcentrations = false) {

	n_ex += 1;

	if (n_ex < 3) {
		throw std::runtime_error("Before trying to run a calculation please first execute initiate and loadMolecules");
	}
#ifdef MEASURE_TIME
	startCalculationMeasurement();
#endif

	py::gil_scoped_acquire acquire;

	loadParametersOnPython(parameters);

	if (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) {
		reloadAllMolecules();
	}

	if (param.sw_alwaysCalculateSizeRelatedParameters == 1 || (param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3)) {
		resizeMonoatomicCations(param, molecules);
	}

	if (param.sw_phi == 1 && int(subcalculations.size()) > 0) {
		const size_t numSubCalcs = subcalculations.size();
		std::vector<int> subcalculationIndices(numSubCalcs);
		for (int i = 0; i < numSubCalcs; i++) {
			subcalculationIndices[i] = i;
		}
		calculateSubcalculations(subcalculationIndices);
	}


	const size_t numCalcs = calculationsOnPython.size();
	std::vector<int> calculationIndices(numCalcs);

	for (int i = 0; i < numCalcs; i++) {

		calculationIndices[i] = calculationsOnPython[i]["index"].cast<int>();

		param.sw_reloadConcentrations = 0;
		param.sw_reloadReferenceConcentrations = 0;

		if (reloadConcentrations == true) {
			param.sw_reloadConcentrations = 1;
			auto concentrations = py::array_t<double>(calculationsOnPython[i]["concentrations"]).unchecked<2>();
			for (int h = 0; h < calculations[calculationIndices[i]].originalNumberOfCalculations; h++) {

				int j = calculations[calculationIndices[i]].actualConcentrationIndices[h];

				std::vector<double> rowConcentration;

				double tempSumOfConcentrations = 0;
				for (int k = 0; k < calculations[calculationIndices[i]].components.size(); k++) {
					double val = (double)concentrations(h, k);
					tempSumOfConcentrations += val;
					calculations[calculationIndices[i]].concentrations[j][k] = val;
				}

				if (abs(1.0f - tempSumOfConcentrations) > MAX_CONCENTRATION_DIFF_FROM_ZERO && (param.sw_phi == 1 != int(calculations[calculationIndices[i]].components.size()) > 2)) {
					throw std::runtime_error("A concentration does not add up to unity.");
				}
			}
		}
		if (reloadReferenceConcentrations == true) {
			param.sw_reloadReferenceConcentrations = 1;
			auto referenceStateTypes = py::array_t<int>(calculationsOnPython[i]["reference_state_types"]).unchecked<1>();
			auto referenceStateConcentrations = py::array_t<double>(calculationsOnPython[i]["reference_state_concentrations"]).unchecked<2>();


			for (int j = 0; j < (size_t)referenceStateConcentrations.shape(0); j++) {
				int referenceStateType = referenceStateTypes(j);
				if (referenceStateType != 2) {
					throw std::runtime_error("reloadReferenceConcentrations only makes sense if the referenceStateTypes == 2.\n");
				}
			}
			if (calculations[calculationIndices[i]].concentrations.size() != calculations[calculationIndices[i]].originalNumberOfCalculations * 2) {
				throw std::runtime_error("The implementation currently assumes that every concentratoin has a unique reference concentration, this could and should be changed in the future.\n");
			}

			for (int h = 0; h < calculations[calculationIndices[i]].originalNumberOfCalculations; h++) {

				std::vector<int> referenceStateCalculationIndices = calculations[calculationIndices[i]].referenceStateCalculationIndices[h];

				double tempSumOfConcentrations = 0;
				for (int k = 0; k < calculations[calculationIndices[i]].components.size(); k++) {
					int referenceStateCalculationIndex = referenceStateCalculationIndices[k];
					double val = (double)referenceStateConcentrations(h, k);
					tempSumOfConcentrations += val;
					calculations[calculationIndices[i]].concentrations[referenceStateCalculationIndex][k] = val;
				}

				if (abs(1.0f - tempSumOfConcentrations) > MAX_CONCENTRATION_DIFF_FROM_ZERO) {
					throw std::runtime_error("A concentration does not add up to unity.");
				}
			}
		}
	}
	calculate(calculationIndices);
	if (param.sw_criticalPoint != 1 && param.sw_azeotropicPoint != 1 && param.sw_criticalPoint_specifiedx != 1) {
		for (int i = 0; i < int(numCalcs); i++) {
			if (calculationsOnPython[i].contains("type")) {
				for (int m = 0; m < calculations[calculationIndices[i]].originalConcentrations.rows(); m++) {
					for (int n = 0; n < calculations[calculationIndices[i]].components.size() - 1; n++) {
						auto concentrations = calculations[calculationIndices[i]].originalConcentrations;
						calculationsOnPython[i]["concentrations"] = py::array(concentrations.size(), concentrations.data());
					}
				}
			}

		}
	}


#ifdef MEASURE_TIME
	stopCalculationMeasurement();
#endif



return calculationsOnPython;

}

#ifdef USE_DOUBLE
PYBIND11_MODULE(openCOSMORS, m) {
#else
PYBIND11_MODULE(openCOSMORS, m) {
#endif
	m.doc() = R"pbdoc(
        openCOSMO-RS
        -----------------------

        .. currentmodule:: openCOSMORS

        .. autosummary::
           :toctree: _generate

           initialize
    )pbdoc";

	m.def("initialize", &initializeOnPython, R"pbdoc(
        Sets the stage to start running the module again.
    )pbdoc");

	m.def("loadMolecules", &loadMoleculesOnPython, R"pbdoc(
        Loads all the sigma profiles of the molecules.
		This needs to be called before calling loadCalculations.
    )pbdoc");

	m.def("loadCalculations", &loadCalculationsOnPython, py::arg("calculationsOnPython"), py::arg("reload") = false, R"pbdoc(
        Loads all calculations.
		This needs to be called before calling calculate.
    )pbdoc");

	m.def("calculate", &calculateOnPython, py::arg("parameters"), py::arg("calculationsOnPython"), py::arg("reloadConcentrations") = false, py::arg("reloadReferenceConcentrations") = false, py::return_value_policy::reference, R"pbdoc(
        Calculates the complete list of calculations with the provided set of parameters.
		These calculations should have been loaded with loadCalculations prior to executing calculate, otherwise this will produce an error.
    )pbdoc");


#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}