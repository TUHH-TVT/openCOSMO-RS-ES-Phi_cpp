/*
    c++ implementation of openCOSMO-RS including multiple segment descriptors
    @author: Simon Mueller, 2022
*/


#pragma once

#include "interaction_matrix.hpp"
#include "contact_statistics.hpp"
#include "COSMOfile_functions.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

// prints only outside OpenMP regions (no GIL on worker threads)
inline void displayIfSerial(const std::string& message) {
#if defined(_OPENMP)
    if (omp_in_parallel()) {
        return;
    }
#endif
    display(message);
}
#include <stdexcept>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>
#include <unordered_map>
#include <string>

#if defined(MEASURE_TIME) 
#include <chrono>

std::atomic<unsigned long> oneIteration_total_ms{0};
std::chrono::high_resolution_clock::time_point oneIteration_last;

std::atomic<unsigned long> calculateTau_total_ms{0};
std::atomic<unsigned long> calculateCOSMOSPACE_total_ms{0};
std::atomic<unsigned long> calculateGammasForMolecules_total_ms{0};
std::atomic<unsigned long> calculateContactStatistics_total_ms{0};
std::atomic<unsigned long> rescaleSegments_total_ms{0};
std::atomic<unsigned long> calculateCombinatorial_total_ms{0};
std::atomic<unsigned long> calculateResidual_total_ms{0};
std::atomic<unsigned long> addContributions_total_ms{0};

// isotherm / EOS timers
std::atomic<unsigned long> calculateIsotherm_total_ms{0};
std::atomic<unsigned long> calculateIsotherm_calls{0};
std::atomic<unsigned long> calculateEqCond_total_ms{0};
std::atomic<unsigned long> calculateEqCond_calls{0};
std::atomic<unsigned long> calculateEOS_total_ms{0};
std::atomic<unsigned long> calculateEOS_calls{0};
std::atomic<unsigned long> calculateEOS_residualGamma_ms{0};   // time inside calculateLnGammaResidual (COSMO-SPACE)
std::atomic<unsigned long> calculateEOS_electrostatic_ms{0};   // time inside Born / PDH pressure & fugacity block
std::atomic<unsigned long> cosmospace_solves{0};               // COSMOSPACE solves
std::atomic<unsigned long> cosmospace_sweeps{0};               // successive-substitution sweeps
std::atomic<unsigned long> fastRoot_success{0};                // brackets found by search
std::atomic<unsigned long> fastRoot_fallback{0};               // searches handed back to the march

// RAII timer
struct ScopedMicrosecondTimer {
    std::atomic<unsigned long>& accumulator;
    std::chrono::high_resolution_clock::time_point start;
    explicit ScopedMicrosecondTimer(std::atomic<unsigned long>& acc)
        : accumulator(acc), start(std::chrono::high_resolution_clock::now()) {}
    ~ScopedMicrosecondTimer() {
        accumulator += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start).count());
    }
};

void startCalculationMeasurement() {

    oneIteration_last = std::chrono::high_resolution_clock::now();

    // by commenting the following lines you achieve a cummulative sum of the time over more than one iteration.
    rescaleSegments_total_ms = 0;
    calculateCombinatorial_total_ms = 0;
    calculateResidual_total_ms = 0;
    calculateTau_total_ms = 0;
    calculateCOSMOSPACE_total_ms = 0;
    calculateContactStatistics_total_ms = 0;
    calculateGammasForMolecules_total_ms = 0;
    addContributions_total_ms = 0;
    oneIteration_total_ms = 0;

    calculateIsotherm_total_ms = 0;
    calculateIsotherm_calls = 0;
    calculateEqCond_total_ms = 0;
    calculateEqCond_calls = 0;
    calculateEOS_total_ms = 0;
    calculateEOS_calls = 0;
    calculateEOS_residualGamma_ms = 0;
    calculateEOS_electrostatic_ms = 0;
    cosmospace_solves = 0;
    cosmospace_sweeps = 0;
    fastRoot_success = 0;
    fastRoot_fallback = 0;
}

void stopCalculationMeasurement() {

    oneIteration_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - oneIteration_last).count());

    // print accumulated times
    displayTime("rescaleSegments_total_ms               ", rescaleSegments_total_ms);
    displayTime("calculateCombinatorial_total_ms        ", calculateCombinatorial_total_ms);
    displayTime("calculateResidual_total_ms             ", calculateResidual_total_ms);
    displayTime("   calculateTau_total_ms               ", calculateTau_total_ms);
    displayTime("   calculateCOSMOSPACE_total_ms        ", calculateCOSMOSPACE_total_ms);
    if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
        displayTime("   calculateContactStatistics_total_ms ", calculateContactStatistics_total_ms);
    }
    displayTime("   calculateGammasForMolecules_total_ms", calculateGammasForMolecules_total_ms);
    displayTime("addContributions_total_ms              ", addContributions_total_ms);
    displayTime("oneIteration_total_ms                  ", oneIteration_total_ms);
    display("\n");

    display("--- isotherm / EOS breakdown (activity coefficient) ---\n");
    displayTime("calculateIsotherm_total_ms             ", calculateIsotherm_total_ms);
    display("   calculateIsotherm_calls                 : " + std::to_string((unsigned long)calculateIsotherm_calls) + "\n");
    displayTime("calculateEqCondForGivenTPn_total_ms    ", calculateEqCond_total_ms);
    display("   calculateEqCondForGivenTPn_calls        : " + std::to_string((unsigned long)calculateEqCond_calls) + "\n");
    displayTime("calculateEOS_total_ms                  ", calculateEOS_total_ms);
    display("   calculateEOS_calls                      : " + std::to_string((unsigned long)calculateEOS_calls) + "\n");
    displayTime("   calculateEOS_residualGamma_ms       ", calculateEOS_residualGamma_ms);
    displayTime("   calculateEOS_electrostatic_ms       ", calculateEOS_electrostatic_ms);
    if ((unsigned long)calculateEOS_calls > 0) {
        display("   calculateEOS_microseconds_per_call      : " + std::to_string((unsigned long)calculateEOS_total_ms / (unsigned long)calculateEOS_calls) + "\n");
    }
    display("cosmospace_solves                          : " + std::to_string((unsigned long)cosmospace_solves) + "\n");
    display("   cosmospace_sweeps                       : " + std::to_string((unsigned long)cosmospace_sweeps) + "\n");
    display("fastRoot_success / fallback                : " + std::to_string((unsigned long)fastRoot_success) + " / " + std::to_string((unsigned long)fastRoot_fallback) + "\n");
    display("\n");
}
#endif

inline float hsum_ps_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline double hsum_pd_sse3(__m128d v) {
    __m128d shuf = _mm_shuffle_pd(v, v, 0x1);
    __m128d sums = _mm_add_pd(v, shuf);
    return _mm_cvtsd_f64(sums);
}

#if defined(__AVX__) || defined(__FMA__)
inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    return hsum_ps_sse3(vlow);
}
inline double hsum256_pd_avx(__m256d v) {
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    return hsum_pd_sse3(vlow);
}
#endif

// OPENCOSMORS_FAST_ROOT=0 switches the fast liquid-root search off
inline bool optFastLiquidRootEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("OPENCOSMORS_FAST_ROOT");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// optional changes compiled into this binary
inline std::string optimization_banner() {
    std::string exact = "";
#ifndef OPT_NO_EARLY_EXIT
    exact += " early-exit";
#endif
#ifndef OPT_NO_MEMO
    exact += " memo";
#endif
#ifndef OPT_NO_DYNAMIC
    exact += " dynamic";
#endif
#ifndef OPT_NO_LOGCACHE
    exact += " logcache";
#endif
#ifndef OPT_NO_CONV_PREFILTER
    exact += " conv-prefilter";
#endif
#if !defined(OPT_NO_HOIST) && defined(USE_DOUBLE)
    exact += " hoist";
#endif
#ifndef OPT_NO_BUFFERS
    exact += " buffers";
#endif
#ifndef OPT_NO_ROOT_HINTS
    exact += " root-hints(fast-root only)";
#endif
#ifndef OPT_NO_VLE_EXITS
    exact += " vle-exits";
#endif
    return "opt (exact):" + (exact.empty() ? std::string(" none") : exact)
        + "\nopt fast liquid root: " + (optFastLiquidRootEnabled() ? "ON (default; OPENCOSMORS_FAST_ROOT=0 disables)" : "off (OPENCOSMORS_FAST_ROOT=0)");
}

void initialize(parameters& param, bool initializeParameters = true, bool initializeMolecules = true, bool initializeCalculations = true, bool showBinarySpecs = true) {


    if (initializeParameters) {
        n_ex = 0;
        param.ChargeRaster.clear();
        param.exp_param.clear();
        param.R_i = std::vector<double>(118, 0.0);
        param.R_i_COSMO = std::vector<double>(118, 0.0);
        param.HBClassElmnt = std::vector<int>(300, 0);

        // Initialize hydrogen bond classes of the elements HBClassElmnt
        // 0 : only non HB  | 1 : potential donor  | 2 : potential acceptor | 3 : potential donor or acceptor
        // classify all hydrogens and some metals as potential donors and all others as potential acceptors.

        for (int atomic_number = 0; atomic_number < param.HBClassElmnt.size(); atomic_number++) {
            if (atomic_number <= 100) param.HBClassElmnt[atomic_number] = 2;
            else if (atomic_number > 100) param.HBClassElmnt[atomic_number] = 1;   // all hydrogens
        }

        // set some values manually
        param.HBClassElmnt[1] = 1;   // hydrogen
        param.HBClassElmnt[3] = 1;   // li 
        param.HBClassElmnt[4] = 1;   // be 
        param.HBClassElmnt[11] = 1;  // na 
        param.HBClassElmnt[12] = 1;  // mg 
        param.HBClassElmnt[13] = 1;  // al 
        param.HBClassElmnt[19] = 1;  // k* 
        param.HBClassElmnt[20] = 1;  // ca 
        param.HBClassElmnt[24] = 1;  // cr 
        param.HBClassElmnt[26] = 1;  // fe 
        param.HBClassElmnt[27] = 1;  // co 
        param.HBClassElmnt[29] = 1;  // cu 
        param.HBClassElmnt[30] = 1;  // zn 
        param.HBClassElmnt[37] = 1;  // rb 
        param.HBClassElmnt[38] = 1;  // sr 
        param.HBClassElmnt[48] = 1;  // cd 
        param.HBClassElmnt[55] = 1;  // cs 
        param.HBClassElmnt[56] = 1;  // ba 


        // initialize charge raster
        int steps = (int)((param.sigmaMax - param.sigmaMin) / param.sigmaStep + 1 + 0.00001);
        int k0 = std::lround(param.sigmaMin / param.sigmaStep); // starting point
        for (int i = 0; i < steps; i++) {
            int k = k0 + i;
            double sigma = k * param.sigmaStep;
            param.ChargeRaster.push_back(sigma);
            //param.ChargeRaster.push_back(param.sigmaMin + param.sigmaStep * i);
        }
    }

    if (initializeMolecules)
        molecules.clear();

    if (initializeCalculations)
        calculations.clear();
    subcalculations.clear();
    criticalPointCache.clear();   // parameters may have changed since the last call

    if (showBinarySpecs)
        display("\nBINARY SPECS\n-------------------------\n" + compilation_mode + "\n" + OPENMP_parallelization + "\n" + vectorization_level + "\n" + precision + "\n" + optimization_banner() + "\n-------------------------\n\n");
}

void averageAndClusterSegments(parameters& param, molecule& _molecule, int approximateNumberOfSegmentTypes = 0) {

    // save reallocation time by specifying the approximate segment type number
    // this is possible whenever the sigma profile is reloaded and the previous number
    // is known
    if (approximateNumberOfSegmentTypes != 0) {
        // as default a vector resizes to the double of the current size when needed
        // here we are taking 110% of the approximateSegmentTypeNumber hopefully  
        // hindering a reallocation
        approximateNumberOfSegmentTypes = int(1.1 * approximateNumberOfSegmentTypes);
        _molecule.segments.reserve(approximateNumberOfSegmentTypes);
    }


    // average the segments
    int numberOfSegments = int(_molecule.segmentAreas.size());
    int numberOfAtoms = int(_molecule.atomAtomicNumbers.size());

    Eigen::VectorXd segmentRadiiSquared = _molecule.segmentAreas / PI;
    double RavSquared = param.Rav * param.Rav;
    double RavCorrSquared = param.RavCorr * param.RavCorr;

    // Just for debugging
    //std::vector<double> segmentAreas = {};
    //std::vector<double> segmentSigmas = {};
    //for (int m = 0; m < _molecule.segmentAreas.size(); m++) {
    //    segmentAreas.push_back(_molecule.segmentAreas(m));
    //    segmentSigmas.push_back(_molecule.segmentSigmas(m));
    //}

    //for (int m = 0; m < _molecule.segmentAreas.size(); m++) {
    //    _molecule.segmentAreas(m) = param.exp_param.at("scaling_factor") * _molecule.segmentAreas(m);
    //}


    Eigen::VectorXd averagedSigmas = Eigen::VectorXd::Zero(numberOfSegments);
    Eigen::VectorXd averagedSigmaCorrs;

    bool calculateMisfitCorrelation = false;
    if (param.sw_misfit == 1) {
        calculateMisfitCorrelation = true;
        averagedSigmaCorrs = Eigen::VectorXd::Zero(numberOfSegments);
    }
    else if (param.sw_misfit == 2 && (_molecule.moleculeCharge == 0 || numberOfAtoms > 1)) {
        calculateMisfitCorrelation = true;
        averagedSigmaCorrs = Eigen::VectorXd::Zero(numberOfSegments);
    }

    double distanceSegmentSegmentSquared;
    double temporaryValue = 0, multiplyingFactor = 0;

    for (int segmentIndexI = 0; segmentIndexI < numberOfSegments; segmentIndexI++) {

        double runningTotalSigmas = 0, runningTotalSigmaCorrs = 0;

        for (int segmentIndexJ = 0; segmentIndexJ < numberOfSegments; segmentIndexJ++) {
            distanceSegmentSegmentSquared = (_molecule.segmentPositions(segmentIndexI, Eigen::indexing::all) - _molecule.segmentPositions(segmentIndexJ, Eigen::indexing::all)).array().square().matrix().sum();

            temporaryValue = segmentRadiiSquared(segmentIndexJ) + RavSquared;
            multiplyingFactor = ((segmentRadiiSquared(segmentIndexJ) * RavSquared) / temporaryValue) * exp(-distanceSegmentSegmentSquared / temporaryValue);

            runningTotalSigmas += multiplyingFactor;
            averagedSigmas(segmentIndexI) = averagedSigmas(segmentIndexI) + _molecule.segmentSigmas(segmentIndexJ) * multiplyingFactor;

            if (calculateMisfitCorrelation == true) {
                temporaryValue = segmentRadiiSquared(segmentIndexJ) + RavCorrSquared;
                multiplyingFactor = ((segmentRadiiSquared(segmentIndexJ) * RavCorrSquared) / temporaryValue) * exp(-distanceSegmentSegmentSquared / temporaryValue);

                runningTotalSigmaCorrs += multiplyingFactor;
                averagedSigmaCorrs(segmentIndexI) = averagedSigmaCorrs(segmentIndexI) + _molecule.segmentSigmas(segmentIndexJ) * multiplyingFactor;
            }
        }
        averagedSigmas(segmentIndexI) = averagedSigmas(segmentIndexI) / runningTotalSigmas;

        if (calculateMisfitCorrelation == true)
            averagedSigmaCorrs(segmentIndexI) = averagedSigmaCorrs(segmentIndexI) / runningTotalSigmaCorrs;
    }

    bool calculateSolvationEnergies = param.dGsolv_E_gas.size() > 0;
    if (calculateSolvationEnergies) {
        if (_molecule.qmMethod != "DFT_CPCM_BP86_def2-TZVP+def2-TZVPD_SP" && _molecule.qmMethod != "DFT_BP86_def2-TZVPD_SP") {
            if (param.sw_dGsolv_calculation_strict == 1) {
                throw std::runtime_error("The QSPR model for the molar volume only works for the quantum chemistry method DFT_BP86_def2-TZVPD_SP");
            }
            else {
                warnings.push_back(" - The QSPR model for the molar volume was parametrized using a different quantum chemistry method than the one you are using. Recommended method: DFT_BP86_def2-TZVPD_SP");
            }
        }

        int numberOfSiAtoms = 0;
        int numberOfHAtoms = 0;
        int numberOfOAtoms = 0;
        for (int i = 0; i < numberOfAtoms; i++) {
            if (_molecule.atomAtomicNumbers[i] == 14)
                numberOfSiAtoms += 1;
            else if (_molecule.atomAtomicNumbers[i] == 1 || _molecule.atomAtomicNumbers[i] > 100)
                numberOfHAtoms += 1;
            else if (_molecule.atomAtomicNumbers[i] == 8)
                numberOfOAtoms += 1;
        }

        if (numberOfAtoms == 3 && numberOfHAtoms == 2 && numberOfOAtoms == 1) {
            _molecule.molarVolumeAt25C = 18.06863632;
        }
        else {
            Eigen::ArrayXd averagedSigmasSquared = averagedSigmas.array() * averagedSigmas.array();
            double secondSigmaMoment = (averagedSigmasSquared * _molecule.segmentAreas.array()).sum() * 10000;
            double fourthSigmaMoment = (averagedSigmasSquared * averagedSigmasSquared * _molecule.segmentAreas.array()).sum() * 100000000;

            _molecule.molarVolumeAt25C = 0.9430785419976806 * numberOfAtoms \
                + 0.6977322963011842 * _molecule.Area \
                - 0.3161763939689293 * secondSigmaMoment \
                + 0.032441059832647084 * fourthSigmaMoment \
                + 8.113026329415828 * numberOfSiAtoms \
                - 0.07066832029215675;
        }
    }

    // determine the hydrogen bonding type based on the atomic number

    for (int i = 0; i < numberOfSegments; i++) {

        int atomicNumber = _molecule.segmentAtomicNumber[i];
        // 0 : non HB  | 1 : potential donor  | 2 : potential acceptor | 3 : potential donor or acceptor
        switch (param.HBClassElmnt[atomicNumber]) {
        case 0:
            _molecule.segmentHydrogenBondingType(i) = 0;
            break;
        case 1:
            _molecule.segmentHydrogenBondingType(i) = averagedSigmas(i) < 0 ? 1 : 0;
            break;
        case 2:
            _molecule.segmentHydrogenBondingType(i) = averagedSigmas(i) < 0 ? 0 : 2;
            break;
        case 3:
            _molecule.segmentHydrogenBondingType(i) = averagedSigmas(i) < 0 ? 1 : 2;
            break;
        default:
            throw std::runtime_error("Unknown HB classification value used.");
        }
    }

    // cluster segments into segment types
    double sigmaLeft = -1;
    double sigmaRight = -1;
    double AsigmaLeft = -1;
    double AsigmaRight = -1;

    double sigmaCorrLeft = -1;
    double sigmaCorrRight = -1;

    double AsigmaLeftSigmaCorrLeft = -1;
    double AsigmaLeftSigmaCorrRight = -1;
    double AsigmaRightSigmaCorrLeft = -1;
    double AsigmaRightSigmaCorrRight = -1;

    unsigned short ind_SigmaCorr_left = 0;

    for (int j = 0; j < numberOfSegments; j++) {

        //unsigned short ind_Sigma_left = int((averagedSigmas(j) - param.sigmaMin) / param.sigmaStep);
        double x = (averagedSigmas(j) - param.sigmaMin) / param.sigmaStep;
        unsigned short ind_Sigma_left = static_cast<unsigned short>(std::floor(x + 1e-12));

        sigmaLeft = param.ChargeRaster[ind_Sigma_left];
        sigmaRight = param.ChargeRaster[ind_Sigma_left + 1];

        AsigmaRight = _molecule.segmentAreas(j) * (averagedSigmas(j) - sigmaLeft) / param.sigmaStep;
        AsigmaLeft = _molecule.segmentAreas(j) * (sigmaRight - averagedSigmas(j)) / param.sigmaStep;
        if (AsigmaLeft < 0 || AsigmaRight < 0) {
            throw std::runtime_error("Calculated negative segment area. This should not happen.");
        }

        if (calculateMisfitCorrelation == true) {
            ind_SigmaCorr_left = int((averagedSigmaCorrs(j) - param.sigmaMin) / param.sigmaStep);

            sigmaCorrLeft = param.ChargeRaster[ind_SigmaCorr_left];
            sigmaCorrRight = param.ChargeRaster[ind_SigmaCorr_left + 1];

            AsigmaLeftSigmaCorrRight = AsigmaLeft * (averagedSigmaCorrs(j) - sigmaCorrLeft) / param.sigmaStep;
            AsigmaLeftSigmaCorrLeft = AsigmaLeft * (sigmaCorrRight - averagedSigmaCorrs(j)) / param.sigmaStep;
            AsigmaRightSigmaCorrRight = AsigmaRight * (averagedSigmaCorrs(j) - sigmaCorrLeft) / param.sigmaStep;
            AsigmaRightSigmaCorrLeft = AsigmaRight * (sigmaCorrRight - averagedSigmaCorrs(j)) / param.sigmaStep;
        }


        unsigned short atomicNumber = _molecule.segmentAtomicNumber(j);
        if (param.sw_atomicNumber == 0) {
            atomicNumber = 0;
        }

        // for monoatomic ions or if misfit correlation is deactivated
        if (_molecule.moleculeGroup == 3 || _molecule.moleculeGroup == 5 || calculateMisfitCorrelation == false) {

            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaLeft, 0.0f, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaLeft, _molecule.index, _molecule.segmentAtomicPolariz(j));
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaRight, 0.0f, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaRight, _molecule.index, _molecule.segmentAtomicPolariz(j));
        }
        else {
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaLeft, sigmaCorrLeft, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaLeftSigmaCorrLeft, _molecule.index, _molecule.segmentAtomicPolariz(j));
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaLeft, sigmaCorrRight, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaLeftSigmaCorrRight, _molecule.index, _molecule.segmentAtomicPolariz(j));
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaRight, sigmaCorrLeft, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaRightSigmaCorrLeft, _molecule.index, _molecule.segmentAtomicPolariz(j));
            _molecule.segments.add(0, _molecule.moleculeGroup, sigmaRight, sigmaCorrRight, _molecule.segmentHydrogenBondingType(j), atomicNumber, AsigmaRightSigmaCorrRight, _molecule.index, _molecule.segmentAtomicPolariz(j));
        }
    }

}

inline Eigen::MatrixXd extractAtomicPolarizabilityTensor(molecule& _molecule, int atomIndex) {

    int numberOfAtoms = int(_molecule.atomAtomicNumbers.size());
    Eigen::MatrixXd polarizabilityTensors = _molecule.atomPolarizabilityTensors;
    Eigen::MatrixXd polarizabilityTensor(3, 3);
    Eigen::VectorXd polarizabilityTensorLine = polarizabilityTensors.row(atomIndex);

    polarizabilityTensor(0, 0) = polarizabilityTensorLine(0);
    polarizabilityTensor(1, 1) = polarizabilityTensorLine(1);
    polarizabilityTensor(2, 2) = polarizabilityTensorLine(2);
    polarizabilityTensor(0, 1) = polarizabilityTensorLine(3);
    polarizabilityTensor(1, 0) = polarizabilityTensorLine(3);
    polarizabilityTensor(0, 2) = polarizabilityTensorLine(4);
    polarizabilityTensor(2, 0) = polarizabilityTensorLine(4);
    polarizabilityTensor(1, 2) = polarizabilityTensorLine(5);
    polarizabilityTensor(2, 1) = polarizabilityTensorLine(5);

    return polarizabilityTensor;
}

void projectSegments(parameters& param, molecule& _molecule) {
    // project polarizability tensors onto segments and scale them

    int numberOfSegments = int(_molecule.segmentAreas.size());
    int numberOfAtoms = int(_molecule.atomAtomicNumbers.size());

    double polarizability_projection;

    Eigen::MatrixXd surfaceCoordinatesInAu = _molecule.segmentPositions / 0.529177249;
    Eigen::VectorXd surfaceAreasInAu = _molecule.segmentAreas / pow(0.529177249, 2);
    Eigen::VectorXd surfaceRadiiSquared = _molecule.segmentAreas / PI;
    Eigen::MatrixXd atomCoordinatesInAu = _molecule.atomPositions / 0.529177249;
    Eigen::VectorXi segmentAtomIndex = _molecule.segmentAtomIndices;
    Eigen::MatrixXd polarizabilityTensor(3, 3);
    Eigen::VectorXd areaOfAtom = Eigen::VectorXd::Zero(numberOfAtoms);
    Eigen::VectorXd segmentsperAtom = Eigen::VectorXd::Zero(numberOfAtoms);
    Eigen::VectorXd distanceSegmentAtom(3);

    if (param.sw_SR_polarizabilities == 0) {
        for (int segmentIndexI = 0; segmentIndexI < numberOfSegments; segmentIndexI++) {
            _molecule.segmentAtomicPolariz(segmentIndexI) = 0;
        }
    }
    else {
        for (int segmentIndexI = 0; segmentIndexI < numberOfSegments; segmentIndexI++) {
            int atomIndex = segmentAtomIndex(segmentIndexI);
            areaOfAtom(atomIndex) = areaOfAtom(atomIndex) + surfaceAreasInAu(segmentIndexI);
            segmentsperAtom(atomIndex) = segmentsperAtom(atomIndex) + 1;
        }


        for (int segmentIndexI = 0; segmentIndexI < numberOfSegments; segmentIndexI++) {
            int atomIndex = segmentAtomIndex(segmentIndexI);
            for (int k = 0; k < 3; k++) {
                distanceSegmentAtom(k) = std::abs(surfaceCoordinatesInAu(segmentIndexI, k) - atomCoordinatesInAu(atomIndex, k));
            }
            double distanceSegmentAtom_norm = distanceSegmentAtom(0) + distanceSegmentAtom(1) + distanceSegmentAtom(2);
            distanceSegmentAtom = distanceSegmentAtom / distanceSegmentAtom_norm;
            polarizabilityTensor = extractAtomicPolarizabilityTensor(_molecule, atomIndex);

            Eigen::RowVector3d distanceSegmentAtom_row = distanceSegmentAtom.transpose();
            Eigen::RowVector3d tmp_vec = distanceSegmentAtom_row * polarizabilityTensor;

            polarizability_projection = tmp_vec.norm();

            if (param.sw_SR_polarizabilities == 5) {
                polarizability_projection = round_and_truncate(polarizability_projection, 1);
            }
            else if (param.sw_SR_polarizabilities == 1) {
                polarizability_projection = round_and_truncate(polarizability_projection / segmentsperAtom(atomIndex), -1);
            }
            else if (param.sw_SR_polarizabilities == 3) {
                polarizability_projection = round_and_truncate(polarizability_projection * surfaceAreasInAu(segmentIndexI) / areaOfAtom(atomIndex), -1);
            }
            _molecule.segmentAtomicPolariz(segmentIndexI) = polarizability_projection;
        }

        if (param.sw_SR_polarizabilities == 6 || param.sw_SR_polarizabilities == 7 || param.sw_SR_polarizabilities == 8) {
            Eigen::VectorXd segmentAtomicPolarizCopy = _molecule.segmentAtomicPolariz;
            for (int segmentIndexI = 0; segmentIndexI < numberOfSegments; segmentIndexI++) {
                Eigen::VectorXd distanceSegmentAllSegments = Eigen::VectorXd::Zero(numberOfSegments);
                Eigen::VectorXd averagingFactors = Eigen::VectorXd::Zero(numberOfSegments);
                double RavSquared = param.Rav * param.Rav;
                distanceSegmentAllSegments(Eigen::indexing::all) = 0.52917721067121 * (surfaceCoordinatesInAu(segmentIndexI, Eigen::indexing::all).replicate(numberOfSegments, 1) - surfaceCoordinatesInAu).array().square().rowwise().sum().sqrt();
                averagingFactors = ((surfaceRadiiSquared * RavSquared).array() / (surfaceRadiiSquared.array() + RavSquared)) * exp(-1 * distanceSegmentAllSegments.array().square() / (surfaceRadiiSquared.array() + RavSquared));
                averagingFactors = averagingFactors / averagingFactors.sum();
                polarizability_projection = (segmentAtomicPolarizCopy.array() * averagingFactors.array()).sum();
                _molecule.segmentAtomicPolariz(segmentIndexI) = round_and_truncate(polarizability_projection, 1);
            }
        }
    }
}

molecule loadNewMolecule(parameters& param, std::string componentPath) {

    molecule newMolecule;
    if (param.sw_COSMOfiles_type == "Turbomole_COSMO_TZVP" || param.sw_COSMOfiles_type == "Turbomole_COSMO_TZVPD_FINE") {
        newMolecule = getMoleculeFromTurbomoleCOSMOfile(componentPath);
    }
    else if (param.sw_COSMOfiles_type == "ORCA_COSMO_TZVPD") {
        newMolecule = getMoleculeFromORCACOSMOfile(componentPath);
    }
    else if (param.sw_COSMOfiles_type == "ML_ORCA_COSMO_TZVPD_SP") {
        newMolecule = getMoleculeFromJSONfile(componentPath);
    }
    else {
        throw std::runtime_error("No method for reading COSMOfiles has been implemented for the following type: " + param.sw_COSMOfiles_type);
    }

    std::string componentName = componentPath;
    std::vector<std::string> parts = split(componentName, '\\');
    parts = split(parts[parts.size() - 1], '/');
    parts = split(parts[parts.size() - 1], '.');
    newMolecule.name = trim(parts[0]);

    // if sigma profile not yet loaded
    if (newMolecule.segments.size() == 0) {
        double sumOfScreeningCharge = (newMolecule.segmentAreas.array() * newMolecule.segmentSigmas.array()).matrix().sum();
        newMolecule.moleculeCharge = (signed char)(std::round(-1.0f * sumOfScreeningCharge));

        int numberOfAtoms = int(newMolecule.atomAtomicNumbers.size());

        // Store atomic radii and check for consistency
        for (int atomIndex = 0; atomIndex < numberOfAtoms; atomIndex++) {
            int AtomicNumber = newMolecule.atomAtomicNumbers(atomIndex);
            if (param.R_i_COSMO[AtomicNumber] != 0 && newMolecule.atomRadii(atomIndex) != param.R_i_COSMO[AtomicNumber]) {
                throw std::runtime_error("Inconsistent radii set for atomic number " + std::to_string(AtomicNumber) + " was found.");
            }
            else if (param.R_i_COSMO[AtomicNumber] == 0 && newMolecule.atomRadii(atomIndex) != 0) {
                param.R_i_COSMO[AtomicNumber] = newMolecule.atomRadii(atomIndex);
            }
        }

        // Calculate distance between atoms
        Eigen::MatrixXd distanceAtomAtomSquared(numberOfAtoms, numberOfAtoms);

        for (int atomIndex = 0; atomIndex < numberOfAtoms; atomIndex++) {
            Eigen::MatrixXd thisAtomPosition = newMolecule.atomPositions(atomIndex, Eigen::indexing::all);
            distanceAtomAtomSquared(Eigen::indexing::all, atomIndex) = (newMolecule.atomPositions - thisAtomPosition.replicate(numberOfAtoms, 1)).array().square().rowwise().sum();
        }

        // set moleculeGroup
        int moleculeGroup = 0;

        if (param.sw_differentiateMoleculeGroups == 1) {

            moleculeGroup = -1;

            if (numberOfAtoms == 3) {
                int numberOfFoundOxygens = 0;
                int numberOfFoundHydrogens = 0;
                for (int atomIndex = 0; atomIndex < numberOfAtoms; atomIndex++) {
                    if (newMolecule.atomAtomicNumbers(atomIndex) == 1) {
                        numberOfFoundHydrogens += 1;
                    }
                    else if (newMolecule.atomAtomicNumbers(atomIndex) == 8) {
                        numberOfFoundOxygens += 1;
                    }
                }

                if (numberOfFoundOxygens == 1 && numberOfFoundHydrogens == 2) {
                    moleculeGroup = 2;
                }
            }

            if (moleculeGroup == -1) {
                if (newMolecule.moleculeCharge == 0) {
                    moleculeGroup = numberOfAtoms == 1 ? 0 : 1;
                }
                else if (newMolecule.moleculeCharge > 0) {
                    moleculeGroup = numberOfAtoms == 1 ? 3 : 4;
                }
                else if (newMolecule.moleculeCharge < 0) {
                    moleculeGroup = numberOfAtoms == 1 ? 5 : 6;
                }
            }
        }
        newMolecule.moleculeGroup = (unsigned short)moleculeGroup;

        // this changes the atomic number of a hydrogen atom to 100 + the atomic number of the closest heavy atom
        // giving the abbility to differentiate between hydrogen atoms depending on the atom they are bound to.
        if (param.sw_differentiateHydrogens == 1) {

            for (int atomIndexI = 0; atomIndexI < numberOfAtoms; atomIndexI++) {
                double minimumDistanceAtomIAtomJSquared = 10e14;
                int closestAtomIndex = -1;
                if (newMolecule.atomAtomicNumbers(atomIndexI) != 1) {
                    continue;
                }

                for (int atomIndexJ = 0; atomIndexJ < numberOfAtoms; atomIndexJ++) {

                    if (atomIndexI == atomIndexJ) {
                        continue;
                    }

                    // search for next atom that is not a Hydrogen unless tmolecule is H2
                    if (newMolecule.atomAtomicNumbers(atomIndexJ) == 1 && newMolecule.atomAtomicNumbers(atomIndexJ) < 100 && numberOfAtoms != 2) {
                        continue;
                    }

                    if (distanceAtomAtomSquared(atomIndexI, atomIndexJ) < minimumDistanceAtomIAtomJSquared) {
                        minimumDistanceAtomIAtomJSquared = distanceAtomAtomSquared(atomIndexI, atomIndexJ);
                        closestAtomIndex = atomIndexJ;
                    }
                }
                int newAtomicNumber = 100 + newMolecule.atomAtomicNumbers(closestAtomIndex);
                newMolecule.atomAtomicNumbers(atomIndexI) = newAtomicNumber;
                param.R_i_COSMO[newAtomicNumber] = param.R_i_COSMO[1];
            }
        }
        else if (param.sw_differentiateHydrogens != 0) {
            throw std::runtime_error("differentiateHydrogens accepts values [0, 1]");
        }

        int numberOfSegments = int(newMolecule.segmentAreas.size());
        newMolecule.segmentAtomicNumber = Eigen::VectorXi(numberOfSegments);

        for (int i = 0; i < numberOfSegments; i++) {

            int atomicNumber = newMolecule.atomAtomicNumbers(newMolecule.segmentAtomIndices(i));
            newMolecule.segmentAtomicNumber(i) = atomicNumber;
        }

        newMolecule.segmentHydrogenBondingType = Eigen::VectorXi(numberOfSegments);
        newMolecule.segmentAtomicPolariz = Eigen::VectorXd(numberOfSegments);

        projectSegments(param, newMolecule);
        averageAndClusterSegments(param, newMolecule);
    }

    newMolecule.clear_unneeded_matrices(param.sw_alwaysReloadSigmaProfiles);
    newMolecule.segments.shrink_to_fit();

#ifdef PRINT_DEBUG_INFO
    //newMolecule.segments.sort();
    //WriteExtendedSigmaProfileToFile(newMolecule.name + ".extsp", newMolecule.segments);
#endif

    return newMolecule;
}

molecule createHoleMolecule(parameters& param) {
    molecule holeMolecule;

    holeMolecule.name = "hole";
    holeMolecule.index = int(molecules.size());
    holeMolecule.moleculeCharge = 0;/////
    holeMolecule.Area = param.hole_area;
    holeMolecule.Volume = 0; // this parameter is only to fullfill the requirements for a standard molecule struct
    // it is never used in calculations as the volume of the holes is dependent on concentrations

    holeMolecule.moleculeGroup = 7;
    holeMolecule.atomPositions = Eigen::MatrixXd::Zero(1, 3);
    holeMolecule.atomAtomicNumbers = Eigen::VectorXi::Zero(1);
    holeMolecule.atomRadii = Eigen::VectorXd::Zero(1);

    holeMolecule.segmentAreas = Eigen::VectorXd::Constant(1, holeMolecule.Area);/////
    holeMolecule.segmentPositions = Eigen::MatrixXd::Zero(1, 3);/////
    holeMolecule.segmentAtomicNumber = Eigen::VectorXi::Zero(1);/////
    holeMolecule.segmentAtomIndices = Eigen::VectorXi::Zero(1); // 299?
    holeMolecule.segmentHydrogenBondingType = Eigen::VectorXi::Zero(1);
    holeMolecule.segmentSigmas = Eigen::VectorXd::Zero(1);/////
    holeMolecule.segmentAtomicPolariz = Eigen::VectorXd::Zero(1);

    averageAndClusterSegments(param, holeMolecule);

    holeMolecule.clear_unneeded_matrices(param.sw_alwaysReloadSigmaProfiles);
    holeMolecule.segments.shrink_to_fit();

    return holeMolecule;
}

void reloadAllMolecules() {
    threadException e;
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < molecules.size(); i++) {
        e.run([=] {
            molecule _molecule = *molecules[i];
            int previousNumberOfSegmentTypes = int(_molecule.segments.size());
            _molecule.segments.clear();
            averageAndClusterSegments(param, _molecule, previousNumberOfSegmentTypes);
            });
    }
    e.rethrow();
}

void resizeMonoatomicCations(parameters& param, std::vector<std::shared_ptr<molecule>> molecules) {
#ifdef MEASURE_TIME
    std::chrono::high_resolution_clock::time_point rescaleSegments_last = std::chrono::high_resolution_clock::now();
#endif
    // scale A and V for monoatomic cations
    for (int i = 0; i < molecules.size(); i++) {

        if (molecules[i]->moleculeGroup == 3) {
            int AN = molecules[i]->atomAtomicNumbers(0);
            double R_i = param.R_i[AN];
            molecules[i]->Area = (4 * PI * R_i * R_i);
            molecules[i]->Volume = (4.0 / 3 * PI * R_i * R_i * R_i);
        }
    }
#ifdef MEASURE_TIME
    rescaleSegments_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - rescaleSegments_last).count());
#endif
}

void rescaleSegments(parameters& param, calculation& _calculation) {
    // rescale segments for monoatomic cations only if present
    if (_calculation.segments.numberOfSegmentsForGroup[3] != 0) {

        // monoatomic cation
        std::unordered_map<unsigned short, int[2]> segmentIndicesBelongingToASpecificAtomicNumber;
        for (int j = _calculation.segments.lowerBoundIndexForGroup[3];
            j < _calculation.segments.upperBoundIndexForGroup[3]; j++) {

            unsigned short AN = _calculation.segments.SegmentTypeAtomicNumber[j];

            if (segmentIndicesBelongingToASpecificAtomicNumber.find(AN) == segmentIndicesBelongingToASpecificAtomicNumber.end()) {
                segmentIndicesBelongingToASpecificAtomicNumber[AN][0] = (unsigned short)j;
            }
            else {
                segmentIndicesBelongingToASpecificAtomicNumber[AN][1] = (unsigned short)j;
            }

        }
        for (auto it = segmentIndicesBelongingToASpecificAtomicNumber.begin(); it != segmentIndicesBelongingToASpecificAtomicNumber.end(); ++it) {

            int AN = it->first;
            int ind_left = it->second[0];
            int ind_right = it->second[1];

            if (ind_right - ind_left > 1) {
                throw std::runtime_error("More than 2 segmentTypes was found for monoatomic cation: " + std::to_string(AN) + ".");
            }

            int ind_molecule = -1;

            for (int k = 0; _calculation.segments.SegmentTypeAreas[ind_left].size(); k++) {
                if (_calculation.segments.SegmentTypeAreas[ind_left][k] > 0.0) {
                    ind_molecule = k;
                    break;
                }
            }

            int screeningCharge = int(_calculation.components[ind_molecule]->moleculeCharge * -1);

            double newArea = (4 * PI * param.R_i[AN] * param.R_i[AN]);
            double newSigma = screeningCharge / newArea;

            //unsigned short ind_Sigma_left = int((newSigma - param.sigmaMin) / param.sigmaStep);
            double x = (newSigma - param.sigmaMin) / param.sigmaStep;
            unsigned short ind_Sigma_left = static_cast<unsigned short>(std::floor(x + 1e-12));

            double sigmaLeft = param.ChargeRaster[ind_Sigma_left];
            double sigmaRight = param.ChargeRaster[ind_Sigma_left + 1];

            double AsigmaRight = newArea * (newSigma - sigmaLeft) / param.sigmaStep;
            double AsigmaLeft = newArea * (sigmaRight - newSigma) / param.sigmaStep;

            _calculation.segments.SegmentTypeAreas[ind_left][ind_molecule] = AsigmaLeft;
            _calculation.segments.SegmentTypeSigma[ind_left] = sigmaLeft;

            _calculation.segments.SegmentTypeAreas[ind_right][ind_molecule] = AsigmaRight;
            _calculation.segments.SegmentTypeSigma[ind_right] = sigmaRight;

        }
    }

    if (_calculation.segments.numberOfSegmentsForGroup[5] != 0) {

        // monoatomic cation
        std::unordered_map<unsigned short, int[2]> segmentIndicesBelongingToASpecificAtomicNumber;
        for (int j = _calculation.segments.lowerBoundIndexForGroup[5];
            j < _calculation.segments.upperBoundIndexForGroup[5]; j++) {

            unsigned short AN = _calculation.segments.SegmentTypeAtomicNumber[j];

            if (segmentIndicesBelongingToASpecificAtomicNumber.find(AN) == segmentIndicesBelongingToASpecificAtomicNumber.end()) {
                segmentIndicesBelongingToASpecificAtomicNumber[AN][0] = (unsigned short)j;
            }
            else {
                segmentIndicesBelongingToASpecificAtomicNumber[AN][1] = (unsigned short)j;
            }

        }
        for (auto it = segmentIndicesBelongingToASpecificAtomicNumber.begin(); it != segmentIndicesBelongingToASpecificAtomicNumber.end(); ++it) {

            int AN = it->first;
            int ind_left = it->second[0];
            int ind_right = it->second[1];

            if (ind_right - ind_left > 1) {
                throw std::runtime_error("More than 2 segmentTypes was found for monoatomic cation: " + std::to_string(AN) + ".");
            }

            int ind_molecule = -1;

            for (int k = 0; _calculation.segments.SegmentTypeAreas[ind_left].size(); k++) {
                if (_calculation.segments.SegmentTypeAreas[ind_left][k] > 0.0) {
                    ind_molecule = k;
                    break;
                }
            }

            int screeningCharge = int(_calculation.components[ind_molecule]->moleculeCharge * -1);

            double newArea = (4 * PI * param.R_i[AN] * param.R_i[AN]);
            double newSigma = screeningCharge / newArea;

            unsigned short ind_Sigma_left = int((newSigma - param.sigmaMin) / param.sigmaStep);

            double sigmaLeft = param.ChargeRaster[ind_Sigma_left];
            double sigmaRight = param.ChargeRaster[ind_Sigma_left + 1];

            double AsigmaRight = newArea * (newSigma - sigmaLeft) / param.sigmaStep;
            double AsigmaLeft = newArea * (sigmaRight - newSigma) / param.sigmaStep;

            _calculation.segments.SegmentTypeAreas[ind_left][ind_molecule] = AsigmaLeft;
            _calculation.segments.SegmentTypeSigma[ind_left] = sigmaLeft;

            _calculation.segments.SegmentTypeAreas[ind_right][ind_molecule] = AsigmaRight;
            _calculation.segments.SegmentTypeSigma[ind_right] = sigmaRight;
        }
        
    }
}

void calculateSegmentConcentrations(calculation& _calculation, std::vector<int> actualConcentrationIndicesToCalculate = {}) {

    if (actualConcentrationIndicesToCalculate.size() == 0) {
        actualConcentrationIndicesToCalculate = std::vector<int>(_calculation.concentrations.size());
        std::iota(std::begin(actualConcentrationIndicesToCalculate), std::end(actualConcentrationIndicesToCalculate), 0);
    }

    //if (_calculation.components[0]->name == "H2O_c000") {
    //    for (int k = 0; k < _calculation.segments.SegmentTypeAreas.size(); k++) {
    //        _calculation.segments.SegmentTypeAreas[k][0] = param.exp_param.at("scaling_factor") * _calculation.segments.SegmentTypeAreas[k][0];
    //    }
    //}

    // calculate the mole fraction of segments for each concentration
    for (auto j : actualConcentrationIndicesToCalculate) {

        int firstNonZeroSegmentIndex = 0;
        int lastNonZeroSegmentIndex = int(_calculation.concentrations.size()) - 1;

        std::vector<double> segmentConcentration(_calculation.segments.size(), 0.0);

        double sumAreaSegmentsConcentrationj = 0.0;

        for (int k = 0; k < _calculation.segments.size(); k++) {

            double areaSegmentK = 0.0;

            for (int m = 0; m < _calculation.components.size(); m++) {
                double thisArea = _calculation.concentrations[j][m] * _calculation.segments.SegmentTypeAreas[k][m];
                areaSegmentK += thisArea;
                sumAreaSegmentsConcentrationj += thisArea;
            }

            segmentConcentration[k] = areaSegmentK;
        }

        double cumulativeSumOfAreasFromSegmentZeroOn = 0.0;

        bool firstNonZeroSegmentIndex_found = false;
        for (int k = 0; k < _calculation.segments.size(); k++) {

            cumulativeSumOfAreasFromSegmentZeroOn += segmentConcentration[k];

            if (cumulativeSumOfAreasFromSegmentZeroOn != 0 && !firstNonZeroSegmentIndex_found) {
                firstNonZeroSegmentIndex = k;
                firstNonZeroSegmentIndex_found = true;
            }
            if (segmentConcentration[k] != 0) {
                lastNonZeroSegmentIndex = k;
            }

            _calculation.segmentConcentrations(k, j) = calcType(segmentConcentration[k] / sumAreaSegmentsConcentrationj);
        }

        _calculation.lowerBoundIndexForCOSMOSPACECalculation[j] = RoundDownToNextMultipleOfEight(firstNonZeroSegmentIndex);
        _calculation.upperBoundIndexForCOSMOSPACECalculation[j] = RoundUpToNextMultipleOfEight(lastNonZeroSegmentIndex + 1);
    }
}

void calculatePartialSegmentConcentrationsPartialv(calculation& _calculation, std::vector<int> actualConcentrationIndicesToCalculate, double PartialholeConcentrationPartialv) {

    // calculate the mole fraction of segments for each concentration
    for (auto j : actualConcentrationIndicesToCalculate) {

        double holeIndex = int(_calculation.components.size() - 1);
        double x_h = _calculation.concentrations[j][holeIndex];

        int firstNonZeroSegmentIndex = 0;
        int lastNonZeroSegmentIndex = int(_calculation.concentrations.size()) - 1;

        std::vector<double> segmentConcentration(_calculation.segments.size(), 0.0);
        std::vector<double> PartialsegmentConcentrationPartialv_2(_calculation.segments.size(), 0.0);

        double sumAreaSegmentsConcentrationj = 0.0;
        double PartialsumAreaSegmentsConcentrationjPartialv_2 = 0.0;
        double sumAreaSegmentsHole = 0.0;

        for (int k = 0; k < _calculation.segments.size(); k++) {

            double areaSegmentK = 0.0;
            double thisArea = 0.0;
            double areaSegmentK_2 = 0.0;
            double thisArea_2 = 0.0;

            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                thisArea = _calculation.concentrations[j][m] * _calculation.segments.SegmentTypeAreas[k][m];
                areaSegmentK += thisArea;
                sumAreaSegmentsConcentrationj += thisArea;
                thisArea_2 = _calculation.concentrations[j][m] / (1 - x_h) * _calculation.segments.SegmentTypeAreas[k][m];
                areaSegmentK_2 += thisArea_2;
                PartialsumAreaSegmentsConcentrationjPartialv_2 += thisArea_2;
            }
            thisArea = _calculation.concentrations[j][holeIndex] * _calculation.segments.SegmentTypeAreas[k][holeIndex];
            thisArea_2 = -1 * _calculation.segments.SegmentTypeAreas[k][holeIndex];
            areaSegmentK += thisArea;
            areaSegmentK_2 += thisArea_2;
            PartialsumAreaSegmentsConcentrationjPartialv_2 += thisArea_2;
            sumAreaSegmentsConcentrationj += thisArea;

            sumAreaSegmentsHole += _calculation.segments.SegmentTypeAreas[k][holeIndex];

            segmentConcentration[k] = areaSegmentK;
            PartialsegmentConcentrationPartialv_2[k] = areaSegmentK_2;
        }

        for (int k = 0; k < _calculation.segments.size(); k++) {

            //double nominator = (-PartialsegmentConcentrationPartialv[k] + _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]) * ((1 - x_h) * PartialsumAreaSegmentsConcentrationjPartialv + x_h * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]) - ((1 - x_h) * PartialsegmentConcentrationPartialv[k] + x_h * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]) * (-PartialsumAreaSegmentsConcentrationjPartialv + _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1]);
            //double denominator = std::pow((1 - x_h) * PartialsumAreaSegmentsConcentrationjPartialv + x_h * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1], 2);

            double nominator_derivative = PartialsegmentConcentrationPartialv_2[k] * (-1) * PartialholeConcentrationPartialv;
            double nominator = segmentConcentration[k];
            double denominator_derivative = PartialsumAreaSegmentsConcentrationjPartialv_2 * (-1) * PartialholeConcentrationPartialv;
            double denominator = sumAreaSegmentsConcentrationj;

            _calculation.PartialsegmentConcentrationsPartialv(k, j) = calcType((nominator_derivative * denominator - nominator * denominator_derivative) / (std::pow(denominator, 2)));
            //_calculation.PartialsegmentConcentrationsPartialv(k, j) = calcType(PartialholeConcentrationPartialv * (_calculation.segments.SegmentTypeAreas[k][holeIndex] * denominator - nominator * sumAreaSegmentsHole) / (std::pow(denominator, 2)));
        }
    }
}


void finishCalculationInitiation(calculation& _calculation, parameters& param) {

    if (_calculation.concentrations.size() > 65535) {
        throw std::runtime_error("Too many calculations, other datatype would be necessary for newCalculation.referenceStates to cope with this amount. (unsigned short used so far allowing for up to 65535)");
    }

    // check charge of all concentrations
    for (int j = 0; j < _calculation.concentrations.size(); j++) {

        double mix_chrg = 0;
        for (int k = 0; k < _calculation.components.size(); k++) {
            mix_chrg = mix_chrg + _calculation.components[k]->moleculeCharge * _calculation.concentrations[j][k];
        }

        if (std::abs(mix_chrg) > MAX_CONCENTRATION_DIFF_FROM_ZERO) {
            throw std::runtime_error("For calculation number " + std::to_string(_calculation.number) + " the mixture number " + std::to_string(j) + " is not electroneutral. residual charge: " + std::to_string(std::abs(mix_chrg)));
        }
    }

    // save sorting of concentrations according to conditions, this clusters the calculation of the interaction matrix
    // sorting by similar concentration does not make sense as the gammas from the previous iteration
    // are saved for every concentration already accelerating the COSMOSPACE convergence
    std::vector<int> sortingVector(_calculation.concentrations.size());
    std::iota(sortingVector.begin(), sortingVector.end(), 0);
    std::sort(sortingVector.begin(), sortingVector.end(),
        [&](int i, int j) {
            if (_calculation.temperatures[i] != _calculation.temperatures[j])
                return _calculation.temperatures[i] < _calculation.temperatures[j];
            else
                return i < j;
        });

    apply_vector_permutation_in_place(_calculation.concentrations, sortingVector);
    apply_vector_permutation_in_place(_calculation.temperatures, sortingVector);

    _calculation.actualConcentrationIndices = std::vector<int>(_calculation.concentrations.size());
    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        _calculation.actualConcentrationIndices[sortingVector[j]] = j;
    }

    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        int TauIndex = _calculation.addOrFindTauIndexForConditions(_calculation.temperatures[j]);
        _calculation.TauConcentrationIndices[TauIndex].push_back(j);
    }

    // initiate arrays for the calculation
    _calculation.segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentGammasPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentGammasPDEVolume = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentGammasSecondPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
    _calculation.segmentConcentrations = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()));
    _calculation.PartialsegmentConcentrationsPartialv = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()));

    _calculation.PhiDash_pxi = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    _calculation.ThetaDash_pxi = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());

    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        _calculation.lowerBoundIndexForCOSMOSPACECalculation.push_back(0);
        _calculation.upperBoundIndexForCOSMOSPACECalculation.push_back(int(_calculation.segments.size()));
    }

    _calculation.shrink_to_fit();
}

calculation loadSubcalculation(std::vector<std::shared_ptr<molecule>> molecules, std::vector<double> temperatures, std::vector<std::vector<double>> concentrationsToLoad, std::vector<double> pressures = { 0.0 }) {

    size_t numberOfComponentsToLoadSegmentsOf = molecules.size();
    size_t numberOfComponents = numberOfComponentsToLoadSegmentsOf - 1;

    calculation newCalculation = calculation(numberOfComponentsToLoadSegmentsOf);

    for (int j = 0; j < numberOfComponentsToLoadSegmentsOf; j++) {
        int moleculeIndex = j;

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
    }

    newCalculation.segments.sort();
    newCalculation.segments.shrink_to_fit();

    // monoatomic ions: rescale the segments to the ideal-sphere sigma
    rescaleSegments(param, newCalculation);

    newCalculation.temperatures = temperatures;
    newCalculation.temperatures.insert(newCalculation.temperatures.end(), temperatures.begin(), temperatures.end());

    size_t numberOfConcentrations = temperatures.size();
    Eigen::MatrixXd concentrations(numberOfConcentrations * 2, numberOfComponents);
    for (int k = 0; k < numberOfConcentrations; k++) {
        for (int l = 0; l < numberOfComponents; l++) {
            concentrations(k, l) = concentrationsToLoad[k][l];
            concentrations(k + numberOfConcentrations, l) = concentrationsToLoad[k][l];
        }
    }

    // concentrations and temperatures

    for (int j = 0; j < (size_t)numberOfConcentrations * 2; j++) {

        std::vector<double> rowConcentration;

        for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {
            double val = 0.0f;
            if (k < numberOfComponents)
                val = (double)concentrations(j, k); // for pure component here was a 0 instead of k

            rowConcentration.push_back(val);
        }

        newCalculation.concentrations.push_back(rowConcentration);
    }

    Eigen::MatrixXd referenceStateConcentrations = Eigen::MatrixXd::Zero(numberOfConcentrations * 2, numberOfComponents);


    newCalculation.originalNumberOfCalculations = (unsigned short)newCalculation.concentrations.size();

    // reference states
    Eigen::MatrixXi referenceStateTypes(numberOfConcentrations * 2, numberOfComponents);
    for (int k = 0; k < numberOfConcentrations * 2; k++) {
        for (int l = 0; l < numberOfComponents; l++) {
            referenceStateTypes(k, l) = (int)5;
        }
    }
    for (int j = 0; j < numberOfConcentrations * 2; j++) {

        int referenceStateType = referenceStateTypes(j, 0);

        newCalculation.referenceStateType.push_back((unsigned short)referenceStateType);

        std::vector<int> thisReferenceStateCalculationIndices;
        for (int k = 0; k < numberOfComponentsToLoadSegmentsOf; k++) {

            std::vector<double> referenceStateConcentration(numberOfComponentsToLoadSegmentsOf, 0.0f);
            referenceStateConcentration[numberOfComponentsToLoadSegmentsOf - 1] = 1.0f;

            double temperature = (double)newCalculation.temperatures[j];
            int referenceStateCalculationIndex = (int)newCalculation.addOrFindArrayIndexForConcentration(referenceStateConcentration, temperature);
            thisReferenceStateCalculationIndices.push_back(referenceStateCalculationIndex);
        }
        newCalculation.referenceStateCalculationIndices.push_back(thisReferenceStateCalculationIndices);
    }

    newCalculation.concentrations_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
    newCalculation.concentrations_data.setZero();
    for (int j = 0; j < numberOfConcentrations; j++) {
        for (int k = 0; k < numberOfComponents; k++) {
            newCalculation.concentrations_data(j, k) = concentrationsToLoad[j][k];
            newCalculation.concentrations_data(numberOfConcentrations + j, k) = concentrationsToLoad[j][k];
        }
    }

    newCalculation.originalConcentrations.~Map();
    new (&newCalculation.originalConcentrations) Eigen::Map<Eigen::MatrixXd>(
        newCalculation.concentrations_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponents);

    for (int j = 0; j < numberOfConcentrations; j++) {
        for (int k = 0; k < numberOfComponents; k++) {
            newCalculation.originalConcentrations(j, k) = concentrationsToLoad[j][k];
            newCalculation.originalConcentrations(numberOfConcentrations + j, k) = concentrationsToLoad[j][k];
        }
    }

    //newCalculation.originalConcentrations = newCalculation.concentrations_data;

    newCalculation.lnGammaCombinatorial_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
    newCalculation.lnGammaCombinatorial_data.setZero();
    new (&newCalculation.lnGammaCombinatorial) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        newCalculation.lnGammaCombinatorial_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponentsToLoadSegmentsOf);

    newCalculation.lnGammaResidual_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
    newCalculation.lnGammaResidual_data.setZero();
    new (&newCalculation.lnGammaResidual) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        newCalculation.lnGammaResidual_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponentsToLoadSegmentsOf);

    newCalculation.lnGammaTotal_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
    newCalculation.lnGammaTotal_data.setZero();
    new (&newCalculation.lnGammaTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        newCalculation.lnGammaTotal_data.data(),
        int(newCalculation.originalNumberOfCalculations),
        numberOfComponentsToLoadSegmentsOf);

    newCalculation.number = (int)0;
    finishCalculationInitiation(newCalculation, param);

    if (param.sw_phi == 1) {

        newCalculation.molarVolume_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.molarVolume_data.setZero();
        new (&newCalculation.molarVolume) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.molarVolume_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.repulsivePressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.repulsivePressure_data.setZero();
        new (&newCalculation.repulsivePressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.repulsivePressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.attractivePressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.attractivePressure_data.setZero();
        new (&newCalculation.attractivePressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.attractivePressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        if (param.sw_electrolyte == 1) {
            newCalculation.BornPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
            newCalculation.BornPressure_data.setZero();
            new (&newCalculation.BornPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.BornPressure_data.data(),
                int(newCalculation.originalNumberOfCalculations));

            newCalculation.electrostaticPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
            newCalculation.electrostaticPressure_data.setZero();
            new (&newCalculation.electrostaticPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.electrostaticPressure_data.data(),
                int(newCalculation.originalNumberOfCalculations));
        }

        newCalculation.totalPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        for (int j = 0; j < numberOfConcentrations * 2; j++) {
            newCalculation.totalPressure_data(j, 0) = double(pressures[0]);
        }
        new (&newCalculation.totalPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.totalPressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.targetPressure_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        for (int j = 0; j < numberOfConcentrations * 2; j++) {
            newCalculation.targetPressure_data(j, 0) = double(pressures[0]);
        }
        new (&newCalculation.targetPressure) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.targetPressure_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.targetTemperature_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        for (int j = 0; j < numberOfConcentrations * 2; j++) {
            newCalculation.targetTemperature_data(j, 0) = double(temperatures[0]);
        }
        new (&newCalculation.targetTemperature) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.targetTemperature_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.lnPhiTotal_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.lnPhiTotal_data.setZero();
        new (&newCalculation.lnPhiTotal) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiTotal_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        newCalculation.lnPhiRepulsive_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.lnPhiRepulsive_data.setZero();
        new (&newCalculation.lnPhiRepulsive) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiRepulsive_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        newCalculation.lnPhiAttractive_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.lnPhiAttractive_data.setZero();
        new (&newCalculation.lnPhiAttractive) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiAttractive_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        if (param.sw_electrolyte == 1) {
            newCalculation.lnPhiElectrostatic_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
            newCalculation.lnPhiElectrostatic_data.setZero();
            new (&newCalculation.lnPhiElectrostatic) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiElectrostatic_data.data(),
                int(newCalculation.originalNumberOfCalculations),
                numberOfComponents);

            newCalculation.lnPhiBorn_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
            newCalculation.lnPhiBorn_data.setZero();
            new (&newCalculation.lnPhiBorn) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.lnPhiBorn_data.data(),
                int(newCalculation.originalNumberOfCalculations),
                numberOfComponents);
        }

        newCalculation.criticalP_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.criticalP_data.setZero();
        new (&newCalculation.criticalP) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.criticalP_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.criticalT_data = Eigen::VectorXd(int(numberOfConcentrations) * 2);
        newCalculation.criticalT_data.setZero();
        new (&newCalculation.criticalT) Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(newCalculation.criticalT_data.data(),
            int(newCalculation.originalNumberOfCalculations));

        newCalculation.criticalx_data = Eigen::MatrixXd(int(numberOfConcentrations) * 2, numberOfComponents);
        newCalculation.criticalx_data.setZero();
        new (&newCalculation.criticalx) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(newCalculation.criticalx_data.data(),
            int(newCalculation.originalNumberOfCalculations),
            numberOfComponents);

        newCalculation.molarResidualEnthalpy = VectorCalcType::Zero(newCalculation.originalNumberOfCalculations);
        newCalculation.residualHeatCapacityAtConstantPressure = VectorCalcType::Zero(newCalculation.originalNumberOfCalculations);
        newCalculation.lnPhiTotalPDETAtConstantPx = MatrixCalcType::Zero(newCalculation.originalNumberOfCalculations, newCalculation.components.size());
        newCalculation.lnGammaResidualPDETatConstantvx = MatrixCalcType::Zero(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
        newCalculation.lnGammaResidualSecondPDETatConstantvx = MatrixCalcType::Zero(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);
        newCalculation.lnGammaResidualPDEvatConstantTx = MatrixCalcType::Zero(int(numberOfConcentrations) * 2, numberOfComponentsToLoadSegmentsOf);

        newCalculation.recalculateConcentrationsForPhi = true;
    }

    return newCalculation;
}


void calculateLnGammaCombinatorial(parameters& param, calculation& _calculation) {

    std::vector<double> averageVolumes(_calculation.concentrations.size(), 0.0);
    std::vector<double> averageAreas(_calculation.concentrations.size(), 0.0);

    // Volume fraction to mole fraction ratio for all compositions
    // Area fraction to mole fraction ratio for all compositions
    for (int j = 0; j < _calculation.concentrations.size(); j++) {
        for (int k = 0; k < _calculation.components.size(); k++) {
            averageVolumes[j] += _calculation.components[k]->Volume * _calculation.concentrations[j][k];
            averageAreas[j] += _calculation.components[k]->Area * _calculation.concentrations[j][k];
        }

        for (int k = 0; k < _calculation.components.size(); k++) {
            _calculation.PhiDash_pxi(j, k) = _calculation.components[k]->Volume / averageVolumes[j];
            _calculation.ThetaDash_pxi(j, k) = _calculation.components[k]->Area / averageAreas[j];
        }
    }

    std::vector<double> qi_std(_calculation.components.size());

    /* Calculation of molecule specific parameters for combinatorial contribution */
    for (int i = 0; i < _calculation.components.size(); i++) {
        qi_std[i] = _calculation.components[i]->Area / param.comb_SG_A_std;
    }

    Eigen::MatrixXd lnGamaForCalculations = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());

    if (param.sw_combTerm == 1) { // tested
        /* Staverman-Guggenheim term
           cp. Kikic (1980) or Lin & Sandler (2002)
           Remark: square brackets in L&S paper for the li term are wrong as can be seen in monograph Prausnitz Lichtenthaler */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {

            double buffdb1 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb1 = _calculation.PhiDash_pxi(i, j) / _calculation.ThetaDash_pxi(i, j);
                lnGamaForCalculations(i, j) = log(_calculation.PhiDash_pxi(i, j)) + 1 - _calculation.PhiDash_pxi(i, j) - \
                    param.comb_SG_z_coord * 0.5 * qi_std[j] * (log(buffdb1) + 1 - buffdb1);
            }
        }
    }
    else if (param.sw_combTerm == 2) { // tested
        /* Klamt (2003) */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {
            double buffdb1 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb1 = buffdb1 + _calculation.concentrations[i][j] * log(_calculation.components[j]->Volume);
            }

            for (int j = 0; j < _calculation.components.size(); j++) {
                lnGamaForCalculations(i, j) = (param.comb_lambda0 * (log(_calculation.components[j]->Volume) - buffdb1)) \
                    - (param.comb_lambda1 * (_calculation.PhiDash_pxi(i, j) - 1)) \
                    - (param.comb_lambda2 * (_calculation.ThetaDash_pxi(i, j) - 1));
            }
        }
    }
    else if (param.sw_combTerm == 3) { // tested
        /*  mod. Staverman-Guggenheim with exponential scaling, compare e.g. Soares (2011), Kikic (1980), Donohue Prausnitz (1975) */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {

            double buffdb1 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb1 += pow(_calculation.components[j]->Volume, param.comb_modSG_exp) * _calculation.concentrations[i][j];
            }

            double buffdb2 = 0;
            for (int j = 0; j < _calculation.components.size(); j++) {
                buffdb2 = _calculation.PhiDash_pxi(i, j) / _calculation.ThetaDash_pxi(i, j);

                double PhiDash_pxi_mod_i = pow(_calculation.components[j]->Volume, param.comb_modSG_exp) / buffdb1;

                lnGamaForCalculations(i, j) = log(PhiDash_pxi_mod_i) + 1 - PhiDash_pxi_mod_i - \
                    param.comb_SG_z_coord * 0.5 * qi_std[j] * (log(buffdb2) + 1 - buffdb2);
            }
        }
    }
    else if (param.sw_combTerm == 4) { // not tested yet
        /*  mod. Staverman-Guggenheim by Grensemann published in Grensemann & Gmehling (2005) especially developed for COSMO-RS */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {

            double sum_qi_div_xi = 0;
            double sum_ri_div_xi = 0;
            double sum_qi_times_xi = 0;
            double sum_ri_times_xi = 0;

            for (int j = 0; j < _calculation.components.size(); j++) {
                sum_qi_div_xi = sum_qi_div_xi + (_calculation.components[j]->Area / _calculation.concentrations[i][j]);
                sum_ri_div_xi = sum_ri_div_xi + (_calculation.components[j]->Volume / _calculation.concentrations[i][j]);
                sum_qi_times_xi = sum_qi_times_xi + (_calculation.components[j]->Area * _calculation.concentrations[i][j]);
                sum_ri_times_xi = sum_ri_times_xi + (_calculation.components[j]->Volume * _calculation.concentrations[i][j]);
            }

            std::vector<double> lnGamaForThisCalculation;
            for (int j = 0; j < _calculation.components.size(); j++) {

                double qi_hash = (1 - _calculation.concentrations[i][j]) * sum_qi_div_xi;
                double ri_hash = (1 - _calculation.concentrations[i][j]) * sum_ri_div_xi;

                double qi_min = std::min(_calculation.components[j]->Area, qi_hash);
                double qi_max = std::max(_calculation.components[j]->Area, qi_hash);

                double ri_min = std::min(_calculation.components[j]->Volume, ri_hash);
                double ri_max = std::max(_calculation.components[j]->Volume, ri_hash);

                double Fi = pow(_calculation.components[j]->Area / sum_qi_times_xi, param.comb_SGG_lambda * (1 - (qi_min / qi_max)));
                double Vi = pow(_calculation.components[j]->Volume / sum_ri_times_xi, param.comb_SGG_beta * (1 - (ri_min / ri_max)));

                lnGamaForCalculations(i, j) = 1 - Vi - log(Vi) + 1 - (Vi / Fi) - log(Vi / Fi);
            }
        }
    }
    else if (param.sw_combTerm == 5) { // not tested yet
        /*  Franke & Hannebauer (2011) */
        for (int i = 0; i < _calculation.concentrations.size(); i++) {
            for (int j = 0; j < _calculation.components.size(); j++) {
                lnGamaForCalculations(i, j) = param.comb_lambda0 * log(_calculation.components[j]->Volume) \
                    + param.comb_lambda1 * (1 - (_calculation.components[j]->Volume / averageVolumes[i]) - log(averageVolumes[i])) \
                    + param.comb_lambda2 * (1 - (_calculation.components[j]->Area / averageAreas[i]) - log(averageAreas[i]));
            }
        }
    }
    else if (param.sw_combTerm != 0) {
        throw std::runtime_error("Error: Invalid switch value for combinatorial term.\n");
    }


    /* Convert activity coefficients to correct reference states. */
    for (int h = 0; h < _calculation.originalNumberOfCalculations; h++) {

        int i = _calculation.actualConcentrationIndices[h];

        for (int j = 0; j < _calculation.components.size(); j++) {

            double multiplier = 1.0;
            // for an ion with reference state PureComponentsOnlyNeutral
            if (_calculation.referenceStateType[h] == 1 && _calculation.referenceStateCalculationIndices[h][j] == -1) {
                multiplier = NAN;
            }
            _calculation.lnGammaCombinatorial(h, j) = multiplier * lnGamaForCalculations(i, j);
#ifdef PRINT_DEBUG_INFO
            displayIfSerial("lnGammaCombinatorial_1_" + std::to_string(h) + ": " + std::to_string(_calculation.lnGammaCombinatorial(h, j)) + "\n");
#endif

            // only subtract reference state value if reference state type is not COSMO
            if (_calculation.referenceStateType[h] != 3 && _calculation.referenceStateType[h] != 4) {
                int indexOfReferenceStateCalculation = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[h][j]];

                _calculation.lnGammaCombinatorial(h, j) -= lnGamaForCalculations(indexOfReferenceStateCalculation, j);
            }
        }
    }
}

// the following declaration is needed because of crossrefs, initialization later
void calculateNumericalDerivative(calculation, double, std::vector<int>);

// the COSMOSPACE equations are solved by successive substitution
// instead of calculating the complete matrix-vector product and then calculating the gammas for the next iteration
// each row is calculated and the gamma for the row just calculated is already used in the calculation of the next row.
// This was done firstly by accident, but it was found, that it accelerated the convergence by a factor of at least 4.
void calculateLnGammaResidual(parameters& param, calculation& _calculation, std::vector<int> actualConcentrationIndicesToCalculate, bool withTDerivative = false, bool withvDerivative = false) {

    const int numberOfSegments = int(_calculation.segments.size());
    const int nMultipleOfEight = RoundUpToNextMultipleOfEight(numberOfSegments);

    Eigen::MatrixXd temporary_lnGammaMolecule = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    Eigen::MatrixXd temporary_lnGammaMolecule_PDETemperature = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    Eigen::MatrixXd temporary_lnGammaMolecule_SecondPDETemperature = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());
    Eigen::MatrixXd temporary_lnGammaMolecule_PDEVolume = Eigen::MatrixXd::Zero(_calculation.concentrations.size(), _calculation.components.size());

    Eigen::Tensor<double, 4, Eigen::RowMajor> temporary_averageInteractionEnergies;
    Eigen::Tensor<double, 3, Eigen::RowMajor> temporary_partialMolarEnergies;

    if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {

        temporary_averageInteractionEnergies = Eigen::Tensor<double, 4, Eigen::RowMajor>(int(_calculation.concentrations.size()),
            param.numberOfPartialInteractionMatrices + 1, // +1 because A_int is the first one
            int(_calculation.components.size()),
            int(_calculation.components.size()));

        temporary_averageInteractionEnergies.setZero();

        if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {
            temporary_partialMolarEnergies = Eigen::Tensor<double, 3, Eigen::RowMajor>(int(_calculation.concentrations.size()),
                param.numberOfPartialInteractionMatrices + 1, // +1 because A_int is the first one
                int(_calculation.components.size()));

            temporary_partialMolarEnergies.setZero();

        }

    }

    double Delta = 0.0001;
    //if (withTDerivative == true && param.exp_param.at("NumericalDerivative") == true) {
    //    calculateNumericalDerivative(_calculation, Delta, actualConcentrationIndicesToCalculate);
    //}

    const double dampingFactor = 0.6;
    const double dampingFactorComplement = 1 - dampingFactor;
    const double convergenceThreshhold = 0.00001;   // originally this was 0.0000001;
                                                    // however when using AVX and vectorized code (especially with floats)
                                                    // the accuracy of the calculation drops
                                                    // without increasing this value the gammas do not converge
                                                    // however since the convergence criteria was changed from
                                                    // gamma to log(gamma) the larger convergence threshhold 
                                                    // still leads to numerically very similar results

    const int maximumNumberOfIterations = 50000;
    double newGamma = 0.0;

#ifndef OPT_NO_BUFFERS
    MatrixCalcType A_int;
    if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
        A_int = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    }
    MatrixCalcType A_int_PDET = MatrixCalcType::Zero(1, 1);
    MatrixCalcType A_int_SecondPDET = MatrixCalcType::Zero(1, 1);
    MatrixCalcType TauX = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
#else
    MatrixCalcType A_int = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    MatrixCalcType A_int_PDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    MatrixCalcType A_int_SecondPDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
    MatrixCalcType TauX = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
#endif
    MatrixCalcType Tau;
    VectorCalcType vtempSum;
    VectorCalcType PartialSegmentGammasPartialT; // for calculations with modified Hessian
    VectorCalcType PartialSegmentGammasPartialTWithHessian; // for calculations with unmodified Hessian according to eq. (52)
    VectorCalcType PartialSegmentGammasPartialv;
    VectorCalcType Partial2SegmentGammasPartialT2;
    if (param.sw_phi == 1) {
        vtempSum = VectorCalcType::Zero(numberOfSegments);
    }

    std::vector<Eigen::MatrixXd> partialInteractionMatrices;

    if (param.numberOfPartialInteractionMatrices > 0) {
        for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
            partialInteractionMatrices.push_back(Eigen::MatrixXd::Zero(numberOfSegments, numberOfSegments));
        }
    }

    for (int g = 0; g < _calculation.TauConcentrationIndices.size(); g++) {

        // conditions
        double temperature = _calculation.TauTemperatures[g];

#ifdef MEASURE_TIME
        std::chrono::high_resolution_clock::time_point calculateTau_last = std::chrono::high_resolution_clock::now();
#endif
        bool TauIsEmpty = _calculation.Taus[g].size() == 0;
        if (TauIsEmpty || withTDerivative) {
#pragma omp critical
            {
                Tau = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
                A_int = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                if (withTDerivative) {
                    A_int_PDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                    A_int_SecondPDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                }
                else {
                    A_int_PDET = MatrixCalcType::Zero(1, 1);
                    A_int_SecondPDET = MatrixCalcType::Zero(1, 1);
                }

                // calculate interaction matrices
                if (param.numberOfPartialInteractionMatrices > 0) {
                    for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                        partialInteractionMatrices[h].setZero();
                    }
                }

                calculateInteractionMatrix(_calculation.segments, A_int, A_int_PDET, A_int_SecondPDET, partialInteractionMatrices, param, temperature);

                // if the full interaction matrix is needed, fill upper right half
                if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0 || withTDerivative) {
                    for (int i = 0; i < numberOfSegments; i++) {
                        for (int j = i + 1; j < numberOfSegments; j++) {
                            // always j >= i + 1
                            A_int(i, j) = A_int(j, i);
                            if (withTDerivative) {
                                A_int_PDET(i, j) = A_int_PDET(j, i);
                                A_int_SecondPDET(i, j) = A_int_SecondPDET(j, i);
                            }
                        }
                    }
                    for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                        for (int i = 0; i < numberOfSegments; i++) {
                            for (int j = i + 1; j < numberOfSegments; j++) {
                                // always j >= i + 1
                                partialInteractionMatrices[h](i, j) = partialInteractionMatrices[h](j, i);
                            }
                        }
                    }
                }

                // calculate Tau from the interaction matrix A_int
                int idx = -1;
                int columnSum = 0;
                const calcType minus_div_RT = calcType(-1.0f / (R_GAS_CONSTANT * temperature));

                calcType val;
                calcType* Tau_1D = &(Tau(0, 0));
                for (int i = 0; i < numberOfSegments; i++) {
                    columnSum = i * nMultipleOfEight;
                    for (int j = i; j < numberOfSegments; j++) {
                        // always j >= i
                        val = exp(A_int(j, i) * minus_div_RT);

                        idx = columnSum + j;
                        Tau_1D[idx] = val;
                        Tau_1D[j * nMultipleOfEight + i] = val;
                    }
                }
                _calculation.Taus[g] = Tau;
            }
        }
        else {
#ifdef OPT_NO_BUFFERS
            Tau = _calculation.Taus[g];
#endif
        }

#ifdef MEASURE_TIME
        calculateTau_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateTau_last).count());
#endif
#ifndef OPT_NO_BUFFERS
        // read the cached Tau in place
        const MatrixCalcType* TauSource = (TauIsEmpty || withTDerivative) ? &Tau : &_calculation.Taus[g];
        const MatrixCalcType& Tau = *TauSource;
#endif

#ifdef PRINT_DEBUG_INFO
        Eigen::MatrixXd A_int_d = A_int.cast<double>();
        for (int i = 0; i < numberOfSegments; i++)
            for (int j = i; j < numberOfSegments; j++)
                A_int_d(i, j) = A_int_d(j, i);

        WriteEigenMatrixToFile("A_int_" + std::to_string(_calculation.number) + "_" + std::to_string(g), A_int_d); // symmetric

        Eigen::MatrixXd Tau_d = Tau.cast<double>();
        WriteEigenMatrixToFile("Tau_" + std::to_string(_calculation.number) + "_" + std::to_string(g), Tau_d);
#endif

        //if (param.exp_param.at("EigenMatrixToFile") == 1) {
        //    std::string tempStr = std::to_string(temperature);
        //    tempStr.erase(
        //        std::remove(tempStr.begin(), tempStr.end(), '.'),
        //        tempStr.end()
        //    );
        //    WriteEigenMatrixToFile(tempStr + "_A_int_" + std::to_string(_calculation.number) + "_" + std::to_string(g), A_int); // symmetric
        //    WriteEigenMatrixToFile(tempStr + "_A_int_PDET" + std::to_string(_calculation.number) + "_" + std::to_string(g), A_int_PDET); // symmetric
        //}

        for (int h = 0; h < _calculation.TauConcentrationIndices[g].size(); h++) {

            int i = _calculation.TauConcentrationIndices[g][h];

            if (actualConcentrationIndicesToCalculate.size() > 0)
                if (std::find(actualConcentrationIndicesToCalculate.begin(), actualConcentrationIndicesToCalculate.end(), i) == actualConcentrationIndicesToCalculate.end())
                    continue;

            calcType* gammas = &(_calculation.segmentGammas(0, i));
            VectorCalcType SegmentGammas = _calculation.segmentGammas.col(i).head(numberOfSegments); // added
            //Eigen::VectorXd vg_exact = Eigen::VectorXd::Zero(numberOfSegments); // added
            //Eigen::VectorXd XjDivGammaJSquared = Eigen::VectorXd::Zero(numberOfSegments); // save evaluations of eq. 53 in Yan, 2024
#ifdef MEASURE_TIME
            std::chrono::high_resolution_clock::time_point calculateCOSMOSPACE_last = std::chrono::high_resolution_clock::now();
#endif
#ifdef OPT_NO_BUFFERS
            TauX = MatrixCalcType::Zero(nMultipleOfEight, numberOfSegments);
#endif
            int lowerBoundIndexForCOSMOSPACECalculation = _calculation.lowerBoundIndexForCOSMOSPACECalculation[i];
            int upperBoundIndexForCOSMOSPACECalculation = _calculation.upperBoundIndexForCOSMOSPACECalculation[i];

            { // as local as possible scope for all variables used in your loops

                const calcType* vTau_1D = &(Tau(0, 0));
                const calcType* vX = &(_calculation.segmentConcentrations(0, i));

                calcType* vTauX_1D = &(TauX(0, 0));
                int idx = 0;
                int columnSum = 0;


                for (int j = 0; j < numberOfSegments; j++) {

                    columnSum = j * nMultipleOfEight;

#if defined(__AVX__) || defined(__FMA__)//AVX
#if defined(USE_DOUBLE)
                    for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 4) {
                        idx = columnSum + k;

                        _mm256_store_pd(vTauX_1D + idx, _mm256_mul_pd(_mm256_load_pd(vTau_1D + idx), _mm256_load_pd(vX + k)));
                    }

#else
                    for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 8) {
                        idx = columnSum + k;

                        _mm256_store_ps(vTauX_1D + idx, _mm256_mul_ps(_mm256_load_ps(vTau_1D + idx), _mm256_load_ps(vX + k)));
                    }
#endif

#else //SSE3
#if defined(USE_DOUBLE)
                    for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 2) {
                        idx = columnSum + k;

                        _mm_store_pd(vTauX_1D + idx, _mm_mul_pd(_mm_load_pd(vTau_1D + idx), _mm_load_pd(vX + k)));
                    }

#else
                    for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 4) {
                        idx = columnSum + k;

                        _mm_store_ps(vTauX_1D + idx, _mm_mul_ps(_mm_load_ps(vTau_1D + idx), _mm_load_ps(vX + k)));
                    }
#endif
#endif

                }
            }

#ifdef PRINT_DEBUG_INFO
            Write1DArrayToFile<calcType>("X_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), &(_calculation.segmentConcentrations(0, i)), 1, nMultipleOfEight);
            Eigen::MatrixXd TauX_d = TauX.cast<double>();
            WriteEigenMatrixToFile("TauX_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), TauX_d);
#endif

            int numberOfConvergedGammas = 0;
            int numberOfIteration = 0;

            while (numberOfConvergedGammas != numberOfSegments) {

                numberOfIteration++;

                if (numberOfIteration > maximumNumberOfIterations) {

                    if (param.sw_skip_COSMOSPACE_errors == 0) {

                        std::string information = "";
                        information += "\nSYSTEM_COMPONENTS:\n";
                        for (int j = 0; j < _calculation.components.size(); j++) {
                            information += " -" + _calculation.components[j]->name + "\n";
                        }

                        const calcType* Tau_1D = &(Tau(0, 0));
                        for (int j = 0; j < numberOfSegments * nMultipleOfEight; j++) {
                            if (std::isnan(Tau_1D[j])) {
                                information += " Some Tau entries are NaN.";
                                break;
                            }
                        }

                        for (int j = 0; j < numberOfSegments * nMultipleOfEight; j++) {
                            if (std::isinf(Tau_1D[j])) {
                                information += " Some Tau entries are inf.";
                                break;
                            }
                        }

                        for (int j = 0; j < numberOfSegments; j++) {
                            if (std::isnan(_calculation.segmentGammas(j, i))) {
                                information += " Some gammas are NaN.";
                                break;
                            }
                        }
                        for (int j = 0; j < numberOfSegments; j++) {
                            if (std::isinf(_calculation.segmentGammas(j, i))) {
                                information += " Some gammas are inf.";
                                break;
                            }
                        }
                        if (A_int.rows() != numberOfSegments) {
                            A_int = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                        }
                        Eigen::MatrixXd A_int_d = A_int.cast<double>();
                        for (int i = 0; i < numberOfSegments; i++)
                            for (int j = i; j < numberOfSegments; j++)
                                A_int_d(i, j) = A_int_d(j, i);

#ifdef PRINT_DEBUG_INFO
                        WriteEigenMatrixToFile("A_int_" + std::to_string(_calculation.number) + "_" + std::to_string(g), A_int_d); // symmetric

                        Eigen::MatrixXd Tau_d = Tau.cast<double>();
                        WriteEigenMatrixToFile("Tau_" + std::to_string(_calculation.number) + "_" + std::to_string(g), Tau_d); // symmetric

                        Write1DArrayToFile<double>("X_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), &(_calculation.segmentConcentrations(0, i)), 1, nMultipleOfEight, true);
                        Eigen::MatrixXd TauX_d = TauX.cast<double>();
                        WriteEigenMatrixToFile("TauX_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), TauX_d);
#endif           

                        displayIfSerial(information + "\nCOSMOSPACE did not converge for calculation " + std::to_string(_calculation.number) + " on concentration " + std::to_string(i) + ": maximum number of iterations reached. (" + std::to_string(numberOfConvergedGammas) + "/" + std::to_string(numberOfSegments) + ")");
                    }
                    else {
                        // set all segment gammas to 1 as initial point for next execution
                        _calculation.segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(_calculation.segments.size())), int(_calculation.concentrations.size()), 1.0);
                    }

                    throw std::runtime_error("COSMOSPACE did not converge for calculation " + std::to_string(_calculation.number) + " on concentration " + std::to_string(i) + ": maximum number of iterations reached. (" + std::to_string(numberOfConvergedGammas) + "/" + std::to_string(numberOfSegments) + ")");
                }

                numberOfConvergedGammas = 0;

                int columnSum = 0;

                for (int j = 0; j < numberOfSegments; j++) {
                    newGamma = 0.0;

                    columnSum = j * nMultipleOfEight;

                    { // as local as possible scope for all variables used in your loops

                        const calcType* vTauX = &(TauX(0, 0));
                        calcType* vGammas = gammas;

#if defined(__FMA__)

#if defined(USE_DOUBLE)
                        __m256d tempVectorSum = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 4) {
                            tempVectorSum = _mm256_fmadd_pd(_mm256_load_pd(vTauX + columnSum + k), _mm256_load_pd(vGammas + k), tempVectorSum); //FMA
                            //tempVectorSum = _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(vTauX + columnSum + k), _mm256_load_pd(vGammas + k)), tempVectorSum); //AVX
                        }

                        newGamma = hsum256_pd_avx(tempVectorSum);

#else
                        __m256 tempVectorSum = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 8) {
                            tempVectorSum = _mm256_fmadd_ps(_mm256_load_ps(vTauX + columnSum + k), _mm256_load_ps(vGammas + k), tempVectorSum); //FMA
                        }

                        newGamma = double(hsum256_ps_avx(tempVectorSum));
#endif

#elif defined(__AVX__)
#if defined(USE_DOUBLE)
                        __m256d tempVectorSum = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 4) {
                            tempVectorSum = _mm256_add_pd(_mm256_mul_pd(_mm256_load_pd(vTauX + columnSum + k), _mm256_load_pd(vGammas + k)), tempVectorSum); //AVX
                        }

                        newGamma = hsum256_pd_avx(tempVectorSum);

#else
                        __m256 tempVectorSum = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 8) {
                            //tempSum += hsum256_ps_avx(_mm256_mul_ps(_mm256_load_ps(vTauX + columnSum + k), _mm256_load_ps(vGammas + k))); // less accurate
                            tempVectorSum = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(vTauX + columnSum + k), _mm256_load_ps(vGammas + k)), tempVectorSum); //AVX
                        }
                        newGamma = double(hsum256_ps_avx(tempVectorSum));
#endif

#else//SSE3
#if defined(USE_DOUBLE)
                        __m128d tempVectorSum = _mm_setr_pd(0.0, 0.0);

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 2) {
                            tempVectorSum = _mm_add_pd(_mm_mul_pd(_mm_load_pd(vTauX + columnSum + k), _mm_load_pd(vGammas + k)), tempVectorSum); //SSE
                        }

                        newGamma = hsum_pd_sse3(tempVectorSum);

#else
                        __m128 tempVectorSum = _mm_setr_ps(0.0, 0.0, 0.0, 0.0);

                        for (int k = lowerBoundIndexForCOSMOSPACECalculation; k < upperBoundIndexForCOSMOSPACECalculation; k += 4) {
                            tempVectorSum = _mm_add_ps(_mm_mul_ps(_mm_load_ps(vTauX + columnSum + k), _mm_load_ps(vGammas + k)), tempVectorSum); //SSE
                        }

                        newGamma = double(hsum_ps_sse3(tempVectorSum));
#endif

#endif
                    }

                    newGamma = 1 / newGamma;
                    double oldGamma = double(gammas[j]);

                    // apply damping
                    if (numberOfIteration > 200) {
                        newGamma = dampingFactor * newGamma + dampingFactorComplement * oldGamma;
                    }

                    // check convergence criteria

                    double temp_convergenceThreshhold = convergenceThreshhold;
                    if (j == numberOfSegments - 1) {
                        temp_convergenceThreshhold = 0.0000001;
                    }
#ifndef OPT_NO_CONV_PREFILTER
                    // decide from the ratio where it is unambiguous, the log test only in between
                    {
                        const double ratioDeviation = std::abs(newGamma / double(gammas[j]) - 1.0);
                        bool isConverged;
                        if (ratioDeviation <= 0.5 * temp_convergenceThreshhold) {
                            isConverged = true;
                        }
                        else if (ratioDeviation >= 2.0 * temp_convergenceThreshhold) {
                            isConverged = false;
                        }
                        else {
                            isConverged = std::abs(log(newGamma) - log(gammas[j])) <= temp_convergenceThreshhold;
                        }
                        if (isConverged) {
                            numberOfConvergedGammas++;
                        }
                    }
#else
                    if (std::abs(log(newGamma) - log(gammas[j])) <= temp_convergenceThreshhold) {
                        //vg_exact(j) = _calculation.segmentConcentrations(j, i) / SegmentGammas(j) - _calculation.segmentConcentrations(j, i) / newGamma; // newGamma is 1 / sum X_nGamma_ntau_mn
                        //XjDivGammaJSquared(j) = 1 / SegmentGammas(j) * (vg_exact(j) + _calculation.segmentConcentrations(j, i) / newGamma);
                        numberOfConvergedGammas++;
                    }
#endif

                    gammas[j] = calcType(newGamma);

#if defined(OPT_NO_HOIST) || !defined(USE_DOUBLE)
                    if (param.sw_phi == 1) {
                        SegmentGammas(j) = calcType(newGamma);
                        vtempSum(j) = calcType(1 / newGamma);
                    }
#endif
                }
            }

#ifdef MEASURE_TIME
            cosmospace_solves += 1;
            cosmospace_sweeps += (unsigned long)numberOfIteration;
#endif

#if !defined(OPT_NO_HOIST) && defined(USE_DOUBLE)
            // filled once from the converged gammas
            if (param.sw_phi == 1) {
                for (int j = 0; j < numberOfSegments; j++) {
                    SegmentGammas(j) = gammas[j];
                    vtempSum(j) = calcType(1 / double(gammas[j]));
                }
            }
#endif

#ifdef PRINT_DEBUG_INFO
            displayIfSerial("COSMOSPACE converged with this number of iterations: " + std::to_string(numberOfIteration) + "\n");
            Write1DArrayToFile<calcType>("Gammas_" + std::to_string(_calculation.number) + "_" + std::to_string(g) + "_" + std::to_string(i), gammas, 1, numberOfSegments, true);
#endif

            if (param.sw_phi == 1 && (withTDerivative == true || withvDerivative == true)) {
                bool iIsReferenceStateIndex = false;
                for (int p = 0; p < _calculation.referenceStateCalculationIndices.size(); p++) {
                    for (int q = 0; q < int(_calculation.components.size() - 1); q++) {
                        if (_calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[p][q]] == i) {
                            iIsReferenceStateIndex = true;
                            break;
                        }
                    }
                    if (iIsReferenceStateIndex == true) {
                        break;
                    }
                }

                if (iIsReferenceStateIndex == true && withTDerivative == true) {
                    //PartialSegmentGammasPartialT = ((- 1 / (R_GAS_CONSTANT * temperature * temperature) * A_int.col(numberOfSegments - 1)).array() * (A_int.col(numberOfSegments - 1) / (R_GAS_CONSTANT * temperature)).array().exp()).matrix();
                    //Partial2SegmentGammasPartialT2 = ((A_int.col(numberOfSegments - 1) / (R_GAS_CONSTANT * temperature)).array().exp() * (2 / (R_GAS_CONSTANT * pow(temperature, 3)) * A_int.col(numberOfSegments - 1) + 1 / (pow(R_GAS_CONSTANT, 2) * pow(temperature, 4)) * A_int.col(numberOfSegments - 1).cwiseProduct(A_int.col(numberOfSegments - 1))).array()).matrix();

                    const int N = numberOfSegments;

                    const long double R = (long double)R_GAS_CONSTANT;
                    const long double T = (long double)temperature;

                    const long double invRT = 1.0L / (R * T);
                    const long double invRT2 = invRT / T;        // 1/(R*T^2)
                    const long double invRT3 = invRT2 / T;       // 1/(R*T^3)
                    const long double invRT4 = invRT3 / T;       // 1/(R*T^4)
                    const long double invR2T4 = invRT4 / R;      // 1/(R^2*T^4)

                    PartialSegmentGammasPartialT.resize(N);
                    Partial2SegmentGammasPartialT2.resize(N);

                    for (int p = 0; p < N; ++p) {

                        long double A = (long double)A_int(p, numberOfSegments - 1);

                        long double arg = A * invRT;
                        long double expval = expl(arg);

                        long double val =
                            -A * invRT2 * expval;

                        PartialSegmentGammasPartialT(p) = (double)val;

                        long double term1 = 2.0L * A * invRT3;
                        long double term2 = A * A * invR2T4;

                        long double val2 =
                            expval * (term1 + term2);

                        Partial2SegmentGammasPartialT2(p) = (double)val2;
                    }

                    _calculation.segmentGammasPDETemperature.col(i).head(numberOfSegments) = PartialSegmentGammasPartialT;
                    _calculation.segmentGammasSecondPDETemperature.col(i).head(numberOfSegments) = Partial2SegmentGammasPartialT2;
                }
                if (iIsReferenceStateIndex == true && withvDerivative == true) {
                    _calculation.segmentGammasPDEVolume.col(i).head(numberOfSegments).setZero();
                }
                if (iIsReferenceStateIndex == false) {
                    VectorCalcType vg = VectorCalcType::Zero(numberOfSegments);
                    VectorCalcType vg_sum = VectorCalcType::Zero(numberOfSegments);
                    MatrixCalcType Hessian = MatrixCalcType::Zero(numberOfSegments, numberOfSegments); // definition according to eq. (52) in Yan, 2024
                    MatrixCalcType modifiedHessian = MatrixCalcType::Zero(numberOfSegments, numberOfSegments); // definition according to eq. (54) in Yan, 2024
                    MatrixCalcType modifiedHessianPDETemperature = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                    VectorCalcType DeltaGamma = VectorCalcType::Zero(numberOfSegments);

                    for (int p = 0; p < numberOfSegments; p++) {

                        for (int q = p; q < numberOfSegments; q++) {

                            long double xp = (long double)_calculation.segmentConcentrations(p, i);
                            long double xq = (long double)_calculation.segmentConcentrations(q, i);
                            long double tau = (long double)Tau(p, q);

                            long double val_ld = -xp * xq * tau;
                            double val = (double)val_ld;

                            //calcType val2 = -1 * _calculation.segmentConcentrations(p, i) * _calculation.segmentConcentrations(q, i) * A_int(p, q) / (R_GAS_CONSTANT * temperature * temperature) * exp(-1 * A_int(p, q) / (R_GAS_CONSTANT * temperature)); // this is the temperature derivative of the Hessian
                            modifiedHessian(q, p) = (calcType)val;
                            modifiedHessian(p, q) = (calcType)val;
                            Hessian(q, p) = (calcType)val;
                            Hessian(p, q) = (calcType)val;
                            //modifiedHessianPDETemperature(q, p) = (calcType)val2;
                            //modifiedHessianPDETemperature(p, q) = (calcType)val2;
                        }

                    }

                    /* ================================
                        Diagonale der modifizierten Hessian
                    ================================ */
                    // The following code uses vector operations
                    //VectorCalcType constantSecondSummandOfModifiedHessian = (_calculation.segmentConcentrations.col(i).head(numberOfSegments).asDiagonal() * _calculation.segmentConcentrations.col(i).head(numberOfSegments)).asDiagonal() * Tau.diagonal().head(numberOfSegments);
                    //VectorCalcType tempvec = _calculation.segmentConcentrations.col(i).head(numberOfSegments).asDiagonal() * vtempSum; 
                    //modifiedHessian.diagonal() = -1 * tempvec.asDiagonal() * SegmentGammas.cwiseInverse() - constantSecondSummandOfModifiedHessian;

                    // This is the computationally heavier version using aggregators
                    for (int p = 0; p < numberOfSegments; ++p) {

                        double xp2 = _calculation.segmentConcentrations(p, i) * _calculation.segmentConcentrations(p, i);
                        long double xp = (long double)_calculation.segmentConcentrations(p, i);
                        long double xpsquared = (long double)xp2;
                        long double gamma_p = (long double)SegmentGammas(p);
                        long double tau_p_p = (long double)Tau(p, p);
                        long double vtempSum_p = (long double)vtempSum(p);

                        long double sum_ld = xpsquared * tau_p_p;

                        long double diag_ld = -(xp * vtempSum_p / gamma_p) - sum_ld;
                        modifiedHessian(p, p) = (calcType)diag_ld;
                    }

                    if (withvDerivative == true) {
                        VectorCalcType dX_dv = _calculation.PartialsegmentConcentrationsPartialv.col(i).head(numberOfSegments);
                        VectorCalcType VectorbForESToDeterminedGamma_dV = VectorCalcType(numberOfSegments);
                        for (int p = 0; p < numberOfSegments; p++) {
                            VectorCalcType Tau_row = Tau.row(p);
                            VectorCalcType CWiseProductSegmentGammasTauRow = SegmentGammas.cwiseProduct(Tau_row);
                            calcType DotProduct = CWiseProductSegmentGammasTauRow.dot(dX_dv);
                            VectorbForESToDeterminedGamma_dV(p) = _calculation.segmentConcentrations(p, i) * DotProduct;
                        }
                        _calculation.segmentGammasPDEVolume.col(i).head(numberOfSegments) = modifiedHessian.ldlt().solve(VectorbForESToDeterminedGamma_dV);
                    }

                    if (withTDerivative == true) {
                        VectorCalcType PartialvgPartialT = VectorCalcType(numberOfSegments);
                        VectorCalcType Partial2vgPartialT2 = VectorCalcType(numberOfSegments);

                        //MatrixCalcType MatrixA = MatrixCalcType::Zero(numberOfSegments, numberOfSegments); // Matrix A in Ax = b to solve for temp derivatives of Segment Gammas
                        //VectorCalcType Vectorb = VectorCalcType::Zero(numberOfSegments);

                        //VectorCalcType dX_dT;
                        //VectorCalcType d2X_dT2;
                        //if (param.exp_param.at("NumericalDerivative") == true) {
                        //    dX_dT = (subcalculations[int(subcalculations.size() - 1)].segmentConcentrations.col(i).head(numberOfSegments) - subcalculations[int(subcalculations.size() - 2)].segmentConcentrations.col(i).head(numberOfSegments)) / (2 * Delta);
                        //    d2X_dT2 = (subcalculations[int(subcalculations.size() - 1)].segmentConcentrations.col(i).head(numberOfSegments) - 2 * _calculation.segmentConcentrations.col(i).head(numberOfSegments) + subcalculations[int(subcalculations.size() - 2)].segmentConcentrations.col(i).head(numberOfSegments)) / (pow(Delta, 2));
                        // }
                        //else {
                        //    dX_dT = VectorCalcType::Zero(numberOfSegments); // later the analytical term could be inserted here
                        //    d2X_dT2 = VectorCalcType::Zero(numberOfSegments);
                        //}

                        MatrixCalcType Tau_PDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);
                        MatrixCalcType Tau_SecondPDET = MatrixCalcType::Zero(numberOfSegments, numberOfSegments);

                        const long double Rl = (long double)R_GAS_CONSTANT;
                        const long double Tl = (long double)temperature;

                        const long double invT = 1.0L / Tl;
                        const long double invT2 = invT * invT;
                        const long double invRT2 = invT2 / Rl;

                        for (int p = 0; p < numberOfSegments; p++) {

                            long double xp = (long double)_calculation.segmentConcentrations(p, i);

                            ///// Calculations for Yan's method /////
                            /* ---- dTau / dT ---- */
                            // The following code lines use vector operations
                            //VectorCalcType Tau_row = Tau.row(p);
                            //VectorCalcType dTau_dT = (((A_int.row(p) - temperature * A_int_PDET.row(p)) / (R_GAS_CONSTANT * temperature * temperature)).array() * Tau_row.transpose().array()).matrix().head(numberOfSegments);
                            //Tau_PDET.row(p) = dTau_dT;

                            // This is the more expensive way using aggregators
                            for (int q = 0; q < numberOfSegments; ++q) {

                                long double A = (long double)A_int(p, q);
                                long double A_T = (long double)A_int_PDET(p, q);
                                long double tau = (long double)Tau(p, q);

                                long double dTau = (A - temperature * A_T) / (R_GAS_CONSTANT * temperature * temperature) * tau;

                                Tau_PDET(p, q) = (calcType)dTau;
                            }


                            /* ---- Partial vg / Partial T ---- */
                            // This is the fast way using vector operations
                            //PartialvgPartialT(p) = -1 * _calculation.segmentConcentrations(p, i) * ((_calculation.segmentConcentrations.col(i).head(numberOfSegments).array() * SegmentGammas.array()).matrix().dot(dTau_dT));

                            // This is the more expensive way using aggregators
                            long double acc_ld = 0.0L;
                            for (int q = 0; q < numberOfSegments; ++q) {
                                long double xq = (long double)_calculation.segmentConcentrations(q, i);
                                long double gamma_q = (long double)SegmentGammas(q);
                                long double dTau = (long double)Tau_PDET(p, q);
                                acc_ld += xq * gamma_q * dTau;
                            }

                            PartialvgPartialT(p) = (calcType)(-xp * acc_ld);


                            ///// Alternative scaled method ///
                            //Eigen::VectorXd dTaudTForRowp = (((A_int.row(p) - temperature * A_int_PDET.row(p)) / (R_GAS_CONSTANT * temperature * temperature)).array() * (-1 * A_int.row(p) / (R_GAS_CONSTANT * temperature)).array().exp()).matrix();

                            //for (int q = 0; q < numberOfSegments; q++) {
                            //    MatrixA(p, q) = SegmentGammas(p) * SegmentGammas(p) * _calculation.segmentConcentrations(q, i) * Tau(p, q);
                            //    Vectorb(p) += _calculation.segmentConcentrations(q, i) * SegmentGammas(q) * dTaudTForRowp(q);
                            //} 
                            //Vectorb(p) *= -1 * SegmentGammas(p) * SegmentGammas(p);
                        }

                        //VectorCalcType VectorWithOnes = VectorCalcType::Constant(numberOfSegments, 1.0);
                        //MatrixA.diagonal() += VectorWithOnes;


                        ///// Calculations with Yans' Method /////

                        // The following expression for vg will always be 0. This is also theoretically expected, as the gradient vanishes at the solution (the maximum of Q).
                        // However, since a convergence threshold > 0 was chosen for the Gammas, the gradient is technically not exactly 0. 
                        // It would then need to be calculated above in the form g_m = X_m div oldgamma_m - X_m div newgamma_m.
                        // However, since the paper also assumes that g_m = 0 for the modified Hessian, this will also be assumed going forward.
                        //vg = _calculation.segmentConcentrations.col(i).head(numberOfSegments).asDiagonal() * SegmentGammas.cwiseInverse() - tempvec; // should be zero after convergence of Gammas

                        PartialSegmentGammasPartialT = modifiedHessian.ldlt().solve(-PartialvgPartialT);


                        for (int p = 0; p < numberOfSegments; p++) {
                            long double xp = (long double)_calculation.segmentConcentrations(p, i);

                            /* ---- d2Tau / dT2 ---- */
                            // The following code lines use vector operations
                            //VectorCalcType Tau_row_l = Tau.row(p);
                            //VectorCalcType dTau_dT_l = Tau_PDET.row(p);
                            //VectorCalcType d2Tau_dT2_l = (((-temperature * A_int_SecondPDET.row(p) + 2 * A_int_PDET.row(p) - 2 * A_int.row(p) / temperature) / (R_GAS_CONSTANT * temperature * temperature)).array() * Tau_row_l.transpose().array()).matrix() + (dTau_dT_l.transpose().array() * ((A_int.row(p) - temperature * A_int_PDET.row(p)) / (R_GAS_CONSTANT * temperature * temperature)).array()).matrix();
                            //Tau_SecondPDET.row(p) = d2Tau_dT2_l;

                            // This is the more expensive way using aggregators
                            for (int q = 0; q < numberOfSegments; ++q) {

                                long double A = (long double)A_int(p, q);
                                long double A_T = (long double)A_int_PDET(p, q);
                                long double A_TT = (long double)A_int_SecondPDET(p, q);
                                long double tau = (long double)Tau(p, q);
                                long double dTau = (long double)Tau_PDET(p, q);

                                /* first factor for d2TaudT2 */
                                long double termA = (-temperature * A_TT + 2.0L * A_T - 2.0L * A * invT) * invRT2;

                                /* second factor for d2TaudT2 */
                                long double termB = (A - temperature * A_T) * invRT2;

                                long double d2Tau = tau * termA + dTau * termB;

                                Tau_SecondPDET(p, q) = (double)d2Tau;
                            }

                            /* ---- Partial2 vg / Partial T2 ---- */
                            // This is the fast way using vector operations (not including terms dX/dT and d^2X/dT^2, so evaluated at constant v and x)
                            //VectorCalcType Tau_row = Tau.row(p).head(numberOfSegments).transpose();
                            //VectorCalcType dTau_dT = Tau_PDET.row(p).head(numberOfSegments).transpose();
                            //VectorCalcType d2Tau_dT2 = Tau_SecondPDET.row(p).head(numberOfSegments).transpose();
                            //Partial2vgPartialT2(p) = 2 * _calculation.segmentConcentrations(p, i) / SegmentGammas(p) * PartialSegmentGammasPartialT(p) * (-PartialvgPartialT(p) / _calculation.segmentConcentrations(p, i) + (_calculation.segmentConcentrations.col(i).head(numberOfSegments).cwiseProduct(Tau_row)).dot(PartialSegmentGammasPartialT)) + _calculation.segmentConcentrations(p, i) * (2 * _calculation.segmentConcentrations.col(i).head(numberOfSegments).cwiseProduct(dTau_dT).dot(PartialSegmentGammasPartialT) + _calculation.segmentConcentrations.col(i).head(numberOfSegments).cwiseProduct(SegmentGammas).dot(d2Tau_dT2));

                            // This is the more expensive way using aggregators (not including terms dX/dT and d^2X/dT^2, so evaluated at constant v and x)
                            long double gamma_p = (long double)SegmentGammas(p);
                            long double dGamma_p = (long double)PartialSegmentGammasPartialT(p);

                            long double term1 = 0.0L;
                            long double term2 = 0.0L;

                            for (int q = 0; q < numberOfSegments; ++q) {
                                long double xq = (long double)_calculation.segmentConcentrations(q, i);
                                long double tau = (long double)Tau(p, q);
                                long double dTau = (long double)Tau_PDET(p, q);
                                long double d2Tau = (long double)Tau_SecondPDET(p, q);
                                long double gamma_q = (long double)SegmentGammas(q);
                                long double dGamma_q = (long double)PartialSegmentGammasPartialT(q);

                                term1 += xq * tau * dGamma_q + xq * gamma_q * dTau;
                                term2 += 2 * xq * dGamma_q * dTau + xq * SegmentGammas(q) * d2Tau;
                            }

                            long double val_ld = 2.0L * xp / gamma_p * dGamma_p * term1 + xp * term2;

                            Partial2vgPartialT2(p) = (double)val_ld;
                        }

                        Partial2SegmentGammasPartialT2 = modifiedHessian.ldlt().solve(Partial2vgPartialT2);


                        ///// Alternative scaled method /////
                        //Eigen::FullPivLU<Eigen::MatrixXd> lu(MatrixA);
                        //if (lu.isInvertible()) {
                        //    PartialSegmentGammasPartialT = lu.solve(Vectorb);
                        //}

                        _calculation.segmentGammasPDETemperature.col(i).head(numberOfSegments) = PartialSegmentGammasPartialT;
                        _calculation.segmentGammasSecondPDETemperature.col(i).head(numberOfSegments) = Partial2SegmentGammasPartialT2;
                    }
                }
            }

#ifdef MEASURE_TIME
            calculateCOSMOSPACE_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateCOSMOSPACE_last).count());
            std::chrono::high_resolution_clock::time_point calculateContactStatistics_last = std::chrono::high_resolution_clock::now();
#endif


            if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
                calculateContactStatistics(_calculation, A_int, partialInteractionMatrices, const_cast<MatrixCalcType&>(Tau), gammas, int(i), temporary_averageInteractionEnergies, temporary_partialMolarEnergies, param);
            }

#ifdef MEASURE_TIME
            calculateContactStatistics_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateContactStatistics_last).count());
            std::chrono::high_resolution_clock::time_point calculateGammasForMolecules_last = std::chrono::high_resolution_clock::now();
#endif
            double div_Aeff = 1 / param.Aeff;

#ifndef OPT_NO_LOGCACHE
            // log(gammas[k]) once per segment
            std::vector<long double> lnGammaOfSegment(numberOfSegments);
            for (int k = 0; k < numberOfSegments; k++) {
                long double gamma = (long double)gammas[k];
                lnGammaOfSegment[k] = log(gamma);
            }
#endif

            for (int j = 0; j < _calculation.components.size(); j++) {


                long double lnGammaMolecule = 0.0L;
                long double lnGammaMoleculePDETAtConstantvx = 0.0L;
                long double lnGammaMoleculeSecondPDETAtConstantvx = 0.0L;
                long double lnGammaMoleculePDEvAtConstantTx = 0.0L;

                for (int k = 0; k < numberOfSegments; k++) {
                    long double Akj = (long double)_calculation.segments.SegmentTypeAreas[k][j];
                    long double divaeff = (long double)div_Aeff;
                    long double nkj = Akj * divaeff;
                    long double gamma = (long double)gammas[k];
#ifndef OPT_NO_LOGCACHE
                    long double ln_gamma = lnGammaOfSegment[k];
#else
                    long double ln_gamma = log(gamma);
#endif

                    lnGammaMolecule += nkj * ln_gamma; // log(gammas[k]) is calculated more than once although not necessary

                    if (param.sw_phi == 1 && withTDerivative) {
                        long double divgamma = 1 / gamma;
                        long double dgammadT = PartialSegmentGammasPartialT(k);
                        lnGammaMoleculePDETAtConstantvx += nkj * divgamma * dgammadT;

                        long double d2gammadT2 = Partial2SegmentGammasPartialT2(k);
                        lnGammaMoleculeSecondPDETAtConstantvx += nkj * (divgamma * d2gammadT2 - std::pow(divgamma * dgammadT, 2));
                    }
                    if (param.sw_phi == 1 && withvDerivative) {
                        long double divgamma = 1 / gamma;
                        long double dgammadv = _calculation.segmentGammasPDEVolume(k, i);
                        lnGammaMoleculePDEvAtConstantTx += nkj * divgamma * dgammadv;
                    }
                }

                //if (param.sw_phi == 1 && withTDerivative) {
                //    std::cout << "lnGammaMoleculePDETAtConstantvx: " << lnGammaMoleculePDETAtConstantvx << std::endl;
                //}

                temporary_lnGammaMolecule(i, j) = double(lnGammaMolecule);
                temporary_lnGammaMolecule_PDETemperature(i, j) = double(lnGammaMoleculePDETAtConstantvx);
                temporary_lnGammaMolecule_SecondPDETemperature(i, j) = double(lnGammaMoleculeSecondPDETAtConstantvx);
                temporary_lnGammaMolecule_PDEVolume(i, j) = double(lnGammaMoleculePDEvAtConstantTx);

#ifdef PRINT_DEBUG_INFO
                displayIfSerial("lnGammaResidual_1_" + std::to_string(h) + ": " + std::to_string(lnGammaMolecule) + "\n");
#endif

                temporary_lnGammaMolecule(i, j) = lnGammaMolecule;
            }

#ifdef MEASURE_TIME
            calculateGammasForMolecules_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateGammasForMolecules_last).count());
#endif
        }
    }

#ifdef MEASURE_TIME
    std::chrono::high_resolution_clock::time_point calculateGammasForMolecules_last = std::chrono::high_resolution_clock::now();
#endif

    /* Convert to correct reference states. */
    for (int h = 0; h < _calculation.originalNumberOfCalculations; h++) {

        int i = _calculation.actualConcentrationIndices[h];

        if (actualConcentrationIndicesToCalculate.size() > 0)
            if (std::find(actualConcentrationIndicesToCalculate.begin(), actualConcentrationIndicesToCalculate.end(), i) == actualConcentrationIndicesToCalculate.end())
                continue;

        for (int j = 0; j < _calculation.components.size(); j++) {


            double multiplier = 1.0;
            // for an ion with reference state PureComponentsOnlyNeutral
            if (_calculation.referenceStateType[h] == 1 && _calculation.referenceStateCalculationIndices[h][j] == -1) {
                multiplier = NAN;
            }
            _calculation.lnGammaResidual(h, j) = multiplier * temporary_lnGammaMolecule(i, j);
            if (withTDerivative) {
                _calculation.lnGammaResidualPDETatConstantvx(h, j) = multiplier * temporary_lnGammaMolecule_PDETemperature(i, j);
                _calculation.lnGammaResidualSecondPDETatConstantvx(h, j) = multiplier * temporary_lnGammaMolecule_SecondPDETemperature(i, j);
            }
            if (withvDerivative) {
                _calculation.lnGammaResidualPDEvatConstantTx(h, j) = multiplier * temporary_lnGammaMolecule_PDEVolume(i, j);
            }

#ifdef PRINT_DEBUG_INFO
            displayIfSerial("lnGammaResidual_2_" + std::to_string(h) + ": " + std::to_string(_calculation.lnGammaResidual(h, j)) + "\n");
#endif

            if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
                for (int k = 0; k < _calculation.components.size(); k++) {
                    _calculation.averageSurfaceEnergies(h, 0, j, k) = multiplier * temporary_averageInteractionEnergies(i, 0, j, k);

                    for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                        _calculation.averageSurfaceEnergies(h, l + 1, j, k) = multiplier * temporary_averageInteractionEnergies(i, l + 1, j, k);
                    }
                }

                if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {
                    _calculation.partialMolarEnergies(h, 0, j) = multiplier * temporary_partialMolarEnergies(i, 0, j);

                    for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                        _calculation.partialMolarEnergies(h, l + 1, j) = multiplier * temporary_partialMolarEnergies(h, l + 1, j);
                    }
                }
            }

            // only subtract reference state value if reference state type is not COSMO
            if (_calculation.referenceStateType[h] != 3 && _calculation.referenceStateType[h] != 4) {
                int indexOfReferenceStateCalculation = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[h][j]];

                _calculation.lnGammaResidual(h, j) -= temporary_lnGammaMolecule(indexOfReferenceStateCalculation, j);
                if (withTDerivative) {
                    _calculation.lnGammaResidualPDETatConstantvx(h, j) -= temporary_lnGammaMolecule_PDETemperature(indexOfReferenceStateCalculation, j);
                    _calculation.lnGammaResidualSecondPDETatConstantvx(h, j) -= temporary_lnGammaMolecule_SecondPDETemperature(indexOfReferenceStateCalculation, j);
                }
                if (withvDerivative) {
                    _calculation.lnGammaResidualPDEvatConstantTx(h, j) -= temporary_lnGammaMolecule_PDEVolume(indexOfReferenceStateCalculation, j);
                }


                if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
                    for (int k = 0; k < _calculation.components.size(); k++) {
                        _calculation.averageSurfaceEnergies(h, 0, j, k) -= temporary_averageInteractionEnergies(indexOfReferenceStateCalculation, 0, j, k);

                        for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                            _calculation.averageSurfaceEnergies(h, l + 1, j, k) -= temporary_averageInteractionEnergies(indexOfReferenceStateCalculation, l + 1, j, k);
                        }
                    }

                    if (param.sw_calculateContactStatisticsAndAdditionalProperties == 2) {
                        _calculation.partialMolarEnergies(h, 0, j) -= temporary_partialMolarEnergies(indexOfReferenceStateCalculation, 0, j);

                        for (int l = 0; l < param.numberOfPartialInteractionMatrices; l++) {
                            _calculation.partialMolarEnergies(h, l + 1, j) -= temporary_partialMolarEnergies(indexOfReferenceStateCalculation, l + 1, j);
                        }
                    }
                }
            }
        }
    }

    //if (withTDerivative && param.exp_param.at("NumericalDerivative") == true) {
    //    // Deleting the subcalculations for the numerical derivative
    //    subcalculations.pop_back();
    //    subcalculations.pop_back();
    //}

#ifdef MEASURE_TIME
    calculateGammasForMolecules_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateGammasForMolecules_last).count());
#endif
}

void calculateLnGammaResidual(parameters& param, calculation& _calculation, bool withTDerivative = false, bool withvDerivative = false) {

    std::vector<int> actualConcentrationIndicesToCalculate(0);

    calculateLnGammaResidual(param, _calculation, actualConcentrationIndicesToCalculate, withTDerivative, withvDerivative);
}


std::tuple<double, double, Eigen::VectorXd> calculateEpsilonMixAndDerivatives(const calculation& _calculation, int i_calculation, bool withvderivative = false, bool withxderivative = false) {

    int numberOfOriginalComponents = _calculation.components.size() - 1;
    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
    double temperature = _calculation.temperatures[actualConcentrationIndex];

    double epsilon_r_mix = 0;
    double depsilon_dv = 0;
    Eigen::VectorXd depsilon_dx = Eigen::VectorXd::Zero(numberOfOriginalComponents);

    if (param.sw_permittivity_mix_model == 0) {
        // Calculate mixture permittivity using correlation of Rueben et al.
        // IMPORTANT: This correlation contains the molar volume of the mixture as independent variable.
        // Since no volume calculation (for given T,P) is performed in this function, only call this function if the correct v has been
        // calculated and saved to _calculation.molarVolume(i_calculation) immediately before the function call.

        double beta = 1 / (kB * temperature);
        double v = _calculation.molarVolume(i_calculation) * pow(10, 30); // Angstrom^3/mol

        Eigen::VectorXd VectorWithPermanentDipoleAndPolarizabilityContribution = Eigen::VectorXd::Zero(numberOfOriginalComponents);
        Eigen::VectorXd VectorWitha2i = Eigen::VectorXd::Zero(numberOfOriginalComponents);

        double CoulombFactor = 1 / (4 * PI * EPSILON0_Fm);
        for (int i = 0; i < numberOfOriginalComponents; i++) {
            // Conversion of a_{11}*mu^2 from D^2 in J*Angstrom^3
            int index = _calculation.components[i]->index;
            
            double a11muconverted = param.permittivity_model_0_parameters(index, 0) * pow(DEBYE, 2) * CoulombFactor * pow(10, 30);
            double PermanentDipoleContribution = double(beta * a11muconverted);

            // Calculation of polarizability contribution
            // PolarizabilityContribution = 3*(a_{12} * alpha) -> Missing the alpha or is it somewhere else?
            double PolarizabilityContribution = double(3 * param.permittivity_model_0_parameters(index, 1));

            // Combination of both contributions in vector entry
            VectorWithPermanentDipoleAndPolarizabilityContribution(i) = PermanentDipoleContribution + PolarizabilityContribution;

            // Save a_2_i to Vector
            VectorWitha2i(i) = param.permittivity_model_0_parameters(index, 2);
        }

        // Effective dipole density yeff and correlation integral I
        double xDotContributions = _calculation.originalConcentrations.row(i_calculation).dot(VectorWithPermanentDipoleAndPolarizabilityContribution);
        double yeff = 4.0 / 9.0 * PI * N_AVOGADRO / v * xDotContributions;
        double xDota2i = _calculation.originalConcentrations.row(i_calculation).dot(VectorWitha2i);
        double I = 1 + xDota2i * (exp(-yeff) - 1);

        // Relative static permittivity of mixture
        double pertubation_term = 1 + yeff + (17.0 / 16.0 * I - 1) * pow(yeff, 2);
        epsilon_r_mix = 1 + 3 * yeff * pertubation_term;

        // Calculate partial derivative with respect to molar volume if required
        if (withvderivative == true) {
            double dyeff_dv = -4.0 / 9.0 * PI * N_AVOGADRO / pow(v, 2) * xDotContributions;
            double dI_dv = -xDota2i * exp(-yeff) * dyeff_dv;
            // The following derivative has the unit epsilon/Angstrom^3
            depsilon_dv = 3 * dyeff_dv * pertubation_term + 3 * yeff * (dyeff_dv + 17.0 / 16.0 * dI_dv * pow(yeff, 2) + 2 * (17.0 / 16.0 * I - 1) * yeff * dyeff_dv);
            // Conversion of derivative to SI units
            depsilon_dv *= 1e30;
        }

        // Calculate partial derivative with respect to mole fraction if required
        if (withxderivative == true) {
            Eigen::VectorXd dyeff_dx = 4 * PI * N_AVOGADRO / (9 * v) * VectorWithPermanentDipoleAndPolarizabilityContribution;
            Eigen::VectorXd dI_dx = VectorWitha2i * (exp(-yeff) - 1.0) - xDota2i * exp(-yeff) * dyeff_dx;
            depsilon_dx = 3 * dyeff_dx * pertubation_term + 3 * yeff * (dyeff_dx + 17.0 / 16.0 * dI_dx * pow(yeff, 2) + 2 * (17.0 / 16.0 * I - 1) * yeff * dyeff_dx);
        }
    }
     
	//std::cout << "epsilon_r_mix: " << epsilon_r_mix << std::endl;

    return { epsilon_r_mix, depsilon_dv, depsilon_dx };
}



// for openCOSMO-RS-phi this could be masde faster by implementing the following line
// and passing it to calculateLnGammaResidual as only some of the concentrations (the ones were the density has not been found)
// need to be recalculated:
// 
// void calculateEOS(parameters& param, calculation& _calculation, const std::vector<int>& originalCalculationsToRun = {}) {

// cation-anion correction of the covolume mixing rule (key bmix_l_ca = l):
//     B = sum_i x_i b_i - l * sum_c sum_a x_c x_a (b_c + b_a)
inline bool covolumeCorrection(const parameters& prm, const calculation& c, int row, double& B, Eigen::VectorXd& bbar) {
    auto it = prm.exp_param.find("bmix_l_ca");
    if (it == prm.exp_param.end() || it->second == 0.0) return false;
    double const l = it->second;
    int const n = int(c.componentCavityVolume.size());
    if (c.componentCharges.size() != n || c.originalConcentrations.cols() < n) return false;   // no charges here: no ions
    Eigen::VectorXd D = Eigen::VectorXd::Zero(n);
    double Q = 0.0;
    for (int i = 0; i < n; i++) {
        if (!(c.componentCharges(i) > 0)) continue;
        double const xi = double(c.originalConcentrations(row, i));
        for (int j = 0; j < n; j++) {
            if (!(c.componentCharges(j) < 0)) continue;
            double const xj = double(c.originalConcentrations(row, j));
            double const s = c.componentCavityVolume(i) + c.componentCavityVolume(j);
            Q += xi * xj * s;
            D(i) += xj * s;
            D(j) += xi * s;
        }
    }
    B -= l * Q;
    bbar = c.componentCavityVolume.array() - l * (D.array() - Q);
    return true;
}

void calculateEOS(parameters& param, calculation& _calculation, int originalConcentrationIndex, bool withTDerivative = false, bool withvDerivative = false, bool DoNotSetGammasToOne = false) {

#ifdef MEASURE_TIME
    calculateEOS_calls += 1;
    ScopedMicrosecondTimer calculateEOS_timer(calculateEOS_total_ms);
#endif

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[originalConcentrationIndex];
    int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[originalConcentrationIndex][0]];

    int numberOfOriginalComponents = int(_calculation.components.size()) - 1;
    int holeIndex = numberOfOriginalComponents; // it's the last component

    double temperature = double(_calculation.temperatures[actualConcentrationIndex]);
    double RT = R_GAS_CONSTANT * temperature;
    double v = _calculation.molarVolume(originalConcentrationIndex);

    // n_total = 1 -> V = v
    // i.e. volume = molar volume
    // corrected covolume (bmix_l_ca)
    double Bmix = _calculation.mixtureCavityVolume(originalConcentrationIndex);
    Eigen::VectorXd cavVolCorrected;
    bool const covolumeCorrected = covolumeCorrection(param, _calculation, originalConcentrationIndex, Bmix, cavVolCorrected);
    Eigen::VectorXd const& cavVol = covolumeCorrected ? cavVolCorrected : _calculation.componentCavityVolume;
    double molarVolumeMinusMixtureCavityVolume = std::max(v - Bmix, 0.0);

    // molesOfHoles can't be negative, its a hardcore and it would lead to negative concentrations
    double molesOfHoles = (1 / _calculation.mixtureHoleVolume(originalConcentrationIndex)) * molarVolumeMinusMixtureCavityVolume;
    double newTotalMols = molesOfHoles + 1; // molesOfHoles + sum of prior n_total
    double holeConcentration = molesOfHoles / newTotalMols;
    double PartialholeConcentrationPartialv = 1 / (_calculation.mixtureHoleVolume(originalConcentrationIndex) * std::pow(newTotalMols, 2));

    if (_calculation.recalculateConcentrationsForPhi) {

        for (int j = 0; j < numberOfOriginalComponents; j++) {
            double newConcentration = _calculation.originalConcentrations(originalConcentrationIndex, j) * (1 - holeConcentration);
            _calculation.concentrations[actualConcentrationIndex][j] = newConcentration;
        }
        _calculation.concentrations[actualConcentrationIndex][numberOfOriginalComponents] = holeConcentration;

        calculateSegmentConcentrations(_calculation, std::vector<int>{actualConcentrationIndex});
        calculatePartialSegmentConcentrationsPartialv(_calculation, std::vector<int>{actualConcentrationIndex}, PartialholeConcentrationPartialv);
        _calculation.recalculateConcentrationsForPhi = true;
    }

    // gamma residual
    if (!DoNotSetGammasToOne) {
        _calculation.segmentGammas(Eigen::indexing::all, actualConcentrationIndex).setOnes();
        _calculation.segmentGammas(Eigen::indexing::all, actualReferenceStateConcentrationIndex).setOnes();
    }
    _calculation.segmentGammasPDETemperature(Eigen::indexing::all, actualConcentrationIndex).setZero();
    _calculation.segmentGammasPDETemperature(Eigen::indexing::all, actualReferenceStateConcentrationIndex).setZero();
    _calculation.segmentGammasPDEVolume(Eigen::indexing::all, actualConcentrationIndex).setZero();
    _calculation.segmentGammasPDEVolume(Eigen::indexing::all, actualReferenceStateConcentrationIndex).setZero();


#ifdef MEASURE_TIME
    std::chrono::high_resolution_clock::time_point calculateEOS_residualGamma_last = std::chrono::high_resolution_clock::now();
#endif
    calculateLnGammaResidual(param, _calculation, std::vector<int>{actualConcentrationIndex, actualReferenceStateConcentrationIndex}, withTDerivative, withvDerivative);
#ifdef MEASURE_TIME
    calculateEOS_residualGamma_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateEOS_residualGamma_last).count());
#endif

    // residual chemical potential
    Eigen::VectorXd residualChemicalPotential = _calculation.lnGammaResidual(originalConcentrationIndex, Eigen::indexing::all).cast<double>() * RT;

    // pressures
    _calculation.repulsivePressure(originalConcentrationIndex) = RT / molarVolumeMinusMixtureCavityVolume;
    _calculation.attractivePressure(originalConcentrationIndex) = std::min(-1 * residualChemicalPotential(holeIndex) / (_calculation.mixtureHoleVolume(originalConcentrationIndex)), 0.0);

    Eigen::VectorXd lnPhiBorn;
    Eigen::VectorXd lnPhiElectrostatic;
    if (param.sw_electrolyte == 1) {
#ifdef MEASURE_TIME
        ScopedMicrosecondTimer calculateEOS_electrostatic_timer(calculateEOS_electrostatic_ms);
#endif
        ///// Born and PDH contribution to pressure /////
        // Calculation of electrostatic pressure
        double RT = R_GAS_CONSTANT * temperature;
        auto result = calculateEpsilonMixAndDerivatives(_calculation, originalConcentrationIndex, true, true);
        double epsilon_r_mix = std::get<0>(result);
        double depsilonrmix_dv = std::get<1>(result);
        Eigen::VectorXd depsilonrmix_dx = std::get<2>(result);
        Eigen::VectorXd zsquared = _calculation.componentCharges.cwiseProduct(_calculation.componentCharges);
        Eigen::VectorXd zSquaredDiva = zsquared.cwiseQuotient(_calculation.componentBornDiameter);
        double ScalarProductxTimeszSquaredDiva = _calculation.originalConcentrations.row(originalConcentrationIndex).dot(zSquaredDiva);
        if (param.sw_Born_term == 1) {
            _calculation.BornPressure(originalConcentrationIndex) = N_AVOGADRO * E_C * E_C / (4 * PI * EPSILON0_Fm * epsilon_r_mix * epsilon_r_mix) * ScalarProductxTimeszSquaredDiva * depsilonrmix_dv;
        }

        // Ix
        double IonicStrength = 0.5 * _calculation.originalConcentrations.row(originalConcentrationIndex).dot(zsquared);
        double sqrtIonicStrength = std::sqrt(IonicStrength);
        Eigen::VectorXd dIonicStrengthdx = 0.5 * zsquared;

        // PDH contribution
        double aPDH = 0.0;
        Eigen::VectorXd daPDHdx = Eigen::VectorXd::Zero(numberOfOriginalComponents);
        _calculation.electrostaticPressure(originalConcentrationIndex) = 0.0;
        if (IonicStrength > 0 && _calculation.salt_diam > 0) {

            // lambda_B
            double FactorForlambdaB = std::pow(E_C, 2) / (4.0 * PI * EPSILON0_Fm * kB * temperature);
            double lambdaB = FactorForlambdaB / epsilon_r_mix;
            double DlambdaBDv = -FactorForlambdaB / (epsilon_r_mix * epsilon_r_mix) * depsilonrmix_dv;
            Eigen::VectorXd DlambdaBDx = -FactorForlambdaB / (epsilon_r_mix * epsilon_r_mix) * depsilonrmix_dx;

            // Ax
            double constInDHParameter = (1.0 / 3.0) * sqrt(2 * PI * N_AVOGADRO) * pow(FactorForlambdaB, 1.5) * sqrt(1 / v);
            double DHParameter = constInDHParameter * pow(1 / epsilon_r_mix, 1.5);
            Eigen::VectorXd dDHParameterdx = constInDHParameter * (-1.5) / pow(epsilon_r_mix, 2.5) * depsilonrmix_dx;
            double dDHParameterdv = DHParameter * (-0.5 / v - 1.5 / epsilon_r_mix * depsilonrmix_dv);

            // b
            double ConstantInClosestApproachParameter = _calculation.salt_diam * sqrt(2 * N_AVOGADRO * E_C * E_C / (v * EPSILON0_Fm * kB * temperature));
            double ClosestApproachParameter = ConstantInClosestApproachParameter * pow(1 / epsilon_r_mix, 0.5);
            Eigen::VectorXd dClosestApproachParameterdx = ConstantInClosestApproachParameter * (-0.5) / pow(epsilon_r_mix, 1.5) * depsilonrmix_dx;
            double dClosestApproachParameterdv = ClosestApproachParameter * (-0.5 / v - 0.5 / epsilon_r_mix * depsilonrmix_dv);

            // bmod
            double bmoddivb = param.omega0 + pow(param.omega1 * lambdaB / _calculation.salt_diam, 1.5);
            double dbmoddivbdv = 1.5 * std::sqrt(param.omega1 * lambdaB / _calculation.salt_diam) * (param.omega1 / _calculation.salt_diam) * DlambdaBDv;
            Eigen::VectorXd dbmoddivbdx = 1.5 * std::sqrt(param.omega1 * lambdaB / _calculation.salt_diam) * (param.omega1 / _calculation.salt_diam) * DlambdaBDx;
            double bmod = bmoddivb * ClosestApproachParameter;
            double dbmoddv = bmoddivb * dClosestApproachParameterdv + dbmoddivbdv * ClosestApproachParameter;
            Eigen::VectorXd dbmoddx = bmoddivb * dClosestApproachParameterdx + dbmoddivbdx * ClosestApproachParameter;

            // aPDH
            double L = log(1 + bmod * sqrtIonicStrength);
            aPDH = RT * (-4 * DHParameter * IonicStrength / bmod * L);

            // electrostatic pressure
            double dalphaPDHdv = -4 * IonicStrength * ((dDHParameterdv * bmod - DHParameter * dbmoddv) / (bmod * bmod) * L
                + DHParameter / bmod * sqrtIonicStrength * dbmoddv / (1 + bmod * sqrtIonicStrength));
            _calculation.electrostaticPressure(originalConcentrationIndex) = -RT * dalphaPDHdv;

            // daPDH/dx
            Eigen::VectorXd summand1 = dDHParameterdx * IonicStrength / bmod * L;
            Eigen::VectorXd summand2 = DHParameter * (bmod * dIonicStrengthdx - IonicStrength * dbmoddx) / pow(bmod, 2) * L;
            Eigen::VectorXd summand3 = DHParameter * IonicStrength / bmod * (bmod * 1.0 / (2 * sqrtIonicStrength) * dIonicStrengthdx + sqrtIonicStrength * dbmoddx) / (1 + bmod * sqrtIonicStrength);
            daPDHdx = RT * (-4 * (summand1 + summand2 + summand3));
        }

        _calculation.totalPressure(originalConcentrationIndex) = std::max(_calculation.repulsivePressure(originalConcentrationIndex) + _calculation.attractivePressure(originalConcentrationIndex) + _calculation.BornPressure(originalConcentrationIndex) + _calculation.electrostaticPressure(originalConcentrationIndex), 1e-8);

        ///// Born and PDH contribution to fugacity coeffcients /////
        // aBorn
        double C = N_AVOGADRO * E_C * E_C / (4 * PI * EPSILON0_Fm);
        double inv_eps = 1.0 / epsilon_r_mix;
        double aBorn = C * (inv_eps - 1.0) * ScalarProductxTimeszSquaredDiva;
        Eigen::VectorXd daBorndx = C * (
            (inv_eps - 1.0) * zSquaredDiva
            - (inv_eps * inv_eps) * depsilonrmix_dx * ScalarProductxTimeszSquaredDiva
            );
        double DotProductxTimesdaBorndx = _calculation.originalConcentrations.row(originalConcentrationIndex).dot(daBorndx);

        // Some general definitions for fugacity coefficient calculation
        const double RT_inv = 1.0 / RT;
        Eigen::VectorXd x = _calculation.originalConcentrations.row(originalConcentrationIndex).transpose();
        int n = x.size();


        // lnPhiElectrostatic
        double pvRT_term = _calculation.electrostaticPressure(originalConcentrationIndex) * v * RT_inv;
        double alphaPDH = aPDH * RT_inv;
        Eigen::VectorXd dalphaPDHdx = daPDHdx * RT_inv;
        double xdot = x.dot(dalphaPDHdx);

        lnPhiElectrostatic =
            Eigen::VectorXd::Constant(n, alphaPDH + pvRT_term)
            + dalphaPDHdx
            - Eigen::VectorXd::Constant(n, xdot);

        lnPhiElectrostatic = (lnPhiElectrostatic.array().isFinite())
            .select(lnPhiElectrostatic, 0.0);
        _calculation.lnPhiElectrostatic(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = lnPhiElectrostatic.cast<double>();

        // lnPhiBorn
        lnPhiBorn = Eigen::VectorXd::Zero(n);
        if (param.sw_Born_term == 1) {
            pvRT_term = _calculation.BornPressure(originalConcentrationIndex) * v * RT_inv;
            double alphaBorn = aBorn * RT_inv;
            Eigen::VectorXd dalpha_dx = daBorndx * RT_inv;
            double x_dot_dalpha = x.dot(dalpha_dx);

            lnPhiBorn =
                Eigen::VectorXd::Constant(n, alphaBorn + pvRT_term)
                + dalpha_dx
                - Eigen::VectorXd::Constant(n, x_dot_dalpha);

            _calculation.lnPhiBorn(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = lnPhiBorn.cast<double>();
        }
    }
    else {
        _calculation.totalPressure(originalConcentrationIndex) = std::max(_calculation.repulsivePressure(originalConcentrationIndex) + _calculation.attractivePressure(originalConcentrationIndex), 1e-8);
    }


    // fugacity coefficients
    double ln_Z = log(_calculation.totalPressure(originalConcentrationIndex) * v / RT);
    Eigen::VectorXd lnPhiRepulsive = (cavVol.array() / molarVolumeMinusMixtureCavityVolume) + log(v / molarVolumeMinusMixtureCavityVolume);

    Eigen::VectorXd hemlholtzEnergyDerivative;
    Eigen::VectorXd bracket;
    if (numberOfOriginalComponents == 1) {
        bracket = _calculation.componentCavityVolume / _calculation.mixtureHoleVolume(originalConcentrationIndex);
        hemlholtzEnergyDerivative = residualChemicalPotential(Eigen::seqN(0, numberOfOriginalComponents)) + _calculation.componentCavityVolume * double(_calculation.attractivePressure(originalConcentrationIndex));
    }
    else {
        bracket = (cavVol / _calculation.mixtureHoleVolume(originalConcentrationIndex)) + ((_calculation.componentHoleVolume / _calculation.mixtureHoleVolume(originalConcentrationIndex)).array() - 1).matrix() * molesOfHoles;
        hemlholtzEnergyDerivative = residualChemicalPotential(Eigen::seqN(0, numberOfOriginalComponents)) - bracket * residualChemicalPotential(holeIndex);
    }
    Eigen::VectorXd lnPhiAttractive = (hemlholtzEnergyDerivative(Eigen::seqN(0, numberOfOriginalComponents)) / RT);

    _calculation.lnPhiAttractive(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = lnPhiAttractive.cast<double>();
    _calculation.lnPhiRepulsive(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = lnPhiRepulsive.cast<double>();

    if (param.sw_electrolyte == 1) {
        _calculation.lnPhiTotal(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = (lnPhiRepulsive.array() + lnPhiAttractive.array() + lnPhiElectrostatic.array() + lnPhiBorn.array() - ln_Z).matrix().cast<double>();
    }
    else {
        _calculation.lnPhiTotal(originalConcentrationIndex, Eigen::seqN(0, numberOfOriginalComponents)) = (lnPhiRepulsive.array() + lnPhiAttractive.array() - ln_Z).matrix().cast<double>();
    }

    // Partial derivatives of derivative properties (molar residual enthalpy and molar heat capacity at consant pressure)
    if (withTDerivative && withvDerivative) {
        // Initializations and pressure derivatives
        Eigen::VectorXd lnGammaResidualPDETatConstantvx = _calculation.lnGammaResidualPDETatConstantvx(originalConcentrationIndex, Eigen::indexing::all).cast<double>();
        Eigen::VectorXd lnGammaResidualSecondPDETatConstantvx = _calculation.lnGammaResidualSecondPDETatConstantvx(originalConcentrationIndex, Eigen::indexing::all).cast<double>();
        Eigen::VectorXd lnGammaResidualPDEvatConstantTx = _calculation.lnGammaResidualPDEvatConstantTx(originalConcentrationIndex, Eigen::indexing::all).cast<double>();
        double dPdTAtConstantvx = -R_GAS_CONSTANT / (_calculation.mixtureHoleVolume(originalConcentrationIndex)) * (residualChemicalPotential(holeIndex) / RT + temperature * lnGammaResidualPDETatConstantvx(holeIndex)) + R_GAS_CONSTANT / molarVolumeMinusMixtureCavityVolume;
        double dPdvAtConstantTx = -RT / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDEvatConstantTx(holeIndex) - RT / std::pow(molarVolumeMinusMixtureCavityVolume, 2);
        double dvdTAtConstantPx = -(dPdTAtConstantvx) / (dPdvAtConstantTx);

        /* ---- Calculation of h^r(T,v,z) = h^R(T,P,z) ---- */
        double RT2 = R_GAS_CONSTANT * std::pow(temperature, 2);
        double hres = 0.0;
        double cpres = 0.0;
        double sum1forcpres = 0.0;
        double sum2forcpres = 0.0;
        for (int j = 0; j < numberOfOriginalComponents; j++) {
            double originalConcentration = _calculation.originalConcentrations(originalConcentrationIndex, j);
            double subtrahend = cavVol(j) / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDETatConstantvx(holeIndex);
            double subtrahend_2 = cavVol(j) / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualSecondPDETatConstantvx(holeIndex);
            double lnGammaResidualPDETDifference = lnGammaResidualPDETatConstantvx(j) - subtrahend;
            double lnGammaResidualSecondPDETDifference = lnGammaResidualSecondPDETatConstantvx(j) - subtrahend_2;
            hres += originalConcentration * lnGammaResidualPDETDifference;
            sum1forcpres += originalConcentration * lnGammaResidualPDETDifference;
            sum2forcpres += originalConcentration * lnGammaResidualSecondPDETDifference;
        }
        hres *= -1 * RT2;
        hres = hres - RT2 * v / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDETatConstantvx(holeIndex);
        hres = hres + _calculation.totalPressure(originalConcentrationIndex) * v - RT;
        _calculation.molarResidualEnthalpy(originalConcentrationIndex) = hres;

        /* ---- Calculation of c_p^r(T,v,z) ---- */
        cpres = -2 * RT * sum1forcpres - RT2 * sum2forcpres;
        cpres = cpres - 2 * RT * v / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualPDETatConstantvx(holeIndex) - RT2 * v / _calculation.mixtureHoleVolume(originalConcentrationIndex) * lnGammaResidualSecondPDETatConstantvx(holeIndex);
        double dPdTAtConstantvxSquared = std::pow(dPdTAtConstantvx, 2);
        cpres = cpres - R_GAS_CONSTANT - temperature * dPdTAtConstantvxSquared / dPdvAtConstantTx;
        _calculation.residualHeatCapacityAtConstantPressure(originalConcentrationIndex) = cpres;
    }
}
//
//void calculateEOS_matrix(parameters& param, calculation& _calculation) {
//
//    int numberOfOriginalComponents = int(_calculation.components.size()) - 1;
//    int holeIndex = numberOfOriginalComponents; // it's the last component
//
//    Eigen::Map<Eigen::VectorXf> temperatures = Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(&(_calculation.temperatures[0]),
//        int(_calculation.originalNumberOfCalculations));
//
//    Eigen::VectorXd RT = R_GAS_CONSTANT * temperatures.cast<double>().array();
//
//    Eigen::VectorXd componentCavityVolumes = Eigen::VectorXd::Zero(numberOfOriginalComponents);
//    Eigen::VectorXd componentHoleVolumes = Eigen::VectorXd::Zero(numberOfOriginalComponents);
//    for (int i = 0; i < numberOfOriginalComponents; i++) {
//        int index = _calculation.components[i]->index;
//        componentCavityVolumes(i) = double(param.phi_param(index, 0)); //b_i,i
//        componentHoleVolumes(i) = double(param.phi_param(index, 3)); //b_h,i
//    }
//
//    Eigen::VectorXd originalConcentrations = _calculation.originalConcentrations.cast<double>();
//    Eigen::VectorXd mixtureHoleVolumes = originalConcentrations * componentHoleVolumes; // bh = sum(x_i * b_h,i) for all real components
//    Eigen::VectorXd molarVolumeMinusHardCoreVolume = (_calculation.molarVolume / VMOL_V) - originalConcentrations * componentCavityVolumes;
//
//    Eigen::VectorXd holeNumbers = (1 / mixtureHoleVolumes.array()) * molarVolumeMinusHardCoreVolume.array();
//    Eigen::VectorXd totalMols = holeNumbers.array() + 1;
//    Eigen::VectorXd holeConcentrations = holeNumbers.array() / totalMols.array();
//
//    if (_calculation.recalculateConcentrationsForPhi) {
//
//        Eigen::MatrixXf newConcentrations = originalConcentrations.array().colwise() / totalMols.array();
//
//        for (int i = 0; i < _calculation.originalNumberOfCalculations; i++) {
//            for (int j = 0; j < numberOfOriginalComponents; j++) {
//                _calculation.concentrations[i][j] = newConcentrations(i, j);
//            }
//            _calculation.concentrations[i][numberOfOriginalComponents] = holeConcentrations(i);
//        }
//
//        calculateSegmentConcentrations(_calculation);
//        _calculation.recalculateConcentrationsForPhi = true;
//    }
//
//    // gamma residual
//    calculateLnGammaResidual(param, _calculation);
//
//    // residual chemical potential
//    Eigen::MatrixXd residualChemicalPotential = _calculation.lnGammaResidual.cast<double>().array().colwise() * RT.array();
//
//    // pressures
//    _calculation.repulsivePressure = (RT.array() / (molarVolumeMinusHardCoreVolume.array() * VMOL_V));
//    _calculation.attractivePressure = -1 * residualChemicalPotential(Eigen::indexing::all, holeIndex).array() / mixtureHoleVolumes.array();
//    _calculation.totalPressure = _calculation.repulsivePressure + _calculation.attractivePressure;
//
//    // fugacity coefficients
//
//    Eigen::VectorXf ln_Z = (_calculation.totalPressure.array() * _calculation.molarVolume.array() / RT.array()).log();
//
//    Eigen::MatrixXf hemlholtzEnergyDerivative;
//    if (numberOfOriginalComponents == 1) {
//        hemlholtzEnergyDerivative = residualChemicalPotential(Eigen::indexing::all, Eigen::seqN(0, numberOfOriginalComponents)).array() - componentCavityVolumes.replicate(_calculation.originalNumberOfCalculations, 1).array() * _calculation.attractivePressure.array();
//    }
//    else {
//        Eigen::MatrixXf temp = (componentCavityVolumes.replicate(_calculation.originalNumberOfCalculations, 1).array().colwise() / mixtureHoleVolumes.array()) + holeConcentrations.array() * ((componentHoleVolumes.replicate(_calculation.originalNumberOfCalculations, 1).array().colwise() / mixtureHoleVolumes.array()) - 1);
//        hemlholtzEnergyDerivative = residualChemicalPotential(Eigen::indexing::all, Eigen::seqN(0, numberOfOriginalComponents)).array() - residualChemicalPotential(Eigen::indexing::all, holeIndex).array() * temp.array();
//
//    }
//    _calculation.lnPhiTotal = (hemlholtzEnergyDerivative(Eigen::indexing::all, Eigen::seqN(0, numberOfOriginalComponents)).array().colwise() / RT.array()) - ln_Z.array();
//}


// liquid-root bracket of the march in calculateIsotherm, by exponential and binary search over k
bool fastLiquidRootBracket(calculation& _calculation, int i_calculation, double v_beforeFirstStep, double targetPressure,
                           std::vector<double>& liquid_v, std::vector<double>& liquid_P, double volumeHint = -1.0) {
    const double multiplier = 1.001;
    const double max_v = 1e8;
    // cap: stay on the liquid side
    const int maxIndex = 2048;

    std::vector<double> volumes;   // volumes[k - 1] = v_k
    volumes.reserve(512);
    auto volumeAt = [&](int k) -> double {
        while (int(volumes.size()) < k) {
            const double previous = volumes.empty() ? v_beforeFirstStep : volumes.back();
            volumes.push_back(previous * multiplier);
        }
        return volumes[k - 1];
    };
    auto pressureAt = [&](int k) -> double {
        _calculation.molarVolume(i_calculation) = volumeAt(k);
        calculateEOS(param, _calculation, i_calculation);
        return _calculation.totalPressure(i_calculation);
    };

    if (volumeAt(1) >= max_v) return false;
    const double P1 = pressureAt(1);
    if (!(P1 < std::numeric_limits<double>::max())) return false;   // NaN / inf: leave it to the march
    if (P1 < targetPressure) {                                      // march: push v_1, then stop
        liquid_v.assign(1, volumeAt(1));
        liquid_P.assign(1, P1);
        return true;
    }

    int lo = 1;
    double P_lo = P1;
    int step = 1;
    int hi = -1;
    double P_hi = 0.0;

#ifndef OPT_NO_ROOT_HINTS
    // try the root of the last call first
    if (volumeHint > v_beforeFirstStep) {
        const double guess = std::ceil(std::log(volumeHint / v_beforeFirstStep) / std::log(multiplier));
        if (guess >= 3.0 && guess <= double(maxIndex)) {
            const int k = int(guess);
            if (volumeAt(k) >= max_v) return false;
            const double P_a = pressureAt(k - 1);
            if (!(P_a < P1)) return false;                          // not falling (or NaN): march
            if (P_a >= targetPressure) {
                // crossing at k or to the right of it
                lo = k - 1;
                P_lo = P_a;
                const double P_b = pressureAt(k);
                if (!(P_b < P_lo)) return false;
                if (P_b < targetPressure) {
                    hi = k;
                    P_hi = P_b;
                }
                else {
                    lo = k;
                    P_lo = P_b;
                }
            }
            else {
                // crossing left of k - 1: gallop towards k = 1 until a probe is back above target
                hi = k - 1;
                P_hi = P_a;
                int back = 1;
                while (true) {
                    const int candidate = hi - back;
                    if (candidate <= 1) {
                        lo = 1;
                        P_lo = P1;
                        break;
                    }
                    const double P_c = pressureAt(candidate);
                    if (!(P_c < P1)) return false;
                    if (P_c >= targetPressure) {
                        lo = candidate;
                        P_lo = P_c;
                        break;
                    }
                    hi = candidate;
                    P_hi = P_c;
                    back *= 2;
                }
            }
        }
    }
#endif

    // exponential search: P(lo) >= target, find hi with P(hi) < target
    while (hi < 0) {
        const int next = lo + step;
        if (next > maxIndex) return false;
        if (volumeAt(next) >= max_v) return false;
        const double P_next = pressureAt(next);
        if (!(P_next < P_lo)) return false;                         // not falling (or NaN): march
        if (P_next < targetPressure) {
            hi = next;
            P_hi = P_next;
            break;
        }
        lo = next;
        P_lo = P_next;
        step *= 2;
    }

    // bisection on the index, keeping P(lo) >= target > P(hi)
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        const double P_mid = pressureAt(mid);
        if (!(P_mid < P_lo)) return false;                          // not falling (or NaN): march
        if (P_mid < targetPressure) {
            hi = mid;
            P_hi = P_mid;
        }
        else {
            lo = mid;
            P_lo = P_mid;
        }
    }

    liquid_v.assign({ volumeAt(lo), volumeAt(hi) });
    liquid_P.assign({ P_lo, P_hi });
    return true;
}

// optTraceScope: 0 full trace, 1 liquid branch only, 2 up to the first vapour point below
// optVaporQueryPressure
std::vector<std::vector<double>> calculateIsotherm(calculation _calculation, int i_calculation, double manual_endP = 10.0, double liquidRootOnlyPressure = -1.0, double liquidRootVolumeHint = -1.0, int optTraceScope = 0, double optVaporQueryPressure = -1.0) {

#ifdef MEASURE_TIME
    calculateIsotherm_calls += 1;
    ScopedMicrosecondTimer calculateIsotherm_timer(calculateIsotherm_total_ms);
#endif

    // This function calculates the full isotherm for specified T (pure component) or specified T, z (mixture)
    // default parameter manual_endP allows to modify end pressure of isotherm which is by default 10. However, in some cases, smaller P have to be calculated.

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
    double temperature = _calculation.temperatures[actualConcentrationIndex];
    double pressureGuess = _calculation.targetPressure(i_calculation);

    // corrected covolume (bmix_l_ca)
    double BmixIsotherm = _calculation.mixtureCavityVolume(i_calculation);
    {
        Eigen::VectorXd unusedPartialCovolumes;
        covolumeCorrection(param, _calculation, i_calculation, BmixIsotherm, unusedPartialCovolumes);
    }
    double current_v = 1.05 * BmixIsotherm;
    double current_P = 11.0;

    // This section ensures that code doesn't start with liquid_P = 1e-8
    _calculation.molarVolume(i_calculation) = current_v;
    calculateEOS(param, _calculation, i_calculation);
    current_P = _calculation.totalPressure(i_calculation);
    if (current_P == 1e-8) {
        int counter = 0;
        while (current_P == 1e-8 && counter < 11) {
            current_v = (1.0 + 0.05 / std::pow(1.5, counter)) * BmixIsotherm;
            _calculation.molarVolume(i_calculation) = current_v;
            calculateEOS(param, _calculation, i_calculation);
            current_P = _calculation.totalPressure(i_calculation);
            counter += 1;
        }
    }
    current_v *= 1 / 1.001;

    double current_dPdv = 0.0;
    double last_P = std::numeric_limits<double>::max();
    double last_v = 0.0;

    std::vector<double> liquid_v = std::vector<double>();
    std::vector<double> liquid_P = std::vector<double>();
    double liquid_minimum_P = std::numeric_limits<double>::max();

    std::vector<double> vapor_v = std::vector<double>();
    std::vector<double> vapor_P = std::vector<double>();

    std::vector<double> twophaseregion_v = std::vector<double>();
    std::vector<double> twophaseregion_P = std::vector<double>();

    std::vector<double> v = std::vector<double>();
    std::vector<double> P = std::vector<double>();
    std::vector<double> P_repulsive = std::vector<double>();
    std::vector<double> P_attractive = std::vector<double>();
    std::vector<double> dPdv = std::vector<double>();

    double div_Aeff = 1 / param.Aeff;
    int numberOfOriginalComponents = _calculation.components.size() - 1;
    const int numberOfSegments = int(_calculation.segments.size());

    // rows without ions (pure solvents, vapour rows) use the analytic dP/dv test
    bool rowHasIons = false;
    if (param.sw_electrolyte == 1) {
        for (int l = 0; l < numberOfOriginalComponents; l++) {
            if (_calculation.components[l]->moleculeCharge != 0 && _calculation.originalConcentrations(i_calculation, l) > 0.0) {
                rowHasIons = true;
                break;
            }
        }
    }
    const bool useAnalyticSlopeTests = (rowHasIons == false);

    std::string mode = "liquid";
    double end_P = 10.0;
    double multiplier = 1.001;
    double max_v = 1e8;
    bool has_consecutive_outlier = false;
    int outlier_count = 0;
    bool is_outlier_last_value = false;


#ifndef OPT_NO_FAST_ROOT
    if (liquidRootOnlyPressure > 0.0 && optFastLiquidRootEnabled()) {
        std::vector<double> bracket_v;
        std::vector<double> bracket_P;
        if (fastLiquidRootBracket(_calculation, i_calculation, current_v, liquidRootOnlyPressure, bracket_v, bracket_P, liquidRootVolumeHint)) {
#ifdef MEASURE_TIME
            fastRoot_success += 1;
#endif
            std::vector<std::vector<double>> bracketReturn;
            bracketReturn.push_back(bracket_v);                 // v
            bracketReturn.push_back(bracket_P);                 // P
            bracketReturn.push_back(bracket_v);                 // liquid_v
            bracketReturn.push_back(bracket_P);                 // liquid_P
            bracketReturn.push_back(std::vector<double>());     // vapor_v
            bracketReturn.push_back(std::vector<double>());     // vapor_P
            bracketReturn.push_back(std::vector<double>());     // P_repulsive
            bracketReturn.push_back(std::vector<double>());     // P_attractive
            if (param.sw_isotherm == 1) {
                bracketReturn.push_back(std::vector<double>());
                bracketReturn.push_back(std::vector<double>());
            }
            return bracketReturn;
        }
#ifdef MEASURE_TIME
        fastRoot_fallback += 1;
#endif
    }
#endif

    while (true) {
        current_v *= multiplier;
        if ((mode == "vapor" && current_P < end_P) || current_v >= max_v)
            break;

        _calculation.molarVolume(i_calculation) = current_v;
        calculateEOS(param, _calculation, i_calculation);
        current_P = _calculation.totalPressure(i_calculation);

        Eigen::VectorXd residualChemicalPotential = _calculation.lnGammaResidual(i_calculation, Eigen::indexing::all).cast<double>() * R_GAS_CONSTANT * double(_calculation.temperatures[_calculation.actualConcentrationIndices[i_calculation]]); // not necessary
        P_attractive.push_back(_calculation.attractivePressure(i_calculation));
        P_repulsive.push_back(_calculation.repulsivePressure(i_calculation));

        v.push_back(current_v);
        P.push_back(current_P);


        if (mode == "liquid") {
            if (current_P < last_P) {
                liquid_v.push_back(current_v);
                liquid_P.push_back(current_P);
#ifndef OPT_NO_EARLY_EXIT
                // liquid-root-only callers need the trace only up to here
                if (liquidRootOnlyPressure > 0.0 && current_P < liquidRootOnlyPressure) {
                    break;
                }
#endif
            }
            else {
                calculateEOS(param, _calculation, i_calculation, false, false, false);
                if (useAnalyticSlopeTests) {
                    if (current_P != 1e-8 && last_P != 1e-8) {
                        calculateEOS(param, _calculation, i_calculation, false, true, false);
                        double sum = 0;
                        for (int k = 0; k < numberOfSegments; k++) {
                            double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationIndex));
                            sum += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
                        }
                        current_dPdv = -(R_GAS_CONSTANT * temperature) / std::pow(current_v - BmixIsotherm, 2) - (R_GAS_CONSTANT * temperature) / _calculation.mixtureHoleVolume(i_calculation) * sum;
                    }
                    else {
                        current_dPdv = 5; // random
                    }

                    if (current_dPdv > 0 || (is_outlier_last_value && has_consecutive_outlier)) {
                        mode = "search_vapor_start";
                        liquid_minimum_P = last_P;
                        multiplier = 1.01;
                        outlier_count = 0;
                        has_consecutive_outlier = false;
                        is_outlier_last_value = false;
                    }
                }
                else {
                    mode = "search_vapor_start";
                    liquid_minimum_P = last_P;
                    multiplier = 1.01;
                }
            }
        }

#ifndef OPT_NO_EARLY_EXIT
#ifndef OPT_NO_VLE_EXITS
        if (optTraceScope == 1 && rowHasIons && mode != "liquid") {
            break;
        }
#endif
        if (liquidRootOnlyPressure > 0.0 && mode != "liquid") {
            break;
        }
#endif

        if (mode == "search_vapor_start") {
            if (current_P < last_P) {
                bool VaporPhaseStart = false;
                if (useAnalyticSlopeTests) {
                    calculateEOS(param, _calculation, i_calculation, false, true, false);
                    double sum = 0;
                    for (int k = 0; k < numberOfSegments; k++) {
                        double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationIndex));
                        sum += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
                    }
                    current_dPdv = -(R_GAS_CONSTANT * temperature) / std::pow(current_v - BmixIsotherm, 2) - (R_GAS_CONSTANT * temperature) / _calculation.mixtureHoleVolume(i_calculation) * sum;
                    if (current_dPdv < 0) {
                        VaporPhaseStart = true;
                    }
                }
                else {
                    VaporPhaseStart = true;
                }

                if (VaporPhaseStart) {
                    mode = "vapor";
                    if (int(_calculation.components.size()) == 2) {
                        if (manual_endP != 10.0) {
                            end_P = manual_endP; // This case becomes active if in pure component psat calculations an estimated pressure is below 10 Pa so that the isotherm is recalculated
                        }
                        else {
                            end_P = std::max(liquid_minimum_P, 10.0);
                        }
                    }
                    else if (int(_calculation.components.size()) > 2) {
                        if (manual_endP < 10.0) {
                            end_P = manual_endP; // In this case, manual_endP was not provided as the lastP of a different isotherm, but to mark that isotherm calculation should not stop at P = 10 Pa
                        }
                        else {
                            end_P = std::max(manual_endP, 10.0); // manual_endP gives the possibility to provide lastP of a different isotherm
                        }
                    }
                    multiplier = 1.1;
                }
            }
            else if (param.sw_isotherm == 1) {
                twophaseregion_v.push_back(current_v);
                twophaseregion_P.push_back(current_P);
            }
        }

        if (mode == "vapor") {
            if (current_P < last_P) {
                if (current_P > end_P) {
                    vapor_v.push_back(current_v);
                    vapor_P.push_back(current_P);
                    has_consecutive_outlier = false;
                    is_outlier_last_value = false;
                }
            }
            else {
                //std::cout << "Have outlier, current_P: " << current_P << std::endl;
                //std::cout << "Have outlier, last_P: " << last_P << std::endl;
                //std::cout << "Have outlier, current_v: " << current_v << std::endl;
                //std::cout << "Have outlier, last_v: " << last_v << std::endl;
                bool HaveOutlier = false;
                if (useAnalyticSlopeTests) {
                    calculateEOS(param, _calculation, i_calculation, false, true, false);
                    double sum = 0;
                    for (int k = 0; k < numberOfSegments; k++) {
                        double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationIndex));
                        sum += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
                    }
                    current_dPdv = -(R_GAS_CONSTANT * temperature) / std::pow(current_v - BmixIsotherm, 2) - (R_GAS_CONSTANT * temperature) / _calculation.mixtureHoleVolume(i_calculation) * sum;
                    //std::cout << "Have outlier, current_dPdv: " << current_dPdv << std::endl;
                    if (current_dPdv > 0) {
                        HaveOutlier = true;
                    }
                }
                else {
                    HaveOutlier = true;
                }

                if (HaveOutlier > 0) {
                    outlier_count += 1;
                    is_outlier_last_value = true;
                }
                else {
                    has_consecutive_outlier = false;
                }
                if (has_consecutive_outlier == false && outlier_count < 5) {
                    has_consecutive_outlier = true;
                }
                else {
#ifdef DEBUG_INFO
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_v.txt", &v[0], v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P.txt", &P[0], v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_v.txt", &vapor_v[0], vapor_v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_P.txt", &vapor_P[0], vapor_P.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_v.txt", &liquid_v[0], liquid_v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_P.txt", &liquid_P[0], liquid_P.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_attractive.txt", &P_attractive[0], v.size(), 1);
                    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_repulsive.txt", &P_repulsive[0], v.size(), 1);
                    calculateEOS(param, _calculation, i_calculation);
#endif
                    if (rowHasIons) {
                        // ions are non-volatile: stop tracing and return the liquid branch
                        break;
                    }
                    const std::string outlierMessage = "The pressure of the vapor phase increased with the molar volume. This should not happen."
                        " (T = " + std::to_string(temperature) + " K, row " + std::to_string(i_calculation)
                        + ", " + std::to_string(numberOfOriginalComponents) + " component(s), "
                        + std::to_string(vapor_v.size()) + " vapour points so far)";
                    displayIfSerial(outlierMessage + "\n");
                    throw std::runtime_error(outlierMessage);
                }
            }
        }
#ifndef OPT_NO_VLE_EXITS
        if (optTraceScope == 2 && mode == "vapor" && !vapor_P.empty() && vapor_P.back() < optVaporQueryPressure) {
            break;
        }
#endif
        last_P = current_P;
        last_v = current_v;
    }

    std::vector<std::vector<double>> returnVector;
    returnVector.push_back(v);
    returnVector.push_back(P);
    returnVector.push_back(liquid_v);
    returnVector.push_back(liquid_P);
    returnVector.push_back(vapor_v);
    returnVector.push_back(vapor_P);
    returnVector.push_back(P_repulsive);
    returnVector.push_back(P_attractive);
    if (param.sw_isotherm == 1) {
        returnVector.push_back(twophaseregion_v);
        returnVector.push_back(twophaseregion_P);
    }

#ifdef DEBUG_INFO
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_v.txt", &v[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P.txt", &P[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_v.txt", &vapor_v[0], vapor_v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_vapor_P.txt", &vapor_P[0], vapor_P.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_v.txt", &liquid_v[0], liquid_v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_liquid_P.txt", &liquid_P[0], liquid_P.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_attractive.txt", &P_attractive[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_P_repulsive.txt", &P_repulsive[0], v.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_bh.txt", &bh[0], bh.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_mures.txt", &mu_res[0], mu_res.size(), 1);
    Write1DArraytoFile<double>(std::to_string(i_calculation) + "_mixture_cavity_volume.txt", &mixture_cavity_volume[0], mixture_cavity_volume.size(), 1);
    calculateEOS(param, _calculation, i_calculation);
#endif

    return returnVector;

}


void calculateEqCondForGivenTPn(calculation _calculation, int i_calculation, std::vector<double>& vapor_v, std::vector<double>& vapor_P, std::vector<double>& liquid_v, std::vector<double>& liquid_P, bool withIterationForPressure, bool WritePressureGuessIntoTargetPressure = false) {
#ifdef MEASURE_TIME
    calculateEqCond_calls += 1;
    ScopedMicrosecondTimer calculateEqCond_timer(calculateEqCond_total_ms);
#endif
    std::vector<double> pressureGuessesTried = std::vector<double>();
    std::vector<double> OF_P_Tried = std::vector<double>();
    bool isSupercriticalIsotherm = vapor_v.size() == 0; // this means supercritical isotherm

    int i_calculation_vapor = i_calculation;
    int i_calculation_liquid = i_calculation + int(_calculation.originalNumberOfCalculations / 2);

    double min_P = 0.0;
    double max_P = 0.0;
    _calculation.helperValue = -1;
    double pressureGuess = _calculation.targetPressure(i_calculation);

    if (isSupercriticalIsotherm) {
        _calculation.molarVolume(i_calculation_vapor) = 0.0;
        _calculation.molarVolume(i_calculation_liquid) = 0.0;

        _calculation.repulsivePressure(i_calculation_vapor) = 0.0;
        _calculation.repulsivePressure(i_calculation_liquid) = 0.0;

        _calculation.attractivePressure(i_calculation_vapor) = 0.0;
        _calculation.attractivePressure(i_calculation_liquid) = 0.0;

        _calculation.totalPressure(i_calculation_vapor) = 0.0;
        _calculation.totalPressure(i_calculation_liquid) = 0.0;

        _calculation.lnPhiAttractive(i_calculation_vapor) = NAN;
        _calculation.lnPhiAttractive(i_calculation_liquid) = NAN;

        _calculation.lnPhiRepulsive(i_calculation_vapor) = NAN;
        _calculation.lnPhiRepulsive(i_calculation_liquid) = NAN;

        _calculation.lnPhiTotal(i_calculation_vapor) = NAN;
        _calculation.lnPhiTotal(i_calculation_liquid) = NAN;
        return;
    }
    else {
        if (withIterationForPressure == true) {
            min_P = liquid_P[liquid_P.size() - 1];
            max_P = vapor_P[0];
            pressureGuess = (min_P + max_P) / 2;
        }
    }

    bool did_not_converge = false;
    auto f_pressure = [&_calculation, i_calculation_vapor, &vapor_v, &vapor_P, i_calculation_liquid, &liquid_v, &liquid_P, &did_not_converge](double pressureGuess_) {

        auto f_liquid = [&_calculation, i_calculation_liquid, pressureGuess_](double molarVolume) {
            _calculation.molarVolume(i_calculation_liquid) = molarVolume;
            calculateEOS(param, _calculation, i_calculation_liquid);
            double diff = _calculation.totalPressure(i_calculation_liquid) - pressureGuess_;
            //display(std::to_string(molarVolume * 1000) + "  " \
                                                    + std::to_string(_calculation.attractivePressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(_calculation.repulsivePressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(_calculation.totalPressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "  " + std::to_string(1000 * _calculation.lnGammaResidual(i_calculation_liquid, 1)) + "\n");
            return diff;
        };

        auto f_vapor = [&_calculation, i_calculation_vapor, pressureGuess_](double molarVolume) {
            _calculation.molarVolume(i_calculation_vapor) = molarVolume;
            calculateEOS(param, _calculation, i_calculation_vapor);
            double diff = _calculation.totalPressure(i_calculation_vapor) - pressureGuess_;
            //display(std::to_string(molarVolume * 1000) + "  " + \
                                                    std::to_string(_calculation.attractivePressure(i_calculation_vapor)) + "  "\
                                                    + std::to_string(_calculation.repulsivePressure(i_calculation_vapor)) + "  " \
                                                    + std::to_string(_calculation.totalPressure(i_calculation_vapor)) + "  " \
                                                    + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "  " + std::to_string(1000 * _calculation.lnGammaResidual(i_calculation_vapor, 1)) + "\n");
            return diff;
        };

        // liquid phase calculation
        int i_v_lower = -1;
        int i_v_upper = -1;
        for (int i_v = 0; i_v < liquid_v.size(); i_v++) {
            if (liquid_P[i_v] < pressureGuess_) {
                i_v_upper = i_v;
                i_v_lower = i_v - 1;
                break;
            }
        }

        if (i_v_lower == -1 || i_v_upper == -1) {
            did_not_converge = true;
            return 1000.0;
        }

        //display("liquid\n");
        double molarVolumeThresholdLiquid = 1e-17;
        double molarVolumeLiquidGuess = ITP(f_liquid, liquid_v[i_v_lower], liquid_v[i_v_upper], liquid_P[i_v_lower] - pressureGuess_, liquid_P[i_v_upper] - pressureGuess_, molarVolumeThresholdLiquid, 0.1);

        _calculation.molarVolume(i_calculation_liquid) = molarVolumeLiquidGuess;
        calculateEOS(param, _calculation, i_calculation_liquid);

        double diff = std::abs(_calculation.totalPressure(i_calculation_liquid) - pressureGuess_);
        if (diff > 3) {
            molarVolumeLiquidGuess = bisection(f_liquid, liquid_v[i_v_lower], liquid_v[i_v_upper], liquid_P[i_v_lower] - pressureGuess_, liquid_P[i_v_upper] - pressureGuess_, molarVolumeThresholdLiquid);
            _calculation.molarVolume(i_calculation_liquid) = molarVolumeLiquidGuess;
            calculateEOS(param, _calculation, i_calculation_liquid);
            double diffnew = std::abs(_calculation.totalPressure(i_calculation_liquid) - pressureGuess_); // for debugging
            if (std::abs(_calculation.totalPressure(i_calculation_liquid) - pressureGuess_) > diff) {
                molarVolumeLiquidGuess = ITP(f_liquid, liquid_v[i_v_lower], liquid_v[i_v_upper], liquid_P[i_v_lower] - pressureGuess_, liquid_P[i_v_upper] - pressureGuess_, molarVolumeThresholdLiquid, 0.1);
                _calculation.molarVolume(i_calculation_liquid) = molarVolumeLiquidGuess;
                calculateEOS(param, _calculation, i_calculation_liquid);
            }
            //display(std::to_string(molarVolumeLiquidGuess * 1000) + "  " + \
                                                    std::to_string(_calculation.attractivePressure(i_calculation_liquid)) + "  "\
                                                    + std::to_string(_calculation.repulsivePressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(_calculation.totalPressure(i_calculation_liquid)) + "  " \
                                                    + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "\n");
        }

        // vapor phase calculation
        i_v_lower = -1;
        i_v_upper = -1;
        for (int i_v = 0; i_v < vapor_v.size(); i_v++) {
            if (vapor_P[i_v] < pressureGuess_) {
                i_v_upper = i_v;
                i_v_lower = i_v - 1;
                break;
            }
        }

        if (i_v_lower == -1 || i_v_upper == -1) {
            did_not_converge = true;
            return 1000.0;
        }

        //display("vapor\n");
        double molarVolumeThresholdVapor = 1e-14;
        if (_calculation.calculationType == "LL") {
            molarVolumeThresholdVapor = 1e-17;
        }
        double molarVolumeVaporGuess = ITP(f_vapor, vapor_v[i_v_lower], vapor_v[i_v_upper], vapor_P[i_v_lower] - pressureGuess_, vapor_P[i_v_upper] - pressureGuess_, molarVolumeThresholdVapor, 1);

        _calculation.molarVolume(i_calculation_vapor) = molarVolumeVaporGuess;
        calculateEOS(param, _calculation, i_calculation_vapor);

        diff = std::abs(_calculation.totalPressure(i_calculation_vapor) - pressureGuess_);
        if (diff > 3) {
            molarVolumeVaporGuess = bisection(f_vapor, vapor_v[i_v_lower], vapor_v[i_v_upper], vapor_P[i_v_lower] - pressureGuess_, vapor_P[i_v_upper] - pressureGuess_, molarVolumeThresholdVapor);
            _calculation.molarVolume(i_calculation_vapor) = molarVolumeVaporGuess;
            calculateEOS(param, _calculation, i_calculation_vapor);
            if (std::abs(_calculation.totalPressure(i_calculation_vapor) - pressureGuess_) > diff) {
                molarVolumeVaporGuess = ITP(f_vapor, vapor_v[i_v_lower], vapor_v[i_v_upper], vapor_P[i_v_lower] - pressureGuess_, vapor_P[i_v_upper] - pressureGuess_, molarVolumeThresholdVapor, 1);
                _calculation.molarVolume(i_calculation_vapor) = molarVolumeVaporGuess;
                calculateEOS(param, _calculation, i_calculation_vapor);
                //display(std::to_string(molarVolumeVaporGuess * 1000) + "  " + \
                                                        std::to_string(_calculation.attractivePressure(i_calculation_vapor)) + "  "\
                                                        + std::to_string(_calculation.repulsivePressure(i_calculation_vapor)) + "  " \
                                                        + std::to_string(_calculation.totalPressure(i_calculation_vapor)) + "  " \
                                                        + std::to_string(pressureGuess_) + "  " + std::to_string(diff) + "\n");
            }
        }

        double returnValue = 0.0;
        double error = 0.0;

        for (int i_c = 0; i_c < _calculation.components.size() - 1; i_c++) {
            double vaporConcentration = _calculation.originalConcentrations(i_calculation_vapor, i_c);
            double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation_vapor, i_c));
            double liquidConcentration = _calculation.originalConcentrations(i_calculation_liquid, i_c);
            double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation_liquid, i_c));
            if (std::abs((fugacityCoefficientVapor - fugacityCoefficientLiquid) / fugacityCoefficientLiquid) > error) {
                error = std::abs((fugacityCoefficientVapor - fugacityCoefficientLiquid) / fugacityCoefficientLiquid);
                returnValue = vaporConcentration * fugacityCoefficientVapor - liquidConcentration * fugacityCoefficientLiquid;
                _calculation.helperValue = i_c;
            }
        }

        return returnValue;

    };

    double fugacityError = 1000;
    double P_lower = -1;
    double P_upper = -1;
    double OF_P_lower = -1;
    double OF_P_upper = -1;
    double fugacityConvergenceCriteria = 0.0001;

    while (fugacityError > fugacityConvergenceCriteria) {
        pressureGuessesTried.push_back(pressureGuess);
        if (WritePressureGuessIntoTargetPressure == true) {
            _calculation.targetPressure(i_calculation_vapor) = pressureGuess;
            _calculation.targetPressure(i_calculation_liquid) = pressureGuess;
        }
        double OF_P = f_pressure(pressureGuess);
        int decisiveIndex = _calculation.helperValue;
        if (decisiveIndex == -1) {
            did_not_converge = true;
        }
        if (did_not_converge) {
            break;
        }
        double fugacityVapor = _calculation.originalConcentrations(i_calculation_vapor, decisiveIndex) * exp(_calculation.lnPhiTotal(i_calculation_vapor, decisiveIndex));
        double fugacityLiquid = _calculation.originalConcentrations(i_calculation_liquid, decisiveIndex) * exp(_calculation.lnPhiTotal(i_calculation_liquid, decisiveIndex));
        fugacityError = std::abs(OF_P / fugacityLiquid);

        OF_P_Tried.push_back(OF_P);
        if (OF_P < 0) {
            P_lower = pressureGuess;
            OF_P_lower = OF_P;
        }
        else if (OF_P > 0) {
            P_upper = pressureGuess;
            OF_P_upper = OF_P;
        }


        if (P_lower != -1 && P_upper != -1) {
            break;
        }


        if (withIterationForPressure == false) {
            break;
        }
        else {
            pressureGuess *= (fugacityLiquid / fugacityVapor);
            if (pressureGuess < vapor_P[vapor_P.size() - 1]) {
                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation, std::min(pressureGuess - 1.0, 0.1));

                liquid_v = returnVector[2];
                liquid_P = returnVector[3];
                vapor_v = returnVector[4];
                vapor_P = returnVector[5];
            }
        }

        if (pressureGuessesTried.size() > 100) {

            displayIfSerial(std::to_string(_calculation.molarVolume(i_calculation_liquid) * 1000) + " "\
                + std::to_string(_calculation.attractivePressure(i_calculation_liquid)) + " "\
                + std::to_string(_calculation.repulsivePressure(i_calculation_liquid)) + " " \
                + std::to_string(_calculation.lnPhiRepulsive(i_calculation_liquid)) + " " \
                + std::to_string(_calculation.lnPhiAttractive(i_calculation_liquid)) + " " \
                + std::to_string(_calculation.lnPhiTotal(i_calculation_liquid)) + "  " \
                + std::to_string(_calculation.molarVolume(i_calculation_vapor) * 1000) + " "\
                + std::to_string(_calculation.attractivePressure(i_calculation_vapor)) + " "\
                + std::to_string(_calculation.repulsivePressure(i_calculation_vapor)) + " " \
                + std::to_string(_calculation.lnPhiRepulsive(i_calculation_vapor)) + " " \
                + std::to_string(_calculation.lnPhiAttractive(i_calculation_vapor)) + " " \
                + std::to_string(_calculation.lnPhiTotal(i_calculation_vapor)) + " " \
                + std::to_string(pressureGuess) + "\n");

        }

        if (pressureGuessesTried.size() > 250) {
            //display("did not converge");
            did_not_converge = true;
            break;
        }
    }

    if (!did_not_converge) {
        if (P_lower != -1 && P_upper != -1) {
            pressureGuess = ITP(f_pressure, P_lower, P_upper, OF_P_lower, OF_P_upper, 0.01, fugacityConvergenceCriteria);

            if (WritePressureGuessIntoTargetPressure == true) {
                _calculation.targetPressure(i_calculation_vapor) = pressureGuess;
                _calculation.targetPressure(i_calculation_liquid) = pressureGuess;
            }
        }
    }

    if (did_not_converge) {
        _calculation.molarVolume(i_calculation_vapor) = 0.0;
        _calculation.molarVolume(i_calculation_liquid) = 0.0;

        _calculation.repulsivePressure(i_calculation_vapor) = 0.0;
        _calculation.repulsivePressure(i_calculation_liquid) = 0.0;

        _calculation.attractivePressure(i_calculation_vapor) = 0.0;
        _calculation.attractivePressure(i_calculation_liquid) = 0.0;

        _calculation.totalPressure(i_calculation_vapor) = 0.0;
        _calculation.totalPressure(i_calculation_liquid) = 0.0;

        _calculation.lnPhiAttractive(i_calculation_vapor) = NAN;
        _calculation.lnPhiAttractive(i_calculation_liquid) = NAN;

        _calculation.lnPhiRepulsive(i_calculation_vapor) = NAN;
        _calculation.lnPhiRepulsive(i_calculation_liquid) = NAN;

        _calculation.lnPhiTotal(i_calculation_vapor) = NAN;
        _calculation.lnPhiTotal(i_calculation_liquid) = NAN;
    }
}


void findCriticalPointForSpecifiedx(calculation _calculation, int i_calculation) {
    std::vector<double> TemperatureGuessesTried = std::vector<double>();
    int numberOfStatePoints = int(_calculation.originalNumberOfCalculations / 2);
    int i_calculation_vapor = i_calculation;
    int i_calculation_liquid = i_calculation + numberOfStatePoints;
    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
    int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfStatePoints];
    int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
    double TemperatureToSave = _calculation.temperatures[actualConcentrationIndex]; // original given temperature should later be restored
    double TemperatureGuess = 100; // inital value at the moment: T = 100 K
    double TemperatureLimit1 = 100;
    double TemperatureLimit2 = 1e8;
    bool TemperatureLimit2Found = false;
    std::vector<std::vector<double>> returnVector;
    std::vector<double> v;
    std::vector<double> P;
    std::vector<double> liquid_v;
    std::vector<double> liquid_P;
    std::vector<double> vapor_v;
    std::vector<double> vapor_P;
    std::vector<double> two_phase_region_v;
    std::vector<double> two_phase_region_P;
    std::vector<double> P_repulsive;
    std::vector<double> P_attractive;

    // tolerance of the critical temperature in K
    const double criticalTemperatureTolerance = 0.1;
    double stepsize = 50;
    bool StartingTemperatureFound = false;
    double criticalPGuess = 0.0;
    //param.sw_isotherm = 1;

    // cheap exit for a component that is subcritical at its own temperature
    // (not for bubble/dew temperature runs, which need the real critical point)
    const bool skipScanWhenSubcritical = (param.sw_Tbub == 0 && param.sw_Tdew == 0);
    const double subcriticalCriticalTemperatureSentinel = 1.0e8;

    if (skipScanWhenSubcritical) {
        try {
            returnVector = calculateIsotherm(_calculation, i_calculation, 10.0);
            liquid_P = returnVector[3];
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];

            if (int(vapor_v.size()) > 1 && liquid_P.size() > 0 && vapor_P.size() > 0) {
                double midTwoPhasePressure = (liquid_P[liquid_P.size() - 1] + vapor_P[0]) / 2;

                _calculation.criticalT(i_calculation_vapor) = subcriticalCriticalTemperatureSentinel;
                _calculation.criticalT(i_calculation_liquid) = subcriticalCriticalTemperatureSentinel;
                _calculation.criticalP(i_calculation_vapor) = midTwoPhasePressure;
                _calculation.criticalP(i_calculation_liquid) = midTwoPhasePressure;
                _calculation.totalPressure(i_calculation_liquid) = midTwoPhasePressure;

                _calculation.temperatures[actualConcentrationIndex] = TemperatureToSave;
                _calculation.temperatures[actualConcentrationIndex2] = TemperatureToSave;
                _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureToSave;
                _calculation.TauTemperatures[0] = TemperatureToSave;
                return;
            }
        }
        catch (...) {
            // fall through to the full scan below
        }
    }
    // otherwise: full scan
    int criticalScanIterations = 0;
    while (std::abs(TemperatureLimit2 - TemperatureLimit1) > criticalTemperatureTolerance) {
        if (++criticalScanIterations > 200) {
            break;
        }
        _calculation.temperatures[actualConcentrationIndex] = TemperatureGuess;
        _calculation.temperatures[actualConcentrationIndex2] = TemperatureGuess;
        _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureGuess;
        _calculation.TauTemperatures[0] = TemperatureGuess;
        int size_vapor = 0;
        int size_two_phase_region = 0;
        try {
            returnVector = calculateIsotherm(_calculation, i_calculation, 10.0);
            v = returnVector[0];
            P = returnVector[1];
            liquid_v = returnVector[2];
            liquid_P = returnVector[3];
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
            P_repulsive = returnVector[6];
            P_attractive = returnVector[7];
            //two_phase_region_v = returnVector[8];
            //two_phase_region_P = returnVector[9];
            //size_two_phase_region = int(two_phase_region_P.size());
            size_vapor = int(vapor_v.size());
            if (size_vapor == 0 || size_vapor == 1) {
                if (StartingTemperatureFound == false) {
                    TemperatureGuess += 50;
                }
                else {
                    TemperatureLimit2 = TemperatureGuess;
                    TemperatureLimit2Found = true;
                    stepsize = std::abs(TemperatureLimit2 - TemperatureLimit1) / 10;
                    TemperatureGuess = TemperatureLimit1 + stepsize;
                }
            }
            else {
                StartingTemperatureFound = true;
                criticalPGuess = (liquid_P[liquid_P.size() - 1] + vapor_P[0]) / 2;
                TemperatureLimit1 = TemperatureGuess;
                TemperatureGuess = TemperatureGuess + stepsize;
            }
        }
        catch (...) {
            if (StartingTemperatureFound == false) {
                // still looking for the first temperature with a two-phase isotherm
                TemperatureGuess += 50;
            }
            else {
                // a failed isotherm counts as no two-phase region
                TemperatureLimit2 = TemperatureGuess;
                TemperatureLimit2Found = true;
                stepsize = std::abs(TemperatureLimit2 - TemperatureLimit1) / 10;
                TemperatureGuess = TemperatureLimit1 + stepsize;
            }
        }
    }

    if (TemperatureLimit2Found == false) {
        // iteration cap reached before any supercritical temperature was seen
        if (StartingTemperatureFound) {
            TemperatureLimit2 = TemperatureLimit1;   // last temperature that was still two-phase
        }
        else {
            throw std::runtime_error("findCriticalPointForSpecifiedx: no scanned temperature gave a two-phase isotherm, so no critical point can be estimated for this component.");
        }
    }
    //param.sw_isotherm = 0;

    //double decimal = 0.1;
    //int divider = 1;
    //for (int z = 0; z < 7; z++) {
    //    double multiplier = 1;
    //    multiplier += decimal / divider;
    //    TemperatureLimit2Found = false;
    //    TemperatureGuess = TemperatureLimit1;
    //    while (TemperatureLimit2Found == false) {
    //        _calculation.temperatures[actualConcentrationIndex] = TemperatureGuess;
    //        _calculation.temperatures[actualConcentrationIndex2] = TemperatureGuess;
    //        _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureGuess;
    //        _calculation.TauTemperatures[0] = TemperatureGuess;
    //        int size = 0;
    //        returnVector = calculateIsotherm(_calculation, i_calculation);
    //        v = returnVector[0];
    //        P = returnVector[1];
    //        liquid_v = returnVector[2];
    //        liquid_P = returnVector[3];
    //        vapor_v = returnVector[4];
    //        vapor_P = returnVector[5];
    //        P_repulsive = returnVector[6];
    //        P_attractive = returnVector[7];
    //        size = int(vapor_v.size());
    //        if (size == 0 || size == 1) {
    //            TemperatureLimit2 = TemperatureGuess;
    //            TemperatureLimit2Found = true;
    //            break;
    //        }
    //        TemperatureLimit1 = TemperatureGuess;
    //        TemperatureGuess *= multiplier;
    //    }
    //    divider *= 10;
    //}
    _calculation.criticalT(i_calculation_vapor) = TemperatureLimit2;
    _calculation.criticalT(i_calculation_liquid) = TemperatureLimit2;
    _calculation.criticalP(i_calculation_vapor) = criticalPGuess;
    _calculation.totalPressure(i_calculation_liquid) = criticalPGuess;
    _calculation.criticalP(i_calculation_liquid) = criticalPGuess;

    _calculation.temperatures[actualConcentrationIndex] = TemperatureToSave;
    _calculation.temperatures[actualConcentrationIndex2] = TemperatureToSave;
    _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureToSave;
    _calculation.TauTemperatures[0] = TemperatureToSave;
}


double calculatevForGivenTPn(calculation _calculation, int i_calculation, std::vector<double> v_vector, std::vector<double> P_vector, double pressure_) {
    auto f = [&_calculation, i_calculation, pressure_](double molarVolume) {
        _calculation.molarVolume(i_calculation) = molarVolume;
        calculateEOS(param, _calculation, i_calculation);
        double diff = _calculation.totalPressure(i_calculation) - pressure_;
        return diff;
    };

    // liquid phase calculation
    int i_v_lower = -1;
    int i_v_upper = -1;
    for (int i_v = 0; i_v < v_vector.size(); i_v++) {
        if (P_vector[i_v] < pressure_) {
            i_v_upper = i_v;
            i_v_lower = i_v - 1;
            break;
        }
    }

    if (i_v_lower == -1 || i_v_upper == -1) {
        return 1000.0;
    }

    //display("liquid\n");
    double molarVolumeLiquidGuess = ITP(f, v_vector[i_v_lower], v_vector[i_v_upper], P_vector[i_v_lower] - pressure_, P_vector[i_v_upper] - pressure_, 1e-16, 0.1);

    return molarVolumeLiquidGuess;
}


double get_radius(std::shared_ptr<molecule> component, std::string mode = "V") {

    double radius = -1;
    std::unordered_map<int, double> monoatomic_anions_fixed_radii_set = {
    {9, 1.72},        // F
    {17, 2.05},       // Cl
    {35, 2.16},       // Br
    {53, 2.32}        // I
    };

    if (int(component->atomAtomicNumbers.size()) == 1) {
        if (component->moleculeCharge > 0) {
            radius = param.R_i[component->atomAtomicNumbers(0, 0)];
        }
        else if (component->moleculeCharge < 0) {
            auto it = monoatomic_anions_fixed_radii_set.find(component->atomAtomicNumbers(0, 0));
            if (it != monoatomic_anions_fixed_radii_set.end()) {
                radius = it->second;
            }
            else {
                throw std::runtime_error(
                    "No default radii was set for ion with AtomicNumber: " + std::to_string(component->atomAtomicNumbers(0, 0))
                );
            }
        }
    }

    if (radius == -1) {
        if (mode == "A") {
            radius = std::sqrt(component->Area / (4.0 * PI));
        }
        else if(mode == "V") {
            radius = std::cbrt(component->Volume / (4.0 / 3.0 * PI));
        }
    }

    return radius;

}


// Born diameter of a charged component in m: 2 * radius * born_radius_sc
double get_born_diameter(std::shared_ptr<molecule> component) {
    if (component->moleculeCharge == 0) {
        return 1; // neutral component
    }
    double scale = 1.0;
    auto it = param.exp_param.find("born_radius_sc");
    if (it != param.exp_param.end()) {
        scale = it->second;
    }
    return 2.0 * get_radius(component) * scale * 1e-10;
}


void calculateBornAndSaltDiameter(calculation& _calculation) {
    int numberOfOriginalComponents = _calculation.components.size() - 1;

    _calculation.componentCharges = Eigen::VectorXd::Zero(numberOfOriginalComponents); // added for ES
    _calculation.componentBornDiameter = Eigen::VectorXd::Zero(numberOfOriginalComponents); // unit m
    if (param.sw_electrolyte == 1) {
        // Add Born Diameters to Eigen::Vector _calculation.componentBornDiameter and component charges to _calculation.componentCharges
        for (int j = 0; j < numberOfOriginalComponents; j++) {

            _calculation.componentCharges(j) = double(_calculation.components[j]->moleculeCharge);

            if (_calculation.components[j]->moleculeCharge == 0) {
                _calculation.componentBornDiameter(j) = 1; // random assignment to avoid division by 0
            }
            else {
                // Born diameter from the ion radius
                _calculation.componentBornDiameter(j) = get_born_diameter(_calculation.components[j]);
            }
        }

        // Calculate salt diameter a if no standard value was given (similar to part 'get salt diameter and geometrical data' in calculation_functions.py)
        if (std::isnan(_calculation.salt_diam) || _calculation.salt_diam == 0.0) {
            int n_ions = 0;
            if (std::isnan(_calculation.ion_diam_mono) || _calculation.ion_diam_mono == 0.0) {
                _calculation.ion_diam_mono = 0;
                _calculation.ion_diam_poly_sph = 0;

                for (int j = 0; j < numberOfOriginalComponents; j++) {
                    if (_calculation.components[j]->moleculeCharge != 0) {
                        // This means the component is an ion
                        n_ions += 1;
                        double area = _calculation.components[j]->Area;
                        double volume = _calculation.components[j]->Volume;
                        double sphericity = pow(PI, 1.0 / 3.0) * pow(6.0 * volume, 2.0 / 3.0) / area;
                        double radius = get_radius(_calculation.components[j]);
                        if (int(_calculation.components[j]->atomAtomicNumbers.size()) == 1) {
                            // This means ion is monoatomic
                            _calculation.ion_diam_mono += radius;
                        }
                        else {
                            _calculation.ion_diam_poly_sph += sphericity * radius;
                        }
                    }
                }
            }
            if (param.sw_LR_model.find("E-PDH") != std::string::npos) {
                _calculation.salt_diam = param.salt_diam_sc_M * _calculation.ion_diam_mono + param.salt_diam_sc_P * _calculation.ion_diam_poly_sph;
            }
            else {
                _calculation.salt_diam = 0;
                for (int j = 0; j < numberOfOriginalComponents; j++) {
                    if (_calculation.components[j]->moleculeCharge != 0) {
                        n_ions += 1;
                        _calculation.salt_diam = _calculation.salt_diam + get_radius(_calculation.components[j]);
                    }
                }
            }
            _calculation.salt_diam *= pow(10, -10); // Conversion from A to m
        }
    }
}



void calculateSubcalculations(std::vector<int>& calculationIndices) {

    // this is needed to catch exceptions in the OPENMP threads and rethrow them after the parallel section ends
    threadException e;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int i = 0; i < calculationIndices.size(); i++) {

        e.run([=] {
            int calculationIndex = calculationIndices[i];

#ifdef MEASURE_TIME
            std::chrono::high_resolution_clock::time_point rescaleSegments_last = std::chrono::high_resolution_clock::now();
#endif
            if (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) {

                subcalculations[calculationIndex].segments.clear();

                for (int j = 0; j < subcalculations[calculationIndex].components.size(); j++) {

                    std::shared_ptr<molecule> thisMolecule = subcalculations[calculationIndex].components[j];

                    for (int k = 0; k < thisMolecule->segments.size(); k++) {
                        subcalculations[calculationIndex].segments.add((unsigned short)j, thisMolecule->segments.SegmentTypeGroup[k],
                            thisMolecule->segments.SegmentTypeSigma[k],
                            thisMolecule->segments.SegmentTypeSigmaCorr[k],
                            thisMolecule->segments.SegmentTypeHBtype[k],
                            thisMolecule->segments.SegmentTypeAtomicNumber[k],
                            thisMolecule->segments.SegmentTypeAreas[k][0],
                            thisMolecule->segments.SegmentTypeMoleculeIndex[k],
                            thisMolecule->segments.SegmentTypeAtomicPolariz[k]);
                    }
                }
                subcalculations[calculationIndex].segments.sort();
                rescaleSegments(param, subcalculations[calculationIndex]);
                subcalculations[calculationIndex].segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentGammasPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentGammasPDEVolume = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentGammasSecondPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()), 1.0f);
                subcalculations[calculationIndex].segmentConcentrations = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()));
                subcalculations[calculationIndex].PartialsegmentConcentrationsPartialv = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(subcalculations[calculationIndex].segments.size())), int(subcalculations[calculationIndex].concentrations.size()));
            }


            if (param.sw_alwaysCalculateSizeRelatedParameters == 1 || (param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3) \
                || (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) || param.sw_reloadConcentrations == 1 || param.sw_reloadReferenceConcentrations == 1) {

                rescaleSegments(param, subcalculations[calculationIndex]);
                calculateSegmentConcentrations(subcalculations[calculationIndex]);
            }

#ifdef DEBUG_INFO
            display("number of components: " + std::to_string(calculations[calculationIndex].components.size()) + "\n");
            display("number of segments: " + std::to_string(calculations[calculationIndex].segments.size()) + "\n");
            display("number of concentrations: " + std::to_string(calculations[calculationIndex].concentrations.size()) + "\n");
#endif

#ifdef MEASURE_TIME
            rescaleSegments_total_ms += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - rescaleSegments_last).count();
            std::chrono::high_resolution_clock::time_point calculateCombinatorial_last = std::chrono::high_resolution_clock::now();
#endif

            if (param.sw_phi == 1) {

                calculation _calculation = subcalculations[calculationIndex];

                _calculation.originalConcentrations = subcalculations[calculationIndex].originalConcentrations; // for excess properties currently necessary because data behind maps are lost

                int numberOfOriginalComponents = _calculation.components.size() - 1;

                _calculation.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                _calculation.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                _calculation.componentCharges = Eigen::VectorXd::Zero(numberOfOriginalComponents); // added for ES
                _calculation.componentBornDiameter = Eigen::VectorXd::Zero(numberOfOriginalComponents); // unit m
                if (param.sw_electrolyte == 1) {
                    // Add Born Diameters to Eigen::Vector _calculation.componentBornDiameter and component charges to _calculation.componentCharges
                    for (int j = 0; j < numberOfOriginalComponents; j++) {

                        _calculation.componentCharges(j) = double(_calculation.components[j]->moleculeCharge);

                        if (_calculation.components[j]->moleculeCharge == 0) {
                            _calculation.componentBornDiameter(j) = 1; // random assignment to avoid division by 0
                        }
                        else {
                            // Born diameter from the ion radius
                            _calculation.componentBornDiameter(j) = get_born_diameter(_calculation.components[j]);
                        }
                    }

                    // Calculate salt diameter a if no standard value was given (similar to part 'get salt diameter and geometrical data' in calculation_functions.py)
                    calculateBornAndSaltDiameter(_calculation);
                }

                for (int i = 0; i < numberOfOriginalComponents; i++) {
                    int index = _calculation.components[i]->index;
                    _calculation.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                    _calculation.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
                }

                if (numberOfOriginalComponents == 1) {

                    for (int j = 0; j < _calculation.originalConcentrations.rows(); j++) {
                        _calculation.originalConcentrations(j, 0) = 1.0;
                    }

                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                    if (_calculation.originalNumberOfCalculations % 2 != 0)
                        throw std::runtime_error("Both phases are expected to be given as concentrations.");

                    int numberOfStatePoints = int(_calculation.originalNumberOfCalculations / 2);
                    _calculation.SaturationPressures = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), 1);

                    //#if defined(_OPENMP)
                    //#pragma omp parallel for
                    //#endif
                    for (int i_calculation = 0; i_calculation < numberOfStatePoints; i_calculation++) {

                        // critical point of the pure component, reused for the other temperatures
                        const int pureComponentIndex = _calculation.components[0]->index;
                        bool criticalPointWasCached = false;
                        double cachedCriticalT = 0.0;
                        double cachedCriticalP = 0.0;
#pragma omp critical (criticalPointCacheAccess)
                        {
                            std::unordered_map<int, std::pair<double, double>>::const_iterator cacheEntry
                                = criticalPointCache.find(pureComponentIndex);
                            if (cacheEntry != criticalPointCache.end()) {
                                criticalPointWasCached = true;
                                cachedCriticalT = cacheEntry->second.first;
                                cachedCriticalP = cacheEntry->second.second;
                            }
                        }

                        if (criticalPointWasCached) {
                            _calculation.criticalT(i_calculation) = cachedCriticalT;
                            _calculation.criticalT(i_calculation + numberOfStatePoints) = cachedCriticalT;
                            _calculation.criticalP(i_calculation) = cachedCriticalP;
                            _calculation.criticalP(i_calculation + numberOfStatePoints) = cachedCriticalP;
                            _calculation.totalPressure(i_calculation + numberOfStatePoints) = cachedCriticalP;
                        }
                        else {
                            findCriticalPointForSpecifiedx(_calculation, i_calculation);
#pragma omp critical (criticalPointCacheAccess)
                            criticalPointCache[pureComponentIndex] = std::make_pair(
                                double(_calculation.criticalT(i_calculation)),
                                double(_calculation.criticalP(i_calculation)));
                        }

                        double endP = 10.0;
                        if (param.sw_excess == 1) {
                            endP = 9.99;
                        }

                        std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation, endP);

                        std::vector<double> v = returnVector[0];
                        std::vector<double> P = returnVector[1];
                        std::vector<double> liquid_v = returnVector[2];
                        std::vector<double> liquid_P = returnVector[3];
                        std::vector<double> vapor_v = returnVector[4];
                        std::vector<double> vapor_P = returnVector[5];
                        std::vector<double> P_repulsive = returnVector[6];
                        std::vector<double> P_attractive = returnVector[7];

                        if (param.sw_Pdew == 1 || param.sw_Pbub == 1 || param.sw_PT_Flash == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1 || param.sw_excess == 1) {
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfStatePoints];
                            int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];
                            bool calculatePsatWithWilson = false;
                            if (vapor_v.size() == 0 || vapor_v.size() == 1 || temperature > _calculation.criticalT(i_calculation)) {
                                // In this case the given temperature is greater than the component's critical temperature. Then, P_isat should be calculated with Wilson Correlation.
                                calculatePsatWithWilson = true;
                            }
                            else {
                                calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true);
                                if (_calculation.totalPressure(i_calculation) == 0.0) {
                                    calculatePsatWithWilson = true;
                                }
                                //else {
                                //    double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, liquid_v, liquid_P, _calculation.targetPressure(0));
                                //    _calculation.molarVolume(i_calculation + numberOfStatePoints) = v;
                                //    calculateEOS(param, _calculation, i_calculation + numberOfStatePoints, false);

                                //    v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, _calculation.targetPressure(0));
                                //    _calculation.molarVolume(i_calculation) = v;
                                //    calculateEOS(param, _calculation, i_calculation, false);
                                //}
                            }
                            if (calculatePsatWithWilson == true) {
                                double temperatureNeededForAcentricFactor = 0.7 * _calculation.criticalT(i_calculation);
                                _calculation.temperatures[actualConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualConcentrationIndex2] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.TauTemperatures[0] = temperatureNeededForAcentricFactor;
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                std::vector<double> v = returnVector[0];
                                std::vector<double> P = returnVector[1];
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];
                                std::vector<double> P_repulsive = returnVector[6];
                                std::vector<double> P_attractive = returnVector[7];
                                calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true);
                                double reducedSaturationVaporPressure = _calculation.totalPressure(i_calculation) / _calculation.criticalP(i_calculation);
                                double acentricFactor = -1 * log10(reducedSaturationVaporPressure) - 1;
                                // Calculate Saturation Pressure with Wilson Correlation and save it in totalPressure
                                _calculation.totalPressure(i_calculation) = _calculation.criticalP(i_calculation) * exp(5.37 * (1 + acentricFactor) * (1 - _calculation.criticalT(i_calculation) / temperature));
                                _calculation.totalPressure(i_calculation + numberOfStatePoints) = _calculation.criticalP(i_calculation) * exp(5.37 * (1 + acentricFactor) * (1 - _calculation.criticalT(i_calculation) / temperature));
                                // This is just to restore original temperature in the temperature vectors
                                _calculation.temperatures[actualConcentrationIndex] = temperature;
                                _calculation.temperatures[actualConcentrationIndex2] = temperature;
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperature;
                                _calculation.TauTemperatures[0] = temperature;
                            }
                            _calculation.SaturationPressures(i_calculation, 0) = _calculation.totalPressure(i_calculation);
                            subcalculations[calculationIndex].SaturationPressures = _calculation.SaturationPressures;

                            if (param.sw_excess == 1) {
                                double pressure = _calculation.targetPressure(i_calculation);
                                double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, liquid_v, liquid_P, pressure);
                                if (v == 1000.0) {
                                    v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, vapor_v, vapor_P, pressure);
                                }
                                _calculation.molarVolume(i_calculation + numberOfStatePoints) = v;
                                calculateEOS(param, _calculation, i_calculation + numberOfStatePoints, true, true);

                                v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                                if (v == 1000.0) {
                                    v = calculatevForGivenTPn(_calculation, i_calculation, liquid_v, liquid_P, pressure);
                                }
                                _calculation.molarVolume(i_calculation) = v;
                                calculateEOS(param, _calculation, i_calculation, true, true);

                                subcalculations[calculationIndex].SaturationPressures = _calculation.SaturationPressures;
                                subcalculations[calculationIndex].segmentGammas = _calculation.segmentGammas;
                                subcalculations[calculationIndex].segmentGammasPDETemperature = _calculation.segmentGammasPDETemperature;
                                subcalculations[calculationIndex].segmentGammasPDEVolume = _calculation.segmentGammasPDEVolume;
                                subcalculations[calculationIndex].segmentGammasSecondPDETemperature = _calculation.segmentGammasSecondPDETemperature;
                                subcalculations[calculationIndex].repulsivePressure = _calculation.repulsivePressure;
                                subcalculations[calculationIndex].molarResidualEnthalpy = _calculation.molarResidualEnthalpy;
                                subcalculations[calculationIndex].residualHeatCapacityAtConstantPressure = _calculation.residualHeatCapacityAtConstantPressure;
                                //subcalculations[calculationIndex] = _calculation;
                            }
                        }
                        else if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
                            double pr = _calculation.targetPressure(i_calculation) / _calculation.criticalP(i_calculation);  // Reduced pressure
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfStatePoints];
                            int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                            if (pr > 1) {
                                double temperatureNeededForAcentricFactor = 0.7 * _calculation.criticalT(i_calculation);
                                _calculation.temperatures[actualConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualConcentrationIndex2] = temperatureNeededForAcentricFactor;
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperatureNeededForAcentricFactor;
                                _calculation.TauTemperatures[0] = temperatureNeededForAcentricFactor;
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                std::vector<double> v = returnVector[0];
                                std::vector<double> P = returnVector[1];
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];
                                std::vector<double> P_repulsive = returnVector[6];
                                std::vector<double> P_attractive = returnVector[7];
                                calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true);
                                double reducedSaturationVaporPressure = _calculation.totalPressure(i_calculation) / _calculation.criticalP(i_calculation);
                                double acentricFactor = -1 * log10(reducedSaturationVaporPressure) - 1;
                                double TsatWithWilson = _calculation.criticalT(0) / (1 - 1 / (5.373 * (1 + acentricFactor)) * log(pr));
                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex] = TsatWithWilson;
                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex2] = TsatWithWilson;
                                subcalculations[calculationIndex].temperatures[actualReferenceStateConcentrationIndex] = TsatWithWilson;
                                subcalculations[calculationIndex].TauTemperatures[0] = TsatWithWilson;
                            }
                            else {
                                std::vector<double> TemperatureGuessesTried = std::vector<double>();
                                int i_calculation_vapor = i_calculation;
                                int i_calculation_liquid = i_calculation + numberOfStatePoints;
                                double TemperatureGuess = _calculation.temperatures[actualConcentrationIndex];

                                double Tmax = 0.999 * _calculation.criticalT(i_calculation);

                                if (pr < 0.2) {
                                    TemperatureGuess = 0.3 * _calculation.criticalT(i_calculation);
                                }
                                else if (pr < 0.8) {
                                    TemperatureGuess = 0.5 * _calculation.criticalT(i_calculation);
                                }
                                else {
                                    TemperatureGuess = 0.85 * _calculation.criticalT(i_calculation);
                                }
                                double stepsize = 0.01 * std::abs(Tmax - TemperatureGuess);

                                double TemperatureLimit1 = TemperatureGuess;
                                double TemperatureLimit2;
                                bool TemperatureLimit2Found = false;
                                double OF_last = 0;
                                double OF_Limit1 = 0;
                                double OF_Limit2 = 0;
                                std::vector<std::vector<double>> returnVector;
                                std::vector<double> v;
                                std::vector<double> P;
                                std::vector<double> liquid_v;
                                std::vector<double> liquid_P;
                                std::vector<double> vapor_v;
                                std::vector<double> vapor_P;
                                std::vector<double> P_repulsive;
                                std::vector<double> P_attractive;

                                // both loops capped, the result is only an initial estimate
                                int saturationSearchIterations = 0;
                                bool saturationSearchFailed = false;
                                while (TemperatureLimit2Found == false && saturationSearchFailed == false) {
                                    if (++saturationSearchIterations > 200) {
                                        saturationSearchFailed = true;
                                        break;
                                    }
                                    _calculation.temperatures[actualConcentrationIndex] = TemperatureGuess;
                                    _calculation.temperatures[actualConcentrationIndex2] = TemperatureGuess;
                                    _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureGuess;
                                    _calculation.TauTemperatures[0] = TemperatureGuess;
                                    int size = 0;

                                    int isothermAttempts = 0;
                                    while (size < 2) {
                                        if (++isothermAttempts > 50 || TemperatureGuess < 50.0) {
                                            saturationSearchFailed = true;
                                            break;
                                        }
                                        _calculation.temperatures[actualConcentrationIndex] = TemperatureGuess;
                                        _calculation.temperatures[actualConcentrationIndex2] = TemperatureGuess;
                                        _calculation.temperatures[actualReferenceStateConcentrationIndex] = TemperatureGuess;
                                        _calculation.TauTemperatures[0] = TemperatureGuess;
                                        try {
                                            returnVector = calculateIsotherm(_calculation, i_calculation);
                                        }
                                        catch (...) {
                                            // treat as no vapour branch
                                            TemperatureGuess -= 10;
                                            size = 0;
                                            continue;
                                        }
                                        v = returnVector[0];
                                        P = returnVector[1];
                                        liquid_v = returnVector[2];
                                        liquid_P = returnVector[3];
                                        vapor_v = returnVector[4];
                                        vapor_P = returnVector[5];
                                        P_repulsive = returnVector[6];
                                        P_attractive = returnVector[7];
                                        size = int(vapor_v.size());
                                        if (size == 0 || size == 1) {
                                            TemperatureGuess = TemperatureGuess - 10;
                                        }
                                        size = int(liquid_v.size());
                                        if (size == 1) {
                                            TemperatureGuess = TemperatureGuess + stepsize;
                                            if (TemperatureGuess > Tmax) {
                                                TemperatureGuess = Tmax;
                                            }
                                        }
                                    }

                                    if (saturationSearchFailed) {
                                        break;   // no usable isotherm - do not evaluate an empty branch
                                    }
                                    calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

                                    double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation_vapor, 0));
                                    double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation_liquid, 0));
                                    double OF_T = fugacityCoefficientVapor - fugacityCoefficientLiquid;
                                    if (OF_last * OF_T < 0) {
                                        TemperatureLimit2 = TemperatureGuess;
                                        OF_Limit2 = OF_T;
                                        TemperatureLimit2Found = true;
                                    }
                                    else {
                                        TemperatureLimit1 = TemperatureGuess;
                                        OF_Limit1 = OF_T;
                                    }
                                    OF_last = OF_T;
                                    TemperatureGuess += stepsize;
                                    if (TemperatureGuess > Tmax) {
                                        TemperatureGuess = Tmax;
                                    }
                                }


                                auto f_temperature = [&_calculation, i_calculation, actualConcentrationIndex, actualConcentrationIndex2, actualReferenceStateConcentrationIndex](double temperature) {
                                    _calculation.temperatures[actualConcentrationIndex] = temperature;
                                    _calculation.temperatures[actualConcentrationIndex2] = temperature;
                                    _calculation.temperatures[actualReferenceStateConcentrationIndex] = temperature;
                                    _calculation.TauTemperatures[0] = temperature;

                                    std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                    std::vector<double> v = returnVector[0];
                                    std::vector<double> P = returnVector[1];
                                    std::vector<double> liquid_v = returnVector[2];
                                    std::vector<double> liquid_P = returnVector[3];
                                    std::vector<double> vapor_v = returnVector[4];
                                    std::vector<double> vapor_P = returnVector[5];
                                    std::vector<double> P_repulsive = returnVector[6];
                                    std::vector<double> P_attractive = returnVector[7];

                                    calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

                                    double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, 0));
                                    double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + int(_calculation.originalNumberOfCalculations / 2), 0));
                                    double OF_T = fugacityCoefficientVapor - fugacityCoefficientLiquid;

                                    return OF_T;
                                };

                                double temperatureGuess = TemperatureLimit1;
                                if (saturationSearchFailed == false && TemperatureLimit2Found) {
                                    temperatureGuess = ITP(f_temperature, TemperatureLimit1, TemperatureLimit2, OF_Limit1, OF_Limit2, 1e-10, 1e-10);
                                }
                                else {
                                    // no sign change: keep the last guess
                                    displayIfSerial("Saturation-temperature search did not bracket a root; using the last temperature guess as the initial estimate.\n");
                                }

                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex] = temperatureGuess;
                                subcalculations[calculationIndex].temperatures[actualConcentrationIndex2] = temperatureGuess;
                                subcalculations[calculationIndex].temperatures[actualReferenceStateConcentrationIndex] = temperatureGuess;
                                subcalculations[calculationIndex].TauTemperatures[0] = temperatureGuess;
                                //calculateEOS(param, subcalculations[calculationIndex], i_calculation);
                            }
                        }
                        else {
                            _calculation.SaturationPressures(i_calculation, 0) = _calculation.totalPressure(i_calculation);

                            double pressure = _calculation.targetPressure(i_calculation);
                            double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfStatePoints, liquid_v, liquid_P, pressure);
                            _calculation.molarVolume(i_calculation + numberOfStatePoints) = v;
                            calculateEOS(param, _calculation, i_calculation + numberOfStatePoints, true);

                            v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                            _calculation.molarVolume(i_calculation) = v;
                            calculateEOS(param, _calculation, i_calculation, true);
                            subcalculations[calculationIndex].SaturationPressures = _calculation.SaturationPressures;
                            subcalculations[calculationIndex].segmentGammas = _calculation.segmentGammas;
                            subcalculations[calculationIndex].segmentGammasPDETemperature = _calculation.segmentGammasPDETemperature;
                            subcalculations[calculationIndex].segmentGammasPDEVolume = _calculation.segmentGammasPDEVolume;
                            subcalculations[calculationIndex].segmentGammasSecondPDETemperature = _calculation.segmentGammasSecondPDETemperature;
                            subcalculations[calculationIndex].repulsivePressure = _calculation.repulsivePressure;
                            //subcalculations[calculationIndex] = _calculation;

                        }
                    }
                }
            }
            });
    }
    e.rethrow();
}


void calculateBubblePoint(calculation _calculation, int i_calculation, double epsilon, bool passInnerLoop = false) {

    // This function performes the two loops of the Bubble Point algorithm -> Initialization has to be done outside of the function.
    // This means that initial P, K and y guess are assumed to be already given. 

    param.sw_isotherm = 1;

    int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);
    bool bubblePorTFound = false;
    int counter = 0;
    int outlier_counter = 0;
    double last_sum = 100000;
    double TemperatureOrPressureWithSumyLowerNearestOne = 0.0;
    double sumyForTemperatureOrPressureWithSumyLowerNearestOne = 0.0;
    double TemperatureOrPressureWithSumyHigherNearestOne = 0.0;
    double sumyForTemperatureOrPressureWithSumyHigherNearestOne = 1000;
    std::vector<std::vector<double>> returnVector;
    std::vector<double> liquid_v;
    std::vector<double> liquid_P;

    // Outer loop
    while (!bubblePorTFound) {
        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

        // Liquid phase calculation (for Bubble P calculation need to be performed only once as x and T are specified)
        if (counter == 0 || param.sw_Tbub == 1) {
#ifndef OPT_NO_VLE_EXITS
            returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations, 10.0, -1.0, -1.0, 1);
#else
            returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
#endif
            liquid_v = returnVector[2];
            liquid_P = returnVector[3];
        }

        // Inner loop
        double sum = _calculation.originalConcentrations.array().row(i_calculation).sum();;
        double difference = 1000.0;
        int inner_counter = 0;
        double end_P = *std::min_element(liquid_P.begin(), liquid_P.end());
        while (difference > epsilon) {
            if (_calculation.targetPressure(i_calculation) < 10) {
                end_P = std::max(_calculation.targetPressure(i_calculation) - 1.0, 1.0);
            }

            // Vapor phase calculation
            _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
            _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;
#ifndef OPT_NO_VLE_EXITS
            returnVector = calculateIsotherm(_calculation, i_calculation, end_P, -1.0, -1.0, 2, _calculation.targetPressure(i_calculation));
#else
            returnVector = calculateIsotherm(_calculation, i_calculation, end_P);
#endif
            std::vector<double> vapor_v = returnVector[4];
            std::vector<double> vapor_P = returnVector[5];
            std::vector<double> two_phase_region_P = returnVector[9];

            if (two_phase_region_P.size() <= 1 || vapor_P.size() == 0) {
                // In this case there is only one supercritical phase in equilibrium with the liquid phase
                vapor_v = returnVector[2];
                vapor_P = returnVector[3];
            }

            // Calculate Properties for current P, T, x
            calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

            if (_calculation.totalPressure(i_calculation) == 0.0 || passInnerLoop == true) {
                // In this case no equilibrium was found and we stop the calculation 
                //Write1DArrayToFile<double>(std::to_string(i_calculation) + "_v.txt", &returnVector[0][0], returnVector[0].size(), 1);
                //Write1DArrayToFile<double>(std::to_string(i_calculation) + "_P.txt", &returnVector[1][0], returnVector[0].size(), 1);
                break;
            }

            // Update K values and y for all components i
            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                if (_calculation.components[m]->moleculeCharge != 0) {
                    // non-volatile ion: K = 0, y = 0
                    _calculation.PartitionCoefficients(i_calculation, m) = 0.0;
                    _calculation.originalConcentrations(i_calculation, m) = 0.0;
                    continue;
                }
                double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, m));
                double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                _calculation.PartitionCoefficients(i_calculation, m) = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                _calculation.originalConcentrations(i_calculation, m) = _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) * _calculation.PartitionCoefficients(i_calculation, m);
            }

            // Calculate difference = (sum_i y_i)^new - (sum_i y_i)^old, update sum and normalize y_i
            difference = std::abs(_calculation.originalConcentrations.array().row(i_calculation).sum() - sum);
            sum = _calculation.originalConcentrations.array().row(i_calculation).sum();
            _calculation.originalConcentrations.array().row(i_calculation) *= 1 / sum;

            // Inner loop stops if either difference is smaller than epsilon or number of iterations is >= 10
            inner_counter += 1;
            if (inner_counter == 10) {
                break;
            }
        }

        passInnerLoop = false;

        if (_calculation.totalPressure(i_calculation) == 0.0) {
            if (param.sw_Tbub == 1) {
                _calculation.targetTemperature(i_calculation) = 0.0;
            }
            break;
        }

        // Check if sum_i y_i is close enough to target value 1
        if (std::abs(sum - 1.0) < epsilon) {
            bubblePorTFound = true;
            if (param.sw_Tbub == 1) {
                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                _calculation.targetTemperature(i_calculation) = _calculation.temperatures[actualConcentrationIndex];
                _calculation.targetTemperature(i_calculation + numberOfCalculations) = _calculation.temperatures[actualConcentrationIndex];
            }
        }
        else {
            counter += 1;
            if (counter == 20) {
                epsilon *= 10;
                counter = 0;
            }
            if (param.sw_Pbub == 1) {
                // Update pressure if Bubble P is to be calculated
                if (sum < 1 && sum > sumyForTemperatureOrPressureWithSumyLowerNearestOne) {
                    TemperatureOrPressureWithSumyLowerNearestOne = _calculation.targetPressure(i_calculation);
                    sumyForTemperatureOrPressureWithSumyLowerNearestOne = sum;
                }
                else if (sum > 1 && sum < sumyForTemperatureOrPressureWithSumyHigherNearestOne) {
                    TemperatureOrPressureWithSumyHigherNearestOne = _calculation.targetPressure(i_calculation);
                    sumyForTemperatureOrPressureWithSumyHigherNearestOne = sum;
                }
                double new_pressure = _calculation.targetPressure(i_calculation) * sum;
                if (TemperatureOrPressureWithSumyLowerNearestOne != 0.0 && TemperatureOrPressureWithSumyHigherNearestOne != 0.0) {
                    new_pressure = TemperatureOrPressureWithSumyLowerNearestOne + (1.0 - sumyForTemperatureOrPressureWithSumyLowerNearestOne) / (sumyForTemperatureOrPressureWithSumyHigherNearestOne - sumyForTemperatureOrPressureWithSumyLowerNearestOne) * (TemperatureOrPressureWithSumyHigherNearestOne - TemperatureOrPressureWithSumyLowerNearestOne);
                }
                _calculation.targetPressure(i_calculation) = new_pressure;
                _calculation.targetPressure(i_calculation + numberOfCalculations) = new_pressure;
            }
            else if (param.sw_Tbub == 1) {
                // Update temperature if Bubble T is to be calculated
                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfCalculations];
                int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                if (sum < 1 && sum > sumyForTemperatureOrPressureWithSumyLowerNearestOne) {
                    TemperatureOrPressureWithSumyLowerNearestOne = _calculation.temperatures[actualConcentrationIndex];
                    sumyForTemperatureOrPressureWithSumyLowerNearestOne = sum;
                }
                else if (sum > 1 && sum < sumyForTemperatureOrPressureWithSumyHigherNearestOne) {
                    TemperatureOrPressureWithSumyHigherNearestOne = _calculation.temperatures[actualConcentrationIndex];
                    sumyForTemperatureOrPressureWithSumyHigherNearestOne = sum;
                }
                double new_temperature = _calculation.temperatures[actualConcentrationIndex];
                // Usually, new T is chosen based on a linear inerpolation strategy.
                // The routine in the else block is used alternatively.
                if (TemperatureOrPressureWithSumyLowerNearestOne != 0.0 && TemperatureOrPressureWithSumyHigherNearestOne != 0.0) {
                    new_temperature = TemperatureOrPressureWithSumyLowerNearestOne + (1.0 - sumyForTemperatureOrPressureWithSumyLowerNearestOne) / (sumyForTemperatureOrPressureWithSumyHigherNearestOne - sumyForTemperatureOrPressureWithSumyLowerNearestOne) * (TemperatureOrPressureWithSumyHigherNearestOne - TemperatureOrPressureWithSumyLowerNearestOne);
                }
                else {
                    if (sum < 0.98) {
                        sum = 0.98;
                    }
                    else {
                        int a = static_cast<int>(sum);
                        sum = a + (sum - a) / 10;
                    }
                    if (sum > last_sum) {
                        outlier_counter += 1;
                        if (outlier_counter == 3) {
                            _calculation.temperatures[actualConcentrationIndex] = 0.0;
                            _calculation.targetTemperature(i_calculation) = 0.0;
                            //throw std::runtime_error("y_T has increased three times in a row. Bubble T algorithm has not converged.");
                            break;
                        }
                    }
                    else {
                        outlier_counter = 0;
                    }
                    new_temperature = _calculation.temperatures[actualConcentrationIndex] * 1 / sum;
                }
                last_sum = sum;
                _calculation.temperatures[actualConcentrationIndex] = new_temperature;
                _calculation.temperatures[actualConcentrationIndex2] = new_temperature;
                _calculation.temperatures[actualReferenceStateConcentrationIndex] = new_temperature;
                for (int g = 0; g < _calculation.TauConcentrationIndices.size(); g++) {
                    for (int h = 0; h < _calculation.TauConcentrationIndices[g].size(); h++) {
                        if (actualConcentrationIndex == _calculation.TauConcentrationIndices[g][h]) {
                            _calculation.TauTemperatures[g] = new_temperature;
                        }
                    }
                }
            }
            sum = _calculation.originalConcentrations.array().row(i_calculation).sum();
            _calculation.originalConcentrations.array().row(i_calculation) *= 1 / sum;
        }
    }
    param.sw_isotherm = 0;
}


void calculateDewPoint(calculation _calculation, int i_calculation, double epsilon) {

    // This function performes the two loops of the Dew Point algorithm -> Initialization has to be done outside of the function.
    // This means that initial P, K and x guess are assumed to be already given.

    int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);
    bool dewPorTFound = false;
    int counter = 0;
    std::vector<std::vector<double>> returnVector;
    std::vector<double> vapor_v;
    std::vector<double> vapor_P;

    // Outer loop
    while (!dewPorTFound) {
        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

        // Vapor phase calculation (for Dew P calculation need to be performed only once as y and T are specified)
        if (counter == 0 || param.sw_Tdew == 1) {
            returnVector = calculateIsotherm(_calculation, i_calculation);
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
        }

        // Inner loop (iterating on x for fixed T, P, y)
        double sum = _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
        double difference = 1000.0;
        int inner_counter = 0;
        while (difference > epsilon) {

            // Liquid phase calculation
            _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
            _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;
            returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];

            // Calculate Properties for current P, T, x  
            calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

            // Update K values and x
            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, m));
                double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                _calculation.PartitionCoefficients(i_calculation, m) = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = _calculation.originalConcentrations(i_calculation, m) / _calculation.PartitionCoefficients(i_calculation, m);
            }
            difference = std::abs(_calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum() - sum);
            sum = _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
            _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1 / sum;
            inner_counter += 1;
            if (inner_counter == 10) {
                break;
            }
        }

        if (_calculation.totalPressure(i_calculation) == 0.0) {
            if (param.sw_Tbub == 1) {
                _calculation.targetTemperature(i_calculation) = 0.0;
            }
            break;
        }

        // Check if sum_i x_i is close enough to target value 1 
        if (std::abs(sum - 1.0) < epsilon) {
            dewPorTFound = true;
            if (param.sw_Tdew == 1) {
                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                _calculation.targetTemperature(i_calculation) = _calculation.temperatures[actualConcentrationIndex];
                _calculation.targetTemperature(i_calculation + numberOfCalculations) = _calculation.temperatures[actualConcentrationIndex];
            }
        }
        else {
            counter += 1;
            if (counter == 20) {
                epsilon *= 10;
                counter = 0;
            }
            if (param.sw_Pdew == 1) {
                _calculation.targetPressure(i_calculation) *= 1 / sum;
                _calculation.targetPressure(i_calculation + numberOfCalculations) *= 1 / sum;
            }
            else if (param.sw_Tdew == 1) {
                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfCalculations];
                int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                if (sum < 0.95) {
                    sum = 0.95;
                }
                else {
                    int a = static_cast<int>(sum);
                    sum = a + (sum - a) / 10;
                }
                _calculation.temperatures[actualConcentrationIndex] *= sum;
                _calculation.temperatures[actualConcentrationIndex2] *= sum;
                _calculation.temperatures[actualReferenceStateConcentrationIndex] *= sum;
                for (int g = 0; g < _calculation.TauConcentrationIndices.size(); g++) {
                    for (int h = 0; h < _calculation.TauConcentrationIndices[g].size(); h++) {
                        if (actualConcentrationIndex == _calculation.TauConcentrationIndices[g][h]) {
                            _calculation.TauTemperatures[g] *= sum;
                        }
                    }
                }
            }
            sum = _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
            _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1 / sum;
        }
    }
}


std::vector<std::vector<double>> activitySampling(std::vector<std::shared_ptr<molecule>> components, double temperature, double pressure, double stepsize) {
    std::vector<double> x = {};
    std::vector<double> lnGamma1 = {};
    std::vector<double> lnGamma2 = {};
    std::vector<double> a1 = {};
    std::vector<double> a2 = {};

    std::vector<double> pressureVector;
    pressureVector.push_back(double(pressure));
    std::vector<std::vector<double>> concentrationx1(1);
    concentrationx1[0].push_back(0);
    concentrationx1[0].push_back(1);

    calculation calculationScheme = loadSubcalculation(components, { double(temperature) }, concentrationx1, pressureVector);
    int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
    calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});

    int numberOfOriginalComponents = calculationScheme.components.size() - 1;
    calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    calculationScheme.calculationType = "LL";
    calculationScheme.originalNumberOfCalculations = 2;


    for (int i = 0; i < numberOfOriginalComponents; i++) {
        int index = calculationScheme.components[i]->index;
        calculationScheme.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
        calculationScheme.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
    }


    // Calculate ln phi for pure component 2
    calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
    calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;
    std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 0);
    std::vector<double> liquid_v = returnVector[2];
    std::vector<double> liquid_P = returnVector[3];
    double v = calculatevForGivenTPn(calculationScheme, 1, liquid_v, liquid_P, pressure);
    calculationScheme.molarVolume(1) = v;
    calculateEOS(param, calculationScheme, 1);
    double lnPhiComp2 = calculationScheme.lnPhiTotal(1, 1);

    // Calculate ln phi for pure component 2
    Eigen::VectorXd newxconcentration(2);
    newxconcentration(0) = 1;
    newxconcentration(1) = 0;
    calculationScheme.originalConcentrations.row(0) = newxconcentration;
    calculationScheme.originalConcentrations.row(1) = newxconcentration;
    calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
    calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;
    returnVector = calculateIsotherm(calculationScheme, 0);
    liquid_v = returnVector[2];
    liquid_P = returnVector[3];
    v = calculatevForGivenTPn(calculationScheme, 1, liquid_v, liquid_P, pressure);
    calculationScheme.molarVolume(1) = v;
    calculateEOS(param, calculationScheme, 1);
    double lnPhiComp1 = calculationScheme.lnPhiTotal(1, 0);

    newxconcentration(0) = 0;
    newxconcentration(1) = 1;
    calculationScheme.originalConcentrations.row(0) = newxconcentration;
    calculationScheme.originalConcentrations.row(1) = newxconcentration;
    double current_x1 = 0;
    int i_calculation = 0;
    const double epsilon = 0.001; // Toleranz für Änderungen
    const double min_stepsize = 0.0005; // Minimale Schrittweite
    const double max_stepsize = 0.05;  // Maximale Schrittweite

    while (current_x1 < 1.0) {

        // Activity coefficient calculation (could be packed into a seperate function)
        calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
        calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;
        std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 0);
        std::vector<double> liquid_v = returnVector[2];
        std::vector<double> liquid_P = returnVector[3];
        double v = calculatevForGivenTPn(calculationScheme, 1, liquid_v, liquid_P, pressure);
        calculationScheme.molarVolume(1) = v;
        calculateEOS(param, calculationScheme, 1);

        x.push_back(calculationScheme.originalConcentrations(1, 0));
        double lnGamma1_i = calculationScheme.lnPhiTotal(1, 0) - lnPhiComp1;
        double lnGamma2_i = calculationScheme.lnPhiTotal(1, 1) - lnPhiComp2;
        lnGamma1.push_back(lnGamma1_i);
        lnGamma2.push_back(lnGamma2_i);
        int actualsize = int(x.size());
        double a1_i = x[actualsize - 1] * exp(lnGamma1[actualsize - 1]);
        double a2_i = (1 - x[actualsize - 1]) * exp(lnGamma2[actualsize - 1]);
        a1.push_back(a1_i);
        a2.push_back(a2_i);

        Eigen::VectorXd newxconcentration(2);
        if (x.size() > 1) {
            double delta_a1 = std::abs(a1[a1.size() - 1] - a1[a1.size() - 2]);
            double delta_a2 = std::abs(a2[a2.size() - 1] - a2[a2.size() - 2]);
            double factor = std::min(std::sqrt(epsilon / (delta_a1 + 1e-10)), std::sqrt(epsilon / (delta_a2 + 1e-10)));
            stepsize = stepsize * factor;
            stepsize = std::max(min_stepsize, std::min(stepsize, max_stepsize));
        }

        newxconcentration(0) = std::min(x[actualsize - 1] + stepsize, 1.0);
        newxconcentration(1) = 1 - newxconcentration(0);
        calculationScheme.originalConcentrations.row(0) = newxconcentration;
        calculationScheme.originalConcentrations.row(1) = newxconcentration;
        current_x1 = newxconcentration(0);
        //std::cout << current_x1 << std::endl;
    }

    std::vector<std::vector<double>> returnVector2;
    returnVector2.push_back(x);
    returnVector2.push_back(lnGamma1);
    returnVector2.push_back(lnGamma2);
    returnVector2.push_back(a1);
    returnVector2.push_back(a2);

    if (param.exp_param.at("EigenMatrixToFile") == 1) {
        Write1DArrayToFile<double>(std::to_string(i_calculation) + "_x.txt", &x[0], x.size(), 1);
        Write1DArrayToFile<double>(std::to_string(i_calculation) + "_lnGamma1.txt", &lnGamma1[0], lnGamma1.size(), 1);
        Write1DArrayToFile<double>(std::to_string(i_calculation) + "_lnGamma2.txt", &lnGamma2[0], lnGamma2.size(), 1);
        Write1DArrayToFile<double>(std::to_string(i_calculation) + "_a1.txt", &a1[0], a1.size(), 1);
        Write1DArrayToFile<double>(std::to_string(i_calculation) + "_a2.txt", &a2[0], a2.size(), 1);
    }

    return returnVector2;
}


std::vector<std::vector<Eigen::VectorXd>> samplingForVLE(std::vector<std::shared_ptr<molecule>> components, double temperature, int IndexOfSupercriticalComponent, Eigen::VectorXd startx, Eigen::VectorXd starty, Eigen::VectorXd startK, double startpressure, std::vector<double> PureTsat = {}, double end_x = 1.0) {

    std::vector<std::vector<double>> concentrationx1(1);
    int IndexOfSubcriticalComponent;
    if (IndexOfSupercriticalComponent == 0) {
        IndexOfSubcriticalComponent = 1;
        concentrationx1[0].push_back(startx(0) + 0.0001);
        concentrationx1[0].push_back(startx(1) - 0.0001);
    }
    if (IndexOfSupercriticalComponent == 1) {
        IndexOfSubcriticalComponent = 0;
        concentrationx1[0].push_back(startx(0) - 0.0001);
        concentrationx1[0].push_back(startx(1) + 0.0001);
    }

    std::vector<Eigen::VectorXd> x = {};
    std::vector<Eigen::VectorXd> y = {};
    std::vector<Eigen::VectorXd> K = {};
    std::vector<Eigen::VectorXd> Pressures = {};
    std::vector<Eigen::VectorXd> Temperatures = {};

    x.push_back(startx);
    K.push_back(startK);
    y.push_back(starty);
    Eigen::VectorXd pressureVec(1);
    pressureVec(0) = startpressure;
    Pressures.push_back(pressureVec);
    Eigen::VectorXd tempVec(1);
    tempVec(0) = temperature;
    Temperatures.push_back(tempVec);

    std::vector<double> pressureVector;
    pressureVector.push_back(double(startpressure));

    double DeltaT = 0.0;
    if (PureTsat.size() > 0) {
        DeltaT = 0.001 * (PureTsat[IndexOfSupercriticalComponent] - PureTsat[IndexOfSubcriticalComponent]);
    }

    // To not affect the actual concentrations given exernally, the tracing procedure uses a new calculaion structure that is here created first
    calculation calculationScheme = loadSubcalculation(components, { double(temperature + DeltaT) }, concentrationx1, pressureVector);
    int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
    calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});
    calculationScheme.PartitionCoefficients = Eigen::MatrixXd::Zero(2, 2);
    calculationScheme.PartitionCoefficients.row(0) = startK;
    for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
        double value = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
        calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
    }
    calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
    int numberOfOriginalComponents = calculationScheme.components.size() - 1;
    calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    calculationScheme.calculationType = "VL";
    calculationScheme.originalNumberOfCalculations = 2;
    for (int i = 0; i < numberOfOriginalComponents; i++) {
        int index = calculationScheme.components[i]->index;
        calculationScheme.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
        calculationScheme.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
    }


    bool VLEDoesNotConvergeAnymore = false;
    int i_calculation = 0;
    int actualsize = 0;
    double stepsize = 0.0001;
    double last_x = 0;
    int counter_for_x_almost_y = 0;
    while (VLEDoesNotConvergeAnymore != true || last_x <= end_x) {
        calculateBubblePoint(calculationScheme, i_calculation, 0.00001);
        if (param.sw_Pbub == 1 && param.sw_criticalPoint != 1 && param.sw_azeotropicPoint != 1 && calculationScheme.totalPressure(i_calculation) == 0.0) {
            VLEDoesNotConvergeAnymore = true;
            break;
        }
        else if (param.sw_Pbub == 1 && (param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) && calculationScheme.totalPressure(i_calculation) == 0.0) {
            // Try 10 smaller stepsizes
            int counter = 1;
            while (calculationScheme.totalPressure(i_calculation) == 0.0) {
                Eigen::VectorXd newxconcentration(2);
                int index = std::max(0, actualsize - 1);
                newxconcentration(IndexOfSupercriticalComponent) = std::min(x[index](IndexOfSupercriticalComponent) + 1 / std::pow(2, counter) * stepsize, 1.0);
                newxconcentration(IndexOfSubcriticalComponent) = std::max(x[index](IndexOfSubcriticalComponent) - 1 / std::pow(2, counter) * stepsize, 0.0);
                calculationScheme.originalConcentrations.row(1) = newxconcentration;
                for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
                    double value = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
                    calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
                }
                calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
                calculateBubblePoint(calculationScheme, i_calculation, 0.00001);
                counter += 1;
                if (counter > 10) {
                    break;
                }
            }
            if (calculationScheme.totalPressure(i_calculation) == 0.0 || counter > 10) {
                // No convergence (VLE) for the specified x, T anymore
                VLEDoesNotConvergeAnymore = true;
                break;
            }
        }
        else if (param.sw_Tbub == 1 && calculationScheme.targetTemperature(i_calculation) == 0.0) {
            VLEDoesNotConvergeAnymore = true;
            break;
        }
        if ((param.sw_Pbub == 1 && calculationScheme.totalPressure(i_calculation) != 0.0) || (param.sw_Tbub == 1 && calculationScheme.targetTemperature(i_calculation) != 0.0)) {

            Pressures.push_back(calculationScheme.totalPressure.row(1));
            Eigen::VectorXd tempVec(1);
            tempVec(0) = calculationScheme.targetTemperature(i_calculation);
            Temperatures.push_back(tempVec);
            x.push_back(calculationScheme.originalConcentrations.row(1));
            last_x = calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent);

            y.push_back(calculationScheme.originalConcentrations.row(0));
            calculationScheme.PartitionCoefficients.row(0) = (calculationScheme.originalConcentrations.row(0).array() / calculationScheme.originalConcentrations.row(1).array()).matrix();
            K.push_back(calculationScheme.PartitionCoefficients.row(0));

            if (int(x.size()) > 1 && (calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) == 0.0 || calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) == 1.0)) {
                break;
            }

            if ((param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) && std::abs(calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) - calculationScheme.originalConcentrations(0, IndexOfSupercriticalComponent)) < 0.00015 && calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) > 0.02) {

                // In case we search for an azeotropic point or a critical point using the tracing procedure, we stop, if the compositions of the coexisting phases are sufficiently close.

                calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
                calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;
                param.sw_isotherm = 1;
                std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 1);
                param.sw_isotherm = 0;
                std::vector<double> vapor_P = returnVector[5];
                std::vector<double> two_phase_region_P = returnVector[9];
                if (two_phase_region_P.size() > 1 && param.sw_criticalPoint == 1 && counter_for_x_almost_y < 5) {
                    counter_for_x_almost_y += 1;
                    continue;
                }
                else {
                    VLEDoesNotConvergeAnymore = true;
                    break;
                }
            }

            // Calculate new stepsize
            actualsize = int(x.size());
            double stepsize_1 = 0.5 * std::max(std::abs(x[actualsize - 1](IndexOfSupercriticalComponent) - y[actualsize - 1](IndexOfSupercriticalComponent)), 0.0001);
            double stepsize_2 = 0.005 / (std::abs(x[actualsize - 1](IndexOfSupercriticalComponent) - y[actualsize - 1](IndexOfSupercriticalComponent)) + 0.00001);
            stepsize = std::min(stepsize_1, stepsize_2);
            if (stepsize < 0.0001 || std::abs(calculationScheme.originalConcentrations(1, IndexOfSupercriticalComponent) - calculationScheme.originalConcentrations(0, IndexOfSupercriticalComponent)) < 0.01) {
                stepsize = 0.0001;
            }
            if (stepsize > 0.03) {
                stepsize = 0.03;
            }

            // Update x
            Eigen::VectorXd newxconcentration(2);
            newxconcentration(IndexOfSupercriticalComponent) = std::min(x[actualsize - 1](IndexOfSupercriticalComponent) + stepsize, 1.0);
            newxconcentration(IndexOfSubcriticalComponent) = std::max(x[actualsize - 1](IndexOfSubcriticalComponent) - stepsize, 0.0);
            calculationScheme.originalConcentrations.row(1) = newxconcentration;
            for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
                calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
            }
            calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();

            if (param.sw_Tbub == 1) {
                DeltaT = 0.0;
                if (PureTsat.size() > 0) {
                    DeltaT = 0.5 * (newxconcentration(0) - x[x.size() - 1](IndexOfSupercriticalComponent)) * (PureTsat[IndexOfSupercriticalComponent] - PureTsat[IndexOfSubcriticalComponent]);
                }
                int actualConcentrationIndex = calculationScheme.actualConcentrationIndices[i_calculation];
                int actualConcentrationIndex2 = calculationScheme.actualConcentrationIndices[i_calculation + 1];
                int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[i_calculation][0]];
                calculationScheme.temperatures[actualConcentrationIndex] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                calculationScheme.temperatures[actualConcentrationIndex2] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                calculationScheme.temperatures[actualReferenceStateConcentrationIndex] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                for (int g = 0; g < calculationScheme.TauConcentrationIndices.size(); g++) {
                    for (int h = 0; h < calculationScheme.TauConcentrationIndices[g].size(); h++) {
                        if (actualConcentrationIndex == calculationScheme.TauConcentrationIndices[g][h]) {
                            calculationScheme.TauTemperatures[g] = calculationScheme.targetTemperature(i_calculation) + DeltaT;
                        }
                    }
                }
            }
        }
    }
    std::vector<std::vector<Eigen::VectorXd>> returnVector;
    returnVector.push_back(x);
    returnVector.push_back(y);
    returnVector.push_back(K);
    returnVector.push_back(Pressures);
    returnVector.push_back(Temperatures);

    if (param.exp_param.at("EigenMatrixToFile") == 1) {
        Write1DArrayToFile<Eigen::VectorXd>(std::to_string(temperature) + "_" + std::to_string(IndexOfSubcriticalComponent) + "_x.txt", &x[0], x.size(), 1);
        Write1DArrayToFile<Eigen::VectorXd>(std::to_string(temperature) + "_" + std::to_string(IndexOfSubcriticalComponent) + "_y.txt", &y[0], y.size(), 1);
        Write1DArrayToFile<Eigen::VectorXd>(std::to_string(temperature) + "_" + std::to_string(IndexOfSubcriticalComponent) + "_K.txt", &K[0], K.size(), 1);
        Write1DArrayToFile<Eigen::VectorXd>(std::to_string(temperature) + "_" + std::to_string(IndexOfSubcriticalComponent) + "_P.txt", &Pressures[0], Pressures.size(), 1);
        Write1DArrayToFile<Eigen::VectorXd>(std::to_string(temperature) + "_" + std::to_string(IndexOfSubcriticalComponent) + "_T.txt", &Temperatures[0], Temperatures.size(), 1);
    }

    return returnVector;
}


void calculatePTFlash(calculation _calculation, int i_calculation, double ConvergenceCriteria, bool skipThetaCalculation, double thetaGuess = 0.0, bool UseGivenK = false) {

    int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

    // Associated polynom of RR function
    auto f_RRap = [&_calculation, i_calculation](double theta) {
        double functionValue = 0.0;
        for (int m = 0; m < _calculation.components.size() - 1; m++) {
            double val1 = double(_calculation.overallConcentrations(i_calculation, m)) * (_calculation.PartitionCoefficients(i_calculation, m) - 1.0) / (1.0 + theta * (_calculation.PartitionCoefficients(i_calculation, m) - 1.0));
            double val2 = 1.0;
            for (int n = 0; n < _calculation.components.size() - 1; n++) {
                val2 *= (1.0 + theta * (_calculation.PartitionCoefficients(i_calculation, n) - 1.0));
            }
            functionValue += val1 * val2;
        }
        return functionValue;
    };

    // RR function
    auto f_RR = [&_calculation, i_calculation](double theta) {
        double functionValue = 0.0;
        for (int m = 0; m < _calculation.components.size() - 1; m++) {
            functionValue += ((_calculation.PartitionCoefficients(i_calculation, m) - 1.0) * _calculation.overallConcentrations(i_calculation, m)) / (1.0 + (_calculation.PartitionCoefficients(i_calculation, m) - 1.0) * theta);
        }
        return functionValue;
    };

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
    int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
    double temperature = _calculation.temperatures[actualConcentrationIndex];
    double pressure = _calculation.targetPressure(i_calculation);

    // Stability analysis
    double zdivK = 0.0;
    double zdotK = 0.0;
    while (!(zdivK > 1 && zdotK > 1)) {
        for (int m = 0; m < _calculation.components.size() - 1; m++) {
            zdivK += double(_calculation.overallConcentrations(i_calculation, m)) / _calculation.PartitionCoefficients(i_calculation, m);
            zdotK += double(_calculation.overallConcentrations(i_calculation, m)) * _calculation.PartitionCoefficients(i_calculation, m);
        }
        if (zdivK < 1 || zdotK < 1) {
            if (zdotK < 1) {
                double multiplier = 1.05 * (1.0 / zdotK);
                for (int m = 0; m < _calculation.components.size() - 1; m++) {
                    _calculation.PartitionCoefficients(i_calculation, m) *= multiplier;
                }
            }
            if (zdivK < 1) {
                double multiplier = zdivK / 1.05;
                for (int m = 0; m < _calculation.components.size() - 1; m++) {
                    _calculation.PartitionCoefficients(i_calculation, m) *= multiplier;
                }
            }
            zdivK = 0.0;
            zdotK = 0.0;
            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                zdivK += double(_calculation.overallConcentrations(i_calculation, m)) / _calculation.PartitionCoefficients(i_calculation, m);
                zdotK += double(_calculation.overallConcentrations(i_calculation, m)) * _calculation.PartitionCoefficients(i_calculation, m);
            }
        }
    }

    // Earlier, we here had if (_calculation.calculationType == "VL" && (f_RRap(0.0) * f_RRap(1.0) > 0)) {...}, this means, that
    // the "complicated" stability analysis by setting x = z and y = z is only performed if the above scheme does not surely lead to
    // two phases. However, it has been shown that above scheme may lead to a solution for theta in the first iteration but then fails
    // in the second iteration, although two phases are expected. Therefore, it's better to in every case do the scheme below to get good
    // initial K values which more probably lead to convergence of RR if there are actually two phases.

    bool SuccessfullyFoundInitialK = false;

    if (UseGivenK == true) {
        SuccessfullyFoundInitialK = true;
    }

    try {
        if (_calculation.calculationType == "VL" && UseGivenK == false) {
            std::vector<Eigen::VectorXd> sampling_x = {};
            std::vector<Eigen::VectorXd> sampling_y = {};
            std::vector<Eigen::VectorXd> sampling_K = {};
            std::vector<Eigen::VectorXd> sampling_P = {};
            std::vector<Eigen::VectorXd> sampling_T = {};

            // Set x_i = z_i and perform Bubble P calculation
            param.sw_Pbub = 1;

            std::vector<double> pressureVector;
            pressureVector.push_back(double(pressure));
            std::vector<std::vector<double>> concentrationx1(1);
            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                concentrationx1[0].push_back(_calculation.overallConcentrations(i_calculation, m));
            }

            calculation calculationScheme = loadSubcalculation(_calculation.components, { double(temperature) }, concentrationx1, pressureVector);
            int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
            calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});

            Eigen::VectorXd TargetPressureAccordingToRaoult = (_calculation.overallConcentrations * _calculation.SaturationPressures.transpose()).diagonal();

            calculationScheme.PartitionCoefficients = Eigen::MatrixXd::Zero(2, 2);
            for (int i = 0; i < int(calculationScheme.components.size()) - 1; i++) {
                calculationScheme.PartitionCoefficients(0, i) = _calculation.SaturationPressures(i_calculation, i) / TargetPressureAccordingToRaoult(i_calculation);
                calculationScheme.originalConcentrations(0, i) = calculationScheme.originalConcentrations(1, i) * calculationScheme.PartitionCoefficients(0, i);
            }
            calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
            int numberOfOriginalComponents = calculationScheme.components.size() - 1;
            calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
            calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
            calculationScheme.calculationType = "VL";
            calculationScheme.originalNumberOfCalculations = 2;

            for (int i = 0; i < numberOfOriginalComponents; i++) {
                int index = calculationScheme.components[i]->index;
                calculationScheme.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                calculationScheme.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
            }

            calculateBubblePoint(calculationScheme, 0, 0.0001);

            double Pbub = calculationScheme.totalPressure(1);
            Eigen::VectorXd Kbub = (calculationScheme.originalConcentrations.row(0).array() / calculationScheme.originalConcentrations.row(1).array()).matrix();

            param.sw_Pbub = 0;

            if (Pbub != 0.0) {

                // Set y_i = z_i and perform Dew P calculation
                param.sw_Pdew = 1;
                calculationScheme.originalConcentrations.row(0) = _calculation.overallConcentrations.row(i_calculation);
                double dewPressureGuess = 1 / ((calculationScheme.originalConcentrations.row(0).array() / _calculation.SaturationPressures.row(i_calculation).array()).matrix().sum());
                calculationScheme.targetPressure(0) = dewPressureGuess;
                calculationScheme.targetPressure(1) = dewPressureGuess;
                calculationScheme.totalPressure(0) = dewPressureGuess;
                calculationScheme.totalPressure(1) = dewPressureGuess;
                calculationScheme.PartitionCoefficients.row(0) = _calculation.SaturationPressures.row(i_calculation) / dewPressureGuess;
                for (int m = 0; m < calculationScheme.components.size() - 1; m++) {
                    calculationScheme.originalConcentrations(1, m) = calculationScheme.originalConcentrations(0, m) / calculationScheme.PartitionCoefficients(0, m);
                }
                if (param.sw_Tdew == 1) {
                    calculationScheme.originalConcentrations.array().row(1) *= 1 / calculationScheme.originalConcentrations.array().row(1).sum();
                }
                calculateDewPoint(calculationScheme, 0, 0.0001);

                double Pdew = calculationScheme.totalPressure(1);
                Eigen::VectorXd Kdew = (calculationScheme.originalConcentrations.row(0).array() / calculationScheme.originalConcentrations.row(1).array()).matrix();
                param.sw_Pdew = 0;

                // Get and assign interpolated K
                if (Pbub != 0.0 && Pdew != 0.0) {
                    SuccessfullyFoundInitialK = true;
                    Eigen::VectorXd K = interpolateK(pressure, Pdew, Pbub, Kdew, Kbub);
                    _calculation.PartitionCoefficients.row(i_calculation) = K;
                }
                if ((Pbub == 0.0 || Pdew == 0.0) && param.sw_excess == 1) {
                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                        _calculation.originalConcentrations(i_calculation, m) = 0.0;
                        _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = 0.0;
                    }
                }
            }
        }
    }

    catch (...) {
        SuccessfullyFoundInitialK = false;
    }

    if (SuccessfullyFoundInitialK == true) {
        int numberOfConvergedCriteria = 0;
        int numberOfIterations = 0;
        int maxNumberOfIterations = 75;
        float previous_theta = 0;
        std::vector<double> previousDelta(int(_calculation.components.size()), 0.0);


        while (numberOfConvergedCriteria != int(_calculation.components.size())) {
            numberOfConvergedCriteria = 0;
            numberOfIterations += 1;

            // Theta from RRap
            if (skipThetaCalculation == false) {
                if (param.sw_excess == 1) {
                    thetaGuess = ITP(f_RRap, 0.0, 1.0, std::nan("1"), std::nan("1"), 1e-10, 0.000001, std::nan("1"), 2, 1, false);
                    if (std::isnan(thetaGuess)) {
                        if (f_RR(0) < 0) {
                            // single liquid phase
                            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = _calculation.overallConcentrations(i_calculation, m);
                                _calculation.originalConcentrations(i_calculation, m) = 0.0;
                            }
                        }
                        else if (f_RR(1) > 0) {
                            // single gas phase
                            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = 0.0;
                                _calculation.originalConcentrations(i_calculation, m) = _calculation.overallConcentrations(i_calculation, m);
                            }
                        }
                        break;
                    }
                }
                else {
                    thetaGuess = ITP(f_RRap, 0.0, 1.0, std::nan("1"), std::nan("1"), 1e-10, 0.000001);
                }
                //double RR_value = f_RR(thetaGuess);
            }
            if (numberOfIterations == maxNumberOfIterations) {
                break;
            }

            // x_i and y_i from mass balance and possible normalization
            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = _calculation.overallConcentrations(i_calculation, m) / (1.0 + thetaGuess * (_calculation.PartitionCoefficients(i_calculation, m) - 1.0));
                _calculation.originalConcentrations(i_calculation, m) = _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) * _calculation.PartitionCoefficients(i_calculation, m);
            }
            double sum = _calculation.originalConcentrations.array().row(i_calculation).sum();
            _calculation.originalConcentrations.array().row(i_calculation) *= 1 / sum;
            sum = _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
            _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1 / sum;

            // Calculate Isotherms, v and fugacity coefficients with EoS
            _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
            _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

            std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
            std::vector<double> vapor_v; // For LL this is actually L2
            std::vector<double> vapor_P;
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
            if (_calculation.calculationType == "VL") {
                if (vapor_v.size() == 0) {
                    // In this case there is only one supercritical phase in equilibrium with the liquid phase
                    vapor_v = returnVector[2];
                    vapor_P = returnVector[3];
                }
            }
            else if (_calculation.calculationType == "LL") {
                vapor_v = returnVector[2];
                vapor_P = returnVector[3];
            }
            returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];

            calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

            if (_calculation.totalPressure(i_calculation) == 0.0) {
                _calculation.originalConcentrations.row(i_calculation) = _calculation.overallConcentrations.row(i_calculation);
                _calculation.originalConcentrations.row(i_calculation + numberOfCalculations) = _calculation.overallConcentrations.row(i_calculation);
                break;
            }

            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, m));
                double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                double newPartitionCoefficient = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                double delta_new = newPartitionCoefficient - _calculation.PartitionCoefficients(i_calculation, m);
                double delta_old = previousDelta[m];
                bool signChange = (delta_new * delta_old < 0.0);
                if (std::abs(delta_new / newPartitionCoefficient) < ConvergenceCriteria) {
                    numberOfConvergedCriteria += 1;
                }
                if (signChange) {
                    newPartitionCoefficient = 0.5 * newPartitionCoefficient + 0.5 * _calculation.PartitionCoefficients(i_calculation, m);
                }
                _calculation.PartitionCoefficients(i_calculation, m) = newPartitionCoefficient;
                previousDelta[m] = delta_new;
            }
            if (std::abs((_calculation.targetPressure(i_calculation) - pressure) / _calculation.targetPressure(i_calculation)) < ConvergenceCriteria) {
                numberOfConvergedCriteria += 1;
            }
            skipThetaCalculation = false;
            previous_theta = thetaGuess;
        }
        // For some systems small jumps between the total Pressure / K's have been observed and the convergence criterion above is not fullfilled.
        // The algorithm then gets stuck in an infinite loop or converges only very slowly.
        // For this reason, the loop stops after 50 iterations and then another convergence criterion is checked and only if this one is also not met, _calculation.totalPressure(i_calculation) = 0.0 which means no convergence.
        if (numberOfIterations >= maxNumberOfIterations) {
            if (!(std::abs(1 - _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).dot(_calculation.PartitionCoefficients.row(i_calculation))) <= 0.001 || std::abs(thetaGuess - previous_theta) <= 0.01)) {
                _calculation.totalPressure(i_calculation) = 0.0;
            }
        }
    }
    else {
        _calculation.totalPressure(i_calculation) = 0.0;
    }
}


std::vector<Eigen::VectorXd> performCalculationsForExcessAndMixingProperties(calculation _calculation, int i_calculation, int MixtureIndex, std::string mode) {

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[MixtureIndex];
    double temperature = _calculation.temperatures[actualConcentrationIndex];
    double div_Aeff = 1 / param.Aeff;
    int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);
    int numberOfOriginalComponents = _calculation.components.size() - 1;
    Eigen::MatrixXd overallConcentrations = _calculation.originalConcentrations.cast<double>();

    Eigen::VectorXd LnPhiAttPure = Eigen::VectorXd::Zero(_calculation.components.size() - 1); // debugging
    Eigen::VectorXd LnPhiRepPure = Eigen::VectorXd::Zero(_calculation.components.size() - 1); // debugging

    Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = Eigen::VectorXd::Zero(_calculation.components.size() - 1);
    Eigen::VectorXd PureComponentMolarVolumes = Eigen::VectorXd::Zero(numberOfOriginalComponents); // in preparation for excess enthalpy calculation
    Eigen::VectorXd PureComponentDivRepulsivePressure = Eigen::VectorXd::Zero(numberOfOriginalComponents); // in preparation for excess enthalpy calculation
    Eigen::VectorXd PartialTempDerivativeRepulsivePressurePure;
    Eigen::VectorXd SecondPartialTempDerivativeRepulsivePressurePure;
    Eigen::VectorXd PartialTempDerivativeLnPhiAttractivePure = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd PartialTempDerivativeLnPhiRepulsivePure = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd PartialTempDerivativeLnZPure = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd SecondPartialTempDerivativeLnPhiRepulsivePure = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd SecondPartialTempDerivativeLnZPure = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd SecondPartialTempDerivativeLnPhiAttractivePure = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    double pressure = _calculation.targetPressure(MixtureIndex);

    auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
    int index = it - _calculation.uniqueTemperatures.begin();

    ///// Reference state / Pure component calculations /////
    for (int i_c = 0; i_c < _calculation.components.size() - 1; i_c++) {

        int subcalcindex = _calculation.IndicesForPureComponentCalculations(index, i_c);

        // Find actual stable state of component i in pure component state for specificied T, P (relevant for h^M calculation)
        int PureCompIndex;
        if (mode == "mixing") {
            if (temperature >= subcalculations[subcalcindex].criticalT(0) || subcalculations[subcalcindex].SaturationPressures(0) < pressure) {
                // This means the pure component is in liquid state
                PureCompIndex = 1;
            }
            else {
                // This means the pure component is in gaseous state
                PureCompIndex = 0;
            }
        }
        if (mode == "excess") {
            if (_calculation.calculationType == "LL") {
                PureCompIndex = 1;
            }
            else if (MixtureIndex == i_calculation) {
                PureCompIndex = 0;
            }
            else {
                PureCompIndex = 1;
            }
        }

        // Calculate LnGamma = LnPhi of Component i in Mixture - LnPhi of Component i  
        _calculation.lnGammaTotal(MixtureIndex, i_c) = _calculation.lnPhiTotal(MixtureIndex, i_c) - subcalculations[subcalcindex].lnPhiTotal(PureCompIndex, 0);
        // Calculate LnGamma = LnPhiMixtureMinusLnPhiPure in case of excess property and LnActivity = Z_i*LnPhiMixture - LnPhiPure in case of mixture property
        if (mode == "excess") {
            LnPhiMixtureMinusLnPhiPure(i_c) = _calculation.lnPhiTotal(MixtureIndex, i_c) - subcalculations[subcalcindex].lnPhiTotal(PureCompIndex, 0);
        }
        if (mode == "mixing") {
            LnPhiMixtureMinusLnPhiPure(i_c) = log(_calculation.originalConcentrations(MixtureIndex, i_c)) + _calculation.lnPhiTotal(MixtureIndex, i_c) - subcalculations[subcalcindex].lnPhiTotal(PureCompIndex, 0);
        }

        LnPhiAttPure(i_c) = subcalculations[subcalcindex].lnPhiAttractive(PureCompIndex, 0);
        LnPhiRepPure(i_c) = subcalculations[subcalcindex].lnPhiRepulsive(PureCompIndex, 0);

        PureComponentMolarVolumes(i_c) = subcalculations[subcalcindex].molarVolume(PureCompIndex, 0);
        PureComponentDivRepulsivePressure(i_c) = 1 / subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0);

        // Partial Temperature Derivative of Ln Phi Attractive
        int actualConcentrationIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[PureCompIndex];
        int actualReferenceStateIndexPure;
        const int numberOfSegments = int(subcalculations[subcalcindex].segments.size());
        double lnPhiAttractivePDETemperature = 0;
        double lnPhiAttractiveSecondPDETemperature = 0;

        for (int k = 0; k < numberOfSegments; k++) {
            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[PureCompIndex][0]];
            double DivGammaTimesDGamma_DT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualConcentrationIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualConcentrationIndexPure));
            double DivGammaIGTimesDGammaIG_DT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure));
            lnPhiAttractivePDETemperature += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][0] * div_Aeff - _calculation.componentCavityVolume(i_c) / _calculation.componentHoleVolume(i_c) * subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (DivGammaTimesDGamma_DT - DivGammaIGTimesDGammaIG_DT);
            lnPhiAttractiveSecondPDETemperature += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][0] * div_Aeff - _calculation.componentCavityVolume(i_c) / _calculation.componentHoleVolume(i_c) * subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (1 / subcalculations[subcalcindex].segmentGammas(k, actualConcentrationIndexPure) * subcalculations[subcalcindex].segmentGammasSecondPDETemperature(k, actualConcentrationIndexPure) - 1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasSecondPDETemperature(k, actualReferenceStateIndexPure) - pow(DivGammaTimesDGamma_DT, 2) + pow(DivGammaIGTimesDGammaIG_DT, 2));
        }
        PartialTempDerivativeLnPhiAttractivePure(i_c) = lnPhiAttractivePDETemperature;
        SecondPartialTempDerivativeLnPhiAttractivePure(i_c) = lnPhiAttractiveSecondPDETemperature;

        // Partial Temperature Derivative of Ln Phi Repulsive
        PartialTempDerivativeRepulsivePressurePure = Eigen::VectorXd::Zero(subcalculations[subcalcindex].originalNumberOfCalculations);
        SecondPartialTempDerivativeRepulsivePressurePure = Eigen::VectorXd::Zero(subcalculations[subcalcindex].originalNumberOfCalculations);
        for (int k = 0; k < numberOfSegments; k++) {
            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[0][1]];
            double DivGammaTimesDGammaDT = double(1 / subcalculations[subcalcindex].segmentGammas(k, subcalculations[subcalcindex].actualConcentrationIndices[0]) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, subcalculations[subcalcindex].actualConcentrationIndices[0]));
            double DivGammaIGTimesDGammaIGDT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure));
            PartialTempDerivativeRepulsivePressurePure(0) += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (log(subcalculations[subcalcindex].segmentGammas(k, subcalculations[subcalcindex].actualConcentrationIndices[0])) - log(subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure)) + temperature * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT)); // is actually not needed
            SecondPartialTempDerivativeRepulsivePressurePure(0) += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (2 * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT) + temperature * (1 / subcalculations[subcalcindex].segmentGammas(k, subcalculations[subcalcindex].actualConcentrationIndices[0]) * subcalculations[subcalcindex].segmentGammasSecondPDETemperature(k, subcalculations[subcalcindex].actualConcentrationIndices[0]) - 1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasSecondPDETemperature(k, actualReferenceStateIndexPure) - pow(DivGammaTimesDGammaDT, 2) + pow(DivGammaIGTimesDGammaIGDT, 2)));

            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[1][1]];
            DivGammaTimesDGammaDT = 1 / subcalculations[subcalcindex].segmentGammas(k, subcalculations[subcalcindex].actualConcentrationIndices[1]) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, subcalculations[subcalcindex].actualConcentrationIndices[1]);
            DivGammaIGTimesDGammaIGDT = 1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure);
            PartialTempDerivativeRepulsivePressurePure(1) += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (log(subcalculations[subcalcindex].segmentGammas(k, subcalculations[subcalcindex].actualConcentrationIndices[1])) - log(subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure)) + temperature * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT));
            SecondPartialTempDerivativeRepulsivePressurePure(1) += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (2 * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT) + temperature * (1 / subcalculations[subcalcindex].segmentGammas(k, subcalculations[subcalcindex].actualConcentrationIndices[1]) * subcalculations[subcalcindex].segmentGammasSecondPDETemperature(k, subcalculations[subcalcindex].actualConcentrationIndices[1]) - 1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasSecondPDETemperature(k, actualReferenceStateIndexPure) - pow(DivGammaTimesDGammaDT, 2) + pow(DivGammaIGTimesDGammaIGDT, 2)));
        }
        PartialTempDerivativeRepulsivePressurePure(0) *= R_GAS_CONSTANT / _calculation.componentHoleVolume(i_c);
        PartialTempDerivativeRepulsivePressurePure(1) *= R_GAS_CONSTANT / _calculation.componentHoleVolume(i_c);
        SecondPartialTempDerivativeRepulsivePressurePure(0) *= R_GAS_CONSTANT / _calculation.componentHoleVolume(i_c);
        SecondPartialTempDerivativeRepulsivePressurePure(1) *= R_GAS_CONSTANT / _calculation.componentHoleVolume(i_c);

        double num_1 = _calculation.componentCavityVolume(i_c) * (R_GAS_CONSTANT * temperature + subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * _calculation.componentCavityVolume(i_c)) + _calculation.componentCavityVolume(i_c) * R_GAS_CONSTANT * temperature;
        double dnum_1_dT = _calculation.componentCavityVolume(i_c) * (R_GAS_CONSTANT + PartialTempDerivativeRepulsivePressurePure(PureCompIndex) * _calculation.componentCavityVolume(i_c)) + R_GAS_CONSTANT * _calculation.componentCavityVolume(i_c);
        double den_1 = R_GAS_CONSTANT * temperature * (R_GAS_CONSTANT * temperature + subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * _calculation.componentCavityVolume(i_c));
        double dden_1_dT = R_GAS_CONSTANT * (R_GAS_CONSTANT * temperature + subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * _calculation.componentCavityVolume(i_c)) + R_GAS_CONSTANT * temperature * (R_GAS_CONSTANT + PartialTempDerivativeRepulsivePressurePure(PureCompIndex) * _calculation.componentCavityVolume(i_c));
        double num_2 = subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * R_GAS_CONSTANT * temperature * (_calculation.componentCavityVolume(i_c) + _calculation.componentCavityVolume(i_c)) + _calculation.componentCavityVolume(i_c) * subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * _calculation.componentCavityVolume(i_c);
        double dnum_2_dT = R_GAS_CONSTANT * (_calculation.componentCavityVolume(i_c) + _calculation.componentCavityVolume(i_c)) * (subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) + temperature * PartialTempDerivativeRepulsivePressurePure(PureCompIndex)) + 2 * _calculation.componentCavityVolume(i_c) * subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex) * _calculation.componentCavityVolume(i_c);
        double den_2 = R_GAS_CONSTANT * temperature * temperature * (R_GAS_CONSTANT * temperature + subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * _calculation.componentCavityVolume(i_c));
        double dden_2_dT = 2 * R_GAS_CONSTANT * temperature * (R_GAS_CONSTANT * temperature + subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * _calculation.componentCavityVolume(i_c)) + R_GAS_CONSTANT * pow(temperature, 2) * (R_GAS_CONSTANT + PartialTempDerivativeRepulsivePressurePure(PureCompIndex) * _calculation.componentCavityVolume(i_c));
        PartialTempDerivativeLnPhiRepulsivePure(i_c) = (num_1 / den_1) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex) - num_2 / den_2;
        SecondPartialTempDerivativeLnPhiRepulsivePure(i_c) = num_1 / den_1 * SecondPartialTempDerivativeRepulsivePressurePure(PureCompIndex) + (dnum_1_dT * den_1 - num_1 * dden_1_dT) / pow(den_1, 2) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex) - (dnum_2_dT * den_2 - num_2 * dden_2_dT) / pow(den_2, 2);

        // Partial Temperature Derivative of Ln Z
        //std::cout << "Pure Comp v: " << 1 / subcalculations[subcalcindex].molarVolume(PureCompIndex, 0) << std::endl;
        double DivvdvdTPure = 1 / subcalculations[subcalcindex].molarVolume(PureCompIndex, 0) * R_GAS_CONSTANT * (1 / subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) - temperature / (subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0) * subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0)) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex));
        //std::cout << "Pure Comp PDET v: " << DivvdvdTPure << std::endl;
        double d2vdT2Pure = -R_GAS_CONSTANT / pow(subcalculations[subcalcindex].repulsivePressure(PureCompIndex), 2) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex) - R_GAS_CONSTANT * temperature / pow(subcalculations[subcalcindex].repulsivePressure(PureCompIndex), 2) * SecondPartialTempDerivativeRepulsivePressurePure(PureCompIndex) - ((R_GAS_CONSTANT * pow(subcalculations[subcalcindex].repulsivePressure(PureCompIndex), 2) - 2 * R_GAS_CONSTANT * temperature * subcalculations[subcalcindex].repulsivePressure(PureCompIndex) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex)) / (pow(subcalculations[subcalcindex].repulsivePressure(PureCompIndex), 4))) * PartialTempDerivativeRepulsivePressurePure(PureCompIndex);
        //std::cout << "Pure Comp Second PDET v: " << d2vdT2Pure << std::endl;

        PartialTempDerivativeLnZPure(i_c) = DivvdvdTPure - 1 / temperature;
        SecondPartialTempDerivativeLnZPure(i_c) = 1 / subcalculations[subcalcindex].molarVolume(PureCompIndex, 0) * (d2vdT2Pure)-pow(DivvdvdTPure, 2) + 1 / pow(temperature, 2);
    }


    ///// Calculations of the properties of componet i in mixture /////
    Eigen::VectorXd PartialTempDerivativeRepulsivePressure = Eigen::VectorXd::Zero(_calculation.originalNumberOfCalculations);
    Eigen::VectorXd SecondPartialTempDerivativeRepulsivePressure = Eigen::VectorXd::Zero(_calculation.originalNumberOfCalculations);
    Eigen::VectorXd PartialTempDerivativeLnPhiAttractiveMixture = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd PartialTempDerivativeLnPhiResidualMixture = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd SecondPartialTempDerivativeLnPhiRepulsiveMixture = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd PartialTempDerivativeLnZMixture = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd SecondPartialTempDerivativeLnZMixture = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    Eigen::VectorXd SecondPartialTempDerivativeLnPhiAttractiveMixture = Eigen::VectorXd::Zero(numberOfOriginalComponents);
    const int numberOfSegments = int(_calculation.segments.size());

    // Later this for loop could be combined with the one in which the derivatives of ln phi attractive are calculated 
    double DPRepDTAtConstantPVx = 0.0;
    double DPRepDvAtConstantPTx = 0.0;
    int actualConcentrationStateIndex = _calculation.actualConcentrationIndices[MixtureIndex];
    int actualReferenceStateIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[MixtureIndex][numberOfOriginalComponents]];
    for (int k = 0; k < numberOfSegments; k++) {
        double DivGammaTimesDGammaDT = double(1 / _calculation.segmentGammas(k, actualConcentrationStateIndex) * _calculation.segmentGammasPDETemperature(k, actualConcentrationStateIndex));
        double DivGammaIGTimesDGammaIGDT = double(1 / _calculation.segmentGammas(k, actualConcentrationStateIndex) * _calculation.segmentGammasPDETemperature(k, actualReferenceStateIndex));
        DPRepDTAtConstantPVx += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (log(_calculation.segmentGammas(k, actualConcentrationStateIndex)) - log(_calculation.segmentGammas(k, actualReferenceStateIndex)) + temperature * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT));
        double DivGammaTimesDGammaDv = double(1 / _calculation.segmentGammas(k, actualConcentrationStateIndex) * _calculation.segmentGammasPDEVolume(k, actualConcentrationStateIndex));
        //double DivGammaIGTimesDGammaIGDv = double(1 / _calculation.segmentGammas(k, actualReferenceStateIndex) * _calculation.segmentGammasPDEVolume(k, actualReferenceStateIndex)); // is Zero
        DPRepDvAtConstantPTx += (_calculation.segments.SegmentTypeAreas[k][numberOfOriginalComponents] * div_Aeff) * (DivGammaTimesDGammaDv);
    }
    double RT = R_GAS_CONSTANT * temperature;
    double dvdTAtConstantPx = (R_GAS_CONSTANT / _calculation.repulsivePressure(MixtureIndex) - RT / pow(_calculation.repulsivePressure(MixtureIndex), 2) * R_GAS_CONSTANT / _calculation.mixtureHoleVolume(MixtureIndex) * DPRepDTAtConstantPVx) / (1 + RT / pow(_calculation.repulsivePressure(MixtureIndex), 2) * RT / _calculation.mixtureHoleVolume(MixtureIndex) * DPRepDvAtConstantPTx);
    dvdTAtConstantPx = 0;

    PartialTempDerivativeRepulsivePressure(MixtureIndex) = R_GAS_CONSTANT / _calculation.mixtureHoleVolume(MixtureIndex) * DPRepDTAtConstantPVx + RT / _calculation.mixtureHoleVolume(MixtureIndex) * DPRepDvAtConstantPTx * dvdTAtConstantPx;
    SecondPartialTempDerivativeRepulsivePressure(i_calculation) *= R_GAS_CONSTANT / _calculation.mixtureHoleVolume(i_calculation);

    double v = _calculation.molarVolume(MixtureIndex);
    double molarVolumeMinusMixtureCavityVolume = std::max(v - _calculation.mixtureCavityVolume(MixtureIndex), 0.0);
    double molesOfHoles = (1 / _calculation.mixtureHoleVolume(MixtureIndex)) * molarVolumeMinusMixtureCavityVolume;
    double newTotalMols = molesOfHoles + 1; // molesOfHoles + sum of prior n_total
    double holeConcentration = molesOfHoles / newTotalMols;
    Eigen::VectorXd bracket = (_calculation.componentCavityVolume / _calculation.mixtureHoleVolume(MixtureIndex)) + ((_calculation.componentHoleVolume / _calculation.mixtureHoleVolume(MixtureIndex)).array() - 1).matrix() * molesOfHoles;
    Eigen::VectorXd DivvTimesDvDT = Eigen::VectorXd::Zero(numberOfOriginalComponents);

    for (int j = 0; j < numberOfOriginalComponents; j++) {

        int actualConcentrationStateIndex = _calculation.actualConcentrationIndices[MixtureIndex];
        int actualReferenceStateIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[MixtureIndex][j]];

        // Partial Temperature Derivative of Ln Z
        DivvTimesDvDT(j) = 1 / v * dvdTAtConstantPx;
        double dv_dT = dvdTAtConstantPx;
        double d2v_dT2 = -R_GAS_CONSTANT / pow(_calculation.repulsivePressure(MixtureIndex), 2) * PartialTempDerivativeRepulsivePressure(MixtureIndex) - R_GAS_CONSTANT * temperature / pow(_calculation.repulsivePressure(MixtureIndex), 2) * SecondPartialTempDerivativeRepulsivePressure(MixtureIndex) - ((R_GAS_CONSTANT * pow(_calculation.repulsivePressure(MixtureIndex), 2) - 2 * R_GAS_CONSTANT * temperature * _calculation.repulsivePressure(MixtureIndex) * PartialTempDerivativeRepulsivePressure(MixtureIndex)) / (pow(_calculation.repulsivePressure(MixtureIndex), 4))) * PartialTempDerivativeRepulsivePressure(MixtureIndex);

        PartialTempDerivativeLnZMixture(j) = DivvTimesDvDT(j) - 1 / temperature;
        SecondPartialTempDerivativeLnZMixture(j) = 1 / v * (d2v_dT2)-pow(DivvTimesDvDT(j), 2) + 1 / pow(temperature, 2);

        //double lnGammaMolecule = 0; // just for checking 
        double lnPhiAttractivePDETemperature = 0;
        double lnPhiAttractiveSecondPDETemperature = 0;

        for (int k = 0; k < numberOfSegments; k++) {
            //lnGammaMolecule += _calculation.segments.SegmentTypeAreas[k][j] * div_Aeff * (log(_calculation.segmentGammas(k,i_calculation)) - log(_calculation.segmentGammas(k, _calculation.referenceStateCalculationIndices[i_calculation][j])));
            double DivGammaTimesDGamma_DT = double(1 / _calculation.segmentGammas(k, actualConcentrationStateIndex) * (_calculation.segmentGammasPDETemperature(k, actualConcentrationStateIndex) + _calculation.segmentGammasPDEVolume(k, actualConcentrationStateIndex) * dv_dT));
            double DivGammaIGTimesDGammaIG_DT = double(1 / _calculation.segmentGammas(k, actualReferenceStateIndex) * (_calculation.segmentGammasPDETemperature(k, actualReferenceStateIndex)));
            lnPhiAttractivePDETemperature += (_calculation.segments.SegmentTypeAreas[k][j] * div_Aeff - bracket(j) * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1] * div_Aeff) * (DivGammaTimesDGamma_DT - DivGammaIGTimesDGammaIG_DT) - (_calculation.componentHoleVolume(j) / _calculation.mixtureHoleVolume(MixtureIndex) - 1) * 1 / _calculation.mixtureHoleVolume(MixtureIndex) * dv_dT * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1] * div_Aeff * (log(_calculation.segmentGammas(k, actualConcentrationStateIndex)) - log(_calculation.segmentGammas(k, actualReferenceStateIndex)));
            lnPhiAttractiveSecondPDETemperature += (_calculation.segments.SegmentTypeAreas[k][j] * div_Aeff - bracket(j) * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1] * div_Aeff) * (1 / _calculation.segmentGammas(k, actualConcentrationStateIndex) * _calculation.segmentGammasSecondPDETemperature(k, actualConcentrationStateIndex) - 1 / _calculation.segmentGammas(k, actualReferenceStateIndex) * _calculation.segmentGammasSecondPDETemperature(k, actualReferenceStateIndex) - pow(DivGammaTimesDGamma_DT, 2) + pow(DivGammaIGTimesDGammaIG_DT, 2)) - 2 * (_calculation.componentHoleVolume(j) / _calculation.mixtureHoleVolume(MixtureIndex) - 1) * 1 / _calculation.mixtureHoleVolume(MixtureIndex) * v * DivvTimesDvDT(j) * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1] * div_Aeff * (DivGammaTimesDGamma_DT - DivGammaIGTimesDGammaIG_DT) - (_calculation.componentHoleVolume(j) / _calculation.mixtureHoleVolume(MixtureIndex) - 1) * 1 / _calculation.mixtureHoleVolume(MixtureIndex) * d2v_dT2 * _calculation.segments.SegmentTypeAreas[k][_calculation.components.size() - 1] * div_Aeff * (log(_calculation.segmentGammas(k, actualConcentrationStateIndex)) - log(_calculation.segmentGammas(k, actualReferenceStateIndex)));
        }
        PartialTempDerivativeLnPhiAttractiveMixture(j) = lnPhiAttractivePDETemperature;
        SecondPartialTempDerivativeLnPhiAttractiveMixture(j) = lnPhiAttractiveSecondPDETemperature;

        // Partial Temperature Derivative of Ln Phi Repulsive
        double num_1 = _calculation.componentCavityVolume(j) * (R_GAS_CONSTANT * temperature + _calculation.repulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex)) + _calculation.mixtureCavityVolume(MixtureIndex) * R_GAS_CONSTANT * temperature;
        double dnum_1_dT = _calculation.componentCavityVolume(j) * (R_GAS_CONSTANT + PartialTempDerivativeRepulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex)) + R_GAS_CONSTANT * _calculation.mixtureCavityVolume(MixtureIndex);
        double den_1 = R_GAS_CONSTANT * temperature * (R_GAS_CONSTANT * temperature + _calculation.repulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex));
        double dden_1_dT = R_GAS_CONSTANT * (R_GAS_CONSTANT * temperature + _calculation.repulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex)) + R_GAS_CONSTANT * temperature * (R_GAS_CONSTANT + PartialTempDerivativeRepulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex));
        double num_2 = _calculation.repulsivePressure(MixtureIndex) * R_GAS_CONSTANT * temperature * (_calculation.componentCavityVolume(j) + _calculation.mixtureCavityVolume(MixtureIndex)) + _calculation.componentCavityVolume(j) * _calculation.repulsivePressure(MixtureIndex) * _calculation.repulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex);
        double dnum_2_dT = R_GAS_CONSTANT * (_calculation.componentCavityVolume(j) + _calculation.mixtureCavityVolume(MixtureIndex)) * (_calculation.repulsivePressure(MixtureIndex) + temperature * PartialTempDerivativeRepulsivePressure(MixtureIndex)) + 2 * _calculation.componentCavityVolume(j) * _calculation.repulsivePressure(MixtureIndex) * PartialTempDerivativeRepulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex);
        double den_2 = R_GAS_CONSTANT * temperature * temperature * (R_GAS_CONSTANT * temperature + _calculation.repulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex));
        double dden_2_dT = 2 * R_GAS_CONSTANT * temperature * (R_GAS_CONSTANT * temperature + _calculation.repulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex)) + R_GAS_CONSTANT * pow(temperature, 2) * (R_GAS_CONSTANT + PartialTempDerivativeRepulsivePressure(MixtureIndex) * _calculation.mixtureCavityVolume(MixtureIndex));
        PartialTempDerivativeLnPhiResidualMixture(j) = num_1 / den_1 * PartialTempDerivativeRepulsivePressure(MixtureIndex) - num_2 / den_2;
        SecondPartialTempDerivativeLnPhiRepulsiveMixture(j) = num_1 / den_1 * SecondPartialTempDerivativeRepulsivePressure(MixtureIndex) + (dnum_1_dT * den_1 - num_1 * dden_1_dT) / pow(den_1, 2) * PartialTempDerivativeRepulsivePressure(MixtureIndex) - (dnum_2_dT * den_2 - num_2 * dden_2_dT) / pow(den_2, 2);
    }

    double divmolarVolumeMinusMixtureCavityVolume = 1 / molarVolumeMinusMixtureCavityVolume;

    Eigen::VectorXd PureComponentMolarVolumeMinusPureComponentCavityVolume = PureComponentMolarVolumes - _calculation.componentCavityVolume;

    std::vector<Eigen::VectorXd> returnVector;
    returnVector.reserve(15);

    returnVector.push_back(LnPhiMixtureMinusLnPhiPure);
    returnVector.push_back(PartialTempDerivativeLnPhiResidualMixture);
    returnVector.push_back(PartialTempDerivativeLnPhiAttractiveMixture);
    returnVector.push_back(PartialTempDerivativeLnZMixture);
    returnVector.push_back(PartialTempDerivativeLnPhiRepulsivePure);
    returnVector.push_back(PartialTempDerivativeLnPhiAttractivePure);
    returnVector.push_back(PartialTempDerivativeLnZPure);
    returnVector.push_back(SecondPartialTempDerivativeLnPhiRepulsiveMixture);
    returnVector.push_back(SecondPartialTempDerivativeLnPhiAttractiveMixture);
    returnVector.push_back(SecondPartialTempDerivativeLnZMixture);
    returnVector.push_back(SecondPartialTempDerivativeLnPhiRepulsivePure);
    returnVector.push_back(SecondPartialTempDerivativeLnPhiAttractivePure);
    returnVector.push_back(SecondPartialTempDerivativeLnZPure);
    returnVector.push_back(PureComponentMolarVolumeMinusPureComponentCavityVolume);

    if (param.exp_param.at("EigenMatrixToFile") == 1) {
        std::string tempStr = std::to_string(temperature);
        tempStr.erase(
            std::remove(tempStr.begin(), tempStr.end(), '.'),
            tempStr.end()
        );
        WriteEigenVectorToFile(tempStr + "_" + "LnPhiAttPure_" + std::to_string(_calculation.number), LnPhiAttPure, "T = " + std::to_string(temperature) + "_" + std::to_string(i_calculation + numberOfCalculations));
        WriteEigenVectorToFile(tempStr + "_" + "LnPhiRepPure_" + std::to_string(_calculation.number), LnPhiRepPure, "T = " + std::to_string(temperature) + "_" + std::to_string(i_calculation + numberOfCalculations));
        WriteEigenVectorToFile(tempStr + "_" + "RepulsivePressurePDET_" + std::to_string(_calculation.number), PartialTempDerivativeRepulsivePressure, "T = " + std::to_string(temperature) + "_" + std::to_string(i_calculation + numberOfCalculations));
        WriteEigenVectorToFile(tempStr + "_" + "RepulsivePressureSecondPDET_" + std::to_string(_calculation.number), SecondPartialTempDerivativeRepulsivePressure, "T = " + std::to_string(temperature) + "_" + std::to_string(i_calculation + numberOfCalculations));
        WriteEigenVectorToFile(tempStr + "_" + "RepulsivePressurePDET_Pure_" + std::to_string(_calculation.number), PartialTempDerivativeRepulsivePressurePure, "T = " + std::to_string(temperature) + "_" + std::to_string(i_calculation + numberOfCalculations));
        WriteEigenVectorToFile(tempStr + "_" + "RepulsivePressureSecondPDET_Pure_" + std::to_string(_calculation.number), SecondPartialTempDerivativeRepulsivePressurePure, "T = " + std::to_string(temperature) + "_" + std::to_string(i_calculation + numberOfCalculations));
    }

    return returnVector;
}


std::vector<Eigen::VectorXd> calculateMolarResidualEnthalpyForPureComponents(calculation _calculation, int i_calculation, int MixtureIndex, std::string mode) {

    int actualConcentrationIndex = _calculation.actualConcentrationIndices[MixtureIndex];
    double temperature = _calculation.temperatures[actualConcentrationIndex];
    double pressure = _calculation.targetPressure(MixtureIndex);
    double div_Aeff = 1 / param.Aeff;
    int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);
    int numberOfOriginalComponents = _calculation.components.size() - 1;
    Eigen::MatrixXd overallConcentrations = _calculation.originalConcentrations.cast<double>();

    Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = Eigen::VectorXd::Zero(_calculation.components.size() - 1);
    Eigen::VectorXd MolarResidualEnthalpiesOfPureComponents = Eigen::VectorXd::Zero(_calculation.components.size() - 1);
    Eigen::VectorXd MolarCpresOfPureComponents = Eigen::VectorXd::Zero(_calculation.components.size() - 1);


    auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
    int index = it - _calculation.uniqueTemperatures.begin();

    ///// Reference state / Pure component calculations /////
    for (int i_c = 0; i_c < _calculation.components.size() - 1; i_c++) {

        int subcalcindex = _calculation.IndicesForPureComponentCalculations(index, i_c);

        // Find actual stable state of component i in pure component state for specificied T, P (relevant for h^M calculation)
        int PureCompIndex;
        if (mode == "mixing") {
            if (temperature >= subcalculations[subcalcindex].criticalT(0) || subcalculations[subcalcindex].SaturationPressures(0) < pressure) {
                // This means the pure component is in supercritical or liquid state
                PureCompIndex = 1;
            }
            else {
                // This means the pure component is in gaseous state
                PureCompIndex = 0;
            }
        }
        if (mode == "excess") {
            if (_calculation.calculationType == "LL") {
                PureCompIndex = 1;
            }
            else if (MixtureIndex != i_calculation || temperature >= subcalculations[subcalcindex].criticalT(0)) {
                PureCompIndex = 1;
            }
            else {
                PureCompIndex = 0;
            }
        }
        // Calculate LnGamma = LnPhiMixtureMinusLnPhiPure in case of excess property and LnActivity = Z_i*LnPhiMixture - LnPhiPure in case of mixture property
        LnPhiMixtureMinusLnPhiPure(i_c) = _calculation.lnPhiTotal(MixtureIndex, i_c) - subcalculations[subcalcindex].lnPhiTotal(PureCompIndex, 0);
        if (mode == "mixing" && _calculation.originalConcentrations(MixtureIndex, i_c) != 0.0) {
            double log_xi = log(_calculation.originalConcentrations(MixtureIndex, i_c));
            LnPhiMixtureMinusLnPhiPure(i_c) += log_xi;
        }

        double PureComponentMolarVolume = subcalculations[subcalcindex].molarVolume(PureCompIndex, 0);
        double PureComponentRepulsivePressure = subcalculations[subcalcindex].repulsivePressure(PureCompIndex, 0);
        double PureComponentTotalPressure = subcalculations[subcalcindex].totalPressure(PureCompIndex, 0); // should equal target Pressure (of mixture as well as pure component)

        // Calculate h^res(T,v_pure) for current pure component i_c
        int actualConcentrationIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[PureCompIndex];
        int actualReferenceStateIndexPure;
        int numberOfSegments = int(subcalculations[subcalcindex].segments.size());

        double sum_1 = 0;

        for (int k = 0; k < numberOfSegments; k++) {
            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[PureCompIndex][0]];
            double DivGammaTimesDGamma_DT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualConcentrationIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualConcentrationIndexPure));
            double DivGammaIGTimesDGammaIG_DT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure));
            sum_1 += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][0] * div_Aeff - _calculation.componentCavityVolume(i_c) / _calculation.componentHoleVolume(i_c) * subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (DivGammaTimesDGamma_DT - DivGammaIGTimesDGammaIG_DT);
        }

        double sum_3 = 0;

        for (int k = 0; k < numberOfSegments; k++) {
            actualReferenceStateIndexPure = subcalculations[subcalcindex].actualConcentrationIndices[subcalculations[subcalcindex].referenceStateCalculationIndices[PureCompIndex][1]];
            double DivGammaTimesDGammaDT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualConcentrationIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualConcentrationIndexPure));
            double DivGammaIGTimesDGammaIGDT = double(1 / subcalculations[subcalcindex].segmentGammas(k, actualReferenceStateIndexPure) * subcalculations[subcalcindex].segmentGammasPDETemperature(k, actualReferenceStateIndexPure));
            sum_3 += (subcalculations[subcalcindex].segments.SegmentTypeAreas[k][1] * div_Aeff) * (DivGammaTimesDGammaDT - DivGammaIGTimesDGammaIGDT);
        }

        double T2 = temperature * temperature;
        double MinusRT2 = -R_GAS_CONSTANT * T2;
        MolarResidualEnthalpiesOfPureComponents(i_c) = MinusRT2 * sum_1 + MinusRT2 * PureComponentMolarVolume / (_calculation.componentHoleVolume(i_c)) * sum_3 + PureComponentTotalPressure * PureComponentMolarVolume - R_GAS_CONSTANT * temperature;

        // Calculate cp^res(T,v_pure) for current pure component i_c
        MolarCpresOfPureComponents(i_c) = subcalculations[subcalcindex].residualHeatCapacityAtConstantPressure(PureCompIndex);
    }

    std::vector<Eigen::VectorXd> returnVector;
    returnVector.reserve(2);

    returnVector.push_back(LnPhiMixtureMinusLnPhiPure);
    returnVector.push_back(MolarResidualEnthalpiesOfPureComponents);
    returnVector.push_back(MolarCpresOfPureComponents);

    return returnVector;
}


void calculate(std::vector<int>& calculationIndices) {

    // this is needed to catch exceptions in the OPENMP threads and rethrow them after the parallel section ends
    threadException e;

#ifndef OPT_NO_DYNAMIC
    // dynamic schedule, largest calculations first
    std::vector<int> scheduledCalculationIndices(calculationIndices);
    std::stable_sort(scheduledCalculationIndices.begin(), scheduledCalculationIndices.end(), [](int a, int b) {
        const double segmentsA = double(calculations[a].segments.size());
        const double segmentsB = double(calculations[b].segments.size());
        return double(calculations[a].originalNumberOfCalculations) * segmentsA * segmentsA
             > double(calculations[b].originalNumberOfCalculations) * segmentsB * segmentsB;
    });
#else
    std::vector<int>& scheduledCalculationIndices = calculationIndices;
#endif

#if defined(_OPENMP)
    omp_set_nested(1);
#ifndef OPT_NO_DYNAMIC
#pragma omp parallel for schedule(dynamic, 1)
#else
#pragma omp parallel for
#endif
#endif
    for (int i = 0; i < scheduledCalculationIndices.size(); i++) {

        const int calculationIndex = scheduledCalculationIndices[i];

        e.run([=] {
            //#if defined(_OPENMP)
            //int outer_tid = omp_get_thread_num();
            //std::cout << "Outer thread " << outer_tid << "\n";
            //#endif


#ifdef MEASURE_TIME
            std::chrono::high_resolution_clock::time_point rescaleSegments_last = std::chrono::high_resolution_clock::now();
#endif
            if (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) {

                calculations[calculationIndex].segments.clear();

                for (int j = 0; j < calculations[calculationIndex].components.size(); j++) {

                    std::shared_ptr<molecule> thisMolecule = calculations[calculationIndex].components[j];

                    for (int k = 0; k < thisMolecule->segments.size(); k++) {
                        calculations[calculationIndex].segments.add((unsigned short)j, thisMolecule->segments.SegmentTypeGroup[k],
                            thisMolecule->segments.SegmentTypeSigma[k],
                            thisMolecule->segments.SegmentTypeSigmaCorr[k],
                            thisMolecule->segments.SegmentTypeHBtype[k],
                            thisMolecule->segments.SegmentTypeAtomicNumber[k],
                            thisMolecule->segments.SegmentTypeAreas[k][0],
                            thisMolecule->segments.SegmentTypeMoleculeIndex[k],
                            thisMolecule->segments.SegmentTypeAtomicPolariz[k]);
                    }
                }
                calculations[calculationIndex].segments.sort();
                rescaleSegments(param, calculations[calculationIndex]);
                calculations[calculationIndex].segmentGammas = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()), 1.0);
                calculations[calculationIndex].segmentGammasPDETemperature = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()), 1.0);
                calculations[calculationIndex].segmentGammasPDEVolume = MatrixCalcType::Constant(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()), 1.0);
                calculations[calculationIndex].segmentConcentrations = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()));
                calculations[calculationIndex].PartialsegmentConcentrationsPartialv = MatrixCalcType::Zero(RoundUpToNextMultipleOfEight(int(calculations[calculationIndex].segments.size())), int(calculations[calculationIndex].concentrations.size()));
            }


            if (param.sw_alwaysCalculateSizeRelatedParameters == 1 || (param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3) \
                || (param.sw_alwaysReloadSigmaProfiles == 1 && n_ex > 3) || param.sw_reloadConcentrations == 1 || param.sw_reloadReferenceConcentrations == 1) {

                rescaleSegments(param, calculations[calculationIndex]);
                calculateSegmentConcentrations(calculations[calculationIndex]);
            }

#ifdef PRINT_DEBUG_INFO
            display("number of components: " + std::to_string(calculations[calculationIndex].components.size()) + "\n");
            display("number of segments: " + std::to_string(calculations[calculationIndex].segments.size()) + "\n");
            display("number of concentrations: " + std::to_string(calculations[calculationIndex].concentrations.size()) + "\n");
#endif

#ifdef MEASURE_TIME
            rescaleSegments_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - rescaleSegments_last).count());
            std::chrono::high_resolution_clock::time_point calculateCombinatorial_last = std::chrono::high_resolution_clock::now();
#endif

            if (param.sw_phi == 1) {
                // Calculations with EoS openCOSMO-RS-Phi

                calculation _calculation = calculations[calculationIndex];

                int numberOfOriginalComponents = _calculation.components.size() - 1;

                _calculation.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                _calculation.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                _calculation.componentCharges = Eigen::VectorXd::Zero(numberOfOriginalComponents); // added for ES
                _calculation.componentBornDiameter = Eigen::VectorXd::Zero(numberOfOriginalComponents); // unit m

                if (param.sw_electrolyte == 1) {
                    // Add Born Diameters to Eigen::Vector _calculation.componentBornDiameter and component charges to _calculation.componentCharges
                    calculateBornAndSaltDiameter(_calculation);
                }

                for (int i = 0; i < numberOfOriginalComponents; i++) {
                    int index = _calculation.components[i]->index;
                    _calculation.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                    _calculation.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
                }

                if (numberOfOriginalComponents == 1) {
                    // Pure Component Calculations

                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                    if (_calculation.originalNumberOfCalculations % 2 != 0)
                        throw std::runtime_error("Both phases are expected to be given as concentrations.");

                    if (_calculation.calculationType != "VL")
                        throw std::runtime_error("This code only works for VL equilibria.");

                    int numberOfStatePoints = int(_calculation.originalNumberOfCalculations / 2);

                    //#if defined(_OPENMP)
                    //#pragma omp parallel for
                    //#endif
                    _calculation.invalidateTaus();//recalcualte Taus
                    double smallestTemperatureWhichIsSupercritic = std::numeric_limits<double>::max();

#if defined(_OPENMP)
#pragma omp parallel for
#endif
                    for (int i_calculation = 0; i_calculation < numberOfStatePoints; i_calculation++) {

                        //#if defined(_OPENMP)
                        //int inner_tid = omp_get_thread_num();
                        //std::cout << "  Inner thread " << inner_tid << "\n";
                        //#endif

                        if (param.sw_criticalPoint == 1) {
                            findCriticalPointForSpecifiedx(_calculation, i_calculation);
                        }
                        else {
                            // In the else case it is simply assumed that a pure component VLE calculation is to be carried out
                            double pressureGuess = _calculation.targetPressure(i_calculation);

                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];

                            std::vector<double> v;
                            std::vector<double> P;
                            std::vector<double> liquid_v;
                            std::vector<double> liquid_P;
                            std::vector<double> vapor_v;
                            std::vector<double> vapor_P;
                            std::vector<double> P_repulsive;
                            std::vector<double> P_attractive;

                            if (temperature < smallestTemperatureWhichIsSupercritic) {
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);

                                v = returnVector[0];
                                P = returnVector[1];
                                liquid_v = returnVector[2];
                                liquid_P = returnVector[3];
                                vapor_v = returnVector[4];
                                vapor_P = returnVector[5];
                                P_repulsive = returnVector[6];
                                P_attractive = returnVector[7];
                                bool isSupercriticalIsotherm = vapor_v.size() == 0;

                                if (isSupercriticalIsotherm && temperature < smallestTemperatureWhichIsSupercritic)
                                    smallestTemperatureWhichIsSupercritic = temperature;
                            }

                            calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, true, true);
                        }
                    }
                }
                else {
                    // Mixture Calculations

                    _calculation.overallConcentrations = _calculation.originalConcentrations.cast<double>();


                    ///// INITIALIZATION FOR VLE CALCULATIONS /////
                    _calculation.PartitionCoefficients = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), numberOfOriginalComponents);

                    if (_calculation.calculationType == "VL" || param.sw_excess == 1) {

                        std::deque<calculation> IntermediateSubcalculations;   // must match subcalculations

                        if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
                            _calculation.SaturationTemperatures = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), numberOfOriginalComponents);

                            // Find pure component saturation temperature previously calculated for specified P
                            for (int i_pressure = 0; i_pressure < int(_calculation.originalConcentrations.rows()); i_pressure++) {
                                double pressureToFind = _calculation.targetPressure(i_pressure);
                                int index = -1;
                                for (int k = 0; k < int(_calculation.uniquePressures.size()); k++) {
                                    if (_calculation.uniquePressures[k] == pressureToFind) {
                                        index = k;
                                        break;
                                    }
                                }
                                for (int l = 0; l < _calculation.components.size() - 1; l++) {
                                    int subcalcindex = _calculation.IndicesForPureComponentCalculations(index, l);
                                    if (subcalcindex < 0) {
                                        // non-volatile ion: no boiling temperature
                                        _calculation.SaturationTemperatures(i_pressure, l) = 0.0;
                                        continue;
                                    }
                                    _calculation.SaturationTemperatures(i_pressure, l) = subcalculations[subcalcindex].temperatures[0];
                                }
                            }

                            // Initial estimates for boiling temperature
                            Eigen::VectorXd target_temperatures = (_calculation.overallConcentrations * _calculation.SaturationTemperatures.transpose()).diagonal();

                            // initial estimate over the volatile components only
                            for (int i_row = 0; i_row < int(target_temperatures.size()); i_row++) {
                                double neutralMoleFractionSum = 0.0;
                                for (int l = 0; l < int(_calculation.components.size()) - 1; l++) {
                                    if (_calculation.components[l]->moleculeCharge == 0) {
                                        neutralMoleFractionSum += double(_calculation.overallConcentrations(i_row, l));
                                    }
                                }
                                if (neutralMoleFractionSum > 0.0) {
                                    target_temperatures(i_row) /= neutralMoleFractionSum;
                                }
                            }

                            // Save initial boiling temperature estimate in temperatures and TauTemperatures
                            for (int i_pressure = 0; i_pressure < int(_calculation.originalConcentrations.rows()); i_pressure++) {
                                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_pressure];
                                int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_pressure][0]];
                                _calculation.temperatures[actualConcentrationIndex] = target_temperatures(i_pressure);
                                _calculation.temperatures[actualReferenceStateConcentrationIndex] = target_temperatures(i_pressure);
                                for (int g = 0; g < _calculation.TauConcentrationIndices.size(); g++) {
                                    for (int h = 0; h < _calculation.TauConcentrationIndices[g].size(); h++) {
                                        if (actualConcentrationIndex == _calculation.TauConcentrationIndices[g][h]) {
                                            _calculation.TauTemperatures[g] = target_temperatures(i_pressure);
                                        }
                                    }
                                }
                            }

                            // Now where we have first guesses for the temperatures, we calculate the saturation pressures at these temperatures
                            // swap instead of copy: the subcalculations hold Eigen::Maps into their own buffers
                            std::swap(IntermediateSubcalculations, subcalculations);
                            for (int k = 0; k < target_temperatures.size(); k++) {
                                _calculation.uniqueTemperatures.push_back(target_temperatures(k));
                            }
                            std::sort(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end());
                            _calculation.uniqueTemperatures.erase(std::unique(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end()), _calculation.uniqueTemperatures.end());
                            _calculation.IndicesForPureComponentCalculations = Eigen::MatrixXi(int(_calculation.uniqueTemperatures.size()), _calculation.components.size() - 1);
                            for (int j = 0; j < _calculation.components.size() - 1; j++) {
                                // pure ion: no pure-component state (-1)
                                if (_calculation.components[j]->moleculeCharge != 0) {
                                    for (int k = 0; k < int(_calculation.uniqueTemperatures.size()); k++) {
                                        _calculation.IndicesForPureComponentCalculations(k, j) = -1;
                                    }
                                    continue;
                                }
                                std::vector<std::shared_ptr<molecule>> actualMolecules;
                                actualMolecules.push_back(_calculation.components[j]);
                                actualMolecules.push_back(_calculation.components[int(_calculation.components.size() - 1)]);
                                int subcalculationIndex = -2;
                                for (int k = 0; k < _calculation.uniqueTemperatures.size(); k++) {
                                    std::vector<double> temperatureVector;
                                    temperatureVector.push_back(_calculation.uniqueTemperatures[k]);
                                    subcalculationIndex = calculations[i].addOrFindArrayIndexInSubcalculations(subcalculations, actualMolecules, _calculation.uniqueTemperatures[k]);
                                    if (subcalculationIndex == -1) {
                                        subcalculations.push_back(loadSubcalculation(actualMolecules, temperatureVector, { {1.0, 0.0} }));
                                        subcalculationIndex = int(subcalculations.size()) - 1;
                                    }
                                    _calculation.IndicesForPureComponentCalculations(k, j) = subcalculationIndex;
                                }
                            }
                            param.sw_Pbub = 1;
                            const size_t numSubCalcs = subcalculations.size();
                            std::vector<int> subcalculationIndices(numSubCalcs);
                            for (int i = 0; i < numSubCalcs; i++) {
                                subcalculationIndices[i] = i;
                            }
                            calculateSubcalculations(subcalculationIndices);
                            param.sw_Pbub = 0;
                        }

                        if (param.sw_Pdew == 1 || param.sw_Pbub == 1 || param.sw_PT_Flash == 1 || param.sw_Tdew == 1 || param.sw_Tbub == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1 || param.sw_excess == 1) {
                            _calculation.SaturationPressures = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), numberOfOriginalComponents);

                            // Find pure component saturation pressure previously calculated for specified T
                            for (int i_temperature = 0; i_temperature < int(_calculation.originalConcentrations.rows()); i_temperature++) {
                                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_temperature];
                                double temperatureToFind = _calculation.temperatures[actualConcentrationIndex];
                                auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperatureToFind);
                                int index = it - _calculation.uniqueTemperatures.begin();
                                for (int l = 0; l < _calculation.components.size() - 1; l++) {
                                    int subcalcindex = _calculation.IndicesForPureComponentCalculations(index, l);
                                    if (subcalcindex < 0) {
                                        // non-volatile ion: Psat = 0
                                        _calculation.SaturationPressures(i_temperature, l) = 0.0;
                                        continue;
                                    }
                                    _calculation.SaturationPressures(i_temperature, l) = subcalculations[subcalcindex].SaturationPressures(0, 0); // look
                                }
                            }

                            // In case of a Bubble P, Dew P, Critical Point or Azeotropic Point Calculation, total pressure P is unknown and therefore estimated using Raoult's law
                            if (param.sw_Pbub == 1 || param.sw_Pdew == 1 || param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) {
                                Eigen::VectorXd TargetPressureAccordingToRaoult = (_calculation.overallConcentrations * _calculation.SaturationPressures.transpose()).diagonal();
                                _calculation.targetPressure = TargetPressureAccordingToRaoult;
                            }

                            // Calculation of initial K values
                            for (int k = 0; k < _calculation.originalConcentrations.rows(); k++) {
                                for (int l = 0; l < _calculation.components.size() - 1; l++) {
                                    _calculation.PartitionCoefficients(k, l) = _calculation.SaturationPressures(k, l) / _calculation.targetPressure(k);
                                }
                            }

                            if (param.sw_Tdew == 1 || param.sw_Tbub == 1) {
                                std::swap(subcalculations, IntermediateSubcalculations);
                            }
                        }
                    }


                    ///// CRITICAL POINT CALCULATION FOR SPECIFIED X /////
                    if (param.sw_criticalPoint_specifiedx == 1) {
                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {
                            findCriticalPointForSpecifiedx(_calculation, i_calculation);
                        }
                    }


                    ///// CRITICAL OR AZEOTROPIC POINT CALCULATION FOR SPECIFIED T /////
                    if (param.sw_criticalPoint == 1 || param.sw_azeotropicPoint == 1) {
                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

                            std::vector<Eigen::VectorXd> sampling_x = {};
                            std::vector<Eigen::VectorXd> sampling_y = {};
                            std::vector<Eigen::VectorXd> sampling_K = {};
                            std::vector<Eigen::VectorXd> sampling_P = {};
                            std::vector<Eigen::VectorXd> sampling_T = {};

                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];

                            if (int(_calculation.components.size() - 1) > 2) {
                                throw std::runtime_error("Critical/Azeotropic Point calculation currently only works for binary mixtures.");
                            }
                            else {
                                auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
                                int index = it - _calculation.uniqueTemperatures.begin();
                                int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                int IndexOfSupercriticalComponent;
                                int IndexOfSubcriticalComponent;
                                int SubcalculationsIndexOfSupercriticalComponent;
                                int SubcalculationsIndexOfSubcriticalComponent;
                                if (((subcalculations[subcalcindex_comp1].criticalT(0) > temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) < temperature))) {
                                    IndexOfSupercriticalComponent = 1;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                    IndexOfSubcriticalComponent = 0;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                }
                                else if ((subcalculations[subcalcindex_comp1].criticalT(0) < temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) > temperature) || ((subcalculations[subcalcindex_comp1].criticalT(0) > temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) > temperature))) {
                                    IndexOfSupercriticalComponent = 0;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp1;
                                    IndexOfSubcriticalComponent = 1;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp2;
                                }
                                else {
                                    throw std::runtime_error("Both components are in supercritical state for given T, P. For this situation, use different routine.");
                                }

                                double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
                                Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
                                startx(IndexOfSupercriticalComponent) = 0.0;
                                startx(IndexOfSubcriticalComponent) = 1.0;
                                Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
                                startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
                                startK(IndexOfSubcriticalComponent) = 1.0;

                                std::cout << "Start Sampling For Critical Point Calculation" << std::endl;
                                param.sw_Pbub = 1;
                                std::vector<std::vector<Eigen::VectorXd>> samplingResults;
                                samplingResults = samplingForVLE(_calculation.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, 1.0);
                                param.sw_Pbub = 0;
                                sampling_x = samplingResults[0];
                                sampling_y = samplingResults[1];
                                sampling_K = samplingResults[2];
                                sampling_P = samplingResults[3];

                                if (sampling_x[sampling_x.size() - 1](0) == 1 || sampling_x[sampling_x.size() - 1](0) == 0 || std::abs(sampling_y[sampling_y.size() - 1](0) - sampling_x[sampling_x.size() - 1](0)) > 0.001) {
                                    throw std::runtime_error("No critical or azeotropic Point found");
                                }
                                _calculation.criticalT(i_calculation, 0) = temperature;
                                _calculation.criticalP.row(i_calculation) = sampling_P[sampling_P.size() - 1];
                                _calculation.criticalx.row(i_calculation) = (sampling_y[sampling_y.size() - 1] + sampling_x[sampling_x.size() - 1]) / 2;
                            }
                        }

                    }


                    ///// BUBBLE POINT CALCULATION (BUBBLE T OR BUBBLE P) /////
                    if (param.sw_Pbub == 1 || param.sw_Tbub == 1) {
                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        std::vector<Eigen::VectorXd> sampling_x = {};
                        std::vector<Eigen::VectorXd> sampling_y = {};
                        std::vector<Eigen::VectorXd> sampling_K = {};
                        std::vector<Eigen::VectorXd> sampling_P = {};
                        std::vector<Eigen::VectorXd> sampling_T = {};

                        std::cout << "Start Bubble Point Algorithm" << std::endl;

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

                            // Initialization Routine 1: Shortcut K-ratio method / Raoult's law
                            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                _calculation.originalConcentrations(i_calculation, m) = _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) * _calculation.PartitionCoefficients(i_calculation, m);
                            }
                            if (param.sw_Tbub == 1) {
                                _calculation.originalConcentrations.array().row(i_calculation) *= 1 / _calculation.originalConcentrations.array().row(i_calculation).sum();
                            }
                            double startPressure = _calculation.targetPressure(i_calculation);
                            calculateBubblePoint(_calculation, i_calculation, 0.00001);

                            // Initialization Routine 2 if Bubble Point Calculation failed
                            if (param.sw_Pbub == 1 && _calculation.totalPressure(i_calculation) == 0.0) {

                                // In this case no equilibrium was found using the initial values of routine 1
                                if (int(_calculation.components.size() - 1) > 2) {
                                    throw std::runtime_error("Bubble Pressure Algorithm did not converge. Sampling currently only works for binary mixtures.");
                                }

                                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                                double temperature = _calculation.temperatures[actualConcentrationIndex];
                                auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
                                int index = it - _calculation.uniqueTemperatures.begin();
                                int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                int IndexOfSupercriticalComponent;
                                int IndexOfSubcriticalComponent;
                                int SubcalculationsIndexOfSupercriticalComponent;
                                int SubcalculationsIndexOfSubcriticalComponent;
                                if (subcalculations[subcalcindex_comp1].criticalT(0) < temperature && subcalculations[subcalcindex_comp2].criticalT(0) > temperature) {
                                    IndexOfSupercriticalComponent = 0;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp1;
                                    IndexOfSubcriticalComponent = 1;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp2;
                                }
                                else if ((subcalculations[subcalcindex_comp1].criticalT(0) > temperature && subcalculations[subcalcindex_comp2].criticalT(0) < temperature) || (subcalculations[subcalcindex_comp1].criticalT(0) > temperature && subcalculations[subcalcindex_comp2].criticalT(0) > temperature)) {
                                    IndexOfSupercriticalComponent = 1;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                    IndexOfSubcriticalComponent = 0;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                }
                                else {
                                    throw std::runtime_error("Both components are in supercritical state for given T, P. For this situation, sampling currently doesn't work.");
                                }
                                double endx = _calculation.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();

                                if (sampling_x.size() == 0) {

                                    std::vector<std::vector<Eigen::VectorXd>> samplingResults;
                                    if (i_calculation == 0) {

                                        double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
                                        Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
                                        startx(IndexOfSupercriticalComponent) = 0.0;
                                        startx(IndexOfSubcriticalComponent) = 1.0;
                                        Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
                                        startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
                                        startK(IndexOfSubcriticalComponent) = 1.0;
                                        std::cout << "Start SamplingForVLE" << std::endl;
                                        samplingResults = samplingForVLE(_calculation.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, endx);
                                    }
                                    else {
                                        std::cout << "Start SamplingForVLE" << std::endl;
                                        samplingResults = samplingForVLE(_calculation.components, temperature, IndexOfSupercriticalComponent, _calculation.originalConcentrations.row(i_calculation + numberOfCalculations - 1), _calculation.originalConcentrations.row(i_calculation - 1), _calculation.PartitionCoefficients.row(i_calculation - 1), _calculation.totalPressure(i_calculation - 1), {}, endx);
                                    }
                                    sampling_x = samplingResults[0];
                                    sampling_y = samplingResults[1];
                                    sampling_K = samplingResults[2];
                                    sampling_P = samplingResults[3];
                                }
                                double max_x = sampling_x[sampling_x.size() - 1](IndexOfSupercriticalComponent);
                                if (_calculation.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent) <= max_x) {
                                    int index = 0;
                                    for (int i = 0; i < sampling_x.size(); i++) {
                                        if (sampling_x[i](IndexOfSupercriticalComponent) > _calculation.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent)) {
                                            if (i == 0) {
                                                index = i;
                                            }
                                            else {
                                                index = i - 1;
                                            }
                                            break;
                                        }
                                    }
                                    _calculation.originalConcentrations.row(i_calculation) = sampling_y[index];
                                    _calculation.PartitionCoefficients.row(i_calculation) = sampling_K[index];
                                    _calculation.targetPressure.row(i_calculation) = sampling_P[index];
                                    calculateBubblePoint(_calculation, i_calculation, 0.00001);
                                }
                                else {
                                    throw std::runtime_error("For given mixture, only one single phase is to be expected.");
                                }
                            }
                            else if (param.sw_Tbub == 1 && _calculation.targetTemperature(i_calculation) == 0.0) {
                                // In this case no equilibrium was found using the initial values of routine 1

                                if (int(_calculation.components.size() - 1) > 2) {
                                    throw std::runtime_error("Bubble Temperature Algorithm did not converge. Sampling currently only works for binary mixtures.");
                                }

                                double pressure = _calculation.targetPressure(i_calculation);
                                std::cout << pressure << std::endl;
                                double TsatComp1 = _calculation.SaturationTemperatures(i_calculation, 0);
                                double TsatComp2 = _calculation.SaturationTemperatures(i_calculation, 1);
                                std::vector<double> TsatPure;
                                TsatPure.push_back(TsatComp1);
                                TsatPure.push_back(TsatComp2);

                                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                                auto it = std::find(_calculation.uniquePressures.begin(), _calculation.uniquePressures.end(), pressure);
                                int index = it - _calculation.uniquePressures.begin();
                                int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                int IndexOfSupercriticalComponent;
                                int IndexOfSubcriticalComponent;
                                int SubcalculationsIndexOfSupercriticalComponent;
                                int SubcalculationsIndexOfSubcriticalComponent;
                                std::cout << subcalculations[subcalcindex_comp1].criticalP(0) << std::endl;
                                std::cout << subcalculations[subcalcindex_comp2].criticalP(0) << std::endl;
                                //if ((subcalculations[subcalcindex_comp1].criticalP(0) < pressure && subcalculations[subcalcindex_comp2].criticalP(0) > pressure) || ((subcalculations[subcalcindex_comp1].criticalP(0) > pressure) && (subcalculations[subcalcindex_comp2].criticalP(0) > pressure))) {
                                //    IndexOfSupercriticalComponent = 0;
                                //    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp1;
                                //    IndexOfSubcriticalComponent = 1;
                                //    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp2;
                                //}
                                //else if (subcalculations[subcalcindex_comp1].criticalP(0) > pressure && subcalculations[subcalcindex_comp2].criticalP(0) < pressure) {
                                //    IndexOfSupercriticalComponent = 1;
                                //    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                //    IndexOfSubcriticalComponent = 0;
                                //    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                //}
                                //else {
                                //    throw std::runtime_error("Both components are in supercritical state for given T, P. For this situation, sampling currently doesn't work.");
                                //}
                                IndexOfSupercriticalComponent = 0;
                                SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp1;
                                IndexOfSubcriticalComponent = 1;
                                SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp2;

                                if (sampling_x.size() == 0) {

                                    std::vector<std::vector<Eigen::VectorXd>> samplingResults;
                                    if (i_calculation == 0) {
                                        double endx = _calculation.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();
                                        double starttemperature = _calculation.SaturationTemperatures(i_calculation, IndexOfSubcriticalComponent);
                                        Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
                                        startx(IndexOfSupercriticalComponent) = 0.0;
                                        startx(IndexOfSubcriticalComponent) = 1.0;
                                        Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);

                                        startK(IndexOfSupercriticalComponent) = _calculation.SaturationPressures(i_calculation, IndexOfSupercriticalComponent) / _calculation.SaturationPressures(i_calculation, IndexOfSubcriticalComponent);
                                        startK(IndexOfSubcriticalComponent) = 1.0;
                                        std::cout << "Start sampling for Bubble T Calculation" << std::endl;
                                        samplingResults = samplingForVLE(_calculation.components, starttemperature, IndexOfSupercriticalComponent, startx, startx, startK, pressure, TsatPure, endx);
                                    }
                                    else {
                                        std::cout << "Start sampling for Bubble T Calculation" << std::endl;
                                        double endx = _calculation.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();
                                        samplingResults = samplingForVLE(_calculation.components, _calculation.targetTemperature(i_calculation - 1), IndexOfSupercriticalComponent, _calculation.originalConcentrations.row(i_calculation + numberOfCalculations - 1), _calculation.originalConcentrations.row(i_calculation - 1), _calculation.PartitionCoefficients.row(i_calculation - 1), _calculation.targetPressure(i_calculation), TsatPure, endx);
                                    }
                                    sampling_x = samplingResults[0];
                                    sampling_y = samplingResults[1];
                                    sampling_K = samplingResults[2];
                                    sampling_T = samplingResults[4];
                                }
                                double max_x = sampling_x[sampling_x.size() - 1](IndexOfSupercriticalComponent);
                                if (_calculation.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent) <= max_x) {
                                    int index = 0;
                                    for (int i = 0; i < sampling_x.size(); i++) {
                                        if (sampling_x[i](IndexOfSupercriticalComponent) > _calculation.originalConcentrations(i_calculation + numberOfCalculations, IndexOfSupercriticalComponent)) {
                                            if (i == 0) {
                                                index = i;
                                            }
                                            else {
                                                index = i - 1;
                                            }
                                            break;
                                        }
                                    }
                                    _calculation.originalConcentrations.row(i_calculation) = sampling_y[index];
                                    _calculation.PartitionCoefficients.row(i_calculation) = sampling_K[index];
                                    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                                    int actualConcentrationIndex2 = _calculation.actualConcentrationIndices[i_calculation + numberOfCalculations];
                                    int actualReferenceStateConcentrationIndex = _calculation.actualConcentrationIndices[_calculation.referenceStateCalculationIndices[i_calculation][0]];
                                    _calculation.temperatures[actualConcentrationIndex] = sampling_T[index](0);
                                    _calculation.temperatures[actualConcentrationIndex2] = sampling_T[index](0);
                                    _calculation.temperatures[actualReferenceStateConcentrationIndex] = sampling_T[index](0);
                                    for (int g = 0; g < _calculation.TauConcentrationIndices.size(); g++) {
                                        for (int h = 0; h < _calculation.TauConcentrationIndices[g].size(); h++) {
                                            if (actualConcentrationIndex == _calculation.TauConcentrationIndices[g][h]) {
                                                _calculation.TauTemperatures[g] = sampling_T[index](0);
                                            }
                                        }
                                    }
                                    calculateBubblePoint(_calculation, i_calculation, 0.00001);
                                }
                                else {
                                    throw std::runtime_error("For given mixture, only one single phase is to be expected.");
                                }
                            }
                        }
                    }


                    ///// DEW POINT CALCULATION (DEW T OR DEW P) /////
                    if (param.sw_Pdew == 1 || param.sw_Tdew == 1) {
                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

                            // Initialization Routine 1: Raoult's law
                            for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = _calculation.originalConcentrations(i_calculation, m) / _calculation.PartitionCoefficients(i_calculation, m);
                            }
                            // Perform normalization when calculating Dew T because of possible inaccuracies
                            if (param.sw_Tdew == 1) {
                                _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations) *= 1 / _calculation.originalConcentrations.array().row(i_calculation + numberOfCalculations).sum();
                            }
                            calculateDewPoint(_calculation, i_calculation, 0.0001);

                            // Initialization Routine 2: currently not implemented
                        }
                    }


                    ///// PT Flash /////
                    if (param.sw_PT_Flash == 1) {

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        std::vector<Eigen::VectorXd> sampling_x = {};
                        std::vector<Eigen::VectorXd> sampling_y = {};
                        std::vector<Eigen::VectorXd> sampling_K = {};
                        std::vector<Eigen::VectorXd> sampling_P = {};
                        std::vector<Eigen::VectorXd> sampling_T = {};

                        std::cout << "Start pT Flash Algorithm" << std::endl;

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

                            bool skipThetaCalculation = false;
                            bool useGivenK = false;
                            if (param.useGivenInitialK == true && !(_calculation.calculatedPartitionCoefficients.row(i_calculation).isZero(1e-7))) {
                                _calculation.PartitionCoefficients.row(i_calculation) = _calculation.calculatedPartitionCoefficients.row(i_calculation);
                                useGivenK = true;
                            }

                            double thetaGuess = 0;
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];

                            // Initialization Routine 1: Shortcut K-ratio method / Raoult's law
                            double pressure = _calculation.targetPressure(i_calculation);

                            if (_calculation.calculationType == "LL") {

                                if (param.useGivenInitialK == false) {
                                    // Initialization scheme for LLE
                                    std::vector<double> gammas;
                                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                    std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                                    std::vector<double> liquid1_v = returnVector[2];
                                    std::vector<double> liquid1_P = returnVector[3];

                                    double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid1_v, liquid1_P, pressure);
                                    _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                                    calculateEOS(param, _calculation, i_calculation + numberOfCalculations);

                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        double fugacityCoefficientLiquid1 = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                                        _calculation.originalConcentrations.row(i_calculation).setZero();
                                        _calculation.originalConcentrations(i_calculation, m) = 1;
                                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;
                                        std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                        std::vector<double> liquid2_v = returnVector[2];
                                        std::vector<double> liquid2_P = returnVector[3];
                                        double v = calculatevForGivenTPn(_calculation, i_calculation, liquid2_v, liquid2_P, pressure);
                                        _calculation.molarVolume(i_calculation) = v;
                                        calculateEOS(param, _calculation, i_calculation);
                                        double gammamL1 = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m) - _calculation.lnPhiTotal(i_calculation, m));
                                        gammas.push_back(gammamL1);
                                    }

                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        _calculation.originalConcentrations(i_calculation, m) = gammas[m] * _calculation.originalConcentrations(i_calculation + numberOfCalculations, m);
                                    }

                                    double sum = _calculation.originalConcentrations.array().row(i_calculation).sum();
                                    _calculation.originalConcentrations.array().row(i_calculation) *= 1 / sum;

                                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                    returnVector = calculateIsotherm(_calculation, i_calculation);
                                    std::vector<double> liquid2_v = returnVector[2];
                                    std::vector<double> liquid2_P = returnVector[3];

                                    v = calculatevForGivenTPn(_calculation, i_calculation, liquid2_v, liquid2_P, pressure);
                                    _calculation.molarVolume(i_calculation) = v;
                                    calculateEOS(param, _calculation, i_calculation);

                                    _calculation.PartitionCoefficients = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), numberOfOriginalComponents);
                                    int indexWithMaxKi = -1;
                                    double MaxPartitionCoefficient = 0;
                                    int indexWithMinKi = -1;
                                    double MinPartitionCoefficient = 100000;
                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        double fugacityCoefficientLiquid1 = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                                        double fugacityCoefficientLiquid2 = exp(_calculation.lnPhiTotal(i_calculation, m));
                                        double PartitionCoefficient = fugacityCoefficientLiquid1 / fugacityCoefficientLiquid2;
                                        if (PartitionCoefficient > MaxPartitionCoefficient) {
                                            indexWithMaxKi = m;
                                            MaxPartitionCoefficient = PartitionCoefficient;
                                        }
                                        if (PartitionCoefficient < MinPartitionCoefficient) {
                                            indexWithMinKi = m;
                                            MinPartitionCoefficient = PartitionCoefficient;
                                        }
                                        _calculation.PartitionCoefficients(i_calculation, m) = PartitionCoefficient;
                                    }

                                    int counter = 0;
                                    bool eqCondFound = false;

                                    while (eqCondFound != true) {
                                        _calculation.originalConcentrations.row(i_calculation).setZero();
                                        _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).setZero();
                                        _calculation.originalConcentrations(i_calculation, indexWithMaxKi) = 0.98 - counter * 0.002;
                                        _calculation.originalConcentrations(i_calculation, indexWithMinKi) = 0.02 + counter * 0.002;
                                        _calculation.originalConcentrations(i_calculation + numberOfCalculations, indexWithMaxKi) = 0.02 + counter * 0.002;
                                        _calculation.originalConcentrations(i_calculation + numberOfCalculations, indexWithMinKi) = 0.98 - counter * 0.002;
                                        _calculation.PartitionCoefficients.row(i_calculation).setZero();
                                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                        returnVector = calculateIsotherm(_calculation, i_calculation);
                                        std::vector<double> vapor_v; // For LL this is actually L2
                                        std::vector<double> vapor_P;
                                        vapor_v = returnVector[2];
                                        vapor_P = returnVector[3];
                                        returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                                        std::vector<double> liquid_v = returnVector[2];
                                        std::vector<double> liquid_P = returnVector[3];

                                        calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

                                        if (_calculation.totalPressure(i_calculation) != 0) {
                                            eqCondFound = true;
                                            break;
                                        }
                                        else {
                                            counter += 1;
                                        }
                                        if (counter == 100) {
                                            throw std::runtime_error("Initialization scheme for LLE failed, no LLE calculation possible.");
                                        }
                                    }

                                    double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, indexWithMaxKi));
                                    double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, indexWithMaxKi));
                                    _calculation.PartitionCoefficients(i_calculation, indexWithMaxKi) = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                                    fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, indexWithMinKi));
                                    fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, indexWithMinKi));
                                    _calculation.PartitionCoefficients(i_calculation, indexWithMinKi) = fugacityCoefficientLiquid / fugacityCoefficientVapor;

                                    thetaGuess = (_calculation.overallConcentrations(i_calculation, indexWithMaxKi) * (1 - 1 / _calculation.PartitionCoefficients(i_calculation, indexWithMaxKi)) + _calculation.overallConcentrations(i_calculation, indexWithMinKi) * _calculation.PartitionCoefficients(i_calculation, indexWithMinKi)) / (_calculation.overallConcentrations(i_calculation, indexWithMinKi) + _calculation.overallConcentrations(i_calculation, indexWithMaxKi));
                                    skipThetaCalculation = true;
                                }

                                calculatePTFlash(_calculation, i_calculation, 0.001, skipThetaCalculation, thetaGuess, true);
                            }

                            else if (_calculation.calculationType == "VL") {
                                auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
                                int index = it - _calculation.uniqueTemperatures.begin();
                                int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                if (sampling_x.size() == 0) {
                                    calculatePTFlash(_calculation, i_calculation, 0.001, skipThetaCalculation, thetaGuess, useGivenK);
                                }
                            }

                            // Initialization Routine 2 if Shortcut K ratio method failed
                            if (param.noTracing == false && _calculation.calculationType == "VL" && (_calculation.totalPressure(i_calculation) == 0.0 || sampling_x.size() > 0)) {

                                if (int(_calculation.components.size() - 1) > 2) {
                                    throw std::runtime_error("PT Flash Algorithm did not converge. Sampling currently only works for binary mixtures.");
                                }

                                int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                                double temperature = _calculation.temperatures[actualConcentrationIndex];
                                auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
                                int index = it - _calculation.uniqueTemperatures.begin();
                                int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                int IndexOfSupercriticalComponent;
                                int IndexOfSubcriticalComponent;
                                int SubcalculationsIndexOfSupercriticalComponent;
                                int SubcalculationsIndexOfSubcriticalComponent;
                                if (((subcalculations[subcalcindex_comp1].criticalT(0) > temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) > temperature))) {
                                    IndexOfSupercriticalComponent = 1;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                    IndexOfSubcriticalComponent = 0;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                }
                                else if (((subcalculations[subcalcindex_comp1].criticalT(0) > temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) < temperature))) {
                                    IndexOfSupercriticalComponent = 1;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                    IndexOfSubcriticalComponent = 0;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                }
                                else if ((subcalculations[subcalcindex_comp1].criticalT(0) < temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) > temperature)) {
                                    IndexOfSupercriticalComponent = 0;
                                    SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp1;
                                    IndexOfSubcriticalComponent = 1;
                                    SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp2;
                                }
                                else {
                                    throw std::runtime_error("Both components are in supercritical state for given T, P. For this situation, sampling currently doesn't work.");
                                }
                                double endx = _calculation.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();

                                if (sampling_x.size() == 0) {

                                    std::vector<std::vector<Eigen::VectorXd>> samplingResults;
                                    param.sw_Pbub = 1;
                                    if (i_calculation == 0 || param.sw_excess == 1) {
                                        double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
                                        Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
                                        startx(IndexOfSupercriticalComponent) = 0.0;
                                        startx(IndexOfSubcriticalComponent) = 1.0;
                                        Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
                                        startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
                                        startK(IndexOfSubcriticalComponent) = 1.0;
                                        samplingResults = samplingForVLE(_calculation.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, endx);
                                    }
                                    else {
                                        samplingResults = samplingForVLE(_calculation.components, temperature, IndexOfSupercriticalComponent, _calculation.originalConcentrations.row(i_calculation + numberOfCalculations - 1), _calculation.originalConcentrations.row(i_calculation - 1), _calculation.PartitionCoefficients.row(i_calculation - 1), _calculation.totalPressure(i_calculation - 1), {}, endx);
                                    }
                                    sampling_x = samplingResults[0];
                                    sampling_y = samplingResults[1];
                                    sampling_K = samplingResults[2];
                                    sampling_P = samplingResults[3];
                                }
                                double min_conc = 1.0;
                                double max_conc = 0.0;
                                double min_P = sampling_P[0](0);
                                double max_P = sampling_P[0](0);
                                for (int i = 0; i < sampling_P.size(); i++) {
                                    if (sampling_P[i](0) < min_P) {
                                        min_P = sampling_P[i](0);
                                    }
                                    if (sampling_P[i](0) > max_P) {
                                        max_P = sampling_P[i](0);
                                    }
                                    if (sampling_x[i](IndexOfSupercriticalComponent) < min_conc) {
                                        min_conc = sampling_x[i](IndexOfSupercriticalComponent);
                                    }
                                    if (sampling_y[i](IndexOfSupercriticalComponent) < min_conc) {
                                        min_conc = sampling_y[i](IndexOfSupercriticalComponent);
                                    }
                                    if (sampling_x[i](IndexOfSupercriticalComponent) > max_conc) {
                                        max_conc = sampling_x[i](IndexOfSupercriticalComponent);
                                    }
                                    if (sampling_y[i](IndexOfSupercriticalComponent) > max_conc) {
                                        max_conc = sampling_y[i](IndexOfSupercriticalComponent);
                                    }
                                }

                                bool twoPhaseFound = false;
                                bool zInAnySegment = false;
                                bool AggregationStateDetermined = false;
                                int twoPhaseIndex = -1;
                                int zinSegmentIndex = -1;

                                if (sampling_P.size() > 1 &&
                                    min_conc <= _calculation.overallConcentrations(i_calculation, IndexOfSupercriticalComponent) &&
                                    _calculation.overallConcentrations(i_calculation, IndexOfSupercriticalComponent) <= max_conc)
                                {
                                    double z = _calculation.overallConcentrations(i_calculation, IndexOfSupercriticalComponent);
                                    double P_target = _calculation.targetPressure(i_calculation);

                                    for (int i = 1; i < sampling_P.size(); ++i)
                                    {
                                        double xmin_i1 = std::min(sampling_x[i - 1](IndexOfSupercriticalComponent),
                                            sampling_y[i - 1](IndexOfSupercriticalComponent));
                                        double xmax_i1 = std::max(sampling_x[i - 1](IndexOfSupercriticalComponent),
                                            sampling_y[i - 1](IndexOfSupercriticalComponent));

                                        double xmin_i = std::min(sampling_x[i](IndexOfSupercriticalComponent),
                                            sampling_y[i](IndexOfSupercriticalComponent));
                                        double xmax_i = std::max(sampling_x[i](IndexOfSupercriticalComponent),
                                            sampling_y[i](IndexOfSupercriticalComponent));

                                        double zmin = std::min(xmin_i1, xmin_i);
                                        double zmax = std::max(xmax_i1, xmax_i);

                                        if (!(zmin <= z && z <= zmax))
                                            continue;

                                        zInAnySegment = true;
                                        zinSegmentIndex = i;

                                        double P_i1 = sampling_P[i - 1](0);
                                        double P_i = sampling_P[i](0);

                                        double P_low = std::min(P_i1, P_i);
                                        double P_high = std::max(P_i1, P_i);

                                        if (P_low <= P_target && P_target <= P_high)
                                        {
                                            double xmin_seg = std::min({ xmin_i1, xmin_i });
                                            double xmax_seg = std::max({ xmax_i1, xmax_i });

                                            if (xmin_seg <= z && z <= xmax_seg)
                                            {
                                                twoPhaseFound = true;
                                                twoPhaseIndex = i - 1;

                                                break;
                                            }
                                        }
                                    }
                                }

                                if (twoPhaseFound) {
                                    _calculation.originalConcentrations.row(i_calculation) =
                                        sampling_y[twoPhaseIndex];
                                    _calculation.originalConcentrations.row(i_calculation + numberOfCalculations) =
                                        sampling_x[twoPhaseIndex];
                                    _calculation.PartitionCoefficients.row(i_calculation) =
                                        sampling_K[twoPhaseIndex];

                                    std::cout << "Found Two-phase split and calculate Flash" << std::endl;

                                    calculatePTFlash(_calculation, i_calculation, 0.001, false, 0.0, true);

                                    AggregationStateDetermined = true;
                                }
                                else if (zInAnySegment) {
                                    double P_i1 = sampling_P[zinSegmentIndex - 1](0);
                                    double P_i = sampling_P[zinSegmentIndex](0);

                                    double P_low = std::min(P_i1, P_i);
                                    double P_high = std::max(P_i1, P_i);

                                    double P_target = _calculation.targetPressure(i_calculation);

                                    if (P_target < P_low)
                                    {
                                        // single gaseous phase
                                        for (int m = 0; m < _calculation.components.size() - 1; ++m)
                                        {
                                            _calculation.originalConcentrations(i_calculation, m) =
                                                _calculation.overallConcentrations(i_calculation, m);
                                            _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) =
                                                0.0;
                                        }
                                        AggregationStateDetermined = true;
                                    }
                                    else if (P_target > P_high)
                                    {
                                        // single liquid phase
                                        for (int m = 0; m < _calculation.components.size() - 1; ++m)
                                        {
                                            _calculation.originalConcentrations(i_calculation, m) =
                                                0.0;
                                            _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) =
                                                _calculation.overallConcentrations(i_calculation, m);
                                        }
                                        AggregationStateDetermined = true;
                                    }
                                }

                                if (AggregationStateDetermined == false) {
                                    throw std::runtime_error("An error occured, aggregation state of mixture could not be determined.");
                                }
                            }
                        }
                    }


                    ///// ISOTHERM TO TXT FILE /////
                    if (param.sw_isotherm == 1) {
                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {
                            double pressure = _calculation.targetPressure(i_calculation);

                            auto start = std::chrono::steady_clock::now();

                            std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                            std::vector<double> liquid_v = returnVector[2];
                            std::vector<double> liquid_P = returnVector[3];
                            std::vector<double> vapor_v = returnVector[4];
                            std::vector<double> vapor_P = returnVector[5];
                            std::vector<double> two_phase_region_v = returnVector[8];
                            std::vector<double> two_phase_region_P = returnVector[9];

                            auto end = std::chrono::steady_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                            //std::cout << "Dauer: " << duration.count() << " �s\n";

                            Write1DArrayToFile<double>(std::to_string(i_calculation) + "_vapor_v.txt", &vapor_v[0], vapor_v.size(), 1);
                            Write1DArrayToFile<double>(std::to_string(i_calculation) + "_vapor_P.txt", &vapor_P[0], vapor_P.size(), 1);
                            Write1DArrayToFile<double>(std::to_string(i_calculation) + "_liquid_v.txt", &liquid_v[0], liquid_v.size(), 1);
                            Write1DArrayToFile<double>(std::to_string(i_calculation) + "_liquid_P.txt", &liquid_P[0], liquid_P.size(), 1);
                            Write1DArrayToFile<double>(std::to_string(i_calculation) + "_two_phase_region_v.txt", &two_phase_region_v[0], two_phase_region_v.size(), 1);
                            Write1DArrayToFile<double>(std::to_string(i_calculation) + "_two_phase_region_P.txt", &two_phase_region_P[0], two_phase_region_P.size(), 1);
                        }
                    }


                    ///// ACTIVITY COEFFICIENTS /////
                    if (param.sw_lnGamma == 1) {

                        ///// Identifying Solvent and Ionic components /////
                        int numberOfOriginalComponents = int(_calculation.components.size() - 1);
                        std::vector<int> SolventComponentIndices;
                        std::vector<int> IonsIndices = {};
                        if (param.sw_electrolyte == 1) {
                            for (int i = 0; i < numberOfOriginalComponents; i++) {
                                if (_calculation.components[i]->moleculeCharge == 0) {
                                    SolventComponentIndices.push_back(i);
                                }
                                else if (_calculation.components[i]->moleculeCharge != 0) {
                                    IonsIndices.push_back(i);
                                }
                            }
                        }
                        else {
                            SolventComponentIndices.reserve(numberOfOriginalComponents);
                            for (int i = 0; i < numberOfOriginalComponents; ++i) {
                                SolventComponentIndices.push_back(i);
                            }
                        }

                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

#ifndef OPT_NO_MEMO
                        // reuse the roots of rows with the same (T, P, composition) within one call
                        struct MiacSolventRefMemo { int component; double T; double P; double lnPhi; double lnPhiElectrostatic; double lnPhiBorn; };
                        struct MiacIonRefMemo { double T; double P; std::vector<double> x; Eigen::VectorXd lnPhi; Eigen::VectorXd lnPhiElectrostatic; Eigen::VectorXd lnPhiBorn; };
                        struct MiacMixtureVolumeMemo { double T; double P; double b; double bh; std::vector<double> x; double v; };
                        std::vector<MiacSolventRefMemo> solventRefMemo;
                        std::vector<MiacIonRefMemo> ionRefMemo;
                        std::vector<MiacMixtureVolumeMemo> mixtureVolumeMemo;
#endif
#ifndef OPT_NO_ROOT_HINTS
                        // roots of the previous call (0 solvent reference, 1 ion reference, 2 mixture row)
                        const std::vector<calculation::OptRootHint>& previousRootHints = calculations[calculationIndex].optRootHints;
                        std::vector<calculation::OptRootHint> currentRootHints;
                        auto findRootHint = [&previousRootHints](int kind, int component, double T, double P, const std::vector<double>& x) -> double {
                            for (const auto& hint : previousRootHints) {
                                if (hint.kind == kind && hint.component == component && hint.T == T && hint.P == P && hint.x == x) {
                                    return hint.v;
                                }
                            }
                            return -1.0;
                        };
#endif

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];
                            double pressure = _calculation.targetPressure(i_calculation);

                            ///// Calculate Phi in reference state /////
                            Eigen::VectorXd LnPhiRef = Eigen::VectorXd::Zero(int(_calculation.components.size() - 1));
                            Eigen::VectorXd LnPhiRef_Electrostatic = Eigen::VectorXd::Zero(int(_calculation.components.size() - 1));
                            Eigen::VectorXd LnPhiRef_Born = Eigen::VectorXd::Zero(int(_calculation.components.size() - 1));

                            // Solvent Molecules
                            double ConcentrationSumOfAllSolventMolecules = 0.0;
                            double meanMolarMass = 0.0;
                            for (int i = 0; i < int(SolventComponentIndices.size()); i++) {

                                int ComponentIndex = SolventComponentIndices[i];

                                meanMolarMass += _calculation.originalConcentrations(numberOfCalculations + i_calculation, ComponentIndex) * _calculation.components[ComponentIndex]->molarMass;

#ifndef OPT_NO_MEMO
                                {
                                    bool found = false;
                                    for (const auto& memo : solventRefMemo) {
                                        if (memo.component == ComponentIndex && memo.T == temperature && memo.P == pressure) {
                                            LnPhiRef(ComponentIndex) = memo.lnPhi;
                                            if (param.sw_electrolyte == 1) {
                                                LnPhiRef_Electrostatic(ComponentIndex) = memo.lnPhiElectrostatic;
                                                LnPhiRef_Born(ComponentIndex) = memo.lnPhiBorn;
                                            }
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (found) {
                                        ConcentrationSumOfAllSolventMolecules += _calculation.originalConcentrations(i_calculation + numberOfCalculations, ComponentIndex);
                                        continue;
                                    }
                                }
#endif

                                std::vector<double> pressureVector;
                                pressureVector.push_back(double(pressure));

                                std::vector<std::vector<double>> concentration(1, std::vector<double>(_calculation.components.size() - 1, 0.0));
                                concentration[0][0] = 1.0;

                                std::vector<std::shared_ptr<molecule>> componentVector;
                                componentVector.push_back(_calculation.components[ComponentIndex]);
                                componentVector.push_back(_calculation.components[int(_calculation.components.size() - 1)]); // hole always has to be appended

                                calculation calculationScheme = loadSubcalculation(componentVector, { double(temperature) }, concentration, pressureVector);
                                int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
                                calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});
                                calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(1);
                                calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(1);
                                calculationScheme.mixtureCavityVolume = Eigen::VectorXd::Zero(1);
                                calculationScheme.mixtureHoleVolume = Eigen::VectorXd::Zero(1);
                                calculationScheme.originalNumberOfCalculations = 2;

                                int index = _calculation.components[ComponentIndex]->index; // Note: index in calculationScheme is different, therefore use _calculation here
                                calculationScheme.componentCavityVolume(0) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                                calculationScheme.componentHoleVolume(0) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
                                calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
                                calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;

                                calculateBornAndSaltDiameter(calculationScheme);

#ifndef OPT_NO_ROOT_HINTS
                                const double solventRootHint = findRootHint(0, ComponentIndex, temperature, pressure, std::vector<double>());
#else
                                const double solventRootHint = -1.0;
#endif
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 0, 10.0, pressure, solventRootHint);
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];

                                double v = calculatevForGivenTPn(calculationScheme, 1, liquid_v, liquid_P, pressure);
#ifndef OPT_NO_ROOT_HINTS
                                currentRootHints.push_back({ 0, ComponentIndex, temperature, pressure, std::vector<double>(), v });
#endif
                                calculationScheme.molarVolume(1) = v;
                                calculateEOS(param, calculationScheme, 1);

                                LnPhiRef(ComponentIndex) = calculationScheme.lnPhiTotal(1, 0);
                                if (param.sw_electrolyte == 1) {
                                    LnPhiRef_Electrostatic(ComponentIndex) = calculationScheme.lnPhiElectrostatic(1, 0);
                                    LnPhiRef_Born(ComponentIndex) = calculationScheme.lnPhiBorn(1, 0);
                                }
#ifndef OPT_NO_MEMO
                                solventRefMemo.push_back({ ComponentIndex, temperature, pressure, LnPhiRef(ComponentIndex), LnPhiRef_Electrostatic(ComponentIndex), LnPhiRef_Born(ComponentIndex) });
#endif

                                ConcentrationSumOfAllSolventMolecules += _calculation.originalConcentrations(i_calculation + numberOfCalculations, ComponentIndex);
                            }

                            // Ions
                            if (int(IonsIndices.size()) > 0) {

                                if (ConcentrationSumOfAllSolventMolecules != 0.0) {
                                    std::vector<double> pressureVector;
                                    pressureVector.push_back(double(pressure));

                                    std::vector<std::vector<double>> concentration(1);
                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        if (_calculation.components[m]->moleculeCharge == 0) {
                                            concentration[0].push_back(_calculation.overallConcentrations(i_calculation, m) / ConcentrationSumOfAllSolventMolecules);
                                        }
                                        else if (_calculation.components[m]->moleculeCharge != 0) {
                                            concentration[0].push_back(0.0);
                                        }
                                    }

#ifndef OPT_NO_MEMO
                                    const MiacIonRefMemo* ionRefHit = nullptr;
                                    for (const auto& memo : ionRefMemo) {
                                        if (memo.T == temperature && memo.P == pressure && memo.x == concentration[0]) {
                                            ionRefHit = &memo;
                                            break;
                                        }
                                    }
                                    if (ionRefHit != nullptr) {
                                        for (int i = 0; i < int(IonsIndices.size()); i++) {
                                            int ComponentIndex = IonsIndices[i];
                                            LnPhiRef(ComponentIndex) = ionRefHit->lnPhi(ComponentIndex);
                                            LnPhiRef_Electrostatic(ComponentIndex) = ionRefHit->lnPhiElectrostatic(ComponentIndex);
                                            LnPhiRef_Born(ComponentIndex) = ionRefHit->lnPhiBorn(ComponentIndex);
                                        }
                                    }
                                    else
#endif
                                    {
                                    calculation calculationScheme = loadSubcalculation(_calculation.components, { double(temperature) }, concentration, pressureVector);
                                    int actualReferenceStateConcentrationIndex = calculationScheme.actualConcentrationIndices[calculationScheme.referenceStateCalculationIndices[0][0]]; // the reference state is the same for all calculations
                                    calculateSegmentConcentrations(calculationScheme, std::vector<int>{actualReferenceStateConcentrationIndex});
                                    calculationScheme.originalConcentrations.array().row(0) *= 1 / calculationScheme.originalConcentrations.array().row(0).sum();
                                    calculationScheme.componentCavityVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                                    calculationScheme.componentHoleVolume = Eigen::VectorXd::Zero(numberOfOriginalComponents);
                                    calculationScheme.originalNumberOfCalculations = 2;

                                    for (int i = 0; i < numberOfOriginalComponents; i++) {
                                        int index = calculationScheme.components[i]->index;
                                        calculationScheme.componentCavityVolume(i) = double(param.phi_param(index, 0) * VMOL_V); //b_i,i
                                        calculationScheme.componentHoleVolume(i) = double(param.phi_param(index, 3) * VMOL_V); //b_h,i
                                    }

                                    calculationScheme.mixtureCavityVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentCavityVolume;
                                    calculationScheme.mixtureHoleVolume = calculationScheme.originalConcentrations.cast<double>() * calculationScheme.componentHoleVolume;

                                    calculateBornAndSaltDiameter(calculationScheme);

#ifndef OPT_NO_ROOT_HINTS
                                    const double ionRootHint = findRootHint(1, -1, temperature, pressure, concentration[0]);
#else
                                    const double ionRootHint = -1.0;
#endif
                                    std::vector<std::vector<double>> returnVector = calculateIsotherm(calculationScheme, 0, 10.0, pressure, ionRootHint);
                                    std::vector<double> liquid_v = returnVector[2];
                                    std::vector<double> liquid_P = returnVector[3];
                                    std::vector<double> vapor_v = returnVector[4];
                                    std::vector<double> vapor_P = returnVector[5];

                                    double v = calculatevForGivenTPn(calculationScheme, 1, liquid_v, liquid_P, pressure);
#ifndef OPT_NO_ROOT_HINTS
                                    currentRootHints.push_back({ 1, -1, temperature, pressure, concentration[0], v });
#endif
                                    calculationScheme.molarVolume(1) = v;
                                    calculateEOS(param, calculationScheme, 1);

                                    for (int i = 0; i < int(IonsIndices.size()); i++) {
                                        int ComponentIndex = IonsIndices[i];
                                        LnPhiRef(ComponentIndex) = calculationScheme.lnPhiTotal(1, ComponentIndex);
                                        LnPhiRef_Electrostatic(ComponentIndex) = calculationScheme.lnPhiElectrostatic(1, ComponentIndex);
                                        LnPhiRef_Born(ComponentIndex) = calculationScheme.lnPhiBorn(1, ComponentIndex);
                                    }
#ifndef OPT_NO_MEMO
                                    ionRefMemo.push_back({ temperature, pressure, concentration[0], LnPhiRef, LnPhiRef_Electrostatic, LnPhiRef_Born });
#endif
                                    }
                                }
                                else {
                                    for (int i = 0; i < int(IonsIndices.size()); i++) {
                                        int ComponentIndex = IonsIndices[i];
                                        LnPhiRef(ComponentIndex) = 0.0;
                                        LnPhiRef_Electrostatic(ComponentIndex) = 0.0;
                                        LnPhiRef_Born(ComponentIndex) = 0.0;
                                    }
                                }
                            }


                            ///// Calculate Fugacity coefficient (liquid-like root) of all components in mixture
#ifndef OPT_NO_MEMO
                            const int mixtureRow = i_calculation + numberOfCalculations;
                            const double mixtureT = _calculation.temperatures[_calculation.actualConcentrationIndices[mixtureRow]];
                            const double mixtureB = _calculation.mixtureCavityVolume(mixtureRow);
                            const double mixtureBh = _calculation.mixtureHoleVolume(mixtureRow);
                            std::vector<double> mixtureX(numberOfOriginalComponents);
                            for (int m = 0; m < numberOfOriginalComponents; m++) {
                                mixtureX[m] = _calculation.originalConcentrations(mixtureRow, m);
                            }
                            double v = 0.0;
                            bool mixtureHit = false;
                            for (const auto& memo : mixtureVolumeMemo) {
                                if (memo.T == mixtureT && memo.P == pressure && memo.b == mixtureB && memo.bh == mixtureBh && memo.x == mixtureX) {
                                    v = memo.v;
                                    mixtureHit = true;
                                    break;
                                }
                            }
                            if (!mixtureHit) {
#ifndef OPT_NO_ROOT_HINTS
                                const double mixtureRootHint = findRootHint(2, -1, mixtureT, pressure, mixtureX);
#else
                                const double mixtureRootHint = -1.0;
#endif
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, mixtureRow, 10.0, pressure, mixtureRootHint);
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                v = calculatevForGivenTPn(_calculation, mixtureRow, liquid_v, liquid_P, pressure);
#ifndef OPT_NO_ROOT_HINTS
                                currentRootHints.push_back({ 2, -1, mixtureT, pressure, mixtureX, v });
#endif
                                mixtureVolumeMemo.push_back({ mixtureT, pressure, mixtureB, mixtureBh, mixtureX, v });
                            }
#else
                            std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations, 10.0, pressure);
                            std::vector<double> liquid_v = returnVector[2];
                            std::vector<double> liquid_P = returnVector[3];
                            std::vector<double> vapor_v = returnVector[4];
                            std::vector<double> vapor_P = returnVector[5];

                            double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
#endif
                            _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                            calculateEOS(param, _calculation, i_calculation + numberOfCalculations);
                            
                            //auto epsResult = calculateEpsilonMixAndDerivatives(_calculation, i_calculation + numberOfCalculations);
                            //std::cout << "i_calc " << i_calculation
                            //        << "  v = " << _calculation.molarVolume(i_calculation + numberOfCalculations) * 1e6 << " cm3/mol"
                            //        << "  eps_r_mix = " << std::get<0>(epsResult) << std::endl;


                            if (param.sw_electrolyte == 1) {
                                _calculation.lnGammaSRLRTotal.row(i_calculation + numberOfCalculations).head(int(_calculation.components.size() - 1)) = _calculation.lnPhiTotal.row(i_calculation + numberOfCalculations) - LnPhiRef.transpose();
                                _calculation.lnGammaBorn.row(i_calculation + numberOfCalculations).head(int(_calculation.components.size() - 1)) = _calculation.lnPhiBorn.row(i_calculation + numberOfCalculations) - LnPhiRef_Born.transpose();
                                _calculation.lnGammaElectrostatic.row(i_calculation + numberOfCalculations).head(int(_calculation.components.size() - 1)) = _calculation.lnPhiElectrostatic.row(i_calculation + numberOfCalculations) - LnPhiRef_Electrostatic.transpose();
                                
                                // At the moment, it is assumed that there is only one salt in the mixture
                                Eigen::VectorXd molalities = _calculation.originalConcentrations.row(i_calculation + numberOfCalculations) / (meanMolarMass * ConcentrationSumOfAllSolventMolecules);
                                double MolalityOfNondissociatedSolute = molalities(IonsIndices[0]) / _calculation.moleculeStochCoeffs[IonsIndices[0]];
                                int sumOfStochiometricCoefficientsofIons = 0;
                                double MIAC_x = 1;
                                for (int m = 0; m < int(IonsIndices.size()); m++) {
                                    int comp = IonsIndices[m];
                                    sumOfStochiometricCoefficientsofIons += _calculation.moleculeStochCoeffs[comp];
                                    MIAC_x *= pow(exp(_calculation.lnGammaSRLRTotal(i_calculation + numberOfCalculations, comp)), _calculation.moleculeStochCoeffs[comp]);
                                }
                                MIAC_x = pow(MIAC_x, 1.0 / sumOfStochiometricCoefficientsofIons);
                                double MIAC_m = 1000 * MIAC_x / (1 + meanMolarMass * MolalityOfNondissociatedSolute * sumOfStochiometricCoefficientsofIons);
                                for (int m = 0; m < int(IonsIndices.size()); m++) {
                                    int ComponentIndex = IonsIndices[m];
                                    _calculation.MIACInMoleFractionScale(i_calculation + numberOfCalculations, ComponentIndex) = MIAC_x;
                                    _calculation.MIACInMolalityScale(i_calculation + numberOfCalculations, ComponentIndex) = MIAC_m;
                                }
                            }
                            else {
                                _calculation.lnGammaTotal.row(i_calculation + numberOfCalculations).head(int(_calculation.components.size() - 1)) = _calculation.lnPhiTotal.row(i_calculation + numberOfCalculations) - LnPhiRef.transpose();
                            }
                        }
#ifndef OPT_NO_ROOT_HINTS
                        calculations[calculationIndex].optRootHints = currentRootHints;
#endif
                    }


                    ///// PARTITION COEFFICIENTS /////
                    if (param.sw_partitionCoeff == 1) {

                        ///// Identifying Solvent and Ionic components /////
                        int numberOfOriginalComponents = int(_calculation.components.size() - 1);
                        std::vector<int> SolventComponentIndices;
                        std::vector<int> IonsIndices = {};
                        if (param.sw_electrolyte == 1) {
                            for (int i = 0; i < numberOfOriginalComponents; i++) {
                                if (_calculation.components[i]->moleculeCharge == 0) {
                                    SolventComponentIndices.push_back(i);
                                }
                                else if (_calculation.components[i]->moleculeCharge != 0) {
                                    IonsIndices.push_back(i);
                                }
                            }
                        }
                        else {
                            SolventComponentIndices.reserve(numberOfOriginalComponents);
                            for (int i = 0; i < numberOfOriginalComponents; ++i) {
                                SolventComponentIndices.push_back(i);
                            }
                        }

                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

                            // The following properties have the same values for both phases (i_calculation and i_calculation + numberOfCalculations)
                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];
                            double pressure = _calculation.targetPressure(i_calculation);

                            ///// Calculate Fugacity coefficient (liquid-like root) of all components in mixture - Phase 1 /////
                            std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations, 10.0, pressure);
                            std::vector<double> liquid_v = returnVector[2];
                            std::vector<double> liquid_P = returnVector[3];
                            std::vector<double> vapor_v = returnVector[4];
                            std::vector<double> vapor_P = returnVector[5];

                            double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
                            _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                            calculateEOS(param, _calculation, i_calculation + numberOfCalculations);


                            ///// Calculate Fugacity coefficient (liquid-like root) of all components in mixture - Phase 2 /////
                            returnVector = calculateIsotherm(_calculation, i_calculation, 10.0, pressure);
                            liquid_v = returnVector[2];
                            liquid_P = returnVector[3];
                            vapor_v = returnVector[4];
                            vapor_P = returnVector[5];

                            v = calculatevForGivenTPn(_calculation, i_calculation, liquid_v, liquid_P, pressure);
                            _calculation.molarVolume(i_calculation) = v;
                            calculateEOS(param, _calculation, i_calculation);


                            if (param.sw_electrolyte == 1) {
                                // Partition coefficients of solvent molecules can be calculated as usual
                                for (int m = 0; m < int(SolventComponentIndices.size()); m++) {
                                    int comp = SolventComponentIndices[m];
                                    _calculation.calculatedPartitionCoefficients(i_calculation, comp) = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, comp) - _calculation.lnPhiTotal(i_calculation, comp));
                                }
                                // For partition coefficients of ions, mean fugacity coefficient must be used
                                double log_MFC_x_Phase1 = 0;
                                double log_MFC_x_Phase2 = 0;
                                int sumOfStochiometricCoefficientsofIons = 0;
                                for (int m = 0; m < int(IonsIndices.size()); m++) {
                                    int comp = IonsIndices[m];
                                    int nu = _calculation.moleculeStochCoeffs[comp];
                                    sumOfStochiometricCoefficientsofIons += nu;
                                    log_MFC_x_Phase1 += nu * _calculation.lnPhiTotal(i_calculation + numberOfCalculations, comp);
                                    log_MFC_x_Phase2 += nu * _calculation.lnPhiTotal(i_calculation, comp);
                                }
                                log_MFC_x_Phase1 /= sumOfStochiometricCoefficientsofIons;
                                log_MFC_x_Phase2 /= sumOfStochiometricCoefficientsofIons;
                                double Kion = exp(log_MFC_x_Phase1 - log_MFC_x_Phase2);
                                for (int m = 0; m < int(IonsIndices.size()); m++) {
                                    int comp = IonsIndices[m];
                                    _calculation.calculatedPartitionCoefficients(i_calculation, comp) = Kion;
                                }
                            }
                            else {
                                _calculation.calculatedPartitionCoefficients.row(i_calculation).head(int(_calculation.components.size() - 1)) = (_calculation.lnPhiTotal.row(i_calculation + numberOfCalculations) - _calculation.lnPhiTotal.row(i_calculation)).array().exp();
                            }
                        }
                    }


                    ///// EXCESS AND MIXING PROPERTIES /////
                    if (param.sw_excess == 1) {
                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                        int numberOfCalculations = int(_calculation.originalNumberOfCalculations / 2);

                        std::vector<Eigen::VectorXd> sampling_x = {};
                        std::vector<Eigen::VectorXd> sampling_y = {};
                        std::vector<Eigen::VectorXd> sampling_K = {};
                        std::vector<Eigen::VectorXd> sampling_P = {};
                        std::vector<Eigen::VectorXd> sampling_T = {};

                        for (int i_calculation = 0; i_calculation < numberOfCalculations; i_calculation++) {

                            int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                            double temperature = _calculation.temperatures[actualConcentrationIndex];
                            double pressure = _calculation.targetPressure(i_calculation);

                            bool phaseBoundariesKnown = false;

                            // Check if phase boundaries for current T, P are given exernally 
                            int indexInExternalParamVectors = -1;
                            for (size_t i = 0; i < param.temperatures_mixing.size(); ++i) {
                                if (std::abs(param.temperatures_mixing[i] - temperature) <= 1e-6 && std::abs(param.pressures_mixing[i] - pressure) <= 1e-6) {
                                    phaseBoundariesKnown = true;
                                    indexInExternalParamVectors = i;
                                    break;
                                }
                            }

                            std::string MixtureState = "UNKNOWN";

                            // If phase boundaries are given externally for current T, P, actual mixture state is directly known 
                            if (phaseBoundariesKnown) {
                                bool all_zero = std::all_of(param.concentration_x_phase_mixing[indexInExternalParamVectors].begin(), param.concentration_x_phase_mixing[indexInExternalParamVectors].end(), [](double elem) {
                                    return elem == 0.0;
                                    });
                                bool all_one = std::all_of(param.concentration_x_phase_mixing[indexInExternalParamVectors].begin(), param.concentration_x_phase_mixing[indexInExternalParamVectors].end(), [](double elem) {
                                    return elem == 1.0;
                                    });

                                if (all_zero || (_calculation.overallConcentrations(i_calculation, 0) <= param.concentration_x_phase_mixing[indexInExternalParamVectors][0])) {
                                    MixtureState = "liquid";
                                }
                                else if ((param.concentration_x_phase_mixing[indexInExternalParamVectors][0] < _calculation.overallConcentrations(i_calculation, 0)) && (_calculation.overallConcentrations(i_calculation, 0) < param.concentration_y_phase_mixing[indexInExternalParamVectors][0])) {
                                    MixtureState = "TwoPhaseSplit";
                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        _calculation.originalConcentrations(i_calculation + numberOfCalculations, m) = param.concentration_x_phase_mixing[indexInExternalParamVectors][m];
                                        _calculation.originalConcentrations(i_calculation, m) = param.concentration_y_phase_mixing[indexInExternalParamVectors][m];
                                    }
                                }
                                else if (all_one || (_calculation.overallConcentrations(i_calculation, 0) >= param.concentration_y_phase_mixing[indexInExternalParamVectors][0])) {
                                    MixtureState = "gaseous";
                                }
                            }

                            // If phase boundaries are not given externally for current T, P, actual mixture state has to be determined by performing a pT-Flash
                            else if (!(phaseBoundariesKnown)) {
                                double thetaGuess = 0;
                                bool skipThetaCalculation = false;
                                bool skipFlashCalculation = false;
                                auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
                                int index = it - _calculation.uniqueTemperatures.begin();
                                int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                if (subcalculations[subcalcindex_comp1].criticalT(0) < temperature && subcalculations[subcalcindex_comp2].criticalT(0) < temperature) {
                                    skipFlashCalculation = true;
                                    MixtureState = "UNKNOWN";
                                }

                                // Initialization Routine 1: Shortcut K-ratio method / Raoult's law
                                if (_calculation.calculationType == "LL") {
                                    // Initialization scheme for LLE
                                    std::vector<double> gammas;
                                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                    std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                                    std::vector<double> liquid1_v = returnVector[2];
                                    std::vector<double> liquid1_P = returnVector[3];

                                    double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid1_v, liquid1_P, pressure);
                                    _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                                    calculateEOS(param, _calculation, i_calculation + numberOfCalculations);

                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        double fugacityCoefficientLiquid1 = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                                        _calculation.originalConcentrations.row(i_calculation).setZero();
                                        _calculation.originalConcentrations(i_calculation, m) = 1;
                                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;
                                        std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                        std::vector<double> liquid2_v = returnVector[2];
                                        std::vector<double> liquid2_P = returnVector[3];
                                        double v = calculatevForGivenTPn(_calculation, i_calculation, liquid2_v, liquid2_P, pressure);
                                        _calculation.molarVolume(i_calculation) = v;
                                        calculateEOS(param, _calculation, i_calculation);
                                        double gammamL1 = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m) - _calculation.lnPhiTotal(i_calculation, m));
                                        gammas.push_back(gammamL1);
                                    }

                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        _calculation.originalConcentrations(i_calculation, m) = gammas[m] * _calculation.originalConcentrations(i_calculation + numberOfCalculations, m);
                                    }

                                    double sum = _calculation.originalConcentrations.array().row(i_calculation).sum();
                                    _calculation.originalConcentrations.array().row(i_calculation) *= 1 / sum;

                                    _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                    _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                    returnVector = calculateIsotherm(_calculation, i_calculation);
                                    std::vector<double> liquid2_v = returnVector[2];
                                    std::vector<double> liquid2_P = returnVector[3];

                                    v = calculatevForGivenTPn(_calculation, i_calculation, liquid2_v, liquid2_P, pressure);
                                    _calculation.molarVolume(i_calculation) = v;
                                    calculateEOS(param, _calculation, i_calculation);

                                    _calculation.PartitionCoefficients = Eigen::MatrixXd(int(_calculation.originalConcentrations.rows()), numberOfOriginalComponents);
                                    int indexWithMaxKi = -1;
                                    double MaxPartitionCoefficient = 0;
                                    int indexWithMinKi = -1;
                                    double MinPartitionCoefficient = 100000;
                                    for (int m = 0; m < _calculation.components.size() - 1; m++) {
                                        double fugacityCoefficientLiquid1 = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, m));
                                        double fugacityCoefficientLiquid2 = exp(_calculation.lnPhiTotal(i_calculation, m));
                                        double PartitionCoefficient = fugacityCoefficientLiquid1 / fugacityCoefficientLiquid2;
                                        if (PartitionCoefficient > MaxPartitionCoefficient) {
                                            indexWithMaxKi = m;
                                            MaxPartitionCoefficient = PartitionCoefficient;
                                        }
                                        if (PartitionCoefficient < MinPartitionCoefficient) {
                                            indexWithMinKi = m;
                                            MinPartitionCoefficient = PartitionCoefficient;
                                        }
                                        _calculation.PartitionCoefficients(i_calculation, m) = PartitionCoefficient;
                                    }

                                    int counter = 0;
                                    bool eqCondFound = false;

                                    while (eqCondFound != true) {
                                        _calculation.originalConcentrations.row(i_calculation).setZero();
                                        _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).setZero();
                                        _calculation.originalConcentrations(i_calculation, indexWithMaxKi) = 0.98 - counter * 0.002;
                                        _calculation.originalConcentrations(i_calculation, indexWithMinKi) = 0.02 + counter * 0.002;
                                        _calculation.originalConcentrations(i_calculation + numberOfCalculations, indexWithMaxKi) = 0.02 + counter * 0.002;
                                        _calculation.originalConcentrations(i_calculation + numberOfCalculations, indexWithMinKi) = 0.98 - counter * 0.002;
                                        _calculation.PartitionCoefficients.row(i_calculation).setZero();
                                        _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                        _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                        returnVector = calculateIsotherm(_calculation, i_calculation);
                                        std::vector<double> vapor_v; // For LL this is actually L2
                                        std::vector<double> vapor_P;
                                        vapor_v = returnVector[2];
                                        vapor_P = returnVector[3];
                                        returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                                        std::vector<double> liquid_v = returnVector[2];
                                        std::vector<double> liquid_P = returnVector[3];

                                        calculateEqCondForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, liquid_v, liquid_P, false);

                                        if (_calculation.totalPressure(i_calculation) != 0) {
                                            eqCondFound = true;
                                            break;
                                        }
                                        else {
                                            counter += 1;
                                        }
                                        if (counter == 100) {
                                            throw std::runtime_error("Initialization scheme for LLE failed, no LLE calculation possible.");
                                        }
                                    }

                                    double fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, indexWithMaxKi));
                                    double fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, indexWithMaxKi));
                                    _calculation.PartitionCoefficients(i_calculation, indexWithMaxKi) = fugacityCoefficientLiquid / fugacityCoefficientVapor;
                                    fugacityCoefficientVapor = exp(_calculation.lnPhiTotal(i_calculation, indexWithMinKi));
                                    fugacityCoefficientLiquid = exp(_calculation.lnPhiTotal(i_calculation + numberOfCalculations, indexWithMinKi));
                                    _calculation.PartitionCoefficients(i_calculation, indexWithMinKi) = fugacityCoefficientLiquid / fugacityCoefficientVapor;

                                    thetaGuess = (_calculation.overallConcentrations(i_calculation, indexWithMaxKi) * (1 - 1 / _calculation.PartitionCoefficients(i_calculation, indexWithMaxKi)) + _calculation.overallConcentrations(i_calculation, indexWithMinKi) * _calculation.PartitionCoefficients(i_calculation, indexWithMinKi)) / (_calculation.overallConcentrations(i_calculation, indexWithMinKi) + _calculation.overallConcentrations(i_calculation, indexWithMaxKi));
                                    skipThetaCalculation = true;

                                    calculatePTFlash(_calculation, i_calculation, 0.001, skipThetaCalculation, thetaGuess, true);

                                    if (!(_calculation.originalConcentrations.row(i_calculation + numberOfCalculations).isZero(1e-7)) && !(_calculation.originalConcentrations.row(i_calculation).isZero(1e-7)) && !(_calculation.totalPressure(i_calculation) == 0.0)) {
                                        param.temperatures_mixing.push_back(temperature);
                                        param.pressures_mixing.push_back(pressure);
                                        std::vector<double> x_vector = {};
                                        std::vector<double> y_vector = {};
                                        for (size_t i_c = 0; i_c + 1 < _calculation.components.size(); ++i_c) {
                                            x_vector.push_back(_calculation.originalConcentrations(i_calculation + numberOfCalculations, i_c));
                                            y_vector.push_back(_calculation.originalConcentrations(i_calculation, i_c));
                                        }
                                        param.concentration_x_phase_mixing.push_back(x_vector);
                                        param.concentration_y_phase_mixing.push_back(y_vector);
                                        MixtureState = "TwoPhaseSplit";
                                    }
                                }

                                else if (skipFlashCalculation == false && sampling_x.size() == 0 && _calculation.calculationType == "VL") {
                                    calculatePTFlash(_calculation, i_calculation, 0.001, skipThetaCalculation, thetaGuess);
                                    if (!(_calculation.originalConcentrations.row(i_calculation + numberOfCalculations).isZero(1e-7)) && !(_calculation.originalConcentrations.row(i_calculation).isZero(1e-7)) && !(_calculation.totalPressure(i_calculation) == 0.0)) {
                                        param.temperatures_mixing.push_back(temperature);
                                        param.pressures_mixing.push_back(pressure);
                                        std::vector<double> x_vector = {};
                                        std::vector<double> y_vector = {};
                                        for (size_t i_c = 0; i_c + 1 < _calculation.components.size(); ++i_c) {
                                            x_vector.push_back(_calculation.originalConcentrations(i_calculation + numberOfCalculations, i_c));
                                            y_vector.push_back(_calculation.originalConcentrations(i_calculation, i_c));
                                        }
                                        param.concentration_x_phase_mixing.push_back(x_vector);
                                        param.concentration_y_phase_mixing.push_back(y_vector);
                                        MixtureState = "TwoPhaseSplit";
                                    }
                                }

                                // Initialization Routine 2 if Flash Algorithm failed
                                if (skipFlashCalculation == false && _calculation.calculationType == "VL" && (_calculation.totalPressure(i_calculation) == 0.0 || sampling_x.size() > 0)) {
                                    // In this case no equilibrium was found

                                    int actualConcentrationIndex = _calculation.actualConcentrationIndices[i_calculation];
                                    double temperature = _calculation.temperatures[actualConcentrationIndex];
                                    auto it = std::find(_calculation.uniqueTemperatures.begin(), _calculation.uniqueTemperatures.end(), temperature);
                                    int index = it - _calculation.uniqueTemperatures.begin();
                                    int subcalcindex_comp1 = _calculation.IndicesForPureComponentCalculations(index, 0);
                                    int subcalcindex_comp2 = _calculation.IndicesForPureComponentCalculations(index, 1);
                                    int IndexOfSupercriticalComponent;
                                    int IndexOfSubcriticalComponent;
                                    int SubcalculationsIndexOfSupercriticalComponent;
                                    int SubcalculationsIndexOfSubcriticalComponent;
                                    if (((subcalculations[subcalcindex_comp1].criticalT(0) > temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) > temperature))) {
                                        IndexOfSupercriticalComponent = 1;
                                        SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                        IndexOfSubcriticalComponent = 0;
                                        SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                    }
                                    else if (((subcalculations[subcalcindex_comp1].criticalT(0) > temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) < temperature))) {
                                        IndexOfSupercriticalComponent = 1;
                                        SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp2;
                                        IndexOfSubcriticalComponent = 0;
                                        SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp1;
                                    }
                                    else if ((subcalculations[subcalcindex_comp1].criticalT(0) < temperature) && (subcalculations[subcalcindex_comp2].criticalT(0) > temperature)) {
                                        IndexOfSupercriticalComponent = 0;
                                        SubcalculationsIndexOfSupercriticalComponent = subcalcindex_comp1;
                                        IndexOfSubcriticalComponent = 1;
                                        SubcalculationsIndexOfSubcriticalComponent = subcalcindex_comp2;
                                    }
                                    else {
                                        skipFlashCalculation = true;
                                    }
                                    double endx = _calculation.originalConcentrations.col(IndexOfSupercriticalComponent).bottomRows(numberOfCalculations).maxCoeff();

                                    if (skipFlashCalculation == false && sampling_x.size() == 0) {
                                        if (int(_calculation.components.size() - 1) > 2) {
                                            throw std::runtime_error("PT Flash Algorithm did not converge. Sampling currently only works for binary mixtures.");
                                        }
                                        else {
                                            std::vector<std::vector<Eigen::VectorXd>> samplingResults;
                                            param.sw_Pbub = 1;
                                            double startpressure = subcalculations[SubcalculationsIndexOfSubcriticalComponent].SaturationPressures(0);
                                            Eigen::VectorXd startx = Eigen::VectorXd::Zero(2);
                                            startx(IndexOfSupercriticalComponent) = 0.0;
                                            startx(IndexOfSubcriticalComponent) = 1.0;
                                            Eigen::VectorXd startK = Eigen::VectorXd::Zero(2);
                                            startK(IndexOfSupercriticalComponent) = subcalculations[SubcalculationsIndexOfSupercriticalComponent].SaturationPressures(0) / startpressure;
                                            startK(IndexOfSubcriticalComponent) = 1.0;
                                            samplingResults = samplingForVLE(_calculation.components, temperature, IndexOfSupercriticalComponent, startx, startx, startK, startpressure, {}, endx);
                                            sampling_x = samplingResults[0];
                                            sampling_y = samplingResults[1];
                                            sampling_K = samplingResults[2];
                                            sampling_P = samplingResults[3];
                                        }
                                    }
                                    double min_conc = 1.0;
                                    double max_conc = 0.0;
                                    double min_P = sampling_P[0](0);
                                    double max_P = sampling_P[0](0);
                                    for (int i = 0; i < sampling_P.size(); i++) {
                                        if (sampling_P[i](0) < min_P) {
                                            min_P = sampling_P[i](0);
                                        }
                                        if (sampling_P[i](0) > max_P) {
                                            max_P = sampling_P[i](0);
                                        }
                                        if (sampling_x[i](IndexOfSupercriticalComponent) < min_conc) {
                                            min_conc = sampling_x[i](IndexOfSupercriticalComponent);
                                        }
                                        if (sampling_y[i](IndexOfSupercriticalComponent) < min_conc) {
                                            min_conc = sampling_y[i](IndexOfSupercriticalComponent);
                                        }
                                        if (sampling_x[i](IndexOfSupercriticalComponent) > max_conc) {
                                            max_conc = sampling_x[i](IndexOfSupercriticalComponent);
                                        }
                                        if (sampling_y[i](IndexOfSupercriticalComponent) > max_conc) {
                                            max_conc = sampling_y[i](IndexOfSupercriticalComponent);
                                        }
                                    }
                                    std::cout << min_P << std::endl;
                                    std::cout << max_P << std::endl;
                                    std::cout << "min_conc: " << min_conc << std::endl;
                                    std::cout << "max_conc: " << max_conc << std::endl;

                                    bool twoPhaseFound = false;
                                    bool zInAnySegment = false;
                                    bool AggregationStateDetermined = false;
                                    int twoPhaseIndex = -1;
                                    int zinSegmentIndex = -1;

                                    if (sampling_P.size() > 1 &&
                                        min_conc <= _calculation.overallConcentrations(i_calculation, IndexOfSupercriticalComponent) &&
                                        _calculation.overallConcentrations(i_calculation, IndexOfSupercriticalComponent) <= max_conc)
                                    {
                                        double z = _calculation.overallConcentrations(i_calculation, IndexOfSupercriticalComponent);
                                        double P_target = _calculation.targetPressure(i_calculation);

                                        for (int i = 1; i < sampling_P.size(); ++i)
                                        {
                                            double xmin_i1 = std::min(sampling_x[i - 1](IndexOfSupercriticalComponent),
                                                sampling_y[i - 1](IndexOfSupercriticalComponent));
                                            double xmax_i1 = std::max(sampling_x[i - 1](IndexOfSupercriticalComponent),
                                                sampling_y[i - 1](IndexOfSupercriticalComponent));

                                            double xmin_i = std::min(sampling_x[i](IndexOfSupercriticalComponent),
                                                sampling_y[i](IndexOfSupercriticalComponent));
                                            double xmax_i = std::max(sampling_x[i](IndexOfSupercriticalComponent),
                                                sampling_y[i](IndexOfSupercriticalComponent));

                                            double zmin = std::min(xmin_i1, xmin_i);
                                            double zmax = std::max(xmax_i1, xmax_i);

                                            if (!(zmin <= z && z <= zmax))
                                                continue;

                                            zInAnySegment = true;
                                            zinSegmentIndex = i;

                                            double P_i1 = sampling_P[i - 1](0);
                                            double P_i = sampling_P[i](0);

                                            double P_low = std::min(P_i1, P_i);
                                            double P_high = std::max(P_i1, P_i);

                                            if (P_low <= P_target && P_target <= P_high)
                                            {
                                                double xmin_seg = std::min({ xmin_i1, xmin_i });
                                                double xmax_seg = std::max({ xmax_i1, xmax_i });

                                                if (xmin_seg <= z && z <= xmax_seg)
                                                {
                                                    twoPhaseFound = true;
                                                    twoPhaseIndex = i - 1;

                                                    break;
                                                }
                                            }
                                        }
                                    }

                                    if (twoPhaseFound) {
                                        _calculation.originalConcentrations.row(i_calculation) = sampling_y[twoPhaseIndex];
                                        _calculation.originalConcentrations.row(i_calculation + numberOfCalculations) = sampling_x[twoPhaseIndex];
                                        _calculation.PartitionCoefficients.row(i_calculation) = sampling_K[twoPhaseIndex];

                                        std::cout << "Found Two-phase split and calculate Flash" << std::endl;

                                        calculatePTFlash(_calculation, i_calculation, 0.001, false, 0.0, true);

                                        if (!(_calculation.originalConcentrations.row(i_calculation + numberOfCalculations).isZero(1e-7)) && !(_calculation.originalConcentrations.row(i_calculation).isZero(1e-7)) && !(_calculation.totalPressure(i_calculation) == 0.0)) {
                                            param.temperatures_mixing.push_back(temperature);
                                            param.pressures_mixing.push_back(pressure);
                                            std::vector<double> x_vector = {};
                                            std::vector<double> y_vector = {};
                                            for (size_t i_c = 0; i_c + 1 < _calculation.components.size(); ++i_c) {
                                                x_vector.push_back(_calculation.originalConcentrations(i_calculation + numberOfCalculations, i_c));
                                                y_vector.push_back(_calculation.originalConcentrations(i_calculation, i_c));
                                            }
                                            param.concentration_x_phase_mixing.push_back(x_vector);
                                            param.concentration_y_phase_mixing.push_back(y_vector);
                                            MixtureState = "TwoPhaseSplit";
                                            AggregationStateDetermined = true;
                                        }
                                    }
                                    else if (zInAnySegment) {
                                        double P_i1 = sampling_P[zinSegmentIndex - 1](0);
                                        double P_i = sampling_P[zinSegmentIndex](0);

                                        double P_low = std::min(P_i1, P_i);
                                        double P_high = std::max(P_i1, P_i);

                                        double P_target = _calculation.targetPressure(i_calculation);

                                        if (P_target < P_low)
                                        {
                                            // single gaseous phase
                                            MixtureState = "gaseous";
                                            AggregationStateDetermined = true;
                                        }
                                        else if (P_target > P_high)
                                        {
                                            // single liquid phase
                                            MixtureState = "liquid";
                                            AggregationStateDetermined = true;
                                        }
                                    }
                                }
                            }


                            // Indices check
                            int MixtureIndex = -1;
                            double theta = 0.0; // molar proportion of the gas phase
                            double RT = R_GAS_CONSTANT * temperature;
                            if (MixtureState == "UNKNOWN") {
                                // This means that no bubble and/or dew point could be calculated in Flash procedure
                                // First calculate residual molar gibbs energy selecting the liquid-like root

                                std::cout << "Determine stable state of mixture." << std::endl;

                                _calculation.originalConcentrations.row(i_calculation) = _calculation.overallConcentrations.row(i_calculation);
                                _calculation.originalConcentrations.row(i_calculation + numberOfCalculations) = _calculation.overallConcentrations.row(i_calculation);

                                _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                                _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];

                                double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
                                double residualGibbsEnergyLiquid = 1e8;
                                if (v != 1000) {
                                    residualGibbsEnergyLiquid = 0.0;
                                    _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                                    calculateEOS(param, _calculation, i_calculation + numberOfCalculations, true, false);

                                    for (size_t i_c = 0; i_c + 1 < _calculation.components.size(); ++i_c) {
                                        residualGibbsEnergyLiquid += _calculation.originalConcentrations(i_calculation + numberOfCalculations, i_c) * RT * (_calculation.lnGammaResidual(i_calculation + numberOfCalculations, i_c) - _calculation.componentCavityVolume(i_c) / _calculation.mixtureHoleVolume(i_calculation + numberOfCalculations) * _calculation.lnGammaResidual(i_calculation + numberOfCalculations, _calculation.components.size() - 1));
                                    }
                                    double z_liquid = _calculation.totalPressure(i_calculation + numberOfCalculations) * v / RT;
                                    residualGibbsEnergyLiquid = residualGibbsEnergyLiquid - _calculation.attractivePressure(i_calculation + numberOfCalculations) * v - RT * log((v - _calculation.mixtureCavityVolume(i_calculation + numberOfCalculations)) / v) + RT * (z_liquid - 1) - RT * log(z_liquid);
                                }

                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];

                                if (vapor_P.size() > 0 && pressure <= vapor_P[0]) {
                                    double residualGibbsEnergyVapor = 1e8;
                                    v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                                    if (v != 1000) {
                                        residualGibbsEnergyVapor = 0.0;
                                        _calculation.molarVolume(i_calculation) = v;
                                        calculateEOS(param, _calculation, i_calculation, true, false);

                                        for (size_t i_c = 0; i_c + 1 < _calculation.components.size(); ++i_c) {
                                            residualGibbsEnergyVapor += _calculation.originalConcentrations(i_calculation, i_c) * RT * (_calculation.lnGammaResidual(i_calculation, i_c) - _calculation.componentCavityVolume(i_c) / _calculation.mixtureHoleVolume(i_calculation) * _calculation.lnGammaResidual(i_calculation, _calculation.components.size() - 1));
                                        }
                                        double z_vapor = _calculation.totalPressure(i_calculation) * v / RT;
                                        residualGibbsEnergyVapor = residualGibbsEnergyVapor - _calculation.attractivePressure(i_calculation) * v - RT * log((v - _calculation.mixtureCavityVolume(i_calculation)) / v) + RT * (z_vapor - 1) - RT * log(z_vapor);
                                    }

                                    if (residualGibbsEnergyVapor < residualGibbsEnergyLiquid) {
                                        MixtureState = "gaseous";
                                        theta = 1;
                                        MixtureIndex = i_calculation;
                                    }
                                    else {
                                        MixtureState = "liquid";
                                        theta = 0;
                                        MixtureIndex = i_calculation + numberOfCalculations;
                                    }
                                }
                                else {
                                    MixtureState = "liquid";
                                    theta = 0;
                                    MixtureIndex = i_calculation + numberOfCalculations;
                                }
                            }
                            else if (MixtureState == "liquid") {
                                theta = 0;
                                _calculation.originalConcentrations.row(i_calculation + numberOfCalculations) = _calculation.overallConcentrations.row(i_calculation);
                                _calculation.originalConcentrations.row(i_calculation).setZero();
                                MixtureIndex = i_calculation + numberOfCalculations;
                            }
                            else if (MixtureState == "gaseous") {
                                theta = 0;
                                _calculation.originalConcentrations.row(i_calculation) = _calculation.overallConcentrations.row(i_calculation);
                                _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).setZero();
                                MixtureIndex = i_calculation;
                            }
                            else if (MixtureState == "TwoPhaseSplit") {
                                // Mixture splits into two phases
                                theta = std::abs((_calculation.overallConcentrations(i_calculation, 0) - _calculation.originalConcentrations(i_calculation + numberOfCalculations, 0)) / (_calculation.originalConcentrations(i_calculation, 0) - _calculation.originalConcentrations(i_calculation + numberOfCalculations, 0)));
                            }

                            _calculation.mixtureCavityVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentCavityVolume;
                            _calculation.mixtureHoleVolume = _calculation.originalConcentrations.cast<double>() * _calculation.componentHoleVolume;

                            if (MixtureState == "liquid" || MixtureState == "TwoPhaseSplit") {
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation + numberOfCalculations);
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];

                                double v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
                                if (v == 1000.0) {
                                    liquid_v = returnVector[4];
                                    liquid_P = returnVector[5];
                                    v = calculatevForGivenTPn(_calculation, i_calculation + numberOfCalculations, liquid_v, liquid_P, pressure);
                                }
                                _calculation.molarVolume(i_calculation + numberOfCalculations) = v;
                                calculateEOS(param, _calculation, i_calculation + numberOfCalculations, true, true);

                                // Save h^res(T,v,z) for mixture in hM
                                _calculation.hM(i_calculation + numberOfCalculations) = _calculation.molarResidualEnthalpy(i_calculation + numberOfCalculations);
                                _calculation.hE(i_calculation + numberOfCalculations) = _calculation.hM(i_calculation + numberOfCalculations);
                                _calculation.cpM(i_calculation + numberOfCalculations) = _calculation.residualHeatCapacityAtConstantPressure(i_calculation + numberOfCalculations);
                                _calculation.cpE(i_calculation + numberOfCalculations) = _calculation.cpM(i_calculation + numberOfCalculations);
                            }

                            if (MixtureState == "gaseous" || MixtureState == "TwoPhaseSplit") {
                                std::vector<std::vector<double>> returnVector = calculateIsotherm(_calculation, i_calculation);
                                std::vector<double> liquid_v = returnVector[2];
                                std::vector<double> liquid_P = returnVector[3];
                                std::vector<double> vapor_v = returnVector[4];
                                std::vector<double> vapor_P = returnVector[5];
                                if (_calculation.calculationType == "LL") {
                                    // In the case of a (second) liquid phase, the v root should normally be found in returnVector[2] and returnVector[4] is fallback.
                                    vapor_v = returnVector[2];
                                    vapor_P = returnVector[3];
                                    liquid_v = returnVector[4];
                                    liquid_P = returnVector[5];
                                }
                                double v = calculatevForGivenTPn(_calculation, i_calculation, vapor_v, vapor_P, pressure);
                                if (v == 1000.0) {
                                    v = calculatevForGivenTPn(_calculation, i_calculation, liquid_v, liquid_P, pressure);
                                }
                                _calculation.molarVolume(i_calculation) = v;
                                calculateEOS(param, _calculation, i_calculation, true, true);

                                // Initially set mixture properties equal to residual property of mixture (later arithmetic mean of pc properties is subtracted)
                                _calculation.hM(i_calculation) = _calculation.molarResidualEnthalpy(i_calculation);
                                _calculation.hE(i_calculation) = _calculation.hM(i_calculation);
                                _calculation.cpM(i_calculation) = _calculation.residualHeatCapacityAtConstantPressure(i_calculation);
                                _calculation.cpE(i_calculation) = _calculation.cpM(i_calculation);
                            }

                            // Excess Properties // 
                            if (MixtureState != "TwoPhaseSplit") {
                                std::vector<Eigen::VectorXd> returnVector = calculateMolarResidualEnthalpyForPureComponents(_calculation, i_calculation, MixtureIndex, "excess");
                                Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = returnVector[0];
                                Eigen::VectorXd MolarResidualEnthalpiesOfPureComponents = returnVector[1];
                                Eigen::VectorXd MolarResidualCpOfPureComponents = returnVector[2];

                                _calculation.gE(MixtureIndex) = R_GAS_CONSTANT * temperature * _calculation.originalConcentrations.row(MixtureIndex).dot(LnPhiMixtureMinusLnPhiPure);

                                Eigen::VectorXd concentrationVector = _calculation.originalConcentrations.row(MixtureIndex).head(numberOfOriginalComponents).cast<double>();
                                double SumConcentrationsTimesPureComponentMolarResidualEnthalpy = concentrationVector.dot(MolarResidualEnthalpiesOfPureComponents);
                                _calculation.hE(MixtureIndex) += -1 * SumConcentrationsTimesPureComponentMolarResidualEnthalpy;
                                double SumConcentrationsTimesPureComponentcpres = concentrationVector.dot(MolarResidualCpOfPureComponents);
                                _calculation.cpE(MixtureIndex) += -1 * SumConcentrationsTimesPureComponentcpres;
                            }
                            else {
                                std::vector<Eigen::VectorXd> returnVector = calculateMolarResidualEnthalpyForPureComponents(_calculation, i_calculation, i_calculation, "excess");
                                Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = returnVector[0];
                                Eigen::VectorXd MolarResidualEnthalpiesOfPureComponents = returnVector[1];
                                Eigen::VectorXd MolarResidualCpOfPureComponents = returnVector[2];

                                _calculation.gE(i_calculation) = theta * R_GAS_CONSTANT * temperature * _calculation.originalConcentrations.row(i_calculation).dot(LnPhiMixtureMinusLnPhiPure);

                                Eigen::VectorXd concentrationVector = _calculation.originalConcentrations.row(i_calculation).head(numberOfOriginalComponents).cast<double>();
                                double SumConcentrationsTimesPureComponentMolarResidualEnthalpy = concentrationVector.dot(MolarResidualEnthalpiesOfPureComponents);
                                _calculation.hE(i_calculation) += -1 * SumConcentrationsTimesPureComponentMolarResidualEnthalpy;
                                _calculation.hE(i_calculation) *= theta;
                                double SumConcentrationsTimesPureComponentcpres = concentrationVector.dot(MolarResidualCpOfPureComponents);
                                _calculation.cpE(i_calculation) += -1 * SumConcentrationsTimesPureComponentcpres;
                                _calculation.cpE(i_calculation) *= theta;


                                returnVector = calculateMolarResidualEnthalpyForPureComponents(_calculation, i_calculation, i_calculation + numberOfCalculations, "excess");
                                LnPhiMixtureMinusLnPhiPure = returnVector[0];
                                MolarResidualEnthalpiesOfPureComponents = returnVector[1];
                                MolarResidualCpOfPureComponents = returnVector[2];

                                _calculation.gE(i_calculation + numberOfCalculations) = (1 - theta) * R_GAS_CONSTANT * temperature * _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).dot(LnPhiMixtureMinusLnPhiPure);

                                concentrationVector = _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).head(numberOfOriginalComponents).cast<double>();
                                SumConcentrationsTimesPureComponentMolarResidualEnthalpy = concentrationVector.dot(MolarResidualEnthalpiesOfPureComponents);
                                _calculation.hE(i_calculation + numberOfCalculations) += -1 * SumConcentrationsTimesPureComponentMolarResidualEnthalpy;
                                _calculation.hE(i_calculation + numberOfCalculations) *= (1 - theta);
                                SumConcentrationsTimesPureComponentcpres = concentrationVector.dot(MolarResidualCpOfPureComponents);
                                _calculation.cpE(i_calculation + numberOfCalculations) += -1 * SumConcentrationsTimesPureComponentcpres;
                                _calculation.cpE(i_calculation + numberOfCalculations) *= (1 - theta);
                            }

                            ///// Mixing Properties /////
                            if (MixtureState != "TwoPhaseSplit") {
                                std::vector<Eigen::VectorXd> returnVector = calculateMolarResidualEnthalpyForPureComponents(_calculation, i_calculation, MixtureIndex, "mixing");
                                Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = returnVector[0];
                                Eigen::VectorXd MolarResidualEnthalpiesOfPureComponents = returnVector[1];
                                Eigen::VectorXd MolarResidualCpOfPureComponents = returnVector[2];

                                ///// Excess Gibbs Free Energy /////
                                _calculation.gM(MixtureIndex) = R_GAS_CONSTANT * temperature * _calculation.originalConcentrations.row(MixtureIndex).dot(LnPhiMixtureMinusLnPhiPure);

                                Eigen::VectorXd concentrationVector = _calculation.originalConcentrations.row(MixtureIndex).head(numberOfOriginalComponents).cast<double>();
                                double SumConcentrationsTimesPureComponentMolarResidualEnthalpy = concentrationVector.dot(MolarResidualEnthalpiesOfPureComponents);
                                _calculation.hM(MixtureIndex) += -1 * SumConcentrationsTimesPureComponentMolarResidualEnthalpy;
                                double SumConcentrationsTimesPureComponentcpres = concentrationVector.dot(MolarResidualCpOfPureComponents);
                                _calculation.cpM(MixtureIndex) += -1 * SumConcentrationsTimesPureComponentcpres;
                            }
                            else {
                                std::vector<Eigen::VectorXd> returnVector = calculateMolarResidualEnthalpyForPureComponents(_calculation, i_calculation, i_calculation, "mixing");
                                Eigen::VectorXd LnPhiMixtureMinusLnPhiPure = returnVector[0];
                                Eigen::VectorXd MolarResidualEnthalpiesOfPureComponents = returnVector[1];
                                Eigen::VectorXd MolarResidualCpOfPureComponents = returnVector[2];

                                ///// Excess Gibbs Free Energy /////
                                _calculation.gM(i_calculation) = theta * R_GAS_CONSTANT * temperature * _calculation.originalConcentrations.row(i_calculation).dot(LnPhiMixtureMinusLnPhiPure);

                                Eigen::VectorXd concentrationVector = _calculation.originalConcentrations.row(i_calculation).head(numberOfOriginalComponents).cast<double>();
                                double SumConcentrationsTimesPureComponentMolarResidualEnthalpy = concentrationVector.dot(MolarResidualEnthalpiesOfPureComponents);
                                _calculation.hM(i_calculation) += -1 * SumConcentrationsTimesPureComponentMolarResidualEnthalpy;
                                _calculation.hM(i_calculation) *= theta;
                                double SumConcentrationsTimesPureComponentcpres = concentrationVector.dot(MolarResidualCpOfPureComponents);
                                _calculation.cpM(i_calculation) += -1 * SumConcentrationsTimesPureComponentcpres;
                                _calculation.cpM(i_calculation) *= theta;

                                returnVector = calculateMolarResidualEnthalpyForPureComponents(_calculation, i_calculation, i_calculation + numberOfCalculations, "mixing");
                                LnPhiMixtureMinusLnPhiPure = returnVector[0];
                                MolarResidualEnthalpiesOfPureComponents = returnVector[1];
                                MolarResidualCpOfPureComponents = returnVector[2];

                                ///// Excess Gibbs Free Energy /////
                                _calculation.gM(i_calculation + numberOfCalculations) = (1 - theta) * R_GAS_CONSTANT * temperature * _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).dot(LnPhiMixtureMinusLnPhiPure);

                                concentrationVector = _calculation.originalConcentrations.row(i_calculation + numberOfCalculations).head(numberOfOriginalComponents).cast<double>();
                                SumConcentrationsTimesPureComponentMolarResidualEnthalpy = concentrationVector.dot(MolarResidualEnthalpiesOfPureComponents);
                                _calculation.hM(i_calculation + numberOfCalculations) += -1 * SumConcentrationsTimesPureComponentMolarResidualEnthalpy;
                                _calculation.hM(i_calculation + numberOfCalculations) *= (1 - theta);
                                SumConcentrationsTimesPureComponentcpres = concentrationVector.dot(MolarResidualCpOfPureComponents);
                                _calculation.cpM(i_calculation + numberOfCalculations) += -1 * SumConcentrationsTimesPureComponentcpres;
                                _calculation.cpM(i_calculation + numberOfCalculations) *= (1 - theta);
                            }
                        }
                    }
                }
            }
            else {
                // recalculate combinatorial term if needed
                if ((param.sw_alwaysCalculateSizeRelatedParameters == 0 && n_ex == 3) || param.sw_alwaysCalculateSizeRelatedParameters == 1 || param.sw_reloadConcentrations == 1 || param.sw_reloadReferenceConcentrations == 1) {
                    calculateLnGammaCombinatorial(param, calculations[calculationIndex]);;
                }

#ifdef MEASURE_TIME
                calculateCombinatorial_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateCombinatorial_last).count());
                std::chrono::high_resolution_clock::time_point calculateResidual_last = std::chrono::high_resolution_clock::now();
#endif

                // calculate residual part
                calculateLnGammaResidual(param, calculations[calculationIndex]);

#ifdef MEASURE_TIME
                calculateResidual_total_ms += (const unsigned long)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - calculateResidual_last).count());
#endif

                calculations[calculationIndex].lnGammaTotal = calculations[calculationIndex].lnGammaCombinatorial + calculations[calculationIndex].lnGammaResidual;
#ifdef PRINT_DEBUG_INFO
                for (int i_concentration = 0; i_concentration < calculations[calculationIndex].originalNumberOfCalculations; i_concentration++) {
                    for (int i_component = 0; i_component < calculations[calculationIndex].components.size(); i_component++) {
                        display("lnGammaTotal_" + std::to_string(calculationIndex) + "_" + std::to_string(i_concentration) + "_" + std::to_string(i_component) + \
                            ": " + std::to_string(calculations[calculationIndex].lnGammaTotal(i_concentration, i_component)) + "\n");
                    }
                }
#endif

                bool calculateSolvationEnergies = param.dGsolv_E_gas.size() > 0;
                if (calculateSolvationEnergies) {
                    double kcalPerMol_per_Hartree = 2625.499639479 / 4.184;
                    double reference_pressure = 101325; // Pa = 1 atm;
                    std::vector<int> atomicNumbersWithout_dGsolv_tau = std::vector<int>();
                    double approximate_dGsolv_tau = 0.0262; // median of other values
                    for (int i_concentration = 0; i_concentration < calculations[calculationIndex].originalNumberOfCalculations; i_concentration++) {
                        if (calculations[calculationIndex].referenceStateType[i_concentration] == 4) {
                            int i_solvent_component = -1;
                            for (int i_component = 0; i_component < calculations[calculationIndex].components.size(); i_component++) {
                                if (calculations[calculationIndex].concentrations[i_concentration][i_component] == 1.0f) {
                                    i_solvent_component = i_component;
                                    break;
                                }
                            }
                            for (int i_component = 0; i_component < calculations[calculationIndex].components.size(); i_component++) {
                                double dGsolv = 0.0;
                                if (calculations[calculationIndex].concentrations[i_concentration][i_component] == 0.0f) {

                                    double RT = R_GAS_CONSTANT * calculations[calculationIndex].temperatures[i_concentration];
                                    double molar_volume_ideal_gas = RT / reference_pressure;
                                    double RT_kcalPerMol = RT / (1000 * 4.184);

                                    // all energies calculated below this line are in kcal/mol
                                    double E_vdw = 0.0;
                                    segmentTypeCollection segments = calculations[calculationIndex].components[i_component]->segments;
                                    std::unordered_map<int, double> areasByAtomicNumber;
                                    for (int i_segment = 0; i_segment < segments.size(); i_segment++) {
                                        int AN = segments.SegmentTypeAtomicNumber[i_segment];

                                        if (areasByAtomicNumber.find(AN) == areasByAtomicNumber.end())
                                            areasByAtomicNumber[AN] = 0.0;

                                        areasByAtomicNumber[AN] += segments.SegmentTypeAreas[i_segment][0];
                                    }

                                    for (auto& it : areasByAtomicNumber) {
                                        double this_atom_dGsolv_tau = std::abs(param.dGsolv_tau[it.first]);
                                        if (this_atom_dGsolv_tau == 0.0) {
                                            this_atom_dGsolv_tau = approximate_dGsolv_tau;
                                            atomicNumbersWithout_dGsolv_tau.push_back(it.first);
                                        }

                                        E_vdw += this_atom_dGsolv_tau * it.second;
                                    }
                                    double referenceStateCorrection = RT_kcalPerMol * log(molar_volume_ideal_gas / (calculations[calculationIndex].components[i_solvent_component]->molarVolumeAt25C / 1E6));

                                    double E_diel = (calculations[calculationIndex].components[i_component]->epsilonInfinityTotalEnergy - param.dGsolv_E_gas[i_component]) * kcalPerMol_per_Hartree;
                                    double mu_liquid = RT_kcalPerMol * calculations[calculationIndex].lnGammaTotal(i_concentration, i_component);
                                    double E_ring = param.dGsolv_omega_ring * param.dGsolv_numberOfAtomsInRing[i_component];
                                    dGsolv = E_diel + mu_liquid - E_vdw - E_ring - referenceStateCorrection - param.dGsolv_eta;
#ifdef PRINT_DEBUG_INFO
                                    display("dGsolv_1_" + std::to_string(calculationIndex) + ": " + std::to_string(dGsolv) + "\n");
                                    display("E_diel: " + std::to_string(E_diel) + "\n");
                                    display("mu_liquid: " + std::to_string(mu_liquid) + "\n");
                                    display("E_vdw: " + std::to_string(E_vdw) + "\n");
                                    display("E_ring: " + std::to_string(E_ring) + "\n");
                                    display("referenceStateCorrection: " + std::to_string(referenceStateCorrection) + "\n");
                                    display("dGsolv_eta: " + std::to_string(param.dGsolv_eta) + "\n");
#endif

                                }

                                calculations[calculationIndex].dGsolv(i_concentration, i_component) = dGsolv;
                            }
                        }
                    }
                    if (atomicNumbersWithout_dGsolv_tau.size() > 0) {
                        sort(atomicNumbersWithout_dGsolv_tau.begin(), atomicNumbersWithout_dGsolv_tau.end());
                        atomicNumbersWithout_dGsolv_tau.erase(unique(atomicNumbersWithout_dGsolv_tau.begin(), atomicNumbersWithout_dGsolv_tau.end()), atomicNumbersWithout_dGsolv_tau.end());
                        std::string ANs = "";
                        for (int i_AN = 0; i_AN < atomicNumbersWithout_dGsolv_tau.size(); i_AN++) {
                            ANs += std::to_string(atomicNumbersWithout_dGsolv_tau[i_AN]) + ";";
                        }
                        if (param.sw_dGsolv_calculation_strict == 1) {
                            throw std::runtime_error("For the following atomic numbers not all parameters are available for the calculation of solvation energies: " + ANs);
                        }
                        else {
                            warnings.push_back(" - For the following atomic numbers not all parameters are available for the calculation of solvation energies, an estimate was used: " + ANs);
                        }
                    }
                }
            }
            });
    }
    e.rethrow();
}

void calculateNumericalDerivative(calculation _calculation, double Delta, std::vector<int> actualConcentrationIndicesToCalculate) {

    calculation subcalc_lowerlimit = _calculation;
    subcalculations.emplace_back(subcalc_lowerlimit);
    int index_subcalc_lowerlimit = int(subcalculations.size() - 1);
    calculation subcalc_upperlimit = _calculation;
    subcalculations.emplace_back(subcalc_upperlimit);
    int index_subcalc_upperlimit = int(subcalculations.size() - 1);

    for (int index = 0; index < _calculation.temperatures.size(); index++) {
        subcalculations[index_subcalc_lowerlimit].temperatures[index] -= Delta;
        subcalculations[index_subcalc_upperlimit].temperatures[index] += Delta;
    }

    for (int index = 0; index < _calculation.TauTemperatures.size(); index++) {
        subcalculations[index_subcalc_lowerlimit].TauTemperatures[index] -= Delta;
        subcalculations[index_subcalc_upperlimit].TauTemperatures[index] += Delta;
    }

    int numberOfStatePoints = int(_calculation.originalNumberOfCalculations / 2);

    for (int i_calculation = 0; i_calculation < int(_calculation.originalNumberOfCalculations); i_calculation++) {

        //subcalculations[index_subcalc_lowerlimit].originalConcentrations.row(i_calculation) = subcalculations[index_subcalc_lowerlimit].overallConcentrations.row(i_calculation);
        //subcalculations[index_subcalc_lowerlimit].originalConcentrations.row(i_calculation + numberOfStatePoints) = subcalculations[index_subcalc_lowerlimit].overallConcentrations.row(i_calculation);
        //subcalculations[index_subcalc_upperlimit].originalConcentrations.row(i_calculation) = subcalculations[index_subcalc_upperlimit].overallConcentrations.row(i_calculation);
        //subcalculations[index_subcalc_upperlimit].originalConcentrations.row(i_calculation + numberOfStatePoints) = subcalculations[index_subcalc_upperlimit].overallConcentrations.row(i_calculation);

        int actualConcentrationIndex = subcalculations[index_subcalc_lowerlimit].actualConcentrationIndices[i_calculation];

        auto it = std::find(actualConcentrationIndicesToCalculate.begin(), actualConcentrationIndicesToCalculate.end(), actualConcentrationIndex);
        if (it != actualConcentrationIndicesToCalculate.end()) {
            // calculation for lower limit
            std::vector<std::vector<double>> returnVector = calculateIsotherm(subcalculations[index_subcalc_lowerlimit], i_calculation);

            std::vector<double> v = returnVector[0];
            std::vector<double> P = returnVector[1];
            std::vector<double> liquid_v = returnVector[2];
            std::vector<double> liquid_P = returnVector[3];
            std::vector<double> vapor_v = returnVector[4];
            std::vector<double> vapor_P = returnVector[5];
            std::vector<double> P_repulsive = returnVector[6];
            std::vector<double> P_attractive = returnVector[7];

            double pressure = subcalculations[index_subcalc_lowerlimit].targetPressure(i_calculation);

            std::vector<double> v_vector;
            std::vector<double> P_vector;
            if (i_calculation < numberOfStatePoints) {
                v_vector = vapor_v;
                P_vector = vapor_P;
            }
            else {
                v_vector = liquid_v;
                P_vector = liquid_P;
            }


            double volume = calculatevForGivenTPn(subcalculations[index_subcalc_lowerlimit], i_calculation, v_vector, P_vector, pressure);
            subcalculations[index_subcalc_lowerlimit].molarVolume(i_calculation) = volume;
            calculateEOS(param, subcalculations[index_subcalc_lowerlimit], i_calculation);


            // calculation for upper limit
            returnVector = calculateIsotherm(subcalculations[index_subcalc_upperlimit], i_calculation);

            v = returnVector[0];
            P = returnVector[1];
            liquid_v = returnVector[2];
            liquid_P = returnVector[3];
            vapor_v = returnVector[4];
            vapor_P = returnVector[5];
            P_repulsive = returnVector[6];
            P_attractive = returnVector[7];

            if (i_calculation < numberOfStatePoints) {
                v_vector = vapor_v;
                P_vector = vapor_P;
            }
            else {
                v_vector = liquid_v;
                P_vector = liquid_P;
            }

            pressure = subcalculations[index_subcalc_upperlimit].targetPressure(i_calculation);

            volume = calculatevForGivenTPn(subcalculations[index_subcalc_upperlimit], i_calculation, v_vector, P_vector, pressure);
            subcalculations[index_subcalc_upperlimit].molarVolume(i_calculation) = volume;
            calculateEOS(param, subcalculations[index_subcalc_upperlimit], i_calculation);
        }
    }
}