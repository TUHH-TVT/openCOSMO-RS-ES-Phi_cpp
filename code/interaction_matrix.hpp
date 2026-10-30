/*
    c++ implementation of openCOSMO-RS including multiple segment descriptors
    @author: Simon Mueller, 2022
*/


#pragma once

#include "helper_functions.hpp"
#include <stdexcept>
#include <immintrin.h>// always include this as at least SSE3 is required
#include <cmath>
#include <map>
#include <iostream>
// returns lower left triangular matrices, this is because this way the matrix is accessed in sequential order in memory
void calculateInteractionMatrix(segmentTypeCollection& segments, MatrixCalcType& A_int, MatrixCalcType& A_int_PDET, MatrixCalcType& A_int_SecondPDET, std::vector<Eigen::MatrixXd>& partialInteractionMatrices, parameters& param, double temperature) {

    const int numberOfSegments = int(segments.size());
    const int indexOfHole = numberOfSegments - 1;

    double sigmai = 0.0;
    double sigmaj = 0.0;
    double alphai = 0.0;
    double alphaj = 0.0;
    double sigmaCorri = 0.0;
    double sigmaCorrj = 0.0;
    double sigmaTransi = 0.0;
    double sigmaTransj = 0.0;
    double sigmaMFij = 0.0;

    double const sigmaHB = param.SigmaHB;
    double const minus_sigmaHB = -1 * sigmaHB;

    std::vector<double> ChargeRaster = param.ChargeRaster;


    // neutral - neutral interactions -------------------------------------------------------------------------------------------------
    // misfit 
    double const misfit_prefactor = param.Aeff * param.alpha * 0.5;

    double ionizationPotentialCorrection = 0;

    // hb
    double CHB_T = 0;
    double CHB_T_PDETemperature = 0;
    double CHB_T_SecondPDETemperature = 0;
    double buffdb1 = 1.0 - param.CHBT + param.CHBT * (298.15 / (temperature));
    if (buffdb1 > 0) {
        // CHB_T = param.exp_param.at("k") * param.CHB * buffdb1;
        CHB_T = param.CHB * buffdb1;
        CHB_T_PDETemperature = -1 * param.CHB * param.CHBT * (298.15 / (pow(temperature, 2)));
        CHB_T_SecondPDETemperature = 2 * param.CHB * param.CHBT * (298.15 / (pow(temperature, 3)));
    }
    // prefactor ions
    double const ion_prefactor = param.Aeff * param.CHB;

    // temperature dependence of the ion interaction energies: g(T) = 1 - w + w * 298.15/T,
    // optionally + w2 * ln(T/298.15) and per ion class; no key, no temperature dependence
    auto tdep = [&param, temperature](const char* key, const char* fallback, const char* curvature) -> double {
        auto it = param.exp_param.find(key);
        if (it == param.exp_param.end() && fallback != nullptr) it = param.exp_param.find(fallback);
        auto ic = param.exp_param.find(curvature);
        if (it == param.exp_param.end() && ic == param.exp_param.end()) return 1.0;
        double const w = (it == param.exp_param.end()) ? 0.0 : it->second;
        double g = 1.0 - w + w * (298.15 / temperature);
        if (ic != param.exp_param.end()) g += ic->second * std::log(temperature / 298.15);
        return std::max(0.0, g);
    };
    double const g_ion_solvent = tdep("tdep_ion_solvent", nullptr, "tdep2_ion_solvent");
    double const g_ion_ion = tdep("tdep_ion_ion", nullptr, "tdep2_ion_ion");
    double const g_cat_solvent = tdep("tdep_cat_solvent", "tdep_ion_solvent", "tdep2_ion_solvent");
    double const g_hal_solvent = tdep("tdep_hal_solvent", "tdep_ion_solvent", "tdep2_ion_solvent");
    double const g_poly_solvent = tdep("tdep_poly_solvent", "tdep_ion_solvent", "tdep2_ion_solvent");

    double const hb_prefactor = param.Aeff * CHB_T;
    double const hb_prefactor_PDETemperatur = param.Aeff * CHB_T_PDETemperature;
    double const hb_prefactor_SecondPDETemperatur = param.Aeff * CHB_T_SecondPDETemperature;
    double temp = 0;

    double val = 0;
    double val2 = 0; // added for partial derivative with respect to temperature
    double val3 = 0; // added for second partial derivative with respect to temperature
    double vdw_val = 0;

    MatrixCalcType binaryComponentDispersionEnergies;
    MatrixCalcType binaryComponentDispersionEnergies_PDETemperature;
    MatrixCalcType binaryComponentDispersionEnergies_SecondPDETemperature;

    bool withDerivative = false;

    if (param.sw_phi == 1) {
        int numberOfComponents = param.phi_param.rows();

        // exploit the fact that multiplication is commutative and the matrix is symmetric
        // include hole, interaction energy is zero as expected
        binaryComponentDispersionEnergies = MatrixCalcType::Zero(numberOfComponents + 1, numberOfComponents + 1);
        if (A_int_PDET.rows() == numberOfSegments && A_int_PDET.cols() == numberOfSegments) {
            binaryComponentDispersionEnergies_PDETemperature = MatrixCalcType::Zero(numberOfComponents + 1, numberOfComponents + 1);
            binaryComponentDispersionEnergies_SecondPDETemperature = MatrixCalcType::Zero(numberOfComponents + 1, numberOfComponents + 1);
            withDerivative = true;
        }
        for (int i = 0; i < numberOfComponents; i++) {
            double dispTi = param.phi_param(i, 2) * 100;
            double dispT0i = param.phi_param(i, 1) * 4184;
            //double dispi = dispT0i * exp(1 - pow(temperature / dispTi, param.exp_param.at("k")));
            double expi = (1 - exp(-dispTi / temperature));
            double dispi = dispT0i * expi;
            for (int j = i; j < numberOfComponents; j++) {
                double dispTj = param.phi_param(j, 2) * 100;
                double dispT0j = param.phi_param(j, 1) * 4184;
                //double dispj = dispT0j * exp(1 - pow(temperature / dispTj, param.exp_param.at("k")));
                double expj = (1 - exp(-dispTj / temperature));
                double dispj = dispT0j * expj;
                double val = -1 * 0.5 * pow((dispj * dispi), 0.5);
                binaryComponentDispersionEnergies(i, j) = val;
                binaryComponentDispersionEnergies(j, i) = val;
                if (withDerivative) {
                    double num = dispT0i * dispT0j * (dispTi * exp(dispTj / temperature) + dispTj * exp(dispTi / temperature) - dispTj - dispTi) * exp(-dispTj / temperature - dispTi / temperature);
                    double den = 2 * pow(temperature, 2) * pow((dispj * dispi), 0.5);
                    double val2 = 0.5 * num / den;
                    binaryComponentDispersionEnergies_PDETemperature(i, j) = val2;
                    binaryComponentDispersionEnergies_PDETemperature(j, i) = val2;

                    //double nummod = dispTi * exp(-dispTi / temperature) + dispTj * exp(-dispTj / temperature) + (-dispTj - dispTi) * exp(-dispTj / temperature - dispTi / temperature);
                    double nummod = num / (dispT0i * dispT0j);
                    double denmod = den / sqrt(dispT0i * dispT0j);
                    double bracketForMultiplicator2 = dispTi * exp(dispTj / temperature) - dispTi + dispTj * exp(dispTi / temperature) - dispTj;
                    double dden_dT = (4 * temperature) * sqrt(expi * expj) + pow(temperature, 2) / sqrt(expi * expj) * (-1 * (dispTi / pow(temperature, 2) * exp(-1 * dispTi / temperature)) * expj - expi * (dispTj / pow(temperature, 2) * exp(-1 * dispTj / temperature)));
                    double dnum_dT = exp(-1 * dispTj / temperature - dispTi / temperature) * (-1 * (dispTi * dispTj) / pow(temperature, 2) * (exp(dispTj / temperature) + exp(dispTi / temperature)) + bracketForMultiplicator2 * ((dispTj + dispTi) / pow(temperature, 2)));

                    double val3 = 0.5 * sqrt(dispT0i * dispT0j) * (dnum_dT * denmod - nummod * dden_dT) / pow(denmod, 2);

                    binaryComponentDispersionEnergies_SecondPDETemperature(i, j) = val3;
                    binaryComponentDispersionEnergies_SecondPDETemperature(j, i) = val3;
                }
            }
        }
    }
    else if (param.sw_phi == 0) {
        if (param.exp_param.at("with_dispersion") == 1) {
            int numberOfComponents = molecules.size();

            // exploit the fact that multiplication is commutative and the matrix is symmetric
            // include hole, interaction energy is zero as expected
            binaryComponentDispersionEnergies = MatrixCalcType::Zero(numberOfComponents, numberOfComponents);
            for (int i = 0; i < numberOfComponents; i++) {
                double dispTi;
                double dispT0i;
                if (i == 0) {
                    dispTi = param.exp_param.at("deltamT_comp1") * 100;
                    dispT0i = param.exp_param.at("deltam0_comp1") * 4184;
                }
                else if (i == 1) {
                    dispTi = param.exp_param.at("deltamT_comp2") * 100;
                    dispT0i = param.exp_param.at("deltam0_comp2") * 4184;
                }
                //double dispi = dispT0i * exp(1 - pow(temperature / dispTi, param.exp_param.at("k")));
                double expi = (1 - exp(-dispTi / temperature));
                double dispi = dispT0i * expi;
                for (int j = i; j < numberOfComponents; j++) {
                    double dispTj;
                    double dispT0j;
                    if (j == 0) {
                        dispTj = param.exp_param.at("deltamT_comp1") * 100;
                        dispT0j = param.exp_param.at("deltam0_comp1") * 4184;
                    }
                    else if (j == 1) {
                        dispTj = param.exp_param.at("deltamT_comp2") * 100;
                        dispT0j = param.exp_param.at("deltam0_comp2") * 4184;
                    }
                    //double dispj = dispT0j * exp(1 - pow(temperature / dispTj, param.exp_param.at("k")));
                    double expj = (1 - exp(-dispTj / temperature));
                    double dispj = dispT0j * expj;
                    double val = -1 * 0.5 * pow((dispj * dispi), 0.5);
                    binaryComponentDispersionEnergies(i, j) = val;
                    binaryComponentDispersionEnergies(j, i) = val;
                }
            }

        }
    }

    int UpperBoundIndexForNeutralComponents = std::max(segments.upperBoundIndexForGroup[0], segments.upperBoundIndexForGroup[1]);
    UpperBoundIndexForNeutralComponents = std::max(UpperBoundIndexForNeutralComponents, segments.upperBoundIndexForGroup[2]);
    UpperBoundIndexForNeutralComponents = std::max(UpperBoundIndexForNeutralComponents, segments.upperBoundIndexForGroup[7]);

    for (int i = segments.lowerBoundIndexForGroup[0]; i < UpperBoundIndexForNeutralComponents; i++) {

        bool i_is_segment_on_water = i >= segments.lowerBoundIndexForGroup[2] && i < segments.upperBoundIndexForGroup[2];
        sigmai = segments.SegmentTypeSigma[i];
        if (param.sw_electrolyte == 1) alphai = segments.SegmentTypeAtomicPolariz[i];

        if (param.sw_misfit > 0) {
            sigmaCorri = segments.SegmentTypeSigmaCorr[i];
            sigmaTransi = sigmaCorri - 0.816 * sigmai;
        }

        for (int j = i; j < UpperBoundIndexForNeutralComponents; j++) {

            val2 = 0;
            val3 = 0;

            sigmaj = segments.SegmentTypeSigma[j];
            sigmaMFij = sigmai + sigmaj;
            if (param.sw_electrolyte == 1) alphaj = segments.SegmentTypeAtomicPolariz[j];

            if (param.sw_misfit > 0) {
                sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                if (segments.SegmentTypeAtomicNumber[i] == 9 || segments.SegmentTypeAtomicNumber[j] == 9) {
                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj)) + param.E_F_corr;
                }
            }
            else {
                val = misfit_prefactor * sigmaMFij * sigmaMFij;
            }


            if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {

                if (segments.SegmentTypeHBtype[i] == 1 && segments.SegmentTypeHBtype[j] == 2) {

                    val += hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);

                    if (withDerivative) {
                        val2 = hb_prefactor_PDETemperatur * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                        val3 = hb_prefactor_SecondPDETemperatur * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                    }
                }
                else if (segments.SegmentTypeHBtype[i] == 2 && segments.SegmentTypeHBtype[j] == 1) {
                    throw std::runtime_error("This should not happen. Wrong assumption calculating interaction matrix. 1");
                }

            }
            else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                if (segments.SegmentTypeHBtype[i] == 2 && segments.SegmentTypeHBtype[j] == 1) {

                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);

                    if (withDerivative) {
                        val2 = hb_prefactor_PDETemperatur * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                        val3 = hb_prefactor_SecondPDETemperatur * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                    }
                }
                else if (segments.SegmentTypeHBtype[i] == 1 && segments.SegmentTypeHBtype[j] == 2) {
                    throw std::runtime_error("This should not happen. Wrong assumption calculating interaction matrix. 2");
                }

            }

            if (param.sw_phi == 1) {
                int sigmaiMolIndex = segments.SegmentTypeMoleculeIndex[i];
                int sigmajMolIndex = segments.SegmentTypeMoleculeIndex[j];

                val += binaryComponentDispersionEnergies(sigmaiMolIndex, sigmajMolIndex);
                if (withDerivative) {
                    val2 += binaryComponentDispersionEnergies_PDETemperature(sigmaiMolIndex, sigmajMolIndex);
                    val3 += binaryComponentDispersionEnergies_SecondPDETemperature(sigmaiMolIndex, sigmajMolIndex);
                }
            }
            else if (param.sw_phi == 0) {
                if (param.exp_param.at("with_dispersion") == 1) {
                    int sigmaiMolIndex = segments.SegmentTypeMoleculeIndex[i];
                    int sigmajMolIndex = segments.SegmentTypeMoleculeIndex[j];

                    val += binaryComponentDispersionEnergies(sigmaiMolIndex, sigmajMolIndex);
                }
            }

            vdw_val = 0.0;

            if (param.sw_SR_polarizabilities > 0) {

                if (param.sw_SR_polarizabilities == 7) {
                    ionizationPotentialCorrection = 1 / ioniz_potential_ev[segments.SegmentTypeAtomicNumber[i]] / ioniz_potential_ev[segments.SegmentTypeAtomicNumber[j]];
                    vdw_val = param.Aeff * param.m_vdW * sqrt(alphai * alphaj * ionizationPotentialCorrection);
                }
                else if (param.sw_SR_polarizabilities == 8) {
                    ionizationPotentialCorrection = 1 / (ioniz_potential_ev[segments.SegmentTypeAtomicNumber[i]] + ioniz_potential_ev[segments.SegmentTypeAtomicNumber[j]]);
                    vdw_val = param.Aeff * param.m_vdW * sqrt(alphai * alphaj) * ionizationPotentialCorrection;
                }
                else {
                    vdw_val = param.Aeff * param.m_vdW * sqrt(alphai * alphaj);
                }
            }

            // always j >= i
            val -= vdw_val;
            A_int(j, i) = static_cast<calcType>(val);
            if (withDerivative) {
                A_int_PDET(j, i) = static_cast<calcType>(val2);
                A_int_SecondPDET(j, i) = static_cast<calcType>(val3);
            }
        }
    }


    // COSMO-RS-ES+ equations start here 	------------------------------------------------------
    if (param.sw_electrolyte == 1) {

        // ion - organic solvent energies, scaled by g(T)
        double const msf_cP_org = param.msf_cP_org * g_cat_solvent;
        double const msf_aP_org = param.msf_aP_org * g_poly_solvent;
        double const msf_aM_org = param.msf_aM_org * g_hal_solvent;
        double const msf_cM_org = param.msf_cM_org * g_cat_solvent;

        double const fHB_cP_org = param.fHB_cP_org * g_cat_solvent;
        double const fHB_aP_XH = param.fHB_aP_XH * g_poly_solvent;
        double const fHB_A_org = param.fHB_A_org * g_ion_solvent;
        double const fHB_cM_org = param.fHB_cM_org * g_cat_solvent;
        double const fHB_aM_org = param.fHB_aM_org * g_hal_solvent;
        // C1 is not scaled
        double const C1 = param.C1;
        //double const DCO = param.DCO;

        // dispersion for ion-containing pairs (sw_phi == 1)
        auto ion_pair_dispersion = [&](int i_, int j_) -> double {
            if (param.sw_phi != 1) return 0.0;
            return binaryComponentDispersionEnergies(segments.SegmentTypeMoleculeIndex[i_],
                                                     segments.SegmentTypeMoleculeIndex[j_]);
        };

        // organic molecule - -----------------------------------------------------------------------------------------------------------------
        for (int i = segments.lowerBoundIndexForGroup[0]; i < segments.upperBoundIndexForGroup[1]; i++) {

            sigmai = segments.SegmentTypeSigma[i];
            if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                sigmaCorri = segments.SegmentTypeSigmaCorr[i];
                sigmaTransi = sigmaCorri - 0.816 * sigmai;
            }

            // organic molecule - monoatomic cation interactions ------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[3]; j < segments.upperBoundIndexForGroup[3]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit
                sigmaMFij = sigmai + sigmaj;

                // basic COSMO-RS
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cM_org * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cM_org * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // specific ion equations			
                if (sigmai > C1) {
                    val += fHB_cM_org * ion_prefactor * (sigmai - C1) * (sigmaj);// * (1 + DCO * std::abs(sigmaj));
                }

                // write both triangles
                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // organic molecule - polyatomic cation interactions ------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[4]; j < segments.upperBoundIndexForGroup[4]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cP_org * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cP_org * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // hydrogen bonding (basic COSMORS)
                if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {

                    val += hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);

                }
                else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                    if (segments.SegmentTypeAtomicNumber[j] == 108) {
                        val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                    }
                    else if (segments.SegmentTypeAtomicNumber[j] > 100 || segments.SegmentTypeAtomicNumber[j] == 7 || segments.SegmentTypeAtomicNumber[j] == 15 || segments.SegmentTypeAtomicNumber[j] == 16) {
                        val += fHB_cP_org * ion_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                    }
                    else {
                        val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                    }

                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // organic molecule - monoatomic anion interactions -------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[5]; j < segments.upperBoundIndexForGroup[5]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                // basic COSMO-RS
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_aM_org * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_aM_org * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // ion specific equations
                if (sigmai < minus_sigmaHB && sigmaj > 0.0) {
                    val += fHB_aM_org * ion_prefactor * sigmaj * (sigmai + sigmaHB);
                }
                // if (sigmai < minus_sigmaHB && segments.SegmentTypeAtomicNumber[i] > 106) {
                    // val += fHB_aM_org * ion_prefactor * sigmaj * (sigmai + sigmaHB);
                // }
                // else if (sigmai < minus_sigmaHB && sigmaj > sigmaHB && segments.SegmentTypeAtomicNumber[i] != 106) {
                    // val += fHB_A_org * ion_prefactor * (sigmai + sigmaHB) * (sigmaj - sigmaHB);
                // }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // organic molecule - polyatomic anion interactions -------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[6]; j < segments.upperBoundIndexForGroup[6]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_aP_org * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_aP_org * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // ion specific equations
                if (sigmai < minus_sigmaHB && segments.SegmentTypeAtomicNumber[i] > 106 && sigmaj > sigmaHB && segments.SegmentTypeAtomicNumber[j] == 8) {
                    val += fHB_aP_XH * ion_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                }

                // hydrogen bonding (basic COSMORS)
                else if (sigmai < minus_sigmaHB && segments.SegmentTypeAtomicNumber[i] != 106 && sigmaj > sigmaHB) {
                    val += fHB_A_org * hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                }
                //else if (sigmai < minus_sigmaHB && sigmaj > sigmaHB){
                //	val += hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB) ;
                //}
                if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);

                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }
        }

        // attractive term parameters for ions


        // ion - water energies, scaled by g(T)
        double const fHB_cP_H2O = param.fHB_cP_H2O * g_cat_solvent;
        double const fHB_aO_H2O = param.fHB_aO_H2O * g_poly_solvent;
        double const fHB_aP_H2O = param.fHB_aP_H2O * g_poly_solvent;
        double const fHB_cM_H2O = param.fHB_cM_H2O * g_cat_solvent;
        double const fHB_aM_H2O = param.fHB_aM_H2O * g_hal_solvent;

        double const msf_cP_H2O = param.msf_cP_H2O * g_cat_solvent;
        double const msf_cM_H2O = param.msf_cM_H2O * g_cat_solvent;
        double const msf_aP_H2O = param.msf_aP_H2O * g_poly_solvent;
        //double const msf_aM_H2O = param.msf_aM_H2O;
        // C0 and DCW are not scaled
        double const C0 = param.C0;
        double const DCW = param.DCW;

        // water molecule - -----------------------------------------------------------------------------------------------------------------

        for (int i = segments.lowerBoundIndexForGroup[2]; i < segments.upperBoundIndexForGroup[2]; i++) {

            sigmai = segments.SegmentTypeSigma[i];
            if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                sigmaCorri = segments.SegmentTypeSigmaCorr[i];
                sigmaTransi = sigmaCorri - 0.816 * sigmai;
            }

            // water - monoatomic cation interactions ------------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[3]; j < segments.upperBoundIndexForGroup[3]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cM_H2O * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cM_H2O * misfit_prefactor * sigmaMFij * sigmaMFij;
                }
                // attractive
                if (sigmai > sigmaHB) {
                    val += fHB_cM_H2O * ion_prefactor * (sigmai - sigmaHB) * (sigmaj) * (1 + DCW * abs(sigmaj));
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // water - polyatomic cation interactions -----------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[4]; j < segments.upperBoundIndexForGroup[4]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cP_H2O * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cP_H2O * misfit_prefactor * sigmaMFij * sigmaMFij;
                }
                // hydrogen bonding (basic COSMORS) and ion surface specific equations
                if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {
                    val += hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                }
                if (segments.SegmentTypeAtomicNumber[j] == 108 && sigmaj < minus_sigmaHB && sigmai > sigmaHB) {
                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                }
                else if (segments.SegmentTypeAtomicNumber[j] > 100 || segments.SegmentTypeAtomicNumber[j] == 7 || segments.SegmentTypeAtomicNumber[j] == 15 || segments.SegmentTypeAtomicNumber[j] == 16) {
                    if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {
                        val += fHB_cP_H2O * ion_prefactor * (sigmai - sigmaHB) * (sigmaj);
                    }
                }
                else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {
                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                }


                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // water - monoatomic anion interactions -----------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[5]; j < segments.upperBoundIndexForGroup[5]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // ion specific equations
                if (sigmai < minus_sigmaHB && sigmaj > C0) {
                    val += fHB_aM_H2O * ion_prefactor * (sigmaj - C0) * (sigmai + sigmaHB);
                    //val += fHB_aM_H2O * ion_prefactor * (sigmaj) * (sigmai + sigmaHB);
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // water - polyatomic anion interactions ------------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[6]; j < segments.upperBoundIndexForGroup[6]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_aP_H2O * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_aP_H2O * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // ion specific equations
                if (sigmai < minus_sigmaHB && sigmaj > sigmaHB && segments.SegmentTypeAtomicNumber[j] == 8) {
                    val += fHB_aO_H2O * ion_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                }
                else if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {
                    val += fHB_aP_H2O * hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                }
                // hydrogen bonding (basic COSMORS)
                else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);

                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }
        }

        // attractive term parameters for ions
        // ion - ion energies, scaled by g(T)
        double const msf_cP_aP = param.msf_cP_aP * g_ion_ion;
        double const msf_cP_aM = param.msf_cP_aM * g_ion_ion;
        double const msf_cM_aP = param.msf_cM_aP * g_ion_ion;
        double const msf_cM_aM = param.msf_cM_aM * g_ion_ion;

        double const fHB_cM_aM = param.fHB_cM_aM * g_ion_ion;
        double const fHB_cM_aP = param.fHB_cM_aP * g_ion_ion;
        double const fHB_cP_aM = param.fHB_cP_aM * g_ion_ion;
        double const fHB_cP_aP = param.fHB_cP_aP * g_ion_ion;
        double const fHB_iP_iP = param.fHB_iP_iP * g_ion_ion;

        // D1, E1, D2, E2 are not scaled
        double const D1 = param.D1;
        double const E1 = param.E1;
        double const D2 = param.D2;
        double const E2 = param.E2;
        //double const C2 = param.C2;

        // monoatomic cation - -----------------------------------------------------------------------------------------------------------------

        for (int i = segments.lowerBoundIndexForGroup[3]; i < segments.upperBoundIndexForGroup[3]; i++) {

            sigmai = segments.SegmentTypeSigma[i];
            if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                sigmaCorri = segments.SegmentTypeSigmaCorr[i];
                sigmaTransi = sigmaCorri - 0.816 * sigmai;
            }

            // monoatomic cation - monoatomic cation interactions -----------------------------------------------------------------------------------------
            for (int j = i; j < segments.upperBoundIndexForGroup[3]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit
                sigmaMFij = sigmai + sigmaj;
                //val = prefactor_ions * scint_cat_cat * sigmaMFij * sigmaMFij;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }
                // attractive
                // no attractive interaction

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // monoatomic cation - polyatomic cation interactions -----------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[4]; j < segments.upperBoundIndexForGroup[4]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit
                sigmaMFij = sigmai + sigmaj;
                //val = prefactor_ions * scint_cat_cat * sigmaMFij * sigmaMFij;
                // attractive
                // no attractive interaction


                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // hydrogen bonding (basic COSMORS) NOTE AGC: this interaction is unknown !!! A good initial assumption is to treat it as cation - organic, but new parameters will be needed
                if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {
                    val += ion_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // monoatomic cation - monoatomic anion interactions -----------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[5]; j < segments.upperBoundIndexForGroup[5]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cM_aM * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cM_aM * misfit_prefactor * sigmaMFij * sigmaMFij;
                }
                //NOTE AGC: for single precision std::pow is faster; for double precision pow is supposed to be faster
                //temp = std::max(0.0, (1 - D2 * pow(std::abs(sigmai), E2)));
                temp = std::max(0.0, (1 - D1 * pow(std::abs(sigmai), E1)));
                // ion speicfic equations
                if (sigmaj > 0.0 && sigmai < 0.0) {
                    //NOTE AGC: the following equation applies a Gaussian distribution-like form to the ion-ion interaction. It was not used in the final calculation. Left here as example.
                    //val += fHB_cM_aM * ion_prefactor * (sigmaj * sigmai) * (1 - D2 * pow(std::abs(sigmai), E2)) * 1/(std::sqrt(2*M_PI) * D1) * std::exp(-1.0*pow(E1*(sigmai + sigmaj),2)/(2*pow(D1,2)));
                    val += fHB_cM_aM * ion_prefactor * (sigmaj * sigmai) * temp;
                }
                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // monoatomic cation - polyatomic anion interactions ------------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[6]; j < segments.upperBoundIndexForGroup[6]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cM_aP * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cM_aP * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // ion specific equations
                if (sigmaj > sigmaHB && sigmai < 0.0 && segments.SegmentTypeAtomicNumber[j] != 7) {
                    val += fHB_cM_aP * ion_prefactor * (sigmaj * sigmai) * std::max(0.0, (1 - D2 * pow(std::abs(sigmai), E2)));
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

        }

        // polyatomic cation - -----------------------------------------------------------------------------------------------------------------

        for (int i = segments.lowerBoundIndexForGroup[4]; i < segments.upperBoundIndexForGroup[4]; i++) {

            sigmai = segments.SegmentTypeSigma[i];
            if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                sigmaCorri = segments.SegmentTypeSigmaCorr[i];
                sigmaTransi = sigmaCorri - 0.816 * sigmai;
            }

            // polyatomic cation - polyatomic cation interactions -----------------------------------------------------------------------------------------
            for (int j = i; j < segments.upperBoundIndexForGroup[4]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // hydrogen bonding (basic COSMORS)
                if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {

                    val += fHB_iP_iP * hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);

                }
                else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                    val += fHB_iP_iP * hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);


                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // polyatomic cation - monoatomic anion interactions -----------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[5]; j < segments.upperBoundIndexForGroup[5]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cP_aM * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj)); // --------------------------------- msf_iM_iP
                }
                else {
                    val = msf_cP_aM * misfit_prefactor * sigmaMFij * sigmaMFij; 											   // --------------------------------- msf_iM_iP
                }

                // ion specific equations
                if (sigmai < 0.0 && sigmaj > 0.0) {
                    val += fHB_cP_aM * ion_prefactor * (sigmaj * sigmai) * std::max(0.0, (1 - D2 * pow(std::abs(sigmai), E2)));
                    //val += -1.0 * fHB_cP_aM * ion_prefactor * std::max(0.0, sigmaj - C2) * std::max(0.0, std::abs(sigmai) - C2) * std::max(0.0, (1 - D2 * pow(std::abs(sigmai), E2)));
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // polyatomic cation - polyatomic anion interactions ------------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[6]; j < segments.upperBoundIndexForGroup[6]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = msf_cP_aP * misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = msf_cP_aP * misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // ion specific equations
                if (sigmai < 0.0 && sigmaj > 0.0) {
                    val += fHB_cP_aP * ion_prefactor * (sigmaj * sigmai) * std::max(0.0, (1 - D2 * pow(std::abs(sigmai), E2)));
                    //val += -1.0 * fHB_cP_aP * ion_prefactor * std::max(0.0, sigmaj - C2) * std::max(0.0, std::abs(sigmai) - C2) * std::max(0.0, (1 - D2 * pow(std::abs(sigmai), E2)));
                }
                else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {
                    val += hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

        }

        // monoatomic anion - -----------------------------------------------------------------------------------------------------------------

        for (int i = segments.lowerBoundIndexForGroup[5]; i < segments.upperBoundIndexForGroup[5]; i++) {

            sigmai = segments.SegmentTypeSigma[i];
            if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                sigmaCorri = segments.SegmentTypeSigmaCorr[i];
                sigmaTransi = sigmaCorri - 0.816 * sigmai;
            }

            // monoatomic anion - monoatomic anion interactions ------------------------------------------------------------------------------------------
            for (int j = i; j < segments.upperBoundIndexForGroup[5]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];
                sigmaMFij = sigmai + sigmaj;

                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

            // monoatomic anion - polyatomic anion interactions ------------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[6]; j < segments.upperBoundIndexForGroup[6]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // hydrogen bonding (basic COSMORS) NOTE AGC: THIS INTERACTION IS UNKNOWN !!!
                if (sigmaj < minus_sigmaHB) {

                    val += hb_prefactor * (sigmai) * (sigmaj + sigmaHB);

                }


                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }

        }

        // polyatomic anion - -----------------------------------------------------------------------------------------------------------------
        for (int i = segments.lowerBoundIndexForGroup[6]; i < segments.upperBoundIndexForGroup[6]; i++) {

            sigmai = segments.SegmentTypeSigma[i];
            if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                sigmaCorri = segments.SegmentTypeSigmaCorr[i];
                sigmaTransi = sigmaCorri - 0.816 * sigmai;
            }

            // polyatomic anion - polyatomic anion interactions ------------------------------------------------------------------------------------------
            for (int j = segments.lowerBoundIndexForGroup[6]; j < segments.upperBoundIndexForGroup[6]; j++) {
                sigmaj = segments.SegmentTypeSigma[j];
                sigmaMFij = sigmai + sigmaj;

                // misfit (basic COSMORS)
                sigmaMFij = sigmai + sigmaj;
                if (param.sw_misfit == 1 || param.sw_misfit == 3) {
                    sigmaCorrj = segments.SegmentTypeSigmaCorr[j];
                    sigmaTransj = sigmaCorrj - 0.816 * sigmaj;

                    val = misfit_prefactor * sigmaMFij * (sigmaMFij + param.fCorr * (sigmaTransi + sigmaTransj));
                }
                else {
                    val = misfit_prefactor * sigmaMFij * sigmaMFij;
                }

                // hydrogen bonding (basic COSMORS)
                if (sigmai < minus_sigmaHB && sigmaj > sigmaHB) {

                    val += fHB_iP_iP * hb_prefactor * (sigmaj - sigmaHB) * (sigmai + sigmaHB);

                }
                else if (sigmaj < minus_sigmaHB && sigmai > sigmaHB) {

                    val += fHB_iP_iP * hb_prefactor * (sigmai - sigmaHB) * (sigmaj + sigmaHB);

                }

                val += ion_pair_dispersion(i, j);
                A_int(std::max(i, j), std::min(i, j)) = (float)val;
            }
        }
        // ion - hole interactions
        {
            auto const it_msf_ion_hole = param.exp_param.find("msf_ion_hole");
            double const msf_ion_hole = (it_msf_ion_hole == param.exp_param.end())
                ? 1.0 : it_msf_ion_hole->second;

            if (msf_ion_hole != 1.0) {
                for (int group = 3; group <= 6; group++) {
                    for (int i = segments.lowerBoundIndexForGroup[group];
                         i < segments.upperBoundIndexForGroup[group]; i++) {
                        for (int j = segments.lowerBoundIndexForGroup[7];
                             j < segments.upperBoundIndexForGroup[7]; j++) {
                            A_int(std::max(i, j), std::min(i, j)) *= (calcType)msf_ion_hole;
                        }
                    }
                }
            }
        }


    }
    // COSMO-RS-ES+ equations end here 		------------------------------------------------------

    if (param.sw_useSegmentReferenceStateForInteractionMatrix == 1) {

        // apply the required reference state for the interaction matrix
        for (int i = 0; i < numberOfSegments; i++) {
            for (int j = i + 1; j < numberOfSegments; j++) {
                // always j >= i + 1
                A_int(j, i) = A_int(j, i) - 0.5f * (A_int(i, i) + A_int(j, j));
                if (withDerivative) {
                    A_int_PDET(j, i) = A_int_PDET(j, i) - 0.5f * (A_int_PDET(i, i) + A_int_PDET(j, j));
                    A_int_SecondPDET(j, i) = A_int_SecondPDET(j, i) - 0.5f * (A_int_SecondPDET(i, i) + A_int_SecondPDET(j, j));
                }
            }
        }

        // assign zero to the diagonal
        for (int i = 0; i < numberOfSegments; i++) {
            A_int(i, i) = 0.0f;
            if (withDerivative) {
                A_int_PDET(i, i) = 0.0f;
                A_int_SecondPDET(i, i) = 0.0f;
            }
        }

        if (param.sw_calculateContactStatisticsAndAdditionalProperties > 0) {
            for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                for (int i = 0; i < numberOfSegments; i++) {
                    for (int j = i + 1; j < numberOfSegments; j++) {
                        // always j >= i + 1
                        partialInteractionMatrices[h](j, i) = partialInteractionMatrices[h](j, i) - 0.5 * (partialInteractionMatrices[h](i, i) + partialInteractionMatrices[h](j, j));
                    }
                }
            }

            for (int h = 0; h < param.numberOfPartialInteractionMatrices; h++) {
                for (int i = 0; i < numberOfSegments; i++) {
                    partialInteractionMatrices[h](i, i) = 0.0;
                }
            }
        }
    }
    // debug: enumerate all interaction matrix calls, dumping each to a numbered file.
    // The file name contains the call number and the number of segments so that
    // pure-solvent calls, mixture calls and subcalculations can be told apart.
  /*  static int interactionMatrixCallCounter = 0;
    interactionMatrixCallCounter++;
    if (interactionMatrixCallCounter <= 99) {
        std::string callNumber = std::to_string(interactionMatrixCallCounter);
        if (callNumber.size() < 2) callNumber = "0" + callNumber;
        std::cout << "interaction matrix call " << callNumber << ": " << numberOfSegments << " segments, T = " << temperature << " K" << std::endl;
        WriteEigenMatrixToFile("C:/Users/clr8069/source/repos/LeanderX/interaction_matrix_EOS_call" + callNumber + "_" + std::to_string(numberOfSegments) + "seg.txt", A_int.cast<double>());
    }*/

}

