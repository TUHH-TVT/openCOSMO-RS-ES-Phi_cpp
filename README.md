# openCOSMO-RS-ES-Phi

openCOSMO-RS-ES-Phi is a predictive electrolyte equation of state based on openCOSMO-RS. It combines the
COSMO-RS-based equation of state openCOSMO-RS-Phi with the ion interaction energies and the long-range term of
COSMO-RS-ES, refitted on the same simplified short range. Unlike COSMO-RS-ES, which treats the liquid as
incompressible, it gives access to densities and pressure effects.

The permittivity of the mixture includes the dielectric decrement and follows from volume and composition through the
perturbation-theory model of Rueben et al. A Born term is left out, because it counts the solvation of the ions a
second time. The equation of state adds only volume, dispersion and temperature parameters of the ions, fitted to
aqueous mean ionic activity coefficients (MIAC) and densities. These parameters and the parameters of the solvents
are given in the paper and its supporting information.

The model reproduces COSMO-RS-ES in dilute solutions, improves on it in concentrated ones and yields solution
densities. Without mixed-solvent data in its fit, it predicts the MIAC and bubble points of water–alcohol–salt
systems. The main limitation is the transfer of ions between solvents, which aqueous data do not constrain: the
salting-out of the alcohol is missed, and salts come out too soluble in organic solvents.

The model is implemented in C++ and exposed to Python through pybind11 as the module `openCOSMORS`. The Python
scripts in `bindings/` are the example entry point.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `code/` | C++ source and the Python binding `bindings_forPython.cpp`. |
| `bindings/` | Example scripts, COSMO files, and locally built Python extension files. |
| `eigen/`, `pybind11/` | External dependencies as git submodules. |
| `nlohmann/` | nlohmann/json 3.12.0, MIT license. |

## Requirements

- CMake 3.15 or newer.
- A C++14 compiler with OpenMP. Tested with gcc 11.2 on Linux. The Visual Studio solution targets MSVC from
  Visual Studio 2022.
- Python with NumPy. The examples also use pandas and matplotlib. The Visual Studio Python project is configured for
  the conda environment `base311`, which can be changed in the project settings.
- After cloning, initialize the dependencies:

```bash
git submodule update --init --recursive
```

or clone with them:

```bash
git clone --recurse-submodules <repository-url>
```

The submodules are pinned to the same commits as openCOSMO-RS-Phi_cpp: an Eigen development snapshot (3.5.0-dev,
MPL-2.0) and pybind11 3.x (BSD). The Eigen release 3.4.0 does not compile the code, because of its indexing of
vectors with `Eigen::all`.

## Build

### Python Extension

The CMake target builds the `openCOSMORS` Python extension.

```bash
cmake -S . -B build -DOPENCOMSORS_SIMD=FMA
cmake --build build --config Release
```

With a specific Python interpreter, for example a conda environment:

```bash
cmake -S . -B build -DOPENCOMSORS_SIMD=FMA -DPython_EXECUTABLE="$(which python)"
cmake --build build --config Release
```

On Windows with a conda environment:

```powershell
conda activate base311
cmake -S . -B build -DOPENCOMSORS_SIMD=FMA -DPython_EXECUTABLE="$env:CONDA_PREFIX\python.exe"
cmake --build build --config Release
```

If your CMake version does not know the Visual Studio 2022 generator, run from a Visual Studio developer shell or
after `vcvars64.bat` and use NMake:

```powershell
cmake -S . -B build-msvc-nmake -G "NMake Makefiles" -DOPENCOMSORS_SIMD=FMA -DPython_EXECUTABLE="$env:CONDA_PREFIX\python.exe"
cmake --build build-msvc-nmake
```

Without CMake, the module can be compiled directly:

```bash
g++ -fopenmp -mavx2 -mfma -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) \
    code/bindings_forPython.cpp -o openCOSMORS$(python3-config --extension-suffix) \
    -I code -I . -I pybind11/include -I eigen
```

### Visual Studio Solution

`openCOSMORS.sln` contains:

- `openCOSMORS-ES-Phi`: the C++/pybind11 project.
- `openCOSMORS_python_example`: the Python example project.

The C++ project includes Python from `$(CONDA_PREFIX)\include` and links from `$(CONDA_PREFIX)\libs`.
`open_vs_with_conda_env.bat` sets `CONDA_PREFIX` and opens the solution; adjust the path in it to your conda
environment. The project uses `/arch:AVX2`, which selects the `FMA` backend, and OpenMP in the Release
configuration. Debug and Release write `openCOSMORS.pyd` to `bindings\Debug\` and `bindings\Release\`.

## SIMD Backend

The code selects its vector instructions through the compiler macros `__SSE3__`, `__AVX__` and `__FMA__`, set by
`OPENCOMSORS_SIMD`:

| Value | Meaning |
| --- | --- |
| `AUTO` | `FMA` on x86-64. |
| `SSE3` | x86 SSE3. |
| `AVX` | x86 AVX. |
| `FMA` | x86 AVX2/FMA (`-mavx2 -mfma`, `/arch:AVX2` with MSVC). |

Only x86 targets are supported. With the pinned Eigen commit, `AVX` does not compile, because Eigen's code path for
AVX without AVX2 is broken there; use `FMA` or `SSE3`.

## Running Examples

The examples live in `bindings/`. Run them from that directory, with the compiled `.pyd`/shared library either
copied there or on `PYTHONPATH`.

```powershell
conda activate base311
cd bindings
python run_phi_pure_with_densities.py
```

If you built into `build/Release`, use:

```powershell
$env:PYTHONPATH = "$PWD\build\Release;$env:PYTHONPATH"
python bindings\run_phi_pure_with_densities.py
```

On Linux, the module is written to `build/`:

```bash
PYTHONPATH="$PWD/build:$PYTHONPATH" python bindings/run_phi_pure_with_densities.py
```

The examples are the openCOSMO-RS-Phi examples for neutral components. They run openCOSMO-RS-ES-Phi with all
electrolyte switches set to 0 (`sw_electrolyte`, `sw_Born_term`, `sw_partitionCoeff`, `sw_SR_polarizabilities`),
which reduces the model to openCOSMO-RS-Phi. They differ from the openCOSMO-RS-Phi examples in two inputs:

- `alpha` and `CHB` are absolute values in J Å² mol⁻¹ e⁻² (8130 and 42 564 kJ Å² mol⁻¹ e⁻² in the paper).
  openCOSMO-RS-Phi takes `ln_alpha` and `ln_CHB` instead, the logarithms of factors on 5950 and 36 700 kJ Å² mol⁻¹ e⁻².
- Pure-component calculations take the calculation type `VL`.

### Pure-Component Vapor Pressure

`bindings/run_phi_pure.py` is the minimal pure-component vapor-pressure example. It calculates the vapor pressure of
methanol at one user-provided temperature.

### Vapor Pressure With Saturated Densities

`bindings/run_phi_pure_with_densities.py` extends the pure-component workflow. It exposes:

```python
calculate_vapor_pressure_and_densities(
    orcacosmo_file,
    pure_component_parameters,
    temperatures,
    pressure_guess=101325.0,
)
```

The function returns one dictionary per temperature with:

- `temperature_K`
- `vapor_pressure_Pa`
- `vapor_molar_volume_m3_per_mol`
- `liquid_molar_volume_m3_per_mol`
- `vapor_molar_density_mol_per_m3`
- `liquid_molar_density_mol_per_m3`
- `vapor_mass_density_kg_per_m3`
- `liquid_mass_density_kg_per_m3`

The script also contains a runnable methanol example:

```python
methanol_pc_parameters = [
    56.08186814344738,
    1.3518314651065213,
    3.3225493224786162,
    8.713083189565843,
]
methanol_temperatures = np.array([280.0, 298.15, 320.0], np.float64)
results = calculate_vapor_pressure_and_densities(
    "methanol.orcacosmo",
    methanol_pc_parameters,
    methanol_temperatures,
)
```

### Mixture Workflows

`bindings/run_phi_mixture.py` contains examples and helper functions for common binary-mixture workflows, including:

- isothermal VLE diagrams
- isobaric VLE diagrams
- dew-pressure calculations
- pT flash calculations
- activity coefficients
- excess and mixing properties
- critical and azeotropic point searches

The file is intentionally script-oriented: uncomment the calculation block you want to run, set component COSMO files
and pure-component parameters, then run the script from `bindings/`.

### Electrolyte Calculations

For systems with ions, set `sw_electrolyte` to 1. The ion parameters and the solvent parameters are given in the
paper and its supporting information. Compared with the neutral examples, the binding then reads:

| Where | Key | Content |
| --- | --- | --- |
| `options` | `sw_electrolyte` | 1 to activate openCOSMO-RS-ES-Phi. |
| `options` | `sw_Born_term` | 0, as in the paper; 1 adds a Born term. |
| `options` | `sw_permittivity_mix_model` | 0 for the permittivity model of Rueben et al., with the dielectric decrement. |
| `options` | `sw_SR_polarizabilities` | > 0 to use atomic polarizabilities, which needs `E_F_corr` and `m_vdW` in `parameters`. |
| `options` | `sw_partitionCoeff` | 1 to calculate partition coefficients. |
| `parameters` | ion interaction parameters | `salt_diam_sc_*`, `msf_*`, `fHB_*`, `ln_C0`…`ln_C2`, `ln_D1`, `ln_D2`, `ln_E1`, `ln_E2`, `DCO`, `DCW`, `omega0`, `omega1`. |
| `parameters` | `permittivity_model_0_parameters` | Array with three permittivity parameters per component, shape `(n_components, 3)`. |
| calculation | `component_stochiometric_coefficients` | Stoichiometric coefficient of each component in the salt. |
| calculation | `ln_gamma_x_LR_calc`, `ln_gamma_x_Born_calc`, `ln_gamma_x_Electrostatic_calc`, `ln_gamma_x_SRLR_calc` | Result arrays shaped like `ln_gamma_x_SR_calc`. |
| calculation | `MIAC_x_calc`, `MIAC_m_calc` | Mean ionic activity coefficients in the mole-fraction and molality scale, shaped like `ln_gamma_x_SR_calc`. |
| calculation | `salt_diam`, `ion_diam_mono` | Optional ion diameters. |
| calculation, with `sw_phi` | `Born_pressure`, `electrostatic_pressure` | Result vectors with one entry per row. |
| calculation, with `sw_phi` | `ln_phi_Born`, `ln_phi_electrostatic` | Result arrays shaped like `ln_phi`. |

Ions are non-volatile and get no pure-component state.

## Python API Shape

The pybind11 API is dictionary-based, as in openCOSMO-RS-Phi:

```python
import openCOSMORS

openCOSMORS.initialize()
openCOSMORS.loadMolecules(options, parameters, components)
openCOSMORS.loadCalculations(calculations)
calculations = openCOSMORS.calculate(parameters, calculations, False)
```

The result arrays are allocated in Python and mapped by the C++ layer. Keep arrays C-contiguous,
`dtype=np.float64`, and alive until `calculate()` returns. For Phi/EOS calculations, pure-component saturated vapor
and liquid states are stored as paired rows: for `n` temperatures, vapor rows are `0:n` and liquid rows are `n:2*n`.

## Citation

L. Raudszus, S. Müller, *An Electrolyte Equation of State Based on openCOSMO-RS: From Aqueous
Solutions to Non-Aqueous Systems*.

## License

LGPL-3.0, as openCOSMO-RS-Phi, from which the C++ code derives. The included libraries keep their licenses:
Eigen MPL-2.0, pybind11 BSD, nlohmann/json MIT.

## Related Projects

- [openCOSMO-RS-Phi_cpp](https://github.com/TUHH-TVT/openCOSMO-RS-Phi_cpp)
- [openCOSMO-RS_cpp](https://github.com/TUHH-TVT/openCOSMO-RS_cpp)
- [openCOSMO-RS_py](https://github.com/TUHH-TVT/openCOSMO-RS_py)
- [LVPP sigma profile database](https://github.com/lvpp/sigma)
- [Benchmark COSMO-SAC implementation](https://github.com/usnistgov/COSMOSAC)
- [Pysac](https://github.com/lvpp/pysac)
