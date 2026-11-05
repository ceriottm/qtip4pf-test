# metatomic-qtip4pf

The flexible q-TIP4P/F water model ([Habershon, Markland and Manolopoulos, J. Chem.
Phys. 131, 024501 (2009)](https://doi.org/10.1063/1.3167790)) as a
[metatomic](https://docs.metatensor.org/metatomic/) model, written in plain PyTorch.
It only depends on `torch` and `metatomic-torch`. The model can be exported to a
standalone TorchScript file and run in any engine that supports metatomic models
(i-PI, ASE, LAMMPS, ...).

## Installation and usage

```bash
pip install metatomic-qtip4pf

# export the model to qtip4pf.pt
metatomic-qtip4pf -o qtip4pf.pt
```

or, from Python:

```python
from metatomic_qtip4pf import get_model

model = get_model()  # metatomic.torch.AtomisticModel
model.save("qtip4pf.pt")
```

The model needs **atoms ordered as O, H, H, O, H, H, ...**, with each triplet of
consecutive atoms making up one molecule. Molecules do not need to be whole across
periodic boundaries. It provides the total `energy` of each system. The engine gets
forces and stress/virial by automatic differentiation. Systems can be fully
periodic (any cell shape) or non-periodic.

`.pt` files are tied to the version of `torch` used to export them, so you should
export the model in the environment that will run the simulation.

## Model

Internally, the model works in kcal/mol and Å and uses the parameters from Table I
of the paper exactly:

| Term | Functional form | Parameters |
|---|---|---|
| O–H stretch | $D_r[\alpha_r^2\Delta r^2 - \alpha_r^3\Delta r^3 + \tfrac{7}{12}\alpha_r^4\Delta r^4]$, with $\Delta r = r - r_\text{eq}$ | $D_r$ = 116.09 kcal/mol, $\alpha_r$ = 2.287 Å⁻¹, $r_\text{eq}$ = 0.9419 Å |
| H–O–H bend | $\tfrac12 k_\theta(\theta-\theta_\text{eq})^2$ | $k_\theta$ = 87.85 kcal/mol/rad², $\theta_\text{eq}$ = 107.4° |
| O–O Lennard-Jones | $4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]$ | $\varepsilon$ = 0.1852 kcal/mol, $\sigma$ = 3.1589 Å |
| Charges | $+q_M/2$ on each H, $-q_M$ on M | $q_M$ = 1.1128 e |
| M site | $\mathbf r_M = \gamma\mathbf r_O + (1-\gamma)(\mathbf r_{H_1} + \mathbf r_{H_2})/2$ | $\gamma$ = 0.73612 |

The Coulomb interactions include only charges on different molecules; all
intramolecular electrostatics (H–H, H–M) is excluded. The Coulomb constant
$e^2/4\pi\varepsilon_0$ = 332.0637132991921 kcal/mol·Å comes from CODATA 2018.

Some implementation choices:

- **Lennard-Jones** is truncated at `cutoff` (9 Å by default, as in the paper). There is no shift and no tail correction, as in the original simulations and in the i-PI and LAMMPS reference inputs.
- **Periodic electrostatics** uses an Ewald sum with conducting boundary conditions.
  - The real-space sum includes all charge pairs from molecules whose O–O distance is below `cutoff` (a molecule-based cutoff).
  - The splitting parameter and the reciprocal-space cutoff come from `ewald_accuracy` (default 1e-6). For a 216-molecule box, the defaults give energies within 4·10⁻⁶ (relative) of the fully converged Ewald sum and forces within 10⁻⁴ kcal/mol/Å.
  - The cost of this plain Ewald sum grows roughly as N² at fixed cutoff. On a CPU, energy, forces and virial take about 0.3 s for 216 molecules and 7 s for 1728 molecules.
- **Non-periodic systems** use a direct Coulomb sum over all pairs of molecules, without cutoff.

## Validation

The test suite checks:

- the energy against an independent transcription of Eqs. (1)–(4) of the paper, for single molecules and non-periodic clusters;
- the Ewald sum against the Madelung constants of NaCl and CsCl (including a triclinic cell). It also checks that the result does not depend on the splitting parameter, and the analytical finite-size correction of a dimer in a large box;
- forces and virial against finite differences, invariance under translations, rotations, molecule wrapping and changes of the cell representation, and size-extensivity in a supercell;
- energy, forces and virial for a 216-molecule box against the Fortran q-TIP4P/F implementation shipped with i-PI (see [`tests/data/README.md`](tests/data/README.md)). The agreement is ~10⁻⁹ (relative) when both Ewald sums are converged;
- that the exported TorchScript model reproduces the Python one, with automatic unit conversion.

Run the tests with:

```bash
pip install ".[test]"
pytest
```
