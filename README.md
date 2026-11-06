# metatomic-qtip4pf

The flexible q-TIP4P/F water model ([Habershon, Markland and Manolopoulos, J. Chem.
Phys. 131, 024501 (2009)](https://doi.org/10.1063/1.3167790)) as a
[metatomic](https://docs.metatensor.org/metatomic/) model, written in plain PyTorch.
It only depends on `torch` and `metatomic-torch`. The model can be exported to a
standalone TorchScript file and run in any engine that supports metatomic models
(i-PI, ASE, LAMMPS, ...).

## Installation and usage

The package is not on PyPI; install it directly from GitHub:

```bash
pip install git+https://github.com/ceriottm/qtip4pf-test

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
  - The cost of this plain Ewald sum grows roughly as N² at fixed cutoff, so the model is meant for small boxes, up to a few hundred molecules (see [Performance](#performance)).
- **Non-periodic systems** use a direct Coulomb sum over all pairs of molecules, without cutoff.

## Performance

This is a test potential for small boxes. Systems with the same number of atoms that
are passed to the model in a single call (e.g. the beads of a path integral
simulation in i-PI) are evaluated together. On GPU, this removes most of the
overhead of launching many small kernels. On CPU, systems are evaluated in chunks of
about 1000 atoms.

Time (ms per system) for energy, forces and virial with the exported model, without
the neighbor list (computed by the simulation engine), for boxes of water at
0.997 g/cm³ and one or 32 systems per call:

| molecules | CPU (1 core), 1 | CPU (1 core), 32 | H100, 1 | H100, 32 | L40S, 1 | L40S, 32 |
|---|---|---|---|---|---|---|
| 32 | 3.9 | 2.4 | 4.8 | 0.58 | 4.6 | 0.55 |
| 64 | 5.7 | 4.5 | 5.1 | 0.59 | 4.9 | 0.64 |
| 128 | 11.3 | 10.4 | 5.2 | 0.60 | 5.0 | 0.86 |
| 216 | 20.4 | 20.3 | 5.2 | 0.65 | 5.1 | 1.28 |

A single small system on GPU is limited by kernel launch overhead, so it is not
faster than on CPU. On CPU, the default glibc memory allocator returns large blocks
to the operating system after each evaluation, and then has to map them again.
Preloading jemalloc (`LD_PRELOAD=/path/to/libjemalloc.so.2`) or setting
`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1073741824:glibc.malloc.trim_threshold=1073741824`
makes CPU evaluations about 30% faster.

These numbers come from [`benchmarks/benchmark.py`](benchmarks/benchmark.py), which
follows what i-PI does at each step:

```bash
python benchmarks/benchmark.py --device cpu --threads 1 --batch 1 32
python benchmarks/benchmark.py --device cuda --batch 1 32
```

## Validation

The test suite checks:

- the energy against an independent transcription of Eqs. (1)–(4) of the paper, for single molecules and non-periodic clusters;
- the Ewald sum against the Madelung constants of NaCl and CsCl (including a triclinic cell). It also checks that the result does not depend on the splitting parameter, and the analytical finite-size correction of a dimer in a large box;
- forces and virial against finite differences, invariance under translations, rotations, molecule wrapping and changes of the cell representation, and size-extensivity in a supercell;
- energy, forces and virial for a 216-molecule box against the Fortran q-TIP4P/F implementation shipped with i-PI (see [`tests/data/README.md`](tests/data/README.md)). The agreement is ~10⁻⁹ (relative) when both Ewald sums are converged;
- that the exported TorchScript model reproduces the Python one, with automatic unit conversion.

Run the tests from a clone of the repository with:

```bash
git clone https://github.com/ceriottm/qtip4pf-test
cd qtip4pf-test
pip install ".[test]"
pytest
```
