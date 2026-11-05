# Reference data

`water_216.xyz` holds 216 water molecules in a cubic box of 35.233 bohr. The data comes from
`examples/init_files/water_216.xyz` in [i-PI](https://github.com/i-pi/i-pi),
converted from bohr to Å with a0 = 0.529177210903 Å.

`water_216_ipi_fortran.json` holds the energy (kcal/mol), forces (kcal/mol/Å) and
virial (−∂E/∂ε, in kcal/mol) for this structure. They were computed with the q-TIP4P/F
implementation in i-PI's Fortran driver (`drivers/f90/pes/qtip4pf.f90`) using its
hard-coded parameters, which are rounded atomic-unit versions of Table I. The
matching parameters are `IPI_PARAMETERS` in `tests/conftest.py`. Results were
converted from atomic units with a0 = 0.529177210903 Å and Eh = 627.5094740631
kcal/mol (CODATA 2018).

The Fortran Ewald sum targets ~10⁻⁵ accuracy: real-space cutoff at erfc(π) and
an approximate erfc. To get a converged reference, the following changes were
made to a copy of `qtip4pf.f90` before generating the data:

- `walpha = 5.5d0/wrcut` (instead of `pi/wrcut`), with
  `rkmax = 2.d0*5.5d0*walpha` and `kmax = int(5.5d0*walpha*rlong/pi) + 1`;
- the exact `erfc = 1.d0 - erf(x)` replaces the Abramowitz-Stegun 7.1.26 approximation.

The driver also adds a Lennard-Jones tail correction to the virial, but not the
energy, and puts it in the wrong element (`vir(2,3) = vir(2,2) - ptail`).
This correction is not part of the model, so it was removed from the `xx` and `zz`
components, and `yz` was set to `zy`.

With the unmodified driver, energies differ by ~10⁻⁴ (relative), and forces by
~10⁻⁵ atomic units, which is the accuracy of its Ewald sum.
