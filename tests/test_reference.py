"""
Comparison with the q-TIP4P/F implementation in i-PI's Fortran driver
(drivers/f90/pes/qtip4pf.f90), see data/README.md for details.
"""

import json
import os

import numpy as np
import pytest
from conftest import DATA, IPI_PARAMETERS, evaluate

from metatomic_qtip4pf import QTIP4PF


@pytest.fixture
def reference():
    with open(os.path.join(DATA, "water_216_ipi_fortran.json")) as fd:
        data = json.load(fd)
    return data["energy"], np.array(data["forces"]), np.array(data["virial"])


@pytest.mark.parametrize("wrap", [False, True])
def test_ipi_fortran(water_216, reference, wrap):
    types, positions, cell = water_216
    if wrap:
        positions = positions - np.floor(positions / np.diag(cell)) * np.diag(cell)

    ref_energy, ref_forces, ref_virial = reference

    model = QTIP4PF(parameters=IPI_PARAMETERS, ewald_accuracy=1e-10)
    energy, forces, virial = evaluate(model, types, positions, cell)

    assert energy == pytest.approx(ref_energy, rel=1e-9)
    # forces are ~ 30 kcal/mol/A
    assert np.allclose(forces, ref_forces, atol=1e-6)
    # virial components are ~ 1000 kcal/mol
    assert np.allclose(virial, ref_virial, atol=1e-5)


def test_default_accuracy(water_216, reference):
    """The default Ewald accuracy is enough for practical purposes"""
    types, positions, cell = water_216
    ref_energy, ref_forces, ref_virial = reference

    model = QTIP4PF(parameters=IPI_PARAMETERS)
    energy, forces, virial = evaluate(model, types, positions, cell)

    assert energy == pytest.approx(ref_energy, rel=1e-5)
    assert np.allclose(forces, ref_forces, atol=1e-4)
    assert np.allclose(virial, ref_virial, rtol=1e-4, atol=0.1)


def test_paper_parameters(water_216, reference):
    """
    Table I parameters differ from the rounded i-PI ones by up to 2e-4 (relative),
    which leads to small but visible differences
    """
    types, positions, cell = water_216
    ref_energy, ref_forces, _ = reference

    energy, forces, _ = evaluate(QTIP4PF(), types, positions, cell)
    assert energy == pytest.approx(ref_energy, rel=1e-3)
    assert not energy == pytest.approx(ref_energy, rel=1e-5)
    assert np.allclose(forces, ref_forces, atol=0.1)
