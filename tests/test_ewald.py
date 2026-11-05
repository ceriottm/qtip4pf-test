import math

import numpy as np
import pytest
import torch
from conftest import energy_only, water_molecule

from metatomic_qtip4pf import QTIP4PF
from metatomic_qtip4pf._model import ewald_reciprocal


NACL_MADELUNG = 1.747564594633182
CSCL_MADELUNG = 1.762674773070988


def point_charges_energy(positions, charges, cell, alpha=12.0):
    """
    Ewald energy of point charges with a very large alpha, so that the real-space
    sum is negligible (erfc(alpha * r_nn) < 1e-30) and only the reciprocal and
    self terms contribute.
    """
    positions = torch.tensor(positions, dtype=torch.float64)
    charges = torch.tensor(charges, dtype=torch.float64)
    cell = torch.tensor(cell, dtype=torch.float64)
    k_cut = 2 * alpha * math.sqrt(-math.log(1e-16))
    reciprocal = ewald_reciprocal(positions, charges, cell, alpha, k_cut, 1.0)
    self_energy = -alpha / math.sqrt(math.pi) * torch.sum(charges**2)
    return (reciprocal + self_energy).item()


def test_madelung_nacl_conventional():
    a = 1.0
    na = a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cl = na + a * np.array([0.5, 0, 0])
    energy = point_charges_energy(
        np.concatenate([na, cl]), [1.0] * 4 + [-1.0] * 4, a * np.eye(3)
    )
    # 4 ion pairs, nearest neighbor distance a/2
    assert energy == pytest.approx(-4 * NACL_MADELUNG / (a / 2), rel=1e-12)


def test_madelung_nacl_primitive():
    """Triclinic (FCC primitive) cell"""
    a = 1.0
    cell = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    positions = np.array([[0, 0, 0], [a / 2, a / 2, a / 2]])
    energy = point_charges_energy(positions, [1.0, -1.0], cell)
    assert energy == pytest.approx(-NACL_MADELUNG / (a / 2), rel=1e-12)


def test_madelung_cscl():
    a = 1.0
    positions = np.array([[0, 0, 0], [a / 2, a / 2, a / 2]])
    energy = point_charges_energy(positions, [1.0, -1.0], a * np.eye(3))
    assert energy == pytest.approx(-CSCL_MADELUNG / (math.sqrt(3) * a / 2), rel=1e-12)


def test_alpha_independence(small_box):
    """
    Changing ewald_accuracy changes the splitting parameter (and the reciprocal
    cutoff): the total energy should not change, which checks the consistency of
    the real-space, reciprocal, self and intramolecular-exclusion terms.
    """
    types, positions, cell = small_box
    energies = []
    for accuracy in [1e-10, 1e-12, 1e-14]:
        model = QTIP4PF(ewald_accuracy=accuracy)
        energies.append(energy_only(model, types, positions, cell))

    assert energies[1] == pytest.approx(energies[0], rel=1e-9, abs=1e-8)
    assert energies[2] == pytest.approx(energies[0], rel=1e-9, abs=1e-8)

    # the default accuracy is also close to the converged result
    energy = energy_only(QTIP4PF(), types, positions, cell)
    assert energy == pytest.approx(energies[0], abs=1e-3)


@pytest.mark.parametrize("box", [60.0, 90.0])
def test_isolated_limit(box):
    """
    A water dimer in a large cubic periodic box: Ewald (with conducting boundary
    conditions) differs from the isolated system by -2 pi k_e |mu|^2 / (3 V), up
    to higher-order multipole terms, which decay as 1/L^5.
    """
    from metatomic_qtip4pf import COULOMB_CONSTANT, PARAMETERS

    rng = np.random.default_rng(3)
    positions = np.concatenate(
        [water_molecule(rng), np.array([2.9, 0.2, 0.1]) + water_molecule(rng)]
    )
    types = [8, 1, 1] * 2

    gamma = PARAMETERS["gamma"]
    q_M = PARAMETERS["q_M"]
    dipole = np.zeros(3)
    for o, h1, h2 in positions.reshape(-1, 3, 3):
        m = gamma * o + (1 - gamma) * (h1 + h2) / 2
        dipole += q_M / 2 * (h1 + h2) - q_M * m

    model = QTIP4PF(ewald_accuracy=1e-10)
    isolated = energy_only(model, types, positions, np.zeros((3, 3)), pbc=False)
    periodic = energy_only(model, types, positions, box * np.eye(3))
    correction = -2 * math.pi * COULOMB_CONSTANT * dipole @ dipole / (3 * box**3)
    assert abs(correction) > 1e-4
    assert periodic == pytest.approx(isolated + correction, abs=2e-5)
