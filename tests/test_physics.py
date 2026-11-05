"""
Check the model against an independent transcription of the q-TIP4P/F functional
form, Eqs. (1)-(4) and Table I of Habershon, Markland, Manolopoulos, JCP 131,
024501 (2009). Parameters are typed again here on purpose.
"""

import math

import numpy as np
import pytest
from conftest import energy_only, water_molecule

from metatomic_qtip4pf import QTIP4PF


EPSILON = 0.1852  # kcal/mol
SIGMA = 3.1589  # A
Q_M = 1.1128  # e
GAMMA = 0.73612
D_R = 116.09  # kcal/mol
ALPHA_R = 2.287  # 1/A
R_EQ = 0.9419  # A
K_THETA = 87.85  # kcal/mol/rad^2
THETA_EQ = 107.4  # degrees

# e^2/(4 pi eps0) in kcal/mol A, computed from SI constants
K_E = (
    1.602176634e-19**2
    / (4 * math.pi * 8.8541878128e-12)
    * 6.02214076e23
    / 4184.0
    * 1e10
)


def reference_intra(o, h1, h2):
    def v_oh(r):
        dr = r - R_EQ
        return D_R * (
            ALPHA_R**2 * dr**2 - ALPHA_R**3 * dr**3 + 7 / 12 * ALPHA_R**4 * dr**4
        )

    r1 = np.linalg.norm(h1 - o)
    r2 = np.linalg.norm(h2 - o)
    theta = math.acos(np.dot(h1 - o, h2 - o) / (r1 * r2))
    return v_oh(r1) + v_oh(r2) + 0.5 * K_THETA * (theta - math.radians(THETA_EQ)) ** 2


def charged_sites(o, h1, h2):
    m = GAMMA * o + (1 - GAMMA) * (h1 + h2) / 2
    return [(h1, Q_M / 2), (h2, Q_M / 2), (m, -Q_M)]


def reference_energy(positions, lj_cutoff=9.0):
    """Non-periodic q-TIP4P/F energy, by explicit double loops."""
    molecules = positions.reshape(-1, 3, 3)
    energy = sum(reference_intra(*mol) for mol in molecules)
    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            r_oo = np.linalg.norm(molecules[i][0] - molecules[j][0])
            if r_oo < lj_cutoff:
                energy += 4 * EPSILON * ((SIGMA / r_oo) ** 12 - (SIGMA / r_oo) ** 6)
            for ri, qi in charged_sites(*molecules[i]):
                for rj, qj in charged_sites(*molecules[j]):
                    energy += K_E * qi * qj / np.linalg.norm(ri - rj)
    return energy


def test_coulomb_constant():
    from metatomic_qtip4pf import COULOMB_CONSTANT

    assert COULOMB_CONSTANT == pytest.approx(K_E, rel=1e-14)


def test_parameters():
    from metatomic_qtip4pf import PARAMETERS

    assert PARAMETERS == {
        "epsilon": EPSILON,
        "sigma": SIGMA,
        "q_M": Q_M,
        "gamma": GAMMA,
        "D_r": D_R,
        "alpha_r": ALPHA_R,
        "r_eq": R_EQ,
        "k_theta": K_THETA,
        "theta_eq": THETA_EQ,
    }


def test_equilibrium_molecule():
    rng = np.random.default_rng(0)
    positions = water_molecule(rng) + rng.normal(size=3)
    energy = energy_only(QTIP4PF(), [8, 1, 1], positions, np.zeros((3, 3)), pbc=False)
    assert abs(energy) < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_intramolecular(seed):
    rng = np.random.default_rng(seed)
    positions = water_molecule(rng, distort=0.1)
    energy = energy_only(QTIP4PF(), [8, 1, 1], positions, np.zeros((3, 3)), pbc=False)
    assert energy == pytest.approx(reference_intra(*positions), rel=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_cluster(seed):
    """Non-periodic cluster: direct LJ + Coulomb, molecules beyond the LJ cutoff"""
    rng = np.random.default_rng(seed)
    centers = [[0, 0, 0], [2.8, 0.3, 0.1], [0.5, 3.0, -0.6], [1.0, 1.0, 9.5]]
    positions = np.concatenate(
        [np.array(c) + water_molecule(rng, distort=0.05) for c in centers]
    )
    types = [8, 1, 1] * len(centers)

    energy = energy_only(QTIP4PF(), types, positions, np.zeros((3, 3)), pbc=False)
    assert energy == pytest.approx(reference_energy(positions), rel=1e-12)


def test_atom_order():
    rng = np.random.default_rng(0)
    positions = water_molecule(rng)
    with pytest.raises(Exception, match="ordered as O, H, H"):
        energy_only(QTIP4PF(), [1, 8, 1], positions, np.zeros((3, 3)), pbc=False)
    with pytest.raises(Exception, match="ordered as O, H, H"):
        energy_only(QTIP4PF(), [8, 1], positions[:2], np.zeros((3, 3)), pbc=False)


def test_mixed_pbc():
    import torch
    from metatomic.torch import System

    rng = np.random.default_rng(0)
    system = System(
        types=torch.tensor([8, 1, 1], dtype=torch.int32),
        positions=torch.tensor(water_molecule(rng)),
        cell=torch.diag(torch.tensor([20.0, 20.0, 0.0], dtype=torch.float64)),
        pbc=torch.tensor([True, True, False]),
    )
    with pytest.raises(Exception, match="fully periodic or non-periodic"):
        QTIP4PF().compute_energy(system)
