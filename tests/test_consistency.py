import numpy as np
import pytest
from conftest import energy_only, evaluate, random_rotation

from metatomic_qtip4pf import QTIP4PF


@pytest.fixture
def model():
    return QTIP4PF(ewald_accuracy=1e-10)


def triclinic(cell):
    """Another (sheared) representation of the same lattice"""
    a, b, c = cell
    return np.array([a, b + a, c - 2 * b + a])


@pytest.mark.parametrize("shape", ["cubic", "triclinic"])
def test_forces_finite_differences(model, small_box, shape):
    types, positions, cell = small_box
    if shape == "triclinic":
        cell = triclinic(cell)

    _, forces, _ = evaluate(model, types, positions, cell)

    delta = 1e-5
    rng = np.random.default_rng(0)
    for atom in rng.choice(len(types), size=6, replace=False):
        for d in range(3):
            plus = positions.copy()
            plus[atom, d] += delta
            minus = positions.copy()
            minus[atom, d] -= delta
            fd = -(
                energy_only(model, types, plus, cell)
                - energy_only(model, types, minus, cell)
            ) / (2 * delta)
            assert forces[atom, d] == pytest.approx(fd, abs=1e-5, rel=1e-6)


@pytest.mark.parametrize("shape", ["cubic", "triclinic"])
def test_virial_finite_differences(model, small_box, shape):
    types, positions, cell = small_box
    if shape == "triclinic":
        cell = triclinic(cell)

    _, _, virial = evaluate(model, types, positions, cell)
    assert np.allclose(virial, virial.T, atol=1e-8)

    delta = 1e-6
    for i in range(3):
        for j in range(3):
            strain = np.eye(3)
            strain[i, j] += delta
            plus = energy_only(model, types, positions @ strain, cell @ strain)
            strain[i, j] -= 2 * delta
            minus = energy_only(model, types, positions @ strain, cell @ strain)
            fd = -(plus - minus) / (2 * delta)
            assert virial[i, j] == pytest.approx(fd, abs=1e-4, rel=1e-6)


def test_invariances(model, small_box):
    types, positions, cell = small_box
    reference, forces, virial = evaluate(model, types, positions, cell)

    # translation
    shifted = positions + np.array([0.3, -1.7, 12.1])
    assert energy_only(model, types, shifted, cell) == pytest.approx(
        reference, rel=1e-12
    )

    # wrapping all atoms inside the cell breaks molecules across the boundary
    fractional = positions @ np.linalg.inv(cell)
    wrapped = (fractional - np.floor(fractional)) @ cell
    assert not np.allclose(wrapped, positions)
    energy, wrapped_forces, wrapped_virial = evaluate(model, types, wrapped, cell)
    assert energy == pytest.approx(reference, rel=1e-12)
    assert np.allclose(wrapped_forces, forces, atol=1e-9)
    assert np.allclose(wrapped_virial, virial, atol=1e-8)

    # rigid rotation of the whole system, including the cell
    rotation = random_rotation(np.random.default_rng(1))
    energy, rotated_forces, rotated_virial = evaluate(
        model, types, positions @ rotation.T, cell @ rotation.T
    )
    assert energy == pytest.approx(reference, rel=1e-10)
    assert np.allclose(rotated_forces, forces @ rotation.T, atol=1e-7)
    assert np.allclose(rotated_virial, rotation @ virial @ rotation.T, atol=1e-6)

    # different representation of the same lattice
    energy, triclinic_forces, triclinic_virial = evaluate(
        model, types, positions, triclinic(cell)
    )
    assert energy == pytest.approx(reference, rel=1e-10)
    assert np.allclose(triclinic_forces, forces, atol=1e-7)
    assert np.allclose(triclinic_virial, virial, atol=1e-6)


def test_supercell(model, small_box):
    types, positions, cell = small_box
    energy, forces, virial = evaluate(model, types, positions, cell)

    super_types = np.concatenate([types, types])
    super_positions = np.concatenate([positions, positions + cell[0]])
    super_cell = cell.copy()
    super_cell[0] *= 2
    super_energy, super_forces, super_virial = evaluate(
        model, super_types, super_positions, super_cell
    )

    assert super_energy == pytest.approx(2 * energy, rel=1e-10)
    assert np.allclose(super_forces[: len(types)], forces, atol=1e-7)
    assert np.allclose(super_virial, 2 * virial, atol=1e-6)


def test_newton_third_law(model, small_box):
    types, positions, cell = small_box
    _, forces, _ = evaluate(model, types, positions, cell)
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-9)
