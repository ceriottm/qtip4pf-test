"""
Several systems evaluated together give the same results as one at a time, and
results on GPU are the same as on CPU.
"""

import numpy as np
import pytest
import torch
from conftest import make_system

from metatomic_qtip4pf import QTIP4PF


def triclinic(cell):
    a, b, c = cell
    return np.array([a, b + a, c - 2 * b + a])


@pytest.fixture
def model():
    return QTIP4PF(ewald_accuracy=1e-8)


def structures(small_box, water_216):
    """Systems with the same number of atoms but different cells and positions"""
    types, positions, cell = small_box
    rng = np.random.default_rng(0)
    return [
        (types, positions, cell, True),
        (types, positions + 0.05 * rng.normal(size=positions.shape), cell, True),
        # different cell shape and size (and so different reciprocal-space grid)
        (types, 1.1 * positions, 1.1 * triclinic(cell), True),
        (types, positions, np.zeros((3, 3)), False),
        (*water_216, True),
    ]


def energies_and_gradients(model, structures, device="cpu"):
    systems, inputs = [], []
    for types, positions, cell, pbc in structures:
        positions = torch.tensor(positions, device=device, requires_grad=True)
        cell = torch.tensor(cell, device=device, requires_grad=pbc)
        systems.append(make_system(model, types, positions, cell, pbc))
        inputs += [positions, cell] if pbc else [positions]

    energies = model.compute_energies(systems)
    gradients = torch.autograd.grad(energies.sum(), inputs)
    return energies.detach().cpu(), [g.cpu() for g in gradients]


@pytest.mark.parametrize(
    "selection",
    [
        [0, 1, 2],  # periodic, same size: evaluated together
        [3, 3],  # non-periodic, same size: evaluated together
        [0, 3, 4, 1],  # different sizes and periodicity: evaluated one by one
    ],
)
def test_batch(model, small_box, water_216, selection):
    all_structures = structures(small_box, water_216)
    selected = [all_structures[i] for i in selection]

    energies, gradients = energies_and_gradients(model, selected)

    expected_energies, expected_gradients = [], []
    for structure in selected:
        energy, grads = energies_and_gradients(model, [structure])
        expected_energies.append(energy)
        expected_gradients += grads

    # The results are not bitwise identical, since math functions on CPU can give
    # slightly different results (~1e-11 relative) between calls
    assert torch.allclose(energies, torch.cat(expected_energies), rtol=1e-10)
    for gradient, expected in zip(gradients, expected_gradients, strict=True):
        assert torch.allclose(gradient, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda(model, small_box, water_216):
    selected = structures(small_box, water_216)
    energies, gradients = energies_and_gradients(model, selected)

    model = model.to("cuda")
    cuda_energies, cuda_gradients = energies_and_gradients(model, selected, "cuda")

    assert torch.allclose(cuda_energies, energies, rtol=1e-12)
    for gradient, expected in zip(cuda_gradients, gradients, strict=True):
        assert torch.allclose(gradient, expected, rtol=1e-9, atol=1e-9)
