import math
import os

import numpy as np
import pytest
import torch
from metatomic.torch import System
from vesin.metatomic import NeighborList


DATA = os.path.join(os.path.dirname(__file__), "data")

# CODATA 2018, consistent with metatomic_qtip4pf.COULOMB_CONSTANT = EH * A0
A0 = 0.529177210903  # Angstrom
EH = 627.5094740631  # kcal/mol

# Parameters hard-coded (in atomic units) in i-PI's drivers/f90/pes/qtip4pf.f90,
# which are rounded versions of Table I in the original paper.
IPI_PARAMETERS = {
    "epsilon": 2.95147e-4 * EH,
    "sigma": 5.96946 * A0,
    "q_M": 1.1128,
    "gamma": 0.73612,
    "D_r": 0.185 * EH,
    "alpha_r": 1.21 / A0,
    "r_eq": 1.78 * A0,
    "k_theta": 2 * 0.07 * EH,
    "theta_eq": 107.4,
}


def read_xyz(path):
    """Minimal extended XYZ reader: returns types, positions, cell."""
    with open(path) as fd:
        lines = fd.read().splitlines()
    n_atoms = int(lines[0])
    lattice = lines[1].split('Lattice="')[1].split('"')[0]
    cell = np.array([float(x) for x in lattice.split()]).reshape(3, 3)
    types, positions = [], []
    for line in lines[2 : 2 + n_atoms]:
        name, x, y, z = line.split()[:4]
        types.append({"O": 8, "H": 1}[name])
        positions.append([float(x), float(y), float(z)])
    return np.array(types), np.array(positions), cell


def make_system(module, types, positions, cell, pbc=True):
    """Create a metatomic System, with neighbor lists for `module`."""
    device = positions.device
    system = System(
        types=torch.as_tensor(types, dtype=torch.int32, device=device),
        positions=positions,
        cell=cell,
        pbc=torch.tensor([pbc, pbc, pbc], device=device),
    )
    for options in module.requested_neighbor_lists():
        NeighborList(options, "angstrom").add_neighbor_list([system])
    return system


def evaluate(module, types, positions, cell, pbc=True):
    """
    Energy (kcal/mol), forces (kcal/mol/A) and virial (-dE/dstrain, kcal/mol) from
    the eager module, computing derivatives with respect to positions and strain in
    the same way as i-PI does with metatomic models.
    """
    positions = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
    cell = torch.tensor(cell, dtype=torch.float64)
    strain = torch.eye(3, dtype=torch.float64, requires_grad=True)
    system = make_system(module, types, positions @ strain, cell @ strain, pbc)
    energy = module.compute_energy(system)
    if pbc:
        grad_pos, grad_strain = torch.autograd.grad(energy, [positions, strain])
    else:
        (grad_pos,) = torch.autograd.grad(energy, [positions])
        grad_strain = torch.full((3, 3), float("nan"), dtype=torch.float64)
    return energy.item(), -grad_pos.numpy(), -grad_strain.numpy()


def energy_only(module, types, positions, cell, pbc=True):
    positions = torch.tensor(positions, dtype=torch.float64)
    cell = torch.tensor(cell, dtype=torch.float64)
    with torch.no_grad():
        system = make_system(module, types, positions, cell, pbc)
        return module.compute_energy(system).item()


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ]
    )


def water_molecule(rng, r1=0.9419, r2=0.9419, theta=107.4, distort=0.0):
    """O, H, H positions of a randomly oriented water molecule centered on O."""
    r1 = r1 + distort * rng.normal()
    r2 = r2 + distort * rng.normal()
    theta = math.radians(theta + 50 * distort * rng.normal())
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [r1, 0.0, 0.0],
            [r2 * math.cos(theta), r2 * math.sin(theta), 0.0],
        ]
    )
    return local @ random_rotation(rng).T


@pytest.fixture
def small_box():
    """8 distorted water molecules on a cubic lattice with ~1 g/cm^3 density."""
    rng = np.random.default_rng(42)
    spacing = 3.1
    positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                center = spacing * np.array([i, j, k]) + 0.2 * rng.normal(size=3)
                positions.append(center + water_molecule(rng, distort=0.03))
    types = np.array([8, 1, 1] * 8)
    return types, np.concatenate(positions), np.eye(3) * 2 * spacing


@pytest.fixture
def water_216():
    return read_xyz(os.path.join(DATA, "water_216.xyz"))
