"""
Pure-torch implementation of the q-TIP4P/F water model.

Reference: S. Habershon, T. E. Markland, D. E. Manolopoulos, "Competing quantum
effects in the dynamics of a flexible water model", J. Chem. Phys. 131, 024501
(2009), https://doi.org/10.1063/1.3167790

Internally, energies are in kcal/mol and lengths in Angstrom, so that all the
parameters below are exactly those of Table I in the paper.
"""

import math
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System


# Table I of Habershon, Markland, Manolopoulos, JCP 131, 024501 (2009)
PARAMETERS: Dict[str, float] = {
    "epsilon": 0.1852,  # kcal/mol, O-O Lennard-Jones well depth
    "sigma": 3.1589,  # Angstrom, O-O Lennard-Jones diameter
    "q_M": 1.1128,  # |e|, magnitude of the (negative) charge on the M site
    "gamma": 0.73612,  # r_M = gamma r_O + (1 - gamma) (r_H1 + r_H2) / 2
    "D_r": 116.09,  # kcal/mol, O-H stretch "Morse" depth
    "alpha_r": 2.287,  # 1/Angstrom, O-H stretch "Morse" range
    "r_eq": 0.9419,  # Angstrom, O-H equilibrium distance
    "k_theta": 87.85,  # kcal/mol/rad^2, H-O-H bending force constant
    "theta_eq": 107.4,  # degrees, H-O-H equilibrium angle
}

# e^2 / (4 pi eps_0) in kcal/mol * Angstrom, from the exact SI values of e, N_A,
# the thermochemical calorie (4.184 J) and eps_0 = 8.8541878128e-12 F/m (CODATA
# 2018). CODATA 2022 changes this by ~5e-10 relative.
COULOMB_CONSTANT = 332.0637132991921


def _lennard_jones(r: torch.Tensor, epsilon: float, sigma: float) -> torch.Tensor:
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6)


class QTIP4PF(torch.nn.Module):
    """
    q-TIP4P/F energy, following the metatomic ``ModelInterface``.

    Atoms must be ordered as O, H, H, O, H, H, ... with each consecutive triplet
    forming one water molecule. Molecules do not need to be whole across periodic
    boundaries.

    - intramolecular: quartic expansion of a Morse potential for each O-H bond plus a
      harmonic H-O-H bend;
    - intermolecular: O-O Lennard-Jones truncated (no shift, no tail correction) at
      ``cutoff``, plus Coulomb interactions between charges on H and on the M site of
      different molecules (all intramolecular electrostatics is excluded).

    For periodic systems, electrostatics uses an Ewald sum. Real-space interactions
    are computed for all pairs of molecules whose O-O distance is below ``cutoff``
    (molecule-based cutoff); the Ewald splitting parameter and the reciprocal-space
    cutoff are chosen so that both real-space and reciprocal-space truncation errors
    are of order ``ewald_accuracy`` (relative). For non-periodic systems,
    electrostatics is an untruncated direct sum.
    """

    def __init__(
        self,
        cutoff: float = 9.0,
        ewald_accuracy: float = 1e-6,
        parameters: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if not (ewald_accuracy > 0.0 and ewald_accuracy < 1.0):
            raise ValueError("ewald_accuracy must be between 0 and 1")
        # Real-space Coulomb includes all site pairs of molecules within `cutoff`
        # (O-O distance), and charged sites are ~1 A away from their oxygen: the
        # smallest excluded site-site distance is thus around `cutoff - 2 A`.
        if cutoff <= 4.0:
            raise ValueError("cutoff must be larger than 4 A")

        p = dict(PARAMETERS)
        if parameters is not None:
            for key, value in parameters.items():
                if key not in p:
                    raise ValueError(f"unknown q-TIP4P/F parameter '{key}'")
                p[key] = float(value)

        self._epsilon = p["epsilon"]
        self._sigma = p["sigma"]
        self._q_H = 0.5 * p["q_M"]
        self._q_M = -p["q_M"]
        self._gamma = p["gamma"]
        self._D_r = p["D_r"]
        self._alpha_r = p["alpha_r"]
        self._r_eq = p["r_eq"]
        self._k_theta = p["k_theta"]
        self._theta_eq = p["theta_eq"] * math.pi / 180.0
        self._k_e = COULOMB_CONSTANT

        # charges on (H1, H2, M). This is created here and not in forward, since
        # `torch.tensor(List[float])` goes through float32 in TorchScript
        self.register_buffer(
            "_charges",
            torch.tensor([self._q_H, self._q_H, self._q_M], dtype=torch.float64),
        )

        self._cutoff = float(cutoff)
        # erfc(alpha r) ~ exp(-(alpha r)^2) ~ accuracy at r = cutoff - 2 A, and
        # exp(-k^2 / (4 alpha^2)) ~ accuracy at k = k_cut
        log_acc = math.sqrt(-math.log(ewald_accuracy))
        self._ewald_alpha = log_acc / (self._cutoff - 2.0)
        self._ewald_kcut = 2.0 * self._ewald_alpha * log_acc

        self._nl_options = NeighborListOptions(
            cutoff=self._cutoff, full_list=False, strict=False
        )

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self._nl_options]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}

        if outputs["energy"].sample_kind != "system":
            raise ValueError("q-TIP4P/F only provides the total energy of each system")

        if selected_atoms is not None:
            n_atoms = 0
            for system in systems:
                n_atoms += len(system)
            if len(selected_atoms) != n_atoms:
                raise ValueError(
                    "q-TIP4P/F can not compute the energy of a subset of atoms"
                )

        energies: List[torch.Tensor] = []
        for system in systems:
            energies.append(self.compute_energy(system))

        device = systems[0].positions.device
        block = TensorBlock(
            values=torch.stack(energies).reshape(-1, 1),
            samples=Labels(
                ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
            ),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]], device=device)),
        )
        return {
            "energy": TensorMap(Labels("_", torch.tensor([[0]], device=device)), [block])
        }

    def compute_energy(self, system: System) -> torch.Tensor:
        """Total q-TIP4P/F energy of a single system, in kcal/mol."""
        types = system.types
        positions = system.positions
        cell = system.cell
        device = positions.device
        dtype = positions.dtype

        n_atoms = types.shape[0]
        if n_atoms % 3 != 0:
            raise ValueError("q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ...")
        n_molecules = n_atoms // 3
        expected = torch.tensor([8, 1, 1], dtype=types.dtype, device=device)
        if not bool(torch.all(types.reshape(n_molecules, 3) == expected)):
            raise ValueError("q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ...")

        n_periodic = int(system.pbc.sum())
        if n_periodic != 0 and n_periodic != 3:
            raise ValueError("q-TIP4P/F only supports fully periodic or non-periodic systems")
        periodic = n_periodic == 3

        # ============================================================================ #
        # Intramolecular geometry
        oxygens = positions[0::3]
        d1 = positions[1::3] - oxygens
        d2 = positions[2::3] - oxygens
        if periodic:
            # minimum image, so that molecules can be split across the boundary
            inv_cell = torch.linalg.inv(cell)
            d1 = d1 - torch.round(d1 @ inv_cell) @ cell
            d2 = d2 - torch.round(d2 @ inv_cell) @ cell

        r1 = torch.linalg.vector_norm(d1, dim=1)
        r2 = torch.linalg.vector_norm(d2, dim=1)
        cos_theta = torch.sum(d1 * d2, dim=1)
        sin_theta = torch.linalg.vector_norm(torch.linalg.cross(d1, d2), dim=1)
        theta = torch.atan2(sin_theta, cos_theta)

        energy = torch.sum(self._stretch(r1) + self._stretch(r2))
        energy = energy + 0.5 * self._k_theta * torch.sum((theta - self._theta_eq) ** 2)

        # positions of the charged sites (H1, H2, M) relative to the oxygen
        d_M = 0.5 * (1.0 - self._gamma) * (d1 + d2)
        offsets = torch.stack([d1, d2, d_M], dim=1)
        charges = self._charges.to(dtype=dtype, device=device)
        qq = charges.reshape(3, 1) * charges.reshape(1, 3)

        # ============================================================================ #
        # Intermolecular interactions
        if periodic:
            neighbors = system.get_neighbor_list(self._nl_options)
            first = neighbors.samples.column("first_atom").to(torch.long)
            second = neighbors.samples.column("second_atom").to(torch.long)
            vectors = neighbors.values.reshape(-1, 3)

            # only keep O-O pairs inside the cutoff
            mask = torch.logical_and(first % 3 == 0, second % 3 == 0)
            first = first[mask]
            second = second[mask]
            vectors = vectors[mask]
            distances = torch.linalg.vector_norm(vectors, dim=1)
            mask = distances < self._cutoff
            mol_i = first[mask] // 3
            mol_j = second[mask] // 3
            vectors = vectors[mask]
            distances = distances[mask]

            energy = energy + torch.sum(
                _lennard_jones(distances, self._epsilon, self._sigma)
            )

            # real-space Ewald, over all pairs of charged sites in these molecules
            site_vectors = (
                vectors.reshape(-1, 1, 1, 3)
                + offsets[mol_j].reshape(-1, 1, 3, 3)
                - offsets[mol_i].reshape(-1, 3, 1, 3)
            )
            site_distances = torch.linalg.vector_norm(site_vectors, dim=-1)
            energy = energy + self._k_e * torch.sum(
                qq * torch.special.erfc(self._ewald_alpha * site_distances)
                / site_distances
            )

            energy = energy + self._ewald_reciprocal(oxygens, offsets, charges, cell)

            # self-interaction of each charge with its own gaussian
            energy = energy - (
                self._k_e
                * self._ewald_alpha
                / math.sqrt(math.pi)
                * n_molecules
                * (2.0 * self._q_H**2 + self._q_M**2)
            )

            # remove the intramolecular interactions (H1-H2, H1-M, H2-M) that are
            # included in the reciprocal-space sum
            intra = torch.stack([d2 - d1, d_M - d1, d_M - d2], dim=1)
            intra_r = torch.linalg.vector_norm(intra, dim=-1)
            # charge products for the (H1, H2), (H1, M), (H2, M) pairs
            intra_qq = qq[[0, 0, 1], [1, 2, 2]]
            energy = energy - self._k_e * torch.sum(
                intra_qq * torch.special.erf(self._ewald_alpha * intra_r) / intra_r
            )
        else:
            # direct sum over all pairs of molecules, no electrostatic cutoff
            pairs = torch.triu_indices(n_molecules, n_molecules, offset=1, device=device)
            mol_i = pairs[0]
            mol_j = pairs[1]
            vectors = oxygens[mol_j] - oxygens[mol_i]
            distances = torch.linalg.vector_norm(vectors, dim=1)

            lj = _lennard_jones(distances, self._epsilon, self._sigma)
            energy = energy + torch.sum(
                torch.where(distances < self._cutoff, lj, torch.zeros_like(lj))
            )

            site_vectors = (
                vectors.reshape(-1, 1, 1, 3)
                + offsets[mol_j].reshape(-1, 1, 3, 3)
                - offsets[mol_i].reshape(-1, 3, 1, 3)
            )
            site_distances = torch.linalg.vector_norm(site_vectors, dim=-1)
            energy = energy + self._k_e * torch.sum(qq / site_distances)

        return energy

    def _stretch(self, r: torch.Tensor) -> torch.Tensor:
        a_dr = self._alpha_r * (r - self._r_eq)
        a_dr2 = a_dr * a_dr
        return self._D_r * (a_dr2 - a_dr2 * a_dr + (7.0 / 12.0) * a_dr2 * a_dr2)

    def _ewald_reciprocal(
        self,
        oxygens: torch.Tensor,
        offsets: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:
        """Reciprocal-space part of the Ewald sum, for charges at oxygens + offsets."""
        sites = (oxygens.reshape(-1, 1, 3) + offsets).reshape(-1, 3)
        site_charges = charges.repeat(oxygens.shape[0])
        return ewald_reciprocal(
            sites, site_charges, cell, self._ewald_alpha, self._ewald_kcut, self._k_e
        )


def ewald_reciprocal(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float,
    k_cut: float,
    k_e: float,
) -> torch.Tensor:
    """
    Reciprocal-space Ewald energy
    ``k_e (2 pi / V) sum_{0 < |k| < k_cut} exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2``
    with ``S(k) = sum_i q_i exp(i k r_i)``. ``cell`` contains the lattice vectors as
    rows, and can be triclinic.
    """
    device = positions.device
    dtype = positions.dtype

    reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).T
    volume = torch.abs(torch.linalg.det(cell))

    # largest integer n_i such that |k| < k_cut is possible, since n_i = k . a_i / 2 pi
    lengths = torch.linalg.vector_norm(cell.detach(), dim=1)
    n_max = torch.ceil(k_cut * lengths / (2.0 * math.pi)).to(torch.long)
    n0 = torch.arange(0, int(n_max[0]) + 1, device=device)
    n1 = torch.arange(-int(n_max[1]), int(n_max[1]) + 1, device=device)
    n2 = torch.arange(-int(n_max[2]), int(n_max[2]) + 1, device=device)
    n = torch.cartesian_prod(n0, n1, n2)

    # only keep half of the k-vectors, since S(-k) = S(k)^*
    half = torch.logical_or(
        n[:, 0] > 0,
        torch.logical_or(
            torch.logical_and(n[:, 0] == 0, n[:, 1] > 0),
            torch.logical_and(
                torch.logical_and(n[:, 0] == 0, n[:, 1] == 0), n[:, 2] > 0
            ),
        ),
    )
    n = n[half]
    k_vectors = n.to(dtype) @ reciprocal
    k2 = torch.sum(k_vectors**2, dim=1)
    inside = k2.detach() < k_cut * k_cut
    k_vectors = k_vectors[inside]
    k2 = k2[inside]

    phases = positions @ k_vectors.T
    s_cos = charges @ torch.cos(phases)
    s_sin = charges @ torch.sin(phases)

    # factor 2 for the k-vectors in the other half of reciprocal space
    prefactor = 4.0 * math.pi * k_e / volume
    return prefactor * torch.sum(
        torch.exp(-k2 / (4.0 * alpha * alpha)) / k2 * (s_cos**2 + s_sin**2)
    )
