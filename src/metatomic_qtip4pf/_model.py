"""
Pure-torch implementation of the q-TIP4P/F water model.

Reference: S. Habershon, T. E. Markland, D. E. Manolopoulos, "Competing quantum
effects in the dynamics of a flexible water model", J. Chem. Phys. 131, 024501
(2009), https://doi.org/10.1063/1.3167790

Internally, energies are in kcal/mol and lengths in Angstrom, so that all the
parameters below are exactly those of Table I in the paper.
"""

import math
from typing import Dict, List, Optional, Tuple

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
        self._gamma = p["gamma"]
        self._D_r = p["D_r"]
        self._alpha_r = p["alpha_r"]
        self._r_eq = p["r_eq"]
        self._k_theta = p["k_theta"]
        self._theta_eq = p["theta_eq"] * math.pi / 180.0
        self._k_e = COULOMB_CONSTANT

        # Charges on (H1, H2, M), and their products. These are created here and not
        # in forward, since `torch.tensor(List[float])` goes through float32 in
        # TorchScript
        q_H = 0.5 * p["q_M"]
        q_M = -p["q_M"]
        charges = torch.tensor([q_H, q_H, q_M], dtype=torch.float64)
        self.register_buffer("_charges", charges)
        self.register_buffer("_qq", charges.reshape(3, 1) * charges.reshape(1, 3))
        # charge products for the (H1, H2), (H2, M), (H1, M) pairs
        self.register_buffer(
            "_intra_qq",
            torch.tensor([q_H * q_H, q_H * q_M, q_H * q_M], dtype=torch.float64),
        )
        self.register_buffer("_types", torch.tensor([8, 1, 1], dtype=torch.int32))

        self._cutoff = float(cutoff)
        # erfc(alpha r) ~ exp(-(alpha r)^2) ~ accuracy at r = cutoff - 2 A, and
        # exp(-k^2 / (4 alpha^2)) ~ accuracy at k = k_cut
        log_acc = math.sqrt(-math.log(ewald_accuracy))
        self._ewald_alpha = log_acc / (self._cutoff - 2.0)
        self._ewald_kcut = 2.0 * self._ewald_alpha * log_acc
        # interaction of each charge with its own gaussian, per molecule
        self._ewald_self = (
            self._k_e * self._ewald_alpha / math.sqrt(math.pi) * (2.0 * q_H**2 + q_M**2)
        )

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

        energies = self.compute_energies(systems)

        device = energies.device
        block = TensorBlock(
            values=energies.reshape(-1, 1),
            samples=Labels(
                ["system"], torch.arange(len(systems), device=device).reshape(-1, 1)
            ),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]], device=device)),
        )
        return {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=device)), [block]
            )
        }

    def compute_energy(self, system: System) -> torch.Tensor:
        """Total q-TIP4P/F energy of a single system, in kcal/mol."""
        return self.compute_energies([system])[0]

    def compute_energies(self, systems: List[System]) -> torch.Tensor:
        """
        Total q-TIP4P/F energy of each system, in kcal/mol. Systems with the same
        number of atoms and periodicity (e.g. the replicas of a path integral
        simulation) are evaluated together.
        """
        n_systems = len(systems)
        for system in systems:
            if len(system) % 3 != 0:
                raise ValueError(
                    "q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ..."
                )

        # Everything that is needed on the host, gathered in a single transfer to
        # avoid multiple synchronizations when running on GPU
        types = torch.cat([system.types for system in systems])
        pbc = torch.stack([system.pbc for system in systems])
        cells = torch.stack([system.cell.detach() for system in systems])
        dtype = cells.dtype
        host: List[float] = torch.cat(
            [
                torch.all(types.reshape(-1, 3) == self._types).reshape(1).to(dtype),
                pbc.reshape(-1).to(dtype),
                torch.linalg.vector_norm(cells, dim=2).reshape(-1),
            ]
        ).tolist()
        if host[0] != 1.0:
            raise ValueError("q-TIP4P/F expects atoms ordered as O, H, H, O, H, H, ...")

        periodic: List[bool] = []
        n_max: List[List[int]] = []
        for i in range(n_systems):
            n_periodic = int(host[1 + 3 * i] + host[2 + 3 * i] + host[3 + 3 * i])
            if n_periodic != 0 and n_periodic != 3:
                raise ValueError(
                    "q-TIP4P/F only supports fully periodic or non-periodic systems"
                )
            periodic.append(n_periodic == 3)
            lengths = host[1 + 3 * n_systems + 3 * i : 4 + 3 * n_systems + 3 * i]
            n_max.append(_ewald_n_max(lengths, self._ewald_kcut))

        all_same = True
        for i in range(n_systems):
            if len(systems[i]) != len(systems[0]) or periodic[i] != periodic[0]:
                all_same = False

        energies: List[torch.Tensor] = []
        if all_same:
            # the reciprocal-space grid must contain the k-vectors of all systems
            grid_n_max = [0, 0, 0]
            for i in range(n_systems):
                for d in range(3):
                    grid_n_max[d] = max(grid_n_max[d], n_max[i][d])

            # On CPU, evaluating many systems together is only faster as long as
            # the intermediate arrays stay small, so systems are evaluated in
            # chunks of about 1000 atoms
            chunk = n_systems
            if cells.device.type == "cpu":
                chunk = max(1, 1000 // max(1, len(systems[0])))

            for start in range(0, n_systems, chunk):
                energies.append(
                    self._energies(
                        systems[start : start + chunk], periodic[0], grid_n_max
                    )
                )
        else:
            for i in range(n_systems):
                energies.append(self._energies([systems[i]], periodic[i], n_max[i]))

        return torch.cat(energies)

    def _energies(
        self, systems: List[System], periodic: bool, n_max: List[int]
    ) -> torch.Tensor:
        """
        Energies of systems with the same number of atoms, which are all periodic
        (``periodic=True``) or all non-periodic.
        """
        n_systems = len(systems)
        positions = torch.stack([system.positions for system in systems])
        cells = torch.stack([system.cell for system in systems])
        device = positions.device
        dtype = positions.dtype
        n_molecules = positions.shape[1] // 3

        # ============================================================================ #
        # Intramolecular geometry
        positions = positions.reshape(n_systems, n_molecules, 3, 3)
        oxygens = positions[:, :, 0]
        # O-H vectors, shape (n_systems, n_molecules, 2, 3)
        oh = positions[:, :, 1:] - positions[:, :, :1]
        # only used for periodic systems
        inverse = cells
        volumes = cells[:, 0, 0]
        if periodic:
            inverse, volumes = _inverse_cell(cells)
            # minimum image, so that molecules can be split across the boundary
            images = torch.round(oh.reshape(n_systems, -1, 3) @ inverse) @ cells
            oh = oh - images.reshape(oh.shape)

        r_oh = torch.linalg.vector_norm(oh, dim=-1)
        d1 = oh[:, :, 0]
        d2 = oh[:, :, 1]
        cos_theta = torch.sum(d1 * d2, dim=-1)
        sin_theta = torch.linalg.vector_norm(torch.linalg.cross(d1, d2, dim=-1), dim=-1)
        theta = torch.atan2(sin_theta, cos_theta)

        a_dr = self._alpha_r * (r_oh - self._r_eq)
        a_dr2 = a_dr * a_dr
        stretch = a_dr2 * (1.0 - a_dr + (7.0 / 12.0) * a_dr2)
        energies = self._D_r * torch.sum(stretch, dim=(1, 2))
        bend = (theta - self._theta_eq) ** 2
        energies = energies + 0.5 * self._k_theta * torch.sum(bend, dim=1)

        # positions of O and of the charged sites H1, H2, M, with shape
        # (n_systems, n_molecules, 4, 3)
        d_M = ((0.5 * (1.0 - self._gamma)) * (d1 + d2)).unsqueeze(2)
        sites = oxygens.unsqueeze(2) + torch.cat(
            [torch.zeros_like(d_M), oh, d_M], dim=2
        )
        qq = self._qq.to(dtype=dtype, device=device)

        # ============================================================================ #
        # Intermolecular interactions. Molecules are indexed over all systems.
        if periodic:
            # only keep O-O pairs from the neighbor lists. Their vectors are recomputed
            # from the positions instead of using `neighbors.values`, so that gradients
            # only flow through these pairs.
            samples: List[torch.Tensor] = []
            shifts: List[torch.Tensor] = []
            system_index: List[torch.Tensor] = []
            for i, system in enumerate(systems):
                neighbors = system.get_neighbor_list(self._nl_options)
                samples.append(neighbors.samples.values)
                shifts.append(neighbors.samples.values[:, 2:].to(dtype) @ system.cell)
                n_pairs = neighbors.samples.values.shape[0]
                system_index.append(
                    torch.full((n_pairs,), i, dtype=torch.int32, device=device)
                )

            is_oxygen = systems[0].types == 8
            pairs = torch.cat(samples)
            oo_pairs = torch.nonzero(is_oxygen[pairs[:, 0]] & is_oxygen[pairs[:, 1]])
            oo_pairs = oo_pairs.reshape(-1)
            pairs = pairs[oo_pairs]
            shifts = torch.cat(shifts)[oo_pairs]
            pair_system = torch.cat(system_index)[oo_pairs]

            first = pair_system * n_molecules + pairs[:, 0] // 3
            second = pair_system * n_molecules + pairs[:, 1] // 3

            pair_energies = self._pair_energies(
                sites.reshape(-1, 4, 3), first, second, shifts, qq, True
            )
            energies = energies.index_add(0, pair_system, pair_energies)

            # reciprocal-space Ewald
            charged = sites[:, :, 1:].reshape(n_systems, -1, 3)
            energies = energies + _ewald_reciprocal(
                charged @ inverse,
                self._charges.to(dtype=dtype, device=device).repeat(n_molecules),
                inverse,
                volumes,
                n_max,
                self._ewald_alpha,
                self._ewald_kcut,
                self._k_e,
            )

            # self-interaction of each charge with its own gaussian
            energies = energies - self._ewald_self * n_molecules

            # remove the intramolecular interactions (H1-H2, H2-M, H1-M) that are
            # included in the reciprocal-space sum
            intra = torch.cat(
                [
                    sites[:, :, 2:] - sites[:, :, 1:3],
                    sites[:, :, 3:] - sites[:, :, 1:2],
                ],
                dim=2,
            )
            intra_r = torch.linalg.vector_norm(intra, dim=-1)
            intra_qq = self._intra_qq.to(dtype=dtype, device=device)
            exclusion = (
                intra_qq * torch.special.erf(self._ewald_alpha * intra_r) / intra_r
            )
            energies = energies - self._k_e * torch.sum(exclusion, dim=(1, 2))
        else:
            # direct sum over all pairs of molecules, no electrostatic cutoff
            pairs = torch.triu_indices(
                n_molecules, n_molecules, offset=1, device=device
            )
            offsets = n_molecules * torch.arange(n_systems, device=device).reshape(
                -1, 1
            )
            pair_system = torch.arange(n_systems, device=device).repeat_interleave(
                pairs.shape[1]
            )
            pair_energies = self._pair_energies(
                sites.reshape(-1, 4, 3),
                (pairs[0] + offsets).reshape(-1),
                (pairs[1] + offsets).reshape(-1),
                torch.zeros((1, 3), dtype=dtype, device=device),
                qq,
                False,
            )
            energies = energies.index_add(0, pair_system, pair_energies)

        return energies

    def _pair_energies(
        self,
        sites: torch.Tensor,
        mol_i: torch.Tensor,
        mol_j: torch.Tensor,
        shifts: torch.Tensor,
        qq: torch.Tensor,
        ewald: bool,
    ) -> torch.Tensor:
        """
        Intermolecular energy for each pair of molecules (``mol_i``, ``mol_j``), with
        ``sites[mol_j] + shifts`` being the periodic image of molecule j interacting
        with molecule i: Lennard-Jones between oxygens within the cutoff, and either
        the real-space part of the Ewald sum (``ewald=True``) or the bare Coulomb
        interaction (``ewald=False``) between charged sites.
        """
        # Positions of (O, H1, H2, M) in molecules i and j, with shape (3, 4, n_pairs).
        # The pairs are the last dimension to make element-wise operations efficient.
        sites = sites.transpose(1, 2).reshape(-1, 12)
        sites_i = sites[mol_i].T.contiguous().reshape(3, 4, -1)
        sites_j = sites[mol_j].T.contiguous().reshape(3, 4, -1)
        sites_j = sites_j + shifts.T.contiguous().reshape(3, 1, -1)

        oo = sites_j[:, 0] - sites_i[:, 0]
        r2_oo = torch.sum(oo * oo, dim=0)
        sr6 = (self._sigma * self._sigma / r2_oo) ** 3
        lj = 4.0 * self._epsilon * (sr6 * sr6 - sr6)

        # vectors between the charged sites, with shape (3, 3, 3, n_pairs)
        vectors = sites_j[:, 1:].reshape(3, 1, 3, -1) - sites_i[:, 1:].reshape(
            3, 3, 1, -1
        )
        r = torch.sqrt(torch.sum(vectors * vectors, dim=0))
        qq = qq.reshape(3, 3, 1)

        inside = r2_oo < self._cutoff * self._cutoff
        if ewald:
            coulomb = qq * torch.special.erfc(self._ewald_alpha * r) / r
            energies = lj + self._k_e * torch.sum(coulomb, dim=(0, 1))
            return torch.where(inside, energies, torch.zeros_like(energies))
        else:
            # Coulomb is not truncated for non-periodic systems
            coulomb = self._k_e * torch.sum(qq / r, dim=(0, 1))
            return coulomb + torch.where(inside, lj, torch.zeros_like(lj))


def _inverse_cell(cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse and volume of a batch of cells, with lattice vectors as rows"""
    # rows of the inverse transpose are (b x c, c x a, a x b) / det
    cross = torch.linalg.cross(cells.roll(-1, dims=1), cells.roll(-2, dims=1), dim=-1)
    det = torch.sum(cells[:, 0] * cross[:, 0], dim=-1)
    return cross.transpose(1, 2) / det.reshape(-1, 1, 1), torch.abs(det)


def _ewald_n_max(lengths: List[float], k_cut: float) -> List[int]:
    """
    Largest integer n_i such that |k| < k_cut is possible for k-vectors with
    components n_i in the basis of the reciprocal lattice, since n_i = k . a_i / 2 pi
    """
    return [int(math.ceil(k_cut * length / (2.0 * math.pi))) for length in lengths]


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
    inverse, volume = _inverse_cell(cell.unsqueeze(0))
    lengths: List[float] = torch.linalg.vector_norm(cell.detach(), dim=1).tolist()
    energy = _ewald_reciprocal(
        (positions @ inverse[0]).unsqueeze(0),
        charges,
        inverse,
        volume,
        _ewald_n_max(lengths, k_cut),
        alpha,
        k_cut,
        k_e,
    )
    return energy[0]


def _ewald_reciprocal(
    fractional: torch.Tensor,
    charges: torch.Tensor,
    inverse: torch.Tensor,
    volumes: torch.Tensor,
    n_max: List[int],
    alpha: float,
    k_cut: float,
    k_e: float,
) -> torch.Tensor:
    """
    Implementation of :py:func:`ewald_reciprocal` for a batch of systems with the
    same number of charges, from the fractional coordinates ``s_i`` of the charges
    (shape ``(n_systems, n_charges, 3)``), the inverse of the cells, their volumes,
    and the largest components of the k-vectors along each reciprocal lattice vector.

    With ``k = 2 pi n @ inverse.T`` for integer ``n``, ``exp(i k r) = exp(2 pi i n_0
    s_0) exp(2 pi i n_1 s_1) exp(2 pi i n_2 s_2)``, and the structure factors on the
    full grid of ``n`` are computed with a single matrix multiplication. This uses
    real arithmetic, since complex operations are slow on CPU.
    """
    device = fractional.device
    dtype = fractional.dtype
    n_systems = fractional.shape[0]
    n_sites = fractional.shape[1]

    # values of n along each direction. Only n_0 >= 0 is needed since S(-k) = S(k)^*
    m = max(n_max)
    n = torch.arange(-m, m + 1, dtype=dtype, device=device)
    range_0 = (m, m + n_max[0] + 1)
    range_1 = (m - n_max[1], m + n_max[1] + 1)
    range_2 = (m - n_max[2], m + n_max[2] + 1)
    size_01 = (range_0[1] - range_0[0]) * (range_1[1] - range_1[0])
    size_2 = range_2[1] - range_2[0]

    # cos and sin of 2 pi n s for all sites, directions and values of n
    angles = (2.0 * math.pi) * fractional.unsqueeze(3) * n
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # exp(2 pi i (n_0 s_0 + n_1 s_1)), for all (n_0, n_1)
    shape_0 = (n_systems, n_sites, -1, 1)
    shape_1 = (n_systems, n_sites, 1, -1)
    cos_0 = cos[:, :, 0, range_0[0] : range_0[1]].reshape(shape_0)
    sin_0 = sin[:, :, 0, range_0[0] : range_0[1]].reshape(shape_0)
    cos_1 = cos[:, :, 1, range_1[0] : range_1[1]].reshape(shape_1)
    sin_1 = sin[:, :, 1, range_1[0] : range_1[1]].reshape(shape_1)
    cos_01 = (cos_0 * cos_1 - sin_0 * sin_1).reshape(n_systems, n_sites, size_01)
    sin_01 = (sin_0 * cos_1 + cos_0 * sin_1).reshape(n_systems, n_sites, size_01)

    # real and imaginary parts of the structure factors S(n_0, n_1, n_2), from
    # the products of the (cos_01, sin_01) and (q cos_2, q sin_2) blocks
    charges = charges.reshape(1, n_sites, 1)
    cos_2 = charges * cos[:, :, 2, range_2[0] : range_2[1]]
    sin_2 = charges * sin[:, :, 2, range_2[0] : range_2[1]]
    products = torch.cat([cos_01, sin_01], dim=2).transpose(1, 2) @ torch.cat(
        [cos_2, sin_2], dim=2
    )
    s_real = products[:, :size_01, :size_2] - products[:, size_01:, size_2:]
    s_imag = products[:, :size_01, size_2:] + products[:, size_01:, :size_2]
    s2 = (s_real * s_real + s_imag * s_imag).reshape(n_systems, -1)

    n_0 = n[range_0[0] : range_0[1]]
    grid = torch.meshgrid(
        [n_0, n[range_1[0] : range_1[1]], n[range_2[0] : range_2[1]]], indexing="ij"
    )
    # k-vectors are computed with element-wise operations instead of a matrix
    # multiplication, which could round differently for different numbers of systems.
    # This way, the k-vectors inside the cutoff are the same when evaluating systems
    # one by one or together.
    grid = torch.stack(grid).reshape(1, 1, 3, -1)
    k_vectors = torch.sum(((2.0 * math.pi) * inverse).unsqueeze(3) * grid, dim=2)
    k2 = torch.sum(k_vectors * k_vectors, dim=1)

    # n = 0 is excluded, and the n_0 = 0 plane contains both k and -k
    inside = torch.logical_and(k2 > 0.0, k2 < k_cut * k_cut)
    k2 = torch.where(inside, k2, torch.ones_like(k2))
    weights = torch.where(inside, torch.exp(-k2 / (4.0 * alpha * alpha)) / k2, 0.0)
    weights = torch.where(grid[0, 0, 0] > 0, weights, 0.5 * weights)

    return (4.0 * math.pi * k_e) / volumes * torch.sum(weights * s2, dim=1)
