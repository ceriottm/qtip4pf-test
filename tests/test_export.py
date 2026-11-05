import math

import numpy as np
import pytest
import torch
from conftest import make_system
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    load_atomistic_model,
)

from metatomic_qtip4pf import get_model
from metatomic_qtip4pf._cli import main as cli_main


KCAL_MOL_TO_EV = 4184.0 / (6.02214076e23 * 1.602176634e-19)


def run(model, systems, unit="eV", sample_kind="system"):
    options = ModelEvaluationOptions(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(unit=unit, sample_kind=sample_kind)},
    )
    outputs = model(systems, options, check_consistency=True)
    return outputs["energy"].block().values.reshape(-1)


def test_export(tmp_path, small_box, water_216):
    model = get_model()
    assert math.isinf(model.capabilities().interaction_range)

    path = str(tmp_path / "qtip4pf.pt")
    model.save(path)
    loaded = load_atomistic_model(path)

    systems = []
    for types, positions, cell in [small_box, water_216]:
        systems.append(
            make_system(
                loaded,
                types,
                torch.tensor(positions, requires_grad=True),
                torch.tensor(cell),
            )
        )

    energies = run(loaded, systems)
    for system, energy in zip(systems, energies):
        expected = model.module.compute_energy(system) * KCAL_MOL_TO_EV
        assert energy.item() == pytest.approx(expected.item(), rel=1e-12)

    # the exported model can compute forces through autograd
    energies.sum().backward()
    assert systems[0].positions.grad is not None
    assert torch.all(torch.isfinite(systems[1].positions.grad))

    # non-periodic systems
    types, positions, _ = small_box
    system = make_system(
        loaded,
        types,
        torch.tensor(positions),
        torch.zeros((3, 3), dtype=torch.float64),
        pbc=False,
    )
    energy = run(loaded, [system], unit="kcal/mol")
    assert energy.item() == pytest.approx(
        model.module.compute_energy(system).item(), rel=1e-12
    )


def test_per_atom(small_box):
    model = get_model()
    types, positions, cell = small_box
    system = make_system(model, types, torch.tensor(positions), torch.tensor(cell))
    with pytest.raises(Exception):
        run(model, [system], sample_kind="atom")


def test_cli(tmp_path):
    path = tmp_path / "model.pt"
    cli_main(["-o", str(path), "--ewald-accuracy", "1e-8"])
    loaded = load_atomistic_model(str(path))
    assert loaded.metadata().name == "q-TIP4P/F"
    assert np.isinf(loaded.capabilities().interaction_range)
