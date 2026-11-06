"""q-TIP4P/F water model for metatomic, in pure PyTorch."""

from typing import Dict, Optional

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
)

from ._model import COULOMB_CONSTANT, PARAMETERS, QTIP4PF


try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+unknown"


__all__ = ["COULOMB_CONSTANT", "PARAMETERS", "QTIP4PF", "get_model"]


def get_model(
    cutoff: float = 9.0,
    ewald_accuracy: float = 1e-6,
    parameters: Optional[Dict[str, float]] = None,
) -> AtomisticModel:
    """
    Get the q-TIP4P/F water model as a metatomic :py:class:`AtomisticModel`, which can
    be saved to a standalone file with ``get_model().save("qtip4pf.pt")``.

    Atoms must be ordered as O, H, H, O, H, H, ... The model provides the total
    ``"energy"`` of each system; forces and stress/virial are obtained by the
    simulation engine through automatic differentiation.

    :param cutoff: cutoff (in Angstrom) for the O-O Lennard-Jones interaction and for
        the real-space part of the Ewald sum (applied to O-O distances). The original
        model uses 9 Angstrom.
    :param ewald_accuracy: target relative accuracy of the Ewald summation, used to
        set the Ewald splitting parameter and the reciprocal-space cutoff.
    :param parameters: override some of the parameters in :py:data:`PARAMETERS`.
        This is only intended for testing, the default parameters are the ones from
        the original publication.
    """
    model = QTIP4PF(cutoff=cutoff, ewald_accuracy=ewald_accuracy, parameters=parameters)
    model.eval()

    capabilities = ModelCapabilities(
        outputs={
            "energy": ModelOutput(
                unit="kcal/mol",
                sample_kind="system",
                description="q-TIP4P/F potential energy",
            )
        },
        atomic_types=[1, 8],
        # electrostatics is long-ranged
        interaction_range=float("inf"),
        length_unit="angstrom",
        supported_devices=["cpu", "cuda"],
        dtype="float64",
    )

    metadata = ModelMetadata(
        name="q-TIP4P/F",
        description=(
            "Flexible q-TIP4P/F water model, with Ewald electrostatics. Atoms must "
            "be ordered as O, H, H, O, H, H, ..."
        ),
        authors=["Michele Ceriotti <michele.ceriotti@epfl.ch>"],
        references={
            "model": ["https://doi.org/10.1063/1.3167790"],
            "implementation": ["https://github.com/ceriottm/qtip4pf-test"],
        },
    )

    return AtomisticModel(model, metadata, capabilities)
