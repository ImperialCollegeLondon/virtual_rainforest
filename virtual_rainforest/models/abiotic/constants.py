"""The ``models.abiotic.constants`` module contains a set of dataclasses which contain
parameters required by the broader :mod:`~virtual_rainforest.models.abiotic` model.
These parameters are constants in that they should not be changed during a particular
simulation.
"""  # noqa: D205, D415

from dataclasses import dataclass


@dataclass(frozen=True)
class AbioticConsts:
    """Dataclass to store all constants for the `abiotic` model."""
