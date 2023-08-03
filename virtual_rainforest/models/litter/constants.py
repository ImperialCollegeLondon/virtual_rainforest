"""The :mod:`~virtual_rainforest.models.litter.constants` module contains
constants and parameters for the
:mod:`~virtual_rainforest.models.litter.litter_model`. These parameters are constants
in that they should not be changed during a particular simulation.
"""  # noqa: D205, D415

from dataclasses import dataclass


@dataclass(frozen=True)
class LitterConsts:
    """Dataclass to store all constants for the `litter` model."""

    litter_decomp_reference_temp: float = 40.0
    """Reference temperature for litter decomposition [C].

    Value is taken from :cite:t:`kirschbaum_modelling_2002`.
    """

    litter_decomp_offset_temp: float = 31.79
    """Offset temperature for litter decomposition [C].

    Value is taken from :cite:t:`kirschbaum_modelling_2002`.
    """

    litter_decomp_temp_response: float = 3.36
    """Parameter controlling the temperature response strength of litter decomposition.

    [unitless]. Value is taken from :cite:t:`kirschbaum_modelling_2002`.
    """

    litter_decay_constant_metabolic_above: float = 0.56 / 7.0
    """Decay constant for the above ground metabolic litter pool [day^-1].

    Value is taken from :cite:t:`kirschbaum_modelling_2002`.
    """

    litter_decay_constant_structural_above: float = 0.152 / 7.0
    """Decay constant for the above ground structural litter pool [day^-1].

    Value is taken from :cite:t:`kirschbaum_modelling_2002`.
    """

    litter_input_to_metabolic_above: float = 0.000280628
    """Litter input rate to metabolic above ground litter pool [kg C m^-2 day^-1].

    This value was estimated (very unsystematically) from SAFE project data. This
    constant will eventually be removed once the litter is linked to other models.
    """

    litter_input_to_structural_above: float = 0.00071869
    """Litter input rate to metabolic above ground litter pool [kg C m^-2 day^-1].

    This value was estimated (very unsystematically) from SAFE project data. This
    constant will eventually be removed once the litter is linked to other models.
    """