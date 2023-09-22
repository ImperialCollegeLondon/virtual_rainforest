"""The ``models.soil.constants`` module contains a set of dataclasses containing
constants (fitting relationships taken from the literature) required by the broader
:mod:`~virtual_rainforest.models.soil` module
"""  # noqa: D205, D415

from dataclasses import dataclass

# TODO - Need to figure out a sensible area to volume conversion


@dataclass(frozen=True)
class SoilConsts:
    """Dataclass to store all constants for the `soil` model.

    All constants are taken from :cite:t:`abramoff_millennial_2018` unless otherwise
    stated.
    """

    binding_with_ph_slope: float = -0.186
    """Change in the binding affinity of soil mineral with pH.

    Units of [pH^-1]. From linear regression :cite:p:`mayes_relation_2012`."""

    binding_with_ph_intercept: float = -0.216 + 3.0
    """Binding affinity of soil minerals at zero pH.

    Unit of [log(m^3 kg^-1)]. n.b. +3 converts from mg^-1 to kg^-1 and L to m^3. From
    linear regression :cite:p:`mayes_relation_2012`.
    """

    max_sorption_with_clay_slope: float = 0.483
    """Change in the maximum size of the MAOM pool with increasing clay content.

    Units of [(% clay)^-1]. From linear regression :cite:p:`mayes_relation_2012`.
    """

    max_sorption_with_clay_intercept: float = 2.328 - 6.0
    """Maximum size of the MAOM pool at zero clay content.

    Unit of [log(kg C kg soil ^-1)]. n.b. -6 converts from mg to kg. From linear
    regression :cite:p:`mayes_relation_2012`.
    """

    moisture_scalar_coefficient: float = 30.0
    """Used in :cite:t:`abramoff_millennial_2018`, can't find original source.

    Value at zero relative water content (RWC) [unit less].
    """

    moisture_scalar_exponent: float = 9.0
    """Used in :cite:t:`abramoff_millennial_2018`, can't find original source.

    Units of [(RWC)^-1]
    """

    reference_cue: float = 0.6
    """Carbon use efficiency of community at the reference temperature [no units].

    Default value taken from :cite:t:`abramoff_millennial_2018`.
    """

    cue_reference_temp: float = 15.0
    """Reference temperature for carbon use efficiency [degrees C].

    Default value taken from :cite:t:`abramoff_millennial_2018`.
    """

    cue_with_temperature: float = 0.012
    """Change in carbon use efficiency with increasing temperature [degree C^-1].

    Default value taken from :cite:t:`abramoff_millennial_2018`.
    """

    microbial_turnover_rate: float = 0.036
    """Microbial turnover rate [day^-1], this isn't a constant but often treated as one.
    """

    max_uptake_rate_labile_C: float = 0.35
    """Maximum (theoretical) rate at which microbes can take up labile carbon [day^-1].
    """

    necromass_adsorption_rate: float = 0.025
    """Rate at which necromass is adsorbed by soil minerals [day^-1].

    Taken from :cite:t:`abramoff_millennial_2018`, where it was obtained by calibration.
    """

    half_sat_microbial_activity: float = 0.0072
    """Half saturation constant for microbial activity (with increasing biomass).

    Units of [kg C m^-2].
    """

    half_sat_microbial_pom_mineralisation: float = 0.012
    """Half saturation constant for microbial POM mineralisation [kg C m^-2]."""

    max_decomp_rate_pom: float = 0.01
    """Maximum (theoretical) rate for particulate organic matter break down.

    Units of [kg C m^-2 day^-1]. Taken from :cite:t:`abramoff_millennial_2018`, where it
    was obtained by calibration.
    """

    leaching_rate_labile_carbon: float = 1.5e-3
    """Leaching rate for labile carbon (lmwc) [day^-1]."""

    half_sat_pom_decomposition: float = 0.150
    """Half saturation constant for POM decomposition to LMWC [kg C m^-2]."""

    universal_gas_constant: float = 8.314
    """Universal gas constant [J mol^-1 K^-1].

    TODO - This is definitely a core constant
    """

    arrhenius_reference_temp: float = 12.0
    """Reference temperature for the Arrhenius equation [C].

    TODO - When I've added the constants that this reference temperature pertains to,
    mention them here.
    """
