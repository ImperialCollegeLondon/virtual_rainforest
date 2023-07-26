"""The ``models.soil.carbon`` module  simulates the soil carbon cycle for the Virtual
Rainforest. At the moment only two pools are modelled, these are low molecular weight
carbon (LMWC) and mineral associated organic matter (MAOM). More pools and their
interactions will be added at a later date.
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray

from virtual_rainforest.core.logger import LOGGER
from virtual_rainforest.models.soil.constants import SoilConsts


def calculate_soil_carbon_updates(
    soil_c_pool_lmwc: NDArray[np.float32],
    soil_c_pool_maom: NDArray[np.float32],
    soil_c_pool_microbe: NDArray[np.float32],
    soil_c_pool_pom: NDArray[np.float32],
    pH: NDArray[np.float32],
    bulk_density: NDArray[np.float32],
    soil_moisture: NDArray[np.float32],
    soil_temp: NDArray[np.float32],
    percent_clay: NDArray[np.float32],
    delta_pools_ordered: dict[str, NDArray[np.float32]],
    constants: SoilConsts,
) -> NDArray[np.float32]:
    """Calculate net change for each carbon pool.

    This function calls lower level functions which calculate the transfers between
    pools. When all transfers have been calculated the net transfer is used to
    calculate the net change for each pool.

    Args:
        soil_c_pool_lmwc: Low molecular weight carbon pool [kg C m^-3]
        soil_c_pool_maom: Mineral associated organic matter pool [kg C m^-3]
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        soil_c_pool_pom: Particulate organic matter pool [kg C m^-3]
        pH: pH values for each soil grid cell
        bulk_density: bulk density values for each soil grid cell [kg m^-3]
        soil_moisture: relative water content for each soil grid cell [unitless]
        soil_temp: soil temperature for each soil grid cell [degrees C]
        percent_clay: Percentage clay for each soil grid cell
        delta_pools_ordered: Dictionary to store pool changes in the order that pools
            are stored in the initial condition vector.
        constants: Set of constants for the soil model.

    Returns:
        A vector containing net changes to each pool. Order [lmwc, maom].
    """
    # TODO - Add interactions which involve the three missing carbon pools

    # Find scalar factors that multiple rates
    temp_scalar = convert_temperature_to_scalar(
        soil_temp,
        constants.temp_scalar_coefficient_1,
        constants.temp_scalar_coefficient_2,
        constants.temp_scalar_coefficient_3,
        constants.temp_scalar_coefficient_4,
        constants.temp_scalar_reference_temp,
    )
    moist_scalar = convert_moisture_to_scalar(
        soil_moisture,
        constants.moisture_scalar_coefficient,
        constants.moisture_scalar_exponent,
    )
    moist_temp_scalar = moist_scalar * temp_scalar

    # Calculate transfers between pools
    lmwc_to_maom = calculate_mineral_association(
        soil_c_pool_lmwc,
        soil_c_pool_maom,
        pH,
        bulk_density,
        moist_temp_scalar,
        percent_clay,
        constants,
    )
    microbial_uptake = calculate_microbial_carbon_uptake(
        soil_c_pool_lmwc, soil_c_pool_microbe, moist_temp_scalar, soil_temp, constants
    )
    microbial_respiration = calculate_maintenance_respiration(
        soil_c_pool_microbe, moist_temp_scalar, constants.microbial_turnover_rate
    )
    necromass_adsorption = calculate_necromass_adsorption(
        soil_c_pool_microbe, moist_temp_scalar, constants.necromass_adsorption_rate
    )
    labile_carbon_leaching = calculate_labile_carbon_leaching(
        soil_c_pool_lmwc, moist_temp_scalar, constants.leaching_rate_labile_carbon
    )
    litter_input_to_lmwc, litter_input_to_pom = calculate_direct_litter_input_to_pools(
        constants.carbon_input_to_pom, constants.litter_input_rate
    )
    pom_decomposition_to_lmwc = calculate_pom_decomposition(
        soil_c_pool_pom, soil_c_pool_microbe, moist_temp_scalar, constants
    )

    # Determine net changes to the pools
    delta_pools_ordered["soil_c_pool_lmwc"] = (
        litter_input_to_lmwc
        + pom_decomposition_to_lmwc
        - lmwc_to_maom
        - microbial_uptake
        - labile_carbon_leaching
    )
    delta_pools_ordered["soil_c_pool_maom"] = lmwc_to_maom + necromass_adsorption
    delta_pools_ordered["soil_c_pool_microbe"] = (
        microbial_uptake - microbial_respiration - necromass_adsorption
    )
    delta_pools_ordered["soil_c_pool_pom"] = (
        litter_input_to_pom - pom_decomposition_to_lmwc
    )

    # Create output array of pools in desired order
    return np.concatenate(list(delta_pools_ordered.values()))


def calculate_mineral_association(
    soil_c_pool_lmwc: NDArray[np.float32],
    soil_c_pool_maom: NDArray[np.float32],
    pH: NDArray[np.float32],
    bulk_density: NDArray[np.float32],
    moist_temp_scalar: NDArray[np.float32],
    percent_clay: NDArray[np.float32],
    constants: SoilConsts,
) -> NDArray[np.float32]:
    """Calculates net rate of LMWC association with soil minerals.

    Following :cite:t:`abramoff_millennial_2018`, mineral adsorption of carbon is
    controlled by a Langmuir saturation function. At present, binding affinity and
    Q_max are recalculated on every function called based on pH, bulk density and
    clay content. Once a decision has been reached as to how fast pH and bulk
    density will change (if at all), this calculation may need to be moved
    elsewhere.

    Args:
        soil_c_pool_lmwc: Low molecular weight carbon pool [kg C m^-3]
        soil_c_pool_maom: Mineral associated organic matter pool [kg C m^-3]
        pH: pH values for each soil grid cell
        bulk_density: bulk density values for each soil grid cell [kg m^-3]
        moist_temp_scalar: A scalar capturing the combined impact of soil moisture and
            temperature on process rates
        percent_clay: Percentage clay for each soil grid cell
        constants: Set of constants for the soil model.

    Returns:
        The net flux from LMWC to MAOM [kg C m^-3 day^-1]
    """

    # Calculate maximum sorption
    Q_max = calculate_max_sorption_capacity(
        bulk_density,
        percent_clay,
        constants.max_sorption_with_clay_slope,
        constants.max_sorption_with_clay_intercept,
    )
    equib_maom = calculate_equilibrium_maom(pH, Q_max, soil_c_pool_lmwc, constants)

    return (
        moist_temp_scalar * soil_c_pool_lmwc * (equib_maom - soil_c_pool_maom) / Q_max
    )


def calculate_max_sorption_capacity(
    bulk_density: NDArray[np.float32],
    percent_clay: NDArray[np.float32],
    max_sorption_with_clay_slope: float,
    max_sorption_with_clay_intercept: float,
) -> NDArray[np.float32]:
    """Calculate maximum sorption capacity based on bulk density and clay content.

    The maximum sorption capacity is the maximum amount of mineral associated organic
    matter that can exist per unit volume. This expression and its parameters are also
    drawn from :cite:t:`mayes_relation_2012`. In that paper max sorption also depends on
    Fe content, but we are ignoring this for now.

    Args:
        bulk_density: bulk density values for each soil grid cell [kg m^-3]
        percent_clay: Percentage clay for each soil grid cell
        max_sorption_with_clay_slope: Slope of relationship between clay content and
            maximum organic matter sorption [(% clay)^-1]
        max_sorption_with_clay_intercept: Intercept of relationship between clay content
            and maximum organic matter sorption [log(kg C kg soil ^-1)]

    Returns:
        Maximum sorption capacity [kg C m^-3]
    """

    # Check that negative initial values are not given
    if np.any(percent_clay > 100.0) or np.any(percent_clay < 0.0):
        to_raise = ValueError(
            "Relative clay content must be expressed as a percentage!"
        )
        LOGGER.error(to_raise)
        raise to_raise

    Q_max = bulk_density * 10 ** (
        max_sorption_with_clay_slope * np.log10(percent_clay)
        + max_sorption_with_clay_intercept
    )
    return Q_max


def calculate_equilibrium_maom(
    pH: NDArray[np.float32],
    Q_max: NDArray[np.float32],
    lmwc: NDArray[np.float32],
    constants: SoilConsts,
) -> NDArray[np.float32]:
    """Calculate equilibrium MAOM concentration based on Langmuir coefficients.

    Equilibrium concentration of mineral associated organic matter (MAOM) is calculated
    by this function under the assumption that the concentration of low molecular weight
    carbon (LMWC) is fixed.

    Args:
        pH: pH values for each soil grid cell
        Q_max: Maximum sorption capacities [kg C m^-3]
        lmwc: Low molecular weight carbon pool [kg C m^-3]
        constants: Set of constants for the soil model.

    Returns:
        Equilibrium concentration of MAOM [kg C m^-3]
    """

    binding_coefficient = calculate_binding_coefficient(
        pH, constants.binding_with_ph_slope, constants.binding_with_ph_intercept
    )
    return (binding_coefficient * Q_max * lmwc) / (1 + lmwc * binding_coefficient)


def calculate_binding_coefficient(
    pH: NDArray[np.float32],
    binding_with_ph_slope: float,
    binding_with_ph_intercept: float,
) -> NDArray[np.float32]:
    """Calculate Langmuir binding coefficient based on pH.

    This specific expression and its parameters are drawn from
    :cite:t:`mayes_relation_2012`.

    Args:
        pH: pH values for each soil grid cell
        binding_with_ph_slope: Slope of relationship between pH and binding coefficient
            [pH^-1]
        binding_with_ph_intercept: Intercept of relationship between pH and binding
            coefficient [log(m^3 kg^-1)]

    Returns:
        Langmuir binding coefficients for mineral association of labile carbon [m^3
        kg^-1]
    """

    return 10.0 ** (binding_with_ph_slope * pH + binding_with_ph_intercept)


def convert_temperature_to_scalar(
    soil_temp: NDArray[np.float32],
    temp_scalar_coefficient_1: float,
    temp_scalar_coefficient_2: float,
    temp_scalar_coefficient_3: float,
    temp_scalar_coefficient_4: float,
    temp_scalar_reference_temp: float,
) -> NDArray[np.float32]:
    """Convert soil temperature into a factor to multiply rates by.

    This form is used in :cite:t:`abramoff_millennial_2018` to minimise differences with
    the CENTURY model. We very likely want to define our own functional form here. I'm
    also a bit unsure how this form was even obtained, so further work here is very
    needed.

    Args:
        soil_temp: soil temperature for each soil grid cell [degrees C]
        temp_scalar_coefficient_1: Unclear exactly what this parameter is [degrees C]
        temp_scalar_coefficient_2: Unclear exactly what this parameter is [unclear]
        temp_scalar_coefficient_3: Unclear exactly what this parameter is [unclear]
        temp_scalar_coefficient_4: Unclear exactly what this parameter is [unclear]
        temp_scalar_reference_temp: Reference temperature for temperature scalar
            [degrees C]

    Returns:
        A scalar that captures the impact of soil temperature on process rates
    """

    # This expression is drawn from Abramoff et al. (2018)
    numerator = temp_scalar_coefficient_2 + (
        temp_scalar_coefficient_3 / np.pi
    ) * np.arctan(np.pi * (soil_temp - temp_scalar_coefficient_1))

    denominator = temp_scalar_coefficient_2 + (
        temp_scalar_coefficient_3 / np.pi
    ) * np.arctan(
        np.pi
        * temp_scalar_coefficient_4
        * (temp_scalar_reference_temp - temp_scalar_coefficient_1)
    )

    return np.divide(numerator, denominator)


def convert_moisture_to_scalar(
    soil_moisture: NDArray[np.float32],
    moisture_scalar_coefficient: float,
    moisture_scalar_exponent: float,
) -> NDArray[np.float32]:
    """Convert soil moisture into a factor to multiply rates by.

    This form is used in :cite:t:`abramoff_millennial_2018` to minimise differences with
    the CENTURY model. We very likely want to define our own functional form here. I'm
    also a bit unsure how this form was even obtained, so further work here is very
    needed.

    Args:
        soil_moisture: relative water content for each soil grid cell [unitless]
        moisture_scalar_coefficient: [unit less]
        moisture_scalar_exponent: [(Relative water content)^-1]

    Returns:
        A scalar that captures the impact of soil moisture on process rates
    """

    if np.any(soil_moisture > 1.0) or np.any(soil_moisture < 0.0):
        to_raise = ValueError(
            "Relative water content cannot go below zero or above one!"
        )
        LOGGER.error(to_raise)
        raise to_raise

    # This expression is drawn from Abramoff et al. (2018)
    return 1 / (
        1
        + moisture_scalar_coefficient
        * np.exp(-moisture_scalar_exponent * soil_moisture)
    )


def calculate_maintenance_respiration(
    soil_c_pool_microbe: NDArray[np.float32],
    moist_temp_scalar: NDArray[np.float32],
    microbial_turnover_rate: float,
) -> NDArray[np.float32]:
    """Calculate the maintenance respiration of the microbial pool.

    Args:
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        moist_scalar: A scalar capturing the combined impact of soil moisture and
            temperature on process rates
        microbial_turnover_rate: Rate of microbial biomass turnover [day^-1]

    Returns:
        Total respiration for all microbial biomass
    """

    return microbial_turnover_rate * moist_temp_scalar * soil_c_pool_microbe


def calculate_necromass_adsorption(
    soil_c_pool_microbe: NDArray[np.float32],
    moist_temp_scalar: NDArray[np.float32],
    necromass_adsorption_rate: float,
) -> NDArray[np.float32]:
    """Calculate adsorption of microbial necromass to soil minerals.

    Args:
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        moist_temp_scalar: A scalar capturing the combined impact of soil moisture and
            temperature on process rates
        necromass_adsorption_rate: Rate at which necromass is adsorbed by soil minerals

    Returns:
        Adsorption of microbial biomass to mineral associated organic matter (MAOM)
    """

    return necromass_adsorption_rate * moist_temp_scalar * soil_c_pool_microbe


def calculate_carbon_use_efficiency(
    soil_temp: NDArray[np.float32],
    reference_cue: float,
    cue_reference_temp: float,
    cue_with_temperature: float,
) -> NDArray[np.float32]:
    """Calculate the (temperature dependant) carbon use efficiency.

    Args:
        soil_temp: soil temperature for each soil grid cell [degrees C]
        reference_cue: Carbon use efficiency at reference temp [unitless]
        cue_reference_temp: Reference temperature [degrees C]
        cue_with_temperature: Rate of change in carbon use efficiency with increasing
            temperature [degree C^-1]

    Returns:
        The carbon use efficiency (CUE) of the microbial community
    """

    return reference_cue - cue_with_temperature * (soil_temp - cue_reference_temp)


def calculate_microbial_saturation(
    soil_c_pool_microbe: NDArray[np.float32],
    half_sat_microbial_activity: float,
) -> NDArray[np.float32]:
    """Calculate microbial activity saturation.

    This ensures that microbial activity (per unit biomass) drops as biomass density
    increases. This is adopted from Abramoff et al. It feels like an assumption that
    should be revised as the Virtual Rainforest develops.

    Args:
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        half_sat_microbial_activity: Half saturation constant for microbial activity

    Returns:
        A rescaling of microbial biomass that takes into account activity saturation
        with increasing biomass density
    """

    return soil_c_pool_microbe / (soil_c_pool_microbe + half_sat_microbial_activity)


def calculate_microbial_pom_mineralisation_saturation(
    soil_c_pool_microbe: NDArray[np.float32],
    half_sat_microbial_mineralisation: float,
) -> NDArray[np.float32]:
    """Calculate microbial POM mineralisation saturation (with increasing biomass).

    This ensures that microbial mineralisation of POM (per unit biomass) drops as
    biomass density increases. This is adopted from Abramoff et al. This function is
    very similar to the
    :func:`~virtual_rainforest.models.soil.carbon.calculate_microbial_saturation`
    function. They could in theory be reworked into a single function, but it doesn't
    seem worth the effort as we do not anticipate using biomass saturation functions
    beyond the first model draft.

    Args:
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        half_sat_microbial_mineralisation: Half saturation constant for microbial
            mineralisation of POM

    Returns:
        A rescaling of microbial biomass that takes into account POM mineralisation rate
        saturation with increasing biomass density
    """

    return soil_c_pool_microbe / (
        soil_c_pool_microbe + half_sat_microbial_mineralisation
    )


def calculate_pom_decomposition_saturation(
    soil_c_pool_pom: NDArray[np.float32],
    half_sat_pom_decomposition: float,
) -> NDArray[np.float32]:
    """Calculate particulate organic matter (POM) decomposition saturation.

    This ensures that decomposition of POM to low molecular weight carbon (LMWC)
    saturates with increasing POM. This effect arises from the saturation of enzymes
    with increasing substrate.

    Args:
        soil_c_pool_pom: Particulate organic matter (carbon) pool [kg C m^-3]
        half_sat_pom_decomposition: Half saturation constant for POM decomposition

    Returns:
        The saturation of the decomposition process
    """

    return soil_c_pool_pom / (soil_c_pool_pom + half_sat_pom_decomposition)


def calculate_microbial_carbon_uptake(
    soil_c_pool_lmwc: NDArray[np.float32],
    soil_c_pool_microbe: NDArray[np.float32],
    moist_temp_scalar: NDArray[np.float32],
    soil_temp: NDArray[np.float32],
    constants: SoilConsts,
) -> NDArray[np.float32]:
    """Calculate amount of labile carbon taken up by microbes.

    Args:
        soil_c_pool_lmwc: Low molecular weight carbon pool [kg C m^-3]
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        moist_temp_scalar: A scalar capturing the combined impact of soil moisture and
            temperature on process rates
        soil_temp: soil temperature for each soil grid cell [degrees C]
        constants: Set of constants for the soil model.

    Returns:
        Uptake of low molecular weight carbon (LMWC) by the soil microbial biomass.
    """

    # Calculate carbon use efficiency and microbial saturation
    carbon_use_efficency = calculate_carbon_use_efficiency(
        soil_temp,
        constants.reference_cue,
        constants.cue_reference_temp,
        constants.cue_with_temperature,
    )
    microbial_saturation = calculate_microbial_saturation(
        soil_c_pool_microbe, constants.half_sat_microbial_activity
    )

    # TODO - the quantities calculated above can be used to calculate the carbon
    # respired instead of being uptaken. This isn't currently of interest, but will be
    # in future

    return (
        constants.max_uptake_rate_labile_C
        * moist_temp_scalar
        * soil_c_pool_lmwc
        * microbial_saturation
        * carbon_use_efficency
    )


def calculate_labile_carbon_leaching(
    soil_c_pool_lmwc: NDArray[np.float32],
    moist_temp_scalar: NDArray[np.float32],
    leaching_rate: float,
) -> NDArray[np.float32]:
    """Calculate rate at which labile carbon is leached.

    This is adopted from Abramoff et al. We definitely need to give more thought to how
    we model leaching.

    Args:
        soil_c_pool_lmwc: Low molecular weight carbon pool [kg C m^-3]
        moist_temp_scalar: A scalar capturing the combined impact of soil moisture and
            temperature on process rates
        leaching_rate: The rate at which labile carbon leaches from the soil [day^-1]

    Returns:
        The amount of labile carbon leached
    """

    return leaching_rate * moist_temp_scalar * soil_c_pool_lmwc


def calculate_pom_decomposition(
    soil_c_pool_pom: NDArray[np.float32],
    soil_c_pool_microbe: NDArray[np.float32],
    moist_temp_scalar: NDArray[np.float32],
    constants: SoilConsts,
) -> NDArray[np.float32]:
    """Calculate decomposition of particulate organic matter into labile carbon (LMWC).

    This is adopted from Abramoff et al. We definitely want to change this down the line
    to something that uses enzymes explicitly.

    Args:
        soil_c_pool_pom: Particulate organic matter pool [kg C m^-3]
        soil_c_pool_microbe: Microbial biomass (carbon) pool [kg C m^-3]
        moist_temp_scalar: A scalar capturing the combined impact of soil moisture and
            temperature on process rates
        constants: Set of constants for the soil model.

    Returns:
        The amount of particulate organic matter (POM) decomposed into labile carbon
            (LMWC)
    """

    # Calculate the two relevant saturations
    saturation_with_biomass = calculate_microbial_pom_mineralisation_saturation(
        soil_c_pool_microbe, constants.half_sat_microbial_pom_mineralisation
    )
    saturation_with_pom = calculate_pom_decomposition_saturation(
        soil_c_pool_pom, constants.half_sat_pom_decomposition
    )

    return (
        constants.max_decomp_rate_pom
        * saturation_with_pom
        * saturation_with_biomass
        * moist_temp_scalar
    )


def calculate_direct_litter_input_to_pools(
    carbon_input_to_pom: float,
    litter_input_rate: float,
) -> tuple[float, float]:
    """Calculate direct input from litter to LMWC and POM pools.

    This process is very much specific to :cite:t:`abramoff_millennial_2018`, and I
    don't think we want to preserve it long term.

    Args:
        carbon_input_to_pom: Proportion of litter carbon input that goes to POM (rather
            than LMWC) [unitless].
        litter_input_rate: Rate at which carbon moves from litter "pool" to soil carbon
            pools [kg C m^-2 day^-1].

    Returns:
        Amount of carbon directly added to LMWC and POM pools from litter.
    """

    return (
        litter_input_rate * (1 - carbon_input_to_pom),
        litter_input_rate * carbon_input_to_pom,
    )
