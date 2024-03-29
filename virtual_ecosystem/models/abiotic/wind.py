r"""The wind module calculates the above- and within-canopy wind profile for the
Virtual Ecosystem. The wind profile determines the exchange of heat, water, and
:math:`CO_{2}` between soil and atmosphere below the canopy as well as the exchange with
the atmosphere above the canopy.

TODO: add sanity checks, errors and logging
TODO replace leaf area index by plant area index when we have more info about vertical
distribution of leaf and woody parts
TODO change temperatures to Kelvin
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray

from virtual_ecosystem.core.constants import CoreConsts
from virtual_ecosystem.models.abiotic.abiotic_tools import (
    calculate_molar_density_air,
    calculate_specific_heat_air,
)
from virtual_ecosystem.models.abiotic.constants import AbioticConsts


def calculate_zero_plane_displacement(
    canopy_height: NDArray[np.float32],
    leaf_area_index: NDArray[np.float32],
    zero_plane_scaling_parameter: float,
) -> NDArray[np.float32]:
    """Calculate zero plane displacement height, [m].

    The zero displacement height (d) is a concept used in micrometeorology to describe
    the flow of air near the ground or over surfaces like a forest canopy or crops. It
    represents the height above the actual ground where the wind speed is theoretically
    reduced to zero due to the obstruction caused by the roughness elements (like trees
    or buildings). Implementation after :cite:t:`maclean_microclimc_2021`.

    Args:
        canopy_height: Canopy height, [m]
        leaf_area_index: Total leaf area index, [m m-1]
        zero_plane_scaling_parameter: Control parameter for scaling d/h, dimensionless
            :cite:p:`raupach_simplified_1994`

    Returns:
        zero place displacement height, [m]
    """

    # Select grid cells where vegetation is present
    displacement = np.where(leaf_area_index > 0, leaf_area_index, np.nan)

    # Calculate zero displacement height
    scale_displacement = np.sqrt(zero_plane_scaling_parameter * displacement)
    zero_place_displacement = (
        (1 - (1 - np.exp(-scale_displacement)) / scale_displacement) * canopy_height,
    )

    # No displacement in absence of vegetation
    return np.nan_to_num(zero_place_displacement, nan=0.0).squeeze()


def calculate_roughness_length_momentum(
    canopy_height: NDArray[np.float32],
    leaf_area_index: NDArray[np.float32],
    zero_plane_displacement: NDArray[np.float32],
    substrate_surface_drag_coefficient: float,
    roughness_element_drag_coefficient: float,
    roughness_sublayer_depth_parameter: float,
    max_ratio_wind_to_friction_velocity: float,
    min_roughness_length: float,
    von_karman_constant: float,
) -> NDArray[np.float32]:
    """Calculate roughness length governing momentum transfer, [m].

    Roughness length is defined as the height at which the mean velocity is zero due to
    substrate roughness. Real surfaces such as the ground or vegetation are not smooth
    and often have varying degrees of roughness. Roughness length accounts for that
    effect. Implementation after :cite:t:`maclean_microclimc_2021`.

    Args:
        canopy_height: Canopy height, [m]
        leaf_area_index: Total leaf area index, [m m-1]
        zero_plane_displacement: Height above ground within the canopy where the wind
            profile extrapolates to zero, [m]
        substrate_surface_drag_coefficient: Substrate-surface drag coefficient,
            dimensionless
        roughness_element_drag_coefficient: Roughness-element drag coefficient
        roughness_sublayer_depth_parameter: Parameter that characterizes the roughness
            sublayer depth, dimensionless
        max_ratio_wind_to_friction_velocity: Maximum ratio of wind velocity to friction
            velocity, dimensionless
        min_roughness_length: Minimum roughness length, [m]
        von_karman_constant: Von Karman's constant, dimensionless constant describing
            the logarithmic velocity profile of a turbulent fluid near a no-slip
            boundary.

    Returns:
        momentum roughness length, [m]
    """

    # calculate ratio of wind velocity to friction velocity
    ratio_wind_to_friction_velocity = np.sqrt(
        substrate_surface_drag_coefficient
        + (roughness_element_drag_coefficient * leaf_area_index) / 2
    )

    # if the ratio of wind velocity to friction velocity is larger than the set maximum,
    # set the value to set maximum
    set_maximum_ratio = np.where(
        ratio_wind_to_friction_velocity > max_ratio_wind_to_friction_velocity,
        max_ratio_wind_to_friction_velocity,
        ratio_wind_to_friction_velocity,
    )

    # calculate initial roughness length
    initial_roughness_length = (canopy_height - zero_plane_displacement) * np.exp(
        -von_karman_constant * (1 / set_maximum_ratio)
        - roughness_sublayer_depth_parameter
    )

    # if roughness smaller than the substrate surface drag coefficient, set to value to
    # the substrate surface drag coefficient
    roughness_length = np.where(
        initial_roughness_length < substrate_surface_drag_coefficient,
        substrate_surface_drag_coefficient,
        initial_roughness_length,
    )

    return np.where(roughness_length <= 0, min_roughness_length, roughness_length)


def calculate_diabatic_correction_above(
    molar_density_air: float | NDArray[np.float32],
    specific_heat_air: float | NDArray[np.float32],
    temperature: NDArray[np.float32],
    sensible_heat_flux: NDArray[np.float32],
    friction_velocity: NDArray[np.float32],
    wind_heights: NDArray[np.float32],
    zero_plane_displacement: NDArray[np.float32],
    celsius_to_kelvin: float,
    von_karmans_constant: float,
    yasuda_stability_parameters: list[float],
    diabatic_heat_momentum_ratio: float,
) -> dict[str, NDArray[np.float32]]:
    r"""Calculate the diabatic correction factors for momentum and heat above canopy.

    Diabatic correction factor for heat and momentum are used to adjust wind profiles
    for surface heating and cooling :cite:p:`maclean_microclimc_2021`. When the surface
    is strongly heated, the diabatic correction factor for momentum :math:`\Psi_{M}`
    becomes negative and drops to values of around -1.5. In contrast, when the surface
    is much cooler than the air above it, it increases to values around 4.

    Args:
        molar_density_air: molar density of air, [mol m-3]
        specific_heat_air: specific heat of air, [J mol-1 K-1]
        temperature: 2m temperature, [C]
        sensible_heat_flux: Sensible heat flux from canopy to atmosphere above,
            [W m-2], # TODO: could be the top entry of the general sensible heat flux
        friction_velocity: Friction velocity, [m s-1]
        wind_heights: Vector of heights for which wind speed is calculated, [m]
        zero_plane_displacement: Height above ground within the canopy where the wind
            profile extrapolates to zero, [m]
        celsius_to_kelvin: Factor to convert temperature in Celsius to absolute
            temperature in Kelvin
        von_karmans_constant: Von Karman's constant, dimensionless constant describing
            the logarithmic velocity profile of a turbulent fluid near a no-slip
            boundary.
        yasuda_stability_parameters: Parameters to approximate diabatic correction
            factors for heat and momentum after :cite:t:`yasuda_turbulent_1988`
        diabatic_heat_momentum_ratio: Factor that relates diabatic correction
            factors for heat and momentum after :cite:t:`yasuda_turbulent_1988`

    Returns:
        diabatic correction factors for heat (psi_h) and momentum (psi_m) transfer
    """

    # calculate atmospheric stability
    stability = (
        von_karmans_constant
        * (wind_heights - zero_plane_displacement)
        * sensible_heat_flux
    ) / (
        molar_density_air
        * specific_heat_air
        * (temperature + celsius_to_kelvin)
        * friction_velocity
    )

    stable_condition = yasuda_stability_parameters[0] * np.log(1 - stability)
    unstable_condition = -yasuda_stability_parameters[1] * np.log(
        (1 + np.sqrt(1 - yasuda_stability_parameters[2] * stability)) / 2
    )

    diabatic_correction_heat = np.where(
        sensible_heat_flux < 0, stable_condition, unstable_condition
    )

    diabatic_correction_momentum = np.where(
        sensible_heat_flux < 0,
        diabatic_correction_heat,
        diabatic_heat_momentum_ratio * diabatic_correction_heat,
    )

    return {"psi_m": diabatic_correction_momentum, "psi_h": diabatic_correction_heat}


def calculate_mean_mixing_length(
    canopy_height: NDArray[np.float32],
    zero_plane_displacement: NDArray[np.float32],
    roughness_length_momentum: NDArray[np.float32],
    mixing_length_factor: float,
) -> NDArray[np.float32]:
    """Calculate mixing length for canopy air transport, [m].

    The mean mixing length is used to calculate turbulent air transport inside vegetated
    canopies. It is made equivalent to the above canopy value at the canopy surface. In
    absence of vegetation, it is set to zero. Implementation after
    :cite:t:`maclean_microclimc_2021`.

    Args:
        canopy_height: Canopy height, [m]
        zero_plane_displacement: Height above ground within the canopy where the wind
            profile extrapolates to zero, [m]
        roughness_length_momentum: Momentum roughness length, [m]
        mixing_length_factor: Factor in calculation of mean mixing length, dimensionless

    Returns:
        mixing length for canopy air transport, [m]
    """

    mean_mixing_length = (
        mixing_length_factor * (canopy_height - zero_plane_displacement)
    ) / np.log((canopy_height - zero_plane_displacement) / roughness_length_momentum)

    return np.nan_to_num(mean_mixing_length, nan=0)


def generate_relative_turbulence_intensity(
    layer_heights: NDArray[np.float32],
    min_relative_turbulence_intensity: float,
    max_relative_turbulence_intensity: float,
    increasing_with_height: bool,
) -> NDArray[np.float32]:
    """Generate relative turbulence intensity profile, dimensionless.

    At the moment, default values are for a maize crop Shaw et al (1974)
    Agricultural Meteorology, 13: 419-425. TODO adjust default to environment

    Args:
        layer_heights: heights of above ground layers, [m]
        min_relative_turbulence_intensity: minimum relative turbulence intensity,
            dimensionless
        max_relative_turbulence_intensity: maximum relative turbulence intensity,
            dimensionless
        increasing_with_height: increasing logical indicating whether turbulence
            intensity increases (TRUE) or decreases (FALSE) with height

    Returns:
        relative turbulence intensity for each node, dimensionless
    """

    direction = 1 if increasing_with_height else -1

    return (
        min_relative_turbulence_intensity
        + direction
        * (max_relative_turbulence_intensity - min_relative_turbulence_intensity)
        * layer_heights
    )


def calculate_wind_attenuation_coefficient(
    canopy_height: NDArray[np.float32],
    leaf_area_index: NDArray[np.float32],
    mean_mixing_length: NDArray[np.float32],
    drag_coefficient: float,
    relative_turbulence_intensity: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate wind attenuation coefficient, dimensionless.

    The wind attenuation coefficient describes how wind is slowed down by the presence
    of vegetation. In absence of vegetation, the coefficient is set to zero.
    Implementation after :cite:t:`maclean_microclimc_2021`.

    Args:
        canopy_height: Canopy height, [m]
        leaf_area_index: Total leaf area index, [m m-1]
        mean_mixing_length: Mixing length for canopy air transport, [m]
        drag_coefficient: Drag coefficient, dimensionless
        relative_turbulence_intensity: Relative turbulence intensity, dimensionless

    Returns:
        wind attenuation coefficient, dimensionless
    """

    intermediate_coefficient = (
        (drag_coefficient * leaf_area_index * canopy_height)
        / (
            2
            * mean_mixing_length
            * relative_turbulence_intensity[0 : len(leaf_area_index)]
        ),
    )

    attenuation_coefficient = np.nan_to_num(intermediate_coefficient, nan=0).squeeze()
    return np.concatenate(
        [
            [[0.0, 0.0, 0.0]],
            attenuation_coefficient,
            [attenuation_coefficient[-1]] * 2,  # add lowest value to below canopy layer
        ]
    )


def wind_log_profile(
    height: float | NDArray[np.float32],
    zeroplane_displacement: float | NDArray[np.float32],
    roughness_length_momentum: float | NDArray[np.float32],
    diabatic_correction_momentum: float | NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate logarithmic wind profile.

    Args:
        height: Array of heights for which wind speed is calculated, [m]
        zeroplane_displacement: Height above ground within the canopy where the wind
            profile extrapolates to zero, [m]
        roughness_length_momentum: Momentum roughness length, [m]
        diabatic_correction_momentum: Diabatic correction factor for momentum

    Returns:
        logarithmic wind profile
    """

    wind_profile = (
        np.log((height - zeroplane_displacement) / roughness_length_momentum)
        + diabatic_correction_momentum,
    )

    return np.squeeze(wind_profile)


def calculate_friction_velocity(
    wind_speed_ref: NDArray[np.float32],
    reference_height: float | NDArray[np.float32],
    zeroplane_displacement: NDArray[np.float32],
    roughness_length_momentum: NDArray[np.float32],
    diabatic_correction_momentum: float | NDArray[np.float32],
    von_karmans_constant: float,
) -> NDArray[np.float32]:
    """Calculate friction velocity from wind speed at reference height, [m s-1].

    Args:
        wind_speed_ref: Wind speed at reference height, [m s-1]
        reference_height: Height of wind measurement, [m]
        zero_plane_displacement: Height above ground within the canopy where the wind
            profile extrapolates to zero, [m]
        roughness_length_momentum: Momentum roughness length, [m]
        diabatic_correction_momentum: Diabatic correction factor for momentum as
            returned by
            :func:`~virtual_ecosystem.models.abiotic.wind.calculate_diabatic_correction_above`
        von_karmans_constant: Von Karman's constant, dimensionless constant describing
            the logarithmic velocity profile of a turbulent fluid near a no-slip
            boundary.

    Returns:
        friction velocity
    """

    wind_profile_reference = wind_log_profile(
        height=reference_height,
        zeroplane_displacement=zeroplane_displacement,
        roughness_length_momentum=roughness_length_momentum,
        diabatic_correction_momentum=diabatic_correction_momentum,
    )

    return von_karmans_constant * (wind_speed_ref / wind_profile_reference)


def calculate_wind_above_canopy(
    friction_velocity: NDArray[np.float32],
    wind_height_above: NDArray[np.float32],
    zeroplane_displacement: NDArray[np.float32],
    roughness_length_momentum: NDArray[np.float32],
    diabatic_correction_momentum: NDArray[np.float32],
    von_karmans_constant: float,
    min_wind_speed_above_canopy: float,
) -> NDArray[np.float32]:
    """Calculate wind speed above canopy from wind speed at reference height, [m s-1].

    Wind speed above the canopy dictates heat and vapour exchange between the canopy
    and the air above it, and therefore ultimately determines temperature and vapour
    profiles.
    The wind profile above canopy typically follows a logarithmic height profile, which
    extrapolates to zero roughly two thirds of the way to the top of the canopy. The
    profile itself is thus dependent on the height of the canopy, but also on the
    roughness of the vegetation layer, which causes wind shear. We follow the
    implementation by :cite:t:`campbell_introduction_1998` as described in
    :cite:t:`maclean_microclimc_2021`.

    Args:
        friction_velocity: friction velocity, [m s-1]
        wind_height_above: Height above canopy for which wind speed is required, [m]
        zero_plane_displacement: Height above ground within the canopy where the wind
            profile extrapolates to zero, [m]
        roughness_length_momentum: Momentum roughness length, [m]
        diabatic_correction_momentum: Diabatic correction factor for momentum as
            returned by
            :func:`~virtual_ecosystem.models.abiotic.wind.calculate_diabatic_correction_above`
        von_karmans_constant: Von Karman's constant, dimensionless constant describing
            the logarithmic velocity profile of a turbulent fluid near a no-slip
            boundary.

    Returns:
        wind speed at required heights above canopy, [m s-1]
    """

    wind_profile_above = wind_log_profile(
        height=wind_height_above,
        zeroplane_displacement=zeroplane_displacement,
        roughness_length_momentum=roughness_length_momentum,
        diabatic_correction_momentum=diabatic_correction_momentum,
    )
    wind_profile = (friction_velocity / von_karmans_constant) * wind_profile_above

    return np.where(
        wind_profile < min_wind_speed_above_canopy,
        min_wind_speed_above_canopy,
        wind_profile,
    )


def calculate_wind_canopy(
    top_of_canopy_wind_speed: NDArray[np.float32],
    wind_layer_heights: NDArray[np.float32],
    canopy_height: NDArray[np.float32],
    attenuation_coefficient: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate wind speed in a multi-layer canopy, [m s-1].

    This function can be extended to account for edge distance effects.

    Args:
        top_of_canopy_wind_speed: Wind speed at top of canopy layer, [m s-1]
        wind_layer_heights: Heights of canopy layer nodes, [m]
        canopy_height: Height to top of canopy layer, [m]
        attenuation_coefficient: Mean attenuation coefficient based on the profile
            calculated by
            :func:`~virtual_ecosystem.models.abiotic.wind.calculate_wind_attenuation_coefficient`
        min_windspeed_below_canopy: Minimum wind speed below the canopy or in absence of
            vegetation, [m/s]. This value is set to avoid dividion by zero.

    Returns:
        wind speed at height of canopy node, [m s-1]
    """

    zero_displacement = top_of_canopy_wind_speed * np.exp(
        attenuation_coefficient * ((wind_layer_heights / canopy_height) - 1)
    )
    return zero_displacement


def calculate_wind_profile(
    canopy_height: NDArray[np.float32],
    wind_height_above: NDArray[np.float32],
    wind_layer_heights: NDArray[np.float32],
    leaf_area_index: NDArray[np.float32],
    air_temperature: NDArray[np.float32],
    atmospheric_pressure: NDArray[np.float32],
    sensible_heat_flux_topofcanopy: NDArray[np.float32],
    wind_speed_ref: NDArray[np.float32],
    wind_reference_height: float | NDArray[np.float32],
    abiotic_constants: AbioticConsts,
    core_constants: CoreConsts,
) -> dict[str, NDArray[np.float32]]:
    r"""Calculate wind speed above and below the canopy, [m s-1].

    The wind profile above the canopy is described as follows (based on
    :cite:p:`campbell_introduction_1998` as implemented in
    :cite:t:`maclean_microclimc_2021`):

    :math:`u_z = \frac{u^{*}}{0.4} ln \frac{z-d}{z_M} + \Psi_M`

    where :math:`u_z` is wind speed at height :math:`z` above the canopy, :math:`d` is
    the height above ground within the canopy where the wind profile extrapolates to
    zero, :math:`z_m` the roughness length for momentum, :math:`\Psi_M` is a diabatic
    correction for momentum and :math:`u^{*}` is the friction velocity, which gives the
    wind speed at height :math:`d + z_m`.

    The wind profile below canopy is derived as follows:

    :math:`u_z = u_h exp(a(\frac{z}{h} - 1))`

    where :math:`u_z` is wind speed at height :math:`z` within the canopy, :math:`u_h`
    is wind speed at the top of the canopy at height :math:`h`, and :math:`a` is a wind
    attenuation coefficient given by :math:`a = 2 l_m i_w`, where :math:`c_d` is a drag
    coefficient that varies with leaf inclination and shape, :math:`i_w` is a
    coefficient describing relative turbulence intensity and :math:`l_m` is the mean
    mixing length, equivalent to the free space between the leaves and stems. For
    details, see :cite:t:`maclean_microclimc_2021`.

    Args:
        canopy_height: Canopy height, [m]
        wind_height_above: Height above canopy for which wind speed is required, [m]
        wind_layer_heights: Heights of canopy layer nodes, [m]
        leaf_area_index: Leaf area index, [m m-1]
        air_temperature: Air temperature, [C]
        atmospheric_pressure: Atmospheric pressure, [kPa]
        sensible_heat_flux_topofcanopy: Sensible heat flux from the top of the canopy to
            the atmosphere, [W m-2],
        wind_speed_ref: Wind speed at reference height, [m s-1]
        wind_reference_height: Reference height for wind measurement, [m]
        abiotic_constants: Specific constants for the abiotic model
        core_constants: Universal constants shared across all models

    Returns:
        dictionnary that contains wind speed above the canopy, [m s-1], wind speed
        within and below the canopy, [m s-1], and friction velocity, [m s-1]
    """

    output = {}

    # TODO adjust wind to 2m above canopy?

    molar_density_air = calculate_molar_density_air(
        temperature=air_temperature,
        atmospheric_pressure=atmospheric_pressure,
        standard_mole=core_constants.standard_mole,
        standard_pressure=core_constants.standard_pressure,
        celsius_to_kelvin=core_constants.zero_Celsius,
    )

    specific_heat_air = calculate_specific_heat_air(
        temperature=air_temperature,
        molar_heat_capacity_air=core_constants.molar_heat_capacity_air,
        specific_heat_equ_factors=abiotic_constants.specific_heat_equ_factors,
    )

    leaf_area_index_sum = np.nansum(leaf_area_index, axis=0)

    zero_plane_displacement = calculate_zero_plane_displacement(
        canopy_height=canopy_height,
        leaf_area_index=leaf_area_index_sum,
        zero_plane_scaling_parameter=abiotic_constants.zero_plane_scaling_parameter,
    )

    roughness_length_momentum = calculate_roughness_length_momentum(
        canopy_height=canopy_height,
        leaf_area_index=leaf_area_index_sum,
        zero_plane_displacement=zero_plane_displacement,
        substrate_surface_drag_coefficient=(
            abiotic_constants.substrate_surface_drag_coefficient
        ),
        roughness_element_drag_coefficient=(
            abiotic_constants.roughness_element_drag_coefficient
        ),
        roughness_sublayer_depth_parameter=(
            abiotic_constants.roughness_sublayer_depth_parameter
        ),
        max_ratio_wind_to_friction_velocity=(
            abiotic_constants.max_ratio_wind_to_friction_velocity
        ),
        min_roughness_length=abiotic_constants.min_roughness_length,
        von_karman_constant=core_constants.von_karmans_constant,
    )

    friction_velocity_uncorrected = calculate_friction_velocity(
        wind_speed_ref=wind_speed_ref,
        reference_height=wind_reference_height,
        zeroplane_displacement=zero_plane_displacement,
        roughness_length_momentum=roughness_length_momentum,
        diabatic_correction_momentum=0.0,
        von_karmans_constant=core_constants.von_karmans_constant,
    )

    # TODO select above layer (psi) and add function for factor below canopy (phi)
    diabatic_correction_above = calculate_diabatic_correction_above(
        molar_density_air=molar_density_air,
        specific_heat_air=specific_heat_air,
        temperature=air_temperature,
        sensible_heat_flux=sensible_heat_flux_topofcanopy,
        friction_velocity=friction_velocity_uncorrected,
        wind_heights=wind_layer_heights,
        zero_plane_displacement=zero_plane_displacement,
        celsius_to_kelvin=core_constants.zero_Celsius,
        von_karmans_constant=core_constants.von_karmans_constant,
        yasuda_stability_parameters=abiotic_constants.yasuda_stability_parameters,
        diabatic_heat_momentum_ratio=abiotic_constants.diabatic_heat_momentum_ratio,
    )

    friction_velocity = calculate_friction_velocity(
        wind_speed_ref=wind_speed_ref,
        reference_height=wind_reference_height,
        zeroplane_displacement=zero_plane_displacement,
        roughness_length_momentum=roughness_length_momentum,
        diabatic_correction_momentum=diabatic_correction_above["psi_m"],
        von_karmans_constant=core_constants.von_karmans_constant,
    )
    output["friction_velocity"] = friction_velocity

    mean_mixing_length = calculate_mean_mixing_length(
        canopy_height=canopy_height,
        zero_plane_displacement=zero_plane_displacement,
        roughness_length_momentum=roughness_length_momentum,
        mixing_length_factor=abiotic_constants.mixing_length_factor,
    )

    relative_turbulence_intensity = generate_relative_turbulence_intensity(
        layer_heights=wind_layer_heights,
        min_relative_turbulence_intensity=(
            abiotic_constants.min_relative_turbulence_intensity
        ),
        max_relative_turbulence_intensity=(
            abiotic_constants.max_relative_turbulence_intensity
        ),
        increasing_with_height=abiotic_constants.turbulence_sign,
    )

    attennuation_coefficient = calculate_wind_attenuation_coefficient(
        canopy_height=canopy_height,
        leaf_area_index=leaf_area_index,
        mean_mixing_length=mean_mixing_length,
        drag_coefficient=abiotic_constants.drag_coefficient,
        relative_turbulence_intensity=relative_turbulence_intensity,
    )
    wind_speed_above_canopy = calculate_wind_above_canopy(
        friction_velocity=friction_velocity[0],
        wind_height_above=wind_height_above,
        zeroplane_displacement=zero_plane_displacement,
        roughness_length_momentum=roughness_length_momentum,
        diabatic_correction_momentum=diabatic_correction_above["psi_m"][0],
        von_karmans_constant=core_constants.von_karmans_constant,
        min_wind_speed_above_canopy=abiotic_constants.min_wind_speed_above_canopy,
    )
    output["wind_speed_above_canopy"] = wind_speed_above_canopy

    wind_speed_canopy = calculate_wind_canopy(
        top_of_canopy_wind_speed=wind_speed_above_canopy,
        wind_layer_heights=wind_layer_heights,
        canopy_height=canopy_height,
        attenuation_coefficient=attennuation_coefficient,
    )
    output["wind_speed_canopy"] = wind_speed_canopy

    return output
