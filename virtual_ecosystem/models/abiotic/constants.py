"""The ``models.abiotic.constants`` module contains a set of dataclasses which contain
parameters required by the broader
:mod:`~virtual_ecosystem.models.abiotic.abiotic_model` model.
These parameters are constants in that they should not be changed during a particular
simulation.
"""  # noqa: D205, D415

from dataclasses import dataclass, field

from virtual_ecosystem.core.constants_class import ConstantsDataclass


@dataclass(frozen=True)
class AbioticConsts(ConstantsDataclass):
    """Dataclass to store all constants for the `abiotic` model."""

    wind_reference_height: float = 10.0
    """Reference height for wind speed above the canopy.
    The reference height for horizontal wind is typically 10m above ground compared to
    2m for other atmospheric variables such as temperature and relative humidity. We
    assume here that the reference height is above the canopy, please check the input
    data carefully and be aware of limitations."""

    specific_heat_equ_factors: list[float] = field(
        default_factory=lambda: [2e-05, 0.0002]
    )
    """Factors in calculation of molar specific heat of air.

    Implementation after :cite:t:`maclean_microclimc_2021`."""

    latent_heat_vap_equ_factors: list[float] = field(
        default_factory=lambda: [1.91846e6, 33.91]
    )
    """Factors in calculation of latent heat of vapourisation.

    Implementation after :cite:t:`maclean_microclimc_2021`, value is taken from
    :cite:t:`henderson-sellers_new_1984`.
    """

    zero_plane_scaling_parameter: float = 7.5
    """Control parameter for scaling zero displacement to height, dimensionless.

    Implementation after :cite:t:`maclean_microclimc_2021`, value is taken from
    :cite:t:`raupach_simplified_1994`."""

    substrate_surface_drag_coefficient: float = 0.003
    """Substrate-surface drag coefficient, dimensionless.

    The substrate-surface drag coefficient represents the resistance encountered by an
    object moving on or through a surface and varies based on the nature of the surface
    and the object's properties. Here, it affects how wind speed is altered by a surface
    . Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    roughness_element_drag_coefficient: float = 0.3
    """Roughness-element drag coefficient, dimensionless.

    The roughness-element drag coefficient refers to the dimensionless coefficient used
    to quantify the drag force exerted by individual roughness elements (such as
    buildings, trees, or surface irregularities) on airflow, influencing the overall
    aerodynamic characteristics of a surface within the atmospheric boundary layer.
    Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    roughness_sublayer_depth_parameter: float = 0.193
    """Parameter characterizes the roughness sublayer depth.

    The roughness sublayer depth refers to the layer near the surface where the effects
    of surface roughness significantly influence airflow, turbulence, and momentum
    transfer, typically extending up to about 10% of the height of the roughness
    elements or features on the surface. This layer is characterized by intense
    turbulence and rapid velocity changes due to surface irregularities.
    Implentation and value taken from :cite:p:`maclean_microclimc_2021`."""

    max_ratio_wind_to_friction_velocity: float = 0.3
    """Maximum ratio of wind velocity to friction velocity, dimensionless.

    Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    drag_coefficient: float = 0.2
    """Drag coefficient, dimensionless.

    The drag coefficient is a dimensionless quantity that characterizes the drag or
    resistance experienced by an object moving through a fluid (here the atmosphere) and
    is defined as the ratio of the drag force on the object to the dynamic pressure of
    the fluid flow and the reference area of the object.
    Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    relative_turbulence_intensity: float = 0.5
    """Relative turbulence intensity, dimensionless.

    The relative turbulence intensity is a proportionality factor that relates the mean
    eddy velocity is assumed to the local wind speed below the canopy. Implementation
    and value from :cite:t:`maclean_microclimc_2021`."""

    diabatic_correction_factor_below: float = 1
    """Diabatic correction factor below canopy, dimensionless.

    The diabatic correction factor is a scaling adjustment used to compensate for the
    effects of vertical heat transfer or thermal non-adiabaticity on atmospheric
    variables or processes, particularly when estimating or interpreting measurements
    across different heights or conditions. This factor is used to adjust wind profiles
    below the canopy. Implementation and value from :cite:t:`maclean_microclimc_2021`.
    """

    mixing_length_factor: float = 0.32
    """Factor in calculation of mixing length, dimensionless.

    Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    min_relative_turbulence_intensity: float = 0.36
    """Minimum relative turbulence intensity, dimensionless.

    See :attr:`relative_turbulence_intensity`.
    The default value is taken from Shaw et al (1974) Agricultural Meteorology, 13:
    419-425. TODO this is not representative of a rainforest environment and needs to be
    adjusted.
    """

    max_relative_turbulence_intensity: float = 0.9
    """Maximum relative turbulence intensity, dimensionless.

    See :attr:`relative_turbulence_intensity`.
    The default value from Shaw et al (1974) Agricultural Meteorology, 13: 419-425.
    TODO this is not representative of a rainforest environment and needs to be
    adjusted."""

    min_wind_speed_above_canopy: float = 0.1
    """Minimum wind speed above the canopy, [m s-1].

    Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    min_windspeed_below_canopy: float = 0.001
    """Minimum wind speed below the canopy or in absence of vegetation, [m s-1]."""

    min_roughness_length: float = 0.05
    """Minimum roughness length, [m].

    The minimum roughness length represents the lowest height at which the surface
    roughness significantly affects the wind flow over a particular terrain or surface.
    Implementation and value from :cite:t:`maclean_microclimc_2021`."""

    yasuda_stability_parameters: list[float] = field(
        default_factory=lambda: [6.0, 2.0, 16.0]
    )
    """Parameters to approximate diabatic correction factors for heat and momentum.

    Dimenionless parameter, implementation after :cite:t:`maclean_microclimc_2021` and
    values taken from :cite:t:`yasuda_turbulent_1988`."""

    diabatic_heat_momentum_ratio: float = 0.6
    """Factor that relates diabatic correction factors for heat and momentum.

    Dimenionless parameter, implementation after :cite:t:`maclean_microclimc_2021` and
    values taken from :cite:t:`yasuda_turbulent_1988`."""

    turbulence_sign: bool = True
    """Flag indicating if turbulence increases or decreases with height."""

    canopy_temperature_ini_factor: float = 0.01
    """Factor used to initialise canopy temperature as a function of air temperature and
    absorbed shortwave radiation."""

    light_extinction_coefficient: float = 0.01
    """Light extinction coefficient for canopy."""

    gas_constant_water_vapour: float = 461.51
    """Gas constant for water vapour, [J kg -1 K-1]."""

    specific_heat_capacity_leaf: float = 2760.0
    """Specific heat capacity of leaf, [J kg-1 K-1], :cite:p:`aston_heat_1985`.
    """

    leaf_heat_transfer_coefficient: float = 50.0
    """Leaf heat transfer coefficient, [s^1/2 m^-1/2],
    :cite:p:`linacre_determinations_1964`.
    """

    stomatal_resistance: float = 200.0
    """Default stomatal resistance, [s m2 mumol-1]."""

    soil_thermal_conductivity: float = 0.7
    """Soil thermal conductivity, [W m-1 K-1], :cite:p:`monteith_principles_1990`.
    """

    specific_heat_capacity_soil: float = 2.7e6
    """Specific heat capacity of soil, [J kg-1 K-1], :cite:p:`monteith_principles_1990`.
    """

    initial_air_conductivity: float = 50.0
    """Initial air conductivity, [mol m-2 s-1]."""

    top_leaf_vapour_conductivity: float = 0.32
    """Initial leaf vapour conductivity at the top of the canopy, [mol m-2 s-1]."""

    bottom_leaf_vapour_conductivity: float = 0.25
    """Initial leaf vapour conductivity at the bottom of the canopy, [mol m-2 s-1]."""

    top_leaf_air_conductivity: float = 0.19
    """Initial leaf air heat conductivity at the top of the canopy, [mol m-2 s-1]."""

    bottom_leaf_air_conductivity: float = 0.13
    """Initial leaf air heat conductivity at the bottom of the canopy, [mol m-2 s-1]."""

    surface_albedo: float = 0.125
    """Mean surface albedo of a tropical rainforest in South East Asia, dimensionless.

    The value is takes from a study that compares changes in surface albedo before and
    after deforestation in South East Asia :cite:p:`wilson_role_2020`."""

    soil_emissivity: float = 0.8
    """Soil emissivity, dimensionless."""

    surface_layer_depth: float = 0.1
    """Surface layer depth, [m].

    This depth defines the soil depth that is directly involved in the surface energy
    balance.
    """

    volume_to_weight_conversion: float = 1000.0
    """Factor to convert between soil volume and weight in kilograms."""

    kinematic_viscosity_parameters: list[float] = field(
        default_factory=lambda: [0.0908, 11.531]
    )
    """Parameters in calculation of kinematic viscosity
    :cite:p:`campbell_introduction_2012`.
    """

    thermal_diffusivity_parameters: list[float] = field(
        default_factory=lambda: [0.1285, 16.247]
    )
    """Parameter in calculation of thermal diffusivity
    :cite:p:`campbell_introduction_2012`.
    """

    grashof_parameter: float = 9.807
    """Parameter in calculation of Grashof number
    :cite:p:`campbell_introduction_2012`.
    """

    forced_conductance_parameter: float = 0.34
    """Parameter in calculation of forced conductance
    :cite:p:`campbell_introduction_2012`.
    """

    positive_free_conductance_parameter: float = 0.54
    """Parameter in calculation of free conductance for positive temperature difference
    :cite:p:`campbell_introduction_2012`.
    """

    negative_free_conductance_parameter: float = 0.26
    """Parameter in calculation of free conductance for negative temperature difference
    :cite:p:`campbell_introduction_2012`.
    """

    leaf_emissivity: float = 0.8
    """Leaf emissivity, dimensionless."""
