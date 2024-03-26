r"""The ``models.abiotic.energy_balance`` module calculates the energy balance for the
Virtual Ecosystem. Given that the time increments of the model are an hour or longer,
we can assume that below-canopy heat and vapour exchange attain steady state and heat
storage in the canopy does not need to be simulated explicitly.
(For application where very fine-temporal resolution data might be needed, heat and
vapour exchange must be modelled as transient processes, and heat storage by the canopy,
and the exchange of heat between different layers of the canopy, must be considered
explicitly, see :cite:t:`maclean_microclimc_2021`. This is currently not implemented.)

Under steady-state, the balance equation for the leaves in each canopy layer is as
follows (after :cite:t:`maclean_microclimc_2021`):

.. math::
    R_{abs} - R_{em} - H - \lambda E
    = R_{abs} - \epsilon_{s} \sigma T_{L}^{4} - c_{P}g_{Ha}(T_{L} - T_{A})
    - \lambda g_{v} \frac {e_{L} - e_{A}}{p_{A}} = 0

where :math:`R_{abs}` is absorbed radiation, :math:`R_{em}` emitted radiation, :math:`H`
the sensible heat flux, :math:`\lambda E` the latent heat flux, :math:`\epsilon_{s}` the
emissivity of the leaf, :math:`\sigma` the Stefan-Boltzmann constant, :math:`T_{L}` the
absolute temperature of the leaf, :math:`T_{A}` the absolute temperature of the air
surrounding the leaf, :math:`\lambda` the latent heat of vapourisation of water,
:math:`e_{L}` the effective vapour pressure of the leaf, :math:`e_{A}` the vapour
pressure of air and :math:`p_{A}` atmospheric pressure. :math:`g_{Ha}` is the heat
conductance between leaf and atmosphere, :math:`g_{v}` represents the conductance
for vapour loss from the leaves as a function of the stomatal conductance :math:`g_{c}`.

A challenge in solving this equation is the dependency of latent heat and emitted
radiation on leaf temperature. We use a linearisation approach to solve the equation for
leaf temperature and air temperature simultaneously after
:cite:t:`maclean_microclimc_2021`.

The soil energy balance functions are described in
:mod:`~virtual_ecosystem.models.abiotic.soil_energy_balance`.

The conductivities are calculated as described in
:mod:`~virtual_ecosystem.models.abiotic.conductivities`.
"""  # noqa: D205, D415

import numpy as np
from numpy.typing import NDArray
from xarray import DataArray

from virtual_ecosystem.core.constants import CoreConsts
from virtual_ecosystem.core.core_components import LayerStructure
from virtual_ecosystem.core.data import Data
from virtual_ecosystem.models.abiotic.conductivities import (
    calculate_current_conductivities,
    interpolate_along_heights,
)
from virtual_ecosystem.models.abiotic.constants import AbioticConsts
from virtual_ecosystem.models.abiotic_simple.constants import AbioticSimpleConsts
from virtual_ecosystem.models.abiotic_simple.microclimate import (
    calculate_saturation_vapour_pressure,
)


def initialise_absorbed_radiation(
    topofcanopy_radiation: NDArray[np.float32],
    leaf_area_index: NDArray[np.float32],
    layer_heights: NDArray[np.float32],
    light_extinction_coefficient: float,
) -> NDArray[np.float32]:
    r"""Calculate initial light absorption profile.

    This function calculates the fraction of radiation absorbed by a multi-layered
    canopy based on its leaf area index (:math:`LAI`) and extinction coefficient
    (:math:`k`) at each layer, the depth of each measurement (:math:`z`), and the
    incoming light intensity at the top of the canopy (:math:`I_{0}`). The
    implementation based on Beer's law:

    .. math:: I(z) = I_{0} * e^{(-k * LAI * z)}

    Args:
        topofcanopy_radiation: Top of canopy radiation shortwave radiation, [W m-2]
        leaf_area_index: Leaf area index of each canopy layer, [m m-1]
        layer_heights: Layer heights, [m]
        light_extinction_coefficient: Light extinction coefficient, [m-1]

    Returns:
        Shortwave radiation absorbed by canopy layers, [W m-2]
    """
    # Calculate the depth of each layer, [m]
    layer_depths = np.abs(np.diff(layer_heights, axis=0, append=0))

    # Calculate the light extinction for each layer
    layer_extinction = np.exp(
        -0.01 * light_extinction_coefficient * layer_depths * leaf_area_index
    )

    # Calculate how much light penetrates through the canopy, [W m-2]
    cumulative_extinction = np.cumprod(layer_extinction, axis=0)
    penetrating_radiation = cumulative_extinction * topofcanopy_radiation

    # Calculate how much light is absorbed in each layer, [W m-2]
    absorbed_radiation = np.abs(
        np.diff(
            penetrating_radiation,
            prepend=np.expand_dims(topofcanopy_radiation, axis=0),
            axis=0,
        )
    )

    return absorbed_radiation


def initialise_canopy_temperature(
    air_temperature: NDArray[np.float32],
    absorbed_radiation: NDArray[np.float32],
    canopy_temperature_ini_factor: float,
) -> NDArray[np.float32]:
    """Initialise canopy temperature.

    Args:
        air_temperature: Air temperature, [C]
        canopy_temperature_ini_factor: Factor used to initialise canopy temperature as a
            function of air temperature and absorbed shortwave radiation
        absorbed_radiation: Shortwave radiation absorbed by canopy

    Returns:
        Initial canopy temperature, [C]
    """
    return air_temperature + canopy_temperature_ini_factor * absorbed_radiation


def initialise_canopy_and_soil_fluxes(
    air_temperature: DataArray,
    topofcanopy_radiation: DataArray,
    leaf_area_index: DataArray,
    layer_heights: DataArray,
    true_canopy_indexes: NDArray[np.integer],
    topsoil_layer_index: int,
    light_extinction_coefficient: float,
    canopy_temperature_ini_factor: float,
) -> dict[str, DataArray]:
    """Initialise canopy temperature and energy fluxes.

    This function initializes the following variables to run the first step of the
    energy balance routine: absorbed radiation (canopy), canopy temperature, sensible
    and latent heat flux (canopy and soil), and ground heat flux.

    Args:
        air_temperature: Air temperature, [C]
        topofcanopy_radiation: Top of canopy radiation, [W m-2]
        leaf_area_index: Leaf area index, [m m-2]
        layer_heights: Layer heights, [m]
        true_canopy_indexes: Indexes of canopy layers that are not NaN (maximum extent
            to capture all depths even if grid cells have different number of layers)
        topsoil_layer_index: Index of topsoil layer
        light_extinction_coefficient: Light extinction coefficient for canopy
        canopy_temperature_ini_factor: Factor used to initialise canopy temperature as a
            function of air temperature and absorbed shortwave radiation

    Returns:
        Dictionary with absorbed radiation (canopy), canopy temperature, sensible
            and latent heat flux (canopy and soil), and ground heat flux.
    """

    output = {}

    # select canopy layers with leaf area index != nan
    leaf_area_index_true = leaf_area_index[true_canopy_indexes]
    layer_heights_canopy = layer_heights[true_canopy_indexes]
    air_temperature_canopy = air_temperature[true_canopy_indexes]

    # Initialize absorbed radiation DataArray
    absorbed_radiation = DataArray(
        np.full_like(layer_heights, np.nan),
        dims=layer_heights.dims,
        coords=layer_heights.coords,
        name="canopy_absorption",
    )

    # Calculate absorbed radiation
    initial_absorbed_radiation = initialise_absorbed_radiation(
        topofcanopy_radiation=topofcanopy_radiation.to_numpy(),
        leaf_area_index=leaf_area_index_true.to_numpy(),
        layer_heights=layer_heights_canopy.to_numpy(),
        light_extinction_coefficient=light_extinction_coefficient,
    )

    # Replace np.nan with new values and write in output dict
    absorbed_radiation[layer_heights_canopy.indexes] = initial_absorbed_radiation
    output["canopy_absorption"] = absorbed_radiation

    # Initialize canopy temperature DataArray
    canopy_temperature = DataArray(
        np.full_like(layer_heights, np.nan),
        dims=layer_heights.dims,
        coords=layer_heights.coords,
        name="canopy_temperature",
    )

    # Calculate initial temperature and write in output dict
    initial_canopy_temperature = initialise_canopy_temperature(
        air_temperature=air_temperature_canopy.to_numpy(),
        absorbed_radiation=initial_absorbed_radiation,
        canopy_temperature_ini_factor=canopy_temperature_ini_factor,
    )
    canopy_temperature[true_canopy_indexes] = initial_canopy_temperature
    output["canopy_temperature"] = canopy_temperature

    # Initialise sensible heat flux with zeros and write in output dict
    sensible_heat_flux = DataArray(
        np.full_like(layer_heights, np.nan),
        dims=layer_heights.dims,
        coords=layer_heights.coords,
        name="sensible_heat_flux",
    )
    sensible_heat_flux[true_canopy_indexes] = 0
    sensible_heat_flux[topsoil_layer_index] = 0
    output["sensible_heat_flux"] = sensible_heat_flux

    # Initialise latent heat flux with zeros and write in output dict
    output["latent_heat_flux"] = sensible_heat_flux.copy().rename("latent_heat_flux")

    # Initialise latent heat flux with zeros and write in output dict
    ground_heat_flux = DataArray(
        np.full_like(layer_heights, np.nan),
        dims=layer_heights.dims,
        coords=layer_heights.coords,
        name="ground_heat_flux",
    )
    ground_heat_flux[topsoil_layer_index] = 0
    output["ground_heat_flux"] = ground_heat_flux

    return output


def calculate_longwave_emission(
    temperature: NDArray[np.float32],
    emissivity: float | NDArray[np.float32],
    stefan_boltzmann: float,
) -> NDArray[np.float32]:
    """Calculate longwave emission using the Stefan Boltzmann law, [W m-2].

    According to the Stefan Boltzmann law, the amount of radiation emitted per unit time
    from the area of a black body at absolute temperature is directly proportional to
    the fourth power of the temperature. Emissivity (which is equal to absorptive power)
    lies between 0 to 1.

    Args:
        temperature: Temperature, [K]
        emissivity: Emissivity, dimensionless
        stefan_boltzmann: Stefan Boltzmann constant, [W m-2 K-4]

    Returns:
        Longwave emission, [W m-2]
    """
    return emissivity * stefan_boltzmann * temperature**4


def calculate_slope_of_saturated_pressure_curve(
    temperature: NDArray[np.float32],
) -> NDArray[np.float32]:
    r"""Calculate slope of the saturated pressure curve.

    TODO move factors to constants, most of them share with simple_abiotic_model

    Args:
        temperature: Temperature, [C]

    Returns:
        Slope of the saturated pressure curve, :math:`\Delta_{v}`
    """

    return (
        4098
        * (0.6108 * np.exp(17.27 * temperature / (temperature + 237.3)))
        / (temperature + 237.3) ** 2
    )


def calculate_leaf_and_air_temperature(
    data: Data,
    time_index: int,
    topsoil_layer_index: int,
    true_canopy_indexes: NDArray[np.integer],
    true_canopy_layers_n: int,
    layer_structure: LayerStructure,
    abiotic_constants: AbioticConsts,
    abiotic_simple_constants: AbioticSimpleConsts,
    core_constants: CoreConsts,
) -> dict[str, DataArray]:
    r"""Calculate leaf and air temperature under steady state.

    The air temperature surrounding the leaf :math:`T_{A}` is assumed to be influenced
    by leaf temperature :math:`T_{L}`, soil temperature :math:`T_{0}`, and reference air
    temperature :math:`T_{R}` as follows:

    .. math::
        g_{tR} c_{p} (T_{R} - T_{A})
        + g_{t0} c_{p} (T_{0} - T_{A})
        + g_{L} c_{p} (T_{L} - T_{A}) = 0

    where :math:`c_{p}` is the specific heat of air at constant pressure and
    :math:`g_{tR}`, :math:`g_{t0}` and :math:`g_{L}` are conductance from reference
    height, the ground and from the leaf, respectively.
    :math:`g_{L} = 1/(1/g_{HA} + 1/g_{z})` where :math:`g_{HA}` is leaf boundary layer
    conductance and :math:`g_{z}` is the sub-canopy turbulent conductance at the height
    of the leaf over the mean distance between the leaf and the air.

    Defining :math:`T_{L} - T_{A}` as :math:`\Delta T` and rearranging gives:

    .. math:: T_{A} = a_{A} + b_{A} \Delta T_{L}

    where :math:`a_{A} = \frac{(g_{tR} T_{R} + g_{t0} T_{0})}{(g_{tR} + g_{t0})}` and
    :math:`b_{A} = \frac{g_{L}}{(g_{tR} + g_{t0})}` .

    The sensible heat flux between the leaf and the air is given by

    .. math:: g_{Ha} c_{p} (T_{L} - T_{A}) = b_{H} \Delta T_{L}

    where :math:`b_{H} = g_{Ha} c_{p}`. The equivalent vapour flux equation is

    .. math:: g_{tR}(e_{R} - e_{a}) + g_{t0} (e_{0} - e_{a}) + g_{v} (e_{L} - e_{a}) = 0

    where :math:`e_{L}`, :math:`e_{A}`, :math:`e_{0}` and :math:`e_{R}` are the vapour
    pressure of the leaf, air, soil and air at reference height, respectively, and
    :math:`g_{v}` is leaf conductance for vapour given by
    :math:`g_{v} = \frac{1}{(\frac{1}{g_{c} + g_{L})}}` where :math:`g_{c}` is stomatal
    conductance. Assuming the leaf to be saturated, and approximated by
    :math:`e_{s} [T_{R}]+\Delta_{v} [T_{R}]\Delta T_{L}` where :math:`\Delta_{v}` is the
    slope of the saturated pressure curve at temperature :math:`T_{R}`, and rearranging
    gives

    .. math:: e_{a} = a_{E} + b_{E} \Delta T_{L}

    where :math:`a_{E} = \frac{(g_{tR} e_{R} + g_{t0} e_{0} + g_{v} e_{s}[T_{R}])}
    {(g_{tR} + g_{t0} + g_{v})}` and
    :math:`b_{E} = \frac{\Delta_{V} [T_{R}])}{(g_{tR} + g_{t0} + g_{v})}`.

    The latent heat term is given by

    .. math:: \lambda E = \frac{\lambda g_{v}}{p_{a}} (e_{L} - e_{A})

    Substituting :math:`e_{A}` for its linearized form, again assuming :math:`e_{L}`
    is approximated by :math:`e_{s} [T_{R}]+\Delta_{v} [T_{R}]\Delta T_{L}`, and
    rearranging gives:

    .. math:: \lambda E = a_{L} + b_{L} \Delta T_{L},

    where :math:`a_{L} = \frac{\lambda g_{v}}{p_{a}} (e_{s} [T_{R}] - a_{E})` and
    :math:`b_{L} = \frac{\lambda g_{v}}{p_{a}} (\Delta_{V} [T_{R}] - b_{E})`.

    The radiation emitted by the leaf :math:`R_{em}` is given by the Stefan Boltzmann
    law and can be linearised as follows:

    .. math:: R_{em} = a_{R} + b_{R} \Delta T_{L}

    where :math:`a_{R} = \epsilon_{s} \sigma a_{A}^{4}` and
    :math:`b_{R} = 4 \epsilon_{s} \sigma (a_{A}^{3} b_{A} + T_{R}^{3})`.

    The full heat balance equation for the difference between leaf and canopy air
    temperature becomes

    .. math:: \Delta T_{L} = \frac{R_{abs} - a_{R} - a_{L}}{(1 + b_{R} + b_{L} + b_{H})}

    The equation is then used to calculate air and leaf temperature as follows:

    .. math:: T_{A} = a_{A} + b_{A} \Delta T_{L}

    and

    .. math:: T_{L} = T_{A} + \Delta T_{L}.

    the data object has to contain the previous and current values for the following:

    * air_temperature_ref: Air temperature at reference height 2m above canopy, [C]
    * vapour_pressure_ref: vapour pressure at reference height 2m above canopy, [kPa]
    * soil_temperature: Soil temperature, [C]
    * soil_moisture: Relative soil moisture, dimensionless
    * atmospheric_pressure_ref: Atmospheric pressure at reference height, [kPa]
    * air_temperature: Air temperature, [C]
    * canopy_temperature: Leaf temperature, [C]
    * latent_heat_vapourisation: Latent heat of vapourisation, [J kg-1]
    * absorbed_radiation: Absorbed radiation, [W m-2]
    * specific_heat_air: Specific heat of air, [J mol-1 K-1]

    TODO
    * add latent heat flux from soil to atmosphere (-> VPD)
    * check time integration
    * set limits to temperature and VPD

    Args:
        data: Instance of data object
        time_index: Time index
        topsoil_layer_index: Index of top soil layer
        true_canopy_indexes: indexes of canopy layers that are not NaN
        true_canopy_layers_n: Maximum number of canopy layers that are not NaN
        layer_structure: Instance of LayerStructure that countains details about layers
        abiotic_constants: Set of abiotic constants
        abiotic_simple_constants: Set of abiotic constants
        core_constants: Set of core constants

    Returns:
        air temperature, [C], canopy temperature, [C], vapour pressure [kPa], vapour
        pressure deficit, [kPa]
    """

    output = {}

    # Select variables for current time step and relevant layers
    topsoil_temperature = data["soil_temperature"][topsoil_layer_index]
    topsoil_moisture = data["soil_moisture"][topsoil_layer_index]
    air_temperature_ref = data["air_temperature_ref"].isel(time_index=time_index)
    vapour_pressure_ref = data["vapour_pressure_ref"].isel(time_index=time_index)
    atmospheric_pressure_ref = data["atmospheric_pressure_ref"].isel(
        time_index=time_index
    )

    # Calculate vapour pressures
    soil_saturated_vapour_pressure = calculate_saturation_vapour_pressure(
        temperature=topsoil_temperature,
        saturation_vapour_pressure_factors=(
            abiotic_simple_constants.saturation_vapour_pressure_factors
        ),
    )
    soil_vapour_pressure = topsoil_moisture * soil_saturated_vapour_pressure
    saturated_vapour_pressure_ref = calculate_saturation_vapour_pressure(
        temperature=air_temperature_ref,
        saturation_vapour_pressure_factors=(
            abiotic_simple_constants.saturation_vapour_pressure_factors
        ),
    )

    # Calculate current conductivities for atmosphere and soil
    current_conductivities = calculate_current_conductivities(
        data=data,
        characteristic_dimension_leaf=core_constants.characteristic_dimension_leaf,
        von_karmans_constant=core_constants.von_karmans_constant,
        abiotic_constants=abiotic_constants,
    )

    conductivity_from_soil = (
        topsoil_moisture * soil_saturated_vapour_pressure
    ).to_numpy()

    # Factors from leaf and air temperature linearisation
    a_A, b_A = leaf_and_air_temperature_linearisation(
        conductivity_from_ref_height=(
            current_conductivities["conductivity_from_ref_height"][true_canopy_indexes]
        ),
        conductivity_from_soil=conductivity_from_soil,
        leaf_air_heat_conductivity=(
            current_conductivities["leaf_air_heat_conductivity"][true_canopy_indexes]
        ),
        air_temperature_ref=air_temperature_ref.to_numpy(),
        top_soil_temperature=topsoil_temperature.to_numpy(),
    )

    # Factors from longwave radiative flux linearisation
    a_R, b_R = longwave_radiation_flux_linearisation(
        a_A=a_A,
        b_A=b_A,
        air_temperature_ref=air_temperature_ref.to_numpy(),
        leaf_emissivity=abiotic_constants.leaf_emissivity,
        stefan_boltzmann_constant=core_constants.stefan_boltzmann_constant,
    )

    # Factors from vapour pressure linearisation
    delta_v_ref = calculate_slope_of_saturated_pressure_curve(
        air_temperature_ref.to_numpy()
    )

    a_E, b_E = vapour_pressure_linearisation(
        vapour_pressure_ref=vapour_pressure_ref.to_numpy(),
        saturated_vapour_pressure_ref=saturated_vapour_pressure_ref.to_numpy(),
        soil_vapour_pressure=soil_vapour_pressure.to_numpy(),
        conductivity_from_soil=conductivity_from_soil,
        leaf_vapour_conductivity=(
            current_conductivities["leaf_vapour_conductivity"][true_canopy_indexes]
        ),
        conductivity_from_ref_height=(
            current_conductivities["conductivity_from_ref_height"][true_canopy_indexes]
        ),
        delta_v_ref=delta_v_ref,
    )

    # Factors from latent heat flux linearisation
    a_L, b_L = latent_heat_flux_linearisation(
        latent_heat_vapourisation=(
            data["latent_heat_vapourisation"][true_canopy_indexes].to_numpy()
        ),
        leaf_vapour_conductivity=(
            current_conductivities["leaf_vapour_conductivity"][true_canopy_indexes]
        ),
        atmospheric_pressure_ref=atmospheric_pressure_ref.to_numpy(),
        saturated_vapour_pressure_ref=saturated_vapour_pressure_ref.to_numpy(),
        a_E=a_E,
        b_E=b_E,
        delta_v_ref=delta_v_ref,
    )

    # Factor from sensible heat flux linearisation
    b_H = (
        current_conductivities["leaf_air_heat_conductivity"][true_canopy_indexes]
        * data["specific_heat_air"][true_canopy_indexes].to_numpy()
    )

    # Calculate new leaf and air temperature
    delta_canopy_temperature = calculate_delta_canopy_temperature(
        absorbed_radiation=data["absorbed_radiation"][true_canopy_indexes].to_numpy(),
        a_R=a_R,
        a_L=a_L,
        b_R=b_R,
        b_L=b_L,
        b_H=b_H,
    )
    new_air_temperature = a_A + b_A * delta_canopy_temperature
    new_canopy_temperature = (
        data["air_temperature"][1 : true_canopy_layers_n + 1]
    ).to_numpy() + delta_canopy_temperature

    # Interpolate temperature below canopy
    # This could potentially be done without explicit below canopy layers
    target_heights = np.vstack(
        [
            data["layer_heights"]
            .where(data["layer_heights"]["layer_roles"] == "subcanopy")
            .dropna(dim="layers"),
            data["layer_heights"]
            .where(data["layer_heights"]["layer_roles"] == "surface")
            .dropna(dim="layers"),
        ]
    )
    below_canopy_temperature = interpolate_along_heights(
        start_height=np.repeat(0.0, data.grid.n_cells),
        end_height=data["layer_heights"][true_canopy_layers_n].to_numpy(),
        target_heights=target_heights,
        start_value=topsoil_temperature.to_numpy(),
        end_value=new_air_temperature[-1],
    )

    # Create arrays and return for data object
    new_temperature_profile = np.vstack(
        [
            air_temperature_ref.to_numpy(),
            new_air_temperature,
            np.full(
                (
                    layer_structure.canopy_layers - true_canopy_layers_n,
                    data.grid.n_cells,
                ),
                np.nan,
            ),
            below_canopy_temperature,
            np.full((len(layer_structure.soil_layers), data.grid.n_cells), np.nan),
        ]
    )
    output["air_temperature"] = DataArray(
        new_temperature_profile,
        dims=["layers", "cell_id"],
        coords=data["layer_heights"].coords,
    )
    output["canopy_temperature"] = DataArray(
        np.vstack(
            [
                np.repeat(np.nan, data.grid.n_cells),
                new_canopy_temperature,
                np.full(
                    (
                        layer_structure.n_layers - true_canopy_layers_n - 1,
                        data.grid.n_cells,
                    ),
                    np.nan,
                ),
            ]
        ),
        dims=["layers", "cell_id"],
    )

    # Calculate vapour pressure
    vapour_pressure_mean = a_E + b_E * delta_canopy_temperature
    vapour_pressure_new = vapour_pressure_ref.to_numpy() + 2 * (
        vapour_pressure_mean - vapour_pressure_ref.to_numpy()
    )

    saturation_vapour_pressure_new = calculate_saturation_vapour_pressure(
        DataArray(new_temperature_profile),
        saturation_vapour_pressure_factors=(
            abiotic_simple_constants.saturation_vapour_pressure_factors
        ),
    )
    saturation_vapour_pressure_new_canopy = (
        saturation_vapour_pressure_new[1 : true_canopy_layers_n + 1]
    ).to_numpy()

    canopy_vapour_pressure = np.where(
        vapour_pressure_new > saturation_vapour_pressure_new_canopy,
        saturation_vapour_pressure_new_canopy,
        vapour_pressure_new,
    )
    below_canopy_vapour_pressure = interpolate_along_heights(
        start_height=np.repeat(0.0, data.grid.n_cells),
        end_height=data["layer_heights"][true_canopy_layers_n].to_numpy(),
        target_heights=target_heights,
        start_value=soil_vapour_pressure.to_numpy(),
        end_value=canopy_vapour_pressure[-1],
    )
    output["vapour_pressure"] = DataArray(
        np.vstack(
            [
                vapour_pressure_ref.to_numpy(),
                canopy_vapour_pressure,
                np.full((7, data.grid.n_cells), np.nan),
                below_canopy_vapour_pressure,
                np.full((2, data.grid.n_cells), np.nan),
            ]
        ),
        dims=["layers", "cell_id"],
    )

    output["vapour_pressure_deficit"] = output["vapour_pressure"] / DataArray(
        saturation_vapour_pressure_new, dims=["layers", "cell_id"]
    )

    # Return current conductivities as DataArrays
    for var in [
        "conductivity_from_ref_height",
        "leaf_air_heat_conductivity",
        "leaf_vapour_conductivity",
    ]:
        output[var] = DataArray(
            current_conductivities[var],
            dims=data["air_temperature"].dims,
            coords=data["air_temperature"].coords,
            name=var,
        )

    # Return latent and sensible heat flux from canopy
    sensible_heat_flux = data["sensible_heat_flux"].copy()
    sensible_heat_flux_canopy = b_H * delta_canopy_temperature
    sensible_heat_flux[topsoil_layer_index] = data["sensible_heat_flux_soil"]
    sensible_heat_flux[true_canopy_indexes] = sensible_heat_flux_canopy
    output["sensible_heat_flux"] = sensible_heat_flux

    latent_heat_flux = data["latent_heat_flux"].copy()
    latent_heat_flux_canopy = a_L + b_L * delta_canopy_temperature
    latent_heat_flux[topsoil_layer_index] = data["latent_heat_flux_soil"]
    latent_heat_flux[true_canopy_indexes] = latent_heat_flux_canopy
    output["latent_heat_flux"] = latent_heat_flux

    return output


def leaf_and_air_temperature_linearisation(
    conductivity_from_ref_height: NDArray[np.float32],
    conductivity_from_soil: NDArray[np.float32],
    leaf_air_heat_conductivity: NDArray[np.float32],
    air_temperature_ref: NDArray[np.float32],
    top_soil_temperature: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate factors for leaf and air temperature linearisation.

    Args:
        conductivity_from_ref_height: Conductivity from reference height, [mol m-2 s-2]
        conductivity_from_soil: Conductivity from soil, [mol m-2 s-2]
        leaf_air_heat_conductivity: Leaf air heat conductivity, [mol m-2 s-2]
        air_temperature_ref: Air temperature at reference height 2m above the canopy,[C]
        top_soil_temperature: Top soil temperature, [C]

    Returns:
        Factors a_A and b_A for leaf and air temperature linearisation
    """

    a_A = (
        (conductivity_from_ref_height * air_temperature_ref)
        + (conductivity_from_soil * top_soil_temperature)
    ) / (conductivity_from_ref_height + conductivity_from_soil)

    b_A = leaf_air_heat_conductivity / (
        conductivity_from_ref_height + conductivity_from_soil
    )
    return a_A, b_A


def longwave_radiation_flux_linearisation(
    a_A: NDArray[np.float32],
    b_A: NDArray[np.float32],
    air_temperature_ref: NDArray[np.float32],
    leaf_emissivity: float,
    stefan_boltzmann_constant: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate factors for longwave radiative flux linearisation.

    Args:
        a_A: Factor for leaf and air temperature linearisation
        b_A: Factor for leaf and air temperature linearisation
        air_temperature_ref: Air temperature at reference height 2m above the canopy,[C]
        leaf_emissivity: Leaf emissivity, dimensionless
        stefan_boltzmann_constant: Stefan Boltzmann constant, [W m-2 K-4]

    Returns:
        Factors a_R and b_R for longwave radiative flux linearisation
    """

    a_R = leaf_emissivity * stefan_boltzmann_constant * a_A**4

    b_R = (
        4
        * leaf_emissivity
        * stefan_boltzmann_constant
        * (a_A**3 * b_A + air_temperature_ref**3)
    )
    return a_R, b_R


def vapour_pressure_linearisation(
    vapour_pressure_ref: NDArray[np.float32],
    saturated_vapour_pressure_ref: NDArray[np.float32],
    soil_vapour_pressure: NDArray[np.float32],
    conductivity_from_soil: NDArray[np.float32],
    leaf_vapour_conductivity: NDArray[np.float32],
    conductivity_from_ref_height: NDArray[np.float32],
    delta_v_ref: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate factors for vapour pressure linearisation.

    Args:
        vapour_pressure_ref: Vapour pressure at reference height 2 m above canopy, [kPa]
        saturated_vapour_pressure_ref: Saturated vapour pressure at reference height 2 m
            above canopy, [kPa]
        soil_vapour_pressure: Soil vapour pressure, [kPa]
        conductivity_from_soil: Conductivity from soil TODO unit
        leaf_vapour_conductivity: Leaf vapour conductivity, [mol m-2 s-1]
        conductivity_from_ref_height: Conductivity frm reference height, [mol m-2 s-1]
        delta_v_ref: Slope of saturated vapour pressure curve

    Returns:
        Factors a_E and b_E for vapour pressure linearisation
    """

    a_E = (
        conductivity_from_ref_height * vapour_pressure_ref
        + conductivity_from_soil * soil_vapour_pressure
        + leaf_vapour_conductivity * saturated_vapour_pressure_ref
    ) / (
        conductivity_from_ref_height + conductivity_from_soil + leaf_vapour_conductivity
    )

    b_E = delta_v_ref / (
        conductivity_from_ref_height + conductivity_from_soil + leaf_vapour_conductivity
    )
    return a_E, b_E


def latent_heat_flux_linearisation(
    latent_heat_vapourisation: NDArray[np.float32],
    leaf_vapour_conductivity: NDArray[np.float32],
    atmospheric_pressure_ref: NDArray[np.float32],
    saturated_vapour_pressure_ref: NDArray[np.float32],
    a_E: NDArray[np.float32],
    b_E: NDArray[np.float32],
    delta_v_ref: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Calculate factors for latent heat flux linearisation.

    Args:
        latent_heat_vapourisation: latent heat of vapourisation
        leaf_vapour_conductivity: leaf vapour conductivity, [mol m-2 s-1]
        atmospheric_pressure_ref: Atmospheric pressure at reference height 2 m above
            canopy, [kPa]
        saturated_vapour_pressure_ref: Satuated vapour pressure at reference height 2 m
            above canopy, [kPa]
        a_E: Factor for vapour pressure linearisation
        b_E: Factor for vapour pressure linearisation
        delta_v_ref: Slope of saturated vapour pressure curve

    Returns:
        Factors a_L and b_L for latent heat flux linearisation
    """

    a_L = (latent_heat_vapourisation * leaf_vapour_conductivity) / (
        atmospheric_pressure_ref * (saturated_vapour_pressure_ref - a_E)
    )

    b_L = (latent_heat_vapourisation * leaf_vapour_conductivity) / (
        atmospheric_pressure_ref * (delta_v_ref - b_E)
    )

    return a_L, b_L


def calculate_delta_canopy_temperature(
    absorbed_radiation: NDArray[np.float32],
    a_R: NDArray[np.float32],
    a_L: NDArray[np.float32],
    b_R: NDArray[np.float32],
    b_L: NDArray[np.float32],
    b_H: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate change in canopy temperature (delta).

    Args:
        absorbed_radiation: Radiation (shortwave) absorved by canopy, [W m-2]
        a_R: Factor for longwave radiation emission linearisation
        a_L: Factor for latent heat flux linearisation
        b_R: Factor for longwave radiation emission linearisation
        b_L: Factor for latent heat flux linearisation
        b_H: Factor for sensible heat flux linearisation

    Returns:
        Change in canopy temperature, [C]
    """

    return (absorbed_radiation - a_R - a_L) / (1 + b_R + b_L + b_H)
