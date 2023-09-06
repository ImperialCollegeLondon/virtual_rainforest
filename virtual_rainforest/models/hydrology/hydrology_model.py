"""The :mod:`~virtual_rainforest.models.hydrology.hydrology_model` module
creates a
:class:`~virtual_rainforest.models.hydrology.hydrology_model.HydrologyModel`
class as a child of the :class:`~virtual_rainforest.core.base_model.BaseModel` class.
At present a lot of the abstract methods of the parent class (e.g.
:func:`~virtual_rainforest.core.base_model.BaseModel.spinup`) are overwritten using
placeholder functions that don't do anything. This will change as the Virtual Rainforest
model develops. The factory method
:func:`~virtual_rainforest.models.hydrology.hydrology_model.HydrologyModel.from_config`
exists in a more complete state, and unpacks a small number of parameters from our
currently pretty minimal configuration dictionary. These parameters are then used to
generate a class instance. If errors crop here when converting the information from the
config dictionary to the required types they are caught and then logged, and at the end
of the unpacking an error is thrown. This error should be caught and handled by
downstream functions so that all model configuration failures can be reported as one.
"""  # noqa: D205, D415

from __future__ import annotations

from math import sqrt
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from pint import Quantity
from xarray import DataArray

from virtual_rainforest.core.base_model import BaseModel
from virtual_rainforest.core.constants import load_constants
from virtual_rainforest.core.data import Data
from virtual_rainforest.core.exceptions import InitialisationError
from virtual_rainforest.core.grid import Grid
from virtual_rainforest.core.logger import LOGGER
from virtual_rainforest.core.utils import set_layer_roles
from virtual_rainforest.models.hydrology.constants import HydroConsts


class HydrologyModel(BaseModel):
    """A class describing the hydrology model.

    Args:
        data: The data object to be used in the model.
        update_interval: Time to wait between updates of the model state.
        soil_layers: The number of soil layers to be modelled.
        canopy_layers: The initial number of canopy layers to be modelled.
        initial_soil_moisture: The initial volumetric relative water content [unitless]
            for all layers.
        constants: Set of constants for the hydrology model.

    Raises:
        InitialisationError: when initial soil moisture is out of bounds.
    """

    model_name = "hydrology"
    """The model name for use in registering the model and logging."""
    lower_bound_on_time_scale = "1 day"
    """Shortest time scale that hydrology model can sensibly capture."""
    upper_bound_on_time_scale = "1 month"
    """Longest time scale that hydrology model can sensibly capture."""
    required_init_vars = (
        ("precipitation", ("spatial",)),
        ("leaf_area_index", ("spatial",)),
        ("air_temperature_ref", ("spatial",)),
        ("relative_humidity_ref", ("spatial",)),
        ("atmospheric_pressure_ref", ("spatial",)),
        ("evapotranspiration", ("spatial",)),
        ("elevation", ("spatial",)),
        ("surface_runoff", ("spatial",)),
        # TODO this requires the plant model to run before the hydrology; this works as
        # long as the p-model does not require soil moisture as an input. If it does, we
        # have to discuss where we move the calculation of stream flow.
    )
    # TODO add time dimension
    """The required variables and axes for the hydrology model"""

    vars_updated = (
        "precipitation_surface",  # precipitation-interception loss, input to `plants`
        "soil_moisture",
        "surface_runoff",  # equivalent to SPLASH runoff
        "vertical_flow",
        "soil_evaporation",
        "stream_flow",  # P-ET; TODO later surface_runoff_acc + below_ground_acc
        "surface_runoff_accumulated",
    )
    """Variables updated by the hydrology model."""

    def __init__(
        self,
        data: Data,
        update_interval: Quantity,
        soil_layers: int,
        canopy_layers: int,
        initial_soil_moisture: float,
        constants: HydroConsts,
        **kwargs: Any,
    ):
        # Sanity checks for initial soil moisture
        if type(initial_soil_moisture) is not float:
            to_raise = InitialisationError("The initial soil moisture must be a float!")
            LOGGER.error(to_raise)
            raise to_raise

        if initial_soil_moisture < 0 or initial_soil_moisture > 1:
            to_raise = InitialisationError(
                "The initial soil moisture has to be between 0 and 1!"
            )
            LOGGER.error(to_raise)
            raise to_raise

        super().__init__(data, update_interval, **kwargs)

        # Create a list of layer roles
        layer_roles = set_layer_roles(canopy_layers, soil_layers)

        self.data
        """A Data instance providing access to the shared simulation data."""
        self.layer_roles = layer_roles
        """A list of vertical layer roles."""
        self.update_interval
        """The time interval between model updates."""
        self.initial_soil_moisture = initial_soil_moisture
        """Initial volumetric relative water content [unitless] for all layers and grid
        cells identical."""
        self.constants = constants
        """Set of constants for the hydrology model"""
        self.data.grid.set_neighbours(distance=sqrt(self.data.grid.cell_area))
        """Set neighbours."""
        self.drainage_map = calculate_drainage_map(
            grid=self.data.grid,
            elevation=np.array(self.data["elevation"]),
        )
        """Upstream neighbours for the calculation of accumulated runoff."""

    @classmethod
    def from_config(
        cls, data: Data, config: dict[str, Any], update_interval: Quantity
    ) -> HydrologyModel:
        """Factory function to initialise the hydrology model from configuration.

        This function unpacks the relevant information from the configuration file, and
        then uses it to initialise the model. If any information from the config is
        invalid rather than returning an initialised model instance an error is raised.

        Args:
            data: A :class:`~virtual_rainforest.core.data.Data` instance.
            config: The complete (and validated) Virtual Rainforest configuration.
            update_interval: Frequency with which all models are updated.
        """

        # Find number of soil and canopy layers
        soil_layers = config["core"]["layers"]["soil_layers"]
        canopy_layers = config["core"]["layers"]["canopy_layers"]
        initial_soil_moisture = config["hydrology"]["initial_soil_moisture"]

        # Load in the relevant constants
        constants = load_constants(config, "hydrology", "HydroConsts")

        LOGGER.info(
            "Information required to initialise the hydrology model successfully "
            "extracted."
        )
        return cls(
            data,
            update_interval,
            soil_layers,
            canopy_layers,
            initial_soil_moisture,
            constants,
        )

    def setup(self) -> None:
        """Function to set up the hydrology model.

        At the moment, this function initializes variables that are required to run the
        first update(). For the within grid cell hydrology, soil moisture is initialised
        homogenously for all soil layers. This design might change with the
        implementation of the SPLASH model in the plant module which will take care of
        the above-ground hydrology. Air temperature and relative humidity below the
        canopy are set to the 2 m reference values.

        For the hydrology across the grid (above-/below-ground and accumulated runoff),
        this function uses the upstream neighbours of each grid cell (see
        :func:`~virtual_rainforest.models.hydrology.hydrology_model.calculate_drainage_map`
        ).

        TODO implement below-ground horizontal flow and update stream flow
        TODO potentially move `calculate_drainage_map` to core
        """

        # Create 1-dimensional numpy array filled with initial soil moisture values for
        # all soil layers and np.nan for atmosphere layers
        soil_moisture_values = np.repeat(
            a=[np.nan, self.initial_soil_moisture],
            repeats=[
                len(self.layer_roles) - self.layer_roles.count("soil"),
                self.layer_roles.count("soil"),
            ],
        )
        # Broadcast 1-dimensional array to grid and assign dimensions and coordinates
        self.data["soil_moisture"] = DataArray(
            np.broadcast_to(
                soil_moisture_values,
                (len(self.data.grid.cell_id), len(self.layer_roles)),
            ).T,
            dims=["layers", "cell_id"],
            coords={
                "layers": np.arange(len(self.layer_roles)),
                "layer_roles": ("layers", self.layer_roles),
                "cell_id": self.data.grid.cell_id,
            },
            name="soil_moisture",
        )

        # Create initial air temperature with reference temperature below the canopy
        # for first soil evaporation update.
        self.data["air_temperature"] = (
            DataArray(self.data["air_temperature_ref"].isel(time_index=0))
            .expand_dims("layers")
            .rename("air_temperature")
            .assign_coords(
                coords={
                    "layers": [self.layer_roles.index("subcanopy")],
                    "layer_roles": ("layers", ["subcanopy"]),
                    "cell_id": self.data.grid.cell_id,
                },
            )
        )

        # Create initial relative humidity with reference humidity below the canopy
        # for first soil evaporation update.
        self.data["relative_humidity"] = (
            DataArray(self.data["relative_humidity_ref"].isel(time_index=0))
            .expand_dims("layers")
            .rename("relative_humidity")
            .assign_coords(
                coords={
                    "layers": [self.layer_roles.index("subcanopy")],
                    "layer_roles": ("layers", ["subcanopy"]),
                    "cell_id": self.data.grid.cell_id,
                },
            )
        )

        # Get the runoff created by SPLASH or initial data set as the initial state:
        initial_runoff = np.array(self.data["surface_runoff"])

        # Set initial accumulated runoff to zero
        accumulated_runoff = np.zeros_like(self.data["elevation"])

        # Calculate accumulated surface runoff for each cell
        new_accumulated_runoff = accumulate_surface_runoff(
            drainage_map=self.drainage_map,
            surface_runoff=initial_runoff,
            accumulated_runoff=accumulated_runoff,
        )

        self.data["surface_runoff_accumulated"] = DataArray(
            new_accumulated_runoff,
            dims="cell_id",
            name="surface_runoff_accumulated",
            coords={"cell_id": self.data.grid.cell_id},
        )

    def spinup(self) -> None:
        """Placeholder function to spin up the hydrology model."""
        # TODO soil moisture and accumulated runoff need a spin up

    def update(self, time_index: int) -> None:
        r"""Function to update the hydrology model.

        At the moment, this step calculates surface precipitation, soil moisture,
        vertical flow, soil evaporation, and surface runoff (per grid cell and
        accumulated), and estimates mean stream flow. These processes are problematic
        at a monthly timestep, which is why - as an intermediate step - the input
        precipitation is divided by 30 days and return variables are means or
        accumulated values.

        Surface runoff is calculated with a simple bucket model based on
        :cite:t:`davis_simple_2017`: if precipitation exceeds top soil moisture capacity
        , the excess water is added to runoff and top soil moisture is set to soil
        moisture capacity value; if the top soil is not saturated, precipitation is
        added to the current soil moisture level and runoff is set to zero. Note that
        this function will likely change with the implementation of the SPLASH model
        :cite:p:`davis_simple_2017` in the plant module which will take care of the grid
        cell based above-ground hydrology. The accumulated surface runoff is calculated
        as the sum of current runoff and the runoff from upstream cells at the previous
        time step.

        Soil evaporation is calculated with classical bulk aerodynamic formulation,
        following the so-called ':math:`\alpha` method', see
        :func:`~virtual_rainforest.models.hydrology.hydrology_model.calculate_soil_evaporation`
        .

        Vertical flow between soil layers is calculated using the Richards equation, see
        :func:`~virtual_rainforest.models.hydrology.hydrology_model.calculate_vertical_flow`
        . That function returns total vertical flow in mm. Note that there are
        severe limitations to this approach on the temporal and spatial scale of this
        model and this can only be treated as a very rough approximation!

        Soil moisture is updated by iteratively updating the soil moisture of individual
        layers under consideration of the vertical flow in and out of each layer, see
        :func:`~virtual_rainforest.models.hydrology.hydrology_model.update_soil_moisture`
        .

        Mean stream flow :math:`Q` is currently estimated with a simple catchment water
        balance as

        :math:`Q = P - ET - \Delta S`

        where :math:`P` is mean precipitation, :math:`ET` is evapotranspiration, and
        :math:`\Delta S` is the change in soil moisture. Note that this has to be called
        after evapotranspiration is calculated by the plant model which works as long as
        the P-model does not require moisture as an input. In the future, this
        might move to a different model or the order of models might change.

        The function requires the following input variables from the data object:

        * air temperature, [C]
        * relative humidity, []
        * atmospheric pressure, [kPa]
        * precipitation, [mm]
        * wind speed (currently not implemented, default = 0.1 m s-1)
        * leaf area index, [m m-2]
        * layer heights, [m]
        * Volumetric relative water content (previous time step), [unitless]
        * evapotranspiration (current time step), [mm]
        * accumulated surface runoff (previous time step), [mm]

        and the following soil parameters (defaults in
        :class:`~virtual_rainforest.models.hydrology.constants.HydroConsts`):

        * soil moisture capacity, [unitless]
        * soil moisture residual, [unitless]
        * soil hydraulic conductivity, [m s-1]
        * soil hydraulic gradient, [m m-1]
        * van Genuchten non-linearity parameter, dimensionless

        and a number of additional parameters that as described in detail in
        :class:`~virtual_rainforest.models.hydrology.constants.HydroConsts`.

        The function updates the following variables in the `data` object:

        * precipitation_surface, [mm]
        * soil_moisture, [-]
        * surface_runoff, [mm], equivalent to SPLASH runoff
        * vertical_flow, [mm/timestep]
        * soil_evaporation, [mm]
        * stream_flow, [mm/timestep], currently simply P-ET
        * surface_runoff_accumulated, [mm]
        """
        # select time conversion factor
        # TODO allow for other time steps and make it an option to loop over days to
        # calculate monthly statistics
        if self.update_interval != Quantity("1 month"):
            to_raise = NotImplementedError("This time step is currently not supported.")
            LOGGER.error(to_raise)
            raise to_raise

        time_conversion_factor = self.constants.seconds_to_month
        days = 30  # TODO this is not permanent

        # Select variables at relevant heights for current time step
        current_precipitation = np.array(
            self.data["precipitation"].isel(time_index=time_index)
        )
        leaf_area_index_sum = np.array(self.data["leaf_area_index"].sum(dim="layers"))
        evapotranspiration = np.array(self.data["evapotranspiration"].sum(dim="layers"))
        subcanopy_temperature = np.array(
            self.data["air_temperature"].isel(
                layers=self.layer_roles.index("subcanopy")
            )
        )
        subcanopy_humidity = np.array(
            self.data["relative_humidity"].isel(
                layers=self.layer_roles.index("subcanopy")
            )
        )
        subcanopy_pressure = np.array(
            self.data["atmospheric_pressure_ref"].isel(time_index=time_index)
        )
        soil_layer_heights = np.array(
            self.data["layer_heights"]
            .where(self.data["layer_heights"].layer_roles == "soil")
            .dropna(dim="layers")
        )

        # Calculate thickness of each layer, [mm]
        soil_layer_thickness = np.array(
            [
                (soil_layer_heights[i] - soil_layer_heights[i - 1])
                * (-self.constants.meters_to_mm)
                if i > 0
                else soil_layer_heights[0] * (-self.constants.meters_to_mm)
                for i in range(len(soil_layer_heights))
            ],
        )

        # Convert soil moisture (volumetric relative water content) to mm as follows:
        # water content in mm = relative water content / 100 * depth in mm
        # Example: for 20% water at 40 cm this would be: 20/100 * 400mm = 80 mm
        soil_moisture_mm = np.array(
            self.data["soil_moisture"]
            .where(self.data["soil_moisture"].layer_roles == "soil")
            .dropna(dim="layers")
            * soil_layer_thickness
        )

        # create output dict as intermediate step to not overwrite data directly
        soil_hydrology = {}

        # TODO The following section calculates an average day, will later need to loop
        current_precipitation = current_precipitation / days

        # Interception of water in canopy, [mm]
        interception = estimate_interception(
            leaf_area_index=leaf_area_index_sum,
            precipitation=current_precipitation,
            intercept_param_1=self.constants.intercept_param_1,
            intercept_param_2=self.constants.intercept_param_2,
            intercept_param_3=self.constants.intercept_param_3,
            veg_density_param=self.constants.veg_density_param,
        )

        # Precipitation that reaches the surface per day, [mm]
        precipitation_surface = current_precipitation - interception

        # Return monthly accumulated precipitation at surface, [mm]
        soil_hydrology["precipitation_surface"] = DataArray(
            precipitation_surface * days,
            dims="cell_id",
            coords={"cell_id": self.data.grid.cell_id},
        )

        # Calculate how much water can be added to soil before capacity is reached, [mm]
        free_capacity_mm = (
            self.constants.soil_moisture_capacity * soil_layer_thickness
            - soil_moisture_mm
        )

        # Calculate daily surface runoff of each grid cell, [mm]; replace by SPLASH
        surface_runoff = np.where(
            precipitation_surface > free_capacity_mm[0],
            precipitation_surface - free_capacity_mm[0],
            0,
        )

        # Return accumulated surface runoff, [mm]
        soil_hydrology["surface_runoff"] = DataArray(
            surface_runoff * days,
            dims="cell_id",
            coords={"cell_id": self.data["soil_moisture"].cell_id},
        )

        # Calculate top soil moisture after infiltration, [mm]
        soil_moisture_infiltrated = np.clip(
            soil_moisture_mm[0] + precipitation_surface,
            0,
            (self.constants.soil_moisture_capacity * soil_layer_thickness[0]),
        )

        # Calculate daily soil evaporation, [mm]
        soil_evaporation = calculate_soil_evaporation(
            temperature=subcanopy_temperature,
            relative_humidity=subcanopy_humidity,
            atmospheric_pressure=subcanopy_pressure,
            soil_moisture=soil_moisture_infiltrated / soil_layer_thickness[0],  # vol
            wind_speed=0.1,  # m/s TODO wind_speed in data object (mechanistic model)
            celsius_to_kelvin=self.constants.celsius_to_kelvin,
            density_air=self.constants.density_air,
            latent_heat_vapourisation=self.constants.latent_heat_vapourisation,
            gas_constant_water_vapour=self.constants.gas_constant_water_vapour,
            heat_transfer_coefficient=self.constants.heat_transfer_coefficient,
        )

        # Return accumulated soil evaporation, [mm]
        soil_hydrology["soil_evaporation"] = DataArray(
            soil_evaporation * days,
            dims="cell_id",
            coords={"cell_id": self.data.grid.cell_id},
        )

        # Calculate top soil moisture after evap and combine with lower layers, [mm]
        soil_moisture_evap: NDArray = np.concatenate(
            (
                # np.expand_dims((soil_moisture_infiltrated - soil_evaporation), axis=0)
                np.expand_dims((soil_moisture_infiltrated), axis=0),
                soil_moisture_mm[1:],
            )
        )

        # Calculate vertical flow between soil layers in mm per time step
        # Note that there are severe limitations to this approach on the temporal and
        # spatial scale of this model and this can only be treated as a very rough
        # approximation to discuss nutrient leaching.

        vertical_flow = calculate_vertical_flow(
            soil_moisture=soil_moisture_evap / soil_layer_thickness,  # vol
            soil_layer_thickness=soil_layer_thickness,  # mm
            soil_moisture_capacity=self.constants.soil_moisture_capacity,  # vol
            soil_moisture_residual=self.constants.soil_moisture_residual,  # vol
            hydraulic_conductivity=self.constants.hydraulic_conductivity,  # m/s
            hydraulic_gradient=self.constants.hydraulic_gradient,  # m/m
            nonlinearily_parameter=self.constants.nonlinearily_parameter,
            groundwater_capacity=self.constants.groundwater_capacity,
            timestep_conversion_factor=time_conversion_factor / days,
        )

        # Return accumulated vertical flow, [mm]
        soil_hydrology["vertical_flow"] = DataArray(
            np.sum(vertical_flow * days, axis=0),
            dims="cell_id",
            coords={"cell_id": self.data.grid.cell_id},
        )

        # Update soil moisture by subtractung/adding vertical flow to each layer, [mm]
        soil_moisture_updated = update_soil_moisture(
            soil_moisture=soil_moisture_evap,
            vertical_flow=vertical_flow,
            soil_moisture_capacity=(
                self.constants.soil_moisture_capacity * soil_layer_thickness
            ),
            soil_moisture_residual=(
                self.constants.soil_moisture_residual * soil_layer_thickness
            ),
        )

        soil_hydrology["sm_updated"] = DataArray(soil_moisture_updated)
        # TODO Remove plant evapotranspiration from second soil layer
        soil_moisture_et = np.where(
            soil_moisture_updated[1] - evapotranspiration / days < 0,
            self.constants.soil_moisture_residual * soil_layer_thickness[1],
            soil_moisture_updated[1] - evapotranspiration / days,
        )

        # Return mean soil moisture, [-], and add atmospheric layers (nan)
        soil_hydrology["soil_moisture"] = DataArray(
            np.concatenate(
                (
                    np.full(
                        (
                            len(self.layer_roles) - self.layer_roles.count("soil"),
                            len(self.data.grid.cell_id),
                        ),
                        np.nan,
                    ),
                    np.expand_dims(
                        soil_moisture_updated[0] / soil_layer_thickness[0], axis=0
                    ),
                    np.expand_dims(soil_moisture_et / soil_layer_thickness[1], axis=0),
                ),
            ),
            dims=self.data["soil_moisture"].dims,
            coords=self.data["soil_moisture"].coords,
        )

        # TODO Convert to matric potential

        # Calculate accumulated surface runoff
        # Get the runoff created by SPLASH or initial data set
        single_cell_runoff = np.array(soil_hydrology["surface_runoff"])

        # Get accumulated runoff from previous time step
        accumulated_runoff = np.array(self.data["surface_runoff_accumulated"])

        # Calculate accumulated runoff for each cell (me + sum of upstream neighbours)
        new_accumulated_runoff = accumulate_surface_runoff(
            drainage_map=self.drainage_map,
            surface_runoff=single_cell_runoff,
            accumulated_runoff=accumulated_runoff,
        )

        soil_hydrology["surface_runoff_accumulated"] = DataArray(
            new_accumulated_runoff, dims="cell_id"
        )

        # Calculate stream flow as Q= P-ET-dS ; vertical flow is not considered
        # TODO add vertical and below-ground horizontal flow
        # The maximum stream flow capacity is set to an arbitray value, could be used to
        # flag flood events

        soil_moisture_change = np.array(
            (
                (self.data["soil_moisture"]).sum(dim="layers")
                / np.sum(soil_layer_thickness)
            )
            - (
                soil_hydrology["soil_moisture"].sum(dim="layers")
                / np.sum(soil_layer_thickness)
            )
        )

        soil_hydrology["stream_flow"] = DataArray(
            np.clip(
                (
                    precipitation_surface * days
                    - evapotranspiration
                    - soil_moisture_change
                ),
                0,
                HydroConsts.stream_flow_capacity,
            ).squeeze(),
            dims="cell_id",
        )

        # Update data object
        self.data.add_from_dict(output_dict=soil_hydrology)

    def cleanup(self) -> None:
        """Placeholder function for hydrology model cleanup."""


def calculate_vertical_flow(
    soil_moisture: NDArray,
    soil_layer_thickness: NDArray,
    soil_moisture_capacity: Union[float, NDArray],
    soil_moisture_residual: Union[float, NDArray],
    hydraulic_conductivity: Union[float, NDArray],
    hydraulic_gradient: Union[float, NDArray],
    nonlinearily_parameter: Union[float, NDArray],
    groundwater_capacity: Union[float, NDArray],
    timestep_conversion_factor: float,
) -> NDArray:
    r"""Calculate vertical water flow through soil column.

    To calculate the flow of water through unsaturated soil, this function uses the
    Richards equation. First, the function calculates the effective saturation :math:`S`
    and effective hydraulic conductivity :math:`K(S)` based on the moisture content
    :math:`\Theta` using the van Genuchten/Mualem model:

    :math:`S = \frac{\Theta - \Theta_{r}}{\Theta_{s} - \Theta_{r}}`

    and

    :math:`K(S) = K_{s}* \sqrt{S} *(1-(1-S^{1/m})^{m})^{2}`

    where :math:`\Theta_{r}` is the residual moisture content, :math:`\Theta_{s}` is the
    saturated moisture content, :math:`K_{s}` is the saturated hydraulic conductivity,
    and :math:`m=1-1/n` is a shape parameter derived from the non-linearity parameter
    :math:`n`. Then, the function applies Darcy's law to calculate the water flow rate
    :math:`q` in :math:`\frac{m^3}{s^1}` considering the effective hydraulic
    conductivity:

    :math:`q = - K(S)*(\frac{dh}{dl}-1)`

    where :math:`\frac{dh}{dl}` is the hydraulic gradient with :math:`l` the
    length of the flow path in meters (here equal to the soil depth).

    Note that there are severe limitations to this approach on the temporal and
    spatial scale of this model and this can only be treated as a very rough
    approximation!

    Args:
        soil_moisture: Volumetric relative water content in top soil, [unitless]
        soil_layer_thickness: Thickness of all soil_layers, [mm]
        soil_moisture_capacity: soil moisture capacity, [unitless]
        soil_moisture_residual: residual soil moisture, [unitless]
        hydraulic_conductivity: hydraulic conductivity of soil, [m/s]
        hydraulic_gradient: hydraulic gradient (change in hydraulic head) along the flow
            path, positive values indicate downward flow, [m/m]
        nonlinearily_parameter: dimensionless parameter in van Genuchten model that
            describes the degree of nonlinearity of the relationship between the
            volumetric water content and the soil matric potential.
        groundwater_capacity: storage capacity of groupwater
        timestep_conversion_factor: factor to convert between m^3 per second and mm per
            model time step

    Returns:
        volumetric flow rate of water, [mm/timestep]
    """
    m = 1 - 1 / nonlinearily_parameter

    # Calculate soil effective saturation in rel. vol. water content for each layer:
    effective_saturation = (soil_moisture - soil_moisture_residual) / (
        soil_moisture_capacity - soil_moisture_residual
    )

    # Calculate the effective hydraulic conductivity in m/s
    effective_conductivity = np.array(
        hydraulic_conductivity
        * np.sqrt(effective_saturation)
        * (1 - (1 - (effective_saturation) ** (1 / m)) ** m) ** 2,
    )

    # Calculate flow from top soil to lower soil in mm per month
    flow = (
        -effective_conductivity * (hydraulic_gradient - 1) * timestep_conversion_factor
    )

    # Make sure that flow does not exceed storage capacity in mm
    available_storage = (soil_moisture - soil_moisture_residual) * soil_layer_thickness

    flow_min = []
    for i in np.arange(len(soil_moisture) - 1):
        flow_layer = np.where(
            effective_conductivity[i] < available_storage[i + 1],
            flow[i],
            available_storage[i + 1],
        )
        flow_min.append(flow_layer)

    groundwater_storage = groundwater_capacity * np.sum(soil_layer_thickness, axis=0)

    outflow = np.where(
        effective_conductivity[-1] < groundwater_storage,
        flow[-1],
        groundwater_storage,
    )
    flow_min.append(outflow)

    return np.array(flow_min)


def calculate_soil_evaporation(
    temperature: NDArray,
    relative_humidity: NDArray,
    atmospheric_pressure: NDArray,
    soil_moisture: NDArray,
    wind_speed: Union[float, NDArray],
    celsius_to_kelvin: float,
    density_air: Union[float, NDArray],
    latent_heat_vapourisation: Union[float, NDArray],
    gas_constant_water_vapour: float,
    heat_transfer_coefficient: float,
) -> NDArray:
    r"""Calculate soil evaporation based classical bulk aerodynamic formulation.

    This function uses the so-called 'alpha' method to estimate the evaporative flux.
    We here use the implementation by Barton (1979):

    :math:`\alpha = \frac{1.8 * soil moisture}{soil moisture + 0.3}`

    :math:`E_{g} = \frac{\rho_{air}}{R_{a}} * (\alpha * q_{sat}(T_{s}) - q_{g})`

    where :math:`E_{g}` is the evaporation flux (W m-2), :math:`\rho_{air}` is the
    density of air (kg m-3), :math:`R_{a}` is the aerodynamic resistance (unitless),
    :math:`q_{sat}(T_{s})` (unitless) is the saturated specific humidity, and
    :math:`q_{g}` is the surface specific humidity (unitless); see Mahfouf (1991).

    TODO add references
    TODO move constants to HydroConsts or CoreConstants and check values

    Args:
        temperature: air temperature at reference height, [C]
        relative_humidity: relative humidity at reference height, []
        atmospheric_pressure: atmospheric pressure at reference height, [kPa]
        soil_moisture: Volumetric relative water content, [unitless]
        wind_speed: wind speed at reference height, [m s-1]
        celsius_to_kelvin: factor to convert teperature from Celsius to Kelvin
        density_air: density if air, [kg m-3]
        latent_heat_vapourisation: latent heat of vapourisation, [J kg-1]
        gas_constant_water_vapour: gas constant for water vapour, [J kg-1 K-1]
        heat_transfer_coefficient: heat transfer coefficient of air

    Returns:
        soil evaporation, [mm]
    """

    # Convert temperature to Kelvin
    temperature_k = temperature + celsius_to_kelvin

    # Estimate alpha using the Barton (1979) equation
    barton_ratio = (1.8 * soil_moisture) / (soil_moisture + 0.3)
    alpha = np.where(barton_ratio > 1, 1, barton_ratio)

    saturation_vapour_pressure = 0.6112 * np.exp(
        (17.67 * (temperature_k)) / (temperature_k + 243.5)
    )

    saturated_specific_humidity = (
        gas_constant_water_vapour / latent_heat_vapourisation
    ) * (
        saturation_vapour_pressure / (atmospheric_pressure - saturation_vapour_pressure)
    )

    specific_humidity_air = (relative_humidity * saturated_specific_humidity) / 100

    aerodynamic_resistance = heat_transfer_coefficient / (wind_speed) ** 2

    evaporative_flux = (density_air / aerodynamic_resistance) * (  # W/m2
        alpha * saturation_vapour_pressure - specific_humidity_air
    )

    # Return surface evaporation in mm; TODO note that this is just for step
    return (evaporative_flux / latent_heat_vapourisation).squeeze()


def find_lowest_neighbour(
    neighbours: list[np.ndarray],
    elevation: np.ndarray,
) -> list[int]:
    """Find lowest neighbour for each grid cell from digital elevation model.

    This function finds the cell IDs of the lowest neighbour for each grid cell. This
    can be used to determine in which direction surface runoff flows.

    Args:
        neighbours: list of neighbour IDs
        elevation: elevation, [m]

    Returns:
        list of lowest neighbour IDs
    """
    lowest_neighbour = []
    for cell_id, neighbors_id in enumerate(neighbours):
        downstream_id_loc = np.argmax(elevation[cell_id] - elevation[neighbors_id])
        lowest_neighbour.append(neighbors_id[downstream_id_loc])

    return lowest_neighbour


def find_upstream_cells(lowest_neighbour: list[int]) -> list[list[int]]:
    """Find all upstream cell IDs for all grid cells.

    This function identifies all cell IDs that are upstream of each grid cell. This can
    be used to calculate the water flow that goes though a grid cell.

    Args:
        lowest_neighbour: list of lowest neighbour cell_ids

    Returns:
        lists of all upstream IDs for each grid cell
    """
    upstream_ids: list = [[] for i in range(len(lowest_neighbour))]

    for down_s, up_s in enumerate(lowest_neighbour):
        upstream_ids[up_s].append(down_s)

    return upstream_ids


def accumulate_surface_runoff(
    drainage_map: dict[int, list[int]],
    surface_runoff: np.ndarray,
    accumulated_runoff: np.ndarray,
) -> np.ndarray:
    """Calculate accumulated surface runoff for each grid cell.

    This function takes the accumulated surface runoff from the previous timestep and
    adds all surface runoff of the current time step from upstream cell IDs.

    The function currently raises a `ValueError` if accumulated runoff is negative.

    Args:
        drainage_map: dict of all upstream IDs for each grid cell
        surface_runoff: surface runoff of the current time step, [mm]
        accumulated_runoff: accumulated surface runoff from previous time step, [mm]

    Returns:
        accumulated surface runoff, [mm]
    """

    for cell_id, upstream_ids in enumerate(drainage_map.values()):
        accumulated_runoff[cell_id] += np.sum(surface_runoff[upstream_ids])

    if (accumulated_runoff < 0.0).any():
        to_raise = ValueError("The accumulated surface runoff should not be negative!")
        LOGGER.error(to_raise)
        raise to_raise

    return accumulated_runoff


# TODO move this to core.grid once we decided on common use
def calculate_drainage_map(grid: Grid, elevation: np.ndarray) -> dict[int, list[int]]:
    """Calculate drainage map based on digital elevation model.

    This function finds the lowest neighbour for each grid cell, identifies all upstream
    IDs and creates a dictionary that provides all upstream cell IDs for each grid
    cell. This function currently supports only square grids.

    Args:
        grid: grid object
        elevation: elevation, [m]

    Returns:
        dictionary of cell IDs and their upstream neighbours
    """

    if grid.grid_type != "square":
        to_raise = ValueError("This grid type is currently not supported!")
        LOGGER.error(to_raise)
        raise to_raise

    grid.set_neighbours(distance=sqrt(grid.cell_area))
    lowest_neighbours = find_lowest_neighbour(grid.neighbours, elevation)
    upstream_ids = find_upstream_cells(lowest_neighbours)

    return dict(enumerate(upstream_ids))


def estimate_interception(
    leaf_area_index: NDArray,
    precipitation: NDArray,
    intercept_param_1: float,
    intercept_param_2: float,
    intercept_param_3: float,
    veg_density_param: float,
) -> NDArray:
    r"""Estimate canopy interception.

    This function estimates canopy interception using the following storage-based
    equation after :cite:t:`aston_rainfall_1979` and :cite:t:`merriam_note_1960`:

    :math:`Int = S_{max} * [1 - e \frac{(-k*R*\delta t}{S_{max}})]`

    where :math:`Int` [mm] is the interception per time step, :math:`S_{max}` [mm] is
    the maximum interception, :math:`R` is the rainfall intensity per time step [mm] and
    the factor :math:`k` accounts for the density of the vegetation.

    :math:`S_{max}` is calculated using an empirical equation
    :cite:p:`von_hoyningen-huene_interzeption_1981`:

    :math:`S_{max} = 0.935 + 0.498 * LAI - 0.00575 * LAI^{2}` for [LAI > 0.1], and

    :math:`S_{max} = 0` for [LAI ≤ 0.1]

    where LAI is the average Leaf Area Index [m2 m-2]. :math:`k` is estimated as:

    :math:`k=0.046 * LAI`

    Args:
        leaf_area_index: leaf area index summed over all canopy layers, [m2 m-2]
        precipitation: precipitation, [mm]
        intercept_parameter_1: Parameter in equation that estimates maximum canopy
            interception capacity
        intercept_parameter_2: Parameter in equation that estimates maximum canopy
            interception capacity
        intercept_parameter_3: Parameter in equation that estimates maximum canopy
            interception capacity
        veg_density_param: Parameter used to estimate vegetation density for maximum
            canopy interception capacity estimate

    Returns:
        interception, [mm]
    """

    capacity = (
        intercept_param_1
        + intercept_param_2 * leaf_area_index
        - intercept_param_3 * leaf_area_index**2
    )
    max_capacity = np.where(leaf_area_index > 0.1, capacity, 0)

    canopy_density_factor = veg_density_param * leaf_area_index

    return np.nan_to_num(
        max_capacity
        * (1 - np.exp(-canopy_density_factor * precipitation / max_capacity)),
        nan=0.0,
    )


def update_soil_moisture(
    soil_moisture: NDArray,
    vertical_flow: NDArray,
    soil_moisture_capacity: NDArray,
    soil_moisture_residual: NDArray,
) -> NDArray:
    """Update soil moisture profile.

    This function calculates soil moisture for each layer by removing the vertical flow
    of the current layer and adding it to the layer below.

    Args:
        soil_moisture: soil moisture after infiltration and surface evaporation, [mm]
        vertical_flow: vertical flow between all layers, [mm]
        soil_moisture_capacity: soil moisture capacity for each layer, [mm]
        soil_moisture_residual: residual soil moisture for each layer, [mm]

    Returns:
        updated soil moisture profile, relative volumetric water content, dimensionless
    """
    # TODO this is currently not conserving water
    top_soil_moisture = np.clip(
        soil_moisture[0] - vertical_flow[0],
        soil_moisture_residual[0],
        soil_moisture_capacity[0],
    )

    lower_soil_moisture = [
        np.clip(
            (soil_moisture[i + 1] + vertical_flow[i] - vertical_flow[i + 1]),
            soil_moisture_residual[i + 1],
            soil_moisture_capacity[i + 1],
        )
        for sm, vf in zip(
            soil_moisture[1:],
            vertical_flow[1:],
        )
        for i in range(len(soil_moisture) - 1)
    ]

    # Combine all levels and convert to relative volumetric water content
    return np.concatenate(([top_soil_moisture], lower_soil_moisture))
