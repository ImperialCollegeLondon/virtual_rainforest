"""The `abiotic.energy_balance` module.

This module calculates the energy balance for the Virtual Rainforest.
The sequence of processes is based on Maclean et al, 2021: Microclimc: A mechanistic
model of above, below and within-canopy microclimate. Ecological Modelling
Volume 451, 109567. https://doi.org/10.1016/j.ecolmodel.2021.109567.
"""

from math import exp as exp

import numpy as np
from numpy.typing import NDArray

# from core.constants import CONSTANTS as C
CELCIUS_TO_KELVIN = 273.15  # calculate absolute temperature in Kelvin
MOLAR_GAS_CONCENTRATION = 44.6  # pure ideal gas at 0 °C and 101.325 kPa
VAPOR_PRESSURE_FACTOR1 = 0.6108 # constant in calculation of vapor pressure
VAPOR_PRESSURE_FACTOR2 = 17.27 # constant in calculation of vapor pressure
VAPOR_PRESSURE_FACTOR3 = 237.7 # constant in calculation of vapor pressure

from scipy.constants import physical_constants

# import data
data = {"air_temperature_2m": 25,
        "relative_humidity_2m": 90,
        "atmospheric_pressure_2m": 101.325,
        "wind_speed_2m": 10.0,
        "mean_annual_temperature": 25
}

class EnergyBalance:
    """EnergyBalance method."""

    def __init__(self) -> None:
        """Initializes point-based energy_balance method."""
        raise NotImplementedError("Implementation of this feature is still missing")

    def calc_energy_balance(self, args: None, kwargs: None) -> None:
        """Calculates energy balance under steady state.

        Args:

        Returns:
            air_temperature
            canopy_temperature
            soil_temperature
            vapor_pressure_deficit

        """

    # --- this is the sequence of processes in microclimc for one time step ---
    # paraminit
    # soilinit
    # runonetimestep:
    # Check whether any vegetation layers have zero PAI
    # == Unpack climate variables ==
    # == calculate baseline variables: ==
    # - molar density of air
    # - specific heat of air
    # - latent heat of vaporisation
    # Adjust wind to 2 m above canopy
    # Generate heights of nodes
    # Set z above
    # == Calculate diabatic correction factors ==
    # Calculate temperatures and relative humidities for top of canopy
    # Set limits to temp can
    # Adjust relative humidity
    # == Calculate wind speed and turbulent conductances ==
    # == Calculate canopy turbulences ==
    # == Calculate conductivity to top of canopy and merge ==
    # Turbulent air conductivity and layer merge
    # == Calculate absorbed radiation
    # == Conductivities ==
    # Vapour conductivity
    # Leaf conductivity
    # == Soil conductivity ==
    # conductivity and specific heat
    # soil heat
    # Canopy air layer not in equilibrium with above canopy:
    # Heat to add / loose
    # vapour exchange
    # Interpolate
    # Canopy air layer in equilibrium with above canopy
    # Set limits to soil temperatures
    # Calculate Heat flux
    # Latent heat
    # dewpoints
    # Incoming radiation
    # internal function to sort out vegetation parameters

    def thomas(
        self,
        air_temperature_0: NDArray[np.float32],
        soil_temperature_0: NDArray[np.float32],
        mean_annual_temperature: NDArray[np.float32],
        air_temperature_2m: NDArray[np.float32],
        thermal_conductivity: NDArray[np.float32],
        thermal_heat_capacity: NDArray[np.float32],
        added_temperatures: NDArray[np.float32] = np.ndarray(0.0, type=float),
        weighting_factor: float = 0.6,
    ) -> NDArray[np.float32]:
        """Thomas algorithm for solving simultanious heat fluxes between soil/air.

        Args:
            air_temperature_0: NDArray[np.float32], air temperatures from previous
                timestep [C]
            soil_temperature_0: NDArray[np.float32], soil temperatures from previous
                timestep [C]
            mean_annual_temperature: NDArray[np.float32], mean annual temperature,
                here used as temperature of deepest soil layer [C]
            air_temperature_2m: NDArray[np.float32], air temperature at reference
                height 2 m above canopy in current time step [C]
            thermal_conductivity: NDArray[np.float32], thermal conductances between
                layers [W m-2 K-1], see details
            thermal_heat_capacity: NDArray[np.float32], thermal heat capacity of
                layers [J m-3 K-1]
            added_temperatures: NDArray[np.float32], temperatures to be added
                resulting from e.g. leaf heat fluxes or radiation absorbed by top soil
            weighting_factor:float = 0.6, forward / backward weighting of algorithm, see
                details

        Returns:
            air_temperature: NDArray[np.float32], air temperatures from current
                timestep [C]
            soil_temperature: NDArray[np.float32], soil temperatures from current
                timestep [C]

        Details:
            The vector `previous_temperatures` that combines air and soil temperatures
            must be ordered with reference air temperature first and the soil
            temperature of the deepest layer (= mean annual temperature) last. I.e. the
            length of the vector is the number of nodes + 2.
            The vector `thermal_conductivity` is the conductivity between each node and
            that dirctly below it, the first value representing conductivity between
            reference height and the top canopy node. I.e. the length of the vector is
            the number of nodes + 1.
            The vector `thermal_heat_capacity` is the heat storage at each node. I.e.
            the length of the vector is the same as the number of nodes.
            The `weighting_factor` may range from 0 to 1. If `weighting_factor` = 0, the
            flux is determined by the temperature difference at the beginning of the
            time step. If `weighting_factor` = 0.5, the average of the old and new
            temperatures is used to compute heat flux. If `weighting_factor` = 1, fluxes
            are computed using only the new temperatures. The best value to use for
            `weighting_factor` is determined by considerations of numerical stability
            and accuracy and experimentation may be required. If `weighting_factor` = 0
            more heat transfer between nodes is predicted than would actually occur, and
            can therefore become unstable if time steps are too large. When
            `weighting_factor` > 0.5, stable solutions will always be obtained, but heat
            flux will be underestimated. The best accuracy is obtained with
            `weighting_factor`around 0.4, best stability is at `weighting_factor`= 1.
            A typical compromise is `weighting_factor` = 0.6.

            m = n_layers ; check difference in starting from 0 or 1
        """
        # combine temperatures from previous timestep in one vector
        previous_temperatures = [
            air_temperature_2m,
            air_temperature_0,
            soil_temperature_0,
            mean_annual_temperature,
        ]

        # n_layers includes all soil and air layers
        n_layers = len(previous_temperatures) - 2

        # create empty array for current temperatures
        air_temperature = np.array(n_layers + 2, type=float)

        # assign first and last value, indices from maclean R code, needs changing!!
        air_temperature[n_layers + 2] = mean_annual_temperature
        air_temperature[1] = air_temperature_2m

        # set `matrix zeros`
        a = [0, 0]
        b = 0
        cc = 0
        d = 0

        xx = [x for x in range(2, (n_layers + 1))]  ### as a list

        previous_temperatures[xx] = (
            previous_temperatures[xx] + (1 - weighting_factor) * added_temperatures
        )
        cc[xx] = - thermal_conductivity[xx] * weighting_factor
        a[xx + 1] = cc[xx]
        b[xx] = (
            weighting_factor * (thermal_conductivity[xx] + thermal_conductivity[xx - 1])
            + thermal_heat_capacity
        )
        d[xx] = (
            (1 - weighting_factor)
            * thermal_conductivity[xx - 1]
            * previous_temperatures[xx - 1]
            + (
                thermal_heat_capacity
                - (1 - weighting_factor)
                * (thermal_conductivity[xx] + thermal_conductivity[xx - 1])
            )
            * previous_temperatures[xx]
            + (1 - weighting_factor)
            * thermal_conductivity[xx]
            * previous_temperatures[xx + 1]
        )
        d[2] = d[2] + thermal_conductivity[1] * air_temperature[1] * weighting_factor
        d[n_layers + 1] = (
            d[n_layers + 1]
            + thermal_conductivity[n_layers + 1]
            * weighting_factor
            * air_temperature[n_layers + 2]
        )

        for i in range(2, n_layers):
            cc[i] = cc[i] / b[i]
            d[i] = d[i] / b[i]
            b[i + 1] = b[i + 1] - a[i + 1] * cc[i]
            d[i + 1] = d[i + 1] - a[i + 1] * d[i]

        air_temperature[n_layers + 1] = d[n_layers + 1] / b[n_layers + 1]

        for i in range(
            2, n_layers
        ):  ### double-check, looks like error in original code
            air_temperature[i] = d[i] - cc[i] * air_temperature[i + 1]

        x_min = pmin(
            air_temperature[xx],
            air_temperature[xx - 1],
            air_temperature[xx + 1],
        )
        x_max = pmax(
            air_temperature[xx],
            air_temperature[xx - 1],
            air_temperature[xx + 1],
        )
        air_temperature[xx] = ifelse(
            air_temperature[xx] < x_min, x_min, air_temperature[xx]
        )
        air_temperature[xx] = ifelse(
            air_temperature[xx] > x_max, x_max, air_temperature[xx]
        )
        air_temperature[xx] = (
            air_temperature[xx] + weighting_factor * added_temperatures
        )

        return air_temperature

    def thomasV(self, air_vapour_concentration_0: NDArray[np.float32],
            air_temperature_0: NDArray[np.float32],
            relative_humidity_0: NDArray[np.float32],
            top_soil_moisture_0: NDArray[np.floa32],
            soil_temperature_0: NDArray[np.float32],
            atmospheric_pressure_0: NDArray[np.float32],
            air_temperature_2m: NDArray[np.float32],
            relative_humidity_2m: NDArray[np.float32],
            atmospheric_pressure: NDArray[np.float32],
            air_temperature: NDArray[np.float32],
            soil_temperature: NDArray[np.float32],
            top_soil_moisture: NDArray[np.floa32],
            vapour_flux: NDArray[np.float32],
            node_height_differences: NDArray[np.float32],
            canopy_node_conductance: NDArray[np.float32],
            weighting_factor:float = 0.6,
            soilp = None
) -> NDArray[np.foat32]:
        """Thomas algorithm for solving simultanious vapour fluxes between air layers.

        Args:
            air_vapour_concentration_0: NDArray[np.float32],air vapour concentrations for
                each canopy node in the previous timestep (mol fraction)
            air_temperature_0: NDArray[np.float32], air temperature from previous time
                step [C]
            realtive_humidity_0: NDArray[np.float32], relative humidity from previous
                time step [percentage]
            top_soil_moisture_0: NDArray[np.floa32], Volumetric water content of the
                upper most soil layer in the previous time step [m3 m-3]
            soil_temperature_0: NDArray[np.float32], soil temperatures from previous
                timestep [C]
            atmospheric_pressure_0: NDArray[np.float32], atmospheric pressure previous
                time step [kPa]
            air_temperature_2m: NDArray[np.float32], air temperature at reference
                height 2 m above canopy in current time step [C]
            relative_humidity_2m: NDArray[np.float32], relative humidity (percentage) at
                reference height 2 m above canopy in current time step (percentage)
            atmospheric_pressure: NDArray[np.float32], atmospheric pressure current time
                step [kPa]
            air_temperature: NDArray[np.float32],air temperatures for each canopy
                node in the current timestep [C]
            soil_temperature: NDArray[np.float32], soil temperatures from current
                time step [C]
            top_soil_moisture: NDArray[np.floa32], Volumetric water content of
                the upper most soil layer in the current time step [m3 m-3]
            vapour_flux: NDArray[np.float32], Total vapour flux from leaves to air
                [mol m-3]
            node_height_differences: NDArray[np.float32], height difference between each
                canopy node and that directly below it. The first value is the height
                difference between the lowest canopy node and the ground.
            canopy_node_conductance: NDArray[np.float32], molar conductances between each
                canopy node at that directly below it [mol m-2 sec-1]. The first value
                is the conductivity between the ground and the lowest node, and the last
                value the conductivity between the highest node and reference height.
            weighting_factor:float = 0.6, forward / backward weighting of algorithm, see
                details (as for [thomas()])
            soilp = None, a list of soil parameters as returned by [soilinit()]

        Returns:
            air_vapour_concentration: NDArray[np.float32], vapour concentrations
                for each canopy node in the current time step [mole fractions].
                The first value is that for the ground and the last value that at
                reference height.
        """
        n_layers = len(node_height_differences)
 
        # Calculate molar density of air
        molar_density_air = (
        MOLAR_GAS_CONCENTRATION
        * (atmospheric_pressure/physical_constants['standard atmosphere'])
        * (CELCIUS_TO_KELVIN / (air_temperature_2m + CELCIUS_TO_KELVIN))
        )

        # Calculate actual vapor pressure for current time step
        actual_vapor_pressure = (
        VAPOR_PRESSURE_FACTOR1 
        * exp(VAPOR_PRESSURE_FACTOR2 
        * (air_temperature_2m / (air_temperature_2m + VAPOR_PRESSURE_FACTOR3))) 
        * (relative_humidity_2m / 100)
        )

        # Calculate actual vapor pressure for previous timestep
        actual_vapor_pressure_0 = (
            VAPOR_PRESSURE_FACTOR1
        * exp(VAPOR_PRESSURE_FACTOR2
        * air_temperature_0 / (air_temperature_0 + VAPOR_PRESSURE_FACTOR3))
        * (relative_humidity_0/100)
        )
        
        # Calculate air vapor concentration?
        Vair = actual_vapor_pressure / atmospheric_pressure

        # calculate soil relative humidity, function in microclimctools
        relative_humidity_soil = soilrh(
            top_soil_moisture,
            soilp$b,
            soilp$psi_e,
            soilp$Smax,
            soil_temperature
            )

        relative_humidity_soil_0 = soilrh(
            top_soil_moisture_0,
            soilp$b,
            soilp$psi_e,
            soilp$Smax,
            soil_temperature_0[1]
            )

        relative_humidity_soil[relative_humidity_soil>1] = 1
        relative_humidity_soil_0[relative_humidity_soil_0>1] = 1

        # calculate current soil vapor pressure ??
        actual_vapor_pressure_soil = (
            VAPOR_PRESSURE_FACTOR1
            * exp(VAPOR_PRESSURE_FACTOR2
            * soil_temperature/(soil_temperature+VAPOR_PRESSURE_FACTOR3))
            * relative_humidity_soil
        )

        # calculate previous timestep soil vapor pressure ??
        actual_vapor_pressure_soil_0 = (
            VAPOR_PRESSURE_FACTOR1
            * exp(VAPOR_PRESSURE_FACTOR2
            * soil_temperature_0/(soil_temperature_0+VAPOR_PRESSURE_FACTOR3))
            * relative_humidity_soil_0
        )

        Vsoil = actual_vapor_pressure_soil/atmospheric_pressure
        Vo = [actual_vapor_pressure_soil_0/atmospheric_pressure_0, Vo, actual_vapor_pressure_0/atmospheric_pressure_0]

        Vn = Thomas(
            rev(Vo),
            Vsoil,
            Vair,
            rev(canopy_node_conductance),
            rev(molar_density_air),
            weighting_factor,
            vapor_flux)

        Vn = rev(Vn)
}