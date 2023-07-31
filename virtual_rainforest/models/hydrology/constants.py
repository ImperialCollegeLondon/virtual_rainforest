"""The :mod:`~virtual_rainforest.models.hydrology.constants` module contains a set of
dataclasses containing and parameters required by the
:mod:`~virtual_rainforest.models.hydrology.hydrology_model`. These parameters are
constants in that they should not be changed during a particular simulation.

TODO Soil parameters vary strongly with soil type and will require literature search and
sensitivity analysis to produce meaningful results. The current default values are just
examples within reasonable bounds.
"""  # noqa: D205, D415

from dataclasses import dataclass


@dataclass(frozen=True)
class HydroConsts:
    """Dataclass to store all constants for the `hydrology` model."""

    soil_moisture_capacity: float = 0.9
    """Soil moisture capacity, also known as field capacity or water holding capacity,
    refers to the maximum amount of water that a soil can retain against the force of
    gravity after it has been saturated and excess water has drained away. The value is
    soil type specific, the unit here is relative water content (between 0 and 1).
    """

    soil_moisture_residual: float = 0.1
    """Residual soil moisture refers to the water that remains in the soil after
    prolonged drainage due to the force of gravity. It is the water that is tightly held
    by soil particles and is not easily available for plant roots to extract. The value
    is soil specific, the unit here is relative water content (between 0 and 1).
    """

    water_interception_factor: float = 0.1
    """Water interception factor describes the proportion of rainfall that is
    intercepted by a canopy her unit leaf area.
    """

    hydraulic_conductivity: float = 0.001
    """Hydraulic conductivity (m/s) is the measure of a soil's ability to transmit water
    through its pores. More specifically, is defined as the volumetric flow rate of
    water passing through a unit cross-sectional area of soil under a unit hydraulic
    gradient (pressure difference).
    """

    hydraulic_gradient: float = 0.01
    """The hydraulic gradient (m/m) is a measure of the change in hydraulic head
    (pressure) per unit of distance in a particular direction within a fluid or porous
    medium, such as soil or an aquifer. It represents the driving force behind the
    movement of water and indicates the direction in which water will flow.
    """

    seconds_to_month: float = 2.628e6
    """Factor to convert variable unit from seconds to month."""

    nonlinearily_parameter: float = 2.0
    """Nonlinearity parameter n (dimensionless) in van Genuchten model for hydraulic
    conductivity :cite:p:`van_genuchten_describing_1985`."""

    meters_to_millimeters: float = 1000
    """Factor to convert variable unit from meters to millimeters."""

    celsius_to_kelvin: float = 273.15
    """Factor to convert variable unit from Celsius to Kelvin."""

    density_air: float = 1.225
    """Density of air under standard atmosphere, kg m-3"""

    latent_heat_vapourisation: float = 2.45
    """Latent heat of vapourisation under standard atmosphere, MJ kg-1"""

    gas_constant_water_vapour: float = 461.51
    """Gas constant for water vapour, J kg-1 K-1"""

    heat_transfer_coefficient: float = 12.5
    """Heat transfer coefficient, :cite:p:`van_de_griend_bare_1994` """

    flux_to_mm_conversion: float = 3.35e-4
    """Factor to convert evaporative flux to mm."""

    stream_flow_capacity: float = 5000.0
    """Stream flow capacity, mm per timestep. This is curretly an arbitrary value, but
    could be used in the future to flag flood events."""