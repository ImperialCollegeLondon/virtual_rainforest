"""The `abiotic.radiation` module.

The radiation balance at the top of the canopy at a given location depends on

1. extra-terrestrial radiation (affected by the earth's orbit, date, and location),
2. terrestrial radiation (affected by atmospheric composition and clouds),
3. topography (elevation, slope and aspect),
4. surface albedo (vegetation type and fraction of vegetation/bare soil), and
5. emitted longwave radiation.

The preprocessing module takes extra-terrestrial radiation as an input and adjusts for
the effects of topography (slope and aspect). Here, the effects of atmospheric
filtering (elevation-dependent) and cloud cover are added to calculate photosynthetic
photon flux density (PPFD) at the top of the canopy which is a crucial input to the
plant module. The implementation is based on :cite:t:`Davis2017`.

Cloud cover and surface albedo also determine how much of the shortwave radiation that
reaches the top of the canopy is reflected and how much remains to be absorbed via
photosynthesis and re-emitted as longwave radiation by vegetation and forest floor.
At this stage, scattering and re-absorption of longwave radiation are not considered.

"""
# the following structural components are not implemented yet
# TODO include time dimension
# TODO logging, raise errors

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from virtual_rainforest.core.data import Data
from virtual_rainforest.core.logger import LOGGER
from virtual_rainforest.core.model import InitialisationError


# from some external source, could be core.constants or abiotic.config
@dataclass
class RadiationConstants:
    """Radiation constants dataclass."""

    CLOUDY_TRANSMISSIVITY: float = 0.25
    """Cloudy transmittivity :cite:p:`Linacre1968`"""
    TRANSMISSIVITY_COEFFICIENT: float = 0.50
    """Angular coefficient of transmittivity :cite:p:`Linacre1968`"""
    FLUX_TO_ENERGY: float = 2.04
    """From flux to energy conversion, umol J-1 :cite:p:`Meek1984`"""
    STEFAN_BOLTZMANN_CONSTANT: float = 5.67e-8
    """Stefan-Boltzmann constant W m-2 K-4"""
    SOIL_EMISSIVITY: float = 0.95
    """Soil emissivity, default for tropical rainforest"""
    CANOPY_EMISSIVITY: float = 0.95
    """Canopy emissivity, default for tropical rainforest"""
    BEER_REGRESSION: float = 2.67e-5
    """Parameter in equation for atmospheric transmissivity based on regression of
    Beer's radiation extinction function :cite:p:`Allen1996`"""
    CELSIUS_TO_KELVIN: float = 273.15
    """Factor to convert temperature in Celsius to absolute temperature in Kelvin"""


class Radiation:
    """Radiation balance dataclass.

    This class uses a :class:`~virtual_rainforest.core.data.Data` object to populate
    and store radiation-related attributes which serve as inputs to other modules.
    Elevation information from a digital elevation model is used for the topographic
    adjustment of incoming shortwave radiation. The gross radiation that reaches the top
    of the canopy is reduced as it penetrates through the canopy (the absorption by
    individual canopy layers is provided by the plants module). What remains is the net
    shortwave radiation at the surface (= forest floor), which is an input to the
    :class:`~virtual_rainforest.models.abiotic.Energy_balance` class. Top of canopy
    photosynthetic photon flux density is the key input for
    :mod:`~virtual_rainforest.models.plants` on which the photosythesis is based.
    Longwave radiation from individual canopy layers and longwave radiation from soil
    serve as inputs to the :class:`~virtual_rainforest.models.abiotic.Energy_balance`
    class.

    The dataclass populates the elevation attribute, further attributes are
    calculate in a __post_init__ functionality which calls a number of helper functions.

    Init method populates radiation balance attributes from data object.

    Creating an instance of this class expects a data object that contains the following
    variables:
    * elevation: elevation above sea level, [m]
    * shortwave_in: downward shortwave radiation, [J m-2]
    * canopy_temperature: canopy temperature of individual layers, [C]
    * surface_temperature: surface soil temperature, [C]
    * canopy_absorption: absorption by canopy, [J m-2]

    The data object can also optionally provide these variables, but will default to the
    values given below:
    * albedo_vis: visible light albedo, default = 0.03
    * albedo_shortwave: shortwave albedo, default = 0.17
    * sunshine_fraction: fraction of sunshine hours, between 0 (100% cloud cover)
        and 1 (cloud free sky), default = 1

    Args:
        data: A Virtual Rainforest Data object.
        const: A RadiationConstants instance.
    """

    def __init__(
        self, data: Data, const: RadiationConstants = RadiationConstants()
    ) -> None:

        # check that elevation is above sea level
        if np.any(data["elevation"] < 0):
            to_raise = InitialisationError(
                "Initial elevation contains at least one negative value!"
            )
            LOGGER.error(to_raise)
            raise to_raise

        # TODO - think about xarray/numpy array and ArrayLike typing

        # Set the default values if variables not provided in data
        if "sunshine_fraction" not in data:
            sunshine_fraction = np.array(1.0, dtype=np.float32)
        else:
            sunshine_fraction = data["sunshine_fraction"]

        if "albedo_vis" not in data:
            albedo_vis = np.array(0.03, dtype=np.float32)
        else:
            albedo_vis = data["albedo_vis"]

        if "albedo_shortwave" not in data:
            albedo_shortwave = np.array(0.17, dtype=np.float32)
        else:
            albedo_shortwave = data["albedo_shortwave"]

        # ** arraylike? **
        # ppfd and topofcanopy_radiation radiation could be calculated across all time
        # steps in the abiotic module __init__. Leaving here for now.
        self.ppfd: NDArray[np.float32] = calculate_ppfd(
            shortwave_in=data["shortwave_in"],
            elevation=data["elevation"],
            sunshine_fraction=sunshine_fraction,
            albedo_vis=albedo_vis,
            flux_to_energy=const.FLUX_TO_ENERGY,
        )
        """Top of canopy photosynthetic photon flux density, [mol m-2]"""

        self.topofcanopy_radiation: NDArray[
            np.float32
        ] = calculate_topofcanopy_radiation(
            shortwave_in=data["shortwave_in"],
            elevation=data["elevation"],
            sunshine_fraction=sunshine_fraction,
            albedo_shortwave=albedo_shortwave,
        )
        """Top of canopy downward shortwave radiation, [J m-2]"""

        self.longwave_radiation: NDArray[np.float32] = calculate_longwave_radiation(
            canopy_temperature=data["canopy_temperature"],
            surface_temperature=data["surface_temperature"],
            canopy_emissivity=const.CANOPY_EMISSIVITY,
            soil_emissivity=const.SOIL_EMISSIVITY,
            STEFAN_BOLTZMANN_CONSTANT=const.STEFAN_BOLTZMANN_CONSTANT,
            CELSIUS_TO_KELVIN=const.CELSIUS_TO_KELVIN,
        )
        """Longwave radiation from canopy layers and soil, [J m-2]"""

        self.netradiation_surface: NDArray[np.float32] = calculate_netradiation_surface(
            topofcanopy_radiation=self.topofcanopy_radiation,
            canopy_absorption=data["canopy_absorption"],
            longwave_radiation=self.longwave_radiation,
        )
        """Net shortwave radiation at the surface (= forest floor), [J m-2]"""


# helper functions
def calculate_atmospheric_transmissivity(
    elevation: NDArray[np.float32],
    sunshine_fraction: NDArray[np.float32] = np.array(1.0, dtype=np.float32),
    cloudy_transmissivity: float = RadiationConstants.CLOUDY_TRANSMISSIVITY,
    transmissivity_coefficient: float = RadiationConstants.TRANSMISSIVITY_COEFFICIENT,
    beer_regression: float = RadiationConstants.BEER_REGRESSION,
    # const: RadiationConstants = RadiationConstants() ## alternative approach
) -> NDArray[np.float32]:
    """Calculate atmospheric transmissivity (tau).

    Args:
        elevation: elevation above sea level, [m]
        sunshine_fraction: fraction of sunshine hours, between 0 (100% cloud cover)
            and 1 (cloud free sky), default = 1
        cloudy_transmissivity: cloudy transmittivity :cite:p:`Linacre1968`, default set
            in config
        transmissivity_coefficient: angular coefficient of transmittivity
            :cite:p:`Linacre1968`, default set in config
        beer_regression: parameter in equation for atmospheric transmissivity based on
            regression of Beer's radiation extinction function :cite:p:`Allen1996`,
            default set in config

    Returns:
        atmospheric transmissivity, unitless
    """

    # check sunshine fraction between 0 and 1
    if 0 > np.any(sunshine_fraction) > 1:
        to_raise = ValueError(
            "The fraction of sunshine hours needs to be between 0 and 1!"
        )
        LOGGER.critical(to_raise)
        raise to_raise

    # calculate transmissivity (tau), unitless
    tau_o = cloudy_transmissivity + transmissivity_coefficient * sunshine_fraction
    return tau_o * (1.0 + beer_regression * elevation)


def calculate_ppfd(
    shortwave_in: NDArray[np.float32],
    elevation: NDArray[np.float32],
    sunshine_fraction: NDArray[np.float32] = np.array(1.0, dtype=np.float32),
    albedo_vis: NDArray[np.float32] = np.array(0.03, dtype=np.float32),
    flux_to_energy: float = RadiationConstants.FLUX_TO_ENERGY,
) -> NDArray[np.float32]:
    """Calculate top of canopy photosynthetic photon flux density, [mol m-2].

    Args:
        shortwave_in: downward shortwave radiation, [J m-2]
        elevation: elevation above sea level, [m]
        sunshine_fraction: fraction of sunshine hours, between 0 (100% cloud cover)
            and 1 (cloud free sky), default = 1
        albedo_vis: visible light albedo, default = 0.03
        flux_to_energy: flux to energy conversion factor, [umol J-1],default from config

    Returns:
        photosynthetic photon flux density, [mol m-2]

    Reference: :cite:t:`Davis2017`
    """

    tau = calculate_atmospheric_transmissivity(elevation, sunshine_fraction)
    return (1.0e-6) * flux_to_energy * (1.0 - albedo_vis) * tau * shortwave_in


def calculate_topofcanopy_radiation(
    shortwave_in: NDArray[np.float32],
    elevation: NDArray[np.float32],
    sunshine_fraction: NDArray[np.float32] = np.array(1.0, dtype=np.float32),
    albedo_shortwave: NDArray[np.float32] = np.array(0.17, dtype=np.float32),
) -> NDArray[np.float32]:
    """Calculate top of canopy shortwave radiation, [J m-2].

    Args:
        shortwave_in: downward shortwave radiation, [J m-2]
        elevation: elevation above sea level, [m]
        sunshine_fraction: fraction of sunshine hours, between 0 (100% cloud cover)
            and 1 (cloud free sky), default = 1
        albedo_shortwave: shortwave albedo, default = 0.17

    Returns:
        top of canopy radiation shortwave radiation, [J m-2]
    """

    tau = calculate_atmospheric_transmissivity(elevation, sunshine_fraction)
    return (1.0 - albedo_shortwave) * tau * shortwave_in


def calculate_longwave_radiation(
    canopy_temperature: NDArray[np.float32],
    surface_temperature: NDArray[np.float32],
    canopy_emissivity: float = RadiationConstants.CANOPY_EMISSIVITY,
    soil_emissivity: float = RadiationConstants.SOIL_EMISSIVITY,
    STEFAN_BOLTZMANN_CONSTANT: float = RadiationConstants.STEFAN_BOLTZMANN_CONSTANT,
    CELSIUS_TO_KELVIN: float = RadiationConstants.CELSIUS_TO_KELVIN,
) -> NDArray[np.float32]:
    """Calculate longwave emission from canopy and forest floor, [J m-2].

    Args:
        canopy_temperature: canopy temperature of n layers, [C]; the array size is
            set to max number of layers (n_max) and filled with NaN where n < n_max
        surface_temperature: surface soil temperature, [C]
        canopy_emissivity: canopy emissivity, default set in config
        soil_emissivity: soil emissivity, default set in config
        STEFAN_BOLTZMANN_CONSTANT: Stefan-Boltzmann constant [W m-2 K-4]
        CELSIUS_TO_KELVIN: factor to convert temperature in Celsius to absolute
            temperature in Kelvin

    Returns:
        longwave radiation from n individual canopy layers and soil, [J m-2]
    """
    # longwave emission canopy
    longwave_canopy = (
        canopy_emissivity
        * STEFAN_BOLTZMANN_CONSTANT
        * (CELSIUS_TO_KELVIN + canopy_temperature) ** 4
    )

    # longwave emission surface
    longwave_soil = (
        soil_emissivity
        * STEFAN_BOLTZMANN_CONSTANT
        * (CELSIUS_TO_KELVIN + surface_temperature) ** 4
    )

    # return array of longwave radiation for all canopy layers and surface

    # temporary hack pending resolution of dicussion of NDArray/DataArray
    # - assuming that if these are not numpy arrays, then they are DataArrays
    #   which we then need to coerce to numpy arrays to use the nparray.reshape API
    if not isinstance(longwave_canopy, np.ndarray):
        longwave_canopy = longwave_canopy.to_numpy()
    if not isinstance(longwave_soil, np.ndarray):
        longwave_soil = longwave_soil.to_numpy()

    return np.append(
        longwave_canopy.transpose(),
        longwave_soil.reshape([1, len(surface_temperature)]),
        axis=0,
    ).transpose()


def calculate_netradiation_surface(
    topofcanopy_radiation: NDArray[np.float32],
    canopy_absorption: NDArray[np.float32],
    longwave_radiation: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Calculate net shortwave radiation at the surface, [J m-2].

    Args:
        topofcanopy_radiation: top of canopy radiation shortwave radiation, [J m-2]
        canopy_absorption: shortwave radiation absorbed by canopy layers, [J m-2]
        longwave_radiation: longwave radiation from canopy layers and soil, [J m-2]

    Returns:
        net shortwave radiation at the surface ( = forest floor), [J m-2]
    """
    return (
        topofcanopy_radiation
        - np.sum(canopy_absorption, axis=1)  # sum over all canopy layers
        - np.sum(longwave_radiation, axis=1)  # sum over all canopy layers and topsoil
    )
