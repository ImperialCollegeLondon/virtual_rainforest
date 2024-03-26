"""Collection of fixtures to assist the testing scripts."""

from logging import DEBUG
from typing import Any

import numpy as np
import pytest
import xarray as xr
from xarray import DataArray

# An import of LOGGER is required for INFO logging events to be visible to tests
# This can be removed as soon as a script that imports logger is imported
from virtual_ecosystem.core.logger import LOGGER

# Class uses DEBUG
LOGGER.setLevel(DEBUG)


def log_check(
    caplog: pytest.LogCaptureFixture,
    expected_log: tuple[tuple],
    subset: slice | None = None,
) -> None:
    """Helper function to check that the captured log is as expected.

    Arguments:
        caplog: An instance of the caplog fixture
        expected_log: An iterable of 2-tuples containing the log level and message.
        subset: Only check a specified subset of the captured log.
    """

    # caplog.records is just a list of LogRecord objects, so can use a slice to drop
    # down to a subset of the records.
    if subset is None:
        captured_records = caplog.records
    else:
        captured_records = caplog.records[subset]

    assert len(expected_log) == len(captured_records)

    assert all(
        [exp[0] == rec.levelno for exp, rec in zip(expected_log, captured_records)]
    )
    assert all(
        [exp[1] in rec.message for exp, rec in zip(expected_log, captured_records)]
    )


def record_found_in_log(
    caplog: pytest.LogCaptureFixture,
    find: tuple[int, str],
) -> bool:
    """Helper function to look for a specific logging record in the captured log.

    Arguments:
        caplog: An instance of the caplog fixture
        find: A tuple giving the logging level and message to look for
    """

    try:
        # Iterate over the record tuples, ignoring the leading element
        # giving the logger name
        _ = next(msg for msg in caplog.record_tuples if msg[1:] == find)
        return True
    except StopIteration:
        return False


@pytest.fixture(autouse=True)
def reset_module_registry():
    """Reset the module registry.

    The register_module function updates the MODULE_REGISTRY, which persists between
    tests. This autouse fixture is used to ensure that the registry is always cleared
    before tests start, so that the correct registration of modules within tests is
    enforced.
    """
    from virtual_ecosystem.core.registry import MODULE_REGISTRY

    MODULE_REGISTRY.clear()


# Shared fixtures


@pytest.fixture
def fixture_square_grid():
    """Create a square grid fixture.

    A 10 x 10 grid of 1 hectare cells, with non-zero origin.
    """

    from virtual_ecosystem.core.grid import Grid

    grid = Grid(
        grid_type="square",
        cell_area=10000,
        cell_nx=10,
        cell_ny=10,
        xoff=500000,
        yoff=200000,
    )

    return grid


@pytest.fixture
def fixture_square_grid_simple():
    """Create a square grid fixture.

    A 2 x 2 grid centred on x=1,1,2,2 y=1,2,1,2
    """

    from virtual_ecosystem.core.grid import Grid

    grid = Grid(
        grid_type="square",
        cell_area=1,
        cell_nx=2,
        cell_ny=2,
        xoff=0.5,
        yoff=0.5,
    )

    return grid


@pytest.fixture
def fixture_data(fixture_square_grid_simple):
    """A Data instance fixture for use in testing."""

    from virtual_ecosystem.core.data import Data

    data = Data(fixture_square_grid_simple)

    # Create an existing variable to test replacement
    data["existing_var"] = DataArray([1, 2, 3, 4], dims=("cell_id",))

    return data


@pytest.fixture
def data_instance():
    """Creates an empty data instance."""
    from virtual_ecosystem.core.data import Data
    from virtual_ecosystem.core.grid import Grid

    grid = Grid()
    return Data(grid)


@pytest.fixture
def fixture_config():
    """Simple configuration fixture for use in tests."""

    from virtual_ecosystem.core.config import Config

    cfg_string = """
        [core]
        [core.grid]
        cell_nx = 10
        cell_ny = 10
        [core.timing]
        start_date = "2020-01-01"
        update_interval = "2 weeks"
        run_length = "50 years"
        [core.data_output_options]
        save_initial_state = true
        save_final_state = true
        out_initial_file_name = "model_at_start.nc"
        out_final_file_name = "model_at_end.nc"

        [core.layers]
        canopy_layers = 10
        soil_layers = [-0.25, -1.0]
        above_canopy_height_offset = 2.0
        surface_layer_height = 0.1
        subcanopy_layer_height = 1.5

        [plants]
        a_plant_integer = 12
        [[plants.ftypes]]
        pft_name = "shrub"
        max_height = 1.0
        [[plants.ftypes]]
        pft_name = "broadleaf"
        max_height = 50.0

        [[animals.functional_groups]]
        name = "carnivorous_bird"
        taxa = "bird"
        diet = "carnivore"
        metabolic_type = "endothermic"
        birth_mass = 0.1
        adult_mass = 1.0
        [[animals.functional_groups]]
        name = "herbivorous_bird"
        taxa = "bird"
        diet = "herbivore"
        metabolic_type = "endothermic"
        birth_mass = 0.05
        adult_mass = 0.5
        [[animals.functional_groups]]
        name = "carnivorous_mammal"
        taxa = "mammal"
        diet = "carnivore"
        metabolic_type = "endothermic"
        birth_mass = 4.0
        adult_mass = 40.0
        [[animals.functional_groups]]
        name = "herbivorous_mammal"
        taxa = "mammal"
        diet = "herbivore"
        metabolic_type = "endothermic"
        birth_mass = 1.0
        adult_mass = 10.0
        [[animals.functional_groups]]
        name = "carnivorous_insect"
        taxa = "insect"
        diet = "carnivore"
        metabolic_type = "ectothermic"
        birth_mass = 0.001
        adult_mass = 0.01
        [[animals.functional_groups]]
        name = "herbivorous_insect"
        taxa = "insect"
        diet = "herbivore"
        metabolic_type = "ectothermic"
        birth_mass = 0.0005
        adult_mass = 0.005
        """

    return Config(cfg_strings=cfg_string)


@pytest.fixture
def fixture_core_components(fixture_config):
    """A CoreComponents instance for use in testing."""
    from virtual_ecosystem.core.core_components import CoreComponents

    return CoreComponents(fixture_config)


@pytest.fixture
def dummy_carbon_data(fixture_core_components):
    """Creates a dummy carbon data object for use in tests."""

    from virtual_ecosystem.core.data import Data
    from virtual_ecosystem.core.grid import Grid

    # Setup the data object with four cells.
    grid = Grid(cell_nx=4, cell_ny=1)
    data = Data(grid)

    # The required data is now added. This includes the four carbon pools: mineral
    # associated organic matter, low molecular weight carbon, microbial carbon and
    # particulate organic matter. It also includes various factors of the physical
    # environment: pH, bulk density, soil moisture, soil temperature, percentage clay in
    # soil.
    data["soil_c_pool_lmwc"] = DataArray([0.05, 0.02, 0.1, 0.005], dims=["cell_id"])
    """Low molecular weight carbon pool (kg C m^-3)"""
    data["soil_c_pool_maom"] = DataArray([2.5, 1.7, 4.5, 0.5], dims=["cell_id"])
    """Mineral associated organic matter pool (kg C m^-3)"""
    data["soil_c_pool_microbe"] = DataArray([5.8, 2.3, 11.3, 1.0], dims=["cell_id"])
    """Microbial biomass (carbon) pool (kg C m^-3)"""
    data["soil_c_pool_pom"] = DataArray([0.1, 1.0, 0.7, 0.35], dims=["cell_id"])
    """Particulate organic matter pool (kg C m^-3)"""
    data["soil_enzyme_pom"] = DataArray(
        [0.022679, 0.009576, 0.050051, 0.003010], dims=["cell_id"]
    )
    """Soil enzyme that breaks down particulate organic matter (kg C m^-3)"""
    data["soil_enzyme_maom"] = DataArray(
        [0.0356, 0.0117, 0.02509, 0.00456], dims=["cell_id"]
    )
    """Soil enzyme that breaks down mineral associated organic matter (kg C m^-3)"""
    data["pH"] = DataArray([3.0, 7.5, 9.0, 5.7], dims=["cell_id"])
    data["bulk_density"] = DataArray([1350.0, 1800.0, 1000.0, 1500.0], dims=["cell_id"])
    data["clay_fraction"] = DataArray([0.8, 0.3, 0.1, 0.9], dims=["cell_id"])
    data["litter_C_mineralisation_rate"] = DataArray(
        [0.00212106, 0.00106053, 0.00049000, 0.0055], dims=["cell_id"]
    )
    # Data for average vertical flow
    data["vertical_flow"] = DataArray([0.1, 0.5, 2.5, 1.59], dims=["cell_id"])

    # The layer dependant data has to be handled separately
    data["soil_moisture"] = xr.concat(
        [
            DataArray(np.full((13, 4), np.nan), dims=["layers", "cell_id"]),
            # At present the soil model only uses the top soil layer, so this is the
            # only one with real test values in
            DataArray(
                [[0.9304620050, 0.787549327, 0.504263188, 0.302527807]],
                dims=["layers", "cell_id"],
            ),
            DataArray(np.full((1, 4), np.nan), dims=["layers", "cell_id"]),
        ],
        dim="layers",
    )
    data["soil_moisture"] = data["soil_moisture"].assign_coords(
        {
            "layers": np.arange(0, 15),
            "layer_roles": (
                "layers",
                fixture_core_components.layer_structure.layer_roles,
            ),
            "cell_id": data.grid.cell_id,
        }
    )
    data["matric_potential"] = xr.concat(
        [
            DataArray(np.full((13, 4), np.nan), dims=["layers", "cell_id"]),
            # At present the soil model only uses the top soil layer, so this is the
            # only one with real test values in
            DataArray([[-3.0, -10.0, -250.0, -10000.0]], dims=["layers", "cell_id"]),
            DataArray(np.full((1, 4), np.nan), dims=["layers", "cell_id"]),
        ],
        dim="layers",
    ).assign_coords(
        {
            "layers": np.arange(0, 15),
            "layer_roles": (
                "layers",
                fixture_core_components.layer_structure.layer_roles,
            ),
            "cell_id": data.grid.cell_id,
        }
    )
    data["soil_temperature"] = xr.concat(
        [
            DataArray(np.full((13, 4), np.nan), dims=["dim_0", "cell_id"]),
            # At present the soil model only uses the top soil layer, so this is the
            # only one with real test values in
            DataArray([[35.0, 37.5, 40.0, 25.0]], dims=["dim_0", "cell_id"]),
            DataArray(np.full((1, 4), 22.5), dims=["dim_0", "cell_id"]),
        ],
        dim="dim_0",
    )
    data["soil_temperature"] = (
        data["soil_temperature"]
        .rename({"dim_0": "layers"})
        .assign_coords(
            {
                "layers": np.arange(0, 15),
                "layer_roles": (
                    "layers",
                    fixture_core_components.layer_structure.layer_roles,
                ),
                "cell_id": data.grid.cell_id,
            }
        )
    )

    return data


@pytest.fixture
def top_soil_layer_index(fixture_core_components):
    """The index of the top soil layer in the data fixtures."""
    return fixture_core_components.layer_structure.layer_roles.index("soil")


@pytest.fixture
def surface_layer_index(fixture_core_components):
    """The index of the top soil layer in the data fixtures."""
    return fixture_core_components.layer_structure.layer_roles.index("surface")


@pytest.fixture
def new_axis_validators():
    """Create new axis validators to test methods and registration."""
    from virtual_ecosystem.core.axes import AxisValidator
    from virtual_ecosystem.core.grid import Grid

    # Create a new subclass.
    class TestAxis(AxisValidator):
        core_axis = "testing"
        dim_names = {"test"}

        def can_validate(self, value: DataArray, grid: Grid, **kwargs: Any) -> bool:
            return True if value.sum() > 10 else False

        def run_validation(
            self, value: DataArray, grid: Grid, **kwargs: Any
        ) -> DataArray:
            return value * 2

    # Create a new duplicate subclass to check mutual exclusivity test
    class TestAxis2(AxisValidator):
        core_axis = "testing"
        dim_names = {"test"}

        def can_validate(self, value: DataArray, grid: Grid, **kwargs: Any) -> bool:
            return True if value.sum() > 10 else False

        def run_validation(
            self, value: DataArray, grid: Grid, **kwargs: Any
        ) -> DataArray:
            return value * 2


@pytest.fixture
def dummy_climate_data(fixture_core_components):
    """Creates a dummy climate data object for use in tests."""

    from virtual_ecosystem.core.data import Data
    from virtual_ecosystem.core.grid import Grid

    # Setup the data object with four cells.
    grid = Grid(
        grid_type="square",
        cell_nx=3,
        cell_ny=1,
        cell_area=3,
        xoff=0,
        yoff=0,
    )
    data = Data(grid)

    # Reference data
    data["air_temperature_ref"] = DataArray(
        np.full((3, 3), 30.0),
        dims=["cell_id", "time_index"],
    )
    data["wind_speed_ref"] = DataArray(
        np.full((3, 3), 1.0),
        dims=["time_index", "cell_id"],
    )
    data["mean_annual_temperature"] = DataArray(
        np.full((3), 20.0),
        dims=["cell_id"],
    )
    data["relative_humidity_ref"] = DataArray(
        np.full((3, 3), 90.0),
        dims=["cell_id", "time_index"],
    )
    data["vapour_pressure_deficit_ref"] = DataArray(
        np.full((3, 3), 0.14),
        dims=["cell_id", "time_index"],
    )
    data["vapour_pressure_ref"] = DataArray(
        np.full((3, 3), 0.14),
        dims=["cell_id", "time_index"],
    )
    data["atmospheric_pressure_ref"] = DataArray(
        np.full((3, 3), 96.0),
        dims=["cell_id", "time_index"],
    )
    data["atmospheric_co2_ref"] = DataArray(
        np.full((3, 3), 400.0),
        dims=["cell_id", "time_index"],
    )
    data["precipitation"] = DataArray(
        np.full((3, 3), 200.0),
        dims=["time_index", "cell_id"],
    )

    data["elevation"] = DataArray([200, 100, 10], dims="cell_id")
    data["topofcanopy_radiation"] = DataArray(
        np.full((3, 3), 100.0), dims=["cell_id", "time_index"]
    )

    # Simulation data
    full_coordinates = {
        "layers": np.arange(15),
        "layer_roles": ("layers", fixture_core_components.layer_structure.layer_roles),
        "cell_id": data.grid.cell_id,
    }

    # Structural variables
    leaf_area_index = np.repeat(a=[np.nan, 1.0, np.nan], repeats=[1, 3, 11])
    data["leaf_area_index"] = DataArray(
        np.broadcast_to(leaf_area_index, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="leaf_area_index",
    )

    layer_heights = np.repeat(
        a=[32.0, 30.0, 20.0, 10.0, np.nan, 1.5, 0.1, -0.5, -1.0],
        repeats=[1, 1, 1, 1, 7, 1, 1, 1, 1],
    )
    data["layer_heights"] = DataArray(
        np.broadcast_to(layer_heights, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="layer_heights",
    )
    data["canopy_height"] = DataArray([32, 32, 32], dims=["cell_id"])

    # Microclimate and energy balance
    wind_speed = np.repeat(a=[0.1, np.nan, 0.1, np.nan], repeats=[4, 7, 2, 2])
    data["wind_speed"] = DataArray(
        np.broadcast_to(wind_speed, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="wind_speed",
    )
    pressure = np.repeat(a=[96.0, np.nan, 96.0, np.nan], repeats=[4, 7, 2, 2])
    data["atmospheric_pressure"] = DataArray(
        np.broadcast_to(pressure, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="atmospheric_pressure",
    )

    data["air_temperature"] = xr.concat(
        [
            DataArray(
                [
                    [30.0, 30.0, 30.0],
                    [29.844995, 29.844995, 29.844995],
                    [28.87117, 28.87117, 28.87117],
                    [27.206405, 27.206405, 27.206405],
                ],
                dims=["layers", "cell_id"],
            ),
            DataArray(np.full((7, 3), np.nan), dims=["layers", "cell_id"]),
            DataArray(
                [
                    [22.65, 22.65, 22.65],
                    [16.145945, 16.145945, 16.145945],
                ],
                dims=["layers", "cell_id"],
            ),
            DataArray(np.full((2, 3), np.nan), dims=["layers", "cell_id"]),
        ],
        dim="layers",
    ).assign_coords(full_coordinates)

    data["soil_temperature"] = xr.concat(
        [DataArray(np.full((13, 3), np.nan)), DataArray(np.full((2, 3), 20))],
        dim="dim_0",
    )

    data["soil_temperature"] = (
        data["soil_temperature"]
        .rename({"dim_0": "layers", "dim_1": "cell_id"})
        .assign_coords(full_coordinates)
    )

    data["relative_humidity"] = xr.concat(
        [
            DataArray(
                [
                    [90.0, 90.0, 90.0],
                    [90.341644, 90.341644, 90.341644],
                    [92.488034, 92.488034, 92.488034],
                    [96.157312, 96.157312, 96.157312],
                ],
                dims=["layers", "cell_id"],
            ),
            DataArray(np.full((7, 3), np.nan), dims=["layers", "cell_id"]),
            DataArray(
                [
                    [100, 100, 100],
                    [100, 100, 100],
                ],
                dims=["layers", "cell_id"],
            ),
            DataArray(np.full((2, 3), np.nan), dims=["layers", "cell_id"]),
        ],
        dim="layers",
    ).assign_coords(full_coordinates)

    data["shortwave_radiation_surface"] = DataArray(
        np.array([100, 10, 0]), dims="cell_id"
    )

    absorbed = np.repeat(a=[np.nan, 10.0, np.nan], repeats=[1, 3, 11])
    data["absorbed_radiation"] = DataArray(
        np.broadcast_to(absorbed, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="absorbed_radiation",
        coords=full_coordinates,
    )
    data["sensible_heat_flux_topofcanopy"] = DataArray([100, 50, 10], dims=["cell_id"])
    data["sensible_heat_flux_soil"] = DataArray([1, 1, 1], dims=["cell_id"])
    data["latent_heat_flux_soil"] = DataArray([1, 1, 1], dims=["cell_id"])
    data["friction_velocity"] = DataArray([12, 5, 2], dims=["cell_id"])
    sensible_heat_flux = np.repeat(a=[0.0, np.nan, 0.0, np.nan], repeats=[4, 9, 1, 1])
    data["sensible_heat_flux"] = DataArray(
        np.broadcast_to(sensible_heat_flux, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="sensible_heat_flux",
        coords=full_coordinates,
    )
    latent_heat_flux = np.repeat(a=[0.0, np.nan, 0.0, np.nan], repeats=[4, 9, 1, 1])
    data["latent_heat_flux"] = DataArray(
        np.broadcast_to(latent_heat_flux, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="latent_heat_flux",
        coords=full_coordinates,
    )
    molar_density_air = np.repeat(a=[38.0, np.nan, 38.0, np.nan], repeats=[4, 7, 2, 2])
    data["molar_density_air"] = DataArray(
        np.broadcast_to(molar_density_air, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="molar_density_air",
        coords=full_coordinates,
    )
    specific_heat_air = np.repeat(a=[29.0, np.nan, 29.0, np.nan], repeats=[4, 7, 2, 2])
    data["specific_heat_air"] = DataArray(
        np.broadcast_to(specific_heat_air, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="specific_heat_air",
        coords=full_coordinates,
    )
    data["zero_displacement_height"] = DataArray(np.repeat(20.0, 3), dims="cell_id")
    data["diabatic_correction_heat_above"] = DataArray(
        np.repeat(0.1, 3),
        dims=["cell_id"],
        name="diabatic_correction_heat_above",
    )
    data["diabatic_correction_heat_canopy"] = DataArray(
        np.repeat(1.0, 3),
        dims=["cell_id"],
        name="diabatic_correction_heat_canopy",
    )
    data["diabatic_correction_momentum_above"] = DataArray(
        np.repeat(0.1, 3),
        dims=["cell_id"],
        name="diabatic_correction_momentum_above",
    )
    data["diabatic_correction_momentum_canopy"] = DataArray(
        np.repeat(1.0, 3),
        dims=["cell_id"],
        name="diabatic_correction_momentum_canopy",
    )
    attenuation_coefficient = np.repeat(
        a=[13.0, np.nan, 2.0, np.nan], repeats=[4, 7, 2, 2]
    )
    data["attenuation_coefficient"] = DataArray(
        np.broadcast_to(attenuation_coefficient, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="attenuation_coefficient",
        coords=full_coordinates,
    )
    data["mean_mixing_length"] = DataArray(np.repeat(1.3, 3), dims="cell_id")

    relative_turbulence_intensity = np.repeat(
        a=[[17.64, 16.56, 11.16, 5.76, np.nan, 1.17, 0.414, np.nan]],
        repeats=[1, 1, 1, 1, 7, 1, 1, 2],
    )
    data["relative_turbulence_intensity"] = DataArray(
        np.broadcast_to(relative_turbulence_intensity, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="relative_turbulence_intensity",
        coords=full_coordinates,
    )
    data["aerodynamic_resistance_surface"] = DataArray(
        np.repeat(12.5, 3), dims="cell_id"
    )
    latent_heat_vapourisation = np.repeat(
        a=[2254.0, np.nan, 2254.0, np.nan], repeats=[4, 7, 2, 2]
    )
    data["latent_heat_vapourisation"] = DataArray(
        np.broadcast_to(latent_heat_vapourisation, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="latent_heat_vapourisation",
        coords=full_coordinates,
    )
    canopy_temperature = np.repeat(a=[np.nan, 25.0, np.nan], repeats=[1, 3, 11])
    data["canopy_temperature"] = DataArray(
        np.broadcast_to(canopy_temperature, (3, 15)).T,
        dims=["layers", "cell_id"],
        name="canopy_temperature",
    ).assign_coords(full_coordinates)
    leaf_air_cond = np.repeat(a=[np.nan, 0.13, np.nan], repeats=[1, 3, 11])
    data["leaf_air_heat_conductivity"] = DataArray(
        np.broadcast_to(leaf_air_cond, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="leaf_air_heat_conductivity",
    )

    leaf_vap_cond = np.repeat(a=[np.nan, 0.2, np.nan], repeats=[1, 3, 11])
    data["leaf_vapour_conductivity"] = DataArray(
        np.broadcast_to(leaf_vap_cond, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="leaf_vapour_conductivity",
    )

    ref_cond = np.repeat(a=[np.nan, 3.0, np.nan, 3.0, np.nan], repeats=[1, 3, 7, 2, 2])
    data["conductivity_from_ref_height"] = DataArray(
        np.broadcast_to(ref_cond, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="conductivity_from_ref_height",
    )

    stomatal_conductance = np.repeat(a=[np.nan, 15.0, np.nan], repeats=[1, 3, 11])
    data["stomatal_conductance"] = DataArray(
        np.broadcast_to(stomatal_conductance, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="stomatal_conductance",
    )

    # Hydrology
    evapotranspiration = np.repeat(a=[np.nan, 20.0, np.nan], repeats=[1, 3, 11])
    data["evapotranspiration"] = DataArray(
        np.broadcast_to(evapotranspiration, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=full_coordinates,
        name="evapotranspiration",
    )
    data["soil_evaporation"] = DataArray(np.array([0.001, 0.01, 0.1]), dims="cell_id")
    data["surface_runoff"] = DataArray([10, 50, 100], dims="cell_id")
    data["surface_runoff_accumulated"] = DataArray([0, 10, 300], dims="cell_id")
    data["subsurface_flow_accumulated"] = DataArray([10, 10, 30], dims="cell_id")
    data["soil_moisture"] = xr.concat(
        [
            DataArray(np.full((13, 3), np.nan), dims=["layers", "cell_id"]),
            DataArray(np.full((2, 3), 0.20), dims=["layers", "cell_id"]),
        ],
        dim="layers",
    )
    data["groundwater_storage"] = DataArray(
        np.full((2, 3), 450.0),
        dims=("groundwater_layers", "cell_id"),
    )

    return data
