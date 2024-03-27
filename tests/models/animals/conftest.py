"""Collection of fixtures to assist the animal model testing scripts."""

import numpy as np
import pytest
import xarray
from xarray import DataArray


@pytest.fixture
def plant_data_instance():
    """Fixture returning a simple data instance containing plant resource data."""
    from virtual_ecosystem.core.data import Data
    from virtual_ecosystem.core.grid import Grid

    # Populate data with a layers x cell id layer_leaf_mass array
    data = Data(grid=Grid(cell_nx=3, cell_ny=3))
    leaf_mass = np.full((15, 9), fill_value=np.nan)
    leaf_mass[1:4, :] = 10000
    data["layer_leaf_mass"] = xarray.DataArray(
        data=leaf_mass, dims=["layers", "cell_id"]
    )

    return data


@pytest.fixture
def plant_climate_data_instance(fixture_core_components):
    """Fixture returning a combination of plant and air temperature data."""

    from virtual_ecosystem.core.data import Data
    from virtual_ecosystem.core.grid import Grid

    # Setup the data object with four cells.
    grid = Grid(
        grid_type="square",
        cell_nx=3,
        cell_ny=1,
    )
    data = Data(grid)

    leaf_mass = np.full((15, 3), fill_value=np.nan)
    leaf_mass[1:4, :] = 10000
    data["layer_leaf_mass"] = xarray.DataArray(
        data=leaf_mass, dims=["layers", "cell_id"]
    )
    data["air_temperature"] = xarray.concat(
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
    ).assign_coords(
        {
            "layers": np.arange(0, 15),
            "layer_roles": (
                "layers",
                fixture_core_components.layer_structure.layer_roles[0:15],
            ),
            "cell_id": data.grid.cell_id,
        }
    )

    return data


@pytest.fixture
def constants_instance():
    """Fixture for an instance of animal constants."""
    from virtual_ecosystem.models.animals.constants import AnimalConsts

    return AnimalConsts()


@pytest.fixture
def functional_group_list_instance(shared_datadir, constants_instance):
    """Fixture for an animal functional group used in tests."""
    from virtual_ecosystem.models.animals.functional_group import (
        import_functional_groups,
    )

    file = shared_datadir / "example_functional_group_import.csv"
    fg_list = import_functional_groups(file, constants_instance)

    return fg_list


@pytest.fixture
def animal_model_instance(
    data_instance,
    fixture_core_components,
    functional_group_list_instance,
    constants_instance,
):
    """Fixture for an animal model object used in tests."""

    from virtual_ecosystem.models.animals.animal_model import AnimalModel

    return AnimalModel(
        data=data_instance,
        core_components=fixture_core_components,
        functional_groups=functional_group_list_instance,
        model_constants=constants_instance,
    )


@pytest.fixture
def animal_community_instance(
    functional_group_list_instance,
    animal_model_instance,
    plant_data_instance,
    constants_instance,
):
    """Fixture for an animal community used in tests."""
    from virtual_ecosystem.models.animals.animal_communities import AnimalCommunity

    return AnimalCommunity(
        functional_groups=functional_group_list_instance,
        data=plant_data_instance,
        community_key=4,
        neighbouring_keys=[1, 3, 5, 7],
        get_destination=animal_model_instance.get_community_by_key,
        constants=constants_instance,
    )


@pytest.fixture
def herbivore_functional_group_instance(shared_datadir, constants_instance):
    """Fixture for an animal functional group used in tests."""
    from virtual_ecosystem.models.animals.functional_group import (
        import_functional_groups,
    )

    file = shared_datadir / "example_functional_group_import.csv"
    fg_list = import_functional_groups(file, constants_instance)

    return fg_list[3]


@pytest.fixture
def herbivore_cohort_instance(herbivore_functional_group_instance, constants_instance):
    """Fixture for an animal cohort used in tests."""
    from virtual_ecosystem.models.animals.animal_cohorts import AnimalCohort

    return AnimalCohort(
        herbivore_functional_group_instance, 10000.0, 1, 10, constants_instance
    )


@pytest.fixture
def excrement_pool_instance():
    """Fixture for a soil pool used in tests."""
    from virtual_ecosystem.models.animals.decay import ExcrementPool

    return ExcrementPool(100000.0, 0.0)


@pytest.fixture
def plant_instance(plant_data_instance, constants_instance):
    """Fixture for a plant community used in tests."""
    from virtual_ecosystem.models.animals.plant_resources import PlantResources

    return PlantResources(
        data=plant_data_instance, cell_id=4, constants=constants_instance
    )


@pytest.fixture
def plant_list_instance(plant_data_instance, constants_instance):
    """Fixture providing a list of plant resources."""
    from virtual_ecosystem.models.animals.plant_resources import PlantResources

    return [
        PlantResources(
            data=plant_data_instance, cell_id=4, constants=constants_instance
        ),
        PlantResources(
            data=plant_data_instance, cell_id=4, constants=constants_instance
        ),
        PlantResources(
            data=plant_data_instance, cell_id=4, constants=constants_instance
        ),
    ]


@pytest.fixture
def animal_list_instance(herbivore_functional_group_instance, constants_instance):
    """Fixture providing a list of animal cohorts."""
    from virtual_ecosystem.models.animals.animal_cohorts import AnimalCohort

    return [
        AnimalCohort(
            herbivore_functional_group_instance, 10000.0, 1, 10, constants_instance
        ),
        AnimalCohort(
            herbivore_functional_group_instance, 10000.0, 1, 10, constants_instance
        ),
        AnimalCohort(
            herbivore_functional_group_instance, 10000.0, 1, 10, constants_instance
        ),
    ]
