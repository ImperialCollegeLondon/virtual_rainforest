"""Test module for animal_cohorts.py."""

import pytest
from numpy import isclose, timedelta64


@pytest.fixture
def predator_functional_group_instance(shared_datadir):
    """Fixture for an animal functional group used in tests."""
    from virtual_rainforest.models.animals.functional_group import (
        import_functional_groups,
    )

    file = shared_datadir / "example_functional_group_import.csv"
    fg_list = import_functional_groups(file)

    return fg_list[2]


@pytest.fixture
def predator_cohort_instance(predator_functional_group_instance):
    """Fixture for an animal cohort used in tests."""
    from virtual_rainforest.models.animals.animal_cohorts import AnimalCohort

    return AnimalCohort(predator_functional_group_instance, 10000.0, 1)


@pytest.fixture
def ectotherm_functional_group_instance(shared_datadir):
    """Fixture for an animal functional group used in tests."""
    from virtual_rainforest.models.animals.functional_group import (
        import_functional_groups,
    )

    file = shared_datadir / "example_functional_group_import.csv"
    fg_list = import_functional_groups(file)

    return fg_list[5]


@pytest.fixture
def ectotherm_cohort_instance(ectotherm_functional_group_instance):
    """Fixture for an animal cohort used in tests."""
    from virtual_rainforest.models.animals.animal_cohorts import AnimalCohort

    return AnimalCohort(ectotherm_functional_group_instance, 100.0, 1)


@pytest.fixture
def prey_cohort_instance(herbivore_functional_group_instance):
    """Fixture for an animal cohort used in tests."""
    from virtual_rainforest.models.animals.animal_cohorts import AnimalCohort

    return AnimalCohort(herbivore_functional_group_instance, 100.0, 1)


@pytest.fixture
def carcass_instance():
    """Fixture for an carcass pool used in tests."""
    from virtual_rainforest.models.animals.decay import CarcassPool

    return CarcassPool(0.0, 0.0)


class TestAnimalCohort:
    """Test AnimalCohort class."""

    def test_initialization(self, herbivore_cohort_instance):
        """Testing initialization of derived parameters for animal cohorts."""
        assert herbivore_cohort_instance.individuals == 1
        assert herbivore_cohort_instance.stored_energy == pytest.approx(
            56531469253.03123, rel=1e-6
        )

    @pytest.mark.parametrize(
        "functional_group, mass, age, error_type",
        [
            (lambda fg: fg, -1000.0, 1.0, ValueError),
            (lambda fg: fg, 1000.0, -1.0, ValueError),
        ],
    )
    def test_invalid_animal_cohort_initialization(
        self,
        herbivore_functional_group_instance,
        functional_group,
        mass,
        age,
        error_type,
    ):
        """Test for invalid inputs during AnimalCohort initialization."""
        from virtual_rainforest.models.animals.animal_cohorts import AnimalCohort

        with pytest.raises(error_type):
            AnimalCohort(
                functional_group(herbivore_functional_group_instance), mass, age
            )

    @pytest.mark.parametrize(
        "dt, initial_energy, temperature, final_energy",
        [
            (
                timedelta64(1, "D"),
                28266000000.0,
                298.0,
                27989441986.150745,
            ),  # normal case
            (timedelta64(1, "D"), 500.0, 298.0, 0.0),  # edge case: low energy
            (timedelta64(1, "D"), 0.0, 298.0, 0.0),  # edge case: zero energy
            (timedelta64(3, "D"), 28266000000.0, 298.0, 27436325958.45224),  # 3 days
        ],
    )
    def test_metabolize_endotherm(
        self, herbivore_cohort_instance, dt, initial_energy, temperature, final_energy
    ):
        """Testing metabolize with an endothermic metabolism."""
        herbivore_cohort_instance.stored_energy = initial_energy
        herbivore_cohort_instance.metabolize(temperature, dt)
        assert isclose(herbivore_cohort_instance.stored_energy, final_energy, rtol=1e-9)

    @pytest.mark.parametrize(
        "dt, initial_energy, temperature, final_energy",
        [
            (
                timedelta64(1, "D"),
                28266000000.0,
                298.0,
                28265999999.700752,
            ),  # normal case
            (timedelta64(10, "D"), 1.0, 298.0, 0.0),  # edge case: low energy
            (timedelta64(1, "D"), 0.0, 298.0, 0.0),  # edge case: zero energy
        ],
    )
    def test_metabolize_ectotherm(
        self, ectotherm_cohort_instance, dt, initial_energy, temperature, final_energy
    ):
        """Testing metabolize."""
        ectotherm_cohort_instance.stored_energy = initial_energy
        ectotherm_cohort_instance.metabolize(temperature, dt)
        assert isclose(ectotherm_cohort_instance.stored_energy, final_energy, rtol=1e-9)

    @pytest.mark.parametrize(
        "dt, initial_energy, temperature, error_type",
        [
            (timedelta64(-1, "D"), 28266000000.0, 298.0, ValueError),
            (timedelta64(1, "D"), -100.0, 298.0, ValueError),
            # Add more invalid cases as needed
        ],
    )
    def test_metabolize_invalid_input(
        self, herbivore_cohort_instance, dt, initial_energy, temperature, error_type
    ):
        """Testing metabolize for invalid input."""
        herbivore_cohort_instance.stored_energy = initial_energy
        with pytest.raises(error_type):
            herbivore_cohort_instance.metabolize(temperature, dt)

    @pytest.mark.parametrize(
        "scav_initial, scav_final, decomp_initial, decomp_final, consumed_energy",
        [
            (1000.0, 1050.0, 0.0, 50.0, 1000.0),
            (0.0, 50.0, 1000.0, 1050.0, 1000.0),
            (1000.0, 1000.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1000.0, 1000.0, 0.0),
        ],
    )
    def test_excrete(
        self,
        herbivore_cohort_instance,
        excrement_instance,
        scav_initial,
        scav_final,
        decomp_initial,
        decomp_final,
        consumed_energy,
    ):
        """Testing excrete() for varying soil energy levels."""
        excrement_instance.scavengeable_energy = scav_initial
        excrement_instance.decomposed_energy = decomp_initial
        herbivore_cohort_instance.excrete(excrement_instance, consumed_energy)
        assert excrement_instance.scavengeable_energy == scav_final
        assert excrement_instance.decomposed_energy == decomp_final

    @pytest.mark.parametrize(
        "dt, initial_age, final_age",
        [
            (timedelta64(0, "D"), 0.0, 0.0),
            (timedelta64(1, "D"), 0.0, 1.0),
            (timedelta64(0, "D"), 3.0, 3.0),
            (timedelta64(90, "D"), 10.0, 100.0),
        ],
    )
    def test_increase_age(self, herbivore_cohort_instance, dt, initial_age, final_age):
        """Testing aging at varying ages."""
        herbivore_cohort_instance.age = initial_age
        herbivore_cohort_instance.increase_age(dt)
        assert herbivore_cohort_instance.age == final_age

    @pytest.mark.parametrize(
        argnames=[
            "number_dead",
            "initial_pop",
            "final_pop",
            "initial_carcass",
            "final_carcass",
            "decomp_carcass",
        ],
        argvalues=[
            (0, 0, 0, 0.0, 0.0, 0.0),
            (0, 1000, 1000, 0.0, 0.0, 0.0),
            (1, 1, 0, 1.0, 56000001.0, 1.4e7),
            (100, 200, 100, 0.0, 5.6e9, 1.4e9),
        ],
    )
    def test_die_individual(
        self,
        herbivore_cohort_instance,
        number_dead,
        initial_pop,
        final_pop,
        carcass_instance,
        initial_carcass,
        final_carcass,
        decomp_carcass,
    ):
        """Testing death."""
        herbivore_cohort_instance.individuals = initial_pop
        carcass_instance.scavengeable_energy = initial_carcass
        herbivore_cohort_instance.die_individual(number_dead, carcass_instance)
        assert herbivore_cohort_instance.individuals == final_pop
        assert carcass_instance.scavengeable_energy == final_carcass
        assert carcass_instance.decomposed_energy == decomp_carcass

    def test_get_eaten(
        self, prey_cohort_instance, predator_cohort_instance, carcass_instance
    ):
        """Testing get_eaten.

        Currently, this just tests rough execution. As the model gets paramterized,
        these tests will be expanded to specific values.
        """

        initial_individuals = prey_cohort_instance.individuals
        initial_scavengeable_energy = carcass_instance.scavengeable_energy

        # Execution
        prey_cohort_instance.get_eaten(predator_cohort_instance, carcass_instance)

        # Assertions
        assert prey_cohort_instance.individuals < initial_individuals
        assert carcass_instance.scavengeable_energy > initial_scavengeable_energy
        assert carcass_instance.decomposed_energy > 0.0

    def test_forage_cohort(
        self, predator_cohort_instance, prey_cohort_instance, mocker
    ):
        """Testing forage_cohort."""
        # Setup
        from virtual_rainforest.models.animals.animal_cohorts import AnimalCohort
        from virtual_rainforest.models.animals.animal_traits import DietType
        from virtual_rainforest.models.animals.decay import CarcassPool, ExcrementPool
        from virtual_rainforest.models.animals.dummy_plants import PlantCommunity

        # Mocking the eat method of AnimalCohort
        mock_eat = mocker.patch.object(AnimalCohort, "eat")

        # Instances
        plant_list_instance = [mocker.MagicMock(spec=PlantCommunity)]
        animal_list_instance = [
            mocker.MagicMock(spec=AnimalCohort) for _ in range(3)
        ]  # Assuming 3 animal cohorts
        carcass_pool_instance = mocker.MagicMock(spec=CarcassPool)
        excrement_pool_instance = mocker.MagicMock(spec=ExcrementPool)
        excrement_pool_instance.scavengeable_energy = 0
        excrement_pool_instance.decomposed_energy = 0

        animal_cohort_instances = [predator_cohort_instance, prey_cohort_instance]

        for animal_cohort_instance in animal_cohort_instances:
            # Execution
            animal_cohort_instance.forage_cohort(
                plant_list=plant_list_instance,
                animal_list=animal_list_instance,
                carcass_pool=carcass_pool_instance,
                excrement_pool=excrement_pool_instance,
            )

            # Assertions
            if animal_cohort_instance.functional_group.diet == DietType.HERBIVORE:
                mock_eat.assert_called_with(
                    plant_list_instance[0], excrement_pool_instance
                )  # Assuming just one plant instance for simplicity
            elif animal_cohort_instance.functional_group.diet == DietType.CARNIVORE:
                # Ensure eat was called for each animal in the list
                assert len(mock_eat.call_args_list) == 1
                for call in mock_eat.call_args_list:
                    # Ensure each call had a single AnimalCohort and the CarcassPool
                    args, _ = call
                    assert args[0] in animal_list_instance
                    assert args[1] == carcass_pool_instance

            # Reset mock_eat for next iteration
            mock_eat.reset_mock()

    def test_eat(self, herbivore_cohort_instance, mocker):
        """Testing eat."""
        from virtual_rainforest.models.animals.protocols import Pool, Resource

        mock_food = mocker.MagicMock(spec=Resource)
        mock_pool = mocker.MagicMock(spec=Pool)

        herbivore_cohort_instance.individuals = (
            10  # Setting a non-zero value for individuals
        )
        herbivore_cohort_instance.stored_energy = (
            0  # Setting initial energy to 0 for simplicity
        )

        # Mocking get_eaten to return a fixed energy value
        mock_energy_return = 100  # Example energy return value
        mock_food.get_eaten.return_value = mock_energy_return

        # Execution
        herbivore_cohort_instance.eat(mock_food, mock_pool)

        # Assertions
        mock_food.get_eaten.assert_called_once_with(
            herbivore_cohort_instance, mock_pool
        )
        assert (
            herbivore_cohort_instance.stored_energy
            == mock_energy_return / herbivore_cohort_instance.individuals
        )

        # Test ValueError for zero individuals
        herbivore_cohort_instance.individuals = 0
        with pytest.raises(ValueError, match="Individuals cannot be 0."):
            herbivore_cohort_instance.eat(mock_food, mock_pool)

    def test_can_reproduce_method(self, herbivore_cohort_instance):
        """Test the can_reproduce method of AnimalCohort."""

        # 1. Test when stored_energy is exactly equal to the threshold
        herbivore_cohort_instance.stored_energy = (
            herbivore_cohort_instance.reproduction_energy_threshold
        )
        assert herbivore_cohort_instance.can_reproduce()

        # 2. Test when stored_energy is just below the threshold
        herbivore_cohort_instance.stored_energy = (
            herbivore_cohort_instance.reproduction_energy_threshold - 0.01
        )
        assert not herbivore_cohort_instance.can_reproduce()

        # 3. Test when stored_energy is above the threshold
        herbivore_cohort_instance.stored_energy = (
            herbivore_cohort_instance.reproduction_energy_threshold + 0.01
        )
        assert herbivore_cohort_instance.can_reproduce()

        # 4. Test with stored_energy set to 0
        herbivore_cohort_instance.stored_energy = 0.0
        assert not herbivore_cohort_instance.can_reproduce()

    @pytest.mark.parametrize(
        "initial_individuals, number_days, mortality_prob",
        [(100, 10.0, 0.01), (1000, 20.0, 0.05), (0, 10.0, 0.01), (100, 10.0, 0.0)],
    )
    def test_inflict_natural_mortality(
        self,
        herbivore_cohort_instance,
        carcass_instance,
        mocker,
        initial_individuals,
        number_days,
        mortality_prob,
    ):
        """Testing inflict natural mortality method."""
        from random import seed

        from numpy import floor

        seed(42)

        expected_deaths = initial_individuals * (
            1 - (1 - mortality_prob) ** number_days
        )
        expected_deaths = int(floor(expected_deaths))

        # Set individuals and adult natural mortality probability
        herbivore_cohort_instance.individuals = initial_individuals
        herbivore_cohort_instance.adult_natural_mortality_prob = mortality_prob

        # Mock the random.binomial call
        mocker.patch(
            "virtual_rainforest.models.animals.animal_cohorts.random.binomial",
            return_value=expected_deaths,
        )
        # Keep a copy of initial individuals to validate number_of_deaths
        initial_individuals_copy = herbivore_cohort_instance.individuals

        # Call the inflict_natural_mortality method
        herbivore_cohort_instance.inflict_natural_mortality(
            carcass_instance, number_days
        )

        # Verify the number_of_deaths and remaining individuals
        assert (
            herbivore_cohort_instance.individuals
            == initial_individuals_copy - expected_deaths
        )
