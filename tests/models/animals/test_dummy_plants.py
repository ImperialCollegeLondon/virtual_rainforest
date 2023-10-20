"""Test module for dummy_plants.py."""

import pytest


class TestPlantCommunity:
    """Test Plant class."""

    @pytest.mark.parametrize(
        "initial, final",
        [(10000.0, 10000.0), (500.0, 975.0), (0.0, 0.0)],
    )
    def test_grow(self, plant_instance, initial, final):
        """Testing grow at 100%, 50%, and 0% maximum energy."""
        plant_instance.mass_current = initial
        plant_instance.grow()
        assert plant_instance.mass_current == pytest.approx(final, rel=1e-6)

    def test_die(self, plant_instance):
        """Testing die."""
        assert plant_instance.is_alive
        plant_instance.die()
        assert not plant_instance.is_alive

    def test_get_eaten(
        self, plant_instance, herbivore_cohort_instance, excrement_instance
    ):
        """Testing get_eaten.

        Currently, this just tests rough execution. As the model gets paramterized,
        these tests will be expanded to specific values.
        """

        initial_plant_mass = plant_instance.mass_current
        initial_decay_pool_energy = excrement_instance.decomposed_energy

        # Execution
        plant_instance.get_eaten(herbivore_cohort_instance, excrement_instance)

        # Assertions
        assert plant_instance.mass_current < initial_plant_mass
        assert excrement_instance.decomposed_energy > initial_decay_pool_energy
