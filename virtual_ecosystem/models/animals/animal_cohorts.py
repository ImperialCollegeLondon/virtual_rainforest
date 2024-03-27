"""The ''animals'' module provides animal module functionality.

Notes:
- assume each grid = 1 km2
- assume each tick = 1 day (28800s)
- damuth ~ 4.23*mass**(-3/4) indiv / km2
"""  # noqa: #D205, D415

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, exp

from numpy import random, timedelta64

import virtual_ecosystem.models.animals.scaling_functions as sf
from virtual_ecosystem.core.logger import LOGGER
from virtual_ecosystem.models.animals.animal_traits import DietType
from virtual_ecosystem.models.animals.constants import AnimalConsts
from virtual_ecosystem.models.animals.decay import CarcassPool
from virtual_ecosystem.models.animals.functional_group import FunctionalGroup
from virtual_ecosystem.models.animals.protocols import Consumer, DecayPool, Resource


class AnimalCohort:
    """This is a class of animal cohorts."""

    def __init__(
        self,
        functional_group: FunctionalGroup,
        mass: float,
        age: float,
        individuals: int,
        constants: AnimalConsts = AnimalConsts(),
    ) -> None:
        if age < 0:
            raise ValueError("Age must be a positive number.")
        """Check if age is a positive number. """

        if mass < 0:
            raise ValueError("Mass must be a positive number.")
        """Check if mass is a positive number."""

        """The constructor for the AnimalCohort class."""
        self.functional_group = functional_group
        """The functional group of the animal cohort which holds constants."""
        self.name = functional_group.name
        """The functional type name of the animal cohort."""
        """The average mass of an individual in the animal cohort [kg]."""
        self.mass_current = mass
        """The current average body mass of an individual [kg]."""
        self.age = age
        """The age of the animal cohort [days]."""
        self.individuals = individuals
        """The number of individuals in this cohort."""
        self.constants = constants
        """Animal constants."""
        self.damuth_density: int = sf.damuths_law(
            self.functional_group.adult_mass, self.functional_group.damuths_law_terms
        )
        """The number of individuals in an average cohort of this type."""
        self.is_alive: bool = True
        """Whether the cohort is alive [True] or dead [False]."""
        self.reproductive_mass: float = 0.0
        """The pool of biomass from which the material of reproduction is drawn."""

        self.intake_rate: float = sf.intake_rate_scaling(
            self.functional_group.adult_mass, self.functional_group.intake_rate_terms
        )
        """The individual rate of plant mass consumption over an 8hr foraging day
        [kg/day]."""
        self.prey_groups = sf.prey_group_selection(
            self.functional_group.diet,
            self.functional_group.adult_mass,
            self.functional_group.prey_scaling,
        )
        """The identification of useable food resources."""

        self.adult_natural_mortality_prob = sf.natural_mortality_scaling(
            self.functional_group.adult_mass, self.functional_group.longevity_scaling
        )
        # TODO: Distinguish between background, senesence, and starvation mortalities.
        """The per-day probability of an individual dying to natural causes."""

        # TODO - In future this should be parameterised using a constants dataclass, but
        # this hasn't yet been implemented for the animal model
        self.decay_fraction_excrement: float = self.constants.decay_fraction_excrement
        """The fraction of excrement which decays before it gets consumed."""
        self.decay_fraction_carcasses: float = self.constants.decay_fraction_carcasses
        """The fraction of carcass biomass which decays before it gets consumed."""

    def metabolize(self, temperature: float, dt: timedelta64) -> float:
        """The function to reduce mass_current through basal metabolism.

        TODO: Implement distinction between field and basal rates.
        TODO: Implement proportion of day active.
        TODO: clean up units
        TODO: implement distinction between metabolic waste flow to atmosphere vs urea

        Args:
            temperature: Current air temperature (K)
            dt: Number of days over which the metabolic costs should be calculated.

        Returns:
            The mass of metabolic waste produced.

        """

        if dt < timedelta64(0, "D"):
            raise ValueError("dt cannot be negative.")

        if self.mass_current < 0:
            raise ValueError("mass_current cannot be negative.")

        #  kg/day metabolic rate * number of days
        potential_mass_metabolized = sf.metabolic_rate(
            self.mass_current,
            temperature,
            self.functional_group.metabolic_rate_terms,
            self.functional_group.metabolic_type,
        ) * float(dt / timedelta64(1, "D"))

        actual_mass_metabolized = min(self.mass_current, potential_mass_metabolized)

        self.mass_current -= actual_mass_metabolized

        # returns total metabolic waste from cohort to animal_communities for tracking
        # in data object
        return actual_mass_metabolized * self.individuals

    def excrete(
        self,
        excrement_pool: DecayPool,
        mass_consumed: float,
    ) -> None:
        """Transfer waste mass from an animal cohort to the excrement pool.

        Currently, this function is in an inbetween state where mass is removed from
        the animal cohort but it is recieved by the litter pool as energy. This will be
        fixed once the litter pools are updated for mass.

        TODO: Rework after update litter pools for mass

        Args:
            excrement_pool: The local ExcrementSoil pool in which waste is deposited.
            mass_consumed: The amount of mass flowing through cohort digestion.
        """
        # Find total waste mass, the total amount of waste is then found by the
        # average cohort member * number individuals.
        waste_energy = mass_consumed * self.functional_group.conversion_efficiency

        # This total waste is then split between decay and scavengeable excrement
        excrement_pool.scavengeable_energy += (
            (1 - self.decay_fraction_excrement) * waste_energy * self.individuals
        )
        excrement_pool.decomposed_energy += (
            self.decay_fraction_excrement * waste_energy * self.individuals
        )

    def increase_age(self, dt: timedelta64) -> None:
        """The function to modify cohort age as time passes.

        Args:
            dt: The amount of time that should be added to cohort age.

        """
        self.age += float(dt / timedelta64(1, "D"))

    def die_individual(self, number_dead: int, carcass_pool: CarcassPool) -> None:
        """The function to reduce the number of individuals in the cohort through death.

        Currently, all cohorts are crafted as single km2 grid cohorts. This means that
        very large animals will have one or fewer cohort members per grid. As changes
        are made to capture large body size and multi-grid occupancy, this will be
        updated.

        Currently, this function is in an inbetween state where mass is removed from
        the animal cohort but it is recieved by the litter pool as energy. This will be
        fixed once the litter pools are updated for mass.

        TODO: Rework after update litter pools for mass

        Args:
            number_dead: The number of individuals by which to decrease the population
                count.
            carcass_pool: The resident pool of animal carcasses to which the dead
                individuals are delivered.

        """
        self.individuals -= number_dead

        # Find total mass contained in the carcasses
        carcass_mass = number_dead * self.mass_current

        # Split this mass between carcass decay, and scavengeable carcasses
        carcass_pool.scavengeable_energy += (
            1 - self.decay_fraction_carcasses
        ) * carcass_mass
        carcass_pool.decomposed_energy += self.decay_fraction_carcasses * carcass_mass

    def update_carcass_pool(self, carcass_mass: float, carcass_pool: DecayPool) -> None:
        """Updates the carcass pool based on consumed mass and predator's efficiency.

        Args:
            carcass_mass: The total mass consumed from the prey cohort.
            carcass_pool: The pool to which remains of eaten individuals are delivered.
        """

        # Update the carcass pool with the remainder
        carcass_pool.scavengeable_energy += (
            1 - self.decay_fraction_carcasses
        ) * carcass_mass
        carcass_pool.decomposed_energy += self.decay_fraction_carcasses * carcass_mass

    def get_eaten(
        self,
        potential_consumed_mass: float,
        predator: Consumer,
        carcass_pool: DecayPool,
    ) -> float:
        """Adjusts individuals from the prey cohort based on the consumed mass.

        Args:
            potential_consumed_mass: The mass intended to be consumed by the predator.
            predator: The predator consuming the cohort.
            carcass_pool: The pool to which remains of eaten individuals are delivered.

        Returns:
            The actual mass consumed by the predator, closely matching consumed_mass.
        """
        individual_mass = (
            self.mass_current
        )  # Mass of an average individual in the cohort
        max_individuals_eaten = ceil(potential_consumed_mass / individual_mass)
        actual_individuals_eaten = min(max_individuals_eaten, self.individuals)

        # Calculate the mass represented by the individuals actually eaten
        actual_killed_mass = actual_individuals_eaten * individual_mass

        # Calculate the actual amount of mass consumed by the predator
        actual_consumed_mass = min(actual_killed_mass, potential_consumed_mass)

        # Calculate the amount of mass that goes into carcass pool
        carcass_mass = (actual_killed_mass - actual_consumed_mass) + (
            actual_consumed_mass * (1 - predator.functional_group.mechanical_efficiency)
        )

        # Update the number of individuals in the prey cohort
        self.individuals -= actual_individuals_eaten

        # Update the carcass pool with carcass mass
        self.update_carcass_pool(carcass_mass, carcass_pool)

        return actual_consumed_mass

    def calculate_alpha(self) -> float:
        """Calculate search efficiency.

        This utilizes the alpha_i_k scaling function to determine the effective rate at
        which an individual herbivore searches its environment, factoring in the
        herbivore's current mass.

        Returns:
            A float representing the search efficiency rate in [ha/(day*g)].
        """

        return sf.alpha_i_k(self.constants.alpha_0_herb, self.mass_current)

    def calculate_potential_consumed_biomass(
        self, target_plant: Resource, alpha: float
    ) -> float:
        """Calculate potential consumed biomass for the target plant.

        This method computes the potential consumed biomass based on the search
        efficiency (alpha),the fraction of the total plant stock available to the cohort
        (phi), and the biomass of the target plant.

        Args:
            target_plant: The plant resource being targeted by the herbivore cohort.
            alpha: The search efficiency rate of the herbivore cohort.

        Returns:
            A float representing the potential consumed biomass of the target plant by
            the cohort [g/day].
        """

        phi = self.functional_group.constants.phi_herb_t
        A_cell = 1.0  # temporary
        return sf.k_i_k(alpha, phi, target_plant.mass_current, A_cell)

    def calculate_total_handling_time_for_herbivory(
        self, plant_list: Sequence[Resource], alpha: float
    ) -> float:
        """Calculate total handling time across all plant resources.

        This aggregates the handling times for consuming each plant resource in the
        list, incorporating the search efficiency and other scaling factors to compute
        the total handling time required by the cohort.

        Args:
            plant_list: A sequence of plant resources available for consumption by the
            cohort.
            alpha: The search efficiency rate of the herbivore cohort.

        Returns:
            A float representing the total handling time in days required by the cohort
            for all available plant resources.
        """

        phi = self.functional_group.constants.phi_herb_t
        A_cell = 1.0  # temporary
        return sum(
            sf.k_i_k(alpha, phi, plant.mass_current, A_cell)
            + sf.H_i_k(
                self.constants.h_herb_0,
                self.constants.M_herb_ref,
                self.mass_current,
                self.constants.b_herb,
            )
            for plant in plant_list
        )

    def F_i_k(self, plant_list: Sequence[Resource], target_plant: Resource) -> float:
        """Method to determine instantaneous herbivory rate on plant k.

        This method integrates the calculated search efficiency, potential consumed
        biomass of the target plant, and the total handling time for all available
        plant resources to determine the rate at which the target plant is consumed by
        the cohort.

        Args:
            plant_list: A sequence of plant resources available for consumption by the
                 cohort.
            target_plant: The specific plant resource being targeted by the herbivore
                 cohort for consumption.

        Returns:
            The instantaneous consumption rate [g/day] of the target plant resource by
              the herbivore cohort.
        """
        alpha = self.calculate_alpha()
        k = self.calculate_potential_consumed_biomass(target_plant, alpha)
        total_handling_t = self.calculate_total_handling_time_for_herbivory(
            plant_list, alpha
        )
        B_k = target_plant.mass_current  # current plant biomass
        N = self.individuals  # herb cohort size
        return N * (k / (1 + total_handling_t)) * (1 / B_k)

    def calculate_theta_opt_i(self) -> float:
        """Calculate the optimal predation param based on predator-prey mass ratio.

        Returns:
            Float value of the optimal predation parameter for use in calculating the
            probability of a predation event being successful.

        """
        return sf.theta_opt_i(
            self.constants.theta_opt_min_f,
            self.constants.theta_opt_f,
            self.constants.sigma_opt_f,
        )

    def calculate_predation_success_probability(self, M_target: float) -> float:
        """Calculate the probability of a successful predation event.

        Args:
            M_target: the body mass of the animal cohort being targeted for predation.

        Returns:
            A float value of the probability that a predation event is successful.

        """
        M_i = self.mass_current
        theta_opt_i = self.calculate_theta_opt_i()
        return sf.w_bar_i_j(
            M_i,
            M_target,
            theta_opt_i,
            self.constants.sigma_opt_pred_prey,
        )

    def calculate_predation_search_rate(self, w_bar: float) -> float:
        """Calculate the search rate of the predator.

        Args:
            w_bar: Probability of successfully capturing prey.

        Returns:
            A float value of the search rate in ha/day

        """
        return sf.alpha_i_j(self.constants.alpha_0_pred, self.mass_current, w_bar)

    def calculate_potential_prey_consumed(
        self, alpha: float, theta_i_j: float
    ) -> float:
        """Calculate the potential number of prey consumed.

        Args:
            alpha: the predation search rate
            theta_i_j: The cumulative density of organisms with a mass lying within the
              same predator specific mass bin.

        Returns:
            The potential number of prey items consumed.

        """
        A_cell = 1.0  # temporary
        return sf.k_i_j(alpha, self.individuals, A_cell, theta_i_j)

    def calculate_total_handling_time_for_predation(self) -> float:
        """Calculate the total handling time for preying on available animal cohorts.

        Returns:
            A float value of handling time in days.

        """
        return sf.H_i_j(
            self.constants.h_pred_0,
            self.constants.M_pred_ref,
            self.mass_current,
            self.constants.b_pred,
        )

    def F_i_j_individual(
        self, animal_list: Sequence[AnimalCohort], target_cohort: AnimalCohort
    ) -> float:
        """Method to determine instantaneous predation rate on cohort j.

        Args:
            animal_list: A sequence of animal cohorts that can be consumed by the
                predator.
            target_cohort: The prey cohort from which mass will be consumed.

        Returns:
            Float fraction of target cohort consumed per day.


        """
        w_bar = self.calculate_predation_success_probability(target_cohort.mass_current)
        alpha = self.calculate_predation_search_rate(w_bar)
        theta_i_j = self.theta_i_j(animal_list)  # Assumes implementation of theta_i_j
        k_target = self.calculate_potential_prey_consumed(alpha, theta_i_j)
        total_handling_t = self.calculate_total_handling_time_for_predation()
        N_i = self.individuals
        N_target = target_cohort.individuals

        return N_i * (k_target / (1 + total_handling_t)) * (1 / N_target)

    def calculate_consumed_mass_predation(
        self, animal_list: Sequence[AnimalCohort], target_cohort: AnimalCohort
    ) -> float:
        """Calculates the mass to be consumed from a prey cohort by the predator.

        This method utilizes the F_i_j_individual method to determine the rate at which
        the target cohort is consumed, and then calculates the actual mass to be
        consumed based on this rate and other model parameters.

        Args:
            animal_list: A sequence of animal cohorts that can be consumed by the
                predator.
            target_cohort: The prey cohort from which mass will be consumed.

        Returns:
            The mass to be consumed from the target cohort by the predator (in kg).
        """
        F = self.F_i_j_individual(animal_list, target_cohort)
        delta_t = 30.0  # days, TODO: Replace with actual reference or model parameter

        # Calculate the consumed mass based on Mad. formula for delta_mass_predation
        consumed_mass = (
            target_cohort.mass_current
            * target_cohort.individuals
            * (
                1
                - exp(-(F * delta_t * self.constants.tau_f * self.constants.sigma_f_t))
            )
        )

        return consumed_mass

    def delta_mass_predation(
        self,
        animal_list: Sequence[AnimalCohort],
        excrement_pool: DecayPool,
        carcass_pool: CarcassPool,
    ) -> float:
        """This method handles mass assimilation from predation.

        This is Madingley's delta_assimilation_mass_predation

        TODO: Replace delta_t values with actual reference
        TODO: add epsilon conversion efficiency

        Args:
            animal_list: A sequence of animal cohorts that can be consumed by the
                         predator.

        Returns:
            The change in mass experienced by the predator.
        """

        total_consumed_mass = 0.0  # Initialize the total consumed mass

        for cohort in animal_list:
            # Calculate the mass to be consumed from this cohort
            consumed_mass = self.calculate_consumed_mass_predation(animal_list, cohort)
            # Call get_eaten on the prey cohort to update its mass and individuals
            actual_consumed_mass = cohort.get_eaten(consumed_mass, self, carcass_pool)
            # Update total mass gained by the predator
            total_consumed_mass += actual_consumed_mass

        # Process waste generated from predation, separate from herbivory b/c diff waste
        self.excrete(excrement_pool, total_consumed_mass)
        return total_consumed_mass

    def calculate_consumed_mass_herbivory(
        self, plant_list: Sequence[Resource], target_plant: Resource
    ) -> float:
        """Calculates the mass to be consumed from a plant resource by the herbivore.

        This method utilizes the F_i_k method to determine the rate at which the target
        plant is consumed, and then calculates the actual mass to be consumed based on
        this rate and other model parameters.

        Args:
            target_plant: The plant resource from which mass will be consumed.

        Returns:
            The mass to be consumed from the target plant by the herbivore (in kg).
        """
        F = self.F_i_k(plant_list, target_plant)  # Adjusting this call as necessary
        delta_t = 30.0  # days, TODO: Replace with actual reference or model parameter

        consumed_mass = target_plant.mass_current * (
            1 - exp(-(F * delta_t * self.constants.tau_f * self.constants.sigma_f_t))
        )
        return consumed_mass

    def delta_mass_herbivory(
        self, plant_list: Sequence[Resource], excrement_pool: DecayPool
    ) -> float:
        """This method handles mass assimilation from herbivory.

        Args:
            plant_list: A sequence of plant resources available for herbivory.
            excrement_pool: A pool representing the excrement in the grid cell.

        Returns:
            A float of the total plant mass consumed by the animal cohort in g.

        """
        total_consumed_mass = 0.0  # Initialize the total consumed mass

        for plant in plant_list:
            # Calculate the mass to be consumed from this plant
            consumed_mass = self.calculate_consumed_mass_herbivory(plant_list, plant)
            # Update the plant resource's state based on consumed mass
            actual_consumed_mass = plant.get_eaten(consumed_mass, self, excrement_pool)
            # Update total mass gained by the herbivore
            total_consumed_mass += actual_consumed_mass

        return total_consumed_mass

    def forage_cohort(
        self,
        plant_list: Sequence[Resource],
        animal_list: Sequence[AnimalCohort],
        excrement_pool: DecayPool,
        carcass_pool: CarcassPool,
    ) -> None:
        """This function handles selection of resources from a list for consumption.

        Args:
            plant_list: A sequence of plant resources available for herbivory.
            animal_list: A sequence of animal cohorts available for predation.
            excrement_pool: A pool representing the excrement in the grid cell.
            carcass_pool: A pool representing the carcasses in the grid cell.

        Return:
            A float value of the net change in consumer mass due to foraging.
        """
        if self.individuals == 0:
            LOGGER.warning("No individuals in cohort to forage.")
            return

        # Herbivore diet
        if self.functional_group.diet == DietType.HERBIVORE and plant_list:
            consumed_mass = self.delta_mass_herbivory(
                plant_list, excrement_pool
            )  # Directly modifies the plant mass
            self.eat(consumed_mass)  # Accumulate net mass gain from each plant

        # Carnivore diet
        elif self.functional_group.diet == DietType.CARNIVORE and animal_list:
            # Calculate the mass gained from predation
            consumed_mass = self.delta_mass_predation(
                animal_list, excrement_pool, carcass_pool
            )
            # Update the predator's mass with the total gained mass
            self.eat(consumed_mass)

        else:
            # No appropriate food sources for the diet type
            raise ValueError(
                f"No appropriate foods available for {self.functional_group.diet} diet."
            )

    def theta_i_j(self, animal_list: Sequence[AnimalCohort]) -> float:
        """Cumulative density method for delta_mass_predation.

        The cumulative density of organisms with a mass lying within the same predator
        specific mass bin as Mi.

        Madingley

        TODO: current format makes no sense, dig up the details in the supp
        TODO: update A_cell with real reference

        Args:
            animal_list: A sequence of animal cohorts that can be consumed by the
                         predator.

        Returns:
            The float value of theta.
        """
        A_cell = 1.0  # temporary

        return sum(
            cohort.individuals / A_cell
            for cohort in animal_list
            if self.mass_current == cohort.mass_current
        )

    def eat(self, mass_consumed: float) -> None:
        """Handles the mass gain from consuming food.

        This method updates the consumer's mass based on the amount of food consumed.
        It assumes the `mass_consumed` has already been calculated and processed
        through `get_eaten`.

        Args:
            mass_consumed: The mass consumed by this consumer, calculated externally.

        Returns:
            The amount of consumed mass, adjusted for efficiency and used for waste
            output.
        """
        if self.individuals == 0:
            return

        # Adjust mass gain based on the consumer's current mass and reproductive
        # threshold
        if self.is_below_mass_threshold(
            self.constants.flow_to_reproductive_mass_threshold
        ):
            self.mass_current += mass_consumed  # Gains to reproductive or body mass
        else:
            self.reproductive_mass += mass_consumed

    def is_below_mass_threshold(self, mass_threshold: float) -> bool:
        """Check if cohort's total mass is below a certain threshold.

        Currently used for thesholding: birth, dispersal, trophic flow to reproductive
        mass.

        Args:
            mass_threshold: a float value holding a threshold ratio of current total
                mass to standard adult mass.

        Return:
            A bool of whether the current mass state is above the migration threshold.
        """
        return (
            self.mass_current + self.reproductive_mass
        ) / self.functional_group.adult_mass < mass_threshold

    def inflict_natural_mortality(
        self, carcass_pool: CarcassPool, number_days: float
    ) -> None:
        """The function to cause natural mortality in a cohort.

        TODO Find a more efficient structure so we aren't recalculating the
        time_step_mortality. Probably pass through the initialized timestep size to the
        scaling function

        Args:
            carcass_pool: The grid-local carcass pool to which the dead matter is
                transferred.
            number_days: Number of days over which the metabolic costs should be
                calculated.

        """

        # Calculate the mortality probability for the entire time step
        time_step_mortality_prob = (
            1 - (1 - self.adult_natural_mortality_prob) ** number_days
        )
        # Draw the number of deaths from a binomial distribution
        number_of_deaths = random.binomial(
            n=self.individuals, p=time_step_mortality_prob
        )

        self.die_individual(number_of_deaths, carcass_pool)
