"""The `models.animals.scaling_functions` module contains a set of functions containing
scaling equations" (relationships between body-mass and a trait) required by the broader
:mod:`~virtual_ecosystem.models.animals` module

To Do:
- streamline units of scaling functions [kg]->[kg] etc

"""  # noqa: D205, D415

from math import ceil, exp, log

import numpy as np

from virtual_ecosystem.models.animals.animal_traits import DietType, MetabolicType
from virtual_ecosystem.models.animals.constants import BOLTZMANN_CONSTANT


def damuths_law(mass: float, terms: tuple) -> int:
    """The function set initial population densities .

        Currently, this function just employs Damuth's Law (Damuth 1987) for
        terrestrial herbivorous mammals. Later, it will be expanded to other types. The
        current form takes the ceiling of the population density to ensure there is a
        minimum of 1 individual and integer values. This will be corrected once the
        multi-grid occupation system for large animals is implemented.

    Args:
        mass: The body-mass [kg] of an AnimalCohort.
        terms: The tuple of population density terms used, default to Damuth.

    Returns:
        The population density of that AnimalCohort [individuals/km2].

    """

    return ceil(terms[1] * mass ** terms[0])


def metabolic_rate_energy(
    mass: float, temperature: float, terms: tuple, metabolic_type: MetabolicType
) -> float:
    """Calculates the metabolic rate of animal cohorts.

    TODO: No longer in use. Remove this method after constants rework.

    Args:
         mass: The body-mass [kg] of an AnimalCohort.
         temperature: The temperature [Celsius] of the environment.
         terms: The tuple of metabolic rate terms used.
         metabolic_type: The metabolic type of the animal [ENDOTHERMIC or ECTOTHERMIC].

    Returns:
         The metabolic rate of an individual of the given cohort in [J/s].

    """
    mass_g = mass * 1000  # Convert mass to grams
    temperature_k = temperature + 273.15  # Convert temperature to Kelvin

    if metabolic_type == MetabolicType.ENDOTHERMIC:
        return terms[1] * mass_g ** terms[0]
    elif metabolic_type == MetabolicType.ECTOTHERMIC:
        b0, exponent = terms
        return b0 * mass_g**exponent * exp(-0.65 / (BOLTZMANN_CONSTANT * temperature_k))
    else:
        raise ValueError("Invalid metabolic type: {metabolic_type}")


def metabolic_rate(
    mass: float,
    temperature: float,
    terms: dict,
    metabolic_type: MetabolicType,
) -> float:
    """Calculates metabolic rate in grams of body mass per day.

    This follows the Madingley implementation, assuming a power-law relationship with
    mass and an exponential relationship with temperature.

    TODO: Implement activity windows to properly paramterize sigma.
    TODO: Move constants to constants file after constants rework.

    Args:
        mass: The body-mass [kg] of an AnimalCohort.
        temperature: The temperature [Celsius] of the environment.
        terms: The tuple of metabolic rate terms used.
        metabolic_type: The metabolic type of the animal [ENDOTHERMIC or ECTOTHERMIC].

    Returns:
        The metabolic rate of an individual of the given cohort in [g/d].
    """

    Es = 3.7 * 10 ** (-2)  # energy to mass conversion constant (g/kJ)
    sig = 0.5  # proportion of time-step with temp in active range (toy)
    Ea = 0.69  # aggregate activation energy of metabolic reactions
    kB = BOLTZMANN_CONSTANT
    mass_g = mass * 1000  # convert mass to grams

    if metabolic_type == MetabolicType.ENDOTHERMIC:
        Ib, bf = terms["basal"]  # field metabolic constant and exponent
        If, bb = terms["field"]  # basal metabolic constant and exponent
        Tk = 310.0  # body temperature of the individual (K)
        return (
            Es
            * (
                (sig * If * exp(-(Ea / (kB * Tk)))) * mass_g**bf
                + ((1 - sig) * Ib * exp(-(Ea / (kB * Tk)))) * mass_g**bb
            )
            / 1000  # convert back to kg
        )
    elif metabolic_type == MetabolicType.ECTOTHERMIC:
        Ib, bf = terms["basal"]  # field metabolic constant and exponent
        If, bb = terms["field"]  # basal metabolic constant and exponent
        Tk = temperature + 274.15  # body temperature of the individual (K)
        return (
            Es
            * (
                (sig * If * exp(-(Ea / (kB * Tk)))) * mass_g**bf
                + ((1 - sig) * Ib * exp(-(Ea / (kB * Tk)))) * mass_g**bb
            )
            / 1000  # convert back to kg
        )
    else:
        raise ValueError("Invalid metabolic type: {metabolic_type}")


def muscle_mass_scaling(mass: float, terms: tuple) -> float:
    """The function to set the amount of muscle mass on individual in an AnimalCohort.

    Currently, this scaling relationship is only accurate for terrestrial mammals.
        This will later be updated for additional functional types.

    Args:
        mass: The body-mass [kg] of an AnimalCohort.
        terms: The tuple of muscle scaling terms used.

    Returns:
        The mass [g] of muscle on an individual of the animal cohort.

    """

    return terms[1] * (mass * 1000) ** terms[0]


def fat_mass_scaling(mass: float, terms: tuple) -> float:
    """The function to set the amount of fat mass on individual in an AnimalCohort.

        Currently, this scaling relationship is only accurate for terrestrial mammals.
        This will later be updated for additional functional types.

    Args:
        mass: The body-mass [kg] of an AnimalCohort.
        terms: The tuple of fat scaling terms used.

    Returns:
        The mass [g] of fat on an individual of the animal cohort.

    """

    return terms[1] * (mass * 1000) ** terms[0]


def energetic_reserve_scaling(
    mass: float, muscle_terms: tuple, fat_terms: tuple
) -> float:
    """The function to set the energetic reserve of an individual in an AnimalCohort.

        Currently, this scaling relationship is only accurate for terrestrial mammals.
        This will later be updated for additional functional types.

    Args:
        mass: The body-mass [kg] of an AnimalCohort.
        muscle_terms: The tuple of muscle scaling terms used.
        fat_terms: The tuple of fat scaling terms used.

    Returns:
        The energetic reserve [J] of  an individual of the animal cohort.

    """
    return (
        muscle_mass_scaling(mass, muscle_terms) + fat_mass_scaling(mass, fat_terms)
    ) * 7000.0  # j/g


def intake_rate_scaling(mass: float, terms: tuple) -> float:
    """The function to set the intake rate of an individual in an AnimalCohort.

        Currently, this scaling relationship is only accurate for terrestrial
        herbivorous mammals interacting with plant foods. This will later be updated
        for additional functional types and interactions.

        The function form converts the original g/min rate into a kg/day rate, where a
        day is an 8hr foraging window.

    Args:
        mass: The body-mass [kg] of an AnimalCohort.
        terms: The tuple of intake rate terms used.

    Returns:
        The intake rate [kg/day] of an individual of the animal cohort.

    """

    return terms[1] * mass ** terms[0] * 480 * (1 / 1000)


def prey_group_selection(
    diet_type: DietType, mass: float, terms: tuple
) -> dict[str, tuple[float, float]]:
    """The function to set the type selection and mass scaling of predators.

    Currently, this function is in a toy form. It exists so the forage_community
    structure can be built properly. In the parameterization stage of development this
    will be expanded into something realistic. I suspect some/much of the content will
    be shifted into functional_group definitions.

    TODO: Implement real pred-prey mass ratio.

    Args:
        mass: The body-mass [kg] of an AnimalCohort
        terms: The tuple of predator-prey scaling terms used.

    Returns:
        The dictionary of functional group names and mass ranges that the predator
        can prey upon.

    """

    if diet_type == DietType.HERBIVORE:
        return {"plants": (0.0, 0.0)}
    elif diet_type == DietType.CARNIVORE:
        return {
            "herbivorous_mammal": (0.1, 1000.0),
            "carnivorous_mammal": (0.1, 1000.0),
            "herbivorous_bird": (0.1, 1000.0),
            "carnivorous_bird": (0.1, 1000.0),
            "herbivorous_insect": (0.1, 1000.0),
            "carnivorous_insect": (0.1, 1000.0),
        }
    else:
        raise ValueError("Invalid diet type: {diet_type}")


def natural_mortality_scaling(mass: float, terms: tuple) -> float:
    """The function to determine the natural mortality rate of animal cohorts.

    Relationship from: Dureuil & Froese 2021

    M = - ln(P) / tmax   (annual, year^-1, instantaneous rate)
    tmax = mean maximum age
    P = 0.015 # proportion surviving to tmax

    Transform yearly rate to daily rate
    transform daily rate to daily probability
    prob = 1 - e^-M

    Args:
        mass: The body-mass [kg] of an AnimalCohort.

    Returns:
        The allometric natural mortality rate as a daily probability of death.

    """
    tmax = terms[1] * mass ** terms[0]
    annual_mortality_rate = -log(0.015) / tmax
    daily_mortality_rate = annual_mortality_rate / 365.0
    daily_mortality_prob = 1 - exp(-daily_mortality_rate)

    return daily_mortality_prob


def alpha_i_k(alpha_0_herb: float, mass: float) -> float:
    """Effective rate at which an individual herbivore searches its environment.

    This is linear scaling of herbivore search times with current body mass.

    Madingley

    Args:
        alpha_0_herb: Effective rate per unit body mass at which a herbivore searches
          its environment.
        mass: The current body mass of the foraging herbivore.

    Returns:
        A float of the effective search rate in [ha/day]

    """

    return alpha_0_herb * mass


def k_i_k(alpha_i_k: float, phi_herb_t: float, B_k_t: float, A_cell: float) -> float:
    """The potential biomass (g) of plant k eaten by cohort i, per day.

    Madingley

    Args:
        alpha_i_k: Effective rate at which an individual herbivore searches its
          environment.
        phi_herb_t: Fraction of the total plant stock that is available to any one
          herbivore cohort
        B_k_t: Plant resource bool biomass.
        A_cell: The area of one cell [standard = 1 ha]

    Returns:
        A float of The potential biomass (g) of plant k eating by cohort i, per day
        [g/day]

    """

    return alpha_i_k * ((phi_herb_t * B_k_t) / A_cell) ** 2


def H_i_k(h_herb_0: float, M_ref: float, M_i_t: float, b_herb: float) -> float:
    """Handling time of plant resource k by cohort i.

    Time (days) for an individual of cohort i to handle 1 gram of plant resource.

    Madingley

    Args:
        h_herb_0: Time in days that it would take a herbivore of mass = M_ref to handle
          1g of autotroph mass.
        M_ref: Reference body mass.
        M_i_t: Current herbivore mass
        b_herb: Exponent of the power-law function relating the handling time of
          autotroph matter to herbivore mass

    Returns:
        A float of the handling time (days).

    """

    return h_herb_0 * (M_ref / M_i_t) ** b_herb


def theta_opt_i(
    theta_opt_min_f: float, theta_opt_f: float, sigma_opt_f: float
) -> float:
    """Optimum predator-prey mass ratio.

    Madingley

    Args:
        theta_opt_min_f: The minimum optimal prey-predator body mass ratio.
        theta_opt_f: The mean optimal prey-predator body mass ratio, from which actual
          cohort optima are drawn.
        sigma_opt_f: The standard deviation of optimal predator-prey mass ratios among
          cohorts.

    Returns:
        A float measure of the optimum ratio.

    """

    return max(theta_opt_min_f, np.random.normal(theta_opt_f, sigma_opt_f))


def w_bar_i_j(
    mass_predator: float,
    mass_prey: float,
    theta_opt_i: float,
    sigma_opt_pred_prey: float,
) -> float:
    """The probability of successfully capturing a prey item.

    Madingley

    Args:
        mass_predator: Current mass of the predator..
        mass_prey: Current mass of the prey.
        theta_opt_i: The optimum predator-prey mass ratio.
        sigma_opt_pred_prey: The standard deviation of the mass ration.

    Returns:
        A float probability [0.0-1.0] that a predation encounter is successful.

    """

    return exp(
        -(
            ((log(mass_prey / mass_predator) - log(theta_opt_i)) / sigma_opt_pred_prey)
            ** 2
        )
    )


def alpha_i_j(alpha_0_pred: float, mass: float, w_bar_i_j: float) -> float:
    """Rate at which an individual predator searches its environment and kills prey.

    This is linear scaling of herbivore search times with current body mass.

    Madingley


    Args:
        alpha_0_pred: Constant describing effective rate per unit body mass at which any
          predator searches its environment in ha/(day*g).
        mass: The current body mass of the foraging herbivore.
        w_bar_i_j: The probability of successfully capturing a prey item.

    Returns:
        A float of the effective search rate in [ha/day]

    """

    return alpha_0_pred * mass * w_bar_i_j


def k_i_j(alpha_i_j: float, N_i_t: float, A_cell: float, theta_i_j: float) -> float:
    """Potential number of prey items eaten off j by i.

    TODO: Finish docstring
    TODO: double check output needs to be float, might be int

    Madingley

    Args:
        alpha_i_j: Rate at which an individual predator searches its environment and
          kills prey.
        N_i_t: Number of consumer individuals.
        A_cell: The area of a grid cell.
        theta_i_j: The cumulative density of organisms with a mass lying within the
              same predator specific mass bin.

    Returns:
        Potential number of prey items eaten off j by i [integer number of individuals]


    """

    return alpha_i_j * (N_i_t / A_cell) * theta_i_j


def H_i_j(h_pred_0: float, M_ref: float, M_i_t: float, b_pred: float) -> float:
    """Handling time of prey cohort j by cohort i.

    Time (days) for an individual of cohort i to handle 1 individual of cohort j.

    Madingley

    Args:
        h_pred_0: Time that it would take a predator of body mass equal to the reference
          mass, to handle a prey individual of body mass equal to one gram.
        M_ref: Reference body mass.
        M_i_t: Current predator mass
        b_pred: Exponent of the power-law function relating the handling time of
          prey to predator mass

    Returns:
        A float of the handling time (days).

    """

    return h_pred_0 * ((M_ref / M_i_t) ** b_pred) * M_i_t
