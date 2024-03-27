"""The `models.animals.constants` module contains a set of dataclasses containing
constants" (fitting relationships taken from the literature) required by the broader
:mod:`~virtual_ecosystem.models.animals` module

The near-future intention is to rework the relationship between these constants and the
AnimalCohort objects in which they are used such that there is a FunctionalType class
in-between them. This class will hold the specific scaling, rate, and conversion
parameters required for determining the function of a specific AnimalCohort and will
avoid frequent searches through this constants file for values.
"""  # noqa: D205, D415

from dataclasses import dataclass, field

from virtual_ecosystem.core.constants_class import ConstantsDataclass
from virtual_ecosystem.models.animals.animal_traits import (
    DietType,
    MetabolicType,
    TaxaType,
)


@dataclass(frozen=True)
class AnimalConsts(ConstantsDataclass):
    """Dataclass to store all constants related to metabolic rates.

    TODO: The entire constants fille will be reworked in this style after the energy to
    mass conversion.

    """

    metabolic_rate_terms: dict[MetabolicType, dict[str, tuple[float, float]]] = field(
        default_factory=lambda: {
            # Parameters from Madingley, mass-based metabolic rates
            MetabolicType.ENDOTHERMIC: {
                "basal": (4.19e10, 0.69),
                "field": (9.08e11, 0.7),
            },
            MetabolicType.ECTOTHERMIC: {
                "basal": (4.19e10, 0.69),
                "field": (1.49e11, 0.88),
            },
        }
    )

    damuths_law_terms: dict[TaxaType, dict[DietType, tuple[float, float]]] = field(
        default_factory=lambda: {
            TaxaType.MAMMAL: {
                DietType.HERBIVORE: (-0.75, 4.23),
                DietType.CARNIVORE: (-0.75, 1.00),
            },
            TaxaType.BIRD: {
                DietType.HERBIVORE: (-0.75, 5.00),
                DietType.CARNIVORE: (-0.75, 2.00),
            },
            TaxaType.INSECT: {
                DietType.HERBIVORE: (-0.75, 5.00),
                DietType.CARNIVORE: (-0.75, 2.00),
            },
        }
    )

    fat_mass_terms: dict[TaxaType, tuple[float, float]] = field(
        default_factory=lambda: {
            TaxaType.MAMMAL: (1.19, 0.02),  # Scaling of mammalian herbivore fat mass
            TaxaType.BIRD: (1.19, 0.05),  # Toy Values
            TaxaType.INSECT: (1.19, 0.05),  # Toy Values
        }
    )

    muscle_mass_terms: dict[TaxaType, tuple[float, float]] = field(
        default_factory=lambda: {
            TaxaType.MAMMAL: (1.0, 0.38),  # Scaling of mammalian herbivore muscle mass
            TaxaType.BIRD: (1.0, 0.40),  # Toy Values
            TaxaType.INSECT: (1.0, 0.40),  # Toy Values
        }
    )

    intake_rate_terms: dict[TaxaType, tuple[float, float]] = field(
        default_factory=lambda: {
            TaxaType.MAMMAL: (0.71, 0.63),  # Mammalian maximum intake rate
            TaxaType.BIRD: (0.7, 0.50),  # Toy Values
            TaxaType.INSECT: (0.7, 0.50),  # Toy Values
        }
    )

    energy_density: dict[str, float] = field(
        default_factory=lambda: {
            "meat": 7000.0,  # Energy of mammal meat [J/g]
            "plant": 18200000.0,  # Energy of plant food [J/g]
        }
    )

    conversion_efficiency: dict[DietType, float] = field(
        default_factory=lambda: {
            DietType.HERBIVORE: 0.1,  # Toy value
            DietType.CARNIVORE: 0.25,  # Toy value
        }
    )

    mechanical_efficiency: dict[DietType, float] = field(
        default_factory=lambda: {
            DietType.HERBIVORE: 0.9,  # Toy value
            DietType.CARNIVORE: 0.8,  # Toy value
        }
    )

    prey_mass_scaling_terms: dict[
        MetabolicType, dict[TaxaType, tuple[float, float]]
    ] = field(
        default_factory=lambda: {
            MetabolicType.ENDOTHERMIC: {
                TaxaType.MAMMAL: (1.0, 1.0),  # Toy values
                TaxaType.BIRD: (1.0, 1.0),  # Toy values
            },
            MetabolicType.ECTOTHERMIC: {TaxaType.INSECT: (1.0, 1.0)},  # Toy values
        }
    )

    longevity_scaling_terms: dict[TaxaType, tuple[float, float]] = field(
        default_factory=lambda: {
            TaxaType.MAMMAL: (0.25, 0.02),  # Toy values
            TaxaType.BIRD: (0.25, 0.05),  # Toy values
            TaxaType.INSECT: (0.25, 0.05),  # Toy values
        }
    )

    birth_mass_threshold: float = 1.5  # Threshold for reproduction
    flow_to_reproductive_mass_threshold: float = (
        1.0  # Threshold of trophic flow to reproductive mass
    )
    dispersal_mass_threshold: float = 0.75  # Threshold for dispersal
    energy_percentile_threshold: float = 0.5  # Threshold for initiating migration
    decay_fraction_excrement: float = 0.5  # Decay fraction for excrement
    decay_fraction_carcasses: float = 0.2  # Decay fraction for carcasses

    # Madingley Foraging Parameters

    tau_f = 0.5  # tau_f
    """Proportion of time for which functional group is active."""
    sigma_f_t = 0.5  # sigma_f(t) - TODO: find real value
    """Proportion of the time step in which it's suitable to be active for functional
    group f."""

    # Trophic paramters

    alpha_0_herb = 1.0e-11  # alpha_herb_0 [Madingley] ha/(day*g)
    """Effective rate per unit mass at which a herbivore searches its environment."""
    alpha_0_pred = 1.0e-6  # alpha_pred_0 [Madingley] ha/(day*g)
    """Effective rate per unit mass at which a predator searches its environment."""

    phi_herb_t = 0.1  # phi_herb_t
    """Fraction of the resource stock that is available to any one herbivore cohort."""

    b_herb = 0.7  # ( ),b_herb)
    """Herbivore exponent of the power-law function relating the handling time of
      autotroph matter to herbivore mass."""

    b_pred = 0.05  # Toy Values
    """Carnivore exponent of the power-law relationship between the handling time of
      prey and the ratio of prey to predator body mass."""

    M_herb_ref = 1.0  # M_herb_ref [Madingley] g
    """Reference mass for herbivore handling time."""
    M_herb_0 = 0.7  # M_herb_0 [Madingley] (days)
    """Time that it would take a herbivore of body mass equal to the reference mass,
    to handle one gram of autotroph biomass."""
    h_herb_0 = 0.7  # h_pred_0 [Madingley]
    """Time that it would take a herbivore of body mass equal to the reference mass,
    to handle one gram of autotroph biomass"""

    M_pred_ref = 1.0  # toy value TODO: find real value
    """The reference value for predator mass."""
    sigma_opt_pred_prey = 0.7  # sigma_opt_pred-prey [Madingley]
    """Standard deviation of the normal distribution describing realized attack rates
    around the optimal predator-prey body mass ratio."""
    theta_opt_min_f = 0.01  # theta_opt_min_f [Madingley]
    """The minimum optimal prey-predator body mass ratio."""
    theta_opt_f = 0.1  # theta_opt_f [Madingley]
    """The mean optimal prey-predator body mass ratio, from which actual cohort optima
    are drawn."""
    sigma_opt_f = 0.02  # sigma_opt_f [Madingley]
    """The standard deviation of optimal predator-prey mass ratios among cohorts."""
    N_sigma_opt_pred_prey = 3.0  # N_sigma_opt_pred-prey [Madingley]
    """The standard deviations of the realized attack rates around the optimal
    predator-prey body mass ratio for which to calculate predator specific cumulative
    prey densities."""
    h_pred_0 = 0.5  # h_pred_0 [Madingley]
    """Time that it would take a predator of body mass equal to the reference mass,
    to handle a prey individual of body mass equal to one gram."""

    # Activity parameters
    m_tol = 1.6  # m_tol_terrestrial [Madingley]
    """Slope of the relationship between monthly temperature variability and the upper
    critical temperature limit relative to annual mean temperature, for terrestrial
    ectothermic functional groups."""

    c_tol = 6.61  # c_tol_terrestrial [Madingley] (degrees C)
    """Intercept of the relationship between monthly temperature variability and the
    upper critical temperature limit relative to annual mean temperature, for
    terrestrial ectothermic functional groups."""

    m_tsm = 1.53  # m_tsm [Madingley]
    """Slope of the relationship between monthly temperature variability and the optimal
    temperature relative to annual mean temperature, for terrestrial ectothermic
    functional groups."""

    c_tsm = 1.51  # c_tsm [Madingley] (degrees C)
    """Intercept of the relationship between monthly temperature variability and the
    optimal temperature relative to annual mean temperature, for terrestrial
    ectothermic functional groups."""


DECAY_FRACTION_EXCREMENT: float = 0.5
"""Fraction of excrement that is assumed to decay rather than be consumed [unitless].

TODO - The number given here is very much made up. In future, we either need to find a
way of estimating this from data, or come up with a smarter way of handling this
process.
"""

DECAY_FRACTION_CARCASSES: float = 0.2
"""Fraction of carcass biomass that is assumed to decay rather than be consumed.

[unitless]. TODO - The number given here is very much made up, see
:attr:`DECAY_FRACTION_EXCREMENT` for details of how this should be changed in future.
"""
BOLTZMANN_CONSTANT: float = 8.617333262145e-5  # Boltzmann constant [eV/K]

TEMPERATURE: float = 37.0  # Toy temperature for setting up metabolism [C].
