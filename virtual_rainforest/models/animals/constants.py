"""The `models.animals.constants` module contains a set of dictionaries containing
constants" (fitting relationships taken from the literature) required by the broader
:mod:`~virtual_rainforest.models.animals` module

The near-future intention is to rework the relationship between these constants and the
AnimalCohort objects in which they are used such that there is a FunctionalType class
in-between them. This class will hold the specific scaling, rate, and conversion
parameters required for determining the function of a specific AnimalCohort and will
avoid frequent searches through this constants file for values.
"""  # noqa: D205, D415

from virtual_rainforest.models.animals.animal_traits import (
    DietType,
    MetabolicType,
    TaxaType,
)

METABOLIC_RATE_TERMS_ENERGY: dict[
    MetabolicType, dict[TaxaType, tuple[float, float]]
] = {
    MetabolicType.ENDOTHERMIC: {
        TaxaType.MAMMAL: (0.75, 0.018),
        # [assumes kg mass]
        TaxaType.BIRD: (0.75, 0.05),
        # Toy values.
    },
    MetabolicType.ECTOTHERMIC: {
        TaxaType.INSECT: (0.75, 0.08)
        # Toy values.
    },
}

METABOLIC_RATE_TERMS: dict[MetabolicType, dict[str, tuple[float, float]]] = {
    # Parameters from Madingley, mass based metabolic rates
    MetabolicType.ENDOTHERMIC: {
        "basal": (4.19 * 10**10, 0.69),
        "field": (9.08 * 10**11, 0.7),
    },
    MetabolicType.ECTOTHERMIC: {
        "basal": (4.19 * 10**10, 0.69),
        "field": (1.49 * 10**11, 0.88),
    },
}

DAMUTHS_LAW_TERMS: dict[TaxaType, dict[DietType, tuple[float, float]]] = {
    TaxaType.MAMMAL: {
        DietType.HERBIVORE: (-0.75, 4.23),
        # Mammalian herbivore population density, observed allometry (Damuth 1987).
        # [assumes kg mass]
        DietType.CARNIVORE: (-0.75, 1.00),
        # Toy values.
    },
    TaxaType.BIRD: {
        DietType.HERBIVORE: (-0.75, 5.00),
        # Toy values.
        DietType.CARNIVORE: (-0.75, 2.00),
        # Toy values.
    },
    TaxaType.INSECT: {
        DietType.HERBIVORE: (-0.75, 5.00),
        # Toy values.
        DietType.CARNIVORE: (-0.75, 2.00),
        # Toy values.
    },
}

FAT_MASS_TERMS: dict[TaxaType, tuple[float, float]] = {
    TaxaType.MAMMAL: (1.19, 0.02),
    # Scaling of mammalian herbivore fat mass (citation from Rallings). [assumes g mass]
    TaxaType.BIRD: (1.19, 0.05),
    # Toy Values
    TaxaType.INSECT: (1.19, 0.05),
    # Toy Values
}

MUSCLE_MASS_TERMS: dict[TaxaType, tuple[float, float]] = {
    TaxaType.MAMMAL: (1.0, 0.38),
    # Scaling of mammalian herbivore muscle mass (citation from Rallings).
    # [assumes g mass]
    TaxaType.BIRD: (1.0, 0.40),
    # Toy Values
    TaxaType.INSECT: (1.0, 0.40),
    # Toy Values
}

INTAKE_RATE_TERMS: dict[TaxaType, tuple[float, float]] = {
    TaxaType.MAMMAL: (0.71, 0.63),
    # Mammalian maximum intake rate (g/min) from (Shipley 1994). [assumes kg mass]
    TaxaType.BIRD: (0.7, 0.50),
    # Toy Values
    TaxaType.INSECT: (0.7, 0.50),
    # Toy Values
}


ENERGY_DENSITY: dict[str, float] = {
    "meat": 7000.0,
    # The energy of a unit mass of mammal meat (check citation from Rallings). [J/g]
    "plant": 18200000.0
    # Temporary realistic plant food value: Alfalfa ¬ 18,200,000 J/kg DM.
}

CONVERSION_EFFICIENCY: dict[DietType, float] = {
    DietType.HERBIVORE: 0.1,
    # Toy value [unitless].
    DietType.CARNIVORE: 0.25,
    # Toy value [unitless].
}

MECHANICAL_EFFICIENCY: dict[DietType, float] = {
    DietType.HERBIVORE: 0.9,
    # Toy value [unitless].
    DietType.CARNIVORE: 0.8,
    # Toy Value [unitless]
}

PREY_MASS_SCALING_TERMS: dict[MetabolicType, dict[TaxaType, tuple[float, float]]] = {
    MetabolicType.ENDOTHERMIC: {
        TaxaType.MAMMAL: (1.0, 1.0),
        # Toy values.
        TaxaType.BIRD: (1.0, 1.0),
        # Toy values.
    },
    MetabolicType.ECTOTHERMIC: {
        TaxaType.INSECT: (1.0, 1.0)
        # Toy values.
    },
}

LONGEVITY_SCALING_TERMS: dict[TaxaType, tuple[float, float]] = {
    TaxaType.MAMMAL: (0.25, 0.02),
    # Toy values
    TaxaType.BIRD: (0.25, 0.05),
    # Toy Values
    TaxaType.INSECT: (0.25, 0.05),
    # Toy Values
}

BOLTZMANN_CONSTANT: float = 8.617333262145e-5  # Boltzmann constant [eV/K]

TEMPERATURE: float = 37.0  # Toy temperature for setting up metabolism [C].

REPRODUCTIVE_MASS_THRESHOLD: float = 1.5  # Toy value for thresholding reproduction.

DISPERSAL_MASS_THRESHOLD: float = 0.75  # Toy value for thesholding dispersal.

ENERGY_PERCENTILE_THRESHOLD: float = 0.5  # Toy value for initiating migration

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
