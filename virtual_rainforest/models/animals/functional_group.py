"""The `models.animals.functional_group` module contains a class that organizes
constants and rate equations used by AnimalCohorts in the
:mod:`~virtual_rainforest.models.animals` module.
"""  # noqa: D205, D415

import csv

from virtual_rainforest.models.animals.constants import (
    CONVERSION_EFFICIENCY,
    DAMUTHS_LAW_TERMS,
    ECTOTHERMIC_METABOLIC_RATE_TERMS,
    ENDOTHERMIC_METABOLIC_RATE_TERMS,
    FAT_MASS_TERMS,
    INTAKE_RATE_TERMS,
    MUSCLE_MASS_TERMS,
)


class FunctionalGroup:
    """This is a class of animal functional groups.

    The goal of this class is to collect the correct constants and scaling relationships
    needed by an animal cohort such that they are accessed at initialization and stored
    in the AnimalCohort object as attributes. This should result in a system where an
    animal cohort can be auto-generated with a few keywords and numbers but that this
    procedure only need run once, at initialization, and that all further references to
    constants and scaling relationships are accessed through attributes of the
    AnimalCohort in question.

    """

    def __init__(self, name: str, taxa: str, diet: str, metabolic_type: str) -> None:
        """The constructor for the FunctionalGroup class."""
        # Check for valid inputs
        valid_taxa = ["mammal", "bird", "insect"]
        if taxa not in valid_taxa:
            raise ValueError(f"Invalid taxa: {taxa}. Valid options are: {valid_taxa}")

        valid_diets = ["herbivore", "carnivore"]
        if diet not in valid_diets:
            raise ValueError(f"Invalid diet: {diet}. Valid options are: {valid_diets}")

        valid_metabolic_types = ["endothermic", "ectothermic"]
        if metabolic_type not in valid_metabolic_types:
            raise ValueError(
                "Invalid metabolic type: "
                f"{metabolic_type}. Valid options are: "
                f"{valid_metabolic_types}"
            )

        self.name = name
        """The name of the functional group."""
        self.taxa = taxa
        """The taxa of the functional group."""
        self.diet = diet
        """The diet of the functional group."""
        self.metabolic_type = metabolic_type
        """The metabolic type of the functional group"""
        self.metabolic_rate_terms = (
            ENDOTHERMIC_METABOLIC_RATE_TERMS[taxa]
            if metabolic_type == "endothermic"
            else ECTOTHERMIC_METABOLIC_RATE_TERMS[taxa]
        )
        """The coefficient and exponent of metabolic rate."""
        self.damuths_law_terms = DAMUTHS_LAW_TERMS[taxa][diet]
        """The coefficient and exponent of damuth's law for population density."""
        self.muscle_mass_terms = MUSCLE_MASS_TERMS[taxa]
        """The coefficient and exponent of muscle mass allometry."""
        self.fat_mass_terms = FAT_MASS_TERMS[taxa]
        """The coefficient and exponent of fat mass allometry."""
        self.intake_rate_terms = INTAKE_RATE_TERMS[taxa]
        """The coefficient and exponent of intake allometry."""
        self.conversion_efficiency = CONVERSION_EFFICIENCY[diet]
        """The conversion efficiency of the functional group based on diet."""


def import_functional_groups(fg_file: str) -> list[FunctionalGroup]:
    """The function to import pre-defined functional groups.

    This function is a first-pass of how we might import pre-defined functional groups.
    The current expected csv structure is "name", "taxa", "diet" - the specific options
    of which can be found in functional_group.py. This allows a user to set out a basic
    outline of functional groups that accept our definitions of parameters and scaling
    relationships based on those traits.

    We will need a structure for users changing those underlying definitions but that
    can be constructed later.

    Args:
        csv_file: The location of the csv file holding the functional group definitions.

    Returns:
        A list of the FunctionalGroup instances created by the import.

    """
    functional_group_list: list[FunctionalGroup] = []

    with open(fg_file, newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)  # get the header

        # Check that the header has the expected columns
        expected_header = ["name", "taxa", "diet", "metabolic_type"]
        if header != expected_header:
            raise ValueError(
                f"Invalid header. Expected {expected_header}, but got {header}"
            )

        for row in reader:
            # Check that the row has the correct number of values
            # This is important to ensure each value can be properly assigned
            # to name, taxa, diet, and metabolic_type when unpacking the row
            if len(row) != len(expected_header):
                raise ValueError(
                    f"Invalid row: {row}. Expected {len(expected_header)} values."
                )

            name, taxa, diet, metabolic_type = row
            # create the FG instance and append it to the list
            # It's expected that the FunctionalGroup __init__ method
            # will handle further error checking for the values
            functional_group_list.append(
                FunctionalGroup(name, taxa, diet, metabolic_type)
            )

    return functional_group_list
