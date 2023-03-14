"""The `models.animals.functional_group` module contains a class that organizes
constants and rate equations used by AnimalCohorts in the
:mod:`~virtual_rainforest.models.animals` module
"""  # noqa: D205, D415

# from virtual_rainforest.models.animals.constants


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

    def __init__(self) -> None:
        pass
