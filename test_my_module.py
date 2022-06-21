import pytest

# flake8: noqa D103


def test_exponentiate_this_raises():
    # TODO: check two strings raises an Exception

    pass


def test_exponentiate_this():
    # TODO: check two floats return an expected value

    pass


def test_exponentiate_this_param():
    # TODO: use a single parameterisation to check combinations of input signs
    #       in the b and n inputs

    pass


def test_make_full_name_string_raises():
    # TODO: check two floats raises an exception

    pass


def test_make_full_name_string():
    # TODO: check two strings returns the expected value
    
    pass


def make_full_name_string_expected():
    # TODO: make this into a fixture containing expectations for pairwise
    #       combinations of two first and two last names.

    pass


def test_make_full_name_string_twoparam():
    # TODO: add separate parameters for first and last name, each providing two
    # alternative inputs, and use the make_full_name_string_expected fixture to
    # check the results are as expected.
    
    pass


def test_person_init_exception():
    # TODO: Check that non-string inputs to init a Person raises an Exception

    pass


def test_person_init():
    # TODO: Create a person instance and check that the class attributes and the
    #       __repr__ method - via str(instance_name) or repr(instance_name) -
    #       work as expected.

    pass


def person_fixture():
    # TODO: Make this a fixture returning a Person instance to use in the
    #       test_greet tests below.

    pass

def test_person_greet_exceptions():
    # TODO: Make this a parameterised test using person_fixture to check the
    #       three possible exceptions that can be raised when running
    #       Person.greet(). You can have pytest.raises() in your parameters!

    pass


def person_greet_expected():
    # TODO: Make this a fixture containing expectations to use in
    #       test_person_greet 

    pass


def test_person_greet():
    # TODO: Use person_fixture and two parameter blocks to test pairwise
    #       combinations of all valid languages and formalities.

    pass
