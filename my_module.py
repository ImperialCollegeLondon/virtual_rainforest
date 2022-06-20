"""Training module for doctest and pytest.

This module contains some example functions and classes that need to have
doctests and pytests added.
"""

# flake8: noqa D202, D107

def exponentiate_this(b: float, n: float):
    """Calculate a power
    
    Returns b to the power ngit

    Args:
        b: The base
        n: The exponent

    TODO: add 3 doctests to show:
        * basic usage using rounding
        * basic usage using ellipsis
        * failure with non float inputs
    """

    if not (isinstance(b, float) and isinstance(n, float)):
        raise ValueError('Both x and y must be of type float')
    
    return b ** n


class