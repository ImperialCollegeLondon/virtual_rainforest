"""Test module for soil.carbon.py.

This module tests the functionality of the soil carbon module
"""

from contextlib import nullcontext as does_not_raise
from logging import CRITICAL

import numpy as np
import pytest

from virtual_rainforest.core.model import InitialisationError
from virtual_rainforest.soil.carbon import SoilCarbon

from .conftest import log_check


@pytest.mark.parametrize(
    "maom,lmwc,raises,expected_log_entries",
    [
        (
            np.array([23.0, 12.0], dtype=np.float32),
            np.array([98.0, 7.0], dtype=np.float32),
            does_not_raise(),
            (),
        ),
        (
            np.array([23.0, 12.0], dtype=np.float32),
            np.array([98.0], dtype=np.float32),
            pytest.raises(InitialisationError),
            (
                (
                    CRITICAL,
                    "Dimension mismatch for initial carbon pools!",
                ),
            ),
        ),
        (
            np.array([23.0, 12.0], dtype=np.float32),
            np.array([98.0, -24.0], dtype=np.float32),
            pytest.raises(InitialisationError),
            (
                (
                    CRITICAL,
                    "Initial carbon pools contain at least one negative value!",
                ),
            ),
        ),
    ],
)
def test_soil_carbon_class(caplog, maom, lmwc, raises, expected_log_entries):
    """Test SoilCarbon class initialisation."""

    # Check that initialisation fails (or doesn't) as expected
    with raises:
        soil_carbon = SoilCarbon(maom, lmwc)

        assert (soil_carbon.maom == maom).all()
        assert (soil_carbon.lmwc == lmwc).all()

    log_check(caplog, expected_log_entries)


# TEST update_pools()

# TEST mineral_association()

# TEST THE TWO SCALAR FUNCTIONS
