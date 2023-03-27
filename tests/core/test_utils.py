"""Testing the utility functions."""

from contextlib import nullcontext as does_not_raise
from logging import CRITICAL, ERROR
from pathlib import Path

import pytest
from numpy import datetime64, timedelta64

from tests.conftest import log_check
from virtual_rainforest.core.exceptions import ConfigurationError, InitialisationError


@pytest.mark.parametrize(
    argnames=["config", "raises", "timestep", "initial_time", "expected_log"],
    argvalues=[
        (
            {
                "core": {"timing": {"start_date": "2020-01-01"}},
                "soil": {"model_time_step": "12 hours"},
            },
            does_not_raise(),
            timedelta64(720, "m"),
            datetime64("2020-01-01"),
            (),
        ),
        (
            {
                "core": {"timing": {"start_date": "2020-01-01"}},
                "soil": {"model_time_step": "12 interminable hours"},
            },
            pytest.raises(InitialisationError),
            None,
            None,
            (
                (
                    ERROR,
                    "Model timing error: 'interminable' is not defined in the unit "
                    "registry",
                ),
            ),
        ),
        (
            {
                "core": {"timing": {"start_date": "2020-01-01"}},
                "soil": {"model_time_step": "12 kilograms"},
            },
            pytest.raises(InitialisationError),
            None,
            None,
            (
                (
                    ERROR,
                    "Model timing error: Cannot convert from 'kilogram' ([mass]) to "
                    "'second' ([time])",
                ),
            ),
        ),
    ],
)
def test_extract_model_time_details(
    caplog, config, raises, timestep, initial_time, expected_log
):
    """Tests timing details extraction utility."""

    from virtual_rainforest.core.utils import extract_model_time_details

    with raises:
        start_time, update_interval = extract_model_time_details(config, "soil")
        assert start_time == initial_time
        assert update_interval == timestep

    log_check(caplog, expected_log)


@pytest.mark.parametrize(
    "out_path,expected_log_entries",
    [
        (
            "./complete_config.toml",
            (
                (
                    CRITICAL,
                    "A file in the user specified output folder (.) already makes use "
                    "of the specified output file name (complete_config.toml), this "
                    "file should either be renamed or deleted!",
                ),
            ),
        ),
        (
            "bad_folder/complete_config.toml",
            (
                (
                    CRITICAL,
                    "The user specified output directory (bad_folder) doesn't exist!",
                ),
            ),
        ),
        (
            "pyproject.toml/complete_config.toml",
            (
                (
                    CRITICAL,
                    "The user specified output folder (pyproject.toml) isn't a "
                    "directory!",
                ),
            ),
        ),
    ],
)
def test_check_outfile(caplog, mocker, out_path, expected_log_entries):
    """Check that an error is logged if an output file is already saved."""
    from virtual_rainforest.core.utils import check_outfile

    # Configure the mock to return a specific list of files
    if out_path == "./complete_config.toml":
        mock_content = mocker.patch("virtual_rainforest.core.config.Path.exists")
        mock_content.return_value = True

    # Check that check_outfile fails as expected
    with pytest.raises(ConfigurationError):
        check_outfile(Path(out_path))

    log_check(caplog, expected_log_entries)
