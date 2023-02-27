"""Test module for main.py (and associated functionality).

This module tests both the main simulation function `vr_run` and the other functions
defined in main.py that it calls.
"""

from contextlib import nullcontext as does_not_raise
from logging import CRITICAL, ERROR, INFO, WARNING
from pathlib import Path

import pytest
from numpy import datetime64, timedelta64

from virtual_rainforest.core.base_model import BaseModel, InitialisationError
from virtual_rainforest.main import (
    check_for_fast_models,
    configure_models,
    extract_timing_details,
    select_models,
    vr_run,
)
from virtual_rainforest.models.soil.soil_model import SoilModel

from .conftest import log_check


@pytest.mark.parametrize(
    "model_list,no_models,raises,expected_log_entries",
    [
        pytest.param(
            ["soil"],  # valid input
            1,
            does_not_raise(),
            (
                (
                    INFO,
                    "Attempting to configure the following models: ['soil']",
                ),
            ),
            id="valid input",
        ),
        pytest.param(
            ["soil", "core"],
            1,
            does_not_raise(),
            (
                (
                    INFO,
                    "Attempting to configure the following models: ['soil']",
                ),
            ),
            id="ignores core",
        ),
        pytest.param(
            ["soil", "freshwater"],  # Model that hasn't been defined
            0,
            pytest.raises(InitialisationError),
            (
                (
                    INFO,
                    "Attempting to configure the following models: ['freshwater', "
                    "'soil']",
                ),
                (
                    CRITICAL,
                    "The following models cannot be configured as they are not found in"
                    " the registry: ['freshwater']",
                ),
            ),
            id="undefined model",
        ),
    ],
)
def test_select_models(caplog, model_list, no_models, raises, expected_log_entries):
    """Test the model selecting function."""

    with raises:
        models = select_models(model_list)
        assert len(models) == no_models
        assert all([type(model) == type(BaseModel) for model in models])

    log_check(caplog, expected_log_entries)


@pytest.mark.parametrize(
    "config,output,raises,expected_log_entries",
    [
        pytest.param(
            {  # valid config
                "soil": {"model_time_step": "7 days"},
                "core": {"timing": {"start_time": "2020-01-01"}},
            },
            "SoilModel(update_interval = 10080 minutes, next_update = 2020-01-08T00:00"
            ")",
            does_not_raise(),
            (
                (INFO, "Attempting to configure the following models: ['soil']"),
                (
                    INFO,
                    "Information required to initialise the soil model successfully "
                    "extracted.",
                ),
            ),
            id="valid config",
        ),
        pytest.param(
            {  # model_time_step missing units
                "soil": {"model_time_step": "7"},
                "core": {"timing": {}},
            },
            None,
            pytest.raises(InitialisationError),
            (
                (INFO, "Attempting to configure the following models: ['soil']"),
                (
                    ERROR,
                    "Configuration types appear not to have been properly validated. "
                    "This problem prevents initialisation of the soil model. The first "
                    "instance of this problem is as follows: Cannot convert from "
                    "'dimensionless' (dimensionless) to 'minute' ([time])",
                ),
                (
                    CRITICAL,
                    "Could not configure all the desired models, ending the "
                    "simulation.",
                ),
            ),
            id="model_time_step missing units",
        ),
    ],
)
def test_configure_models(
    caplog, data_instance, config, output, raises, expected_log_entries
):
    """Test the function that configures the models."""

    with raises:
        model_list = select_models(["soil"])

        models = configure_models(config, data_instance, model_list)

        if output is None:
            assert models == [None]
        else:
            assert repr(models["soil"]) == output

    log_check(caplog, expected_log_entries)


def test_vr_run_miss_model(mocker, caplog):
    """Test the main `vr_run` function handles missing models correctly."""

    mock_conf = mocker.patch("virtual_rainforest.main.validate_config")
    mock_conf.return_value = {"core": {"modules": ["topsoil"], "data": []}}

    with pytest.raises(InitialisationError):
        vr_run("path/does/not/need/to/exist", Path("./delete_me.toml"))
        # If vr_run is successful (which it shouldn't be) clean up the file
        Path("./delete_me.toml").unlink()

    expected_log_entries = (
        (INFO, "Loading data from configuration"),
        (INFO, "Attempting to configure the following models: ['topsoil']"),
        (
            CRITICAL,
            "The following models cannot be configured as they are not found in the "
            "registry: ['topsoil']",
        ),
    )

    log_check(caplog, expected_log_entries)


def test_vr_run_bad_model(mocker, caplog):
    """Test the main `vr_run` function handles bad model configuration correctly."""

    mock_conf = mocker.patch("virtual_rainforest.main.validate_config")
    mock_conf.return_value = {
        "core": {
            "modules": ["soil"],
            "timing": {
                "start_date": "2020-01-01",
                "end_date": "2120-01-01",
            },
            "data": [],
        },
        "soil": {
            "model_time_step": "0.5 martian days",
        },
    }

    with pytest.raises(InitialisationError):
        vr_run("path/does/not/need/to/exist", Path("./delete_me.toml"))
        # If vr_run is successful (which it shouldn't be) clean up the file
        Path("./delete_me.toml").unlink()

    expected_log_entries = (
        (INFO, "Loading data from configuration"),
        (INFO, "Attempting to configure the following models: ['soil']"),
        (
            INFO,
            "All models found in the registry, now attempting to configure them.",
        ),
        (
            ERROR,
            "Configuration types appear not to have been properly validated. This "
            "problem prevents initialisation of the soil model. The first instance of "
            "this problem is as follows: 'martian' is not defined in the unit registry",
        ),
        (
            CRITICAL,
            "Could not configure all the desired models, ending the simulation. The "
            "following models failed: ['soil'].",
        ),
    )

    log_check(caplog, expected_log_entries)


@pytest.mark.parametrize(
    "config,output,raises,expected_log_entries",
    [
        pytest.param(
            {
                "core": {
                    "timing": {
                        "start_date": "2020-01-01",
                        "update_interval": "10 minutes",
                        "run_length": "30 years",
                    }
                }
            },
            {
                "start_time": datetime64("2020-01-01"),
                "update_interval": timedelta64(10, "m"),
                "end_time": datetime64("2049-12-31T12:00"),
            },
            does_not_raise(),
            (
                (
                    INFO,
                    "Virtual Rainforest simulation will run from 2020-01-01 until 2049-"
                    "12-31T12:00. This is a run length of 15778800 minutes, the user "
                    "requested 15778800 minutes",
                ),
            ),
            id="timing correct",
        ),
        pytest.param(
            {
                "core": {
                    "timing": {
                        "start_date": "2020-01-01",
                        "update_interval": "10 minutes",
                        "run_length": "1 minute",
                    }
                }
            },
            {},  # Fails so no output to check
            pytest.raises(InitialisationError),
            (
                (
                    CRITICAL,
                    "Models will never update as the update interval (10 minutes) is "
                    "larger than the run length (1 minutes)",
                ),
            ),
            id="run length < update interval",
        ),
        pytest.param(
            {
                "core": {
                    "timing": {
                        "start_date": "2020-01-01",
                        "update_interval": "10 minutes",
                        "run_length": "7 short days",
                    }
                }
            },
            {},  # Fails so no output to check
            pytest.raises(InitialisationError),
            (
                (
                    CRITICAL,
                    "Units for core.timing.run_length are not valid time units: 7 short"
                    " days",
                ),
            ),
            id="invalid run length units",
        ),
        pytest.param(
            {
                "core": {
                    "timing": {
                        "start_date": "2020-01-01",
                        "update_interval": "10 long minutes",
                        "run_length": "30 years",
                    }
                }
            },
            {},  # Fails so no output to check
            pytest.raises(InitialisationError),
            (
                (
                    CRITICAL,
                    "Units for core.timing.update_interval are not valid time units: 10"
                    " long minutes",
                ),
            ),
            id="invalid update_interval units",
        ),
    ],
)
def test_extract_timing_details(caplog, config, output, raises, expected_log_entries):
    """Test that function to extract main loop timing works as intended."""

    with raises:
        current_time, update_interval, end_time = extract_timing_details(config)
        assert end_time == output["end_time"]
        assert update_interval == output["update_interval"]
        assert current_time == output["start_time"]

    log_check(caplog, expected_log_entries)


@pytest.mark.parametrize(
    "update_interval,expected_log_entries",
    [
        pytest.param(timedelta64(2, "W"), (), id="valid"),
        pytest.param(
            timedelta64(5, "W"),
            (
                (
                    WARNING,
                    "The following models have shorter time steps than the main model: "
                    "['soil']",
                ),
            ),
            id="fast model",
        ),
    ],
)
def test_check_for_fast_models(caplog, update_interval, expected_log_entries):
    """Test that function to warn user about short module time steps works."""

    # Create SoilModel instance and then populate the update_interval
    model = SoilModel.__new__(SoilModel)
    model.update_interval = timedelta64(3, "W")
    models_cfd = {"soil": model}

    check_for_fast_models(models_cfd, update_interval)

    log_check(caplog, expected_log_entries)
