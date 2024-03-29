"""Test module for litter_model.py."""

from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from logging import CRITICAL, DEBUG, ERROR, INFO

import numpy as np
import pytest
from xarray import DataArray

from tests.conftest import log_check
from virtual_ecosystem.core.exceptions import ConfigurationError, InitialisationError


def test_litter_model_initialization(
    caplog, dummy_litter_data, fixture_core_components
):
    """Test `LitterModel` initialization."""
    from virtual_ecosystem.core.base_model import BaseModel
    from virtual_ecosystem.models.litter.constants import LitterConsts
    from virtual_ecosystem.models.litter.litter_model import LitterModel

    model = LitterModel(
        data=dummy_litter_data,
        core_components=fixture_core_components,
        model_constants=LitterConsts(),
    )

    # In cases where it passes then checks that the object has the right properties
    assert isinstance(model, BaseModel)
    assert model.model_name == "litter"
    assert str(model) == "A litter model instance"
    assert repr(model) == "LitterModel(update_interval=1209600 seconds)"

    # Final check that expected logging entries are produced
    log_check(
        caplog,
        expected_log=(
            (DEBUG, "litter model: required var 'litter_pool_above_metabolic' checked"),
            (
                DEBUG,
                "litter model: required var 'litter_pool_above_structural' checked",
            ),
            (DEBUG, "litter model: required var 'litter_pool_woody' checked"),
            (DEBUG, "litter model: required var 'litter_pool_below_metabolic' checked"),
            (
                DEBUG,
                "litter model: required var 'litter_pool_below_structural' checked",
            ),
            (DEBUG, "litter model: required var 'lignin_above_structural' checked"),
            (DEBUG, "litter model: required var 'lignin_woody' checked"),
            (DEBUG, "litter model: required var 'lignin_below_structural' checked"),
        ),
    )


def test_litter_model_initialization_no_data(caplog, fixture_core_components):
    """Test `LitterModel` initialization fails when all data is missing."""
    from virtual_ecosystem.core.data import Data
    from virtual_ecosystem.core.grid import Grid
    from virtual_ecosystem.models.litter.constants import LitterConsts
    from virtual_ecosystem.models.litter.litter_model import LitterModel

    caplog.clear()

    with pytest.raises(ValueError):
        # Make four cell grid
        grid = Grid(cell_nx=4, cell_ny=1)
        litter_data = Data(grid)

        LitterModel(
            data=litter_data,
            core_components=fixture_core_components,
            model_constants=LitterConsts(),
        )

    # Final check that expected logging entries are produced
    log_check(
        caplog,
        expected_log=(
            (
                ERROR,
                "litter model: init data missing required var "
                "'litter_pool_above_metabolic'",
            ),
            (
                ERROR,
                "litter model: init data missing required var "
                "'litter_pool_above_structural'",
            ),
            (
                ERROR,
                "litter model: init data missing required var " "'litter_pool_woody'",
            ),
            (
                ERROR,
                "litter model: init data missing required var "
                "'litter_pool_below_metabolic'",
            ),
            (
                ERROR,
                "litter model: init data missing required var "
                "'litter_pool_below_structural'",
            ),
            (
                ERROR,
                "litter model: init data missing required var "
                "'lignin_above_structural'",
            ),
            (
                ERROR,
                "litter model: init data missing required var " "'lignin_woody'",
            ),
            (
                ERROR,
                "litter model: init data missing required var "
                "'lignin_below_structural'",
            ),
            (
                ERROR,
                "litter model: error checking required_init_vars, see log.",
            ),
        ),
    )


def test_litter_model_initialization_bad_pool_bounds(
    caplog, dummy_litter_data, fixture_core_components
):
    """Test `LitterModel` initialization fails when litter pools are out of bounds."""
    from virtual_ecosystem.models.litter.constants import LitterConsts
    from virtual_ecosystem.models.litter.litter_model import LitterModel

    with pytest.raises(InitialisationError):
        # Put incorrect data in for lmwc
        dummy_litter_data["litter_pool_above_metabolic"] = DataArray(
            [0.05, 0.02, -0.1], dims=["cell_id"]
        )

        LitterModel(
            data=dummy_litter_data,
            core_components=fixture_core_components,
            model_constants=LitterConsts,
        )

    # Final check that the last log entry is as expected
    log_check(
        caplog,
        expected_log=((ERROR, "Negative pool sizes found in: "),),
        subset=slice(-1, None, None),
    )


def test_litter_model_initialization_bad_lignin_bounds(
    caplog, dummy_litter_data, fixture_core_components
):
    """Test `LitterModel` initialization fails for lignin proportions not in bounds."""
    from virtual_ecosystem.models.litter.constants import LitterConsts
    from virtual_ecosystem.models.litter.litter_model import LitterModel

    with pytest.raises(InitialisationError):
        # Make four cell grid
        litter_data = deepcopy(dummy_litter_data)
        # Put incorrect data in for woody lignin
        litter_data["lignin_woody"] = DataArray([0.5, 0.4, 1.1], dims=["cell_id"])

        LitterModel(
            data=litter_data,
            core_components=fixture_core_components,
            model_constants=LitterConsts,
        )

    # Final check that expected logging entries are produced
    log_check(
        caplog,
        expected_log=((ERROR, "Lignin proportions not between 0 and 1 found in: "),),
        subset=slice(-1, None, None),
    )


@pytest.mark.parametrize(
    "cfg_string,temp_response,raises,expected_log_entries",
    [
        pytest.param(
            "[core]\n[core.timing]\nupdate_interval = '24 hours'\n[litter]\n",
            3.36,
            does_not_raise(),
            (
                (INFO, "Initialised litter.LitterConsts from config"),
                (
                    INFO,
                    "Information required to initialise the litter model successfully "
                    "extracted.",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_above_metabolic' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_above_structural' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_woody' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_below_metabolic' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_below_structural' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'lignin_above_structural' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'lignin_woody' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'lignin_below_structural' checked",
                ),
            ),
            id="default_config",
        ),
        pytest.param(
            "[core]\n[core.timing]\nupdate_interval = '24 hours'\n"
            "[litter.constants.LitterConsts]\nlitter_decomp_temp_response = 4.44\n",
            4.44,
            does_not_raise(),
            (
                (INFO, "Initialised litter.LitterConsts from config"),
                (
                    INFO,
                    "Information required to initialise the litter model successfully "
                    "extracted.",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_above_metabolic' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_above_structural' checked",
                ),
                (DEBUG, "litter model: required var 'litter_pool_woody' checked"),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_below_metabolic' checked",
                ),
                (
                    DEBUG,
                    "litter model: required var 'litter_pool_below_structural' checked",
                ),
                (DEBUG, "litter model: required var 'lignin_above_structural' checked"),
                (DEBUG, "litter model: required var 'lignin_woody' checked"),
                (DEBUG, "litter model: required var 'lignin_below_structural' checked"),
            ),
            id="modified_config_correct",
        ),
        pytest.param(
            "[core.timing]\nupdate_interval = '24 hours'\n"
            "[litter.constants.LitterConsts]\ndecomp_rate = 4.44\n",
            None,
            pytest.raises(ConfigurationError),
            (
                (ERROR, "Unknown names supplied for LitterConsts: decomp_rate"),
                (INFO, "Valid names are: "),
                (CRITICAL, "Could not initialise litter.LitterConsts from config"),
            ),
            id="modified_config_incorrect",
        ),
    ],
)
def test_generate_litter_model(
    caplog,
    dummy_litter_data,
    cfg_string,
    temp_response,
    raises,
    expected_log_entries,
):
    """Test that the function to initialise the litter model behaves as expected."""

    from virtual_ecosystem.core.config import Config
    from virtual_ecosystem.core.core_components import CoreComponents
    from virtual_ecosystem.models.litter.litter_model import LitterModel

    # Build the config object and core components
    config = Config(cfg_strings=cfg_string)
    core_components = CoreComponents(config=config)
    caplog.clear()

    # Check whether model is initialised (or not) as expected
    with raises:
        model = LitterModel.from_config(
            data=dummy_litter_data,
            core_components=core_components,
            config=config,
        )
        assert model.model_constants.litter_decomp_temp_response == temp_response

    # Final check that expected logging entries are produced
    log_check(caplog, expected_log_entries)


def test_update(fixture_litter_model, dummy_litter_data):
    """Test to check that the update step works and increments the update step."""

    end_above_meta = [0.29587973, 0.14851276, 0.07041856]
    end_above_struct = [0.50055126, 0.25010012, 0.0907076]
    end_woody = [4.702103, 11.802315, 7.300997]
    end_below_meta = [0.38949196, 0.36147436, 0.06906041]
    end_below_struct = [0.60011634, 0.30989963, 0.02047753]
    end_lignin_above_struct = [0.4996410, 0.1004310, 0.6964345]
    end_lignin_woody = [0.49989001, 0.79989045, 0.34998229]
    end_lignin_below_struct = [0.499760108, 0.249922519, 0.737107757]
    c_mineral = [0.02987233, 0.02316114, 0.00786517]

    fixture_litter_model.update(time_index=0)

    # Check that data fixture has been updated correctly
    assert np.allclose(dummy_litter_data["litter_pool_above_metabolic"], end_above_meta)
    assert np.allclose(
        dummy_litter_data["litter_pool_above_structural"], end_above_struct
    )
    assert np.allclose(dummy_litter_data["litter_pool_woody"], end_woody)
    assert np.allclose(dummy_litter_data["litter_pool_below_metabolic"], end_below_meta)
    assert np.allclose(
        dummy_litter_data["litter_pool_below_structural"], end_below_struct
    )
    assert np.allclose(
        dummy_litter_data["lignin_above_structural"], end_lignin_above_struct
    )
    assert np.allclose(dummy_litter_data["lignin_woody"], end_lignin_woody)
    assert np.allclose(
        dummy_litter_data["lignin_below_structural"], end_lignin_below_struct
    )
    assert np.allclose(dummy_litter_data["litter_C_mineralisation_rate"], c_mineral)
