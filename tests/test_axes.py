"""Testing the data validators."""

from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from xarray import DataArray


def test_AxisValidator_registration_bad_core_axis():
    """Simple test of AxisValidator registration."""
    from virtual_rainforest.core.axes import AxisValidator
    from virtual_rainforest.core.grid import Grid

    # Registered correctly
    with pytest.raises(ValueError) as excep:

        # Create a new failing subclass.
        class TestAxis(AxisValidator):

            core_axis = ""
            dim_names = {"test"}

            def can_validate(self, value: DataArray, grid: Grid, **kwargs: Any) -> bool:
                return True

            def run_validation(
                self, value: DataArray, grid: Grid, **kwargs: Any
            ) -> DataArray:
                return value

    assert str(excep.value) == "Core axis name cannot be an empty string."


def test_AxisValidator_registration_bad_dim_names():
    """Simple test of AxisValidator registration."""
    from virtual_rainforest.core.axes import AxisValidator
    from virtual_rainforest.core.grid import Grid

    # Registered correctly
    with pytest.raises(ValueError) as excep:

        # Create a new failing subclass.
        class TestAxis(AxisValidator):

            core_axis = "failing_test"
            dim_names = {}

            def can_validate(self, value: DataArray, grid: Grid, **kwargs: Any) -> bool:
                return True

            def run_validation(
                self, value: DataArray, grid: Grid, **kwargs: Any
            ) -> DataArray:
                return value

    assert str(excep.value) == "AxisValidator dim names cannot be an empty set."


def test_AxisValidator_registration(new_axis_validators):
    """Simple test of AxisValidator registration."""
    from virtual_rainforest.core.axes import AXIS_VALIDATORS

    # Registered correctly
    assert "testing" in AXIS_VALIDATORS
    assert len(AXIS_VALIDATORS["testing"]) == 2


def test_AxisValidator_methods(new_axis_validators, fixture_data):
    """Simple test of AxisValidator registration and methods."""
    from virtual_rainforest.core.axes import AXIS_VALIDATORS

    # Use the methods
    test_v7r = AXIS_VALIDATORS["testing"][0]()
    assert not test_v7r.can_validate(DataArray([1, 1, 1, 1, 1]), grid=fixture_data.grid)
    assert test_v7r.can_validate(DataArray([3, 3, 3, 3, 3]), grid=fixture_data.grid)

    validated = test_v7r.run_validation(
        DataArray([3, 3, 3, 3, 3]), grid=fixture_data.grid
    )
    assert np.allclose(validated, DataArray([6, 6, 6, 6, 6]))


@pytest.mark.parametrize(
    argnames=["value", "exp_val_dict", "exp_err", "exp_msg"],
    argvalues=[
        pytest.param(
            DataArray(data=np.arange(4), dims=("cell_id")),
            {"spatial": "Spat_CellId_Dim_Any", "testing": None},
            does_not_raise(),
            None,
            id="Match found",
        ),
        pytest.param(
            DataArray(data=np.arange(4), dims=("x")),
            {},
            pytest.raises(ValueError),
            "DataArray uses 'spatial' axis dimension names but "
            "does not match a validator: x",
            id="Uses dims, no match",
        ),
        pytest.param(
            DataArray(data=np.arange(50), dims=("test")),
            {},
            pytest.raises(RuntimeError),
            "Validators on 'testing' axis not mutually exclusive",
            id="Bad validator setup",
        ),
        pytest.param(
            DataArray(data=np.arange(4), dims=("cell_identities")),
            {"spatial": None, "testing": None},
            does_not_raise(),
            None,
            id="No match found",
        ),
    ],
)
def test_validate_dataarray(
    new_axis_validators, fixture_data, value, exp_val_dict, exp_err, exp_msg
):
    """Test the validate_dataarray function.

    This just checks the pass through and failure modes - the individual AxisValidator
    tests should check the return values
    """

    from virtual_rainforest.core.axes import validate_dataarray

    # Decorate a mock function to test the failure modes
    with exp_err as err:
        value, val_dict = validate_dataarray(value, grid=fixture_data.grid)
        assert exp_val_dict == val_dict
    if err is not None:
        assert str(err.value) == exp_msg


@pytest.mark.parametrize(
    argnames=["grid_args", "darray", "exp_err", "exp_message", "exp_vals"],
    argvalues=[
        (  # grid cell ids not covered by data
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 2},
            DataArray(data=np.arange(6), coords={"cell_id": [1, 2, 3, 4, 5, 9]}),
            pytest.raises(ValueError),
            "The data cell ids are not a superset of grid cell ids.",
            None,
        ),
        (  # Duplicate ids in data
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 2},
            DataArray(data=np.arange(6), coords={"cell_id": [0, 1, 2, 5, 4, 5]}),
            pytest.raises(ValueError),
            "The data cell ids contain duplicate values.",
            None,
        ),
        (  # - same size and order
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 2},
            DataArray(data=np.arange(6), coords={"cell_id": [0, 1, 2, 3, 4, 5]}),
            does_not_raise(),
            None,
            [0, 1, 2, 3, 4, 5],
        ),
        (  # - same order but more ids in cell data
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 2},
            DataArray(
                data=np.arange(9), coords={"cell_id": [0, 1, 2, 3, 4, 5, 6, 7, 8]}
            ),
            does_not_raise(),
            None,
            [0, 1, 2, 3, 4, 5],
        ),
        (  # - different order
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 2},
            DataArray(
                data=np.array([5, 3, 1, 0, 4, 2]),
                coords={"cell_id": [5, 3, 1, 0, 4, 2]},
            ),
            does_not_raise(),
            None,
            [0, 1, 2, 3, 4, 5],
        ),
        (  # - different order and subsetting
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 2},
            DataArray(
                data=np.array([6, 5, 7, 3, 1, 0, 4, 2, 8]),
                coords={"cell_id": [6, 5, 7, 3, 1, 0, 4, 2, 8]},
            ),
            does_not_raise(),
            None,
            [0, 1, 2, 3, 4, 5],
        ),
    ],
)
def test_Spat_CellId_Coord_Any(grid_args, darray, exp_err, exp_message, exp_vals):
    """Test the netdcf variable loader."""

    from virtual_rainforest.core.axes import Spat_CellId_Coord_Any
    from virtual_rainforest.core.data import Data
    from virtual_rainforest.core.grid import Grid

    grid = Grid(**grid_args)
    data = Data(grid)

    v7r = Spat_CellId_Coord_Any()

    can_val = v7r.can_validate(darray, data=data, grid=grid)

    if can_val:
        with exp_err as excep:
            darray = v7r.run_validation(darray, data=data, grid=grid)

            assert isinstance(darray, DataArray)
            assert np.allclose(darray.values, exp_vals)

        if excep is not None:
            assert str(excep.value) == exp_message


@pytest.mark.parametrize(
    argnames=["grid_args", "darray", "exp_err", "exp_message", "exp_vals"],
    argvalues=[
        (
            {"grid_type": "square"},
            DataArray(data=np.arange(50), dims=("cell_id")),
            pytest.raises(ValueError),
            "Grid defines 100 cells, data provides 50",
            None,
        ),
        (
            {"grid_type": "square"},
            DataArray(data=np.arange(100), dims=("cell_id")),
            does_not_raise(),
            None,
            np.arange(100),
        ),
    ],
)
def test_Spat_CellId_Dim_Any(grid_args, darray, exp_err, exp_message, exp_vals):
    """Test the netdcf variable loader."""

    from virtual_rainforest.core.axes import Spat_CellId_Dim_Any
    from virtual_rainforest.core.data import Data
    from virtual_rainforest.core.grid import Grid

    grid = Grid(**grid_args)
    data = Data(grid)

    v7r = Spat_CellId_Dim_Any()

    can_val = v7r.can_validate(darray, data=data, grid=grid)

    if can_val:
        with exp_err as excep:
            darray = v7r.run_validation(darray, data=data, grid=grid)

            assert isinstance(darray, DataArray)
            assert np.allclose(darray.values, exp_vals)

        if excep is not None:
            assert str(excep.value) == exp_message


@pytest.mark.parametrize(
    argnames=["grid_args", "darray", "exp_err", "exp_message", "exp_vals"],
    argvalues=[
        (  # Coords on cell boundaries
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 3, "cell_area": 1},
            DataArray(
                data=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                coords={"y": [2, 1, 0], "x": [2, 1, 0]},
            ),
            pytest.raises(ValueError),
            "Mapped points fall on cell boundaries.",
            None,
        ),
        (  # Does not cover cells
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 3, "cell_area": 1},
            DataArray(
                data=np.array([[0, 1, 2], [3, 4, 5]]),
                coords={"y": [2.5, 1.5], "x": [2.5, 1.5, 0.5]},
            ),
            pytest.raises(ValueError),
            "Mapped points do not cover all cells.",
            None,
        ),
        (  # Irregular sampling on y axis gives multiple points in bottom row
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 3, "cell_area": 1},
            DataArray(
                data=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),
                coords={"y": [2.5, 1.5, 0.5, 0.4], "x": [0.5, 1.5, 2.5]},
            ),
            pytest.raises(ValueError),
            "Some cells contain more than one point.",
            None,
        ),
        (  # All good
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 3, "cell_area": 1},
            DataArray(
                data=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
            ),
            does_not_raise(),
            None,
            np.arange(9),
        ),
    ],
)
def test_Spat_XY_Coord_Square(grid_args, darray, exp_err, exp_message, exp_vals):
    """Test the netdcf variable loader."""

    from virtual_rainforest.core.axes import Spat_XY_Coord_Square
    from virtual_rainforest.core.data import Data
    from virtual_rainforest.core.grid import Grid

    grid = Grid(**grid_args)
    data = Data(grid)

    v7r = Spat_XY_Coord_Square()

    can_val = v7r.can_validate(darray, data=data, grid=grid)

    if can_val:
        with exp_err as excep:
            darray = v7r.run_validation(darray, data=data, grid=grid)

            assert isinstance(darray, DataArray)
            assert np.allclose(darray.values, exp_vals)

        if excep is not None:
            assert str(excep.value) == exp_message


@pytest.mark.parametrize(
    argnames=["grid_args", "darray", "exp_err", "exp_message", "exp_vals"],
    argvalues=[
        (  # Wrong size
            {"grid_type": "square", "cell_nx": 2, "cell_ny": 3},
            DataArray(
                data=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), dims=("y", "x")
            ),
            pytest.raises(ValueError),
            "Data XY dimensions do not match square grid",
            None,
        ),
        (  # All good
            {"grid_type": "square", "cell_nx": 3, "cell_ny": 3},
            DataArray(
                data=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), dims=("y", "x")
            ),
            does_not_raise(),
            None,
            np.arange(9),
        ),
    ],
)
def test_Spat_XY_Dim_Square(grid_args, darray, exp_err, exp_message, exp_vals):
    """Test the netdcf variable loader."""

    from virtual_rainforest.core.axes import Spat_XY_Dim_Square
    from virtual_rainforest.core.data import Data
    from virtual_rainforest.core.grid import Grid

    grid = Grid(**grid_args)
    data = Data(grid)

    v7r = Spat_XY_Dim_Square()

    can_val = v7r.can_validate(darray, data=data, grid=grid)

    if can_val:
        with exp_err as excep:
            darray = v7r.run_validation(darray, data=data, grid=grid)

            assert isinstance(darray, DataArray)
            assert np.allclose(darray.values, exp_vals)

        if excep is not None:
            assert str(excep.value) == exp_message