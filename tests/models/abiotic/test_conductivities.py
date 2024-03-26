"""Test module for abiotic.conductivities.py."""

import numpy as np
from xarray import DataArray

from virtual_ecosystem.core.constants import CoreConsts
from virtual_ecosystem.models.abiotic.constants import AbioticConsts


def test_initialise_conductivities(dummy_climate_data, fixture_core_components):
    """Test conductivities are initialised correctly."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        initialise_conductivities,
    )

    result = initialise_conductivities(
        layer_heights=dummy_climate_data["layer_heights"],
        initial_air_conductivity=50.0,
        top_leaf_vapour_conductivity=0.32,
        bottom_leaf_vapour_conductivity=0.25,
        top_leaf_air_conductivity=0.19,
        bottom_leaf_air_conductivity=0.13,
    )

    coords = {
        "layers": np.arange(15),
        "layer_roles": (
            "layers",
            fixture_core_components.layer_structure.layer_roles,
        ),
        "cell_id": [0, 1, 2],
    }

    air_cond_values = np.repeat(
        a=[3.84615385, 3.33333333, 6.66666667, np.nan], repeats=[1, 11, 1, 2]
    )
    exp_air_cond = DataArray(
        np.broadcast_to(air_cond_values, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=coords,
        name="air_conductivity",
    )

    leaf_vap_values = np.repeat(
        a=[np.nan, 0.254389, 0.276332, 0.298276, np.nan],
        repeats=[1, 1, 1, 1, 11],
    )
    exp_leaf_vap_cond = DataArray(
        np.broadcast_to(leaf_vap_values, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=coords,
        name="leaf_vapour_conductivity",
    )

    leaf_air_values = np.repeat(
        a=[np.nan, 0.133762, 0.152571, 0.171379, np.nan],
        repeats=[1, 1, 1, 1, 11],
    )
    exp_leaf_air_cond = DataArray(
        np.broadcast_to(leaf_air_values, (3, 15)).T,
        dims=["layers", "cell_id"],
        coords=coords,
        name="leaf_air_conductivity",
    )

    np.testing.assert_allclose(
        result["air_conductivity"], exp_air_cond, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        result["leaf_vapour_conductivity"], exp_leaf_vap_cond, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        result["leaf_air_conductivity"], exp_leaf_air_cond, rtol=1e-04, atol=1e-04
    )


def test_interpolate_along_heights(dummy_climate_data):
    """Test linear interpolation along heights."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        interpolate_along_heights,
    )

    layer_heights = dummy_climate_data["layer_heights"]
    atmosphere_layers = layer_heights[layer_heights["layer_roles"] != "soil"]
    result = interpolate_along_heights(
        start_height=layer_heights[-3].to_numpy(),
        end_height=layer_heights[0].to_numpy(),
        target_heights=(layer_heights[atmosphere_layers.indexes].to_numpy()),
        start_value=50.0,
        end_value=20.0,
    )
    exp_result = np.full((13, 3), np.nan)
    row_vals = [20.0, 21.88087774, 31.28526646, 40.68965517, 48.68338558, 50.0]
    exp_result.T[..., [0, 1, 2, 3, 11, 12]] = row_vals
    np.testing.assert_allclose(result, exp_result, rtol=1e-04, atol=1e-04)


def test_interpolate_along_heights_arrays(dummy_climate_data):
    """Test linear interpolation along heights with arrays of boundary values."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        interpolate_along_heights,
    )

    layer_heights = dummy_climate_data["layer_heights"]
    atmosphere_layers = layer_heights[layer_heights["layer_roles"] != "soil"]
    result = interpolate_along_heights(
        start_height=layer_heights[-3].to_numpy(),
        end_height=layer_heights[0].to_numpy(),
        target_heights=(layer_heights[atmosphere_layers.indexes].to_numpy()),
        start_value=np.repeat(50.0, 3),
        end_value=np.repeat(20.0, 3),
    )
    exp_result = np.full((13, 3), np.nan)
    row_vals = [20.0, 21.88087774, 31.28526646, 40.68965517, 48.68338558, 50.0]
    exp_result.T[..., [0, 1, 2, 3, 11, 12]] = row_vals
    np.testing.assert_allclose(result, exp_result, rtol=1e-04, atol=1e-04)


def test_calculate_air_heat_conductivity_above(dummy_climate_data):
    """Test heat conductivity above canopy."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        calculate_air_heat_conductivity_above,
    )

    result = calculate_air_heat_conductivity_above(
        height_above_canopy=dummy_climate_data["layer_heights"][0],
        zero_displacement_height=(
            dummy_climate_data["zero_displacement_height"].to_numpy()
        ),
        canopy_height=dummy_climate_data["layer_heights"][1],
        friction_velocity=dummy_climate_data["friction_velocity"].to_numpy(),
        molar_density_air=dummy_climate_data["molar_density_air"][0].to_numpy(),
        diabatic_correction_heat=(
            dummy_climate_data["diabatic_correction_heat_above"].to_numpy()
        ),
        von_karmans_constant=CoreConsts.von_karmans_constant,
    )
    np.testing.assert_allclose(
        result, np.array([523.39996, 218.083317, 87.233327]), rtol=1e-04, atol=1e-04
    )


def test_calculate_air_heat_conductivity_canopy(dummy_climate_data):
    """Test calculate air heat conductivity in canopy."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        calculate_air_heat_conductivity_canopy,
    )

    result = calculate_air_heat_conductivity_canopy(
        attenuation_coefficient=(
            dummy_climate_data["attenuation_coefficient"][1].to_numpy()
        ),
        mean_mixing_length=dummy_climate_data["mean_mixing_length"].to_numpy(),
        molar_density_air=dummy_climate_data["molar_density_air"][1].to_numpy(),
        upper_height=np.repeat(10.0, 3),
        lower_height=np.repeat(5.0, 3),
        relative_turbulence_intensity=(
            dummy_climate_data["relative_turbulence_intensity"][1].to_numpy()
        ),
        top_of_canopy_wind_speed=np.repeat(1.0, 3),
        diabatic_correction_momentum=(
            dummy_climate_data["diabatic_correction_momentum_canopy"].to_numpy()
        ),
        canopy_height=dummy_climate_data["layer_heights"][1].to_numpy(),
    )
    exp_result = np.repeat(0.236981, 3)
    np.testing.assert_allclose(result, exp_result, rtol=1e-04, atol=1e-04)


def test_calculate_leaf_air_heat_conductivity(dummy_climate_data):
    """Test calculation of leaf air heat conductivity."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        calculate_leaf_air_heat_conductivity,
    )

    abiotic_consts = AbioticConsts()
    result = calculate_leaf_air_heat_conductivity(
        temperature=dummy_climate_data["air_temperature"].to_numpy(),
        wind_speed=dummy_climate_data["wind_speed"].to_numpy(),
        characteristic_dimension_leaf=0.1,
        temperature_difference=(
            dummy_climate_data["canopy_temperature"]
            - dummy_climate_data["air_temperature"]
        ).to_numpy(),
        molar_density_air=dummy_climate_data["molar_density_air"].to_numpy(),
        kinematic_viscosity_parameters=abiotic_consts.kinematic_viscosity_parameters,
        thermal_diffusivity_parameters=abiotic_consts.thermal_diffusivity_parameters,
        grashof_parameter=abiotic_consts.grashof_parameter,
        forced_conductance_parameter=abiotic_consts.forced_conductance_parameter,
        positive_free_conductance_parameter=(
            abiotic_consts.positive_free_conductance_parameter
        ),
        negative_free_conductance_parameter=(
            abiotic_consts.negative_free_conductance_parameter
        ),
    )
    exp_result = np.full((15, 3), np.nan)
    row_vals = [0.065242, 0.065062, 0.064753]
    exp_result.T[..., [1, 2, 3]] = row_vals

    np.testing.assert_allclose(result, exp_result, rtol=1e-04, atol=1e-04)


def test_calculate_leaf_vapour_conductivity():
    """Test calculate leaf vapour conductivity."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        calculate_leaf_vapour_conductivity,
    )

    result = calculate_leaf_vapour_conductivity(
        leaf_air_conductivity=np.repeat(5.0, 3),
        stomatal_conductance=np.repeat(5.0, 3),
    )
    np.testing.assert_allclose(result, np.repeat(2.5, 3), rtol=1e-04, atol=1e-04)


def test_calculate_current_conductivities(dummy_climate_data):
    """Test update current conductivities."""

    from virtual_ecosystem.models.abiotic.conductivities import (
        calculate_current_conductivities,
    )

    result = calculate_current_conductivities(
        data=dummy_climate_data,
        characteristic_dimension_leaf=0.01,
        von_karmans_constant=CoreConsts.von_karmans_constant,
        abiotic_constants=AbioticConsts(),
    )

    exp_gt = np.full((15, 3), np.nan)
    gt_vals = [1.46096e02, 1.95435e03, 1.414247e01, 0.125081, 17.67347, 13.654908]
    exp_gt.T[..., [0, 1, 2, 3, 12, 13]] = gt_vals

    exp_gv = np.full((15, 3), np.nan)
    gv_vals = [0.203513, 0.202959, 0.202009]
    exp_gv.T[..., [1, 2, 3]] = gv_vals

    exp_gha = np.full((15, 3), np.nan)
    gha_vals = [0.206312, 0.205743, 0.204766]
    exp_gha.T[..., [1, 2, 3]] = gha_vals

    exp_gtr = np.full((15, 3), np.nan)
    gtr_vals = [1.954354e03, 1.403429e01, 0.123447, 1.788087, 0.604689]
    exp_gtr.T[..., [1, 2, 3, 12, 13]] = gtr_vals

    np.testing.assert_allclose(
        result["air_heat_conductivity"], exp_gt, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        result["leaf_air_heat_conductivity"], exp_gha, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        result["leaf_vapour_conductivity"], exp_gv, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        result["conductivity_from_ref_height"], exp_gtr, rtol=1e-04, atol=1e-04
    )
