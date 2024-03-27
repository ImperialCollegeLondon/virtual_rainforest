"""Test module for abiotic.wind.py."""

import numpy as np

from virtual_ecosystem.core.constants import CoreConsts
from virtual_ecosystem.models.abiotic.constants import AbioticConsts


def test_calculate_zero_plane_displacement(dummy_climate_data):
    """Test if calculated correctly and set to zero without vegetation."""

    from virtual_ecosystem.models.abiotic.wind import calculate_zero_plane_displacement

    result = calculate_zero_plane_displacement(
        canopy_height=dummy_climate_data.data["canopy_height"].to_numpy(),
        leaf_area_index=np.array([0, 3, 7]),
        zero_plane_scaling_parameter=7.5,
    )

    np.testing.assert_allclose(result, np.array([0.0, 25.312559, 27.58673]))


def test_calculate_roughness_length_momentum(dummy_climate_data):
    """Test roughness length governing momentum transfer."""

    from virtual_ecosystem.models.abiotic.wind import (
        calculate_roughness_length_momentum,
    )

    result = calculate_roughness_length_momentum(
        canopy_height=dummy_climate_data["canopy_height"].to_numpy(),
        leaf_area_index=np.array([0, 3, 7]),
        zero_plane_displacement=np.array([0.0, 25.312559, 27.58673]),
        substrate_surface_drag_coefficient=0.003,
        roughness_element_drag_coefficient=0.3,
        roughness_sublayer_depth_parameter=0.193,
        max_ratio_wind_to_friction_velocity=0.3,
        min_roughness_length=0.05,
        von_karman_constant=CoreConsts.von_karmans_constant,
    )

    np.testing.assert_allclose(
        result, np.array([0.017, 1.4533, 0.9591]), rtol=1e-3, atol=1e-3
    )


def test_calculate_diabatic_correction_above(dummy_climate_data):
    """Test diabatic correction factors for heat and momentum."""

    from virtual_ecosystem.models.abiotic.wind import (
        calculate_diabatic_correction_above,
    )

    abiotic_consts = AbioticConsts()
    core_const = CoreConsts()
    result = calculate_diabatic_correction_above(
        molar_density_air=np.repeat(28.96, 3),
        specific_heat_air=np.repeat(1.0, 3),
        temperature=dummy_climate_data["air_temperature"][0].to_numpy(),
        sensible_heat_flux=(
            dummy_climate_data["sensible_heat_flux_topofcanopy"].to_numpy()
        ),
        friction_velocity=dummy_climate_data["friction_velocity"][0].to_numpy(),
        wind_heights=dummy_climate_data["layer_heights"][0].to_numpy(),
        zero_plane_displacement=np.array([0.0, 25.312559, 27.58673]),
        celsius_to_kelvin=core_const.zero_Celsius,
        von_karmans_constant=core_const.von_karmans_constant,
        yasuda_stability_parameters=abiotic_consts.yasuda_stability_parameters,
        diabatic_heat_momentum_ratio=abiotic_consts.diabatic_heat_momentum_ratio,
    )

    exp_result_h = np.array([0.105164, 0.010235, 0.001342])
    exp_result_m = np.array([0.063098, 0.006141, 0.000805])
    np.testing.assert_allclose(result["psi_h"], exp_result_h, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result["psi_m"], exp_result_m, rtol=1e-4, atol=1e-4)


def test_calculate_diabatic_correction_canopy(dummy_climate_data):
    """Test calculate diabatic correction factors for canopy."""

    from virtual_ecosystem.models.abiotic.wind import (
        calculate_diabatic_correction_canopy,
    )

    result = calculate_diabatic_correction_canopy(
        air_temperature=dummy_climate_data["air_temperature"][1:4].to_numpy(),
        wind_speed=dummy_climate_data["wind_speed"][1:4].to_numpy(),
        layer_heights=dummy_climate_data["layer_heights"][1:4].to_numpy(),
        mean_mixing_length=dummy_climate_data["mean_mixing_length"].to_numpy(),
        gravity=CoreConsts.gravity,
    )
    exp_result_h = np.array([1.0, 1.0, 1.0])
    exp_result_m = np.array([1.0, 1.0, 1.0])
    np.testing.assert_allclose(result["phi_h"], exp_result_h, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result["phi_m"], exp_result_m, rtol=1e-4, atol=1e-4)


def test_calculate_mean_mixing_length(dummy_climate_data):
    """Test mixing length with and without vegetation."""

    from virtual_ecosystem.models.abiotic.wind import calculate_mean_mixing_length

    result = calculate_mean_mixing_length(
        canopy_height=dummy_climate_data["layer_heights"][1].to_numpy(),
        zero_plane_displacement=np.array([0.0, 25.312559, 27.58673]),
        roughness_length_momentum=np.array([0.017, 1.4533, 0.9591]),
        mixing_length_factor=AbioticConsts.mixing_length_factor,
    )

    np.testing.assert_allclose(
        result, np.array([1.284154, 1.280886, 0.836903]), rtol=1e-4, atol=1e-4
    )


def test_generate_relative_turbulence_intensity(dummy_climate_data):
    """Test relative turbulence intensity."""

    from virtual_ecosystem.models.abiotic.wind import (
        generate_relative_turbulence_intensity,
    )

    layer_heights = (
        dummy_climate_data.data["layer_heights"]
        .where(dummy_climate_data.data["layer_heights"].layer_roles != "soil")
        .dropna(dim="layers")
    )
    result_t = generate_relative_turbulence_intensity(
        layer_heights=layer_heights.to_numpy(),
        min_relative_turbulence_intensity=0.36,
        max_relative_turbulence_intensity=0.9,
        increasing_with_height=True,
    )

    exp_result_t = np.array(
        [
            [17.64, 17.64, 17.64],
            [16.56, 16.56, 16.56],
            [11.16, 11.16, 11.166],
            [5.76, 5.76, 5.76],
            [1.17, 1.17, 1.17],
            [0.414, 0.414, 0.414],
        ]
    )
    result_f = generate_relative_turbulence_intensity(
        layer_heights=layer_heights.to_numpy(),
        min_relative_turbulence_intensity=0.36,
        max_relative_turbulence_intensity=0.9,
        increasing_with_height=False,
    )

    exp_result_f = np.array(
        [
            [-16.92, -16.92, -16.92],
            [-15.84, -15.84, -15.84],
            [-10.44, -10.44, -10.44],
            [-5.04, -5.04, -5.04],
            [-0.45, -0.45, -0.45],
            [0.306, 0.306, 0.306],
        ]
    )
    np.testing.assert_allclose(result_t, exp_result_t, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(result_f, exp_result_f, rtol=1e-3, atol=1e-3)


def test_calculate_wind_attenuation_coefficient(dummy_climate_data):
    """Test wind attenuation coefficient with and without vegetation."""

    from virtual_ecosystem.models.abiotic.wind import (
        calculate_wind_attenuation_coefficient,
    )

    # leaf_area_index = (
    #     dummy_climate_data.data["leaf_area_index"]
    #     .where(dummy_climate_data.data["leaf_area_index"].layer_roles != "soil")
    #     .dropna(dim="layers")
    # )
    leaf_area_index = np.array([[1, 1, 1], [1.0, 1, np.nan], [1.0, np.nan, np.nan]])
    relative_turbulence_intensity = np.array(
        [
            [17.64, 17.64, 17.64],
            [16.56, 16.56, 16.56],
            [11.16, 11.16, 11.166],
            [5.76, 5.76, 5.76],
            [1.17, 1.17, 1.17],
            [0.414, 0.414, 0.414],
        ]
    )
    result = calculate_wind_attenuation_coefficient(
        canopy_height=dummy_climate_data.data["canopy_height"].to_numpy(),
        leaf_area_index=leaf_area_index,
        mean_mixing_length=np.array([1.35804, 1.401984, 0.925228]),
        drag_coefficient=AbioticConsts.drag_coefficient,
        relative_turbulence_intensity=relative_turbulence_intensity,
    )

    exp_result = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.133579, 0.129392, 0.196066],
            [0.142291, 0.137831, 0.0],
            [0.211141, 0.0, 0.0],
            [0.211141, 0.0, 0.0],
            [0.211141, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(result, exp_result, rtol=1e-3, atol=1e-3)


def test_wind_log_profile(dummy_climate_data):
    """Test log wind profile."""

    from virtual_ecosystem.models.abiotic.wind import wind_log_profile

    layer_heights = (
        dummy_climate_data.data["layer_heights"]
        .where(dummy_climate_data.data["layer_heights"].layer_roles != "soil")
        .dropna(dim="layers")
    )
    result = wind_log_profile(
        height=layer_heights.to_numpy(),
        zeroplane_displacement=np.array([0.0, 25.312559, 27.58673]),
        roughness_length_momentum=np.array([0.017, 1.4533, 0.9591]),
        diabatic_correction_momentum=np.array([1.0, 0.0, -1.0]),
        min_wind_speed=0.001,
    )

    exp_result = np.array(
        [
            [8.540278, 1.526394, 0.5263758],
            [8.475739, 1.171050, 0.001],
            [8.070274, 0.001, 0.001],
            [7.377127, 0.001, 0.001],
            [5.480007, 0.001, 0.001],
            [2.771957, 0.001, 0.001],
        ]
    )

    np.testing.assert_allclose(result, exp_result, rtol=1e-3, atol=1e-3)


def test_calculate_friction_velocity_reference_height(dummy_climate_data):
    """Calculate friction velocity."""

    from virtual_ecosystem.models.abiotic.wind import (
        calculate_friction_velocity_reference_height,
    )

    diab_correction = np.array([-1.0, 0.0, 1])
    result = calculate_friction_velocity_reference_height(
        wind_speed_ref=(
            dummy_climate_data.data["wind_speed_ref"].isel(time_index=0).to_numpy()
        ),
        reference_height=(dummy_climate_data.data["canopy_height"] + 10).to_numpy(),
        zeroplane_displacement=np.array([0.0, 25.312559, 27.58673]),
        roughness_length_momentum=np.array([0.017, 1.4533, 0.9591]),
        diabatic_correction_momentum=diab_correction,
        min_wind_speed=0.001,
        von_karmans_constant=CoreConsts.von_karmans_constant,
    )
    exp_result = np.array([0.058718, 0.163879, 0.107819])
    np.testing.assert_allclose(result, exp_result, rtol=1e-3, atol=1e-3)


def test_calculate_wind_above_canopy(dummy_climate_data):
    """Wind speed above canopy."""

    from virtual_ecosystem.models.abiotic.wind import calculate_wind_above_canopy

    result = calculate_wind_above_canopy(
        friction_velocity=np.array([0.0, 0.819397, 1.423534]),
        wind_height_above=(dummy_climate_data.data["canopy_height"] + 15).to_numpy(),
        zeroplane_displacement=np.array([0.0, 25.312559, 27.58673]),
        roughness_length_momentum=np.array([0.017, 1.4533, 0.9591]),
        diabatic_correction_momentum=np.array([0.003, 0.026, 0.013]),
        von_karmans_constant=CoreConsts.von_karmans_constant,
        min_wind_speed_above_canopy=0.55,
    )

    exp_result = np.array([0.55, 5.590124, 10.750233])
    np.testing.assert_allclose(result, exp_result, rtol=1e-3, atol=1e-3)


def test_calculate_wind_canopy(dummy_climate_data):
    """Test below canopy wind profile."""

    from virtual_ecosystem.models.abiotic.wind import calculate_wind_canopy

    attenuation_coeff = np.array(
        [
            [0, 0, 0],
            [0.133579, 0.129392, 0.196066],
            [0.142291, 0.137831, 0.208853],
            [0.211141, 0.204523, 0.309744],
            [0.211141, 0.204523, 0.309744],
            [0.211141, 0.204523, 0.309744],
        ]
    )
    layer_heights = (
        dummy_climate_data.data["layer_heights"]
        .where(dummy_climate_data.data["layer_heights"].layer_roles != "soil")
        .dropna(dim="layers")
    )
    result = calculate_wind_canopy(
        top_of_canopy_wind_speed=np.array([0.5, 5.590124, 10.750233]),
        wind_layer_heights=layer_heights.to_numpy(),
        canopy_height=dummy_climate_data["canopy_height"].to_numpy(),
        attenuation_coefficient=attenuation_coeff,
        min_wind_speed=0.001,
    )

    exp_result = np.array(
        [
            [0.5, 5.590124, 10.750233],
            [0.495843, 5.545099, 10.619302],
            [0.47402, 5.308529, 9.940403],
            [0.432442, 4.856859, 8.68831],
            [0.408857, 4.600042, 8.002089],
            [0.405097, 4.559065, 7.894382],
        ]
    )
    np.testing.assert_allclose(result, exp_result, rtol=1e-3, atol=1e-3)


def test_calculate_wind_profile(dummy_climate_data):
    """Test full update of wind profile."""

    from virtual_ecosystem.models.abiotic.wind import calculate_wind_profile

    input_dict = {}
    for var in ["layer_heights", "air_temperature", "leaf_area_index"]:
        input_dict[var] = (
            dummy_climate_data.data[var]
            .where(dummy_climate_data.data[var].layer_roles != "soil")
            .dropna(dim="layers")
        )

    wind_update = calculate_wind_profile(
        canopy_height=dummy_climate_data["canopy_height"].to_numpy(),
        wind_height_above=(dummy_climate_data["canopy_height"] + 15).to_numpy(),
        wind_layer_heights=input_dict["layer_heights"].to_numpy(),
        leaf_area_index=input_dict["leaf_area_index"].to_numpy(),
        air_temperature=input_dict["air_temperature"].to_numpy(),
        atmospheric_pressure=np.array([96.0, 96.0, 96.0]),
        sensible_heat_flux_topofcanopy=np.array([100.0, 50.0, 10.0]),
        wind_speed_ref=np.array([0.1, 5.0, 10.0]),
        wind_reference_height=(dummy_climate_data["canopy_height"] + 10).to_numpy(),
        abiotic_constants=AbioticConsts(),
        core_constants=CoreConsts(),
    )

    friction_velocity_exp = np.array([0.014257, 0.818637, 1.638679])
    wind_speed_exp = np.array(
        [
            [0.109341, 5.536364, 11.07365],
            [0.10846, 5.491774, 10.984462],
            [0.103833, 5.257489, 10.515853],
            [0.094999, 4.810177, 9.621155],
            [0.089976, 4.555839, 9.112435],
            [0.089175, 4.515257, 9.031265],
        ]
    )

    psi_h_exp = np.array([6.081216e-01, 3.880146e-03, 3.869994e-04])
    phi_h_exp = np.array([1.0, 1.0, 1.0])

    np.testing.assert_allclose(
        wind_update["friction_velocity"], friction_velocity_exp, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        wind_update["wind_speed"], wind_speed_exp, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        wind_update["diabatic_correction_heat_above"], psi_h_exp, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        wind_update["diabatic_correction_heat_canopy"], phi_h_exp, rtol=1e-3, atol=1e-3
    )
