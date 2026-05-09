"""Unit tests for the caliper noise-estimation primitives."""

from __future__ import annotations

import numpy as np
import pytest

from karst_analysis.caliper.noise import (
    moving_average_centered,
    lag1_autocorrelation,
    measure_noise_in_interval,
    compare_drilling_methods,
)


# ──────────────────────────────────────────────────────────────────────
#  moving_average_centered
# ──────────────────────────────────────────────────────────────────────
class TestMovingAverageCentered:
    def test_constant_signal_unchanged(self):
        y = np.full(50, 7.0)
        result = moving_average_centered(y, win_pts=11)
        np.testing.assert_allclose(result, y, atol=1e-12)

    def test_output_length_matches_input(self):
        y = np.arange(100, dtype=float)
        result = moving_average_centered(y, win_pts=15)
        assert len(result) == len(y)

    def test_even_window_bumped_to_odd(self):
        """An even win_pts should still produce a valid centred result."""
        y = np.arange(50, dtype=float)
        # Even window
        result_even = moving_average_centered(y, win_pts=10)
        # Equivalent odd window
        result_odd = moving_average_centered(y, win_pts=11)
        np.testing.assert_allclose(result_even, result_odd, atol=1e-12)

    def test_short_input_returns_copy(self):
        y = np.array([3.0])
        result = moving_average_centered(y, win_pts=11)
        np.testing.assert_array_equal(result, y)

    def test_linear_signal_is_preserved_in_middle(self):
        """Centred MA on a linear signal: middle samples equal the input."""
        y = np.arange(100, dtype=float)
        result = moving_average_centered(y, win_pts=11)
        # In the middle, away from the edges
        np.testing.assert_allclose(result[20:80], y[20:80], atol=1e-9)


# ──────────────────────────────────────────────────────────────────────
#  lag1_autocorrelation
# ──────────────────────────────────────────────────────────────────────
class TestLag1Autocorrelation:
    def test_white_noise_near_zero(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0, 1, 5000)
        ac = lag1_autocorrelation(r)
        assert abs(ac) < 0.05

    def test_perfectly_correlated_is_one(self):
        # A constant series has zero variance — undefined. Use a simple
        # linear ramp around mean: each sample = previous + 1, so r[i] - mean
        # follows the same pattern → high autocorrelation.
        r = np.arange(100, dtype=float)
        ac = lag1_autocorrelation(r)
        assert ac > 0.9

    def test_alternating_is_negative(self):
        r = np.array([1.0, -1.0] * 50)
        ac = lag1_autocorrelation(r)
        assert ac < -0.9


# ──────────────────────────────────────────────────────────────────────
#  measure_noise_in_interval
# ──────────────────────────────────────────────────────────────────────
class TestMeasureNoiseInInterval:
    def test_residual_mean_near_zero_for_constant_signal(self):
        z = np.linspace(0, 10, 200)   # BGL-positive
        cal = np.full(200, 16.0)
        result = measure_noise_in_interval(z, cal, 2.0, 8.0,
                                            detrend_window_m=0.30)
        assert abs(result["residual_mean_cm"]) < 1e-9
        assert result["sigma_std_cm"] == pytest.approx(0.0, abs=1e-9)

    def test_too_few_samples_raises(self):
        z = np.array([4.0, 5.0])
        cal = np.array([16.0, 16.1])
        with pytest.raises(ValueError, match="only.*samples"):
            measure_noise_in_interval(z, cal, 4.0, 5.0,
                                       detrend_window_m=0.30)

    def test_returns_required_keys(self):
        rng = np.random.default_rng(0)
        z = np.linspace(0, 10, 300)
        cal = 16.0 + rng.normal(0, 0.1, 300)
        result = measure_noise_in_interval(z, cal, 2.0, 8.0,
                                            detrend_window_m=0.30)
        required = {"well_interval", "n", "median_dz_m", "detrend_window_m",
                    "detrend_window_pts", "cal_mean_cm", "cal_std_cm",
                    "sigma_std_cm", "sigma_MAD_cm", "residual_mean_cm",
                    "lag1_autocorr"}
        assert set(result.keys()) == required


# ──────────────────────────────────────────────────────────────────────
#  compare_drilling_methods
# ──────────────────────────────────────────────────────────────────────
class TestCompareDrillingMethods:
    def test_drilling_zero_when_aw5d_equals_aw5o(self):
        """If AW5D and AW5O have the same sigma, drilling contribution is 0."""
        a = {"sigma_std_cm": 0.1, "sigma_MAD_cm": 0.08}
        b = {"sigma_std_cm": 0.1, "sigma_MAD_cm": 0.08}
        result = compare_drilling_methods(a, b)
        assert result["sigma_drilling_from_std_cm"] == pytest.approx(0.0, abs=1e-12)
        assert result["sigma_drilling_from_MAD_cm"] == pytest.approx(0.0, abs=1e-12)

    def test_drilling_clamped_to_zero_when_aw5d_smaller_than_aw5o(self):
        """If AW5D somehow has SMALLER sigma than AW5O, return 0 (not nan)."""
        aw5o = {"sigma_std_cm": 0.5, "sigma_MAD_cm": 0.4}
        aw5d = {"sigma_std_cm": 0.3, "sigma_MAD_cm": 0.2}
        result = compare_drilling_methods(aw5o, aw5d)
        assert result["sigma_drilling_from_std_cm"] == 0.0
        assert result["sigma_drilling_from_MAD_cm"] == 0.0

    def test_quadrature_decomposition(self):
        """sigma_drilling = sqrt(AW5D^2 - AW5O^2)."""
        aw5o = {"sigma_std_cm": 0.3, "sigma_MAD_cm": 0.2}
        aw5d = {"sigma_std_cm": 0.5, "sigma_MAD_cm": 0.4}
        result = compare_drilling_methods(aw5o, aw5d)
        assert result["sigma_drilling_from_std_cm"] == pytest.approx(
            np.sqrt(0.5**2 - 0.3**2), abs=1e-9,
        )
        assert result["sigma_drilling_from_MAD_cm"] == pytest.approx(
            np.sqrt(0.4**2 - 0.2**2), abs=1e-9,
        )
