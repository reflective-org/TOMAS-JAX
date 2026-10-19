"""Ventilation, surface and wall losses (tomas_jax.box.losses)."""
import numpy as np
import pytest

from tomas_jax.box.losses import (gas_loss_step, lai_nazaroff_beta, particle_loss_step, relax)

ROOM = dict(friction_velocity_cm_s=1.0, surface_to_volume_per_m=3.0, height_m=2.5,
            particle_density_g_cm3=1.4)


def test_relax_is_the_analytic_solution_and_its_integral():
    c1, integral = relax(10.0, 2.0, 0.5, 3.0)
    assert c1 == pytest.approx(2.0 + 8.0 * np.exp(-1.5))
    # the integral of the analytic solution (trapezoid on a fine grid)
    t = np.linspace(0.0, 3.0, 20001)
    assert integral == pytest.approx(np.trapezoid(2.0 + 8.0 * np.exp(-0.5 * t), t), rel=1e-8)
    # no loss: unchanged, integral c0 dt
    c1, integral = relax(np.array([1.0, 4.0]), np.array([0.0, 9.0]), np.array([0.0, 0.0]), 7.0)
    assert c1.tolist() == [1.0, 4.0] and integral.tolist() == [7.0, 28.0]


def test_lai_nazaroff_beta_has_the_expected_shape_and_size():
    dp = np.array([10e-9, 50e-9, 200e-9, 1e-6, 5e-6])
    beta_h = lai_nazaroff_beta(dp, 293.15, 101325.0, **ROOM) * 3600.0
    # V shape: diffusion dominates for the smallest, settling for the largest particles
    assert beta_h[0] > beta_h[1] > beta_h[2] < beta_h[3] < beta_h[4]
    # smooth-surface model values at u* = 1 cm/s, S/V = 3 1/m (0.19, 0.008, 1.6 1/h)
    assert 0.1 < beta_h[0] < 1.0            # 10 nm: diffusion
    assert 0.003 < beta_h[2] < 0.05         # 200 nm: the minimum, under 1 % per hour
    assert 0.5 < beta_h[4] < 10.0           # 5 um: settling
    # more turbulence deposits more
    faster = lai_nazaroff_beta(dp, 293.15, 101325.0, **{**ROOM, "friction_velocity_cm_s": 3.0})
    assert np.all(faster * 3600.0 > beta_h)
    with pytest.raises(ValueError, match="r\\+"):
        lai_nazaroff_beta(np.array([2e-2]), 293.15, 101325.0, **ROOM)


def test_particle_loss_budget_closes_per_bin():
    rng = np.random.default_rng(1)
    nk = rng.uniform(1e9, 1e12, 5)
    mk = rng.uniform(1e-12, 1e-9, (5, 3))
    nk_out, mk_out = 0.5 * nk[::-1], 0.3 * mk[::-1]
    lam = np.full(5, 0.5 / 3600.0)
    beta = np.array([1e-4, 3e-5, 0.0, 2e-5, 5e-4])
    nk1, mk1, b = particle_loss_step(nk, mk, nk_out, mk_out, lam, beta, 300.0)
    np.testing.assert_allclose(nk1 - nk, b["n_in"] - b["n_out"] - b["n_deposited"], rtol=1e-12)
    np.testing.assert_allclose(mk1 - mk, b["m_in"] - b["m_out"] - b["m_deposited"], rtol=1e-12)
    assert np.all(b["n_deposited"] >= 0) and b["n_deposited"][2] == 0.0
    # ventilation only: the exact exponential toward the outdoor state
    nk2, mk2, _ = particle_loss_step(nk, mk, nk_out, mk_out, lam, np.zeros(5), 3600.0)
    np.testing.assert_allclose(nk2, nk_out + (nk - nk_out) * np.exp(-0.5), rtol=1e-12)
    np.testing.assert_allclose(mk2, mk_out + (mk - mk_out) * np.exp(-0.5), rtol=1e-12)
    # nothing on: unchanged
    nk3, mk3, b3 = particle_loss_step(nk, mk, nk_out, mk_out, np.zeros(5), np.zeros(5), 300.0)
    assert np.array_equal(nk3, nk) and np.array_equal(mk3, mk) and not np.any(b3["n_out"])


def test_gas_loss_budget_closes():
    y = np.array([1e12, 5e11, 0.0, 2e10])
    y_out = np.array([8e11, 0.0, 1e11, 0.0])
    lam = np.full(4, 1.0 / 3600.0)
    extra = np.array([3.0 / 3600.0, 0.0, 0.0, 1e-4])
    y1, b = gas_loss_step(y, y_out, lam, extra, 300.0)
    np.testing.assert_allclose(y1 - y, b["in"] - b["out"] - b["sink"], rtol=1e-12, atol=1e-3)
    assert b["sink"][1] == 0.0 and b["sink"][0] > 0
    # the steady state of ventilation plus surface loss: lambda C_out / (lambda + k_s)
    y_ss, _ = gas_loss_step(y, y_out, lam, extra, 1e9)
    assert y_ss[0] == pytest.approx(8e11 / 4.0, rel=1e-9)
