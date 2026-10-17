"""Rosenbrock integrator for one coupling interval with frozen rate constants (KPP form).

Stage i (transformed variables, autonomous ODE y' = f(y)):
    U_i = y + sum_{j<i} a_ij K_j
    (I/(h g) - J) K_i = f(U_i) + sum_{j<i} (c_ij / h) K_j     (f reused when not new_f[i])
    y_new = y + sum_j m_j K_j,  err_vec = sum_j e_j K_j
J is the Jacobian at y (one evaluation per accepted step), one LU of (I/(h g) - J) per
attempt. Error norm, step-size control and rejection rules follow KPP's rosenbrock.f90
(Sandu et al. 1997): err = RMS(err_vec / (atol + rtol max(|y|, |y_new|))),
h_new = h * clip(0.9 err^(-1/elo), 0.2, 6), no growth right after a rejection, and after
two rejections in a row h is cut by 10.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass(frozen=True)
class Tableau:
    name: str
    s: int
    a: tuple          # lower-triangular, row-wise: a[i][j] for j < i
    c: tuple
    new_f: tuple
    m: tuple
    e: tuple
    gamma: float
    elo: float


def _rows(flat, s):
    out, q = [], 0
    for i in range(s):
        out.append(tuple(flat[q:q + i]))
        q += i
    return tuple(out)


# KPP rosenbrock.f90 coefficients
RODAS3 = Tableau("rodas3", 4,
                 _rows((0.0, 2.0, 0.0, 2.0, 0.0, 1.0), 4),
                 _rows((4.0, 1.0, -1.0, 1.0, -1.0, -8.0 / 3.0), 4),
                 (True, False, True, True), (2.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0),
                 0.5, 3.0)
_g2 = 1.0 + 1.0 / np.sqrt(2.0)
ROS2 = Tableau("ros2", 2, _rows((1.0 / _g2,), 2), _rows((-2.0 / _g2,), 2), (True, True),
               (3.0 / (2.0 * _g2), 1.0 / (2.0 * _g2)), (1.0 / (2.0 * _g2), 1.0 / (2.0 * _g2)),
               _g2, 2.0)
RODAS4 = Tableau(
    "rodas4", 6,
    _rows((0.1544000000000000e+01, 0.9466785280815826e+00, 0.2557011698983284e+00,
           0.3314825187068521e+01, 0.2896124015972201e+01, 0.9986419139977817e+00,
           0.1221224509226641e+01, 0.6019134481288629e+01, 0.1253708332932087e+02,
           -0.6878860361058950e+00, 0.1221224509226641e+01, 0.6019134481288629e+01,
           0.1253708332932087e+02, -0.6878860361058950e+00, 1.0), 6),
    _rows((-0.5668800000000000e+01, -0.2430093356833875e+01, -0.2063599157091915e+00,
           -0.1073529058151375e+00, -0.9594562251023355e+01, -0.2047028614809616e+02,
           0.7496443313967647e+01, -0.1024680431464352e+02, -0.3399990352819905e+02,
           0.1170890893206160e+02, 0.8083246795921522e+01, -0.7981132988064893e+01,
           -0.3152159432874371e+02, 0.1631930543123136e+02, -0.6058818238834054e+01), 6),
    (True,) * 6,
    (0.1221224509226641e+01, 0.6019134481288629e+01, 0.1253708332932087e+02,
     -0.6878860361058950e+00, 1.0, 1.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    0.25, 4.0)
ROS3 = Tableau(
    "ros3", 3, _rows((1.0, 1.0, 0.0), 3),
    _rows((-0.10156171083877702091975600115545e+01, 0.40759956452537699824805835358067e+01,
           0.92076794298330791242156818474003e+01), 3),
    (True, True, False),
    (1.0, 0.61697947043828245592553615689730e+01, -0.42772256543218573326238373806514e+00),
    (0.5, -0.29079558716805469821718236208017e+01, 0.22354069897811569627360909276199e+00),
    0.43586652150845899941601945119356e+00, 3.0)
TABLEAUS = {t.name: t for t in (RODAS3, ROS2, ROS3, RODAS4)}

FAC_MIN, FAC_MAX, FAC_SAFE, FAC_REJ = 0.2, 6.0, 0.9, 0.1


def make_integrator(rhs, neg_jac, diag_mask, factor, solve, tab: Tableau, rtol: float,
                    atol: float, max_steps: int = 2000, err_norm: str = "rms"):
    """integrate(y0, k, dt, h0) -> (y_end, h_carry, stats[5], ok).

    stats = accepted, rejected, rhs evaluations, Jacobian evaluations, LU factorisations.
    h0 <= 0 selects the initial step with Hairer's rule (as diffrax does for dt0=None).
    err_norm: "rms" (KPP, diffrax default) or "max" (largest scaled error of any species).
    h_carry is the step size the controller would try next (for carrying it over).
    """
    s, g = tab.s, tab.gamma
    n_newf = sum(1 for i in range(1, s) if tab.new_f[i])

    def norm(v, sc):
        if err_norm == "max":
            return jnp.max(jnp.abs(v / sc))
        return jnp.sqrt(jnp.mean((v / sc) ** 2))

    def initial_step(y0, f0, k, dt):
        sc = atol + rtol * jnp.abs(y0)
        d0, d1 = norm(y0, sc), norm(f0, sc)
        h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
        h0 = jnp.minimum(h0, dt)
        f1 = rhs(y0 + h0 * f0, k)
        d2 = norm(f1 - f0, sc) / h0
        mx = jnp.maximum(d1, d2)
        h1 = jnp.where(mx <= 1e-15, jnp.maximum(1e-6, h0 * 1e-3),
                       (0.01 / mx) ** (1.0 / (tab.elo + 0.0)))
        return jnp.minimum(100.0 * h0, h1)

    def attempt(y, f0, nj, k, h):
        fac = factor(nj + (1.0 / (h * g)) * diag_mask)
        ks = []
        for i in range(s):
            if i == 0:
                fi = f0
            elif tab.new_f[i]:
                u = y
                for j, a in enumerate(tab.a[i]):
                    if a != 0.0:
                        u = u + a * ks[j]
                fi = rhs(u, k)
            b = fi
            for j, c in enumerate(tab.c[i]):
                if c != 0.0:
                    b = b + (c / h) * ks[j]
            ks.append(solve(fac, b))
        y_new, err_v = y, jnp.zeros_like(y)
        for j in range(s):
            if tab.m[j] != 0.0:
                y_new = y_new + tab.m[j] * ks[j]
            if tab.e[j] != 0.0:
                err_v = err_v + tab.e[j] * ks[j]
        sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
        return y_new, norm(err_v, sc), err_v

    def integrate(y0, k, dt, h0):
        f0 = rhs(y0, k)
        nj = neg_jac(y0, k)
        h_init = jnp.where(h0 > 0, h0, initial_step(y0, f0, k, dt))
        h_init = jnp.minimum(h_init, dt)
        n_init_f = jnp.where(h0 > 0, 1, 2)
        # carry: t, y, h, f0, nj, rej_last, rej_more, stats, h_free, fresh
        stats0 = jnp.array([0, 0, 0, 1, 0]).at[2].set(n_init_f)

        def cond(c):
            t, *_, stats, _ = c
            return (t < dt) & (stats[0] + stats[1] < max_steps)

        def body(c):
            t, y, h, f0, nj, rej_last, rej_more, stats, h_free = c
            h_free = h
            h = jnp.minimum(h, dt - t)
            y_new, err, _ = attempt(y, f0, nj, k, h)
            err = jnp.where(jnp.isfinite(err), jnp.maximum(err, 1e-10), 1e10)
            fac = jnp.clip(FAC_SAFE / err ** (1.0 / tab.elo), FAC_MIN, FAC_MAX)
            h_new = h * fac
            acc = err <= 1.0
            # accepted
            t_acc = jnp.where(h >= dt - t, dt, t + h)
            h_acc = jnp.minimum(h_new, dt)
            h_acc = jnp.where(rej_last, jnp.minimum(h_acc, h), h_acc)
            # rejected
            h_rej = jnp.where(rej_more, h * FAC_REJ, h_new)
            # Jacobian and f at the new point only after an accepted step that is not the last
            need = acc & (t_acc < dt)
            f_n, nj_n = lax.cond(need, lambda: (rhs(y_new, k), neg_jac(y_new, k)),
                                 lambda: (f0, nj))
            stats = stats + jnp.array([acc, ~acc, n_newf + need, need, 1], dtype=stats.dtype)
            # step size carried to the next interval: the free proposal, unless the
            # controller asks for a smaller one
            h_carry = jnp.where(acc, jnp.where(h < h_free, jnp.maximum(h_acc, h_free), h_acc),
                                h_free)
            return (jnp.where(acc, t_acc, t), jnp.where(acc, y_new, y),
                    jnp.where(acc, h_acc, h_rej), f_n, nj_n,
                    ~acc, jnp.where(acc, False, rej_last), stats, h_carry)

        c0 = (jnp.asarray(0.0), y0, h_init, f0, nj, jnp.asarray(False), jnp.asarray(False),
              stats0, h_init)
        t, y, h, _, _, _, _, stats, h_carry = lax.while_loop(cond, body, c0)
        return y, h_carry, stats, t >= dt

    def fixed_step(y, k, h):
        """One step of size h; returns (y_new, error vector). For order tests."""
        y_new, _, err_v = attempt(y, rhs(y, k), neg_jac(y, k), k, h)
        return y_new, err_v

    integrate.fixed_step = fixed_step
    return integrate
