"""Phase 0 warm-up: JAX differentiation of the Dunne 2016 nucleation rates.

Small, analytically checkable problems before scaling to full-model
sensitivities (see docs/nucleation_sensitivity.md).

  0a  grad of ln J w.r.t. existing inputs (T, H2SO4) at one stratospheric
      point, checked against closed forms and central finite differences;
      vmap of d lnJ/dT over 190-240 K.

Usage::

    python -m sensitivities.dunne_nuc.phase0_warmup --part 0a
"""
import os
import argparse

import numpy as np
import jax
import jax.numpy as jnp

import tomas_jax.core.config  # noqa: F401  (enables float64)
from tomas_jax.physics import nucleation as nuc

from .common import NUC_PARAMS

OUTDIR = os.path.join(os.path.dirname(__file__), 'results', 'phase0')

# Stratospheric reference point (~20 km)
T_REF = 217.0          # K
P_REF = 5500.0         # Pa
H2SO4_REF = 1e6        # molec/cm3
FION_REF = 10.0        # ion pairs/cm3/s
NH3_REF = 0.0          # molec/cm3 (no ammonia in the stratosphere)


def mair(temp, pres):
    """Air number density [molec/cm3], same formula as nucleation.py."""
    return 2.69e19 * 273.15 / temp * pres / 101325.0


def central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def report(name, ad, ref, fd, rtol=1e-6):
    """Print one check row; return True if AD matches the reference."""
    rel_ref = abs(ad - ref) / max(abs(ref), 1e-300)
    rel_fd = abs(ad - fd) / max(abs(ref), 1e-300)
    ok = rel_ref < rtol
    print(f"  {name:<28s} AD={ad: .10e}  ref={ref: .10e}  FD={fd: .10e}  "
          f"rel(AD,ref)={rel_ref:.1e}  rel(AD,FD)={rel_fd:.1e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# =========================================================================
# 0a: gradients w.r.t. existing inputs
# =========================================================================

def mech(idx, temp, ln_h2so4, fion=FION_REF, nh3=NH3_REF, pres=P_REF):
    """One Dunne mechanism (idx 1=Jbn, 3=Jbi) [cm^-3 s^-1].

    Mair is held fixed at the reference T so that d/dT isolates the rate
    constant and ion-recombination temperature dependence.
    """
    out = nuc.dunne_nucleation_rate(
        temp, fion, jnp.exp(ln_h2so4), nh3, mair(T_REF, pres))
    return out[idx]


def ln_mech(idx, temp, ln_h2so4, fion=FION_REF, nh3=NH3_REF, pres=P_REF):
    """ln of one Dunne mechanism (idx 1=Jbn, 3=Jbi)."""
    return jnp.log(mech(idx, temp, ln_h2so4, fion, nh3, pres))


def analytic_dlnJbn_dT(temp):
    t = temp / 1000.0
    return -(nuc._VBN / 1000.0) * np.exp(nuc._VBN * (t - nuc._WBN))


def analytic_dlnJbi_dT(temp, fion=FION_REF, pres=P_REF):
    t = temp / 1000.0
    dlnk = -(nuc._VBI / 1000.0) * np.exp(nuc._VBI * (t - nuc._WBI))
    M = mair(T_REF, pres)
    alpha = 6e-8 * np.sqrt(300.0 / temp) + 6e-26 * M * (300.0 / temp)**4
    dalpha = (-0.5 * 6e-8 * np.sqrt(300.0) * temp**-1.5
              - 4.0 * 6e-26 * M * 300.0**4 * temp**-5)
    # ionc = sqrt(fion/alpha)  ->  d ln ionc/dT = -0.5 dalpha/alpha
    return dlnk - 0.5 * dalpha / alpha


def part_0a():
    print("=" * 100)
    print( f"AD=automatic differentiation (jax.grad), ref=analytical derivative, FD=central finite differences")
    print(f"0a: d lnJ / d(T, ln H2SO4) at T={T_REF} K, H2SO4={H2SO4_REF:.0e}, "
          f"fion={FION_REF}, NH3={NH3_REF}")
    print("=" * 100)
    T0 = jnp.float64(T_REF)
    L0 = jnp.float64(np.log(H2SO4_REF))
    ok = True

    for idx, name, p, dT_ref in [
        (1, 'Jbn', nuc._PBN, analytic_dlnJbn_dT(T_REF)),
        (3, 'Jbi', nuc._PBI, analytic_dlnJbi_dT(T_REF)),
    ]:
        # argnums=(0, 1): gradient w.r.t. temp and ln_h2so4 in one call
        g = jax.grad(lambda T, L: ln_mech(idx, T, L), argnums=(0, 1))
        dT, dL = (float(x) for x in g(T0, L0))
        fd_T = float(central_fd(lambda T: ln_mech(idx, T, L0), T0, 1e-3))
        fd_L = float(central_fd(lambda L: ln_mech(idx, T0, L), L0, 1e-4))
        ok &= report(f"d ln{name}/dT [1/K]", dT, dT_ref, fd_T)
        ok &= report(f"d ln{name}/d ln[H2SO4]", dL, p, fd_L)

    # vmap: d lnJ/dT across the stratospheric temperature range
    temps = jnp.linspace(190.0, 240.0, 101)
    dT_bn = jax.vmap(jax.grad(lambda T: ln_mech(1, T, L0)))(temps)
    dT_bi = jax.vmap(jax.grad(lambda T: ln_mech(3, T, L0)))(temps)
    max_err = max(
        np.max(np.abs(np.array(dT_bn) - analytic_dlnJbn_dT(np.array(temps)))
               / np.abs(analytic_dlnJbn_dT(np.array(temps)))),
        np.max(np.abs(np.array(dT_bi) - analytic_dlnJbi_dT(np.array(temps)))
               / np.abs(analytic_dlnJbi_dT(np.array(temps)))),
    )
    print(f"  vmap (auto vectorization) over 190-240 K: max rel err vs analytic = {max_err:.1e}")
    ok &= max_err < 1e-6

    # Absolute sensitivity dJ/dT [cm^-3 s^-1 K^-1], differentiating J itself.
    # Check against the chain rule dJ/dT = J * dlnJ/dT.
    J_bn = jax.vmap(lambda T: mech(1, T, L0))(temps)
    J_bi = jax.vmap(lambda T: mech(3, T, L0))(temps)
    dJdT_bn = jax.vmap(jax.grad(lambda T: mech(1, T, L0)))(temps)
    dJdT_bi = jax.vmap(jax.grad(lambda T: mech(3, T, L0)))(temps)
    chain_err = max(
        np.max(np.abs(np.array(dJdT_bn - J_bn * dT_bn) / np.array(dJdT_bn))),
        np.max(np.abs(np.array(dJdT_bi - J_bi * dT_bi) / np.array(dJdT_bi))),
    )
    print(f"  dJ/dT vs J * dlnJ/dT over 190-240 K: max rel err = {chain_err:.1e}")
    ok &= chain_err < 1e-10

    os.makedirs(OUTDIR, exist_ok=True)
    npz_out = os.path.join(OUTDIR, 'part0a_dlnJ_dT.npz')
    np.savez(npz_out,
             temps=np.array(temps), dT_bn=np.array(dT_bn), dT_bi=np.array(dT_bi),
             dJdT_bn=np.array(dJdT_bn), dJdT_bi=np.array(dJdT_bi))
    print(f"  wrote {npz_out}")
    plot_0a()

    if not ok:
        raise RuntimeError("Phase 0a checks FAILED (see table above)")
    print("0a: all checks PASS")


def plot_0a():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fname = os.path.join(OUTDIR, 'part0a_dlnJ_dT.npz')
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing NPZ: {fname}. Run --part 0a first.")
    d = np.load(fname)

    fig, ax = plt.subplots(figsize=(6, 4))
    # dlnJ/dT < 0 (colder -> faster); plot its magnitude. Independent of H2SO4.
    ax.plot(d['temps'], -d['dT_bn'], label='binary neutral (Jbn)')
    ax.plot(d['temps'], -d['dT_bi'], label='binary ion (Jbi)')
    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(r'$-\mathrm{d}\ln J/\mathrm{d}T$  [K$^{-1}$]')
    ax.set_title('Dunne 2016 relative T-sensitivity')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'part0a_dlnJ_dT.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# =========================================================================
# 0b: jacfwd w.r.t. the NucleationParams coefficient pytree
# =========================================================================

def as_jax(params):
    """NucleationParams of Python floats -> float64 JAX scalars."""
    return jax.tree.map(jnp.float64, params)


def ln_mech_p(idx, params, temp=T_REF, h2so4=H2SO4_REF, fion=FION_REF,
              nh3=NH3_REF, pres=P_REF):
    out = nuc.dunne_nucleation_rate(temp, fion, h2so4, nh3, mair(temp, pres),
                                    params)
    return jnp.log(out[idx])


def fd_params(f, params, rel_h=1e-6):
    """Central FD of scalar f w.r.t. every field of a NucleationParams."""
    out = {}
    for name, val in params._asdict().items():
        h = rel_h * max(abs(val), 1e-30)
        up = f(params._replace(**{name: val + h}))
        dn = f(params._replace(**{name: val - h}))
        out[name] = float((up - dn) / (2.0 * h))
    return out


# Which mechanism each coefficient belongs to (index into ln_mechanisms)
OWNER = {
    **{f: 1 for f in ('p_bn', 'u_bn', 'v_bn', 'w_bn')},
    **{f: 2 for f in ('p_tn', 'u_tn', 'v_tn', 'w_tn', 'p_an', 'a_n')},
    **{f: 3 for f in ('p_bi', 'u_bi', 'v_bi', 'w_bi')},
    **{f: 4 for f in ('p_ti', 'u_ti', 'v_ti', 'w_ti', 'p_ai', 'a_i')},
    'k_m': 0, 'delta_h': 0,
}
H2SO4_0B = 5e6   # cna = 5, so d lnJ/dp = ln(5) is non-trivial


def ln_mechanisms(p, h2so4=H2SO4_0B, nh3=1e7, org=1e5):
    """ln of (Riccobono, Jbn, Jtn, Jbi, Jti) at the stratospheric point."""
    ricco = nuc.ricco_nucleation_rate(T_REF, h2so4, org, p)
    _, Jbn, Jtn, Jbi, Jti = nuc.dunne_nucleation_rate(
        T_REF, FION_REF, h2so4, nh3, mair(T_REF, P_REF), p)
    return jnp.log(jnp.stack([ricco, Jbn, Jtn, Jbi, Jti]))


def part_0b():
    P = NUC_PARAMS
    Pj = as_jax(P)
    print("=" * 100)
    print("0b: jacfwd of ln Jbn w.r.t. NucleationParams (analytic + FD check)")
    print("=" * 100)
    ok = True

    # jacfwd returns a NucleationParams of partial derivatives
    J = jax.jacfwd(lambda p: ln_mech_p(1, p, h2so4=H2SO4_0B))(Pj)
    fd = fd_params(lambda p: ln_mech_p(1, p, h2so4=H2SO4_0B), P)

    t = T_REF / 1000.0
    e = np.exp(P.v_bn * (t - P.w_bn))
    analytic = {name: 0.0 for name in P._fields}
    analytic.update(u_bn=1.0, p_bn=np.log(H2SO4_0B * 1e-6),
                    v_bn=-e * (t - P.w_bn), w_bn=P.v_bn * e)
    for name in P._fields:
        ad = float(getattr(J, name))
        if analytic[name] == 0.0:
            if ad != 0.0:
                print(f"  {name:<10s} expected exactly 0, got {ad:.3e}  FAIL")
                ok = False
            continue
        ok &= report(f"d lnJbn/d {name}", ad, analytic[name], fd[name], rtol=1e-10)
    print("  all other fields: exactly 0 (as expected)" if ok else "")

    # Every rate coefficient vs FD of the log of the mechanism it belongs to.
    # (Against ln(J_total), minor mechanisms' sensitivities are ~1e-8 and FD
    # is below float64 roundoff, so that comparison is meaningless.)
    print(f"\n  ln(own mechanism) vs FD, every rate field, H2SO4={H2SO4_0B:.0e}, "
          f"NH3=1e7 (ternary active), org=1e5:")
    J = jax.jacfwd(ln_mechanisms)(Pj)
    worst = 0.0
    for name, idx in OWNER.items():
        ad = float(getattr(J, name)[idx])
        fd = fd_params(lambda p: ln_mechanisms(p)[idx],
                       P._replace())[name]
        rel = abs(ad - fd) / max(abs(ad), 1e-300)
        worst = max(worst, rel)
        print(f"    {name:<8s} (mech {idx})  AD={ad: .6e}  FD={fd: .6e}  rel={rel:.1e}")
    print(f"  worst rel(AD,FD) = {worst:.1e}")
    ok &= worst < 1e-6

    # Cluster properties must not enter the rate at all
    cluster = [float(np.max(np.abs(np.array(getattr(J, f)))))
               for f in ('r_nuc_nm', 'rho_nuc', 'org_frac')]
    print(f"  cluster fields in rate: max |d lnJ| = {max(cluster):.1e} (expect 0)")
    ok &= max(cluster) == 0.0

    # Share of the total rate per mechanism at this point (context for 0c)
    lnm = np.array(ln_mechanisms(P))
    share = np.exp(lnm) / np.exp(lnm).sum()
    print("  rate share (ricco, Jbn, Jtn, Jbi, Jti): "
          + ", ".join(f"{s:.2e}" for s in share))

    # NaN safety: tangents at NH3 = 0, fion = 0, org = 0
    print("\n  NaN check: jacfwd of total J and of nucleation_step outputs "
          "at NH3=0, fion=0, org=0")
    Gc = jnp.zeros(44).at[nuc.SRTSO4].set(1e-10)

    def total_zero(p):
        return nuc.ricco_dunne_nucleation_rate(Gc, T_REF, P_REF, 1e6,
                                               0.0, 0.0, 0.0, params=p)

    def step_zero(p):
        Nk = jnp.full(40, 1e3)
        Mk = jnp.full((40, 44), 1e-20)
        xk = jnp.geomspace(1e-24, 1e-12, 41)
        out = nuc.nucleation_step(Nk, Mk, Gc, xk, T_REF, P_REF, 1e6, 60.0,
                                  0.0, 0.0, 0.0, params=p)
        return jnp.concatenate([out[0], out[1].ravel(), out[2]])

    for name, f in [('rate', total_zero), ('nucleation_step', step_zero)]:
        leaves = jax.tree.leaves(jax.jacfwd(f)(Pj))
        n_bad = sum(int(np.sum(~np.isfinite(np.array(x)))) for x in leaves)
        print(f"    {name:<16s} non-finite tangent entries: {n_bad}")
        ok &= n_bad == 0

    if not ok:
        raise RuntimeError("Phase 0b checks FAILED (see table above)")
    print("0b: all checks PASS")


# =========================================================================
# 0c: jacfwd through 1 h of the full model at one stratospheric point
# =========================================================================

SO2_0C_PPBV = 1.0
FD_FIELDS = ('u_bn', 'p_bn', 'u_bi')


def part_0c():
    import time
    from tomas_jax.core.config import make_grid, XK0
    from .common import (STRAT, lognormal_state, make_runner, ppbv_to_molec,
                         BOXVOL)

    print("=" * 100)
    print(f"0c: jacfwd through 60 x 1-min steps (SO2 chem + nucl + coag + cond), "
          f"40 bins, T={STRAT['temp']} K, SO2={SO2_0C_PPBV} ppbv, NH3=0, org=0")
    print("=" * 100)
    xk = make_grid(40, XK0, 2.0)
    so2 = ppbv_to_molec(SO2_0C_PPBV, STRAT['temp'], STRAT['pres'])
    Nk0, Mk0, Gc0 = lognormal_state(xk, so2=so2)
    run = make_runner(xk, nsteps=60)
    P = NUC_PARAMS
    Pj = as_jax(P)

    def outputs(p):
        Nk, _, _, _ = run(p, Nk0, Mk0, Gc0)
        return jnp.stack([jnp.sum(Nk) / BOXVOL, Nk[0] / BOXVOL])

    f = jax.jit(outputs)
    t0 = time.time()
    y0 = np.array(f(Pj))
    t_fwd = time.time() - t0
    _, _, _, diag = jax.jit(run)(Pj, Nk0, Mk0, Gc0)
    n_sub = np.array(diag['n_sub'])
    clamp = np.array(diag['clamp'])
    h2 = np.array(diag['h2so4'])
    print(f"  forward run (incl. compile): {t_fwd:.1f} s")
    print(f"  N_total = {y0[0]:.4e} cm-3, N_bin0 = {y0[1]:.4e} cm-3")
    print(f"  H2SO4 final = {h2[-1]:.3e} cm-3, peak J = {np.max(np.array(diag['J'])):.3e} "
          f"cm-3 s-1")
    print(f"  regime: steps with n_sub>1: {np.sum(n_sub > 1)}/60 (max {n_sub.max()}), "
          f"clamp fired: {np.sum(clamp)}/60")

    t0 = time.time()
    J = jax.jit(jax.jacfwd(outputs))(Pj)
    print(f"  jacfwd w.r.t. all {len(P._fields)} fields (incl. compile): "
          f"{time.time() - t0:.1f} s")

    ok = True
    for name in P._fields:
        g = np.array(getattr(J, name))
        if not np.all(np.isfinite(g)):
            print(f"  {name}: NON-FINITE tangent {g}")
            ok = False

    # FD at two step sizes; AD should match both if no branch switches
    print(f"\n  d ln(output)/d theta: AD vs central FD (rel steps 1e-4, 1e-5)")
    print(f"  {'field':<8s} {'output':<8s} {'AD':>13s} {'FD(1e-4)':>13s} "
          f"{'FD(1e-5)':>13s} {'rel1':>8s} {'rel2':>8s}")
    rows = []
    for name in FD_FIELDS:
        ad = np.array(getattr(J, name)) / y0
        val = getattr(P, name)
        fds = []
        for rel in (1e-4, 1e-5):
            h = rel * abs(val)
            up = np.array(f(as_jax(P._replace(**{name: val + h}))))
            dn = np.array(f(as_jax(P._replace(**{name: val - h}))))
            fds.append((up - dn) / (2.0 * h) / y0)
        for k, lab in enumerate(('N_total', 'N_bin0')):
            r1 = abs(ad[k] - fds[0][k]) / max(abs(ad[k]), 1e-300)
            r2 = abs(ad[k] - fds[1][k]) / max(abs(ad[k]), 1e-300)
            rows.append((name, lab, ad[k], fds[0][k], fds[1][k], r1, r2))
            print(f"  {name:<8s} {lab:<8s} {ad[k]: .6e} {fds[0][k]: .6e} "
                  f"{fds[1][k]: .6e} {r1:8.1e} {r2:8.1e}")
    worst = max(min(r[5], r[6]) for r in rows)
    print(f"  worst min(rel1, rel2) = {worst:.1e}")
    ok &= worst < 1e-3

    # Reverse mode: expected to fail on the dynamic-trip-count fori_loop
    print("\n  reverse mode (jax.grad) through the same run:")
    try:
        jax.grad(lambda p: outputs(p)[0])(Pj)
        print("    jax.grad SUCCEEDED (unexpected; revisit forward-mode-only note)")
    except Exception as e:  # noqa: BLE001 - we report whatever JAX raises
        print(f"    {type(e).__name__}: {str(e).splitlines()[0][:160]}")

    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(os.path.join(OUTDIR, 'part0c_jacfwd_1h.npz'),
             fields=np.array(P._fields), y0=y0,
             jac=np.array([np.array(getattr(J, n)) for n in P._fields]),
             n_sub=n_sub, clamp=clamp, h2so4=h2, J=np.array(diag['J']),
             N_total=np.array(diag['N_total']))

    # Ranked elasticities d ln y / d ln theta (no sigma yet: Phase 2)
    print("\n  elasticity d ln(N_total) / d ln(theta), top 10:")
    el = {n: float(np.array(getattr(J, n))[0]) * getattr(P, n) / y0[0]
          for n in P._fields}
    for n, v in sorted(el.items(), key=lambda kv: -abs(kv[1]))[:10]:
        print(f"    {n:<10s} {v: .4e}")

    if not ok:
        raise RuntimeError("Phase 0c checks FAILED (see table above)")
    print("0c: all checks PASS")


PARTS = {'0a': part_0a, '0b': part_0b, '0c': part_0c}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--part', choices=list(PARTS) + ['all'], default='all')
    args = parser.parse_args()
    for name, fn in PARTS.items():
        if args.part in (name, 'all'):
            fn()


if __name__ == '__main__':
    main()
