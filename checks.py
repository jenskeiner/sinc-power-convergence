"""Numerical checks of Lemma 4.1, Lemma 5.1 and the bounds (6.2), (6.3) of Theorem 6.1.

Run: uv run python checks.py
"""

import mpmath as mp

from sincpow import S1, I, M, c0, err, lam, satz_bound, tilt_var


def gamma(c):
    """gamma(c) of Lemma 5.1."""
    return mp.exp(2 * mp.pi * c / mp.tan(mp.pi * c) - 2)


def beyond(m, c):
    """beta_m(c) s_1^(2m) of Lemma 5.1 (b)."""
    x = 1 / S1
    return S1 ** (2 * m) * (2 + (m * (x - mp.pi) + x) / (mp.pi * c))


def lemma_4_1() -> None:
    """Lower and upper bound of M_2p(2p(y - 1/2)) divided by its exact value."""
    for s in ("1.1", "1.25", "2"):
        sg = mp.mpf(s)
        y = sg / (2 * sg - 1)
        l = lam(y)
        v = tilt_var(y)
        for p in (1, 2, 4, 16, 64):
            Mv = M(2 * p, 2 * p * (y - mp.mpf(1) / 2))
            e = mp.exp(-2 * p * I(y))
            lo = e / (mp.e * mp.sqrt(24 * p * v))
            hi = l / (1 - mp.exp(-l)) * e
            print("sigma", s, "p", p, f"lo/M {float(lo / Mv):.3f}  hi/M {float(hi / Mv):.3f}")


def lemma_5_1() -> None:
    """gamma(c0) and the factors that enter (6.2) and (6.3)."""
    for s in ("1.01", "1.25", "2", "2.6"):
        q = gamma(c0(mp.mpf(s)))
        print(
            "sigma",
            s,
            f"gamma {float(q):.4f}  1-2gamma/(1-gamma) {float(1 - 2 * q / (1 - q)):.4f}  "
            f"2/(1-gamma) {float(2 / (1 - q)):.4f}",
        )


def theorem_6_1() -> None:
    """|R - 1| at theta_e for three delta at sigma = 5/4, the Satz 1.8 bound B, and the bounds (6.3), (6.2)."""
    sg = mp.mpf(5) / 4
    c = c0(sg)
    th = mp.pi / sg
    y = sg / (2 * sg - 1)
    l = lam(y)
    q = gamma(c)
    v = tilt_var(y)

    for m in range(2, 17):
        vals = {d: err(th, mp.mpf(d), m, c) for d in ("1e-6", "0.25", "0.5")}
        amp = mp.exp(2 * m * I(y))
        lower = (
            (c / m)
            * (1 - mp.exp(-l))
            / l
            * amp
            * (mp.sinc(mp.pi * c) ** (2 * m) * (1 - 2 * q / (1 - q)) - 2 * beyond(m, c))
        )
        upper = 2 * mp.e * c * mp.sqrt(24 * v / m) * amp * (mp.sinc(mp.pi * c) ** (2 * m) / (1 - q) + beyond(m, c))
        print(
            m,
            " ".join(f"{d}:{float(x):.4e}" for d, x in vals.items()),
            f"B {float(satz_bound(sg, m)):.3e}  lower {float(lower):.3e}  upper {float(upper):.3e}",
        )


def main() -> None:
    mp.mp.dps = 40
    lemma_4_1()
    lemma_5_1()
    theorem_6_1()


if __name__ == "__main__":
    main()
