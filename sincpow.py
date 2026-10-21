"""Sinc-power window of the NFFT: rates, optimal width and worst-case error.

References such as (3.1) or Lemma 4.1 refer to the paper. Potts (2003) is the
habilitation thesis that states the bound of Satz 1.8.
"""

from collections.abc import Callable
from functools import cache
from itertools import pairwise, product
from operator import itemgetter
from typing import NamedTuple

import mpmath as mp

mp.mp.dps = 50

type Number = int | float | str | mp.mpf


def root(f: Callable[[mp.mpf], mp.mpf], lo: Number, hi: Number) -> mp.mpf:
    """Bisection on [lo, hi]; f(lo) and f(hi) must have opposite signs."""
    return mp.findroot(f, (mp.mpf(lo), mp.mpf(hi)), solver="bisect", maxsteps=400)


# Cramer rate function of U[0,1], Section 4
def lam(y: Number) -> mp.mpf:
    """Tilt lambda_y of (4.5): the tilted U[0,1] has mean y, y in (1/2, 1)."""

    def g(l):
        return 1 / (-mp.expm1(-l)) - 1 / l - y

    lo, hi = mp.mpf("1e-30"), mp.mpf(1)
    while g(hi) < 0:
        hi *= 2
    return root(g, lo, hi)


def I(y: Number) -> mp.mpf:
    """I(y) = sup_l [l y - log((e^l - 1)/l)] of (4.6); 0 for y <= 1/2, inf for y >= 1."""
    y = mp.mpf(y)
    if y <= mp.mpf(1) / 2:
        return mp.mpf(0)
    if y >= 1:
        return mp.inf
    l = lam(y)
    return l * y - mp.log(mp.expm1(l) / l)


def tilt_var(y: Number) -> mp.mpf:
    """Variance v(lambda_y) of the tilted uniform law."""
    l = lam(y)
    return 1 / l**2 - mp.exp(l) / mp.expm1(l) ** 2


# Centred cardinal B-spline M_p, supp [-p/2, p/2]
def _run(p: int, y: Number) -> list[mp.mpf]:
    """[M_p(y + j - p/2)]_{j=0..p-1} for y in [0,1). All terms are positive, so no cancellation."""
    a = [mp.mpf(1)]
    for q in range(2, p + 1):
        # Cox-de Boor recursion; the zero padding drops the missing terms at both ends.
        a = [((y + j) * hi + (q - (y + j)) * lo) / (q - 1) for j, (lo, hi) in enumerate(pairwise([0, *a, 0]))]
    return a


def M(p: int, x: Number) -> mp.mpf:
    x = mp.mpf(x) + mp.mpf(p) / 2
    if x <= 0 or x >= p:
        return mp.mpf(0)
    j = int(mp.floor(x))
    return _run(p, x - j)[j]


# Window in grid units (2.7): omega(t) = sinc(pi c t/m)^(2m),
# Phi(theta) = int omega e^{-i theta t} dt = (m/c) M_2m(m theta/(2 pi c)).
def omega(t: Number, m: int, c: Number) -> mp.mpf:
    return mp.sinc(mp.pi * c * t / m) ** (2 * m)


def Phi(theta: Number, m: int, c: Number) -> mp.mpf:
    return (m / c) * M(2 * m, m * theta / (2 * mp.pi * c))


# Constants of sinc, Section 5
def _peak(lo: Number, hi: Number) -> mp.mpf:
    """Root of tan x = x, i.e. a local extremum of |sinc|, in (lo, hi)."""
    return root(lambda x: mp.sin(x) - x * mp.cos(x), lo, hi)


X1 = _peak(4.4, 4.6)  # first sidelobe peak
S1 = abs(mp.sin(X1) / X1)  # s_1 = 0.21723...
_PEAKS = [_peak(k * mp.pi + 0.01, (k + 0.5) * mp.pi - 0.01) for k in range(1, 12)]
C_S = root(lambda c: mp.sinc(mp.pi * c) - S1, "0.5", "0.99")  # c_s = 0.81282...
SIGMA_S = 1 / (2 * (1 - C_S))  # sigma_s = 2.6713...


def S(c: Number) -> mp.mpf:
    """S(c) = sup_{x >= pi c} |sinc x| of (6.1). _PEAKS covers c up to about 11."""
    x0 = mp.pi * c
    return max([abs(mp.sinc(x0))] + [abs(mp.sin(x) / x) for x in _PEAKS if x >= x0])


def c0(s: Number) -> mp.mpf:
    return 1 - 1 / (2 * mp.mpf(s))


def c_of_a(s: Number, a: Number) -> mp.mpf:
    """Width c = 1 - a/sigma of the window (7.1); a = 1/2 gives c0."""
    return 1 - mp.mpf(a) / mp.mpf(s)


def y_e(s: Number, c: Number) -> mp.mpf:
    return mp.mpf(1) / 2 + 1 / (4 * c * mp.mpf(s))


def y_a(s: Number, c: Number) -> mp.mpf:
    return mp.mpf(1) / 2 + c0(s) / (2 * c)


# Exponential rates per unit m
def r(s: Number) -> mp.mpf:
    """Rate r(sigma) of (6.1) for the window (2.5), Theorem 6.1."""
    s = mp.mpf(s)
    return 2 * (mp.log(S(c0(s))) + I(s / (2 * s - 1)))


def r_satz(s: Number) -> mp.mpf:
    """Rate of the bound of Potts (2003), Satz 1.8."""
    s = mp.mpf(s)
    return 2 * mp.log(s / (2 * s - 1))


def rho_T(s: Number, c: Number) -> mp.mpf:
    return 2 * (mp.log(S(c)) + I(y_e(s, c)))


def rho_A(s: Number, c: Number) -> mp.mpf:
    ya = y_a(s, c)
    return 2 * (I(y_e(s, c)) - I(ya)) if ya < 1 else -mp.inf


def rho(s: Number, c: Number) -> mp.mpf:
    """rho = max(rho_T, rho_A) of (7.3)."""
    return max(rho_T(s, c), rho_A(s, c))


class OptimalWidth(NamedTuple):
    a: mp.mpf
    c: mp.mpf
    rho: mp.mpf


def astar(s: Number) -> OptimalWidth:
    """Root c* of (7.4) in (c0, 1) by bisection; returns (a*, c*, rho*).

    rho_T - rho_A decreases in c. If it is still positive at c = 1, then a* < 0
    and the limit a -> 0+ (c = 1) is returned.
    """
    s = mp.mpf(s)

    def f(c):
        return mp.log(S(c)) + I(y_a(s, c))

    eps = mp.mpf(10) ** (-mp.mp.dps // 2)
    c = 1 if f(1 - eps) > 0 else root(f, c0(s) + eps, 1 - eps)
    return OptimalWidth(s * (1 - c), c, rho(s, c))


def satz_bound(s: Number, m: int) -> mp.mpf:
    """Bound B(m) of Potts (2003), Satz 1.8."""
    s = mp.mpf(s)
    return (1 / mp.mpf(2 * m - 1)) * (4 / s ** (2 * m) + (s / (2 * s - 1)) ** (2 * m - 1))


# Worst-case error, (3.1) and (3.3)
def err(theta: Number, delta: Number, m: int, c: Number, phi: mp.mpf | None = None) -> mp.mpf:
    """|1 - R(theta, delta)| of (3.1) with truncation radius m. phi = Phi(theta, m, c) if given."""
    if phi is None:
        phi = Phi(theta, m, c)
    js = [j for j in range(-m - 2, m + 2) if abs(delta + j) <= m]
    acc = mp.fsum(omega(delta + j, m, c) * mp.expj(-theta * (delta + j)) for j in js)
    return abs(acc / phi - 1)


class Point(NamedTuple):
    err: mp.mpf
    theta: mp.mpf
    delta: mp.mpf


def worst_case_error(
    s: Number,
    m: int,
    c: Number | None = None,
    nt: int | None = None,
    nd: int = 24,
    starts: int = 3,
    tol: Number = "1e-10",
) -> tuple[mp.mpf, tuple[float, float]]:
    """E_inf(m) of (3.3): sup of |1 - R| over theta in [0, pi/sigma] and delta in (0, 1).

    c defaults to c0, the window (2.5). R oscillates in theta with period about
    theta_e sigma/m, hence a grid of nt + 1 = max(25, 8m + 1) values of theta and
    nd values of delta, followed by a pattern search from the best `starts` grid
    points down to step `tol`. For c = c0 the sup is approached as delta -> 0+,
    so delta = 1e-20 is on the grid.
    Returns (sup, (theta/theta_e, delta)).
    """
    s = mp.mpf(s)
    c = c0(s) if c is None else mp.mpf(c)
    nt = max(24, 8 * m) if nt is None else nt
    te = mp.pi / s
    dlo, dhi = mp.mpf("1e-20"), 1 - mp.mpf("1e-20")
    phi = cache(lambda th: Phi(th, m, c))

    def e(th, d):
        return Point(err(th, d, m, c, phi(th)), th, d)

    deltas = [dlo, *(mp.mpf(i) / nd for i in range(1, nd))]
    pts = [e(te * it / nt, d) for it, d in product(range(nt + 1), deltas)]
    pts.sort(key=itemgetter(0), reverse=True)
    best = pts[0]
    tol = mp.mpf(tol)
    for v, th, d in pts[:starts]:
        ht, hd = te / nt, mp.mpf(1) / nd
        while hd > tol:
            neighbours = {
                (min(max(th + i * ht, 0), te), min(max(d + k * hd, dlo), dhi)) for i, k in product((-1, 0, 1), repeat=2)
            }
            top = max((e(t, x) for t, x in neighbours - {(th, d)}), key=itemgetter(0))
            if top.err > v:
                v, th, d = top
            else:
                ht, hd = ht / 2, hd / 2
        if v > best.err:
            best = Point(v, th, d)
    return best.err, (float(best.theta / te), float(best.delta))
