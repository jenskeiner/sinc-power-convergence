"""Sinc-power window of the NFFT: rates, optimal width and worst-case error.

References such as (3.1) or Lemma 4.1 refer to the paper. Potts (2003) is the
habilitation thesis that states the bound of Satz 1.8.
"""

import mpmath as mp

mp.mp.dps = 50


def root(f, lo, hi, it=130):
    """Bisection; f(lo) and f(hi) must have opposite signs."""
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    slo = f(lo) > 0
    for _ in range(it):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == slo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# Cramer rate function of U[0,1], Section 4
def lam(y):
    """Tilt lambda_y of (4.5): the tilted U[0,1] has mean y, y in (1/2, 1)."""
    g = lambda l: 1 / (-mp.expm1(-l)) - 1 / l - y
    lo, hi = mp.mpf("1e-30"), mp.mpf(1)
    while g(hi) < 0:
        hi *= 2
    return root(g, lo, hi, 200)


def I(y):
    """I(y) = sup_l [l y - log((e^l - 1)/l)] of (4.6); 0 for y <= 1/2, inf for y >= 1."""
    y = mp.mpf(y)
    if y <= mp.mpf(1) / 2:
        return mp.mpf(0)
    if y >= 1:
        return mp.inf
    l = lam(y)
    return l * y - mp.log(mp.expm1(l) / l)


def tilt_var(y):
    """Variance v(lambda_y) of the tilted uniform law."""
    l = lam(y)
    return 1 / l**2 - mp.exp(l) / mp.expm1(l) ** 2


# Centred cardinal B-spline M_p, supp [-p/2, p/2]
def _run(p, y):
    """[M_p(y + j - p/2)]_{j=0..p-1} for y in [0,1). All terms are positive, so no cancellation."""
    a = [mp.mpf(1)]
    for q in range(2, p + 1):
        b = []
        for j in range(q):
            v = mp.mpf(0)
            if j < q - 1:
                v += (y + j) * a[j]
            if j > 0:
                v += (q - (y + j)) * a[j - 1]
            b.append(v / (q - 1))
        a = b
    return a


def M(p, x):
    x = mp.mpf(x) + mp.mpf(p) / 2
    if x <= 0 or x >= p:
        return mp.mpf(0)
    j = int(mp.floor(x))
    return _run(p, x - j)[j]


# Window in grid units (2.7): omega(t) = sinc(pi c t/m)^(2m),
# Phi(theta) = int omega e^{-i theta t} dt = (m/c) M_2m(m theta/(2 pi c)).
def omega(t, m, c):
    return mp.sinc(mp.pi * c * t / m) ** (2 * m)


def Phi(theta, m, c):
    return (m / c) * M(2 * m, m * theta / (2 * mp.pi * c))


# Constants of sinc, Section 5
X1 = mp.findroot(lambda x: mp.sin(x) - x * mp.cos(x), (4.4, 4.6), solver="anderson")  # first sidelobe peak
S1 = abs(mp.sin(X1) / X1)  # s_1 = 0.21723...
_PEAKS = [
    mp.findroot(lambda x: mp.sin(x) - x * mp.cos(x), (k * mp.pi + 0.01, (k + 0.5) * mp.pi - 0.01), solver="anderson")
    for k in range(1, 12)
]
C_S = root(lambda c: mp.sinc(mp.pi * c) - S1, "0.5", "0.99")  # c_s = 0.81282...
SIGMA_S = 1 / (2 * (1 - C_S))  # sigma_s = 2.6713...


def S(c):
    """S(c) = sup_{x >= pi c} |sinc x| of (6.1). _PEAKS covers c up to about 11."""
    x0 = mp.pi * c
    return max([abs(mp.sinc(x0))] + [abs(mp.sin(x) / x) for x in _PEAKS if x >= x0])


def c0(s):
    return 1 - 1 / (2 * mp.mpf(s))


def c_of_a(s, a):
    """Width c = 1 - a/sigma of the window (7.1); a = 1/2 gives c0."""
    return 1 - mp.mpf(a) / mp.mpf(s)


def y_e(s, c):
    return mp.mpf(1) / 2 + 1 / (4 * c * mp.mpf(s))


def y_a(s, c):
    return mp.mpf(1) / 2 + c0(s) / (2 * c)


# Exponential rates per unit m
def r(s):
    """Rate r(sigma) of (6.1) for the window (2.5), Theorem 6.1."""
    s = mp.mpf(s)
    return 2 * (mp.log(S(c0(s))) + I(s / (2 * s - 1)))


def r_satz(s):
    """Rate of the bound of Potts (2003), Satz 1.8."""
    s = mp.mpf(s)
    return 2 * mp.log(s / (2 * s - 1))


def rho_T(s, c):
    return 2 * (mp.log(S(c)) + I(y_e(s, c)))


def rho_A(s, c):
    ya = y_a(s, c)
    return 2 * (I(y_e(s, c)) - I(ya)) if ya < 1 else -mp.inf


def rho(s, c):
    """rho = max(rho_T, rho_A) of (7.3)."""
    return max(rho_T(s, c), rho_A(s, c))


def astar(s):
    """Root c* of (7.4) in (c0, 1) by bisection; returns (a*, c*, rho*).

    rho_T - rho_A decreases in c. If it is still positive at c = 1, then a* < 0
    and the limit a -> 0+ (c = 1) is returned.
    """
    s = mp.mpf(s)
    f = lambda c: mp.log(S(c)) + I(y_a(s, c))
    eps = mp.mpf(10) ** (-mp.mp.dps // 2)
    c = 1 if f(1 - eps) > 0 else root(f, c0(s) + eps, 1 - eps)
    return s * (1 - c), c, rho(s, c)


def satz_bound(s, m):
    """Bound B(m) of Potts (2003), Satz 1.8."""
    s = mp.mpf(s)
    return (1 / mp.mpf(2 * m - 1)) * (4 / s ** (2 * m) + (s / (2 * s - 1)) ** (2 * m - 1))


# Worst-case error, (3.1) and (3.3)
def err(theta, delta, m, c, phi=None):
    """|1 - R(theta, delta)| of (3.1) with truncation radius m. phi = Phi(theta, m, c) if given."""
    phi = Phi(theta, m, c) if phi is None else phi
    js = [j for j in range(-m - 2, m + 2) if abs(delta + j) <= m]
    acc = mp.fsum(omega(delta + j, m, c) * mp.expj(-theta * (delta + j)) for j in js)
    return abs(acc / phi - 1)


def worst_case_error(s, m, c=None, nt=None, nd=24, starts=3, tol="1e-10"):
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
    phis = {}

    def e(th, d):
        if th not in phis:
            phis[th] = Phi(th, m, c)
        return err(th, d, m, c, phis[th])

    pts = [
        (e(te * it / nt, d), te * it / nt, d)
        for it in range(nt + 1)
        for d in [dlo] + [mp.mpf(i) / nd for i in range(1, nd)]
    ]
    pts.sort(key=lambda x: x[0], reverse=True)
    best = pts[0]
    tol = mp.mpf(tol)
    for v, th, d in pts[:starts]:
        ht, hd = te / nt, mp.mpf(1) / nd
        while hd > tol:
            nb = [
                (e(t, x), t, x)
                for t, x in {
                    (min(max(th + i * ht, 0), te), min(max(d + k * hd, dlo), dhi))
                    for i in (-1, 0, 1)
                    for k in (-1, 0, 1)
                }
                - {(th, d)}
            ]
            top = max(nb, key=lambda x: x[0])
            if top[0] > v:
                v, th, d = top
            else:
                ht, hd = ht / 2, hd / 2
        if v > best[0]:
            best = (v, th, d)
    return best[0], (float(best[1] / te), float(best[2]))
