"""Prints the numbers in the text and tables of the paper, in order of appearance.

Values of E_inf(m) are read from data/*.csv, see figdata.py.

Run: uv run python paper_numbers.py
"""

import csv
import os

import mpmath as mp

from sincpow import C_S, S1, SIGMA_S, X1, I, M, astar, c_of_a, r, rho, root

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def table(name):
    with open(os.path.join(DATA, name)) as f:
        return {int(row["m"]): row for row in csv.DictReader(f)}


print("Section 5")
print("  first sidelobe peak x1, s1 =", mp.nstr(X1, 11), mp.nstr(S1, 10))
print("  c_s                       =", mp.nstr(C_S, 10))

print("Section 6")
print("  sigma_s                   =", mp.nstr(SIGMA_S, 5))
print("  sigma_crit (r = 0)        =", mp.nstr(root(r, 1.1, 1.5), 20))
print("  Table 1: r(sigma)")
for s in ("1.1", "1.25", "1.28125", "1.3125", "1.5", "2", "4"):
    print("    %7s  %+.3f" % (s, r(s)))
e05 = table("einf_a05.csv")
print("  Figure 1: E_inf, sigma = 1.25: m = 2 %.2f, m = 32 %.0f" % (float(e05[2]["E_1.25"]), float(e05[32]["E_1.25"])))
e = [float(e05[m]["E_1.28125"]) for m in range(16, 33)]
print("  Figure 1: E_inf, sigma = 1.28125, 16 <= m <= 32: min %.3f, max %.3f" % (min(e), max(e)))
print("  Error 1: M_4(4/3) = %s, (2/pi)^4 = %.4f" % (mp.identify(M(4, mp.mpf(4) / 3)), (2 / mp.pi) ** 4))

print("Section 7")
t = lambda a: mp.log(mp.sinc(mp.pi * (1 - a))) + I(mp.mpf(1) / 2 + 1 / (4 * (1 - a)))
print("  a_1                       =", mp.nstr(root(t, 0.2, 0.4), 15))
print("  Table 2: sigma, a*, rho(a*), rho(0.3), rho(0.35), rho(0.5)")
gap = 0
for s in ("1.01", "1.1", "1.25", "1.5", "2", "4"):
    a, c, v = astar(s)
    rs = [rho(s, c_of_a(s, x)) for x in ("0.3", "0.35", "0.5")]
    gap = max(gap, rs[0] - v)
    print("    %5s  %.3f  %+.3f  %+.3f  %+.3f  %+.3f" % (s, a, v, *rs))
print("  max rho(0.3) - rho(a*) over Table 2 = %.4f" % gap)
print("  a* for sigma in [1.01, 1.6]:", " ".join("%.3f" % astar(s)[0] for s in ("1.01", "1.2", "1.4", "1.6")))
e03 = table("einf_a03.csv")
print(
    "  E_inf, sigma = 1.25, m = 16: a = 1/2 %.1f, a = 0.3 %.1e" % (float(e05[16]["E_1.25"]), float(e03[16]["E_1.25"]))
)
