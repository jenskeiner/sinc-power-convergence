"""Data for the figures. Run: uv run python figdata.py (writes data/*.csv)."""

import csv
import os
from multiprocessing import Pool

import mpmath as mp

import sincpow as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MS = list(range(2, 33))
SIG_A05 = ("1.25", "1.28125", "1.3125", "1.5")
SIG_A03 = ("1.1", "1.25", "1.5", "2")


def job(args):
    sigma, m, a = args
    mp.mp.dps = 50
    c = None if a is None else T.c_of_a(sigma, a)
    return float(T.worst_case_error(sigma, m, c=c)[0])


def write(name, header, rows):
    # Text mode turns '\n' into the line ending of the system.
    with open(os.path.join(OUT, name), "w") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def einf():
    """E_inf(m) for a = 1/2 with the bound B(m) of Satz 1.8 (Figure 1), and for a = 0.3 (Figure 2)."""
    tasks = [(s, m, None) for s in SIG_A05 for m in MS] + [(s, m, "0.3") for s in SIG_A03 for m in MS]
    with Pool() as p:
        e = dict(zip(tasks, p.map(job, tasks, chunksize=1), strict=True))
    mp.mp.dps = 50
    write(
        "einf_a05.csv",
        ["m"] + ["E_" + s for s in SIG_A05] + ["B_" + s for s in SIG_A05],
        [[m] + ["%.8e" % e[s, m, None] for s in SIG_A05] + ["%.8e" % T.satz_bound(s, m) for s in SIG_A05] for m in MS],
    )
    write(
        "einf_a03.csv",
        ["m"] + ["E_" + s for s in SIG_A03],
        [[m] + ["%.8e" % e[s, m, "0.3"] for s in SIG_A03] for m in MS],
    )


def rho():
    """rho(sigma, a) for a = 1/2, 0.35, 0.3 and a*(sigma), sigma - 1 from 1e-3 to 1 (Figure 2)."""
    mp.mp.dps = 30
    rows = []
    for k in range(0, 61):
        s = 1 + mp.mpf(10) ** (-3 + mp.mpf(k) / 20)
        r = [T.rho(s, T.c_of_a(s, a)) for a in ("0.5", "0.35", "0.3")]
        a, c, v = T.astar(s)
        rows.append(["%.8e" % (s - 1)] + ["%.8f" % x for x in r + [v, a]])
    write("rho.csv", ["s1", "R_0.5", "R_0.35", "R_0.3", "R_star", "a_star"], rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    einf()
    rho()
