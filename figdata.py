"""Data for the figures. Run: uv run python figdata.py (writes data/*.csv)."""

import csv
from collections.abc import Iterable
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import mpmath as mp

import sincpow as sp

OUT = Path(__file__).resolve().parent / "data"
MS = range(2, 33)
SIG_A05 = ("1.25", "1.28125", "1.3125", "1.5")
SIG_A03 = ("1.1", "1.25", "1.5", "2")


class Task(NamedTuple):
    sigma: str
    m: int
    a: str | None


def job(task: Task) -> float:
    mp.mp.dps = 50
    c = None if task.a is None else sp.c_of_a(task.sigma, task.a)
    return float(sp.worst_case_error(task.sigma, task.m, c=c)[0])


def write(name: str, header: list[str], rows: Iterable[list]) -> None:
    # newline="" and an explicit line terminator give LF on every system.
    with (OUT / name).open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def sci(x) -> str:
    return f"{float(x):.8e}"


def einf() -> None:
    """E_inf(m) for a = 1/2 with the bound B(m) of Satz 1.8 (Figure 1), and for a = 0.3 (Figure 2)."""
    tasks = [Task(s, m, a) for sigmas, a in ((SIG_A05, None), (SIG_A03, "0.3")) for s, m in product(sigmas, MS)]
    with Pool() as p:
        e = dict(zip(tasks, p.map(job, tasks, chunksize=1), strict=True))
    mp.mp.dps = 50
    write(
        "einf_a05.csv",
        ["m", *(f"E_{s}" for s in SIG_A05), *(f"B_{s}" for s in SIG_A05)],
        ([m, *(sci(e[Task(s, m, None)]) for s in SIG_A05), *(sci(sp.satz_bound(s, m)) for s in SIG_A05)] for m in MS),
    )
    write(
        "einf_a03.csv",
        ["m", *(f"E_{s}" for s in SIG_A03)],
        ([m, *(sci(e[Task(s, m, "0.3")]) for s in SIG_A03)] for m in MS),
    )


def rho() -> None:
    """rho(sigma, a) for a = 1/2, 0.35, 0.3 and a*(sigma), sigma - 1 from 1e-3 to 1 (Figure 2)."""
    mp.mp.dps = 30
    rows = []
    for k in range(61):
        s = 1 + mp.mpf(10) ** (-3 + mp.mpf(k) / 20)
        rates = [sp.rho(s, sp.c_of_a(s, a)) for a in ("0.5", "0.35", "0.3")]
        a, _, v = sp.astar(s)
        rows.append([sci(s - 1), *(f"{float(x):.8f}" for x in (*rates, v, a))])
    write("rho.csv", ["s1", "R_0.5", "R_0.35", "R_0.3", "R_star", "a_star"], rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    einf()
    rho()
