# sinc-power-convergence

Code for the paper

> J. Keiner, *The sinc-power window of the NFFT does not always converge*, 2026.

It computes all numbers, tables and figure data in the paper in 30- to 50-digit
arithmetic with [mpmath](https://mpmath.org). Equation, lemma and theorem numbers
in the code refer to the paper.

## Requirements

[uv](https://docs.astral.sh/uv/). `uv.lock` pins Python dependencies (mpmath 1.3.0, pure Python backend).

## Files

| File | Content |
|---|---|
| `sincpow.py` | Library: rate function `I`, B-spline `M`, window `omega`, `Phi`, rates `r`, `rho`, optimal width `astar`, worst-case error `worst_case_error` |
| `paper_numbers.py` | Prints the numbers in the text, Table 1 and Table 2 (a few seconds) |
| `checks.py` | Checks Lemma 4.1, Lemma 5.1 and the bounds (6.2), (6.3) of Theorem 6.1 numerically (less than a second) |
| `figdata.py` | Writes the CSV files in `data/` (about 3 minutes on 10 cores) |

## Usage

```
uv run python paper_numbers.py
uv run python checks.py
uv run python figdata.py
```

## Figure data

| File | Figure | Content |
|---|---|---|
| `data/einf_a05.csv` | 1 | `E_<sigma>`: worst-case error for the original window (a = 1/2) |
| `data/einf_a03.csv` | 2, right | `E_<sigma>`: worst-case error for a = 0.3 |
| `data/rho.csv` | 2, left | `s1` = sigma - 1; `R_<a>`: rate rho(sigma, a); `R_star`, `a_star`: optimum of (7.4) |

The worst-case error (3.3) is the supremum of |1 - R| from (3.1) over theta in
[0, pi/sigma] and delta in (0, 1). `worst_case_error` evaluates it on a grid of
max(25, 8m + 1) values of theta and 24 values of delta, then refines the best
grid points by a pattern search down to step 1e-10.

## License

MIT, see `LICENSE`.
