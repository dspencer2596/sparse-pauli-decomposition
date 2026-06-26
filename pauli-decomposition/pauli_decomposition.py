"""
pauli_decomposition_simple.py

@author: Daniel J. Spencer @UMD <djspence@umd.edu>
@updated: June 24, 2026

A simpler implementation of the Pauli decomposition algorithm developed in
our work.
"""

from dataclasses import dataclass
from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time


Label = tuple[int, int]  # (x, z)

@dataclass(frozen=True)
class PauliTerm:
    x: int
    z: int
    alpha: complex


def parity(x: int) -> int:
    return x.bit_count() & 1


def chi(z: int, u: int) -> int:
    """ Walsh character chi_z(u) = (-1)^(z dot u). """
    return -1 if parity(z & u) else 1


def i_power(k: int) -> complex:
    return (1, 1j, -1, -1j)[k & 3]


def alpha_to_beta(alpha: complex, x: int, z: int) -> complex:
    """ beta_x(z) = i^|x and z| alpha_{x,z}. """
    return i_power((x & z).bit_count()) * alpha


def beta_to_alpha(beta: complex, x: int, z: int) -> complex:
    return beta / i_power((x & z).bit_count())


def fwht(values: np.ndarray) -> np.ndarray:
    """ Unnormalized Walsh-Hadamard transform. """
    out = values.astype(complex, copy=True)
    h = 1
    while h < len(out):
        out.shape = (-1, 2 * h)
        left = out[:, :h].copy()
        right = out[:, h:]
        out[:, :h] = left + right
        out[:, h:] = left - right
        out.shape = (len(values),)
        h *= 2
    return out


def hash_z(rows: list[int], z: int) -> int:
    """ Return Rz, where each row of R is stored as an integer bit mask. """
    s = 0
    for i, row in enumerate(rows):
        s |= parity(row & z) << i
    return s


def transpose_row_span(rows: list[int]) -> list[int]:
    """ Return all vectors R^T t, indexed by t in {0,1}^m. """
    points = [0] * (1 << len(rows))
    for t in range(1, len(points)):
        lowest_bit = t & -t
        j = lowest_bit.bit_length() - 1
        points[t] = points[t ^ lowest_bit] ^ rows[j]
    return points


class SparsePauliOracle:
    """ Row-query access to a planted sparse Pauli expansion. """

    def __init__(self, n: int, terms: list[PauliTerm], tol: float = 1e-10):
        self.n = n
        self.tol = tol
        self.query_count = 0
        self.alpha: dict[Label, complex] = {}
        self.beta_by_x: dict[int, dict[int, complex]] = {}

        for term in terms:
            label = (term.x, term.z)
            self.alpha[label] = self.alpha.get(label, 0j) + term.alpha

        for (x, z), alpha in self.alpha.items():
            self.beta_by_x.setdefault(x, {})[z] = alpha_to_beta(alpha, x, z)

    @property
    def active_x(self) -> set[int]:
        return set(self.beta_by_x)

    def true_px(self, x: int) -> int:
        return len(self.beta_by_x.get(x, {}))

    def reset_query_count(self) -> None:
        self.query_count = 0

    def bx_exact(self, x: int, u: int) -> complex:
        """ Return b_x(u) = M[x xor u, u] without counting a query. """
        return sum(beta * chi(z, u) for z, beta in self.beta_by_x.get(x, {}).items())

    def bx(self, x: int, u: int) -> complex:
        """ Query one entry of row x xor u. """
        self.query_count += 1
        return self.bx_exact(x, u)

    def row(self, v: int) -> list[tuple[int, complex]]:
        """ Return the nonzero entries in row v as (column, value) pairs. """
        self.query_count += 1
        entries = []
        for x in self.beta_by_x:
            u = x ^ v
            value = self.bx_exact(x, u)
            if abs(value) > self.tol:
                entries.append((u, value))
        return entries

    def dense_matrix(self) -> np.ndarray:
        """ Build the dense matrix, only for small-n comparisons. """
        N = 1 << self.n
        M = np.zeros((N, N), dtype=complex)
        for x, beta_slice in self.beta_by_x.items():
            for z, beta in beta_slice.items():
                for u in range(N):
                    M[x ^ u, u] += beta * chi(z, u)
        return M


def discover_x_support(oracle: SparsePauliOracle, k: int, eta: float, rng: random.Random) -> set[int]:
    X = set()
    for _ in range(math.ceil(k * math.log(k / eta))):
        v = rng.getrandbits(oracle.n)
        for u, _ in oracle.row(v):
            X.add(u ^ v)
    return X


def sign_if_pm_one(value: complex, tol: float) -> int | None:
    if abs(value - 1) <= tol:
        return 1
    if abs(value + 1) <= tol:
        return -1
    return None


def decode_unique_x(oracle: SparsePauliOracle,
                    x: int,
                    k: int,
                    eta: float,
                    rng: random.Random,
                    tol: float) -> tuple[int, complex] | None:
    """ Try to certify that the x-slice contains exactly one z. """
    beta = oracle.bx(x, 0)
    if abs(beta) <= tol:
        return None

    z = 0
    for j in range(oracle.n):
        sign = sign_if_pm_one(oracle.bx(x, 1 << j) / beta, tol)
        if sign is None:
            return None
        if sign == -1:
            z |= 1 << j

    for _ in range(math.ceil((k + 1) * math.log(1 / eta))):
        u = rng.getrandbits(oracle.n)
        if abs(oracle.bx(x, u) - beta * chi(z, u)) > tol:
            return None
    return z, beta


def folded_spectrum(oracle: SparsePauliOracle,
                    x: int,
                    rows: list[int],
                    points: list[int],
                    shift: int,
                    recovered: dict[int, complex]) -> np.ndarray:
    """ Fold the residual x-slice by R and compute its small WHT. """
    samples = np.array([oracle.bx(x, u ^ shift) for u in points], dtype=complex)
    spectrum = fwht(samples) / len(samples)

    for z, beta in recovered.items():
        spectrum[hash_z(rows, z)] -= beta * chi(z, shift)
    return spectrum


def decode_degenerate_x(oracle: SparsePauliOracle,
                        x: int,
                        pmax: int,
                        delta: float,
                        rng: random.Random,
                        c: int = 8,
                        tol: float = 1e-8) -> tuple[dict[int, complex], bool]:
    """ Recover a non-singleton x-slice by repeated folded sparse WHTs. """
    n = oracle.n
    m = min(n, math.ceil(math.log2(c * pmax)))
    B = 1 << m
    rounds = math.ceil(math.log(max(3 * pmax / delta, 2), c))
    certifications = math.ceil((pmax + 1) * math.log(3 * B * rounds / delta))
    residual_checks = math.ceil(pmax * math.log(3 / delta))
    recovered: dict[int, complex] = {}

    for _ in range(rounds):
        rows = [rng.getrandbits(n) for _ in range(m)]
        points = transpose_row_span(rows)

        base = folded_spectrum(oracle, x, rows, points, 0, recovered)
        basis = [folded_spectrum(oracle, x, rows, points, 1 << j, recovered) for j in range(n)]

        shifts = [rng.getrandbits(n) for _ in range(certifications)]
        shifted = [folded_spectrum(oracle, x, rows, points, w, recovered) for w in shifts]

        for s, beta in enumerate(base):
            if abs(beta) <= tol:
                continue

            z = 0
            for j in range(n):
                sign = sign_if_pm_one(basis[j][s] / beta, tol)
                if sign is None:
                    break
                if sign == -1:
                    z |= 1 << j
            else:
                hashes_to_s = hash_z(rows, z) == s
                is_new = z not in recovered
                passes_certification = all(abs(g[s] - beta * chi(z, w)) <= tol for w, g in zip(shifts, shifted))
                if hashes_to_s and is_new and passes_certification:
                    recovered[z] = beta

    for _ in range(residual_checks):
        u = rng.getrandbits(n)
        residual = oracle.bx(x, u)
        residual -= sum(beta * chi(z, u) for z, beta in recovered.items())
        if abs(residual) > tol:
            return recovered, False
    return recovered, True


def pauli_decompose(oracle: SparsePauliOracle,
                    k: int,
                    delta: float,
                    tol: float = 1e-8,
                    seed: int | None = None,
                    c: int = 8) -> tuple[dict[Label, complex], bool]:
    """ Run the full randomized sparse Pauli decomposition algorithm. """
    rng = random.Random(seed)
    eta_X = delta / 3
    eta_x = delta / (3 * k)
    X = discover_x_support(oracle, k, eta_X, rng)

    coefficients: dict[Label, complex] = {}
    success = True

    for x in sorted(X):
        unique = decode_unique_x(oracle, x, k, eta_x, rng, tol)
        if unique is not None:
            z, beta = unique
            coefficients[(x, z)] = beta_to_alpha(beta, x, z)
            continue

        beta_slice, passed_final_check = decode_degenerate_x(oracle, x, k, eta_x, rng, c=c, tol=tol)
        success = success and passed_final_check
        for z, beta in beta_slice.items():
            coefficients[(x, z)] = beta_to_alpha(beta, x, z)

    return coefficients, success


def random_coefficient(rng: random.Random) -> complex:
    coeff = 0j
    while coeff == 0:
        coeff = complex(rng.randint(-9, 9), rng.randint(-9, 9)) / 3
    return coeff


def random_composition_with_mixed_slices(k: int, s: int, rng: random.Random) -> list[int]:
    """ Partition k terms over s active x-slices, with a degenerate slice if possible. """
    if not 1 <= s <= k:
        raise ValueError("Need 1 <= s <= k")

    parts = [1] * s
    remaining = k - s
    if remaining:
        parts[0] += 1
        remaining -= 1
    while remaining:
        parts[rng.randrange(s)] += 1
        remaining -= 1
    rng.shuffle(parts)
    return parts


def planted_instance(n: int,
                     k: int,
                     seed: int | None = None,
                     mode: str = "mixed",
                     n_x_slices: int | None = None) -> list[PauliTerm]:
    """ Make a planted Pauli instance. """
    rng = random.Random(seed)
    N = 1 << n
    if k > N * N:
        raise ValueError("k cannot exceed 4^n distinct Pauli labels")

    if mode == "single_x":
        num_x_slices = 1
    elif mode == "unique_only":
        num_x_slices = k
    elif mode == "mixed":
        num_x_slices = min(k, n_x_slices if n_x_slices is not None else max(2, min(n, k)))
    else:
        raise ValueError("mode must be 'mixed', 'single_x', or 'unique_only'")

    x_values = rng.sample(range(N), num_x_slices)
    if mode == "single_x":
        slice_sizes = [k]
    elif mode == "unique_only":
        slice_sizes = [1] * k
    else:
        slice_sizes = random_composition_with_mixed_slices(k, num_x_slices, rng)

    terms = []
    for x, size in zip(x_values, slice_sizes):
        if size > N:
            raise ValueError("A single x-slice cannot contain more than 2^n z labels")
        for z in rng.sample(range(N), size):
            terms.append(PauliTerm(x, z, random_coefficient(rng)))
    rng.shuffle(terms)
    return terms


def int_to_pauli(n: int, x: int, z: int) -> str:
    letters = []
    for j in range(n - 1, -1, -1):
        xb = (x >> j) & 1
        zb = (z >> j) & 1
        letters.append(("I", "Z", "X", "Y")[(xb << 1) | zb])
    return "".join(letters)


def format_decomposition(coefficients: dict[Label, complex], n: int) -> str:
    pieces = []
    for (x, z), alpha in sorted(coefficients.items()):
        pieces.append(f"({alpha:.3g}) {int_to_pauli(n, x, z)}")
    return " + ".join(pieces)


def max_abs_error(true: dict[Label, complex], recovered: dict[Label, complex]) -> float:
    labels = set(true) | set(recovered)
    return max(abs(true.get(label, 0) - recovered.get(label, 0)) for label in labels)


def compare_decompositions(true: dict[Label, complex],
                           recovered: dict[Label, complex],
                           tol: float = 1e-7) -> dict[str, float | int | bool]:
    labels = set(true) | set(recovered)
    max_err = 0.0
    l2_err_sq = 0.0
    missing = 0
    extra = 0
    wrong = 0
    for label in labels:
        true_value = true.get(label, 0j)
        recovered_value = recovered.get(label, 0j)
        err = abs(recovered_value - true_value)
        max_err = max(max_err, err)
        l2_err_sq += err * err
        if label in true and label not in recovered:
            missing += 1
        elif label in recovered and label not in true:
            extra += 1
        elif err > tol:
            wrong += 1
    return {
        "exact": missing == 0 and extra == 0 and wrong == 0 and max_err <= tol,
        "missing": missing,
        "extra": extra,
        "wrong_coeffs": wrong,
        "max_abs_err": max_err,
        "l2_err": math.sqrt(l2_err_sq),
    }


def run_trial(n: int,
              k: int,
              trial_seed: int,
              delta: float,
              mode: str = "mixed",
              c: int = 8,
              tol: float = 1e-8,
              run_pennylane: bool = False) -> dict[str, float | int | bool | str]:
    terms = planted_instance(n, k, seed=trial_seed, mode=mode)
    oracle = SparsePauliOracle(n, terms, tol=tol / 10)
    oracle.reset_query_count()
    t0 = time.perf_counter()
    recovered, residual_check_passed = pauli_decompose(
        oracle,
        k=k,
        delta=delta,
        tol=tol,
        seed=trial_seed + 987654,
        c=c,
    )
    t_algorithm = time.perf_counter() - t0
    metrics = compare_decompositions(oracle.alpha, recovered, tol=1e-6)

    t_pennylane = np.nan
    err_pennylane = np.nan
    if run_pennylane:
        try:
            import pennylane as qml
            from scipy import sparse
        except ImportError as exc:
            raise RuntimeError("Install pennylane and scipy, or set run_pennylane=False") from exc

        M = oracle.dense_matrix()
        t0 = time.perf_counter()
        H = qml.pauli_decompose(sparse.csr_matrix(M), pauli=True, check_hermitian=False)
        t_pennylane = time.perf_counter() - t0
        M_recon = np.array(qml.matrix(H, wire_order=range(n)), dtype=complex)
        err_pennylane = float(np.max(np.abs(M_recon - M)))

    active_counts = [oracle.true_px(x) for x in sorted(oracle.active_x)]
    return {
        "n": n,
        "N": 1 << n,
        "k": k,
        "trial_seed": trial_seed,
        "mode": mode,
        "t_algorithm": t_algorithm,
        "t_pennylane": t_pennylane,
        "err_pennylane": err_pennylane,
        "ran_pennylane": bool(run_pennylane),
        "query_count": int(oracle.query_count),
        "residual_check_passed": bool(residual_check_passed),
        "exact": bool(metrics["exact"]),
        "missing": int(metrics["missing"]),
        "extra": int(metrics["extra"]),
        "wrong_coeffs": int(metrics["wrong_coeffs"]),
        "max_abs_err": float(metrics["max_abs_err"]),
        "l2_err": float(metrics["l2_err"]),
        "x_true": len(oracle.active_x),
        "max_px_true": max(active_counts) if active_counts else 0,
        "num_degenerate_true": sum(1 for p in active_counts if p >= 2),
    }


def multi_trial_benchmark(nvals,
                          k_func,
                          n_trials: int,
                          delta: float,
                          mode: str,
                          base_seed: int = 42,
                          run_pennylane: bool = False,
                          pennylane_nmax: int | None = None,
                          c: int = 8) -> pd.DataFrame:
    rows = []
    for n in nvals:
        k = int(k_func(n))
        run_pennylane_for_n = run_pennylane and (pennylane_nmax is None or n <= pennylane_nmax)
        print(f"n={n}, k={k}, trials={n_trials}, pennylane={run_pennylane_for_n}")
        for trial in range(n_trials):
            trial_seed = base_seed + 1000 * n + trial
            rows.append(
                run_trial(
                    n=n,
                    k=k,
                    trial_seed=trial_seed,
                    delta=delta,
                    mode=mode,
                    c=c,
                    run_pennylane=run_pennylane_for_n,
                )
            )
    return pd.DataFrame(rows)


def has_pennylane_results(df: pd.DataFrame) -> bool:
    return "t_pennylane" in df.columns and df["t_pennylane"].notna().any()


def fit_power_law(ns: np.ndarray, y: np.ndarray) -> dict[str, object] | None:
    ns = ns.astype(float)
    y = y.astype(float)
    mask = np.isfinite(ns) & np.isfinite(y) & (ns > 0) & (y > 0)
    if mask.sum() < 2:
        return None

    fit_ns = ns[mask]
    fit_y = y[mask]
    exponent, log_prefactor = np.polyfit(np.log(fit_ns), np.log(fit_y), deg=1)
    fit_grid = np.linspace(fit_ns.min(), fit_ns.max(), 200)
    fitted = np.exp(log_prefactor) * fit_ns**exponent
    ss_res = float(np.sum((np.log(fit_y) - np.log(fitted)) ** 2))
    ss_tot = float(np.sum((np.log(fit_y) - np.log(fit_y).mean()) ** 2))
    return {
        "kind": "power",
        "plot_ns": fit_grid,
        "plot_y": np.exp(log_prefactor) * fit_grid**exponent,
        "prefactor": float(np.exp(log_prefactor)),
        "exponent": float(exponent),
        "r2": 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot,
    }


def fit_exponential(ns: np.ndarray, y: np.ndarray) -> dict[str, object] | None:
    ns = ns.astype(float)
    y = y.astype(float)
    mask = np.isfinite(ns) & np.isfinite(y) & (ns > 0) & (y > 0)
    if mask.sum() < 2:
        return None

    fit_ns = ns[mask]
    fit_y = y[mask]
    growth_rate, log_prefactor = np.polyfit(fit_ns, np.log(fit_y), deg=1)
    fit_grid = np.linspace(fit_ns.min(), fit_ns.max(), 200)
    fitted = np.exp(log_prefactor + growth_rate * fit_ns)
    ss_res = float(np.sum((np.log(fit_y) - np.log(fitted)) ** 2))
    ss_tot = float(np.sum((np.log(fit_y) - np.log(fit_y).mean()) ** 2))
    return {
        "kind": "exponential",
        "plot_ns": fit_grid,
        "plot_y": np.exp(log_prefactor + growth_rate * fit_grid),
        "prefactor": float(np.exp(log_prefactor)),
        "growth_rate": float(growth_rate),
        "r2": 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot,
    }


def choose_best_scaling_fit(ns: np.ndarray, y: np.ndarray):
    fits = [fit for fit in (fit_power_law(ns, y), fit_exponential(ns, y)) if fit is not None]
    if not fits:
        return None, []
    return max(fits, key=lambda fit: fit["r2"]), fits


def format_decimal(value: float, significant_digits: int = 3) -> str:
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    decimal_places = max(0, significant_digits - exponent - 1)
    rounded = f"{value:.{decimal_places}f}"
    return rounded.rstrip("0").rstrip(".") if "." in rounded else rounded


def describe_scaling_fit(fit: dict[str, object], symbol: str) -> str:
    prefactor = format_decimal(float(fit["prefactor"]))
    if fit["kind"] == "power":
        return rf"${symbol} \approx {prefactor} n^{{{float(fit['exponent']):.2f}}}$"
    return rf"${symbol} \approx {prefactor} e^{{{float(fit['growth_rate']):.2f} n}}$"


def print_scaling_fits(label: str, symbol: str, best_fit: dict[str, object], fits: list[dict[str, object]]) -> None:
    print(f"{label} scaling fits:")
    for fit in fits:
        if fit["kind"] == "power":
            formula = f"{symbol} ~= {float(fit['prefactor']):.3e} * n^{float(fit['exponent']):.3f}"
        else:
            formula = f"{symbol} ~= {float(fit['prefactor']):.3e} * exp({float(fit['growth_rate']):.3f} n)"
        marker = "selected" if fit is best_fit else "not selected"
        print(f"  {fit['kind']}: {formula} (R^2={float(fit['r2']):.4f}, {marker})")


def set_power_of_two_x_ticks(ax: plt.Axes, ns: np.ndarray) -> None:
    positive_ns = np.asarray(ns, dtype=float)
    positive_ns = positive_ns[np.isfinite(positive_ns) & (positive_ns > 0)]
    if len(positive_ns) == 0:
        return

    min_exp = int(math.ceil(math.log2(float(positive_ns.min()))))
    max_exp = int(math.floor(math.log2(float(positive_ns.max()))))
    if min_exp > max_exp:
        tick = float(positive_ns.min())
        ax.set_xticks([tick])
        ax.set_xticklabels([f"{tick:g}"])
        return

    ticks = np.array([2**j for j in range(min_exp, max_exp + 1)], dtype=float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([rf"$2^{{{j}}}$" for j in range(min_exp, max_exp + 1)])


def print_summary(df: pd.DataFrame, run_pennylane: bool) -> None:
    show_pennylane = run_pennylane and has_pennylane_results(df)
    agg = df.groupby("n").agg(
        k=("k", "first"),
        trials=("exact", "count"),
        success_rate=("exact", "mean"),
        t_median_ms=("t_algorithm", lambda x: 1000 * x.median()),
        q_median=("query_count", "median"),
        err_max=("max_abs_err", "max"),
        missing_mean=("missing", "mean"),
        extra_mean=("extra", "mean"),
        x_true=("x_true", "median"),
        deg_true=("num_degenerate_true", "median"),
    ).reset_index()
    if show_pennylane:
        agg = agg.merge(
            df.groupby("n").agg(t_pl_median_ms=("t_pennylane", lambda x: 1000 * x.median())).reset_index(),
            on="n",
        )

    print("\n" + "=" * 100)
    if show_pennylane:
        print(f"{'n':>3} {'k':>5} {'succ':>7} {'alg ms':>10} {'PL ms':>10} {'queries':>10} {'err max':>11}")
    else:
        print(f"{'n':>3} {'k':>5} {'succ':>7} {'alg ms':>10} {'queries':>10} {'err max':>11}")
    print("=" * 100)
    for _, row in agg.iterrows():
        if show_pennylane:
            pl_ms = "--" if pd.isna(row.t_pl_median_ms) else f"{row.t_pl_median_ms:.3f}"
            print(f"{int(row.n):>3} {int(row.k):>5} {row.success_rate:>7.2f} {row.t_median_ms:>10.3f} {pl_ms:>10} {row.q_median:>10.0f} {row.err_max:>11.2e}")
        else:
            print(f"{int(row.n):>3} {int(row.k):>5} {row.success_rate:>7.2f} {row.t_median_ms:>10.3f} {row.q_median:>10.0f} {row.err_max:>11.2e}")
    print("=" * 100)


def plot_results(df: pd.DataFrame, run_pennylane: bool, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    show_pennylane = run_pennylane and has_pennylane_results(df)
    agg = df.groupby("n").agg(
        t_median=("t_algorithm", "median"),
        t_std=("t_algorithm", "std"),
        q_median=("query_count", "median"),
        q_std=("query_count", "std"),
    ).reset_index()

    if show_pennylane:
        agg = agg.merge(
            df.groupby("n").agg(
                t_pl_median=("t_pennylane", "median"),
                t_pl_std=("t_pennylane", "std"),
            ).reset_index(),
            on="n",
        )

    ns = agg["n"].to_numpy()
    y = agg["t_median"].to_numpy()
    ystd = agg["t_std"].fillna(0).to_numpy()
    runtime_fit, runtime_fits = choose_best_scaling_fit(ns, y)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(ns, np.maximum(y - ystd, 1e-12), y + ystd, color="#9370DB", alpha=0.25)
    ax.loglog(ns, y, "o-", color="#9370DB", label="Our algorithm")
    if runtime_fit is not None:
        ax.loglog(runtime_fit["plot_ns"], runtime_fit["plot_y"], "--", color="#4B0082", linewidth=2,
                  label=f"Fit: {describe_scaling_fit(runtime_fit, 't')}")
        ax.text(0.02, 0.96, rf"$R^2={float(runtime_fit['r2']):.3f}$", transform=ax.transAxes, va="top", ha="left")
        print_scaling_fits("Runtime", "t", runtime_fit, runtime_fits)
    if show_pennylane:
        y2 = agg["t_pl_median"].to_numpy()
        y2std = agg["t_pl_std"].fillna(0).to_numpy()
        pl_mask = np.isfinite(y2) & (y2 > 0)
        ax.fill_between(ns[pl_mask], np.maximum(y2[pl_mask] - y2std[pl_mask], 1e-12), y2[pl_mask] + y2std[pl_mask],
                        color="#B87333", alpha=0.25)
        ax.loglog(ns[pl_mask], y2[pl_mask], "s-", color="#B87333", label="PennyLane PauliDecompose")
        if pl_mask.sum() >= 2 and ns[pl_mask].max() < ns.max():
            pennylane_fit = fit_exponential(ns[pl_mask], y2[pl_mask])
            if pennylane_fit is not None:
                ns_future = np.linspace(float(ns[pl_mask].max()), float(ns.max()), 200)
                y_future = float(pennylane_fit["prefactor"]) * np.exp(float(pennylane_fit["growth_rate"]) * ns_future)
                ax.loglog(ns_future, y_future, "--", color="#B87333", linewidth=2, label="PennyLane extrapolation")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Runtime scaling")
    set_power_of_two_x_ticks(ax, ns)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend()
    runtime_path = outdir / "runtime.pdf"
    fig.savefig(runtime_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {runtime_path}")

    yq = agg["q_median"].to_numpy()
    yqstd = agg["q_std"].fillna(0).to_numpy()
    query_fit, query_fits = choose_best_scaling_fit(ns, yq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(ns, np.maximum(yq - yqstd, 1e-12), yq + yqstd, color="#9370DB", alpha=0.25)
    ax.loglog(ns, yq, "o-", color="#9370DB", label="Our algorithm")
    if query_fit is not None:
        ax.loglog(query_fit["plot_ns"], query_fit["plot_y"], "--", color="#4B0082", linewidth=2,
                  label=f"Fit: {describe_scaling_fit(query_fit, 'Q')}")
        ax.text(0.02, 0.96, rf"$R^2={float(query_fit['r2']):.3f}$", transform=ax.transAxes, va="top", ha="left")
        print_scaling_fits("Query", "Q", query_fit, query_fits)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Oracle queries")
    ax.set_title("Query scaling")
    set_power_of_two_x_ticks(ax, ns)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend()
    query_path = outdir / "queries.pdf"
    fig.savefig(query_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {query_path}")


def save_benchmark_results(df: pd.DataFrame, cache_path: Path, config: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"config": config, "results": df}, cache_path)
    print(f"Saved reusable results cache {cache_path}")


def load_benchmark_results(cache_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    payload = pd.read_pickle(cache_path)
    if isinstance(payload, pd.DataFrame):
        return payload, {}
    return payload["results"], dict(payload.get("config", {}))


def main() -> None:
    nmin = 2
    nmax = 8
    n_trials = 5
    delta = 0.1

    mode = "mixed"
    c = 8

    run_pennylane = True
    pennylane_nmax = 10
    make_plots = True
    outdir = Path("figures")
    load_saved_results = False
    results_stem = "simulation_results_mixed_pennylane" if run_pennylane else "simulation_results_nopennylane"
    cache_path = outdir / f"{results_stem}.pkl"
    csv_path = outdir / f"{results_stem}.csv"

    k_fixed: int | None = None
    k_linear_factor = 2.0
    k_density_cap = 0.5
    k_min = 3

    def k_func(n: int) -> int:
        if k_fixed is not None:
            return k_fixed
        linear_k = int(math.ceil(k_linear_factor * n))
        density_cap = int(k_density_cap * (1 << n))
        return max(k_min, min(linear_k, density_cap))

    config = {
        "nmin": nmin,
        "nmax": nmax,
        "n_trials": n_trials,
        "delta": delta,
        "mode": mode,
        "c": c,
        "run_pennylane": run_pennylane,
        "pennylane_nmax": pennylane_nmax,
        "k_fixed": k_fixed,
        "k_linear_factor": k_linear_factor,
        "k_density_cap": k_density_cap,
        "k_min": k_min,
    }

    if load_saved_results:
        df, saved_config = load_benchmark_results(cache_path)
        run_pennylane = bool(saved_config.get("run_pennylane", run_pennylane)) or has_pennylane_results(df)
        print(f"Loaded {len(df)} rows from {cache_path}")
    else:
        df = multi_trial_benchmark(
            nvals=range(nmin, nmax + 1),
            k_func=k_func,
            n_trials=n_trials,
            delta=delta,
            mode=mode,
            run_pennylane=run_pennylane,
            pennylane_nmax=pennylane_nmax,
            c=c,
        )
        outdir.mkdir(parents=True, exist_ok=True)
        save_benchmark_results(df, cache_path, config)
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")

    print_summary(df, run_pennylane=run_pennylane)
    if make_plots:
        plot_results(df, run_pennylane=run_pennylane, outdir=outdir)


if __name__ == "__main__":
    main()
