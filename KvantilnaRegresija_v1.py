#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
QuantReg — Loto 7/39 — V1

OSNOVNI MODEL
Sedam statsmodels QuantReg regresija, q=0.5.
Ulaz: sedam sortiranih brojeva prethodnog kola.
Izlaz: uslovna medijana svake pozicije narednog kola.

PROŠIRENJE
Zajednička raspodela sedmočlanih vektora prognoznih grešaka.
Svaka greška potiče iz prognoze modela učenog samo na ranijim kolima.

Za kontekst x:
    centar_i(x) = medijana(x) + istorijska_prognozna_greška_i

Nenormalizovana masa kombinacije y:
    sum_i exp(-sum_j abs(y_j - centar_ij(x)) / h_j)

Normalizuje se preko:
    1 <= y1 < y2 < ... < y7 <= 39

Eksponencijalno opadajući kernel je sredstvo izglađivanja empirijske raspodele grešaka. 
Ne nameće se jedna Laplasova, normalna ili uniformna raspodela celoj kombinaciji.

Ograničenje V1:
kontekst menja medijane; oblik zajedničke raspodele grešaka deli se između konteksta.

HRONOLOGIJA
70% trening, 15% validacija, 15% test.
Parametar izglađivanja bira se samo validacijom.
Test ne menja izabrane parametre.
Završni QuantReg koristi sve istorijske prelaze.

Početni deo istorije služi za prvo učenje; za njega se ne izmišljaju
prognozne greške. I ti redovi ulaze u završno učenje QuantReg-a.

Nema slučajnosti, mešanja redova, zaokruživanja predikcija,
popravljanja duplikata, frekvencijskog rangiranja ili kozmetičkih filtera.
"""


import csv
import hashlib
import itertools
import math
import sys
import time
import warnings
from pathlib import Path

try:
    import numpy as np
    import statsmodels
    from numba import njit
    from statsmodels.regression.quantile_regression import QuantReg
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning,
        IterationLimitWarning,
    )
except ImportError as exc:
    raise SystemExit(
        "Potrebni su numpy, statsmodels i numba.\n"
        f"{exc}"
    )


PATHS = (
    "/data/loto7_4706_k84.csv",
    "/data/loto7_4706_k84_loto_2975.csv",
    "/data/loto7_4706_k84_loto_plus_1731.csv",
)

N, K = 39, 7
QUANTILE = 0.5

MAX_ITER = 3000
PARAMETER_TOLERANCE = 1e-6

SMOOTHING_FACTORS = (0.25, 0.50, 1.00, 2.00)
SCORE_BATCH = 8

MAP_SECONDS = 120.0
MAP_CHUNK_NODES = 256


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def fingerprint(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()[:16]


def read_draws(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for line, fields in enumerate(csv.reader(stream), 1):
            if not fields or all(not x.strip() for x in fields):
                continue

            require(
                len(fields) == K,
                f"{path}, red {line}: očekivano sedam polja.",
            )
            try:
                row = [int(x.strip()) for x in fields]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: neispravan ceo broj."
                ) from exc

            require(
                len(set(row)) == K and all(1 <= x <= N for x in row),
                f"{path}, red {line}: nelegalno kolo {row}.",
            )
            rows.append(sorted(row))

    require(len(rows) >= 150, f"{path}: potrebno najmanje 150 kola.")
    return np.asarray(rows, dtype=np.int64)


class MedianModel:
    def __init__(self, mean, scale, coefficients, nobs):
        self.mean = mean
        self.scale = scale
        self.coefficients = coefficients
        self.nobs = nobs

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        design = np.column_stack((
            np.ones(len(x)),
            (x-self.mean)/self.scale,
        ))
        predictions = design @ self.coefficients
        require(np.all(np.isfinite(predictions)),
                "Nefinitne QuantReg predikcije.")
        return predictions


def fit_quantreg(x, y, label):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Standardizacija karakteristika koristi samo dati trening.
    # Nije pretpostavka normalne raspodele.
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    design = np.column_stack((
        np.ones(len(x)),
        (x-mean)/scale,
    ))
    require(
        np.linalg.matrix_rank(design) == design.shape[1],
        f"{label}: linearno zavisne ulazne karakteristike.",
    )

    coefficients = np.empty((design.shape[1], K))
    for j in range(K):
        started = time.perf_counter()

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = QuantReg(y[:, j], design).fit(
                q=QUANTILE,
                max_iter=MAX_ITER,
                p_tol=PARAMETER_TOLERANCE,
            )

        convergence_messages = [
            str(item.message)
            for item in recorded
            if issubclass(
                item.category, (ConvergenceWarning, IterationLimitWarning)
            )
        ]
        require(
            not convergence_messages,
            f"{label}, pozicija {j+1}: QuantReg nije uredno konvergirao: "
            +"; ".join(convergence_messages),
        )
        require(
            np.all(np.isfinite(result.params)),
            f"{label}, pozicija {j+1}: neispravni koeficijenti.",
        )
        coefficients[:, j] = result.params

        for item in recorded:
            if not issubclass(
                item.category, (ConvergenceWarning, IterationLimitWarning)
            ):
                print(
                    f"NAPOMENA [{label}, pozicija {j+1}]: {item.message}",
                    flush=True,
                )

        print(
            f"{label}: pozicija {j+1}/7, "
            f"iteracije={result.iterations}, "
            f"{time.perf_counter()-started:.1f}s",
            flush=True,
        )

    return MedianModel(mean, scale, coefficients, len(x))


def initial_error_archive(x, y, train_end):
    """
    Dva uzastopna bloka prognoza unutar treninga.
    Za svaki blok model vidi isključivo ranije redove.
    """
    warmup = max(40, train_end//3)
    middle = (warmup+train_end)//2
    require(warmup < middle < train_end, "Nedovoljno podataka za arhivu.")

    blocks = []
    for start, stop in ((warmup, middle), (middle, train_end)):
        model = fit_quantreg(
            x[:start], y[:start],
            f"ARHIVA: trening do prelaza {start}",
        )
        predicted = model.predict(x[start:stop])
        blocks.append(y[start:stop]-predicted)

    errors = np.ascontiguousarray(np.vstack(blocks), dtype=np.float64)
    require(np.all(np.isfinite(errors)), "Neispravna arhiva grešaka.")

    print(
        f"Početno učenje: {warmup} prelaza; "
        f"hronološka arhiva: {len(errors)} vektora grešaka.",
        flush=True,
    )
    return errors, warmup


@njit(cache=False)
def nll_batch(predictions, observed, errors, widths, n):
    """
    Tačna normalizacija po legalnim kombinacijama.
    Zbir po empirijskim komponentama; nema slučajnog uzorkovanja.
    """
    count = len(errors)
    k = errors.shape[1]
    output = np.empty(len(predictions), dtype=np.float64)

    for row in range(len(predictions)):
        numerator = 0.0
        denominator = 0.0

        for i in range(count):
            exponent = 0.0
            for t in range(k):
                center = predictions[row, t]+errors[i, t]
                exponent -= abs(observed[row, t]-center)/widths[t]
            numerator += math.exp(exponent)

            following = np.ones(n+2, dtype=np.float64)
            for t in range(k-1, -1, -1):
                current = np.zeros(n+2, dtype=np.float64)
                center = predictions[row, t]+errors[i, t]
                for a in range(n, 0, -1):
                    kernel = math.exp(-abs(a-center)/widths[t])
                    current[a] = (
                        current[a+1] + kernel*following[a+1]
                    )
                following = current
            denominator += following[1]

        if (
            numerator <= 0.0
            or denominator <= 0.0
            or not math.isfinite(numerator)
            or not math.isfinite(denominator)
        ):
            output[row] = np.nan
        else:
            output[row] = math.log(denominator)-math.log(numerator)

    return output


def evaluate_nll(predictions, observed, errors, widths, label):
    values = []
    started = last_report = time.perf_counter()

    for start in range(0, len(observed), SCORE_BATCH):
        stop = min(start+SCORE_BATCH, len(observed))
        block = nll_batch(
            np.ascontiguousarray(predictions[start:stop]),
            np.ascontiguousarray(observed[start:stop], dtype=np.float64),
            errors, widths, N,
        )
        require(
            np.all(np.isfinite(block)) and np.all(block >= -1e-8),
            f"{label}: neispravna masa ili numerički podliv.",
        )
        values.extend(block.tolist())

        now = time.perf_counter()
        if now-last_report >= 10:
            print(
                f"{label}: {stop}/{len(observed)}, "
                f"{now-started:.1f}s",
                flush=True,
            )
            last_report = now

    return np.asarray(values)


def select_smoothing(predictions, observed, errors):
    # Skala u jedinicama brojeva, iz isključivo trening grešaka.
    # Donja granica sprečava degeneraciju širine na nulu.
    scale = np.maximum(np.median(np.abs(errors), axis=0), 1.0)

    best = None
    for factor in SMOOTHING_FACTORS:
        widths = np.ascontiguousarray(scale*factor)
        values = evaluate_nll(
            predictions, observed, errors, widths,
            f"VALIDACIJA faktor={factor:g}",
        )
        score = float(values.mean())
        print(
            f"VALIDACIJA faktor={factor:g}, NLL={score:.8f}",
            flush=True,
        )

        if best is None or score < best["score"]:
            best = {
                "factor": factor,
                "widths": widths.copy(),
                "score": score,
            }

    print(
        f"IZBOR: faktor={best['factor']:g}, "
        f"validation NLL={best['score']:.8f}\n"
        f"Širine kernela: {np.array2string(best['widths'], precision=4)}",
        flush=True,
    )
    if best["factor"] in (SMOOTHING_FACTORS[0], SMOOTHING_FACTORS[-1]):
        print("NAPOMENA: izabran je rub pretraženog raspona.", flush=True)
    return best


def make_kernel_table(centers, widths, n):
    k = centers.shape[1]
    table = np.zeros((k, n+1, len(centers)), dtype=np.float64)
    values = np.arange(1, n+1, dtype=np.float64)[:, None]
    for t in range(k):
        table[t, 1:] = np.exp(
            -np.abs(values-centers[None, :, t])/widths[t]
        )
    require(np.all(np.isfinite(table)), "Neispravna tabela kernela.")
    return table


def suffix_tables(table):
    k, width, count = table.shape
    n = width-1
    sums = np.zeros((k+1, n+2, count), dtype=np.float64)
    maxima = np.zeros_like(sums)
    sums[k] = 1.0
    maxima[k] = 1.0

    for t in range(k-1, -1, -1):
        for a in range(n, 0, -1):
            sums[t, a] = (
                sums[t, a+1] + table[t, a]*sums[t+1, a+1]
            )
            maxima[t, a] = np.maximum(
                maxima[t, a+1],
                table[t, a]*maxima[t+1, a+1],
            )

    require(
        np.all(np.isfinite(sums)) and float(sums[0, 1].sum()) > 0,
        "Neispravna normalizacija legalnog prostora.",
    )
    return sums, maxima


def mass_for_rows(rows, table, weights):
    result = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        product = weights.copy()
        for t, number in enumerate(row):
            product *= table[t, int(number)]
        result[i] = product.sum()
    return result


def audit_distribution(table, sums, weights):
    k, width, count = table.shape
    n = width-1
    z = float(weights @ sums[0, 1])
    require(np.isfinite(z) and z > 0, "Neispravan Z.")

    before = np.zeros((n+1, count), dtype=np.float64)
    before[1:] = 1.0
    marginals = np.zeros(n, dtype=np.float64)
    forward_z = 0.0

    for t in range(k):
        current = before*table[t]
        for a in range(1, n+1):
            marginals[a-1] += float(
                weights @ (current[a]*sums[t+1, a+1])
            )/z

        if t == k-1:
            forward_z = float(weights @ current.sum(axis=0))

        cumulative = np.cumsum(current, axis=0)
        before = np.zeros_like(current)
        before[1:] = cumulative[:-1]

    np.testing.assert_allclose(forward_z/z, 1.0, atol=2e-9, rtol=0)
    np.testing.assert_allclose(marginals.sum(), k, atol=2e-8, rtol=0)
    require(
        np.all(marginals >= -1e-10)
        and np.all(marginals <= 1+1e-9),
        "Neispravne marginale.",
    )
    return z, marginals


def combination_mask(row):
    mask = 0
    for number in row:
        mask |= 1 << (int(number)-1)
    return mask


def mask_combination(mask, n):
    return tuple(i+1 for i in range(n) if (int(mask) >> i) & 1)


@njit(cache=False)
def search_chunk(
    table, maxima, states, levels, lows, masks, bounds,
    top, best, best_mask, max_nodes
):
    k = table.shape[0]
    n = table.shape[1]-1
    count = table.shape[2]
    visited = 0

    while top > 0 and visited < max_nodes:
        top -= 1
        t = levels[top]
        low = lows[top]
        mask = masks[top]
        bound = bounds[top]
        prefix = states[top].copy()
        visited += 1

        tolerance = 1e-12*max(abs(bound), abs(best), 1e-300)
        if bound+tolerance < best:
            continue

        upper_number = n-(k-t)+1

        if t == k-1:
            for a in range(low, upper_number+1):
                value = 0.0
                for i in range(count):
                    value += prefix[i]*table[t, a, i]
                candidate_mask = mask | (np.int64(1) << (a-1))
                if value > best or (
                    value == best and candidate_mask < best_mask
                ):
                    best = value
                    best_mask = candidate_mask
            continue

        child_bounds = np.zeros(n+1, dtype=np.float64)
        for a in range(low, upper_number+1):
            value = 0.0
            for i in range(count):
                value += (
                    prefix[i]*table[t, a, i]*maxima[t+1, a+1, i]
                )
            child_bounds[a] = value

        order = np.argsort(child_bounds)
        for index in range(len(order)):
            a = order[index]
            if a < low or a > upper_number:
                continue

            value = child_bounds[a]
            tolerance = 1e-12*max(abs(value), abs(best), 1e-300)
            if value+tolerance < best:
                continue

            if top >= states.shape[0]:
                raise RuntimeError("Nedovoljan kapacitet MAP steka.")

            for i in range(count):
                states[top, i] = prefix[i]*table[t, a, i]
            levels[top] = t+1
            lows[top] = a+1
            masks[top] = mask | (np.int64(1) << (a-1))
            bounds[top] = value
            top += 1

    return top, best, best_mask, visited


def exact_map(table, maxima, weights, seeds, seconds, verbose=True):
    k, width, count = table.shape
    n = width-1

    # Legalne istorijske kombinacije samo daju početnu donju granicu.
    # Konačna pretraga nije ograničena na njih.
    seed_values = mass_for_rows(seeds, table, weights)
    index = int(np.argmax(seed_values))
    best = float(seed_values[index])
    require(best > 0, "Nulta početna masa MAP pretrage.")
    best_mask = np.int64(combination_mask(seeds[index]))

    capacity = 2+k*(n+1)
    states = np.zeros((capacity, count), dtype=np.float64)
    levels = np.zeros(capacity, dtype=np.int64)
    lows = np.zeros(capacity, dtype=np.int64)
    masks = np.zeros(capacity, dtype=np.int64)
    bounds = np.zeros(capacity, dtype=np.float64)

    states[0] = weights
    lows[0] = 1
    bounds[0] = float(weights @ maxima[0, 1])
    top = 1
    visited = 0
    started = last_report = time.perf_counter()

    while top:
        top, best, best_mask, done = search_chunk(
            table, maxima, states, levels, lows, masks, bounds,
            top, best, best_mask, MAP_CHUNK_NODES,
        )
        visited += done
        now = time.perf_counter()

        if verbose and now-last_report >= 5:
            upper = max(best, float(bounds[:top].max())) if top else best
            print(
                f"  MAP: čvorovi={visited:,}; "
                f"gornja/donja granica={upper/best:.7g}; "
                f"{now-started:.1f}s",
                flush=True,
            )
            last_report = now

        if top and now-started >= seconds:
            upper = max(best, float(bounds[:top].max()))
            raise RuntimeError(
                f"MAP nije potvrđen za {seconds:g}s. "
                f"Gornja/donja granica={upper/best:.8g}. "
                "Nepotvrđeni kandidat neće biti izdat kao NEXT."
            )

    return mask_combination(best_mask, n), best, visited


def legal_rows(n, k):
    return np.asarray(
        list(itertools.combinations(range(1, n+1), k)),
        dtype=np.int64,
    )


def self_checks():
    print("Provere raspodele, normalizacije, NLL-a i MAP-a...", flush=True)
    n, k = 6, 2
    centers = np.asarray([
        [1.2, 4.8],
        [2.8, 3.1],
        [0.5, 6.2],
    ])
    widths = np.asarray([0.8, 1.1])
    weights = np.full(len(centers), 1.0/len(centers))
    candidates = legal_rows(n, k)

    table = make_kernel_table(centers, widths, n)
    sums, maxima = suffix_tables(table)
    z, marginals = audit_distribution(table, sums, weights)
    masses = mass_for_rows(candidates, table, weights)
    probabilities = masses/z

    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=2e-12)

    indicators = np.zeros((len(candidates), n))
    indicators[np.arange(len(candidates))[:, None], candidates-1] = 1
    np.testing.assert_allclose(
        probabilities @ indicators, marginals, atol=2e-12
    )

    numbers, maximum, _ = exact_map(
        table, maxima, weights, candidates[:1], 30, verbose=False
    )
    np.testing.assert_allclose(maximum, masses.max(), atol=2e-12)
    require(len(numbers) == k, "Nelegalni MAP u proveri.")

    # Poređenje posebnog NLL računa sa potpunim nabrajanjem.
    predicted = np.zeros((len(candidates), k), dtype=np.float64)
    computed = nll_batch(
        predicted,
        candidates.astype(np.float64),
        np.ascontiguousarray(centers),
        widths, n,
    )
    np.testing.assert_allclose(
        computed, -np.log(probabilities), atol=2e-11, rtol=2e-10
    )

    for boundary_k in (1, n):
        rows = legal_rows(n, boundary_k)
        c = rows[:min(3, len(rows))].astype(np.float64)+0.15
        h = np.full(boundary_k, 0.9)
        w = np.full(len(c), 1.0/len(c))
        tab = make_kernel_table(c, h, n)
        ss, mm = suffix_tables(tab)
        zz, marginal = audit_distribution(tab, ss, w)
        values = mass_for_rows(rows, tab, w)
        np.testing.assert_allclose(values.sum()/zz, 1.0, atol=2e-12)
        _, maximum, _ = exact_map(
            tab, mm, w, rows[:1], 30, verbose=False
        )
        np.testing.assert_allclose(maximum, values.max(), atol=2e-12)
        np.testing.assert_allclose(marginal.sum(), boundary_k, atol=2e-12)

    print("Ugrađene numeričke provere prošle.", flush=True)


def longest_arithmetic(numbers):
    dp = [dict() for _ in numbers]
    best = ()
    for i in range(len(numbers)):
        for j in range(i):
            step = numbers[i]-numbers[j]
            candidate = dp[j].get(step, (numbers[j],))+(numbers[i],)
            if len(candidate) > len(dp[i].get(step, ())):
                dp[i][step] = candidate
            if len(candidate) > len(best):
                best = candidate
    return best


def diagnostic_messages(numbers, draws, medians):
    messages = []
    sequence = longest_arithmetic(numbers)
    if len(sequence) >= 5:
        messages.append(f"Aritmetički podniz: {sequence}.")
    if all(number <= 19 for number in numbers):
        messages.append("Svih sedam brojeva je u 1–19.")
    if all(number >= 20 for number in numbers):
        messages.append("Svih sedam brojeva je u 20–39.")

    span = numbers[-1]-numbers[0]
    if span <= float(np.quantile(draws[:, -1]-draws[:, 0], 0.01)):
        messages.append("Raspon je u donjih 1% istorijskih raspona.")

    repeats = int(np.all(draws == np.asarray(numbers), axis=1).sum())
    if repeats:
        messages.append(f"Kombinacija postoji {repeats} puta u istoriji.")

    if np.any(np.diff(medians) <= 0):
        messages.append(
            "Uslovne medijane pozicija nisu strogo rastuće. "
            "Nisu sortirane ili popravljene."
        )
    if np.any((medians < 1) | (medians > N)):
        messages.append(
            "Neka uslovna medijana je izvan 1–39. "
            "Nije odsečena na granicu."
        )
    return messages


def pinball_loss(observed, predicted):
    residual = observed-predicted
    return float(np.mean(np.maximum(
        QUANTILE*residual, (QUANTILE-1)*residual
    )))


def process_file(path):
    print("\n"+"="*72, flush=True)
    print(Path(path).name, flush=True)
    draws = read_draws(path)
    x = draws[:-1].astype(np.float64)
    y = draws[1:].astype(np.float64)

    total = len(y)
    train_end = int(0.70*total)
    validation_end = int(0.85*total)

    print(
        f"Kola={len(draws)}; prelazi={total}; train={train_end}; "
        f"validation={validation_end-train_end}; "
        f"test={total-validation_end}.",
        flush=True,
    )

    errors, warmup = initial_error_archive(x, y, train_end)

    train_model = fit_quantreg(
        x[:train_end], y[:train_end], "TRAIN"
    )
    validation_predictions = train_model.predict(
        x[train_end:validation_end]
    )
    best = select_smoothing(
        validation_predictions, y[train_end:validation_end], errors
    )
    widths = best["widths"]

    # Tek nakon izbora parametra dodajemo prognozne greške validacije.
    validation_errors = (
        y[train_end:validation_end]-validation_predictions
    )
    test_archive = np.ascontiguousarray(
        np.vstack((errors, validation_errors))
    )
    del train_model

    test_model = fit_quantreg(
        x[:validation_end], y[:validation_end], "TRAIN+VALIDATION"
    )
    test_predictions = test_model.predict(x[validation_end:])
    test_nll = evaluate_nll(
        test_predictions, y[validation_end:],
        test_archive, widths, "ODVOJENI TEST",
    )
    print(
        f"ODVOJENI TEST: NLL={test_nll.mean():.8f}; "
        f"pinball(q=0.5)="
        f"{pinball_loss(y[validation_end:], test_predictions):.8f}",
        flush=True,
    )

    # Test služi za izveštaj, ne za novi izbor parametara.
    # Nakon završenog testa cela poznata istorija sme u završni model.
    test_errors = y[validation_end:]-test_predictions
    final_errors = np.ascontiguousarray(
        np.vstack((test_archive, test_errors))
    )
    del test_model

    require(
        len(final_errors) == total-warmup,
        "Neispravna dužina hronološke arhive.",
    )

    final_model = fit_quantreg(x, y, "CELA ISTORIJA")
    require(final_model.nobs == total,
            "Završni QuantReg ne koristi sve prelaze.")

    medians = final_model.predict(draws[-1:].astype(np.float64))[0]
    centers = final_errors+medians[None, :]
    weights = np.full(len(centers), 1.0/len(centers))

    print(
        "Uslovne medijane, bez zaokruživanja: "
        +np.array2string(medians, precision=5),
        flush=True,
    )
    print(
        f"Završno učenje: svih {total} prelaza. "
        f"Arhiva: {len(final_errors)} ranije prognoziranih grešaka.",
        flush=True,
    )

    table = make_kernel_table(centers, widths, N)
    sums, maxima = suffix_tables(table)
    z, marginals = audit_distribution(table, sums, weights)

    seeds = np.unique(draws, axis=0)
    print(f"MAP pretraga; limit={MAP_SECONDS:g}s.", flush=True)
    numbers, maximum, visited = exact_map(
        table, maxima, weights, seeds, MAP_SECONDS
    )

    require(
        len(numbers) == K
        and len(set(numbers)) == K
        and all(1 <= number <= N for number in numbers),
        "Nelegalni NEXT.",
    )
    selected = np.asarray([numbers], dtype=np.int64)
    direct_mass = mass_for_rows(selected, table, weights)[0]
    np.testing.assert_allclose(
        maximum, direct_mass, atol=1e-14, rtol=2e-9
    )

    probability = maximum/z
    require(
        np.isfinite(probability) and 0 < probability <= 1+1e-9,
        "Neispravna normalizovana masa NEXT-a.",
    )

    # Nezavisan put računanja iste legalno normalizovane mase.
    independent_nll = nll_batch(
        np.ascontiguousarray(medians[None, :]),
        selected.astype(np.float64),
        final_errors, widths, N,
    )[0]
    np.testing.assert_allclose(
        independent_nll, -math.log(probability),
        atol=2e-8, rtol=2e-8,
    )

    messages = diagnostic_messages(numbers, draws, medians)
    for message in messages:
        print("DIJAGNOSTIKA: "+message, flush=True)

    print(
        f"Zbir marginala={marginals.sum():.10f}; "
        f"MAP čvorovi={visited:,}.",
        flush=True,
    )
    print(
        f"\nNEXT [{Path(path).name}]: "
        +" ".join(f"{number:02d}" for number in numbers),
        flush=True,
    )
    print(
        f"Interna modelska masa={probability:.12g}. "
        "Nije potvrđena verovatnoća budućeg izvlačenja.",
        flush=True,
    )
    if messages:
        print(
            "STATUS: maksimum je pronađen; "
            "dijagnostika označava rezultat za proveru.",
            flush=True,
        )
    else:
        print(
            "STATUS: maksimum je pronađen; "
            "to samo po sebi ne potvrđuje prediktivnu vrednost.",
            flush=True,
        )

    return {
        "file": Path(path).name,
        "numbers": numbers,
        "marginals": marginals,
        "input_hash": fingerprint(draws),
        "model_hash": fingerprint(final_model.coefficients),
        "factor": best["factor"],
    }


def main():
    print(
        "QuantReg — Loto 7/39 — V1\n"
        f"Statsmodels verzija: {statsmodels.__version__}\n"
        "q=0.5; zajednička raspodela hronoloških prognoznih grešaka.\n"
        "Bez slučajnosti, zaokruživanja i kozmetičkih filtera.",
        flush=True,
    )

    for path in PATHS:
        require(Path(path).is_file(), f"Nedostaje CSV: {path}")

    self_checks()
    results = []
    failures = []

    for path in PATHS:
        try:
            results.append(process_file(path))
        except Exception as exc:
            name = Path(path).name
            failures.append(name)
            print(
                f"\nNEUSPEH [{name}]: {type(exc).__name__}: {exc}\n"
                "Zamenska kombinacija nije generisana.",
                file=sys.stderr, flush=True,
            )

    for i, left in enumerate(results):
        for right in results[i+1:]:
            if left["numbers"] != right["numbers"]:
                continue
            difference = float(np.max(np.abs(
                left["marginals"]-right["marginals"]
            )))
            print(
                "\nISTI NEXT — POTREBNA PROVERA:\n"
                f"{left['file']} / {right['file']}\n"
                f"Ulazi: {left['input_hash']} / {right['input_hash']}\n"
                f"QuantReg: {left['model_hash']} / {right['model_hash']}\n"
                f"Faktor: {left['factor']} / {right['factor']}\n"
                f"Maksimalna razlika marginala={difference:.9g}\n"
                "Kombinacije nisu menjane da bi bile različite.",
                flush=True,
            )

    if failures:
        print(
            "\nNije potvrđen rezultat za: "+", ".join(failures),
            flush=True,
        )
        return 1

    print("\nSva tri CSV-a obrađena.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "\nPrekinuto. Nedovršena MAP pretraga ne potvrđuje maksimum.",
            file=sys.stderr,
        )
        sys.exit(130)



"""
QuantReg — Loto 7/39 — V1
Statsmodels verzija: 0.14.6
q=0.5; zajednička raspodela hronoloških prognoznih grešaka.
Bez slučajnosti, zaokruživanja i kozmetičkih filtera.
Provere raspodele, normalizacije, NLL-a i MAP-a...
Ugrađene numeričke provere prošle.


========================================================================
loto7_4706_k84.csv
Kola=4706; prelazi=4705; train=3293; validation=706; test=706.
ARHIVA: trening do prelaza 1097: pozicija 1/7, iteracije=115, 0.0s
ARHIVA: trening do prelaza 1097: pozicija 2/7, iteracije=101, 0.0s
ARHIVA: trening do prelaza 1097: pozicija 3/7, iteracije=110, 0.0s
ARHIVA: trening do prelaza 1097: pozicija 4/7, iteracije=153, 0.0s
ARHIVA: trening do prelaza 1097: pozicija 5/7, iteracije=60, 0.0s
ARHIVA: trening do prelaza 1097: pozicija 6/7, iteracije=248, 0.0s
ARHIVA: trening do prelaza 1097: pozicija 7/7, iteracije=471, 0.0s
ARHIVA: trening do prelaza 2195: pozicija 1/7, iteracije=33, 0.0s
ARHIVA: trening do prelaza 2195: pozicija 2/7, iteracije=522, 0.1s
ARHIVA: trening do prelaza 2195: pozicija 3/7, iteracije=236, 0.0s
ARHIVA: trening do prelaza 2195: pozicija 4/7, iteracije=288, 0.1s
ARHIVA: trening do prelaza 2195: pozicija 5/7, iteracije=1376, 0.2s
ARHIVA: trening do prelaza 2195: pozicija 6/7, iteracije=146, 0.0s
ARHIVA: trening do prelaza 2195: pozicija 7/7, iteracije=352, 0.0s
Početno učenje: 1097 prelaza; hronološka arhiva: 2196 vektora grešaka.
TRAIN: pozicija 1/7, iteracije=27, 0.0s
TRAIN: pozicija 2/7, iteracije=91, 0.0s
TRAIN: pozicija 3/7, iteracije=203, 0.0s
TRAIN: pozicija 4/7, iteracije=170, 0.0s
TRAIN: pozicija 5/7, iteracije=79, 0.0s
TRAIN: pozicija 6/7, iteracije=53, 0.0s
TRAIN: pozicija 7/7, iteracije=101, 0.0s
VALIDACIJA faktor=0.25, NLL=17.61721703
VALIDACIJA faktor=0.5, NLL=16.77899656
VALIDACIJA faktor=1, NLL=16.71018489
VALIDACIJA faktor=2, NLL=16.66959302
IZBOR: faktor=2, validation NLL=16.66959302
Širine kernela: [4.4784 7.294  8.5444 8.6813 8.2089 7.1097 4.4458]
NAPOMENA: izabran je rub pretraženog raspona.
TRAIN+VALIDATION: pozicija 1/7, iteracije=26, 0.0s
TRAIN+VALIDATION: pozicija 2/7, iteracije=82, 0.0s
TRAIN+VALIDATION: pozicija 3/7, iteracije=186, 0.0s
TRAIN+VALIDATION: pozicija 4/7, iteracije=176, 0.0s
TRAIN+VALIDATION: pozicija 5/7, iteracije=137, 0.0s
TRAIN+VALIDATION: pozicija 6/7, iteracije=35, 0.0s
TRAIN+VALIDATION: pozicija 7/7, iteracije=65, 0.0s
ODVOJENI TEST: NLL=16.65771982; pinball(q=0.5)=2.05883146
CELA ISTORIJA: pozicija 1/7, iteracije=22, 0.0s
CELA ISTORIJA: pozicija 2/7, iteracije=135, 0.0s
CELA ISTORIJA: pozicija 3/7, iteracije=291, 0.1s
CELA ISTORIJA: pozicija 4/7, iteracije=214, 0.1s
CELA ISTORIJA: pozicija 5/7, iteracije=399, 0.1s
CELA ISTORIJA: pozicija 6/7, iteracije=38, 0.0s
CELA ISTORIJA: pozicija 7/7, iteracije=56, 0.0s
Uslovne medijane, bez zaokruživanja: [ 4.       9.00002 14.86484 20.39539 25.44788 31.      36.     ]
Završno učenje: svih 4705 prelaza. Arhiva: 3608 ranije prognoziranih grešaka.
MAP pretraga; limit=120s.
  MAP: čvorovi=141,056; gornja/donja granica=11.65463; 5.0s
  MAP: čvorovi=285,440; gornja/donja granica=10.66675; 10.0s
  MAP: čvorovi=425,472; gornja/donja granica=8.795095; 15.0s
Zbir marginala=7.0000000000; MAP čvorovi=508,814.

NEXT [loto7_4706_k84.csv]: 04 x 15 y 27 z 37
Interna modelska masa=1.47992254519e-07. Nije potvrđena verovatnoća budućeg izvlačenja.
STATUS: maksimum je pronađen; to samo po sebi ne potvrđuje prediktivnu vrednost.



========================================================================
loto7_4706_k84_loto_2975.csv
Kola=2975; prelazi=2974; train=2081; validation=446; test=447.
ARHIVA: trening do prelaza 693: pozicija 1/7, iteracije=62, 0.0s
ARHIVA: trening do prelaza 693: pozicija 2/7, iteracije=213, 0.0s
ARHIVA: trening do prelaza 693: pozicija 3/7, iteracije=207, 0.0s
ARHIVA: trening do prelaza 693: pozicija 4/7, iteracije=148, 0.0s
ARHIVA: trening do prelaza 693: pozicija 5/7, iteracije=61, 0.0s
ARHIVA: trening do prelaza 693: pozicija 6/7, iteracije=216, 0.0s
ARHIVA: trening do prelaza 693: pozicija 7/7, iteracije=207, 0.0s
ARHIVA: trening do prelaza 1387: pozicija 1/7, iteracije=81, 0.0s
ARHIVA: trening do prelaza 1387: pozicija 2/7, iteracije=237, 0.0s
ARHIVA: trening do prelaza 1387: pozicija 3/7, iteracije=118, 0.0s
ARHIVA: trening do prelaza 1387: pozicija 4/7, iteracije=239, 0.1s
ARHIVA: trening do prelaza 1387: pozicija 5/7, iteracije=207, 0.0s
ARHIVA: trening do prelaza 1387: pozicija 6/7, iteracije=75, 0.0s
ARHIVA: trening do prelaza 1387: pozicija 7/7, iteracije=230, 0.0s
Početno učenje: 693 prelaza; hronološka arhiva: 1388 vektora grešaka.
TRAIN: pozicija 1/7, iteracije=49, 0.0s
TRAIN: pozicija 2/7, iteracije=165, 0.0s
TRAIN: pozicija 3/7, iteracije=24, 0.0s
TRAIN: pozicija 4/7, iteracije=743, 0.1s
TRAIN: pozicija 5/7, iteracije=61, 0.0s
TRAIN: pozicija 6/7, iteracije=109, 0.0s
TRAIN: pozicija 7/7, iteracije=308, 0.1s
VALIDACIJA faktor=0.25, NLL=17.78487854
VALIDACIJA faktor=0.5, NLL=16.81088977
VALIDACIJA faktor=1, NLL=16.70601511
VALIDACIJA faktor=2, NLL=16.65918399
IZBOR: faktor=2, validation NLL=16.65918399
Širine kernela: [4.8323 6.9818 8.2991 8.5225 8.2569 6.9743 4.5262]
NAPOMENA: izabran je rub pretraženog raspona.
TRAIN+VALIDATION: pozicija 1/7, iteracije=47, 0.0s
TRAIN+VALIDATION: pozicija 2/7, iteracije=118, 0.0s
TRAIN+VALIDATION: pozicija 3/7, iteracije=29, 0.0s
TRAIN+VALIDATION: pozicija 4/7, iteracije=149, 0.0s
TRAIN+VALIDATION: pozicija 5/7, iteracije=109, 0.0s
TRAIN+VALIDATION: pozicija 6/7, iteracije=63, 0.0s
TRAIN+VALIDATION: pozicija 7/7, iteracije=145, 0.0s
ODVOJENI TEST: NLL=16.67984252; pinball(q=0.5)=2.10172325
CELA ISTORIJA: pozicija 1/7, iteracije=36, 0.0s
CELA ISTORIJA: pozicija 2/7, iteracije=66, 0.0s
CELA ISTORIJA: pozicija 3/7, iteracije=79, 0.0s
CELA ISTORIJA: pozicija 4/7, iteracije=44, 0.0s
CELA ISTORIJA: pozicija 5/7, iteracije=163, 0.0s
CELA ISTORIJA: pozicija 6/7, iteracije=163, 0.0s
CELA ISTORIJA: pozicija 7/7, iteracije=561, 0.1s
Uslovne medijane, bez zaokruživanja: [ 4.       9.      14.99999 20.30934 25.00886 30.99999 35.99232]
Završno učenje: svih 2974 prelaza. Arhiva: 2281 ranije prognoziranih grešaka.
MAP pretraga; limit=120s.
  MAP: čvorovi=220,160; gornja/donja granica=10.83168; 5.0s
  MAP: čvorovi=446,208; gornja/donja granica=6.970178; 10.0s
Zbir marginala=7.0000000000; MAP čvorovi=482,177.

NEXT [loto7_4706_k84_loto_2975.csv]: 04 x 15 y 25 z 36
Interna modelska masa=1.46047654568e-07. Nije potvrđena verovatnoća budućeg izvlačenja.
STATUS: maksimum je pronađen; to samo po sebi ne potvrđuje prediktivnu vrednost.



========================================================================
loto7_4706_k84_loto_plus_1731.csv
Kola=1731; prelazi=1730; train=1211; validation=259; test=260.
ARHIVA: trening do prelaza 403: pozicija 1/7, iteracije=102, 0.0s
ARHIVA: trening do prelaza 403: pozicija 2/7, iteracije=194, 0.0s
ARHIVA: trening do prelaza 403: pozicija 3/7, iteracije=245, 0.0s
ARHIVA: trening do prelaza 403: pozicija 4/7, iteracije=152, 0.0s
ARHIVA: trening do prelaza 403: pozicija 5/7, iteracije=2509, 0.1s
ARHIVA: trening do prelaza 403: pozicija 6/7, iteracije=1856, 0.1s
ARHIVA: trening do prelaza 403: pozicija 7/7, iteracije=128, 0.0s
ARHIVA: trening do prelaza 807: pozicija 1/7, iteracije=121, 0.0s
ARHIVA: trening do prelaza 807: pozicija 2/7, iteracije=322, 0.0s
ARHIVA: trening do prelaza 807: pozicija 3/7, iteracije=330, 0.0s
ARHIVA: trening do prelaza 807: pozicija 4/7, iteracije=73, 0.0s
ARHIVA: trening do prelaza 807: pozicija 5/7, iteracije=164, 0.0s
ARHIVA: trening do prelaza 807: pozicija 6/7, iteracije=69, 0.0s
ARHIVA: trening do prelaza 807: pozicija 7/7, iteracije=92, 0.0s
Početno učenje: 403 prelaza; hronološka arhiva: 808 vektora grešaka.
TRAIN: pozicija 1/7, iteracije=128, 0.0s
TRAIN: pozicija 2/7, iteracije=1675, 0.1s
TRAIN: pozicija 3/7, iteracije=436, 0.0s
TRAIN: pozicija 4/7, iteracije=184, 0.0s
TRAIN: pozicija 5/7, iteracije=192, 0.0s
TRAIN: pozicija 6/7, iteracije=59, 0.0s
TRAIN: pozicija 7/7, iteracije=43, 0.0s
VALIDACIJA faktor=0.25, NLL=18.53661118
VALIDACIJA faktor=0.5, NLL=16.98966344
VALIDACIJA faktor=1, NLL=16.71262696
VALIDACIJA faktor=2, NLL=16.64342820
IZBOR: faktor=2, validation NLL=16.64342820
Širine kernela: [4.2524 7.2442 8.3418 8.9421 8.3807 7.1726 4.234 ]
NAPOMENA: izabran je rub pretraženog raspona.
TRAIN+VALIDATION: pozicija 1/7, iteracije=145, 0.0s
TRAIN+VALIDATION: pozicija 2/7, iteracije=1021, 0.1s
TRAIN+VALIDATION: pozicija 3/7, iteracije=89, 0.0s
TRAIN+VALIDATION: pozicija 4/7, iteracije=578, 0.1s
TRAIN+VALIDATION: pozicija 5/7, iteracije=675, 0.1s
TRAIN+VALIDATION: pozicija 6/7, iteracije=93, 0.0s
TRAIN+VALIDATION: pozicija 7/7, iteracije=34, 0.0s
ODVOJENI TEST: NLL=16.63258318; pinball(q=0.5)=2.01336218
CELA ISTORIJA: pozicija 1/7, iteracije=124, 0.0s
CELA ISTORIJA: pozicija 2/7, iteracije=147, 0.0s
CELA ISTORIJA: pozicija 3/7, iteracije=537, 0.1s
CELA ISTORIJA: pozicija 4/7, iteracije=82, 0.0s
CELA ISTORIJA: pozicija 5/7, iteracije=957, 0.1s
CELA ISTORIJA: pozicija 6/7, iteracije=34, 0.0s
CELA ISTORIJA: pozicija 7/7, iteracije=52, 0.0s
Uslovne medijane, bez zaokruživanja: [ 4.       9.00001 14.37349 20.47235 26.15759 31.      36.     ]
Završno učenje: svih 1730 prelaza. Arhiva: 1327 ranije prognoziranih grešaka.
MAP pretraga; limit=120s.
  MAP: čvorovi=399,104; gornja/donja granica=8.50834; 5.0s
Zbir marginala=7.0000000000; MAP čvorovi=473,562.

NEXT [loto7_4706_k84_loto_plus_1731.csv]: 04 x 15 y 27 z 37
Interna modelska masa=1.54498094185e-07. Nije potvrđena verovatnoća budućeg izvlačenja.
STATUS: maksimum je pronađen; to samo po sebi ne potvrđuje prediktivnu vrednost.


Sva tri CSV-a obrađena.
"""

 

"""
Kvantilna regresija — QuantReg. Statsmodels. 

QuantReg procenjuje uslovne kvantile jedne izlazne promenljive; sam ne daje zajedničku raspodelu kombinacije 7/39.
Sedam odvojenih regresija ne bi naučilo njihove međusobne zavisnosti. 
Potrebno je dodatno konstruisati diskretnu zajedničku raspodelu i pronaći njen maksimum. 
To bi bilo proširenje metode, ne sam izvorni QuantReg.

QuantReg procenjuje sedam uslovnih medijana, a zajednička raspodela vektora grešaka daje raspodelu cele kombinacije. 
Greške ću računati iz hronoloških prognoza, koristeći samo ranija kola. 
NEXT se bira među legalnim kombinacijama, bez zaokruživanja i popravljanja duplikata.

V1: QuantReg(q=0.5) + zajednička raspodela hronoloških grešaka. Medijane se ne zaokružuju. 
Raspodela zadržava kompletne sedmočlane vektore grešaka; širina njihovog izglađivanja bira se validacijom.
Proširenje izvornog QuantReg, ne njegova ugrađena funkcija. 
MAP pretraga ima limit od 120 sekundi po CSV-u.
"""
