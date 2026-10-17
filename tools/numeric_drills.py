#!/usr/bin/env python3
"""Programmatic seismology computation drills with exact answers (stdlib only, no API).

The model's numerical failure is not the reward or the decoder: handed the right equations, or
most of the worked solution, it still gets the final arithmetic wrong (audit §15.5). These
drills teach the two things a calculator cannot do for it — recall the formula and bind the
values from the question — and leave the arithmetic to a calculator: every computation in a
reference solution is written as

    M0 = μ·A·D = <<3e10 * 4e8 * 1.5>> = 1.8e+19 N·m

and the decoder (src.rl.rollout, calculator=True) evaluates the `<<…>>` expression with
src.rl.verifiers.safe_eval_number and inserts " = value" itself.

Each template draws realistic parameters, states the formula in some phrasings and only names
it in others (so recall is exercised), and computes the answer in Python — the label is exact.

    python3 tools/numeric_drills.py --n 6000 --seed 1 --out data/sources/drills_train.jsonl
    python3 tools/numeric_drills.py --n 300  --seed 2 --out data/eval/drills_test.jsonl --rl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from typing import Callable, Dict, List, Tuple


def fmt(v: float) -> str:
    """The one number format used everywhere: in solutions and by the decoder's calculator."""
    return f"{v:.4g}"


def calc(expr: str, value: float) -> str:
    return f"<<{expr}>> = {fmt(value)}"


Drill = Tuple[str, str, str, float, str]   # question, think, final answer, value, unit


def _mw_from_m0(r: random.Random) -> Drill:
    m, e = round(r.uniform(1.0, 9.9), 2), r.randint(14, 22)
    m0 = m * 10 ** e
    mw = (2 / 3) * (math.log10(m0) - 9.1)
    q = r.choice([
        f"An earthquake has a seismic moment of {m} × 10^{e} N·m. Calculate its moment magnitude Mw using Mw = (2/3)(log10 M0 − 9.1) with M0 in N·m.",
        f"Calculate the moment magnitude of an earthquake whose seismic moment is {m}e{e} N·m (Hanks and Kanamori relation).",
        f"What is the moment magnitude Mw for a seismic moment M0 = {m} × 10^{e} N·m?",
    ])
    t = (f"Moment magnitude: Mw = (2/3)(log10 M0 − 9.1) with M0 in N·m. M0 = {m}e{e} N·m. "
         f"Mw = {calc(f'(2/3) * (log10({m}e{e}) - 9.1)', mw)}.")
    return q, t, f"The moment magnitude is Mw ≈ {fmt(mw)}.", mw, ""


def _m0_from_mw(r: random.Random) -> Drill:
    mw = round(r.uniform(4.0, 9.5), 1)
    m0 = 10 ** (1.5 * mw + 9.1)
    q = r.choice([
        f"Calculate the seismic moment M0 in N·m for an earthquake of moment magnitude Mw {mw}, using log10 M0 = 1.5 Mw + 9.1.",
        f"Estimate the seismic moment (in N·m) of a magnitude Mw {mw} earthquake.",
        f"An earthquake has Mw = {mw}. What is its seismic moment in N·m?",
    ])
    t = f"log10 M0 = 1.5 Mw + 9.1 (N·m). M0 = 10^(1.5 × {mw} + 9.1) = {calc(f'10**(1.5 * {mw} + 9.1)', m0)} N·m."
    return q, t, f"The seismic moment is approximately {fmt(m0)} N·m.", m0, "N·m"


def _m0_from_fault(r: random.Random) -> Drill:
    L, W, D, mu = r.choice([20, 30, 40, 50, 80, 100, 150, 200, 300]), r.choice([10, 12, 15, 20, 25, 30, 50, 80]), round(r.uniform(0.5, 12), 1), r.choice([30, 32, 35, 40])
    m0 = mu * 1e9 * L * 1e3 * W * 1e3 * D
    q = r.choice([
        f"A fault ruptures over a length of {L} km and a width of {W} km with an average slip of {D} m. Assuming a shear modulus of {mu} GPa, calculate the seismic moment using M0 = μAD.",
        f"Calculate the seismic moment (N·m) for a rupture {L} km long and {W} km wide with {D} m of average slip, taking μ = {mu} GPa.",
    ])
    t = (f"M0 = μ A D. μ = {mu} GPa = {mu}e9 Pa, A = {L} km × {W} km = {L}e3 m × {W}e3 m, D = {D} m. "
         f"M0 = {calc(f'{mu}e9 * {L}e3 * {W}e3 * {D}', m0)} N·m.")
    return q, t, f"The seismic moment is approximately {fmt(m0)} N·m.", m0, "N·m"


def _slip_from_mw(r: random.Random) -> Drill:
    mw, L, W, mu = round(r.uniform(5.5, 9.0), 1), r.choice([20, 40, 60, 100, 200, 400]), r.choice([10, 15, 20, 30, 50, 100]), r.choice([30, 32, 35])
    m0 = 10 ** (1.5 * mw + 9.1)
    D = m0 / (mu * 1e9 * L * 1e3 * W * 1e3)
    q = r.choice([
        f"An Mw {mw} earthquake ruptured a fault {L} km long and {W} km wide. Assuming a shear modulus of {mu} GPa, estimate the average slip.",
        f"Estimate the average coseismic slip for a magnitude Mw {mw} event with a rupture area of {L} km × {W} km (μ = {mu} GPa). Use log10 M0 = 1.5 Mw + 9.1.",
    ])
    t = (f"First the seismic moment: M0 = 10^(1.5 × {mw} + 9.1) = {calc(f'10**(1.5 * {mw} + 9.1)', m0)} N·m. "
         f"Then D = M0 / (μ L W) = {calc(f'{fmt(m0)} / ({mu}e9 * {L}e3 * {W}e3)', D)} m.")
    return q, t, f"The average slip is approximately {fmt(D)} m.", D, "m"


def _energy_ratio(r: random.Random) -> Drill:
    m1 = round(r.uniform(5.0, 9.0), 1); m2 = round(m1 - r.choice([0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]), 1)
    ratio = 10 ** (1.5 * (m1 - m2))
    q = r.choice([
        f"How many times more energy does a magnitude {m1} earthquake release than a magnitude {m2} earthquake? Use log10 E = 1.5M + 4.8.",
        f"Calculate the ratio of radiated energy between an M {m1} and an M {m2} earthquake.",
        f"Compared with a magnitude {m2} event, how much more seismic energy does a magnitude {m1} event release?",
    ])
    t = f"log10 E = 1.5M + 4.8, so E1/E2 = 10^(1.5 (M1 − M2)) = {calc(f'10**(1.5 * ({m1} - {m2}))', ratio)}."
    return q, t, f"It releases about {fmt(ratio)} times more energy.", ratio, ""


def _amplitude_ratio(r: random.Random) -> Drill:
    m1 = round(r.uniform(3.0, 8.0), 1); dm = r.choice([0.5, 1.0, 1.5, 2.0]); m2 = round(m1 - dm, 1)
    ratio = 10 ** (m1 - m2)
    q = f"On a logarithmic magnitude scale where M = log10 A + const, how many times larger is the ground-motion amplitude of a magnitude {m1} event than a magnitude {m2} event at the same distance?"
    t = f"Each magnitude unit is a factor of 10 in amplitude: A1/A2 = 10^(M1 − M2) = {calc(f'10**({m1} - {m2})', ratio)}."
    return q, t, f"The amplitude is about {fmt(ratio)} times larger.", ratio, ""


def _radiated_energy(r: random.Random) -> Drill:
    m = round(r.uniform(4.0, 9.0), 1)
    E = 10 ** (1.5 * m + 4.8)
    q = r.choice([f"Estimate the radiated seismic energy (in joules) of a magnitude {m} earthquake using log10 E = 1.5M + 4.8.",
                  f"How much seismic energy in joules does an M {m} earthquake radiate?"])
    t = f"log10 E = 1.5M + 4.8 with E in J. E = 10^(1.5 × {m} + 4.8) = {calc(f'10**(1.5 * {m} + 4.8)', E)} J."
    return q, t, f"The radiated energy is approximately {fmt(E)} J.", E, "J"


def _gr_count(r: random.Random) -> Drill:
    a, b, m = round(r.uniform(3.0, 7.5), 1), round(r.uniform(0.7, 1.3), 2), round(r.uniform(3.0, 7.0), 1)
    N = 10 ** (a - b * m)
    q = r.choice([
        f"A region's Gutenberg–Richter relation is log10 N = {a} − {b} M, with N the annual number of events of magnitude ≥ M. Calculate the expected annual number of earthquakes with M ≥ {m}.",
        f"With a = {a} and b = {b} in the Gutenberg–Richter law, how many earthquakes of magnitude {m} or larger are expected per year?",
    ])
    t = f"log10 N = a − b M, so N = 10^({a} − {b} × {m}) = {calc(f'10**({a} - {b} * {m})', N)}."
    return q, t, f"About {fmt(N)} earthquakes per year are expected.", N, ""


def _gr_ratio(r: random.Random) -> Drill:
    b = round(r.uniform(0.7, 1.3), 2); m1 = round(r.uniform(3.0, 5.5), 1); m2 = round(m1 + r.choice([0.5, 1.0, 1.5, 2.0]), 1)
    ratio = 10 ** (b * (m2 - m1))
    q = f"With a Gutenberg–Richter b-value of {b}, how many times more earthquakes of magnitude ≥ {m1} than of magnitude ≥ {m2} are expected?"
    t = f"N(≥M) ∝ 10^(−bM), so N(≥{m1}) / N(≥{m2}) = 10^(b ({m2} − {m1})) = {calc(f'10**({b} * ({m2} - {m1}))', ratio)}."
    return q, t, f"About {fmt(ratio)} times more events are expected.", ratio, ""


def _sp_distance(r: random.Random) -> Drill:
    dt, vp = round(r.uniform(2.0, 40.0), 1), round(r.uniform(5.5, 7.0), 1)
    ratio = round(r.uniform(1.70, 1.80), 2); vs = round(vp / ratio, 2)
    d = dt * vp * vs / (vp - vs)
    q = r.choice([
        f"The S−P time at a station is {dt} s. With Vp = {vp} km/s and Vs = {vs} km/s, calculate the hypocentral distance.",
        f"A seismogram shows the S wave arriving {dt} seconds after the P wave. Assuming P and S velocities of {vp} and {vs} km/s, how far away is the earthquake?",
    ])
    t = f"Δt = d/Vs − d/Vp, so d = Δt · Vp · Vs / (Vp − Vs) = {calc(f'{dt} * {vp} * {vs} / ({vp} - {vs})', d)} km."
    return q, t, f"The hypocentral distance is approximately {fmt(d)} km.", d, "km"


def _eew_lead_time(r: random.Random) -> Drill:
    D, vs, tw = r.choice([30, 40, 50, 60, 80, 100, 120, 150]), round(r.uniform(3.2, 3.8), 1), round(r.uniform(3.0, 10.0), 1)
    lead = D / vs - tw
    q = f"An earthquake early warning is issued {tw} s after the origin time. With an S-wave speed of {vs} km/s, how much warning time does a city {D} km from the epicenter receive before the S waves arrive?"
    t = f"S waves arrive at t = D / Vs = {calc(f'{D} / {vs}', D / vs)} s. Lead time = {fmt(D / vs)} − {tw} = {calc(f'{D} / {vs} - {tw}', lead)} s."
    return q, t, f"The warning time is about {fmt(lead)} s.", lead, "s"


def _stress_drop(r: random.Random) -> Drill:
    mw, rad = round(r.uniform(4.0, 7.5), 1), r.choice([1, 2, 3, 5, 8, 10, 15, 20, 30])
    m0 = 10 ** (1.5 * mw + 9.1)
    ds = 7 * m0 / (16 * (rad * 1e3) ** 3) / 1e6
    q = r.choice([
        f"For an Mw {mw} earthquake on a circular fault of radius {rad} km, estimate the static stress drop in MPa using Δσ = 7 M0 / (16 r³).",
        f"Estimate the stress drop (MPa) of a magnitude Mw {mw} event with a circular rupture of radius {rad} km.",
    ])
    t = (f"M0 = 10^(1.5 × {mw} + 9.1) = {calc(f'10**(1.5 * {mw} + 9.1)', m0)} N·m. Circular crack: Δσ = 7 M0 / (16 r³) with r = {rad}e3 m. "
         f"Δσ = {calc(f'7 * {fmt(m0)} / (16 * ({rad}e3)**3) / 1e6', ds)} MPa.")
    return q, t, f"The stress drop is approximately {fmt(ds)} MPa.", ds, "MPa"


def _recurrence(r: random.Random) -> Drill:
    D, v = round(r.uniform(0.5, 10.0), 1), round(r.uniform(1.0, 40.0), 1)
    T = D / (v * 1e-3)
    q = r.choice([
        f"A fault slips {D} m in each characteristic earthquake and has a long-term slip rate of {v} mm/yr. Calculate the average recurrence interval.",
        f"With {D} m of slip per event and a slip rate of {v} mm/year, how often do characteristic earthquakes recur on this fault?",
    ])
    t = f"T = D / v with v = {v} mm/yr = {v}e-3 m/yr. T = {calc(f'{D} / ({v}e-3)', T)} years."
    return q, t, f"The recurrence interval is about {fmt(T)} years.", T, "yr"


def _poisson_prob(r: random.Random) -> Drill:
    T, t = r.choice([50, 75, 100, 150, 200, 250, 300, 400, 475, 500, 700, 1000, 1500, 2475]), r.randint(5, 100)
    p = 100 * (1 - math.exp(-t / T))
    q = r.choice([
        f"Earthquakes on a fault follow a Poisson process with a mean recurrence interval of {T} years. Calculate the probability (in %) of at least one event in the next {t} years.",
        f"For a Poisson model with a {T}-year return period, what is the probability of at least one occurrence in {t} years?",
    ])
    t_ = f"P = 1 − exp(−t/T) = 1 − exp(−{t}/{T}). In percent: {calc(f'100 * (1 - exp(-{t} / {T}))', p)} %."
    return q, t_, f"The probability is about {fmt(p)}%.", p, "%"


def _rupture_area(r: random.Random) -> Drill:
    mw = round(r.uniform(5.0, 8.0), 1)
    A = 10 ** (-3.49 + 0.91 * mw)
    q = r.choice([f"Using the Wells and Coppersmith relation log10(RA) = −3.49 + 0.91 Mw, estimate the rupture area (km²) of an Mw {mw} earthquake.",
                  f"Estimate the expected rupture area in km² for a magnitude {mw} earthquake (Wells and Coppersmith, all slip types)."])
    t = f"log10(RA) = −3.49 + 0.91 Mw. RA = 10^(−3.49 + 0.91 × {mw}) = {calc(f'10**(-3.49 + 0.91 * {mw})', A)} km²."
    return q, t, f"The rupture area is approximately {fmt(A)} km².", A, "km²"


TEMPLATES: Dict[str, Callable[[random.Random], Drill]] = {
    "mw_from_m0": _mw_from_m0, "m0_from_mw": _m0_from_mw, "m0_from_fault": _m0_from_fault,
    "slip_from_mw": _slip_from_mw, "energy_ratio": _energy_ratio, "amplitude_ratio": _amplitude_ratio,
    "radiated_energy": _radiated_energy, "gr_count": _gr_count, "gr_ratio": _gr_ratio,
    "sp_distance": _sp_distance, "eew_lead_time": _eew_lead_time, "stress_drop": _stress_drop,
    "recurrence": _recurrence, "poisson_prob": _poisson_prob, "rupture_area": _rupture_area,
}


def make(n: int, seed: int, rl: bool) -> List[dict]:
    r = random.Random(seed)
    names = sorted(TEMPLATES)
    out, seen, attempts = [], set(), 0
    while len(out) < n:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError(f"only {len(out)} unique drills after {attempts} draws; widen the parameter ranges")
        name = names[r.randrange(len(names))]
        q, think, final, value, _ = TEMPLATES[name](r)
        if q in seen:
            continue
        seen.add(q)
        cid = hashlib.md5(f"drill::{name}::{q}".encode()).hexdigest()
        response = f"<think>{think}</think>{final}"
        if rl:
            out.append({"id": len(out), "prompt": q, "type": "numerical", "chunk_id": cid, "template": name,
                        "reference_response": response, "reference_final_answer": final, "reference_think": think,
                        "final_numerical_value": value, "key_concepts": [], "label_source": "exact_drill"})
        else:
            out.append({"instruction": q, "response": response, "type": "numerical", "chunk_id": cid,
                        "source": "drills", "template": name})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rl", action="store_true", help="write the RL/eval schema (prompt, final_numerical_value, ...)")
    args = ap.parse_args()
    rows = make(args.n, args.seed, args.rl)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} drills ({len(TEMPLATES)} templates) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
