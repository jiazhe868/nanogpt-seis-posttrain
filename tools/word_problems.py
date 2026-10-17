#!/usr/bin/env python3
"""Lexically diverse seismology word problems with exact answers (stdlib only).

Why (audit §15.9): trained on counterfactual re-instantiations of 1,248 real questions, the 113M
model learns to bind values within those question wordings (fresh numbers: 3% -> 16%), but
nothing transfers to a new wording -- not even when the held-out computation is a plain ratio,
which 212 training items share. tools/numeric_drills.py has the opposite problem: 15 templates,
one phrasing each. This generator varies the WORDING as much as the operation: ~60 quantities,
many entity types, several phrasings per operation, both directions of each ratio, distractor
numbers and context sentences, so the same operation is never asked the same way twice.

Every question stores its template key; `--holdout` reserves a fraction of question phrasings
for a cross-wording test set that shares no phrasing with training.

    python tools/word_problems.py --n 20000 --out data/sources/word_problems_train.jsonl \\
        --heldout-out data/eval/word_problems_heldout.jsonl --holdout 0.2
"""

from __future__ import annotations

import argparse
import json
import math
import random

# (name, unit, lo, hi, decimals, log-uniform)
QUANTITIES = [
    ("felt area", "km²", 1e3, 3e6, 0, True), ("rupture length", "km", 5, 1200, 0, True),
    ("rupture width", "km", 5, 200, 0, True), ("average slip", "m", 0.1, 25, 1, True),
    ("slip rate", "mm/yr", 0.5, 45, 1, True), ("recurrence interval", "years", 20, 5000, 0, True),
    ("number of aftershocks", "events", 10, 20000, 0, True), ("number of stations", "stations", 3, 400, 0, True),
    ("peak ground acceleration", "g", 0.02, 1.8, 2, True), ("peak ground velocity", "cm/s", 1, 200, 0, True),
    ("shear-wave velocity", "m/s", 150, 1500, 0, True), ("S–P time", "s", 1, 120, 1, True),
    ("hypocentral depth", "km", 2, 650, 0, True), ("epicentral distance", "km", 5, 3000, 0, True),
    ("population exposed", "people", 1e3, 3e7, 0, True), ("reported fatalities", "deaths", 5, 3e5, 0, True),
    ("economic loss", "million USD", 5, 3e5, 0, True), ("number of damaged buildings", "buildings", 20, 5e5, 0, True),
    ("tsunami run-up height", "m", 0.5, 40, 1, True), ("ground displacement", "cm", 1, 600, 0, True),
    ("stress drop", "MPa", 0.3, 30, 1, True), ("fault dip", "degrees", 10, 89, 0, False),
    ("event duration", "s", 2, 300, 0, True), ("corner frequency", "Hz", 0.02, 15, 2, True),
    ("seismicity rate", "events per year", 0.5, 2000, 1, True), ("b-value", "", 0.6, 1.6, 2, False),
    ("strain rate", "nanostrain per year", 1, 300, 0, True), ("uplift", "m", 0.1, 12, 1, True),
    ("landslide count", "landslides", 10, 60000, 0, True), ("liquefaction area", "km²", 0.5, 800, 1, True),
    ("warning time", "s", 2, 90, 0, True), ("P-wave velocity", "km/s", 4.5, 8.5, 1, False),
    ("rupture velocity", "km/s", 1.0, 4.0, 1, False), ("coseismic subsidence", "cm", 5, 250, 0, True),
    ("moment rate", "N·m per year", 1e15, 1e20, 0, True), ("aftershock zone length", "km", 5, 1500, 0, True),
    ("intensity-VI area", "km²", 100, 9e5, 0, True), ("number of felt reports", "reports", 20, 3e5, 0, True),
    ("pore-pressure increase", "MPa", 0.01, 5, 2, True), ("injection volume", "m³", 1e3, 5e6, 0, True),
]
ENTITIES = [
    "the {y} {r} earthquake", "an M{m} event in {r}", "the {r} mainshock", "a {y} sequence near {r}",
    "event A", "event B", "the {r} swarm", "a moderate {r} earthquake", "the largest {y} aftershock",
    "a deep earthquake beneath {r}", "a strike-slip event on the {r} fault", "a megathrust rupture off {r}",
]
REGIONS = ["Tohoku", "Nepal", "Chile", "Sumatra", "Turkey", "Napa", "Christchurch", "Kaikōura", "Ridgecrest",
           "Haiti", "Wenchuan", "Alaska", "Kumamoto", "Lombok", "Oklahoma", "Mexico City", "Taiwan", "Iran",
           "Peru", "Italy", "Greece", "Japan", "Virginia", "Utah", "Nevada", "Ecuador", "the Philippines"]
CONTEXT = [
    "Seismologists compiled the observations from regional networks and field surveys.",
    "Both events were recorded by the same seismic network.",
    "The values below come from a post-event reconnaissance report.",
    "Such comparisons are routinely used in seismic hazard studies.",
    "The data were reported in a review of recent sequences.",
    "Engineers use these numbers to compare damage potential.",
    "", "", "",
]


def sample_value(q, rng):
    _, _, lo, hi, dec, logu = q
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if logu else rng.uniform(lo, hi)
    v = round(v, dec)
    if dec == 0:
        v = int(max(v, 1))
        if v > 1000:  # realistic rounding of big counts
            mag = 10 ** (len(str(v)) - 2)
            v = int(round(v / mag) * mag)
    return v


def fmt_in(v):
    """How a value appears in a question: thousands separators for large integers."""
    if isinstance(v, int):
        return f"{v:,}" if v >= 10000 else str(v)
    return f"{v:g}"


def nc(v):
    """The value as typed in an expression: exactly the question's digits, separators included
    (the calculator in src.rl.rollout drops thousands separators), so binding is a verbatim copy."""
    return fmt_in(v)


def fmt_out(v):
    return f"{v:.4g}"


def entity(rng):
    return rng.choice(ENTITIES).format(y=rng.randint(1900, 2024), r=rng.choice(REGIONS), m=round(rng.uniform(4, 8.5), 1))


def distractor(rng):
    return rng.choice([
        f"The event occurred at a depth of {rng.randint(5, 60)} km.",
        f"It was recorded in {rng.randint(1950, 2024)}.",
        f"The nearest station was {rng.randint(3, 300)} km away.",
        f"The region has about {rng.randint(2, 90)} active faults.",
        f"Its magnitude was {round(rng.uniform(4, 8.8), 1)}.",
        "",
    ])


# Each template: key, question pattern, expression builder, answer unit rule.
# {A}/{B}: entity phrases; {q}: quantity name; {a}/{b}: values with unit.
RATIO_Q = [
    ("ratio1", "How many times larger is the {q} of {A} than that of {B}?"),
    ("ratio2", "By what factor does the {q} of {A} exceed that of {B}?"),
    ("ratio3", "What is the ratio of the {q} of {A} to the {q} of {B}?"),
    ("ratio4", "Compute {A}'s {q} divided by {B}'s {q}."),
    ("ratio5", "Express the {q} of {A} as a multiple of the {q} of {B}."),
    ("ratio6", "How does the {q} of {A} compare with that of {B}? Give the ratio."),
    ("ratio7", "Calculate how many times the {q} of {B} fits into the {q} of {A}."),
    ("ratio8", "Estimate the {q} ratio between {A} and {B} (first relative to second)."),
    ("ratio9", "What multiple of {B}'s {q} is {A}'s {q}?"),
    ("ratio10", "Find the factor by which the {q} of {B} would have to be multiplied to equal that of {A}."),
]
PCT_Q = [
    ("pct1", "What percentage of the {q} of {A} does the {q} of {B} represent?"),
    ("pct2", "Express the {q} of {B} as a percentage of that of {A}."),
    ("pct3", "The {q} of {B} is what percent of the {q} of {A}?"),
    ("pct4", "Calculate the {q} of {B} relative to {A}, in percent."),
    ("pct5", "What fraction of {A}'s {q}, in percent, is {B}'s {q}?"),
]
CHANGE_Q = [
    ("chg1", "By what percentage did the {q} change from {A} to {B}?"),
    ("chg2", "Compute the percent change in {q} going from {A} to {B}."),
    ("chg3", "What is the relative change, in percent, of the {q} of {B} with respect to {A}?"),
    ("chg4", "How much larger or smaller, in percent, is the {q} of {B} than that of {A}?"),
]
DIFF_Q = [
    ("diff1", "How much greater is the {q} of {A} than that of {B}?"),
    ("diff2", "What is the difference in {q} between {A} and {B}?"),
    ("diff3", "Compute the {q} of {A} minus the {q} of {B}."),
    ("diff4", "By how much does the {q} of {A} exceed the {q} of {B}?"),
]
SUM_Q = [
    ("sum1", "What is the combined {q} of {A} and {B}?"),
    ("sum2", "Compute the total {q} for {A} and {B} together."),
    ("sum3", "Add the {q} of {A} and the {q} of {B}."),
]
MW_Q = [
    ("mw1", "{A} had a seismic moment of {M0} N·m. Using Mw = (2/3)(log10 M0 − 9.1), what is its moment magnitude?"),
    ("mw2", "Compute the moment magnitude of {A}, whose seismic moment was {M0} N·m (Mw = (2/3)(log10 M0 − 9.1))."),
    ("mw3", "Given a seismic moment of {M0} N·m for {A}, find Mw with the Hanks–Kanamori relation Mw = (2/3)(log10 M0 − 9.1)."),
]
M0_Q = [
    ("m01", "{A} had moment magnitude Mw {m}. Using M0 = 10^(1.5 Mw + 9.1), what was its seismic moment in N·m?"),
    ("m02", "Compute the seismic moment of an Mw {m} earthquake ({A}) with M0 = 10^(1.5 Mw + 9.1)."),
    ("m03", "What seismic moment (N·m) corresponds to {A} of magnitude Mw {m}? Use M0 = 10^(1.5 Mw + 9.1)."),
]
ENERGY_Q = [
    ("en1", "How many times more energy did {A} (M{m1}) release than {B} (M{m2})? Use E ∝ 10^(1.5M)."),
    ("en2", "Using log10 E = 1.5M + 4.8, by what factor does the radiated energy of an M{m1} event exceed that of an M{m2} event?"),
    ("en3", "{A} had magnitude {m1} and {B} had magnitude {m2}. Compute the ratio of their radiated energies (energy scales as 10^(1.5M))."),
]
AMP_Q = [
    ("amp1", "How many times larger is the recorded amplitude of an M{m1} event than an M{m2} event at the same distance (amplitude ∝ 10^M)?"),
    ("amp2", "{A} (magnitude {m1}) and {B} (magnitude {m2}) were recorded at the same station. What is their amplitude ratio, given amplitude scales as 10^M?"),
]
RATE_Q = [
    ("rate1", "A fault slips at {r} mm/yr. How much slip, in meters, accumulates over {t} years?"),
    ("rate2", "With a slip rate of {r} mm/yr, compute the slip deficit in meters after {t} years."),
    ("rate3", "If strain accumulates at {r} mm/yr for {t} years without an earthquake, how many meters of slip are stored?"),
]
SP_Q = [
    ("sp1", "The S–P time at a station is {t} s. Using distance ≈ 8 × (S–P time), how far is the hypocenter in km?"),
    ("sp2", "A station records an S–P interval of {t} s. With the rule of thumb d = 8 km/s × t, estimate the hypocentral distance."),
    ("sp3", "Estimate the distance to {A} from an S–P time of {t} s, taking 8 km per second of S–P time."),
]
TRAVEL_Q = [
    ("tt1", "A P wave travels {d} km at {v} km/s. How long does it take, in seconds?"),
    ("tt2", "Compute the travel time of a wave crossing {d} km at {v} km/s."),
    ("tt3", "How many seconds does it take for a {v} km/s wave to reach a station {d} km away?"),
]


def solution(given, asked, expr, value, unit):
    g = "; ".join(f"{n} = {v}" + (f" {u}" if u else "") for n, v, u in given)
    return (f"<think>The question asks for {asked}. Given: {g}. {asked[0].upper() + asked[1:]} = <<{expr}>> = "
            f"{fmt_out(value)}.</think>Answer: {fmt_out(value)}{(' ' + unit) if unit else ''}.")


def two_quantity(rng, family):
    q = rng.choice(QUANTITIES)
    name, unit = q[0], q[1]
    A, B = entity(rng), entity(rng)
    while B == A:
        B = entity(rng)
    a, b = sample_value(q, rng), sample_value(q, rng)
    if a == b:
        return None
    if family == "ratio":
        if a < b and rng.random() < 0.8:
            a, b = b, a
        key, pat = rng.choice(RATIO_Q)
        v, expr, asked, out_unit = a / b, f"{nc(a)} / {nc(b)}", f"the ratio of the {name} of the first to that of the second", ""
    elif family == "pct":
        key, pat = rng.choice(PCT_Q)
        v, expr, asked, out_unit = b / a * 100, f"{nc(b)} / {nc(a)} * 100", f"the {name} of the second as a percentage of the first", "%"
    elif family == "chg":
        key, pat = rng.choice(CHANGE_Q)
        v, expr, asked, out_unit = (b - a) / a * 100, f"({nc(b)} - {nc(a)}) / {nc(a)} * 100", f"the percent change in {name}", "%"
    elif family == "diff":
        if a < b:
            a, b = b, a
        key, pat = rng.choice(DIFF_Q)
        v, expr, asked, out_unit = a - b, f"{nc(a)} - {nc(b)}", f"the difference in {name}", unit
    else:
        key, pat = rng.choice(SUM_Q)
        v, expr, asked, out_unit = a + b, f"{nc(a)} + {nc(b)}", f"the combined {name}", unit
    u = f" {unit}" if unit else ""
    facts = [f"The {name} of {A} was {fmt_in(a)}{u}.", f"For {B}, the {name} was {fmt_in(b)}{u}."]
    if rng.random() < 0.3:
        facts = [f"{A[0].upper() + A[1:]} had a {name} of {fmt_in(a)}{u}, while {B} had {fmt_in(b)}{u}."]
    ctx = [rng.choice(CONTEXT), distractor(rng)] + facts
    question = " ".join(s for s in ctx if s) + " " + pat.format(q=name, A=A, B=B)
    given = [(f"{name} of {A}", fmt_in(a), unit), (f"{name} of {B}", fmt_in(b), unit)]
    return key, question, solution(given, asked, expr, v, out_unit), v


def special(rng, family):
    A, B = entity(rng), entity(rng)
    if family == "mw":
        m0 = float(f"{10 ** rng.uniform(15, 22.5):.2e}")
        key, pat = rng.choice(MW_Q)
        q = pat.format(A=A, M0=f"{m0:.2e}".replace("e+", "e"))
        v = (2 / 3) * (math.log10(m0) - 9.1)
        return key, q, solution([("seismic moment M0", f"{m0:.2e}", "N·m")], "the moment magnitude Mw",
                                f"(2/3) * (log10({m0:.2e}) - 9.1)", v, ""), v
    if family == "m0":
        m = round(rng.uniform(3.5, 9.2), 1)
        key, pat = rng.choice(M0_Q)
        v = 10 ** (1.5 * m + 9.1)
        return key, pat.format(A=A, m=m), solution([("moment magnitude Mw", m, "")], "the seismic moment M0",
                                                   f"10**(1.5 * {m} + 9.1)", v, "N·m"), v
    if family in ("en", "amp"):
        m1 = round(rng.uniform(4, 9), 1)
        m2 = round(m1 - rng.choice([0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]), 1)
        key, pat = rng.choice(ENERGY_Q if family == "en" else AMP_Q)
        k = 1.5 if family == "en" else 1.0
        v = 10 ** (k * (m1 - m2))
        asked = "the energy ratio" if family == "en" else "the amplitude ratio"
        return key, pat.format(A=A, B=B, m1=m1, m2=m2), solution(
            [("magnitude of the larger event", m1, ""), ("magnitude of the smaller event", m2, "")], asked,
            f"10**({k} * ({m1} - {m2}))", v, ""), v
    if family == "rate":
        r = round(rng.uniform(0.5, 45), 1)
        t = rng.choice([50, 80, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000])
        key, pat = rng.choice(RATE_Q)
        v = r * t / 1000
        return key, pat.format(r=r, t=t), solution([("slip rate", r, "mm/yr"), ("time", t, "years")],
                                                   "the accumulated slip in meters", f"{r} * {t} / 1000", v, "m"), v
    if family == "sp":
        t = round(rng.uniform(1, 90), 1)
        key, pat = rng.choice(SP_Q)
        v = 8 * t
        return key, pat.format(t=t, A=A), solution([("S–P time", t, "s")], "the hypocentral distance",
                                                   f"8 * {t}", v, "km"), v
    d = rng.randint(10, 3000)
    vv = round(rng.uniform(3.0, 8.5), 1)
    key, pat = rng.choice(TRAVEL_Q)
    v = d / vv
    return key, pat.format(d=d, v=vv), solution([("distance", d, "km"), ("velocity", vv, "km/s")],
                                                "the travel time in seconds", f"{d} / {vv}", v, "s"), v


# Formula application: the question states a relation and its parameters; the answer is the
# substitution. Real questions do this constantly ("Use M0 = μAD with μ = 3×10^10 Pa ...").
VARS = [("M0", "N·m"), ("Mw", ""), ("A", "km²"), ("D", "m"), ("L", "km"), ("W", "km"), ("R", "km"),
        ("PGA", "g"), ("Vs", "m/s"), ("T", "s"), ("N", "events"), ("E", "J"), ("Δσ", "MPa"), ("t", "years"),
        ("v", "km/s"), ("h", "km"), ("f", "Hz"), ("Q", ""), ("K", ""), ("S", "m"), ("P", "")]
FORMULAS = [  # (relation with {y},{x1},{x2},{x3},{a},{b}, python with the same slots)
    ("{y} = {a} × {x1} + {b}", "{a} * {x1} + {b}"),
    ("{y} = {a} × {x1}^{b}", "{a} * {x1}**{b}"),
    ("{y} = {a} log10({x1}) + {b}", "{a} * log10({x1}) + {b}"),
    ("{y} = {x1} × {x2} / {x3}", "{x1} * {x2} / {x3}"),
    ("{y} = {x1} / ({x2} × {x3})", "{x1} / ({x2} * {x3})"),
    ("{y} = sqrt({x1} × {x2})", "sqrt({x1} * {x2})"),
    ("{y} = 10^({a} × {x1} + {b})", "10**({a} * {x1} + {b})"),
    ("{y} = {a} × ({x1} − {x2})", "{a} * ({x1} - {x2})"),
    ("{y} = ({x1} + {x2}) / {a}", "({x1} + {x2}) / {a}"),
    ("{y} = {x1} × {x2} × {x3}", "{x1} * {x2} * {x3}"),
    ("{y} = {a} × exp(−{x1} / {x2})", "{a} * exp(-{x1} / {x2})"),
]
FORMULA_Q = [
    ("fa1", "Using the relation {rel}, compute {y} for {vals}."),
    ("fa2", "Given that {y} is related to the other quantities by {rel}, what is {y} when {vals}?"),
    ("fa3", "{ctx}The empirical relation {rel} applies. Calculate {y}, taking {vals}."),
    ("fa4", "If {vals}, find {y} from {rel}."),
    ("fa5", "Apply {rel} with {vals}. What value of {y} results?"),
]
CHAIN_Q = [
    ("ch1", "A fault is {l} km long and {w} km wide, with an average slip of {d} m. Using M0 = μ A D with μ = {mu} Pa, compute the seismic moment in N·m (convert the area to m²)."),
    ("ch2", "Compute the seismic moment of a rupture {l} km long and {w} km wide that slipped {d} m on average, assuming a rigidity of {mu} Pa (M0 = μAD, area in m²)."),
    ("ch3", "For a rupture area of {l} km by {w} km and mean slip {d} m, with shear modulus {mu} Pa, what is M0 = μAD in N·m?"),
]


def formula_problem(rng):
    rel_t, py_t = rng.choice(FORMULAS)
    names = rng.sample(VARS, 4)
    y, xs = names[0], names[1:]
    a = round(rng.uniform(0.2, 3.0), 2)
    b = round(rng.uniform(-5, 5), 1) if "{b}" in rel_t and "^{b}" not in rel_t else round(rng.uniform(0.3, 2.5), 2)
    if "10^(" in rel_t:
        a, b = round(rng.uniform(0.5, 1.5), 2), round(rng.uniform(-2, 4), 1)
    xv = [round(math.exp(rng.uniform(math.log(0.5), math.log(500))), rng.choice([0, 1, 2])) for _ in xs]
    xv = [int(v) if float(v).is_integer() else v for v in xv]
    if "exp(" in rel_t:
        xv[0] = round(rng.uniform(0.1, 5), 1)
    used = [k for k, slot in enumerate(("{x1}", "{x2}", "{x3}")) if slot in rel_t]
    if "-" in py_t and "exp" not in py_t and len(used) >= 2 and xv[0] <= xv[1]:
        xv[0], xv[1] = xv[1] + xv[0], xv[0]
    sub = {"y": y[0], "a": a, "b": b}
    for k, slot in enumerate(("x1", "x2", "x3")):
        sub[slot] = xs[k][0]
    rel = rel_t.format(**sub)
    vals_txt = ", ".join(f"{xs[k][0]} = {fmt_in(xv[k])}" + (f" {xs[k][1]}" if xs[k][1] else "") for k in used)
    pysub = {"a": a, "b": b, "y": y[0]}
    for k, slot in enumerate(("x1", "x2", "x3")):
        pysub[slot] = nc(xv[k]) if k in used else "0"
    expr = py_t.format(**pysub)
    key, pat = rng.choice(FORMULA_Q)
    q = pat.format(rel=rel, y=y[0], vals=vals_txt, ctx=(rng.choice(CONTEXT) + " ") if rng.random() < 0.5 else "")
    from_ = [(xs[k][0], fmt_in(xv[k]), xs[k][1]) for k in used]
    try:
        v = eval(expr.replace(",", ""), {"log10": math.log10, "sqrt": math.sqrt, "exp": math.exp})
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if not isinstance(v, float) and not isinstance(v, int) or not math.isfinite(v) or v == 0 or abs(v) > 1e30:
        return None
    resp = (f"<think>The question asks for {y[0]} from {rel}. Given: " +
            "; ".join(f"{n} = {val}" + (f" {u}" if u else "") for n, val, u in from_) +
            f". Substituting: {y[0]} = <<{expr}>> = {fmt_out(v)}.</think>Answer: {fmt_out(v)}" +
            (f" {y[1]}" if y[1] else "") + ".")
    return key, q, resp, float(v)


def chain_problem(rng):
    l, w = rng.randint(5, 900), rng.randint(5, 200)
    d = round(rng.uniform(0.1, 20), 1)
    mu = rng.choice(["3e10", "3.0e10", "3.2e10", "4e10", "2.5e10", "3.3e10"])
    key, pat = rng.choice(CHAIN_Q)
    area = l * w * 1e6
    v = float(mu) * area * d
    resp = (f"<think>The question asks for the seismic moment M0 = μAD. Given: L = {l} km; W = {w} km; D = {d} m; "
            f"μ = {mu} Pa. Area A = <<{l} * {w} * 1e6>> = {fmt_out(area)} m². M0 = <<{mu} * {fmt_out(area)} * {d}>> = "
            f"{fmt_out(v)}.</think>Answer: {fmt_out(v)} N·m.")
    return key, pat.format(l=l, w=w, d=d, mu=mu), resp, v


FAMILIES = ["fa"] * 6 + ["ch"] + ["ratio"] * 5 + ["pct"] * 2 + ["chg"] * 2 + ["diff"] * 2 + ["sum"] + \
           ["mw", "m0", "en", "amp", "rate", "sp", "tt"]


def all_keys():
    groups = [RATIO_Q, PCT_Q, CHANGE_Q, DIFF_Q, SUM_Q, MW_Q, M0_Q, ENERGY_Q, AMP_Q, RATE_Q, SP_Q, TRAVEL_Q,
              FORMULA_Q, CHAIN_Q]
    return [k for g in groups for k, _ in g]


def make(n, seed, allowed, max_tries=50):
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        fam = rng.choice(FAMILIES)
        for _ in range(max_tries):
            r = (formula_problem(rng) if fam == "fa" else chain_problem(rng) if fam == "ch" else
                 two_quantity(rng, fam) if fam in ("ratio", "pct", "chg", "diff", "sum") else special(rng, fam))
            if r is None:
                continue
            key, q, resp, v = r
            if key in allowed and q not in seen and math.isfinite(v):
                break
        else:
            continue
        seen.add(q)
        out.append({"instruction": q, "response": resp, "type": "numerical", "template": key,
                    "family": fam, "target": v, "chunk_id": f"wp-{key}"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-heldout", type=int, default=600)
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of question phrasings held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-out", required=True)
    args = ap.parse_args()
    keys = all_keys()
    rng = random.Random(args.seed)
    # hold out phrasings within every family, so each family is still trained
    by_fam = {}
    for k in keys:
        by_fam.setdefault(k.rstrip("0123456789"), []).append(k)
    held = set()
    for fam, ks in by_fam.items():
        ks = sorted(ks)
        rng.shuffle(ks)
        held.update(ks[:max(1, round(args.holdout * len(ks)))])
    train = make(args.n, args.seed + 1, set(keys) - held)
    test = make(args.n_heldout, args.seed + 2, held)
    tq = {r["instruction"] for r in train}
    test = [r for r in test if r["instruction"] not in tq]
    for path, rows in ((args.out, train), (args.heldout_out, test)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(keys)} phrasings, {len(held)} held out ({sorted(held)}) | train {len(train)} -> {args.out} | "
          f"held-out phrasings {len(test)} -> {args.heldout_out}")


if __name__ == "__main__":
    main()
