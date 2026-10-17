"""The numerical answer a reference derivation actually concludes.

This sets the label that both the RL reward (NumericalVerifier) and the eval score against,
so a wrong label caps the reward and the benchmark alike.

The v1 rule (`prepare_rl_dataset.legacy_extract_final_numerical_value`) took the last regex
match of '[=≈~]' / 'is …' in the reference's final-answer text. That produced labels like the
age in "…avulsion of the Ganges at ~2.5 ka" instead of the Mw 7.5 the derivation computes, the
"120 encounters" given in the question, or 0 and 1 on derive/outline questions that have no
numeric answer at all. Only ~60% of held-out labels equalled the derivation's own last result.

The rule here, in order:

  1. No label unless the question asks for a quantity (calculate / estimate / how many / …)
     and is not a purely symbolic derivation: with no numbers in the question,
     "derive / outline / show how" asks for an expression.
  2. An explicit "the answer is X" / "Final answer: X" that the final answer restates wins.
  3. Otherwise the last value the reasoning COMPUTES — the number after an '=' / '≈' whose
     left side contains arithmetic, so assignments like "M = 7" do not count — that the final
     answer restates (within 5%, or as a percentage of a fraction). If a second computed
     value restates the same final-answer number more exactly, it wins (the reference
     considered the first as an alternative).
  4. If the reasoning concludes nothing the final answer restates, use what the final answer
     computes beyond it.

Validated by blind adjudication (old and new label shown in random order, judged from the
question and reference only): on 40 untouched items the frozen rule's labels were right
for 80% against 50% for v1; of the 22 items where they differ, the new label won 16 and v1
won 4 (sign test p = 0.012). The remaining errors are mostly references whose last
computation is a side quantity the question never asked for.

    from src.rl.numerical_target import derive_numerical_target
    value, source = derive_numerical_target(prompt, reference_think, reference_final_answer)
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

# "1.8 × 10^19", "1.8 x 10**19", "1.8·10^19" -> "1.8e19"
SCI_NOTATION = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*[×x·*]\s*10\s*(?:\^|\*\*)?\s*[({\[]?([-+]?\d+)[)}\]]?"
)
# Bare "10^19" (no mantissa) -> "1e19"
BARE_POWER = re.compile(r"(?<![\d.eE])10\s*(?:\^|\*\*)\s*[({\[]?([-+]?\d+)[)}\]]?")


def normalize_scientific_notation(text: str) -> str:
    text = SCI_NOTATION.sub(lambda m: f"{m.group(1)}e{m.group(2)}", text)
    text = BARE_POWER.sub(lambda m: f"1e{m.group(1)}", text)
    return text


NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
NUM_AT = re.compile(NUM)
# a number that stands on its own (a given value), not part of an identifier like "M0" or "x_1"
NUM_FREE = re.compile(rf"(?<![A-Za-z_\d.^]){NUM}(?![A-Za-z_\d])|(?<![A-Za-z_\d.^]){NUM}(?=\s*[A-Za-z%°])")
DEC_POWER = re.compile(r"(?<![\d.])10\s*\^\s*\(?\s*([-+]?\d+\.\d+)\s*\)?")
ARITH = re.compile(
    r"[+*/×·√]|\^|(?<=[\d)])\s*[-−]\s*(?=[\d(.])|\s[-−]\s|\(\s*-|"
    r"\b(?:log|ln|exp|sqrt|sin|cos|tan|arcsin|arccos|arctan|asin|acos|atan|sinh|cosh|tanh|Phi|frac|times|cdot)\b|Φ"
)
BOUNDARY = re.compile(r"[.!?](?=\s)|\n|;")
COMPUTE = re.compile(
    r"\b(?:calculat\w*|comput\w*|estimat(?:e|es|ed|ing)\b|determin\w*|how (?:many|much|long|far|large|big|fast|deep)|"
    r"what (?:is|would be|are|was) the|find\b|convert\w*|quantif\w*|how many times|ratio|percent\w*|"
    r"how (?:does|do|did|would)[^?]{0,90}compare)",
    re.I,
)
SYMBOLIC = re.compile(r"\b(?:deriv\w*|outline\w*|describe the steps|show how)\b", re.I)
ANSWER_MARK = re.compile(
    r"\b(?:final answer|the answer|answer)\s*(?:is|=|:)\s*(?:approximately|about|roughly|~|≈)?\s*\$?(" + NUM + r")",
    re.I,
)


def norm(text: str) -> str:
    """Scientific notation to e-notation, thousands separators removed, 10^1.5 evaluated
    (and marked with '^' so it still counts as a computation)."""
    text = DEC_POWER.sub(lambda m: f"^{10 ** float(m.group(1)):.6g}", text)
    text = normalize_scientific_notation(text)
    text = re.sub(r"(\d),(\d{3}),(\d{3})\b", r"\1\2\3", text)
    return re.sub(r"(\d),(\d{3})\b", r"\1\2", text)


def close(a: float, b: float, tol: float) -> bool:
    d = abs(b) if abs(b) > 1e-12 else 1.0
    return abs(a - b) / d <= tol


def given_values(prompt: str) -> List[float]:
    out = []
    for m in NUM_FREE.finditer(norm(prompt)):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def all_numbers(text: str) -> List[float]:
    out = []
    for m in NUM_AT.finditer(norm(text).replace("^", " ")):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def computed_results(text: str) -> List[float]:
    """Values that END a computation: the number right after '=' / '≈' when the expression
    leading to it (back to the previous '=' or the sentence start) contains arithmetic, or when
    it continues a chain that did (unit conversions). Assignments ('M = 7') do not count."""
    t = norm(text)
    out: List[float] = []
    marks = list(re.finditer(r"[=≈]", t))
    prev_end, prev_computed = 0, False
    for i, m in enumerate(marks):
        lo = max(m.start() - 160, prev_end)
        cut = [b.end() for b in BOUNDARY.finditer(t, lo, m.start())]
        chained = i > 0 and not cut and prev_computed           # same sentence as a computed '='
        lead = t[cut[-1] if cut else lo:m.start()]
        j = m.end()
        while j < len(t) and t[j] == " ":
            j += 1
        is_pow = t.startswith("^", j)
        n = NUM_AT.match(t, j + is_pow)
        prev_end = m.end()
        computed_here = bool(n) and (is_pow or bool(ARITH.search(lead)) or chained)
        if not n:
            prev_computed = bool(ARITH.search(lead)) or chained
            continue
        if re.match(r"\s*(?:[+*/×·^(]|-\s*[\d(])", t[n.end():n.end() + 4]):   # expression continues
            prev_computed = computed_here or prev_computed
            continue
        prev_computed = computed_here
        if computed_here:
            v = float(n.group())
            if math.isfinite(v):
                out.append(v)
    return out


def has_given_data(prompt: str) -> bool:
    return bool(given_values(prompt))


def asks_for_number(prompt: str) -> bool:
    """A quantity is asked for, and the question is not a purely symbolic derivation."""
    if not COMPUTE.search(prompt):
        return False
    return has_given_data(prompt) or not SYMBOLIC.search(prompt)


def _match(f_values: List[float], r: float, tol: float) -> Optional[float]:
    """The final-answer value that restates r: directly, or as a percentage of a fraction."""
    for f in f_values:
        if close(f, r, tol):
            return r                       # same form: keep the derivation's (more precise) value
    for f in f_values:
        if close(f, 100 * r, tol) or close(f, r / 100, tol):
            return f                       # restated as a percentage: use the stated form
    return None


def derive_numerical_target(prompt: str, think: str, final: str, tol: float = 0.05) -> Tuple[Optional[float], str]:
    """(value, source) — see the module docstring for the rule and its validation."""
    if not asks_for_number(prompt):
        return None, "question_asks_for_no_number"
    F = all_numbers(final)
    marks = list(ANSWER_MARK.finditer(norm(think))) + list(ANSWER_MARK.finditer(norm(final)))
    for m in reversed(marks):
        v = _match(F, float(m.group(1)), tol)
        if v is not None:
            return v, "explicit_answer_statement"
    T_all = all_numbers(think)
    R_think = computed_results(think)
    R_final = computed_results(final)
    extended = [r for r in R_final
                if not any(close(t, r, 0.01) or close(t, 100 * r, 0.01) or close(t, r / 100, 0.01) for t in T_all)]
    for r in reversed(R_think):
        v = _match(F, r, tol)
        if v is None:
            continue
        f_star = min((f for f in F if close(f, v, tol)), key=lambda f: abs(f - v), default=None)
        if f_star is not None and not close(v, f_star, 0.01):
            exact = [x for x in R_think if close(x, f_star, 0.01)]
            if exact:
                return exact[-1], "last_computed_result"
        return v, "last_computed_result"
    if extended:          # the reasoning concluded nothing the final answer restates
        return extended[-1], "final_answer_extends_derivation"
    for r in reversed(R_final):
        return r, "final_answer_result"
    return None, "no_consistent_answer"


# ===========================================================================
# Regression suite: one case per failure mode found while adjudicating labels
# ===========================================================================

def test_numerical_target() -> None:
    cases = [
        # (name, prompt, think, final, expected value or None)
        ("trailing unrelated number (v1 took the age 2.5 ka)",
         "Estimate Mw using Mw = (2/3) log10(M0) - 10.7 for a slip of 2 m over 100 km x 30 km.",
         "M0 = 3e11 * 3e13 * 200 = 1.8e27 dyne-cm. Mw = (2/3)*27.2553 - 10.7 = 7.47.",
         "Mw is about 7.47, consistent with the dikes that formed at ~2.5 ka.", 7.47),
        ("a given value is not the answer (v1 took 5.8)",
         "A magnitude 5.8 earthquake was felt over 50,000 km2, another over 1,000,000 km2. How many times larger is the felt area?",
         "The ratio is 1,000,000 / 50,000 = 20.",
         "The felt area was 20 times larger than that of the magnitude 5.8 event.", 20.0),
        ("assignment 'M = 7' is not a computed result",
         "Calculate L for M=7.0 using lg L = 0.43M - 1.27, then the circular area.",
         "lg L = 0.43*7.0 - 1.27 = 1.74, L = 55 km. A = π * (27.5)^2 ≈ 2376 km². For M=7 typical lengths are 50-100 km.",
         "L is about 55 km and the area about 2376 km² for magnitude 7.", 2376.0),
        ("symbolic derivation question has no numeric label",
         "Derive the binned maximum likelihood estimator for a power law and provide the likelihood function.",
         "L(α) = Π [x_i^(1-α) - x_(i+1)^(1-α)]^(n_i), so log L = Σ n_i log(...) = 0 at the optimum.",
         "The MLE solves d log L / dα = 0 for α > 1.", None),
        ("decimal power 10^1.5 counts as computation",
         "How much more energy does an M 8.1 release than an M 7.1? Use log E = 1.5M + 4.8.",
         "E1/E2 = 10^(1.5*1) = 10^1.5 ≈ 31.62.",
         "It releases about 31.62 times more energy.", 31.62),
        ("minus sign without spaces is arithmetic",
         "With p = 0.2 for 30 people, calculate the probability of at least 4 leaders.",
         "P(X<4) = 0.1227, so P(X>=4)=1-0.12271844=0.87728156.",
         "The probability is approximately 0.8773 (or 87.73%).", 0.87728156),
        ("percentage restatement of a fraction keeps the stated form",
         "Using a Poisson model with a rate of 0.004 per year, what is the probability of at least one event in 30 years?",
         "P = 1 - exp(-0.004*30) = 1 - 0.8869 = 0.1131.",
         "The probability is about 11.3%.", 11.3),
        ("arctan result counts as computed",
         "The elevation difference is 1,500 m over 2,000 m. Calculate the slope angle in degrees.",
         "tan(θ) = 1500/2000 = 0.75. arctan(0.75) ≈ 36.87 degrees.",
         "The slope angle is about 36.87°.", 36.87),
        ("the stated value beats a near alternative",
         "The computed base shear is 120 kN and the measured 115 kN; compute the percentage difference.",
         "(5/115)*100% ≈ 4.35%. Alternatively, relative to computed: (5/120)*100% ≈ 4.17%.",
         "The computed value exceeds the measured one by about 4.35%.", 4.35),
        ("unit-conversion chain continues a computation",
         "Each of 4 sessions lasts 60 minutes. Calculate the total therapy time in hours.",
         "Total = 4 * 60 minutes = 240 minutes = 4 hours.",
         "The total therapy time is 4 hours.", 4.0),
    ]
    failures = []
    for name, prompt, think, final, want in cases:
        got, source = derive_numerical_target(prompt, think, final)
        ok = (got is None and want is None) or (got is not None and want is not None and close(got, want, 0.01))
        if not ok:
            failures.append(f"{name}: want {want}, got {got} ({source})")
    assert not failures, "numerical_target regressions:\n  " + "\n  ".join(failures)
    print(f"[ok] numerical_target: {len(cases)} label regression cases pass")


if __name__ == "__main__":
    test_numerical_target()
