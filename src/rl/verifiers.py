"""The reward: programmatic scoring of Phase III rollouts.

Seven verifiers, combined by CompositeRewardEngine. Each one is a hypothesis
about a construct, and therefore an attack surface — so every one has an
adversarial regression test at the bottom of this file, and the whole contract
is tabulated in README section 5.5.

Also holds safe_eval_number: a restricted arithmetic evaluator, because these
functions run on strings the policy wrote and `eval()` would be a
code-execution hole.

    python -m src.rl.verifiers     # run the adversarial test suite
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, List, Optional, Set


def get_ngrams(text: str, n: int = 2) -> Set[str]:
    """Extract word n-grams from text."""
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


class EchoPenaltyVerifier:
    """Detects and penalizes prompt echo / regurgitation attacks."""

    # Normalising by the CANDIDATE's n-gram count (the original design) made the
    # penalty divisible: appending unrelated filler inflates the denominator and
    # walks the ratio back under the threshold without removing any copied text.
    # Measured on the held-out set, "verbose filler + correct answer" scored
    # +0.139 ABOVE the genuine reference answer -- a live hacking channel, and
    # one that also penalised 11% of genuine answers for legitimately restating
    # the question.
    #
    # Normalising by the PROMPT instead measures the construct we actually care
    # about -- "how much of the question did you regurgitate" -- and is immune to
    # padding: a literal echo scores 1.000 whether or not filler is appended.
    # Thresholds recalibrated on 808 genuine reference answers (max 0.760):
    # 0.85 penalises 0.00% of them, 0.70 penalises 0.62%, while a literal echo
    # attack scores 1.000 and is always caught.
    HARD_ECHO_RATIO = 0.85
    SOFT_ECHO_RATIO = 0.70

    def __call__(self, candidate_text: str, prompt_text: str) -> float:
        cand_clean = re.sub(r"<think>.*?</think>", "", candidate_text, flags=re.DOTALL)
        cand_ngrams = get_ngrams(cand_clean, n=2)
        prompt_ngrams = get_ngrams(prompt_text, n=2)

        if not cand_ngrams or not prompt_ngrams:
            return 0.0

        overlap = cand_ngrams.intersection(prompt_ngrams)
        overlap_ratio = len(overlap) / len(prompt_ngrams)

        if overlap_ratio > self.HARD_ECHO_RATIO:
            return -1.5
        elif overlap_ratio > self.SOFT_ECHO_RATIO:
            return -0.5
        return 0.0


class FormatVerifier:
    """Verifies ChatML / reasoning format compliance (<think>...</think> tags)."""

    def __call__(self, text: str, expect_think: bool = False) -> float:
        n_start = text.count("<think>")
        n_end = text.count("</think>")

        if n_start == 0 and n_end == 0:
            # Numerical tasks are trained with a reasoning block; skipping it
            # is a (mild) format miss. Conceptual answers have no think block.
            return -0.25 if expect_think else 0.0

        # Exactly one well-ordered pair. Counting occurrences, rather than
        # comparing first indices, closes a hole: a completion could open one
        # block and sprinkle extra "</think>" through the prose and still
        # collect the full +0.5.
        if n_start == 1 and n_end == 1 and text.find("<think>") < text.find("</think>"):
            return 0.5
        return -0.5


class CleanLanguageVerifier:
    """Penalizes context-dependent lead-in phrases ('as described in the text', 'based on the passage').

    Returns 0.0 for clean text (a constant positive reward cancels in the
    group-relative advantage and only inflates logged/benchmark numbers).
    """

    LEADIN_REGEX = re.compile(
        r"(?:based on|according to|as stated in|as described in|in) the (?:provided |given )?(?:text|passage|study|article|paper|excerpt)|"
        r"the (?:provided |given )?(?:text|passage|study|article|paper|excerpt) (?:states|indicates|shows|discusses|describes|explains|notes|mentions)|"
        r"\bthe authors?\b|\bet al\b",
        re.IGNORECASE,
    )

    def __call__(self, text: str) -> float:
        if self.LEADIN_REGEX.search(text):
            return -0.5
        return 0.0


class KeyConceptVerifier:
    """Evaluates coverage of key scientific domain concepts for conceptual/summarization tasks."""

    # Simple verb/clause check to detect ungrammatical keyword dumps
    VERB_PATTERN = re.compile(
        r"\b(?:is|are|was|were|show|shows|showed|result|results|calculate|calculated|indicates|observed|causes|caused|determines|found|analysis|describes|implies)\b",
        re.IGNORECASE,
    )

    def __call__(self, candidate_text: str, key_concepts: List[str], is_echo: bool = False) -> float:
        if not key_concepts or is_echo:
            return 0.0

        # Concepts must appear in the visible answer. Matching against the raw
        # completion would let the policy stuff every keyword inside <think>
        # (invisible to the user) and collect the maximum reward.
        cand_clean = re.sub(r"<think>.*?</think>", "", candidate_text, flags=re.DOTALL)

        # Check for ungrammatical keyword dumps (no verbs or standard prose structure)
        if not self.VERB_PATTERN.search(cand_clean) and len(cand_clean.split()) < 15:
            return -0.5  # Penalize raw keyword dumping

        cand_lower = cand_clean.lower()
        matched = sum(1 for kw in key_concepts if kw.lower() in cand_lower)
        ratio = matched / len(key_concepts)

        if ratio >= 0.75:
            return 0.8
        elif ratio >= 0.50:
            return 0.4
        elif ratio >= 0.25:
            return 0.1
        else:
            return -0.5


class NumericalVerifier:
    """Verifies numerical precision against single final_numerical_value in final answer."""

    NUMBER_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    # "1.8 × 10^19" / "1.8x10**19" -> "1.8e19" so the value parses as one number
    SCI_NOTATION = re.compile(
        r"([-+]?\d+(?:\.\d+)?)\s*[×x·*]\s*10\s*(?:\^|\*\*)?\s*[({\[]?([-+]?\d+)[)}\]]?"
    )
    # Score only the tail of the final answer. Taking min-error over the whole
    # answer hands the policy one lottery ticket per number emitted; measured on
    # the 166 held-out numerical items, a question-blind policy spraying 8
    # plausible values scored 14.3% @5% under min-over-first-8 -- ABOVE the real
    # model (13.3%). Restricting to the last 3 numbers both lowers that floor to
    # 6.0% and RAISES the ceiling on reference answers from 68.1% to 86.1%,
    # because the stated answer sits near the end and "first 8" often misses it.
    #
    #   rule                 ceiling   floor
    #   min over first 8      68.1%    14.3%
    #   last 1                58.4%     1.9%
    #   last 2                76.5%     3.8%
    #   last 3 (chosen)       86.1%     6.0%
    #   last 5                92.8%     9.3%
    ANSWER_TAIL_NUMBERS = 3

    def extract_numbers(self, text: str) -> List[float]:
        text = self.SCI_NOTATION.sub(lambda m: f"{m.group(1)}e{m.group(2)}", text)
        matches = self.NUMBER_REGEX.findall(text)
        numbers = []
        for m in matches:
            try:
                val = float(m)
                if not (math.isnan(val) or math.isinf(val)):
                    numbers.append(val)
            except ValueError:
                pass
        return numbers

    def __call__(self, candidate_text: str, target_val: float | None) -> float:
        if target_val is None:
            return 0.0

        # Evaluate numbers strictly from post-<think> final answer
        think_match = re.search(r"<think>(.*?)</think>", candidate_text, flags=re.DOTALL)
        if think_match:
            answer_text = candidate_text[think_match.end():]
        else:
            answer_text = candidate_text

        cand_numbers = self.extract_numbers(answer_text)
        if not cand_numbers:
            return -1.0  # Failed to produce any numbers in final answer

        # NOTE: a hard reject at len(numbers) > 8 used to live here as the
        # anti-spam defence. Tail scoring subsumes it -- an attacker
        # emitting 40 numbers still gets only ANSWER_TAIL_NUMBERS chances, so
        # the spray success rate is 5.7% either way. Meanwhile the cap fired on
        # 40.4% of genuine reference answers (median 6 numbers, p90 34: a
        # detailed derivation legitimately cites many values), dragging mean
        # r_num on the ground truth down to +0.274. Removing it restores +0.723
        # without reopening the hole.

        # Check the stated answer (tail of the final-answer span), not every
        # number the completion happens to contain -- see ANSWER_TAIL_NUMBERS.
        denom = abs(target_val) if abs(target_val) > 1e-7 else 1.0
        scored = cand_numbers[-self.ANSWER_TAIL_NUMBERS:]
        best_rel_err = min(abs(cand - target_val) / denom for cand in scored)

        if best_rel_err <= 0.01:
            return 1.0  # 1% precision match
        elif best_rel_err <= 0.05:
            return 0.5  # 5% precision match
        else:
            return -1.0


class LengthPenaltyVerifier:
    """Applies soft penalty if <think> block exceeds length threshold."""

    def __call__(self, text: str, max_chars: int = 2500) -> float:
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if not think_match:
            return 0.0

        think_len = len(think_match.group(1))
        if think_len > max_chars:
            overflow = think_len - max_chars
            penalty = -min(0.5, (overflow / max_chars) * 0.3)
            return penalty
        return 0.0


class TerminationVerifier:
    """Rewards completions that actually end, instead of running to the token cap.

    Nothing in the v2 reward valued stopping, while KeyConceptVerifier rewards
    keyword coverage — which longer text accumulates more of. Measured
    consequence: the GRPO policy's stop rate FELL from 52.5% (SFT) to 27.5%,
    i.e. RL taught the model to ramble past <|im_end|>. This term supplies the
    missing incentive; the trainer passes `terminated` from the rollout.
    """

    def __call__(self, terminated: bool) -> float:
        return 0.0 if terminated else -0.5


class CompositeRewardEngine:
    """Adversarially hardened reward engine."""

    def __init__(self):
        self.echo_verifier = EchoPenaltyVerifier()
        self.format_verifier = FormatVerifier()
        self.clean_verifier = CleanLanguageVerifier()
        self.num_verifier = NumericalVerifier()
        self.concept_verifier = KeyConceptVerifier()
        self.length_verifier = LengthPenaltyVerifier()
        self.termination_verifier = TerminationVerifier()

    def score(self, candidate_text: str, rl_item: Dict[str, Any],
              terminated: bool = True) -> Dict[str, float]:
        prompt = rl_item.get("prompt", "")
        task_type = rl_item.get("type", "conceptual")
        target_val = rl_item.get("final_numerical_value")
        key_concepts = rl_item.get("key_concepts", [])

        # Check echo attack
        r_echo = self.echo_verifier(candidate_text, prompt)
        is_echo = (r_echo < 0.0)

        is_numerical = (task_type == "numerical" and target_val is not None)
        r_format = self.format_verifier(candidate_text, expect_think=is_numerical)
        r_clean = self.clean_verifier(candidate_text)
        r_length = self.length_verifier(candidate_text)
        r_term = self.termination_verifier(terminated)

        if is_numerical:
            r_num = self.num_verifier(candidate_text, target_val)
            r_concept = 0.0
        else:
            r_num = 0.0
            r_concept = self.concept_verifier(candidate_text, key_concepts, is_echo=is_echo)

        total_reward = r_echo + r_num + r_concept + r_format + r_clean + r_length + r_term

        return {
            "r_echo": r_echo,
            "r_num": r_num,
            "r_concept": r_concept,
            "r_format": r_format,
            "r_clean": r_clean,
            "r_length": r_length,
            "r_term": r_term,
            "total_reward": round(total_reward, 4),
        }


# ===========================================================================
# Restricted arithmetic evaluation
# ===========================================================================
# Used to execute expressions the model wrote (src.analysis.diagnose_capability
# oracle). `eval()` here would run whatever a rollout happened to emit, so this
# walks the AST and permits only arithmetic: numeric literals, + - * / // % **,
# unary +/-, parentheses, a whitelist of math functions, and pi/e/tau. Names,
# attributes, subscripts, comprehensions, and any other call are rejected.
# Beyond the whitelist: exponents are capped so `9**9**9` cannot hang the
# caller, results must be finite, and expression length is capped.

MAX_EXPR_LEN = 300
MAX_ABS_EXPONENT = 64          # 10**64 is already far outside any seismology value
MAX_ABS_RESULT = 1e300

_allowed_functions = {
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "sqrt": math.sqrt, "abs": abs,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "radians": math.radians, "degrees": math.degrees,
    "floor": math.floor, "ceil": math.ceil,
    "pow": pow, "round": round, "min": min, "max": max,
}
_allowed_constants = {"pi": math.pi, "e": math.e, "tau": math.tau}
_allowed_binary_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_allowed_unary_ops = (ast.UAdd, ast.USub)


class UnsafeExpression(ValueError):
    pass


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression(f"non-numeric literal: {node.value!r}")
        return node.value

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _allowed_unary_ops):
            raise UnsafeExpression(f"unary op {type(node.op).__name__}")
        value = _eval_node(node.operand)
        return +value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _allowed_binary_ops):
            raise UnsafeExpression(f"binary op {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_ABS_EXPONENT:
                raise UnsafeExpression(f"exponent too large: {right}")
            return left ** right
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise UnsafeExpression("division by zero")
        return {
            ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right,
        }[type(node.op)]()

    if isinstance(node, ast.Name):
        if node.id in _allowed_constants:
            return _allowed_constants[node.id]
        raise UnsafeExpression(f"name not allowed: {node.id}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _allowed_functions:
            raise UnsafeExpression("call to non-whitelisted function")
        if node.keywords:
            raise UnsafeExpression("keyword arguments not allowed")
        return _allowed_functions[node.func.id](*[_eval_node(a) for a in node.args])

    raise UnsafeExpression(f"node type {type(node).__name__}")


def safe_eval_number(expr: str) -> Optional[float]:
    """Evaluate an arithmetic expression; None if unsafe or invalid.

    Never raises on hostile input — callers scoring model output want a value
    or nothing, not an exception mid-rollout.
    """
    if not expr or len(expr) > MAX_EXPR_LEN:
        return None
    try:
        value = _eval_node(ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_ABS_RESULT:
        return None
    return result


def test_safe_eval() -> None:
    valid = {
        "3e10 * 5e7 * 1": 1.5e18,
        "2 + 3 * 4": 14,
        "(2/3) * log10(1.5e18) - 10.7": (2 / 3) * math.log10(1.5e18) - 10.7,
        "10 ** 3": 1000,
        "sqrt(16)": 4.0,
        "-5.5e-3": -5.5e-3,
    }
    for expr, want in valid.items():
        got = safe_eval_number(expr)
        assert got is not None and abs(got - want) < 1e-9 * max(1, abs(want)), (expr, got, want)

    blocked = [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "(lambda: 1)()",
        "9**9**9",
        "[i for i in range(10**9)]",
        "x + 1",
        "1/0",
        "print(1)",
        "().__class__.__bases__",
        "eval('1+1')",
        "a" * 400,
    ]
    for expr in blocked:
        assert safe_eval_number(expr) is None, f"NOT blocked: {expr}"

    print(f"[ok] safe_eval: {len(valid)} valid expressions evaluated, "
          f"{len(blocked)} hostile inputs blocked")


# ===========================================================================
# Adversarial regression suite
# ===========================================================================


def test_verifiers():
    """Every known attack on the reward, kept as a permanent regression test."""
    print("=" * 80)
    print("Adversarial regression suite for the reward engine")
    print("=" * 80)

    engine = CompositeRewardEngine()

    sample_rl_item = {
        "prompt": "Calculate seismic moment M0 for rupture area A=400, slip d=1.5, shear modulus mu=30.",
        "type": "numerical",
        "final_numerical_value": 1.8e19,
    }

    # Test 1: Genuine Correct Answer
    good_resp = "<think>Compute M0 = 30e9 * 400e6 * 1.5 = 1.8e19</think>\nFinal Answer: M0 = 1.8e19 N*m"
    s1 = engine.score(good_resp, sample_rl_item)

    # Test 2: Echo Attack (Copying prompt parameters)
    echo_resp = "<think>Thinking...</think>\nCalculate seismic moment M0 for rupture area A=400, slip d=1.5, shear modulus mu=30."
    s2 = engine.score(echo_resp, sample_rl_item)

    # Test 3: Correct answer in scientific "×10^" notation
    sci_resp = "<think>M0 = mu*A*d</think>\nThe seismic moment is 1.8 × 10^19 N·m."
    s3 = engine.score(sci_resp, sample_rl_item)

    # Test 4: Number spam (enumerating values to game min-error matching)
    spam_resp = "<think>hmm</think>\nCandidates: " + ", ".join(f"{v}e19" for v in range(1, 20))
    s4 = engine.score(spam_resp, sample_rl_item)

    # Test 5: Keyword stuffing hidden inside <think> must NOT earn concept reward
    concept_item = {
        "prompt": "Explain why subduction zones host deep earthquakes.",
        "type": "conceptual",
        "final_numerical_value": None,
        "key_concepts": ["transformational", "faulting", "olivine", "spinel", "dehydration", "embrittlement", "slab", "metastable"],
    }
    stuff_resp = "<think>transformational faulting olivine spinel dehydration embrittlement slab metastable</think>\nIt is important to consider many factors."
    s5 = engine.score(stuff_resp, concept_item)

    # Test 6: extra stray </think> tags must not earn the format reward
    stray_resp = ("<think>compute M0</think>The moment is 1.8e19 N*m. </think>For a strike-slip "
                  "fault the value differs. </think>In summary it is 1.8e19 N*m.")
    s6 = engine.score(stray_resp, sample_rl_item)

    # Test 7: ADAPTIVE number spray. An attacker who knows only the answer
    # distribution emits 40 plausible values hoping one lands within tolerance.
    # Tail scoring must bound them to ANSWER_TAIL_NUMBERS chances, so a spray
    # whose tail misses earns nothing however long it is.
    spray_resp = ("<think>Consider the range.</think>Possible values: "
                  + ", ".join(f"{v}e18" for v in range(1, 41)))
    s7 = engine.score(spray_resp, sample_rl_item)

    # Test 8: PADDING DILUTION. Appending unrelated filler to an echo must not
    # walk the echo penalty back under threshold. Regression for the +0.139
    # channel found by src.analysis.validate_measurement hacking.
    padded_echo = echo_resp + " Further analysis confirms this. " * 20
    s8 = engine.score(padded_echo, sample_rl_item)

    print(f"Good Response Score:      {s1}")
    print(f"Echo Attack Score:        {s2}")
    print(f"Sci-Notation Score:       {s3}")
    print(f"Number-Spam Score:        {s4}")
    print(f"Think-Stuffing Score:     {s5}")
    print(f"Stray-</think> Score:     {s6}")
    print(f"40-Number Spray Score:    {s7}")
    print(f"Padded-Echo Score:        {s8}")

    assert s1["total_reward"] > 1.0, "Good answer must have high positive reward"
    assert s2["total_reward"] < -0.5, "Echo attack must be heavily penalized"
    assert s3["r_num"] == 1.0, "Scientific ×10^ notation must parse as a correct match"
    assert s4["r_num"] < 0.0, "Number spam must not earn the numerical reward"
    assert s5["r_concept"] <= 0.0, "Keyword stuffing inside <think> must earn no concept reward"
    assert s6["r_format"] < 0.0, "Extra stray </think> tags must not earn the format reward"
    assert s7["r_num"] < 0.0, "A 40-number spray whose tail misses must earn no numerical reward"
    assert s8["r_echo"] <= s2["r_echo"], (
        "Padding an echo must not reduce the echo penalty (dilution channel)")
    assert s8["total_reward"] <= s1["total_reward"], (
        "A padded echo must never out-score a genuine answer")

    print("\n[SUCCESS] all 8 adversarial verifier tests passed")


if __name__ == "__main__":
    test_verifiers()
    test_safe_eval()
