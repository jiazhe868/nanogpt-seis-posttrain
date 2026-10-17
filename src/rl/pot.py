"""Program-of-thought: the model writes a short program, an interpreter computes the numbers.

Diagnosis (audit §15.9): free-prose derivations make the 113M model write every intermediate
number itself (~4 tokens each, top-1 0.61-0.74 per token on held-out/train), so a 300-token
derivation almost never survives sampling, and even a finished derivation is restated correctly
only 39-53% of the time. The small models that do well on numerical word problems (TinyGSM,
Physics of LMs iGSM) answer with executable programs instead. Here the reasoning block is a
program with named variables:

    <think>
    area_ca = 50000  # km², felt area of the M5.8 California event
    area_va = 1000000  # km², felt area of the 2011 Virginia event
    answer = area_va / area_ca
    </think>Answer: 20.

The model only copies the question's numbers and writes relations; `run_program` computes the
rest, and the decoder (src.rl.rollout, `pot=True`) forces "Answer: <value>" after </think>.

The interpreter is deliberately tiny: assignments to plain names, arithmetic (+ - * / // % **),
unary +/-, the math functions and constants allowed by src.rl.verifiers.safe_eval_number, and
names already assigned. No loops, attributes, subscripts, imports or calls outside the whitelist;
the same exponent and magnitude caps apply. Comments are ignored by the parser.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Dict, Optional, Tuple

from src.rl.verifiers import (MAX_ABS_EXPONENT, MAX_ABS_RESULT, _allowed_binary_ops,
                              _allowed_constants, _allowed_functions, _allowed_unary_ops)

MAX_PROGRAM_CHARS = 2000
MAX_STATEMENTS = 40
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)
MATH_IMPORT = re.compile(r"^\s*(?:import math|from math import \*|from math import [\w, ]+)\s*$", re.M)
MATH_PREFIX = re.compile(r"\bmath\.(?=[a-z_][a-z0-9_]*\b)")


class ProgramError(ValueError):
    pass


def _eval(node, env: Dict[str, float]):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ProgramError(f"non-numeric literal {node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _allowed_constants:
            return _allowed_constants[node.id]
        raise ProgramError(f"undefined name {node.id}")
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _allowed_unary_ops):
            raise ProgramError("unary op")
        v = _eval(node.operand, env)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _allowed_binary_ops):
            raise ProgramError("binary op")
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow):
            if abs(b) > MAX_ABS_EXPONENT:
                raise ProgramError("exponent too large")
            return a ** b
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
            raise ProgramError("division by zero")
        return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b,
                ast.Mod: lambda: a % b}[type(node.op)]()
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _allowed_functions or node.keywords:
            raise ProgramError("call not allowed")
        return _allowed_functions[node.func.id](*[_eval(a, env) for a in node.args])
    raise ProgramError(f"{type(node).__name__} not allowed")


# Paper notation (notation=True; 2026-10-16, audit §15.14). The base model was pretrained on
# papers and Wikipedia, not code, so solutions can instead be written the way papers write them:
# one equation per line, `SYMBOL = expression [unit]`, with short standard symbols (Mw, M0, μ, v_p),
# ^ for powers, log10 / ln. This normalisation is deterministic -- no guessing, unlike
# run_program_lenient -- and strict mode is unchanged, so earlier results stay reproducible.
_NOTATION_SUBS = [("×", "*"), ("·", "*"), ("−", "-"), ("–", "-"), ("÷", "/"), ("^", "**"), ("π", "pi")]
_UNIT_BRACKET = re.compile(r"\s*\[[^\]]*\]\s*$")
_BARE_LOG = re.compile(r"\blog\s*\(")
_LN = re.compile(r"\bln\s*\(")


def normalize_notation(src: str) -> Optional[str]:
    """Paper notation -> the interpreter's arithmetic; None if a line uses a bare `log(`, which
    seismology papers use for log10 but Python for ln -- solutions must say log10 or ln."""
    out = []
    for raw in src.split("\n"):
        line = _UNIT_BRACKET.sub("", raw.split("#", 1)[0]).rstrip()
        if not line.strip():
            continue
        for x, y in _NOTATION_SUBS:
            line = line.replace(x, y)
        if _BARE_LOG.search(line):
            return None
        out.append(_LN.sub("log(", line))
    return "\n".join(out)


def extract_program(text: str) -> str:
    """The program inside a reasoning block: a ```python fence if present, else the text."""
    m = CODE_FENCE.search(text)
    return (m.group(1) if m else text).strip()


def run_program(src: str, notation: bool = False) -> Tuple[Optional[float], Dict[str, float]]:
    """Execute a restricted program -> (answer, variables). answer is `answer` if assigned,
    else the last assigned value; None if the program is invalid or unsafe. With `notation`,
    paper notation is normalised first (normalize_notation)."""
    src = THOUSANDS.sub("", extract_program(src))
    # teachers (and students imitating them) write `import math` / `math.log10(x)`: the same
    # whitelisted functions, so the prefix is accepted; any other import still fails
    src = MATH_IMPORT.sub("", src)
    src = MATH_PREFIX.sub("", src)
    if notation:
        src = normalize_notation(src)
        if src is None:
            return None, {}
    if not src or len(src) > MAX_PROGRAM_CHARS:
        return None, {}
    try:
        tree = ast.parse(src, mode="exec")
    except (SyntaxError, ValueError):
        return None, {}
    if len(tree.body) > MAX_STATEMENTS:
        return None, {}
    env: Dict[str, float] = {}
    last = None
    try:
        for st in tree.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                v = _eval(st.value, env)
                name = st.targets[0].id
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id in env:
                v = _eval(ast.BinOp(left=ast.Name(id=st.target.id, ctx=ast.Load()), op=st.op, right=st.value), env)
                name = st.target.id
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                continue                                   # a docstring-style note
            else:
                return None, {}
            v = float(v)
            if not math.isfinite(v) or abs(v) > MAX_ABS_RESULT:
                return None, {}
            env[name] = v
            last = v
    except (ProgramError, ArithmeticError, ValueError, TypeError, OverflowError, RecursionError):
        return None, {}
    return env.get("answer", last), env


# ---------------------------------------------------------------------------
# Lenient reading of the student's working (2026-10-16, audit §15.13).
# The base model was pretrained on papers and Wikipedia, not code: 45.5% of its programs crash,
# mostly on a misspelled variable (30% of samples) or syntax (12%). `run_program_lenient` reads
# the working the way a person would instead of the way Python does: paper notation is
# normalised (^ × · − and units written after a number), lines that do not parse are skipped,
# and an undefined name resolves to the closest defined one when that match is unambiguous.
# ---------------------------------------------------------------------------
_NOTATION = [("×", "*"), ("·", "*"), ("−", "-"), ("–", "-"), ("^", "**"), ("÷", "/")]
_UNIT_TAIL = re.compile(r"^(\s*[A-Za-z_]\w*\s*=\s*[-+]?\d[\d.,]*(?:[eE][-+]?\d+)?)\s+[A-Za-zµμ°%²³/·*^.\-\s]+$")


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _resolve(name: str, env: Dict[str, float]) -> Optional[str]:
    """The defined name a misspelling most plausibly means, or None if ambiguous/too far."""
    if not env:
        return None
    norm = lambda x: x.lower().replace("_", "")
    exact = [k for k in env if norm(k) == norm(name)]
    if len(exact) == 1:
        return exact[0]
    scored = sorted((_edit(norm(name), norm(k)), k) for k in env)
    best, k = scored[0]
    if best <= max(2, len(name) // 4) and (len(scored) == 1 or scored[1][0] > best):
        return k
    return None


def run_program_lenient(src: str) -> Tuple[Optional[float], Dict[str, float], Dict[str, int]]:
    """Execute line by line; returns (answer, variables, repairs-made counts)."""
    src = MATH_PREFIX.sub("", MATH_IMPORT.sub("", THOUSANDS.sub("", extract_program(src))))
    if not src or len(src) > MAX_PROGRAM_CHARS:
        return None, {}, {}
    env: Dict[str, float] = {}
    last, fixes = None, {"notation": 0, "units": 0, "skipped_line": 0, "renamed": 0}
    for raw in src.split("\n")[:MAX_STATEMENTS * 2]:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        fixed = line
        for x, y in _NOTATION:
            fixed = fixed.replace(x, y)
        fixed = fixed.replace("****", "**")
        m = _UNIT_TAIL.match(fixed)
        if m:
            fixed = m.group(1)
        fixes["notation"] += fixed != line and not m
        fixes["units"] += bool(m)
        try:
            tree = ast.parse(fixed.strip(), mode="exec")
        except SyntaxError:
            fixes["skipped_line"] += 1
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign) or len(tree.body[0].targets) != 1 \
                or not isinstance(tree.body[0].targets[0], ast.Name):
            fixes["skipped_line"] += 1
            continue
        st = tree.body[0]
        for _ in range(4):                       # repair up to 4 misspelled names in a line
            try:
                v = float(_eval(st.value, env))
                break
            except ProgramError as e:
                msg = str(e)
                if not msg.startswith("undefined name "):
                    v = None
                    break
                bad = msg.split("undefined name ", 1)[1]
                good = _resolve(bad, env)
                if good is None:
                    v = None
                    break
                for node in ast.walk(st.value):
                    if isinstance(node, ast.Name) and node.id == bad:
                        node.id = good
                fixes["renamed"] += 1
            except (ArithmeticError, ValueError, TypeError, OverflowError, RecursionError):
                v = None
                break
        else:
            v = None
        if v is None or not math.isfinite(v) or abs(v) > MAX_ABS_RESULT:
            fixes["skipped_line"] += 1
            continue
        env[st.targets[0].id] = v
        last = v
    return env.get("answer", last), env, fixes


def test_pot() -> None:
    assert run_program("a = 50,000\nb = 1,000,000\nanswer = b / a")[0] == 20.0
    assert abs(run_program("mw = 7.2  # moment magnitude\nm0 = 10**(1.5*mw + 9.1)")[0] - 10 ** 19.9) < 1e12
    assert run_program("```python\nx = 3\ny = x**2 + sqrt(16)\n```")[0] == 13.0
    assert run_program("x = 2\nx *= 5")[0] == 10.0
    assert run_program("import math\nanswer = math.log10(1000) + math.pi * 0")[0] == 3.0
    for bad in ("import os", "x = __import__('os')", "x = (1).real", "for i in range(3): x = i",
                "x = y + 1", "x = 9**9**9", "x = [1, 2]", "x = 1/0", "x = open('f')"):
        assert run_program(bad)[0] is None, bad
    a, _, f = run_program_lenient("rupture_speed_km_s = 3.2\nlength_km = 100\nanswer = length_km / rupture_speed_kms")
    assert abs(a - 31.25) < 1e-9 and f["renamed"] == 1, (a, f)
    a, _, f = run_program_lenient("mw = 6.5\nm0 = 10^(1.5*mw + 9.1)\narea = 10 km²\nanswer = m0 / (3e10 * area * 1e6)")
    assert abs(a - 10 ** 18.85 / 3e17) / (10 ** 18.85 / 3e17) < 1e-9 and f["units"] == 1, (a, f)
    assert run_program_lenient("x = 2\nthis is prose\nanswer = x * 3")[0] == 6.0
    # paper notation: units in brackets, ^ powers, Greek symbols, × ; strict mode still rejects ^
    a, _ = run_program("Mw = 6.5 [-]\nM0 = 10^(1.5*(Mw + 6.07)) [N m]\nμ = 3e10 [Pa]\nA = 10e6 [m^2]\n"
                       "D = M0 / (μ × A) [m]", notation=True)
    want = 10 ** (1.5 * 12.57) / (3e10 * 10e6)
    assert abs(a - want) / want < 1e-12, a
    assert run_program("x = log(100)", notation=True)[0] is None          # ambiguous base
    assert abs(run_program("x = ln(e) [-]", notation=True)[0] - 1.0) < 1e-12
    assert abs(run_program("y = log10(1000)\nz = 2^3 × π", notation=True)[0] - 8 * math.pi) < 1e-12
    assert run_program("y = 2^3")[0] is None                              # strict: ^ is not a power
    # equidistant misspelling: ambiguous, so the line is skipped and the last value stands
    assert run_program_lenient("a_long = 1\nb_long = 2\nanswer = c_long")[0] == 2.0
    assert run_program_lenient("import os\nanswer = 2")[0] == 2.0       # the import line is skipped, never run
    print("[ok] pot: interpreter regression cases pass")


if __name__ == "__main__":
    test_pot()
