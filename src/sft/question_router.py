"""Does a question ask for a computed number? Routes the deployed model's decoding (src/sft/chat.py).

The deployed model answers numerical questions with executed paper-notation equations and vote@8, which
decode without a repetition penalty. Questions that do not ask for a number get prose decoding instead
(repetition penalty 1.2): without the penalty, conceptual answers loop and cover fewer key concepts.
Tuned on data/rl/val.jsonl; on the held-out set it sends all 153 scored numerical questions to the
equations and 640 of the 641 others to prose.
"""
import re

PROSE_START = re.compile(r"^\W*(explain|summari[sz]e|extract|describe|discuss|compare|contrast|outline|list|why|what are|"
                         r"what were|how do|how does|how did|identify|interpret|assess|evaluate|analy[sz]e|critique|"
                         r"characteri[sz]e)\b", re.I)
COMPUTE_START = re.compile(r"^\W*(given|using|calculate|compute|derive|estimate|determine|assuming|assume|suppose|if|"
                           r"consider|for a|a|an)\b", re.I)
PROSE_ASK = re.compile(r"\b(outline|explain|describe|discuss|summari[sz]e|interpret|why)\b", re.I)
COMPUTE = re.compile(r"\b(calculate|compute|estimate|determine|derive|find|how (many|much|long|far|large|deep|high)|"
                     r"what (is|was|would be) the (value|magnitude|moment|ratio|rate|number|distance|depth|time|energy|"
                     r"velocity|period|frequency|stress|strain|probability|area|width|length))\b|"
                     r"\bin (n m|nm|km|m/s|mpa|pa|kpa|gpa|hz|s|years?|%|percent|joules?|j|g)\b", re.I)
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


def is_numerical_question(q: str) -> bool:
    """True when the question asks for a computed number."""
    if PROSE_START.search(q):
        return False
    if COMPUTE_START.search(q) and not PROSE_ASK.search(q):
        return True
    nums = [n for n in NUMBER.findall(q) if not re.fullmatch(r"(1[6-9]|20)\d\d", n)]   # ignore years
    return bool(COMPUTE.search(q)) and len(nums) >= 1 or len(nums) >= 3
