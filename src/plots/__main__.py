"""Render every figure in assets/.

    python -m src.plots            # all
    python -m src.plots sft grpo   # a subset

Figures are regenerated from the run logs and results files in checkpoints/
and docs/, never from numbers pasted into the plotting code -- a defect this
project actually shipped once (PITFALLS F2).
"""

import sys

from src.plots import set_publication_style

GROUPS = ("workflow", "sft", "grpo", "pitfalls")


def main() -> int:
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(GROUPS)
    unknown = [w for w in wanted if w not in GROUPS]
    if unknown:
        print(f"unknown group(s): {unknown}; choose from {GROUPS}")
        return 2

    set_publication_style()
    for group in wanted:
        print(f"[{group}]")
        if group == "workflow":
            from src.plots import workflow
            workflow.main()
        elif group == "sft":
            from src.plots import sft
            sft.main()
        elif group == "grpo":
            from src.plots import grpo
            grpo.main()
        elif group == "pitfalls":
            from src.plots import pitfalls
            pitfalls.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
