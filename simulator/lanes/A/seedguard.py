"""Lane A seed guard -- catches what cruse1.used_seeds() cannot see.

THE HOLE (34th defect, 2026-08-23)
----------------------------------
cruse1.used_seeds() walks every results JSON but matches only the
literal keys "seeds" (list) and "seed" (scalar). Every Lane A search
script records its consumed seeds under **search_seeds**, which falls
to the walker's `else: walk(v)` branch, recurses into a list of bare
ints, and collects nothing. Same for playback_seeds.

Consequence, measured: 24000-24003 were consumed by CD1-BM's parameter
search and were INVISIBLE to the guard, so CD1-CM's "fresh" draw picked
them and CD1-CM and CD1-CT both ran on four already-burnt seeds.

cruse1.py is shared with Lane B and is not Lane A's to edit -- logged in
exchange/NEEDS-ROBIN.md. This module is the route around it.

USE
---
    from seedguard import fresh
    seeds = fresh(200000, 12)     # draws past BOTH guards, or raises
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def hidden_seeds():
    """Every int under any key ending in 'seeds' that is not 'seeds'."""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if (k.endswith("seeds") and k != "seeds"
                        and isinstance(v, list) and v
                        and all(isinstance(x, (int, float)) for x in v)):
                    out.update(int(x) for x in v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    for f in glob.glob("lanes/*/*.json") + glob.glob("results-*.json"):
        try:
            walk(json.load(open(f)))
        except Exception:
            continue
    return out


def prose_seeds():
    """Seeds recorded only in prose (.log/.md) -- the 2026-08-30 fleet
    defect (Lane A 89811164, judge-verified): consumption written to log
    or markdown files was invisible to both JSON-only guards, and 105
    seeds were silently reused across 22 committed records. Parses three
    forms, aggressively -- overcounting is SAFE for a consumption guard
    (a false-used seed only shrinks the draw space):
      literal:  seeds [70000, 70001, ...]   (elided lists included)
      range:    seeds 90000-90012
      reuse:    --reuse 96000,96001
    Judge-owned; probe = tests/fixture in the patch commit message."""
    import re
    pat_list = re.compile(r"seeds?\s*[\[(]([0-9,\s.\u2026]+)")
    pat_range = re.compile(r"seeds?\s+(\d{4,6})\s*[-\u2013]\s*(\d{4,6})")
    pat_reuse = re.compile(r"--reuse[= ]([0-9,\s]+)")
    out = set()
    # Consumption sources ONLY: logs are execution output, and .md files
    # count only when named as results. Packets/registrations/findings
    # narrate seeds in any modality ("seeds 60121-60124 remain fresh"
    # burnt four registered seeds in the v1 sweep, 2026-08-30) and are
    # excluded here; the burnt-seed backfill carries the same rule.
    files = (glob.glob("lanes/*/*.log")
             + [f for f in glob.glob("lanes/*/*.md") if "result" in f.lower()]
             + [f for f in glob.glob("exchange/*.md") if "result" in f.lower()]
             + glob.glob("results-*.md"))
    for f in files:
        try:
            txt = open(f, errors="replace").read()
        except Exception:
            continue
        import re as _re
        for m in pat_list.finditer(txt):
            for tok in _re.findall(r"\d{4,6}", m.group(1)):
                out.add(int(tok))
        for m in pat_range.finditer(txt):
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a <= 2000:
                out.update(range(a, b + 1))
        for m in pat_reuse.finditer(txt):
            for tok in _re.findall(r"\d{4,6}", m.group(1)):
                out.add(int(tok))
    return {x for x in out if 0 <= x <= 999999}


def all_used():
    from cruse1 import used_seeds
    return used_seeds() | hidden_seeds() | prose_seeds()


BLOCK_FILE = "seed_blocks.json"
LANE = "lane-a"


def lane_block():
    """Lane A's binding block from seed_blocks.json on main.

    PINNED 2026-08-25 (judge's accountability item 1, evening notes
    2026-08-24): fresh() may only draw inside this block. The final
    warning fired because 194000-194011 and 32000 were drawn outside the
    then-active 70000-79999 block; a draw outside the current block is a
    governance violation
    even if the seeds come back unused.
    """
    b = json.load(open(BLOCK_FILE))["blocks"][LANE]
    return int(b[0]), int(b[1])


def fresh(start=None, n=1, stop=None):
    """n seeds that BOTH guards agree are unused, drawn ONLY from Lane
    A's block in seed_blocks.json. start/stop outside the block raise
    instead of drawing -- there is deliberately no override argument."""
    lo, hi = lane_block()
    start = lo if start is None else start
    stop = (hi + 1) if stop is None else stop
    if not (lo <= start <= hi and lo < stop <= hi + 1):
        raise RuntimeError(
            f"seedguard: requested range [{start},{stop}) is outside Lane "
            f"A's block [{lo},{hi}] (seed_blocks.json). Refusing to draw.")
    u = all_used()
    got = [s for s in range(start, stop) if s not in u][:n]
    if len(got) < n:
        raise RuntimeError(f"only {len(got)} fresh seeds in "
                           f"[{start},{stop}); widen the range")
    return got


if __name__ == "__main__":
    from cruse1 import used_seeds
    v, h = used_seeds(), hidden_seeds()
    print(f"cruse1.used_seeds():      {len(v)}")
    print(f"hidden under *_seeds:     {len(h)}")
    print(f"INVISIBLE to cruse1:      {len(sorted(h - v))} seeds")
    lo, hi = lane_block()
    print(f"lane block (pinned):      [{lo},{hi}]")
    for bad in (32000, 194000, 20000, 69999):
        try:
            fresh(bad, 1)
            print(f"PIN SELF-TEST FAILED: fresh({bad}) drew outside the block")
            raise SystemExit(1)
        except RuntimeError as e:
            print(f"refused {bad}: OK ({str(e)[:60]}...)")
    got = fresh(n=4)
    assert all(lo <= s <= hi for s in got), got
    print(f"fresh(n=4) from block:    {got}")
    print("PIN SELF-TEST PASS")
