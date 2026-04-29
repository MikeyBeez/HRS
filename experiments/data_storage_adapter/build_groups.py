"""Group existing per-passage Dickens library entries into 4 thematic groups
for the rank × size sweep, plus a 5-entry Part A subset.

Group 1 (Pip childhood)             : library entries 0..9
Group 2 (Domestic / Joe / clerks)   : library entries 10..19
Group 3 (Mid-book characters/places): library entries 20..29
Group 4 (Late-book named entities)  : library entries 30..39

Part A: first 5 of group 1 (Pip's parents, marshes, brothers, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")


def main():
    src = REPO / "experiments/per_passage_dickens/data/library.json"
    library = json.loads(src.read_text())
    assert len(library) >= 40, f"need >=40 library entries, got {len(library)}"

    groups = {
        "G1_pip_childhood": list(range(0, 10)),
        "G2_domestic_joe":  list(range(10, 20)),
        "G3_midbook":       list(range(20, 30)),
        "G4_latebook":      list(range(30, 40)),
    }

    by_id = {e["id"]: e for e in library}
    out = {"groups": {}, "part_a_ids": list(range(0, 5))}
    for gname, ids in groups.items():
        out["groups"][gname] = [by_id[i] for i in ids]

    dst = REPO / "experiments/data_storage_adapter/data/groups.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"Saved {dst}")
    for gname, ids in groups.items():
        print(f"  {gname}: ids {ids[0]}..{ids[-1]}")
        for i in ids[:2]:
            print(f"    [{i}] {by_id[i]['fact']}")
    print(f"\nPart A: {out['part_a_ids']}")
    for i in out["part_a_ids"]:
        print(f"  [{i}] {by_id[i]['fact']}")


if __name__ == "__main__":
    main()
