"""Export the identifier-only view of the frozen core-study item list.

The archive statement of the thesis says that the four source datasets are not
redistributed and that the item lists refer to them by identifier. The working
list `data/core_prompt/items.jsonl` does embed the item text (it has to: every
prompt is built from it, and the scorer compares against the references), so
this script derives the redistribution-safe view of the same list:

    dataset, id, answerable, sentinel, position

`position` is the 0-based index in items.jsonl, i.e. the frozen order produced
by build_items.py with SEED = 20260901. The full list can be rebuilt from the
identifiers alone with `python build_items.py`, which re-samples with the same
seed and therefore reproduces this order exactly.

Usage: python export_item_ids.py [--items items.jsonl] [--out items.ids.jsonl]
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "core_prompt"
FIELDS = ("dataset", "id", "answerable", "sentinel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DATA / "items.jsonl"))
    ap.add_argument("--out", default=str(DATA / "items.ids.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for pos, r in enumerate(rows):
            f.write(json.dumps({**{k: r.get(k, False) for k in FIELDS}, "position": pos}) + "\n")
    n_sent = sum(1 for r in rows if r.get("sentinel"))
    print(f"wrote {a.out}: {len(rows)} items, {n_sent} sentinel")


if __name__ == "__main__":
    main()
