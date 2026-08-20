#!/usr/bin/env python3
"""Write an evidence-only R0-R3 matrix; missing values remain blank."""
import argparse
import csv
import json
from pathlib import Path


def value(number):
    return "" if number is None else f"{number:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for name in ("R0", "R1", "R2", "R3"):
        summary_path = root / "eval" / name / "summary.json"
        aggregate = {}
        if summary_path.exists():
            aggregate = json.loads(summary_path.read_text(encoding="utf-8")).get("aggregate", {})
        rows.append({
            "Method": name,
            "Clean": value(aggregate.get("clean")),
            "Rand": value(aggregate.get("rand")),
            "Avg": value(aggregate.get("avg")),
            "Hard-Clean": "",
            "Hard-Rand": "",
            "Hard-Avg": "",
        })
    destination = root / "summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destination)


if __name__ == "__main__":
    main()
