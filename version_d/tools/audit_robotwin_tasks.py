#!/usr/bin/env python3
"""Create a traceable, keyword-based audit of RoboTwin task prompts."""
import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_KEYWORDS = ["dual", "three", "stapler", "hammer", "cabinet", "switch", "stamp"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    args = parser.parse_args()

    keywords = [item.lower() for item in args.keywords]
    rows = []
    fields = Counter()
    with Path(args.tasks).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            fields.update(record.keys())
            text = " ".join(
                str(record.get(key, ""))
                for key in ("task", "task_name", "instruction", "language_instruction", "prompt")
            ).strip()
            text_lower = text.lower()
            hits = [keyword for keyword in keywords if keyword in text_lower]
            rows.append({
                "line": line_number,
                "task": record.get("task") or record.get("task_name") or "",
                "instruction": record.get("instruction") or record.get("language_instruction") or record.get("prompt") or "",
                "hard_keyword_hits": hits,
                "is_keyword_hard": bool(hits),
            })

    unique_tasks = sorted({str(row["task"]) for row in rows if row["task"]})
    output = {
        "source": str(Path(args.tasks).resolve()),
        "keywords": keywords,
        "records": len(rows),
        "unique_tasks": unique_tasks,
        "unique_task_count": len(unique_tasks),
        "keyword_hard_records": sum(row["is_keyword_hard"] for row in rows),
        "keyword_hard_tasks": sorted({row["task"] for row in rows if row["is_keyword_hard"] and row["task"]}),
        "observed_fields": sorted(fields),
        "rows": rows,
        "note": "This is a sampling audit only. It does not define official Hard evaluation metrics.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("records", "unique_task_count", "keyword_hard_records", "keyword_hard_tasks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
