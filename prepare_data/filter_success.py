import argparse
import json
import os
from typing import List, Dict, Any, Iterable

def filter_success(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter for successful records and strip validation fields."""
    result: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("validation_status") == "success":
            cleaned = {k: v for k, v in rec.items() if not k.startswith("validation_")}
            result.append(cleaned)
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter successful Spring validation records.")
    default_input = os.path.join(os.path.dirname(__file__), "..", "data_filtered", "codes", "codes_spring_mini.json")
    default_output = os.path.join(os.path.dirname(__file__), "..", "data_filtered", "codes", "codes_spring_success_mini.json")

    parser.add_argument("--input", "-i", default=default_input, help="Path to input JSON file.")
    parser.add_argument("--output", "-o", default=default_output, help="Path to output JSON file.")
    args = parser.parse_args()

    print(f"Loading records from {args.input}…")
    records = json.load(open(args.input))
    print("Filtering successful records…")
    success_records = filter_success(records)
    print(f"Found {len(success_records)} successful records. Writing to {args.output}…")
    json.dump(success_records, open(args.output, "w"), ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
