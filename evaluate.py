"""Measure how accurate the extractor is on your own labelled samples.

Put invoices in samples/ (e.g. bill1.pdf) and a matching answer file (bill1.json)
with the correct values, in the same shape as the output. Then run:

    python evaluate.py

The accuracy numbers it prints are the ones you can honestly put on your resume.
"""
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from extractor import ALLOWED_EXTENSIONS, process, to_number  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
TEXT_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "currency"]
NUMBER_FIELDS = ["subtotal", "tax", "total"]


def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def main():
    files = [p for p in sorted(SAMPLES.iterdir())
             if p.suffix.lower() in ALLOWED_EXTENSIONS and p.with_suffix(".json").exists()]
    if not files:
        print("No labelled samples found. Add file + .json pairs to samples/.")
        sys.exit(1)

    correct, total = {}, {}
    for path in files:
        truth = json.loads(path.with_suffix(".json").read_text())
        try:
            got = process(str(path))["fields"]
        except Exception as exc:
            print(f"{path.name}: FAILED ({exc})")
            got = {}

        for field in TEXT_FIELDS:
            if field not in truth:
                continue
            total[field] = total.get(field, 0) + 1
            if norm(got.get(field)) == norm(truth[field]):
                correct[field] = correct.get(field, 0) + 1

        for field in NUMBER_FIELDS:
            if field not in truth:
                continue
            total[field] = total.get(field, 0) + 1
            a, b = to_number(got.get(field)), to_number(truth[field])
            if a is not None and b is not None and abs(a - b) < 0.01:
                correct[field] = correct.get(field, 0) + 1

    print(f"\nEvaluated {len(files)} documents\n")
    for field in TEXT_FIELDS + NUMBER_FIELDS:
        if field in total:
            pct = 100 * correct.get(field, 0) / total[field]
            print(f"{field:16} {correct.get(field, 0)}/{total[field]}  {pct:5.1f}%")
    all_ok, all_n = sum(correct.values()), sum(total.values())
    print(f"\nOverall field accuracy: {all_ok}/{all_n} = {100 * all_ok / all_n:.1f}%")


if __name__ == "__main__":
    main()
