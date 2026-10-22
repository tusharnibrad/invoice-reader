"""Measure how accurate the extractor is on your own labelled samples.

Put invoices in samples/ (e.g. bill1.pdf) and a matching answer file (bill1.json).
Then run:   python evaluate.py

- Every mismatch is printed, and the full report is saved to results.txt.
- Results are cached in .eval_cache/, so re-running does not spend API quota on
  files that were already processed (the cache resets when the file or
  extractor.py changes).
- If the daily API quota runs out, it stops early instead of wasting requests.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from extractor import ALLOWED_EXTENSIONS, process, to_number  # noqa: E402

HERE = Path(__file__).parent
SAMPLES = HERE / "samples"
REPORT = HERE / "results.txt"
CACHE = HERE / ".eval_cache"
TEXT_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "currency"]
NUMBER_FIELDS = ["subtotal", "tax", "total"]
RETRIES = 2  # extra tries for network hiccups (never used for quota errors)


def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def cache_file(path):
    digest = hashlib.sha256()
    digest.update(path.name.encode())
    digest.update(path.read_bytes())
    digest.update((HERE / "extractor.py").read_bytes())
    return CACHE / f"{digest.hexdigest()[:24]}.json"


def is_quota_error(message):
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def extract(path):
    """Return (fields, error, from_cache)."""
    cached = cache_file(path)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8")), None, True

    for attempt in range(RETRIES + 1):
        try:
            fields = process(str(path))["fields"]
            CACHE.mkdir(exist_ok=True)
            cached.write_text(json.dumps(fields), encoding="utf-8")
            return fields, None, False
        except Exception as exc:
            message = str(exc)
            if is_quota_error(message) or attempt == RETRIES:
                return {}, message[:300], False
            time.sleep(5)


def main():
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    files = [p for p in sorted(SAMPLES.iterdir())
             if p.suffix.lower() in ALLOWED_EXTENSIONS and p.with_suffix(".json").exists()]
    if not files:
        print("No labelled samples found. Add file + .json pairs to samples/.")
        sys.exit(1)

    correct, total = {}, {}
    done = 0
    for path in files:
        truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        got, error, from_cache = extract(path)
        say(f"Processing {path.name} ..." + (" (cached)" if from_cache else ""))

        if error:
            say(f"  FAILED: {error}")
            if is_quota_error(error):
                say("\nAPI quota is used up. Stopping here. Run again later: finished files are cached.")
                break
            continue  # a failed file is skipped, not counted as wrong

        done += 1
        for field in TEXT_FIELDS:
            if field not in truth:
                continue
            total[field] = total.get(field, 0) + 1
            if norm(got.get(field)) == norm(truth[field]):
                correct[field] = correct.get(field, 0) + 1
            else:
                say(f"  MISMATCH {field}: expected={truth[field]!r} got={got.get(field)!r}")

        for field in NUMBER_FIELDS:
            if field not in truth:
                continue
            total[field] = total.get(field, 0) + 1
            a, b = to_number(got.get(field)), to_number(truth[field])
            if a is not None and b is not None and abs(a - b) < 0.01:
                correct[field] = correct.get(field, 0) + 1
            else:
                say(f"  MISMATCH {field}: expected={truth[field]!r} got={got.get(field)!r}")

    say(f"\nEvaluated {done} of {len(files)} documents\n")
    for field in TEXT_FIELDS + NUMBER_FIELDS:
        if field in total:
            pct = 100 * correct.get(field, 0) / total[field]
            say(f"{field:16} {correct.get(field, 0)}/{total[field]}  {pct:5.1f}%")
    all_ok, all_n = sum(correct.values()), sum(total.values())
    if all_n:
        say(f"\nOverall field accuracy: {all_ok}/{all_n} = {100 * all_ok / all_n:.1f}%")
    if done < len(files):
        say("Note: this number covers only the documents that finished. Do not use it until all are evaluated.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull report saved to {REPORT.name}", flush=True)


if __name__ == "__main__":
    main()
