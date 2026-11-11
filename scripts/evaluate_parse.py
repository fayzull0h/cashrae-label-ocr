"""Per-field accuracy evaluation for label parsing (Phase 2, Stage 2).

Runs OCR + :func:`scripts.label_parse.parse_fields` over the bundled rating
plates and compares the parsed brand, model number and serial number against
``data/samples/ground_truth.json``.

Typical usage::

    python scripts/evaluate_parse.py
    python scripts/evaluate_parse.py --dir data/samples --json
    python scripts/evaluate_parse.py --engine tesseract

Serial numbers are compared prefix-tolerantly: the verified plates sometimes
omit a leading letter (for example ``E140302785`` vs ``140302785``), and that
prefix is deliberately not scored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow direct execution as ``python scripts/evaluate_parse.py``.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.appliance_lookup import normalize_alphanum  # noqa: E402
from scripts.label_ocr import SUPPORTED_OCR_ENGINES, find_images  # noqa: E402
from scripts.label_parse import parse_fields  # noqa: E402

DEFAULT_SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
DEFAULT_GROUND_TRUTH = DEFAULT_SAMPLES_DIR / "ground_truth.json"
FIELDS = ("brand", "model_number", "serial_number")


def _matches(field: dict[str, Any], expected: str, *, serial: bool = False) -> bool:
    actual = field.get("normalized")
    wanted = normalize_alphanum(expected)
    if not actual or not wanted:
        return False
    if serial:
        return actual == wanted or actual.endswith(wanted)
    return actual == wanted


def evaluate_image(
    image_path: Path,
    expected: dict[str, Any],
    backend,
) -> dict[str, Any]:
    """Run OCR + parsing on one image and score it against ``expected``."""
    from scripts.label_ocr import extract_text

    started = time.perf_counter()
    ocr = extract_text(image_path, backend=backend)
    latency = time.perf_counter() - started

    parsed = parse_fields(ocr.get("raw_text", ""))
    fields: dict[str, Any] = {}
    for name in FIELDS:
        wanted = expected.get(name)
        field = parsed[name]
        fields[name] = {
            "parsed": field["value"],
            "expected": wanted,
            "match": _matches(field, wanted or "", serial=(name == "serial_number")),
            "method": field["method"],
            "confidence": field["confidence"],
        }

    return {
        "image": image_path.name,
        "image_path": str(image_path),
        "backend": ocr.get("backend"),
        "ocr_confidence": ocr.get("confidence"),
        "latency_s": round(latency, 3),
        "fields": fields,
        "all_match": all(field["match"] for field in fields.values()),
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    summary: dict[str, Any] = {
        "images": total,
        "fields": {},
        "all_match": {
            "matched": sum(1 for result in results if result["all_match"]),
            "total": total,
        },
    }
    for name in FIELDS:
        matched = sum(1 for result in results if result["fields"][name]["match"])
        summary["fields"][name] = {"matched": matched, "total": total}
    for section in [*summary["fields"].values(), summary["all_match"]]:
        section["accuracy"] = round(section["matched"] / total, 4) if total else 0.0
    return summary


def evaluate_directory(
    directory: str | Path,
    ground_truth_path: str | Path,
    backend,
) -> dict[str, Any]:
    directory = Path(directory)
    ground_truth_path = Path(ground_truth_path)
    truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    missing: list[str] = []
    skipped: list[str] = []

    by_name = {path.name: path for path in find_images(directory)}
    for name, expected in truth.items():
        image_path = by_name.get(name)
        if image_path is None:
            missing.append(name)
            continue
        try:
            results.append(evaluate_image(image_path, expected, backend))
        except Exception as exc:  # noqa: BLE001 - one bad image must not abort
            results.append(
                {
                    "image": name,
                    "image_path": str(image_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "fields": {},
                    "all_match": False,
                }
            )

    for name in sorted(set(by_name) - set(truth)):
        skipped.append(name)

    return {
        "directory": str(directory),
        "ground_truth_file": str(ground_truth_path),
        "backend": getattr(backend, "name", type(backend).__name__),
        "images": results,
        "missing_images": missing,
        "skipped_without_ground_truth": skipped,
        "summary": _summarize([result for result in results if "error" not in result]),
    }


def _print_report(payload: dict[str, Any]) -> None:
    print("=" * 78)
    print("Appliance label field parsing — evaluation")
    print("=" * 78)
    print(f"Directory:     {payload['directory']}")
    print(f"Ground truth:  {payload['ground_truth_file']}")
    print(f"Engine:        {payload['backend']}")
    print()

    header = f"{'Image':32} {'Brand':6} {'Model':6} {'Serial':7} {'OCR':6} {'Time':>7}"
    print(header)
    print("-" * len(header))
    for result in payload["images"]:
        if "error" in result:
            print(f"{result['image']:32} ERROR  {result['error']}")
            continue
        fields = result["fields"]
        marks = {
            name: "OK" if fields[name]["match"] else "--" for name in FIELDS
        }
        confidence = result.get("ocr_confidence")
        confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
        print(
            f"{result['image']:32} {marks['brand']:6} {marks['model_number']:6} "
            f"{marks['serial_number']:7} {confidence_text:6} {result['latency_s']:6.2f}s"
        )

    summary = payload["summary"]
    print()
    print("Per-field accuracy:")
    for name in FIELDS:
        section = summary["fields"][name]
        print(
            f"  {name:14} {section['matched']}/{section['total']} "
            f"({section['accuracy'] * 100:.1f}%)"
        )
    overall = summary["all_match"]
    print(
        f"  {'all three':14} {overall['matched']}/{overall['total']} "
        f"({overall['accuracy'] * 100:.1f}%)"
    )

    if payload["missing_images"]:
        print()
        print("Missing images (ground truth present, file not found):")
        for name in payload["missing_images"]:
            print(f"  - {name}")
    if payload["skipped_without_ground_truth"]:
        print()
        print("Skipped (no ground truth):")
        for name in payload["skipped_without_ground_truth"]:
            print(f"  - {name}")
    print()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Evaluate brand/model/serial parsing on label photos")
    parser.add_argument("--dir", default=str(DEFAULT_SAMPLES_DIR), help="Directory of label images")
    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help="Ground-truth JSON keyed by image filename",
    )
    parser.add_argument(
        "--engine",
        choices=SUPPORTED_OCR_ENGINES,
        default="auto",
        help="OCR engine: auto (Paddle if available, else Tesseract), paddleocr, or tesseract",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text report")
    parser.add_argument("--output", help="Write JSON results to this file")
    args = parser.parse_args(argv)

    try:
        from scripts.ocr_backends import get_backend

        backend = get_backend(args.engine)
        payload = evaluate_directory(args.dir, args.ground_truth, backend)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - OCR init failures get a clear message
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {output_path}")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_report(payload)

    return 0 if payload["summary"]["images"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
