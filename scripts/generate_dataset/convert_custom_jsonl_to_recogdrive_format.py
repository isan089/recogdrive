#!/usr/bin/env python3
"""
Convert a custom JSONL dataset into a ReCogDrive-friendly training JSONL.

The input format is expected to look like the provided sample.jsonl, e.g.:
{
  "id": "...",
  "image": ["obs://.../frame.parquet"],
  "driving command": "lane change right",
  "velocity": "(11.39,0.00)",
  "acceleration": "(0.10,0.05)",
  "history_trajectory": "-16.17,0.18,-0.02,...",
  "future_traj": "6.16,0.04,0.02,..."
}

The output format is:
{
  "id": "...",
  "image_path": "obs://.../frame.parquet",
  "route_id": 0,
  "route_name": "lane change right",
  "history_trajectory": [[x, y, heading], ... 4 items],
  "velocity": [vx, vy],
  "acceleration": [ax, ay],
  "future_trajectory": [[x, y, heading], ... 8 items]
}

By default, the input image path is preserved exactly as-is.
Floating-point values are rounded to 2 decimals by default.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a custom JSONL dataset into ReCogDrive-friendly JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--route-vocab-in",
        type=Path,
        default=None,
        help=(
            "Optional JSON file that defines a stable route order. "
            "Expected format: {'id_to_route': [...]} or {'route_to_id': {...}}."
        ),
    )
    parser.add_argument(
        "--route-vocab-out",
        type=Path,
        default=None,
        help="Optional JSON path to save the discovered route vocabulary.",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=4,
        help="Number of history trajectory points. Default: 4",
    )
    parser.add_argument(
        "--future-points",
        type=int,
        default=8,
        help="Number of future trajectory points. Default: 8",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        default=2,
        help="Number of decimal places for floating-point values. Default: 2",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on malformed records instead of skipping them.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc


def find_first_key(record: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        if key in record:
            return key
    raise KeyError(f"Missing required field. Tried keys: {keys}")


def extract_float_list(value: object) -> List[float]:
    if isinstance(value, list):
        floats: List[float] = []
        for item in value:
            if isinstance(item, (int, float)):
                floats.append(float(item))
            else:
                floats.extend(extract_float_list(item))
        return floats

    if isinstance(value, (int, float)):
        return [float(value)]

    if not isinstance(value, str):
        raise TypeError(f"Unsupported numeric field type: {type(value)!r}")

    matches = FLOAT_PATTERN.findall(value)
    if not matches:
        raise ValueError(f"Could not parse any numeric value from: {value!r}")
    return [float(item) for item in matches]


def round_float(value: float, decimal_places: int) -> float:
    rounded = round(float(value), decimal_places)
    if rounded == 0:
        return 0.0
    return rounded


def reshape_triplets(
    values: Sequence[float],
    expected_points: int,
    field_name: str,
    decimal_places: int,
) -> List[List[float]]:
    expected_values = expected_points * 3
    if len(values) != expected_values:
        raise ValueError(
            f"{field_name} expects {expected_values} floats "
            f"({expected_points} points x 3), but got {len(values)}"
        )
    return [
        [
            round_float(values[idx], decimal_places),
            round_float(values[idx + 1], decimal_places),
            round_float(values[idx + 2], decimal_places),
        ]
        for idx in range(0, expected_values, 3)
    ]


def reshape_vector(
    values: Sequence[float],
    expected_dims: int,
    field_name: str,
    decimal_places: int,
) -> List[float]:
    if len(values) != expected_dims:
        raise ValueError(f"{field_name} expects {expected_dims} floats, but got {len(values)}")
    return [round_float(item, decimal_places) for item in values]


def extract_image_source(record: Dict) -> str:
    key = find_first_key(record, ("image", "images", "image_path", "image_paths"))
    value = record[key]

    if isinstance(value, list):
        if not value:
            raise ValueError("The image field is an empty list.")
        first_image = value[0]
    else:
        first_image = value

    if not isinstance(first_image, str):
        raise TypeError(f"Expected image path/uri to be a string, got {type(first_image)!r}")
    return first_image


def normalize_route_name(route_name: str) -> str:
    route = route_name.strip()
    if not route:
        raise ValueError("Route name is empty after stripping whitespace.")
    return route


def discover_route_vocab(records: Iterable[Tuple[int, Dict]]) -> Dict[str, int]:
    route_names = set()
    for _, record in records:
        key = find_first_key(record, ("driving command", "driving_command", "route", "route_name"))
        route_names.add(normalize_route_name(str(record[key])))

    ordered_routes = sorted(route_names)
    return {route_name: idx for idx, route_name in enumerate(ordered_routes)}


def load_route_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if "route_to_id" in payload:
        route_to_id = payload["route_to_id"]
        return {str(key): int(value) for key, value in route_to_id.items()}

    if "id_to_route" in payload:
        id_to_route = payload["id_to_route"]
        return {str(route_name): idx for idx, route_name in enumerate(id_to_route)}

    raise ValueError(
        f"Unsupported route vocab format in {path}. "
        "Expected {'route_to_id': {...}} or {'id_to_route': [...]}."
    )


def write_route_vocab(path: Path, route_vocab: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    id_to_route = [route for route, _ in sorted(route_vocab.items(), key=lambda item: item[1])]
    payload = {
        "route_to_id": route_vocab,
        "id_to_route": id_to_route,
    }
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def convert_record(
    record: Dict,
    route_vocab: Dict[str, int],
    history_points: int,
    future_points: int,
    decimal_places: int,
) -> Dict:
    record_id_key = find_first_key(record, ("id", "sample_id", "token"))
    route_key = find_first_key(record, ("driving command", "driving_command", "route", "route_name"))
    velocity_key = find_first_key(record, ("velocity", "ego_velocity"))
    acceleration_key = find_first_key(record, ("acceleration", "ego_acceleration"))
    history_key = find_first_key(record, ("history_trajectory", "history_traj"))
    future_key = find_first_key(record, ("future_traj", "future_trajectory", "trajectory"))

    route_name = normalize_route_name(str(record[route_key]))
    if route_name not in route_vocab:
        raise KeyError(f"Route {route_name!r} is not in the provided route vocabulary.")

    image_source = extract_image_source(record)

    history_values = extract_float_list(record[history_key])
    future_values = extract_float_list(record[future_key])
    velocity_values = extract_float_list(record[velocity_key])
    acceleration_values = extract_float_list(record[acceleration_key])

    converted = {
        "id": str(record[record_id_key]),
        "image_path": image_source,
        "route_id": route_vocab[route_name],
        "route_name": route_name,
        "history_trajectory": reshape_triplets(
            history_values, history_points, "history_trajectory", decimal_places
        ),
        "velocity": reshape_vector(velocity_values, 2, "velocity", decimal_places),
        "acceleration": reshape_vector(acceleration_values, 2, "acceleration", decimal_places),
        "future_trajectory": reshape_triplets(
            future_values, future_points, "future_trajectory", decimal_places
        ),
    }

    if "datasource" in record:
        converted["datasource"] = record["datasource"]

    return converted


def dump_json_with_fixed_floats(value: object, decimal_places: int) -> str:
    if value is None or isinstance(value, bool):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float is not supported in JSON output: {value}")
        formatted = f"{value:.{decimal_places}f}"
        if float(formatted) == 0.0 and formatted.startswith("-"):
            formatted = formatted[1:]
        return formatted

    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, list):
        return "[" + ", ".join(dump_json_with_fixed_floats(item, decimal_places) for item in value) + "]"

    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            encoded_key = json.dumps(str(key), ensure_ascii=False)
            encoded_value = dump_json_with_fixed_floats(item, decimal_places)
            parts.append(f"{encoded_key}: {encoded_value}")
        return "{" + ", ".join(parts) + "}"

    raise TypeError(f"Unsupported value type for JSON serialization: {type(value)!r}")


def main() -> None:
    args = parse_args()

    if args.route_vocab_in is not None:
        route_vocab = load_route_vocab(args.route_vocab_in)
    else:
        route_vocab = discover_route_vocab(iter_jsonl(args.input))

    if args.route_vocab_out is not None:
        write_route_vocab(args.route_vocab_out, route_vocab)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    with args.output.open("w", encoding="utf-8") as out_fp:
        for line_no, record in iter_jsonl(args.input):
            try:
                converted = convert_record(
                    record=record,
                    route_vocab=route_vocab,
                    history_points=args.history_points,
                    future_points=args.future_points,
                    decimal_places=args.decimal_places,
                )
            except Exception as exc:  # noqa: BLE001
                if args.strict:
                    raise ValueError(f"Failed to convert line {line_no}: {exc}") from exc
                skipped_count += 1
                print(f"[skip] line {line_no}: {exc}")
                continue

            out_fp.write(dump_json_with_fixed_floats(converted, args.decimal_places) + "\n")
            converted_count += 1

    print(f"Converted {converted_count} records -> {args.output}")
    if skipped_count:
        print(f"Skipped {skipped_count} malformed records")


if __name__ == "__main__":
    main()
