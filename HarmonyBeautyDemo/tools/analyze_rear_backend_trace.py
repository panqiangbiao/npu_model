#!/usr/bin/env python3
"""Summarize complete rear-camera inference frames from Harmony text hitrace."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


TRACE_RE = re.compile(
    r"^\s*(?P<comm>.+)-(?P<tid>\d+)\s+\(\s*(?P<tgid>\d+)\)\s+"
    r"\[(?P<cpu>\d+)\].*?\s(?P<ts>\d+\.\d+):\s+"
    r"(?P<event>[A-Za-z0-9_]+):\s*(?P<details>.*)$"
)
ASYNC_RE = re.compile(
    r"(?P<phase>[SF])\|(?P<pid>\d+)\|H:(?P<name>HBAI_TRACE/[^|]+)\|(?P<id>\d+)"
)
COUNTER_RE = re.compile(
    r"C\|(?P<pid>\d+)\|H:(?P<name>HBAI_TRACE/[^|]+)\|(?P<value>-?\d+)"
)


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p90_ms": round(percentile(values, 0.90), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def parse(path, backend):
    starts = {}
    spans = {}
    counters = defaultdict(list)
    trace_start = None
    trace_end = None

    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            match = TRACE_RE.match(line)
            if not match:
                continue
            timestamp = float(match.group("ts"))
            trace_start = timestamp if trace_start is None else min(trace_start, timestamp)
            trace_end = timestamp if trace_end is None else max(trace_end, timestamp)
            if match.group("event") != "tracing_mark_write":
                continue
            details = match.group("details")
            async_match = ASYNC_RE.search(details)
            if async_match:
                key = (
                    int(async_match.group("pid")),
                    async_match.group("name"),
                    int(async_match.group("id")),
                )
                if async_match.group("phase") == "S":
                    starts[key] = timestamp
                elif key in starts:
                    spans[(key[1], key[2])] = (timestamp - starts.pop(key)) * 1000
                continue
            counter_match = COUNTER_RE.search(details)
            if counter_match:
                counters[counter_match.group("name")].append(int(counter_match.group("value")))

    frame_name = f"HBAI_TRACE/Frame/Total/{backend}"
    face_name = f"HBAI_TRACE/Face/Infer/{backend}"
    landmark_name = f"HBAI_TRACE/Landmarks/Infer/{backend}"
    frame_ids = {trace_id for name, trace_id in spans if name == frame_name}
    face_ids = {trace_id for name, trace_id in spans if name == face_name}
    landmark_ids = {trace_id for name, trace_id in spans if name == landmark_name}
    complete_ids = sorted(frame_ids & face_ids & landmark_ids)

    stage_names = {
        "total": frame_name,
        "face_infer": face_name,
        "face_decode": "HBAI_TRACE/Face/DecodeSsd",
        "landmark_preprocess": "HBAI_TRACE/Landmarks/Preprocess",
        "landmark_infer": landmark_name,
        "landmark_decode": "HBAI_TRACE/Landmarks/Decode",
        "set_face_region": "HBAI_TRACE/Output/SetFaceRegion",
        "set_landmarks": "HBAI_TRACE/Output/SetLandmarks",
    }
    stages = {
        label: summarize([spans[(name, trace_id)] for trace_id in complete_ids if (name, trace_id) in spans])
        for label, name in stage_names.items()
    }

    confidence = counters.get("HBAI_TRACE/State/FaceConfidencePermille", [])
    landmark_counts = counters.get("HBAI_TRACE/State/LandmarkCount", [])
    duration = (trace_end - trace_start) if trace_start is not None and trace_end is not None else 0
    return {
        "backend": backend,
        "trace_duration_s": round(duration, 3),
        "all_frame_spans": len(frame_ids),
        "complete_frames": len(complete_ids),
        "complete_fps": round(len(complete_ids) / duration, 3) if duration else None,
        "stages": stages,
        "confidence_permille": {
            "count": len(confidence),
            "min": min(confidence) if confidence else None,
            "mean": round(statistics.mean(confidence), 3) if confidence else None,
            "max": max(confidence) if confidence else None,
        },
        "landmark_count_values": sorted(set(landmark_counts)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse(args.trace, args.backend)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
