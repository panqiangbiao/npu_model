#!/usr/bin/env python3
"""Summarize fixed-input backend spans and MNN USER_0 internal stages."""

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
SYNC_BEGIN_RE = re.compile(r"B\|(?P<pid>\d+)\|H:(?P<name>[^|]+)")
SYNC_END_RE = re.compile(r"E\|(?P<pid>\d+)")


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


def parse_trace(path):
    async_starts = {}
    sync_stacks = defaultdict(list)
    spans = defaultdict(list)

    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            match = TRACE_RE.match(line)
            if not match or match.group("event") != "tracing_mark_write":
                continue
            tid = int(match.group("tid"))
            timestamp = float(match.group("ts"))
            details = match.group("details")

            async_match = ASYNC_RE.search(details)
            if async_match:
                key = (
                    int(async_match.group("pid")),
                    async_match.group("name"),
                    int(async_match.group("id")),
                )
                if async_match.group("phase") == "S":
                    async_starts[key] = timestamp
                elif key in async_starts:
                    spans[key[1]].append((timestamp - async_starts.pop(key)) * 1000)
                continue

            begin_match = SYNC_BEGIN_RE.search(details)
            if begin_match:
                sync_stacks[tid].append((begin_match.group("name"), timestamp))
                continue

            if SYNC_END_RE.search(details) and sync_stacks[tid]:
                name, started_at = sync_stacks[tid].pop()
                if name.startswith("HBAI_TRACE/"):
                    duration_ms = (timestamp - started_at) * 1000
                    spans[name].append(duration_ms)
                    if name.startswith("HBAI_TRACE/MNNNPU/User0/"):
                        ancestors = [item[0] for item in sync_stacks[tid]]
                        if "HBAI_TRACE/MNNNPU/User0/Face/Total" in ancestors:
                            spans[name.replace("User0/", "User0/Face/")].append(duration_ms)
                        elif "HBAI_TRACE/MNNNPU/User0/Landmarks/Total" in ancestors:
                            spans[name.replace("User0/", "User0/Landmarks/")].append(duration_ms)

    wanted = {
        name: summarize(values)
        for name, values in sorted(spans.items())
        if name.startswith("HBAI_TRACE/Static/")
        or name.startswith("HBAI_TRACE/MNNNPU/User0/")
    }
    return wanted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_trace(args.trace)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
