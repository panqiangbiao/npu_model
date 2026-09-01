#!/usr/bin/env python3
"""Summarize HarmonyOS backend comparison hilog and text hitrace captures."""

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TRACE_RE = re.compile(
    r"^\s*(?P<comm>.+)-(?P<pid>\d+)\s+\(\s*(?P<tgid>\d+)\)\s+"
    r"\[(?P<cpu>\d+)\].*?\s(?P<ts>\d+\.\d+):\s+"
    r"(?P<event>[A-Za-z0-9_]+):\s*(?P<details>.*)$"
)
SWITCH_RE = re.compile(r"prev_pid=(\d+).*?next_pid=(\d+)")
WAKE_RE = re.compile(r"pid=(\d+)")
CPU_FREQ_RE = re.compile(r"state=(\d+)\s+cpu_id=(\d+)")
CLOCK_RE = re.compile(r"(?P<name>\S+)\s+state=(?P<state>\d+)")
RSS_RE = re.compile(r"member=(?P<member>\d+)\s+size=(?P<size>\d+)B")
ASYNC_TRACE_RE = re.compile(
    r"(?P<phase>[SF])\|(?P<pid>\d+)\|H:(?P<name>HBAI_TRACE/[^|]+)\|(?P<id>\d+)"
)
SYNC_TRACE_BEGIN_RE = re.compile(r"B\|(?P<pid>\d+)\|H:(?P<name>[^|]+)")
LOG_PATTERNS = {
    "face_ms": re.compile(r"\b(?:NPU|CPU|GPU) live frame \d+: (\d+)ms"),
    "landmarks_ms": re.compile(r"\b(?:NPU|CPU|GPU) landmarks frame \d+: (\d+)ms"),
    "total_ms": re.compile(r"\b(?:NPU|CPU|GPU) total frame \d+: (\d+)ms"),
}


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": max(values),
    }


def parse_hilog(path):
    values = {name: [] for name in LOG_PATTERNS}
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            for name, pattern in LOG_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    values[name].append(int(match.group(1)))
    return {name: stats(samples) for name, samples in values.items()}


def add_residency(store, key, timestamp, state):
    previous = store.get(key)
    if previous is None:
        store[key] = [timestamp, state, defaultdict(float)]
        return
    last_time, last_state, durations = previous
    durations[last_state] += max(0.0, timestamp - last_time)
    previous[0] = timestamp
    previous[1] = state


def finish_residency(store, end_time):
    result = {}
    for key, (last_time, last_state, durations) in store.items():
        durations[last_state] += max(0.0, end_time - last_time)
        covered = sum(durations.values())
        weighted = sum(state * duration for state, duration in durations.items())
        result[str(key)] = {
            "average": round(weighted / covered, 1) if covered else None,
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "coverage_s": round(covered, 3),
            "residency_s": {
                str(state): round(duration, 3)
                for state, duration in sorted(durations.items())
            },
        }
    return result


def parse_trace(path, app_pid):
    pid_to_tgid = {}
    runtime_by_pid = defaultdict(float)
    running = {}
    schedule_ins = Counter()
    wakeups = Counter()
    migrations = Counter()
    last_cpu = {}
    event_counts = Counter()
    app_event_counts = Counter()
    app_rss_sizes = defaultdict(list)
    cpu_freq = {}
    clocks = {}
    npu_completion_irqs = 0
    async_trace_starts = {}
    trace_span_ms = defaultdict(list)
    trace_span_by_key = {}
    mnn_active_inference = {}
    trace_start = None
    trace_end = None

    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            match = TRACE_RE.match(line)
            if not match:
                continue
            pid = int(match.group("pid"))
            tgid = int(match.group("tgid"))
            cpu = int(match.group("cpu"))
            timestamp = float(match.group("ts"))
            event = match.group("event")
            details = match.group("details")
            pid_to_tgid[pid] = tgid
            trace_start = timestamp if trace_start is None else min(trace_start, timestamp)
            trace_end = timestamp if trace_end is None else max(trace_end, timestamp)
            event_counts[event] += 1
            if event == "tracing_mark_write":
                async_trace = ASYNC_TRACE_RE.search(details)
                if async_trace:
                    key = (
                        int(async_trace.group("pid")),
                        async_trace.group("name"),
                        int(async_trace.group("id")),
                    )
                    if async_trace.group("phase") == "S":
                        async_trace_starts[key] = timestamp
                        if key[1] in (
                            "HBAI_TRACE/Face/Infer/GPU",
                            "HBAI_TRACE/Landmarks/Infer/GPU",
                            "HBAI_TRACE/Face/Infer/MNN-NPU",
                            "HBAI_TRACE/Landmarks/Infer/MNN-NPU",
                        ):
                            mnn_active_inference[key[0]] = {
                                "name": key[1],
                                "trace_id": key[2],
                                "started_at": timestamp,
                                "backend": "MNNNPU" if "/MNN-NPU" in key[1] else "GPU",
                            }
                    elif key in async_trace_starts:
                        duration_ms = (timestamp - async_trace_starts.pop(key)) * 1000
                        trace_span_ms[key[1]].append(duration_ms)
                        trace_span_by_key[key] = duration_ms
                        active = mnn_active_inference.get(key[0])
                        if active and active["name"] == key[1] and active["trace_id"] == key[2]:
                            label = "Face" if "/Face/" in key[1] else "Landmarks"
                            backend = active["backend"]
                            upload_at = active.get("upload_at")
                            run_at = active.get("run_at")
                            download_at = active.get("download_at")
                            if upload_at is not None and run_at is not None:
                                trace_span_ms[f"HBAI_TRACE/{backend}/{label}/UploadDerived"].append(
                                    (run_at - upload_at) * 1000
                                )
                            if run_at is not None and download_at is not None:
                                trace_span_ms[f"HBAI_TRACE/{backend}/{label}/RunDerived"].append(
                                    (download_at - run_at) * 1000
                                )
                            if download_at is not None:
                                trace_span_ms[f"HBAI_TRACE/{backend}/{label}/DownloadDerived"].append(
                                    (timestamp - download_at) * 1000
                                )
                            mnn_active_inference.pop(key[0], None)
                else:
                    sync_begin = SYNC_TRACE_BEGIN_RE.search(details)
                    if sync_begin:
                        marker_pid = int(sync_begin.group("pid"))
                        marker_name = sync_begin.group("name")
                        active = mnn_active_inference.get(marker_pid)
                        if active:
                            prefix = f"HBAI_TRACE/{active['backend']}/"
                            if marker_name == prefix + "TensorUpload":
                                active["upload_at"] = timestamp
                            elif marker_name == prefix + "RunSession":
                                active["run_at"] = timestamp
                            elif marker_name == prefix + "TensorDownload":
                                active["download_at"] = timestamp
            if tgid == app_pid:
                app_event_counts[event] += 1
                if event == "rss_stat":
                    rss = RSS_RE.search(details)
                    if rss:
                        app_rss_sizes[int(rss.group("member"))].append(int(rss.group("size")))

            if event == "sched_switch":
                switch = SWITCH_RE.search(details)
                if switch:
                    previous_pid = int(switch.group(1))
                    next_pid = int(switch.group(2))
                    current = running.get(cpu)
                    if current is not None:
                        current_pid, current_since = current
                        runtime_by_pid[current_pid] += max(0.0, timestamp - current_since)
                    running[cpu] = (next_pid, timestamp)
                    schedule_ins[next_pid] += 1
                    if next_pid in last_cpu and last_cpu[next_pid] != cpu:
                        migrations[next_pid] += 1
                    last_cpu[next_pid] = cpu
                    pid_to_tgid.setdefault(previous_pid, tgid)
            elif event in ("sched_wakeup", "sched_waking"):
                wake = WAKE_RE.search(details)
                if wake:
                    wakeups[int(wake.group(1))] += 1
            elif event == "cpu_frequency":
                frequency = CPU_FREQ_RE.search(details)
                if frequency:
                    add_residency(cpu_freq, int(frequency.group(2)), timestamp, int(frequency.group(1)))
            elif event == "clock_set_rate":
                clock = CLOCK_RE.search(details)
                if clock:
                    add_residency(clocks, clock.group("name"), timestamp, int(clock.group("state")))
            elif event == "irq_handler_entry" and "name=npu_cq_report_handler" in details:
                npu_completion_irqs += 1

    if trace_end is None or trace_start is None:
        raise RuntimeError(f"No trace events parsed from {path}")
    for current_pid, current_since in running.values():
        runtime_by_pid[current_pid] += max(0.0, trace_end - current_since)

    app_pids = {pid for pid, tgid in pid_to_tgid.items() if tgid == app_pid}
    app_pids.add(app_pid)
    app_runtime = sum(runtime_by_pid[pid] for pid in app_pids)
    system_busy = sum(duration for pid, duration in runtime_by_pid.items() if pid != 0)
    duration = trace_end - trace_start
    cpu_count = max(running.keys(), default=-1) + 1
    capacity = duration * cpu_count
    clock_summary = finish_residency(clocks, trace_end)
    selected_clocks = {
        name: value for name, value in clock_summary.items()
        if name.startswith("ddr_cluster") or name in ("gpufreq", "gpuload", "l3c_cluster2_freq")
    }
    complete_frames = {}
    for backend in ("NPU", "MNN-NPU"):
        frame_name = f"HBAI_TRACE/Frame/Total/{backend}"
        landmark_name = f"HBAI_TRACE/Landmarks/Infer/{backend}"
        landmark_ids = {
            (pid, trace_id)
            for (pid, name, trace_id) in trace_span_by_key
            if name == landmark_name
        }
        values = [
            duration_ms
            for (pid, name, trace_id), duration_ms in trace_span_by_key.items()
            if name == frame_name and (pid, trace_id) in landmark_ids
        ]
        if values:
            complete_frames[backend] = stats(values)
    return {
        "duration_s": round(duration, 3),
        "cpu_count": cpu_count,
        "app_cpu_time_s": round(app_runtime, 3),
        "app_average_cpu_percent": round(app_runtime / duration * 100, 2),
        "app_capacity_percent": round(app_runtime / capacity * 100, 3) if capacity else None,
        "main_thread_cpu_time_s": round(runtime_by_pid[app_pid], 3),
        "system_busy_cpu_time_s": round(system_busy, 3),
        "system_capacity_percent": round(system_busy / capacity * 100, 3) if capacity else None,
        "app_schedule_ins": sum(schedule_ins[pid] for pid in app_pids),
        "app_wakeups": sum(wakeups[pid] for pid in app_pids),
        "app_migrations": sum(migrations[pid] for pid in app_pids),
        "app_rss_stat_events": app_event_counts["rss_stat"],
        "app_rss_stat_events_per_s": round(app_event_counts["rss_stat"] / duration, 2),
        "app_rss_member_bytes": {
            str(member): {
                "min": min(sizes),
                "max": max(sizes),
                "range": max(sizes) - min(sizes),
            }
            for member, sizes in sorted(app_rss_sizes.items())
        },
        "npu_completion_irqs": npu_completion_irqs,
        "npu_completion_irqs_per_s": round(npu_completion_irqs / duration, 2),
        "cpu_frequency_khz": finish_residency(cpu_freq, trace_end),
        "selected_clocks": selected_clocks,
        "trace_spans_ms": {
            name: stats(values) for name, values in sorted(trace_span_ms.items())
        },
        "complete_frame_ms": complete_frames,
        "top_events": event_counts.most_common(20),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--hilog", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "trace": parse_trace(args.trace, args.pid),
        "hilog": parse_hilog(args.hilog),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
