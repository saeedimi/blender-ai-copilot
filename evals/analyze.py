
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (
    mutation_execution_counts,
    percentile,
)


def analyze_trace_dir(trace_dir):
    trace_dir = Path(trace_dir)
    traces = []

    for path in sorted(trace_dir.glob("*.json")):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("trace_id"):
            traces.append(payload)

    total = len(traces)
    successes = [bool(trace.get("success")) for trace in traces]
    verified = [
        bool(trace.get("verification_complete"))
        for trace in traces
    ]
    latencies = [
        float(trace["total_time"])
        for trace in traces
        if trace.get("total_time") is not None
    ]

    duplicate_renders = 0
    no_replay_events = 0
    unresolved_references = 0

    for trace in traces:
        counts = mutation_execution_counts(trace)
        if counts.get("render_scene", 0) > 1:
            duplicate_renders += 1

        for event in trace.get("live_events", []):
            if event.get("event") == "side_effect_executed_verification_failed_no_replay":
                no_replay_events += 1
            elif event.get("event") == "reference_unresolved":
                unresolved_references += 1

    summary = {
        "trace_count": total,
        "success_rate": (
            sum(successes) / total if total else 0.0
        ),
        "verification_complete_rate": (
            sum(verified) / total if total else 0.0
        ),
        "avg_total_time": (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        ),
        "p95_total_time": percentile(latencies, 0.95),
        "duplicate_render_incidents": duplicate_renders,
        "no_replay_guard_events": no_replay_events,
        "unresolved_reference_events": unresolved_references,
    }

    return {
        "summary": summary,
        "traces": traces,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze existing Blender Copilot trace JSON files."
    )
    parser.add_argument(
        "--trace-dir",
        default="traces",
    )
    args = parser.parse_args()

    result = analyze_trace_dir(args.trace_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
