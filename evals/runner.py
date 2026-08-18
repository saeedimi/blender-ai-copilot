
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import (
    dump_json,
    load_json,
    new_run_variables,
    normalize_trace,
    percentile,
    render_template,
)
from .reporting import write_report
from .scoring import score_turn


def post_json(url, payload, timeout=1000):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class ConversationState:
    def __init__(self, recent_turns=3):
        self.recent_turns = max(1, min(int(recent_turns), 6))
        self.recent_messages = []
        self.memory_summary = ""
        self.structured_memory = {}

    def context(self):
        return {
            "recent_messages": self.recent_messages[-(self.recent_turns * 2):],
            "memory_summary": self.memory_summary,
            "structured_memory": self.structured_memory,
        }

    def update(self, prompt, response):
        answer = str(response.get("answer", "") or "").strip()
        self.recent_messages.append(
            {"role": "user", "content": prompt}
        )
        if answer:
            self.recent_messages.append(
                {"role": "assistant", "content": answer}
            )

        if isinstance(response.get("memory_summary"), str):
            self.memory_summary = response["memory_summary"]

        if isinstance(response.get("structured_memory"), dict):
            self.structured_memory = response["structured_memory"]


def finish_approvals(backend, response, auto_approve, timeout):
    current = response
    approval_ids = []

    while current.get("status") == "approval_required":
        approval_id = current.get("approval_id")
        if not approval_id:
            raise RuntimeError("approval_required response had no approval_id.")

        approval_ids.append(approval_id)

        if not auto_approve:
            raise RuntimeError(
                "Suite hit a high-risk tool. Re-run with --auto-approve "
                "only in a disposable/safe Blender scene."
            )

        current = post_json(
            backend.rstrip("/") + "/approve",
            {
                "approval_id": approval_id,
                "approved": True,
            },
            timeout=timeout,
        )

    return current, approval_ids


def run_suite(
    suite_path,
    backend,
    output_root,
    auto_approve=False,
    recent_turns=3,
    timeout=1000,
    dry_run=False,
):
    raw_suite = load_json(suite_path)
    variables = new_run_variables()
    variables.update(
        {
            "ox_plus_6": variables["ox"] + 6.0,
            "oy_minus_3": variables["oy"] - 3.0,
            "oy_minus_8": variables["oy"] - 8.0,
        }
    )
    suite = render_template(raw_suite, variables)

    output_dir = Path(output_root) / variables["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_json(output_dir / "rendered_suite.json", suite)
    dump_json(output_dir / "run_variables.json", variables)

    if dry_run:
        print(json.dumps(suite, indent=2, ensure_ascii=False))
        print()
        print("Dry run only; no backend calls were made.")
        return {
            "run_id": variables["run_id"],
            "dry_run": True,
            "output_dir": str(output_dir),
        }

    results = []
    session = ConversationState(recent_turns=recent_turns)

    started = time.perf_counter()

    for case_index, case in enumerate(suite.get("cases", []), start=1):
        case_id = case["id"]
        case_session = session if case.get("continue_global_context", False) else ConversationState(recent_turns=recent_turns)

        if case.get("seed_global_context", False):
            case_session = session

        case_result = {
            "id": case_id,
            "description": case.get("description", ""),
            "turns": [],
        }

        for turn_index, turn in enumerate(case.get("turns", []), start=1):
            prompt = turn["prompt"]
            print(f"[{case_index}] {case_id} / turn {turn_index}")
            print(f"  > {prompt}")

            response = post_json(
                backend.rstrip("/") + "/chat",
                {
                    "message": prompt,
                    "conversation_context": case_session.context(),
                },
                timeout=timeout,
            )

            response, approval_ids = finish_approvals(
                backend,
                response,
                auto_approve=auto_approve,
                timeout=timeout,
            )

            trace = normalize_trace(response)
            score = score_turn(trace, turn.get("expect", {}))

            case_session.update(prompt, response)

            turn_result = {
                "turn_index": turn_index,
                "prompt": prompt,
                "approval_ids": approval_ids,
                "response_status": response.get("status"),
                "answer": response.get("answer"),
                "trace_id": response.get("trace_id"),
                "trace": trace,
                "score": score,
            }
            case_result["turns"].append(turn_result)

            status = "PASS" if score["passed"] else "FAIL"
            print(
                f"    {status} | "
                f"{score['metrics']['total_time'] or 0:.2f}s | "
                f"LLM={score['metrics']['llm_calls']} | "
                f"tools={score['metrics']['tool_steps']}"
            )

        case_result["passed"] = all(
            turn["score"]["passed"]
            for turn in case_result["turns"]
        )
        results.append(case_result)

        if case.get("commit_context_to_global", False):
            session = case_session

    elapsed = time.perf_counter() - started

    summary = summarize_results(results, elapsed)
    payload = {
        "run_id": variables["run_id"],
        "suite": suite.get("name"),
        "backend": backend,
        "variables": variables,
        "results": results,
        "summary": summary,
    }

    dump_json(output_dir / "results.json", payload)
    write_report(output_dir / "report.md", payload)

    print()
    print("=" * 68)
    print(f"Evaluation run: {variables['run_id']}")
    print(f"Turn pass rate: {summary['turn_pass_rate']:.1%}")
    print(f"Case pass rate: {summary['case_pass_rate']:.1%}")
    print(f"Semantic false-success rate: {summary['semantic_false_success_rate']:.1%}")
    print(f"Average latency: {summary['avg_total_time']:.2f}s")
    if summary["p95_total_time"] is not None:
        print(f"P95 latency: {summary['p95_total_time']:.2f}s")
    print(f"Average LLM calls: {summary['avg_llm_calls']:.2f}")
    print(f"Average tool steps: {summary['avg_tool_steps']:.2f}")
    print(f"Duplicate render incidents: {summary['duplicate_render_incidents']}")
    print(f"Report: {output_dir / 'report.md'}")
    print("=" * 68)

    return payload


def summarize_results(results, elapsed):
    turns = [
        turn
        for case in results
        for turn in case.get("turns", [])
    ]

    case_passes = [bool(case.get("passed")) for case in results]
    turn_passes = [bool(turn["score"]["passed"]) for turn in turns]
    false_success = [
        bool(turn["score"]["semantic_false_success"])
        for turn in turns
    ]

    latencies = [
        turn["score"]["metrics"]["total_time"]
        for turn in turns
        if turn["score"]["metrics"]["total_time"] is not None
    ]

    discovery_precisions = [
        turn["score"]["metrics"]["discovery_precision"]
        for turn in turns
        if turn["score"]["metrics"]["discovery_precision"] is not None
    ]
    discovery_recalls = [
        turn["score"]["metrics"]["discovery_recall"]
        for turn in turns
        if turn["score"]["metrics"]["discovery_recall"] is not None
    ]

    duplicate_render_incidents = sum(
        1
        for turn in turns
        if turn["score"]["metrics"]["render_executions"] > 1
    )

    return {
        "case_count": len(results),
        "turn_count": len(turns),
        "case_pass_rate": (
            sum(case_passes) / len(case_passes)
            if case_passes
            else 0.0
        ),
        "turn_pass_rate": (
            sum(turn_passes) / len(turn_passes)
            if turn_passes
            else 0.0
        ),
        "semantic_false_success_rate": (
            sum(false_success) / len(false_success)
            if false_success
            else 0.0
        ),
        "avg_total_time": (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        ),
        "p95_total_time": percentile(latencies, 0.95),
        "avg_llm_calls": (
            sum(turn["score"]["metrics"]["llm_calls"] for turn in turns)
            / len(turns)
            if turns
            else 0.0
        ),
        "avg_tool_steps": (
            sum(turn["score"]["metrics"]["tool_steps"] for turn in turns)
            / len(turns)
            if turns
            else 0.0
        ),
        "avg_prompt_eval_tokens": (
            sum(turn["score"]["metrics"]["prompt_eval_tokens"] for turn in turns)
            / len(turns)
            if turns
            else 0.0
        ),
        "avg_generated_tokens": (
            sum(turn["score"]["metrics"]["generated_tokens"] for turn in turns)
            / len(turns)
            if turns
            else 0.0
        ),
        "discovery_precision": (
            sum(discovery_precisions) / len(discovery_precisions)
            if discovery_precisions
            else None
        ),
        "discovery_recall": (
            sum(discovery_recalls) / len(discovery_recalls)
            if discovery_recalls
            else None
        ),
        "duplicate_render_incidents": duplicate_render_incidents,
        "wall_clock_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Blender AI Copilot evaluation suites."
    )
    parser.add_argument(
        "--suite",
        default="evals/suites/smoke.json",
    )
    parser.add_argument(
        "--backend",
        default="http://127.0.0.1:8765",
    )
    parser.add_argument(
        "--output-root",
        default="eval_reports",
    )
    parser.add_argument(
        "--recent-turns",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Automatically approve high-risk Blender tools. "
            "Use only in a safe/disposable evaluation scene."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    try:
        payload = run_suite(
            suite_path=args.suite,
            backend=args.backend,
            output_root=args.output_root,
            auto_approve=args.auto_approve,
            recent_turns=args.recent_turns,
            timeout=args.timeout,
            dry_run=args.dry_run,
        )
    except urllib.error.URLError as exc:
        print(f"Backend connection failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise

    if not args.dry_run:
        if payload["summary"]["turn_pass_rate"] < 1.0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
