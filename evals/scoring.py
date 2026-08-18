
from __future__ import annotations

from .common import (
    discovery_tool_names,
    event_rows,
    goal_tools,
    live_event_names,
    mutation_execution_counts,
    pending_goal_tools,
    successful_tool_names,
    tool_steps,
    values_equal,
    walk_path,
)


def _check(name, passed, detail=None, severity="error"):
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "severity": severity,
    }


def _all_discovered_sets(trace):
    sets = []
    for event in trace.get("live_events", []):
        if event.get("event") == "llm_end":
            sets.append(set(event.get("selected_tool_names", []) or []))
    return sets


def score_turn(trace, expectations):
    expectations = expectations or {}
    checks = []

    expected_status = expectations.get("status", "success")
    actual_success = bool(trace.get("success"))

    checks.append(
        _check(
            "request_status",
            actual_success == (expected_status == "success"),
            {
                "expected": expected_status,
                "actual": "success" if actual_success else "failed",
            },
        )
    )

    if expectations.get("verification_complete", True):
        checks.append(
            _check(
                "verification_complete",
                trace.get("verification_complete") is True,
                {"actual": trace.get("verification_complete")},
            )
        )

    pending = pending_goal_tools(trace)
    if expectations.get("all_goals_satisfied", True):
        checks.append(
            _check(
                "all_goals_satisfied",
                not pending,
                {"pending_goals": pending},
            )
        )

    executed_success = set(successful_tool_names(trace))
    discovered = discovery_tool_names(trace)
    goals = set(goal_tools(trace))
    events = set(live_event_names(trace))

    required_tools = set(expectations.get("required_tools", []))
    forbidden_tools = set(expectations.get("forbidden_tools", []))

    for tool in sorted(required_tools):
        checks.append(
            _check(
                f"required_tool:{tool}",
                tool in executed_success,
                {"successful_tools": sorted(executed_success)},
            )
        )

    for tool in sorted(forbidden_tools):
        checks.append(
            _check(
                f"forbidden_tool:{tool}",
                tool not in executed_success,
                {"successful_tools": sorted(executed_success)},
            )
        )

    required_discovered = set(
        expectations.get("required_discovered_tools", [])
    )
    forbidden_discovered = set(
        expectations.get("forbidden_discovered_tools", [])
    )

    for tool in sorted(required_discovered):
        checks.append(
            _check(
                f"required_discovered:{tool}",
                tool in discovered,
                {"discovered_tools": sorted(discovered)},
            )
        )

    for tool in sorted(forbidden_discovered):
        checks.append(
            _check(
                f"forbidden_discovered:{tool}",
                tool not in discovered,
                {"discovered_tools": sorted(discovered)},
            )
        )

    required_goal_tools = set(expectations.get("required_goal_tools", []))
    for tool in sorted(required_goal_tools):
        checks.append(
            _check(
                f"required_goal:{tool}",
                tool in goals,
                {"goal_tools": sorted(goals)},
            )
        )

    for event_name in expectations.get("required_events", []):
        checks.append(
            _check(
                f"required_event:{event_name}",
                event_name in events,
                {"events": sorted(events)},
            )
        )

    for event_name in expectations.get("forbidden_events", []):
        checks.append(
            _check(
                f"forbidden_event:{event_name}",
                event_name not in events,
                {"events": sorted(events)},
            )
        )

    final_answer = str(trace.get("final_answer") or "")
    for term in expectations.get("forbidden_final_answer_terms", []):
        passed = str(term).lower() not in final_answer.lower()
        checks.append(
            _check(
                f"forbidden_final_answer_term:{term}",
                passed,
                {"final_answer": final_answer},
            )
        )

    reference = expectations.get("reference_target")
    if reference:
        resolution = trace.get("reference_resolution", {}) or {}
        checks.append(
            _check(
                "reference_target_type",
                resolution.get("target_type") == reference.get("type"),
                {
                    "expected": reference.get("type"),
                    "actual": resolution.get("target_type"),
                },
            )
        )
        checks.append(
            _check(
                "reference_target_name",
                resolution.get("target_name") == reference.get("name"),
                {
                    "expected": reference.get("name"),
                    "actual": resolution.get("target_name"),
                },
            )
        )

    for assertion in expectations.get("tool_argument_equals", []):
        tool = assertion["tool"]
        path = assertion["path"]
        expected = assertion.get("value")
        matching = tool_steps(trace, tool)
        actual_values = [
            walk_path(step.get("arguments", {}), path)
            for step in matching
        ]
        passed = any(
            values_equal(value, expected)
            for value in actual_values
        )
        checks.append(
            _check(
                f"tool_argument:{tool}.{path}",
                passed,
                {
                    "expected": expected,
                    "actual_values": actual_values,
                },
            )
        )

    for assertion in expectations.get("tool_result_equals", []):
        tool = assertion["tool"]
        path = assertion["path"]
        expected = assertion.get("value")
        matching = tool_steps(trace, tool)
        actual_values = [
            walk_path(step.get("result", {}), path)
            for step in matching
        ]
        passed = any(
            values_equal(value, expected)
            for value in actual_values
        )
        checks.append(
            _check(
                f"tool_result:{tool}.{path}",
                passed,
                {
                    "expected": expected,
                    "actual_values": actual_values,
                },
            )
        )

    max_tool_repeats = expectations.get("max_tool_repeats", {})
    execution_counts = mutation_execution_counts(trace)
    for tool, max_count in max_tool_repeats.items():
        actual = execution_counts.get(tool, 0)
        checks.append(
            _check(
                f"max_repeat:{tool}",
                actual <= int(max_count),
                {
                    "max": int(max_count),
                    "actual": actual,
                },
            )
        )

    max_render_executions = expectations.get("max_render_executions")
    if max_render_executions is not None:
        actual = execution_counts.get("render_scene", 0)
        checks.append(
            _check(
                "render_at_most_once",
                actual <= int(max_render_executions),
                {
                    "max": int(max_render_executions),
                    "actual": actual,
                },
            )
        )

    max_llm_calls = expectations.get("max_llm_calls")
    llm_calls = len(event_rows(trace, "llm_end"))
    if max_llm_calls is not None:
        checks.append(
            _check(
                "max_llm_calls",
                llm_calls <= int(max_llm_calls),
                {
                    "max": int(max_llm_calls),
                    "actual": llm_calls,
                },
                severity="warning",
            )
        )

    max_total_seconds = expectations.get("max_total_seconds")
    if max_total_seconds is not None:
        total = trace.get("total_time")
        checks.append(
            _check(
                "max_total_seconds",
                total is not None and float(total) <= float(max_total_seconds),
                {
                    "max": float(max_total_seconds),
                    "actual": total,
                },
                severity="warning",
            )
        )

    allowed_discovered = expectations.get("allowed_discovered_tools")
    discovery_precision = None
    if allowed_discovered is not None:
        allowed = set(allowed_discovered)
        if discovered:
            discovery_precision = len(discovered & allowed) / len(discovered)
        else:
            discovery_precision = 1.0 if not allowed else 0.0

    discovery_recall = None
    if required_discovered:
        discovery_recall = (
            len(discovered & required_discovered)
            / len(required_discovered)
        )

    # Strict semantic false-success: controller says success, but the harness
    # expectation layer found one or more ERROR-level failures.
    error_failures = [
        check
        for check in checks
        if not check["passed"] and check["severity"] == "error"
    ]

    semantic_false_success = bool(
        trace.get("success") is True
        and error_failures
    )

    return {
        "passed": not error_failures,
        "checks": checks,
        "semantic_false_success": semantic_false_success,
        "metrics": {
            "total_time": trace.get("total_time"),
            "llm_calls": llm_calls,
            "tool_steps": len(trace.get("steps", [])),
            "successful_tools": len(successful_tool_names(trace)),
            "approval_count": len(event_rows(trace, "approval_required")),
            "render_executions": execution_counts.get("render_scene", 0),
            "discovery_precision": discovery_precision,
            "discovery_recall": discovery_recall,
            "prompt_eval_tokens": sum(
                int(event.get("prompt_eval_count") or 0)
                for event in trace.get("live_events", [])
                if event.get("event") == "llm_end"
            ),
            "generated_tokens": sum(
                int(event.get("eval_count") or 0)
                for event in trace.get("live_events", [])
                if event.get("event") == "llm_end"
            ),
        },
    }
