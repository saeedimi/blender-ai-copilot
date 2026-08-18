
from __future__ import annotations

from pathlib import Path


def _fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path, payload):
    path = Path(path)
    summary = payload["summary"]

    lines = [
        f"# Blender AI Copilot Evaluation Report",
        "",
        f"**Run:** `{payload['run_id']}`  ",
        f"**Suite:** `{payload.get('suite')}`  ",
        f"**Backend:** `{payload.get('backend')}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Case pass rate | {summary['case_pass_rate']:.1%} |",
        f"| Turn pass rate | {summary['turn_pass_rate']:.1%} |",
        f"| Semantic false-success rate | {summary['semantic_false_success_rate']:.1%} |",
        f"| Average latency | {_fmt(summary['avg_total_time'])} s |",
        f"| P95 latency | {_fmt(summary['p95_total_time'])} s |",
        f"| Average LLM calls | {_fmt(summary['avg_llm_calls'])} |",
        f"| Average tool steps | {_fmt(summary['avg_tool_steps'])} |",
        "| Discovery precision | {} |".format('n/a' if summary['discovery_precision'] is None else f"{summary['discovery_precision']:.1%}"),
        "| Discovery recall | {} |".format('n/a' if summary['discovery_recall'] is None else f"{summary['discovery_recall']:.1%}"),
        f"| Duplicate render incidents | {summary['duplicate_render_incidents']} |",
        "",
        "## Cases",
        "",
    ]

    for case in payload["results"]:
        lines.append(
            f"### {'✅' if case['passed'] else '❌'} {case['id']}"
        )
        if case.get("description"):
            lines.append(case["description"])
        lines.append("")

        for turn in case["turns"]:
            score = turn["score"]
            lines.append(
                f"**Turn {turn['turn_index']} — "
                f"{'PASS' if score['passed'] else 'FAIL'}**"
            )
            lines.append("")
            lines.append(f"> {turn['prompt']}")
            lines.append("")
            lines.append(
                f"- Trace: `{turn.get('trace_id')}`"
            )
            lines.append(
                f"- Latency: {_fmt(score['metrics']['total_time'])} s"
            )
            lines.append(
                f"- LLM calls: {score['metrics']['llm_calls']}"
            )
            lines.append(
                f"- Tool steps: {score['metrics']['tool_steps']}"
            )
            lines.append(
                f"- Semantic false-success: "
                f"{score['semantic_false_success']}"
            )

            failed = [
                check
                for check in score["checks"]
                if not check["passed"]
            ]
            if failed:
                lines.append("- Failed checks:")
                for check in failed:
                    lines.append(
                        f"  - `{check['name']}` ({check['severity']}): "
                        f"{check.get('detail')}"
                    )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
