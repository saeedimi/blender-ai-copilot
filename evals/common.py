
from __future__ import annotations

import json
import math
import re
import time
import uuid
import zlib
from pathlib import Path
from typing import Any


MUTATING_TOOL_NAMES = {
    "create_cube",
    "move_object",
    "delete_object",
    "create_material",
    "set_material_color",
    "assign_material",
    "add_bevel_modifier",
    "set_bevel_modifier",
    "add_subdivision_modifier",
    "set_subdivision_modifier",
    "remove_modifier",
    "apply_modifier",
    "create_camera",
    "move_camera",
    "set_camera_lens",
    "set_active_camera",
    "aim_camera_at_object",
    "create_light",
    "move_light",
    "set_light_energy",
    "set_light_color",
    "set_area_light_size",
    "aim_light_at_object",
    "set_render_engine",
    "set_render_resolution",
    "set_render_samples",
    "set_render_output",
    "set_render_transparent",
    "render_scene",
    "create_uv_sphere",
    "create_cylinder",
    "create_cone",
    "create_plane",
    "create_torus",
    "shade_smooth",
    "recalculate_normals",
    "scale_mesh_geometry",
    "extrude_top_face",
    "bevel_mesh_edges",
    "inset_top_face",
    "subdivide_mesh",
    "translate_top_face",
    "scale_top_face",
    "merge_by_distance",
    "solidify_mesh",
}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def percentile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return values[lo]
    fraction = position - lo
    return values[lo] * (1.0 - fraction) + values[hi] * fraction


def new_run_variables():
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    slot = zlib.crc32(run_id.encode("utf-8")) % 500

    # Each run gets a distant origin so repeated evaluation runs do not
    # stack newly-created objects on previous evaluation objects.
    origin_x = 100.0 + float(slot * 24)
    origin_y = 100.0 + float((slot % 17) * 28)

    return {
        "run_id": run_id,
        "ox": origin_x,
        "oy": origin_y,
        "oz": 0.0,
    }


def render_template(value, variables):
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [render_template(item, variables) for item in value]
    if isinstance(value, dict):
        return {
            key: render_template(item, variables)
            for key, item in value.items()
        }
    return value


def walk_path(payload, path, default=None):
    current = payload
    for part in str(path).split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except Exception:
                return default
        else:
            return default
    return current


def values_equal(actual, expected, tolerance=1e-5):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    if isinstance(expected, list) and isinstance(actual, list):
        return (
            len(actual) == len(expected)
            and all(
                values_equal(a, e, tolerance=tolerance)
                for a, e in zip(actual, expected)
            )
        )
    return actual == expected


def step_tool_names(trace):
    return [
        step.get("tool")
        for step in trace.get("steps", [])
        if step.get("tool")
    ]


def successful_tool_names(trace):
    return [
        step.get("tool")
        for step in trace.get("steps", [])
        if step.get("tool") and step.get("status") == "success"
    ]


def discovery_tool_names(trace):
    names = set()
    for event in trace.get("live_events", []):
        if event.get("event") != "llm_end":
            continue
        for name in event.get("selected_tool_names", []) or []:
            names.add(name)
    return names


def live_event_names(trace):
    return [
        event.get("event")
        for event in trace.get("live_events", [])
        if event.get("event")
    ]


def goal_tools(trace):
    return [
        goal.get("tool")
        for goal in trace.get("goal_ledger", [])
        if goal.get("tool")
    ]


def pending_goal_tools(trace):
    return [
        goal.get("tool")
        for goal in trace.get("goal_ledger", [])
        if goal.get("tool") and goal.get("status") != "satisfied"
    ]


def mutation_execution_counts(trace):
    counts = {}
    for step in trace.get("steps", []):
        tool = step.get("tool")
        if tool not in MUTATING_TOOL_NAMES:
            continue
        if step.get("status") not in {"success", "verification_failed"}:
            continue
        counts[tool] = counts.get(tool, 0) + 1
    return counts


def tool_steps(trace, tool_name):
    return [
        step
        for step in trace.get("steps", [])
        if step.get("tool") == tool_name
    ]


def event_rows(trace, event_name):
    return [
        event
        for event in trace.get("live_events", [])
        if event.get("event") == event_name
    ]


def normalize_trace(response):
    trace = response.get("trace") if isinstance(response, dict) else None
    if isinstance(trace, dict):
        return trace
    return {}
