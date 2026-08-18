"""Main Blender AI Copilot agent and localhost HTTP backend."""

from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import re
import threading
import time
import uuid

import requests

try:
    from .bridge import BlenderBridge
    from .rag import BlenderRAGRetriever
    from .router import ToolRouter
    from .tools import (
        MUTATING_TOOLS,
        OBSERVATION_TOOLS,
        VERIFICATION_TOOL_BY_MUTATION,
        OLLAMA_TOOLS,
        OLLAMA_TOOL_BY_NAME,
        TOOL_CATEGORIES,
        CATEGORY_CORE_TOOLS,
        CATEGORY_FALLBACK_TOOLS,
        TOOL_DISCOVERY_HINTS,
        TOOL_BEHAVIOR_GROUPS,
        NO_AUTO_REPLAY_TOOLS,
        get_tool_domain,
        get_tool_behavior,
        get_tool_risk,
        tool_call_to_command,
        validate_tool_call,
    )
except ImportError:
    from bridge import BlenderBridge
    from rag import BlenderRAGRetriever
    from router import ToolRouter
    from tools import (
        MUTATING_TOOLS,
        OBSERVATION_TOOLS,
        VERIFICATION_TOOL_BY_MUTATION,
        OLLAMA_TOOLS,
        OLLAMA_TOOL_BY_NAME,
        TOOL_CATEGORIES,
        CATEGORY_CORE_TOOLS,
        CATEGORY_FALLBACK_TOOLS,
        TOOL_DISCOVERY_HINTS,
        TOOL_BEHAVIOR_GROUPS,
        NO_AUTO_REPLAY_TOOLS,
        get_tool_domain,
        get_tool_behavior,
        get_tool_risk,
        tool_call_to_command,
        validate_tool_call,
    )


SYSTEM_PROMPT = """
You are a Blender AI copilot operating through a controlled tool interface.

You have two kinds of tools:

KNOWLEDGE:
- search_blender_docs
  Retrieves official Blender Manual / Blender Python API passages.
  It never modifies Blender.

BLENDER:
- get_scene_objects
- get_materials
- create_cube
- move_object
- delete_object
- create_material
- set_material_color
- assign_material
- get_modifiers
- add_bevel_modifier
- set_bevel_modifier
- add_subdivision_modifier
- set_subdivision_modifier
- remove_modifier
- apply_modifier
- get_cameras
- create_camera
- move_camera
- set_camera_lens
- set_active_camera
- aim_camera_at_object
- get_lights
- create_light
- move_light
- set_light_energy
- set_light_color
- set_area_light_size
- aim_light_at_object
- get_render_settings
- set_render_engine
- set_render_resolution
- set_render_samples
- set_render_output
- set_render_transparent
- render_scene
- get_mesh_info
- create_uv_sphere
- create_cylinder
- create_cone
- create_plane
- create_torus
- shade_smooth
- recalculate_normals
- scale_mesh_geometry
- extrude_top_face
- bevel_mesh_edges
- get_mesh_regions
- inset_top_face
- subdivide_mesh
- translate_top_face
- scale_top_face
- merge_by_distance
- solidify_mesh

Rules:

1. You may propose one or more semantic tools in a turn. The controller validates and queues multi-tool plans, executing one tool at a time with safety and verification between mutations.
2. For precise Blender behavior, API semantics, modes/context, materials,
   rendering, modifiers, cameras, or uncertain Blender-specific knowledge,
   use search_blender_docs before answering or choosing a documentation-
   dependent action.
3. Do not search documentation unnecessarily for a simple scene inspection
   or a straightforward action already represented by an explicit tool.
4. If documentation is retrieved, read the evidence before taking an action.
5. Retrieved documentation can describe capabilities that this copilot does
   not expose. Never invent a missing action tool and never execute arbitrary
   Python as a workaround.
6. Never claim an action succeeded when a tool reports failure.
7. After a successful mutation, verify the result before finishing:
   - use get_scene_objects for object creation/movement/deletion;
   - use get_materials for material creation/color/assignment;
   - use get_modifiers for modifier creation/change/removal/application;
   - use get_cameras for camera creation/position/lens/aim/active-camera changes;
   - use get_lights for light creation/position/energy/color/size/aim changes;
   - use get_render_settings for render-engine/resolution/sample/output/transparency changes;
   - use get_mesh_info for direct mesh geometry/shading/normal changes.
8. Treat the tool result as the source of truth for the current Blender scene.
9. delete_object is high risk and may require explicit user approval.
10. When an answer relies on retrieved documentation, cite labels such as
    [DOC1] and [DOC2].
11. apply_modifier permanently changes geometry and is high risk. Never claim
    it was applied unless the approval flow and tool execution both succeeded.
12. If the user explicitly asks to apply a named modifier to a named object,
    call apply_modifier directly. Do NOT call search_blender_docs or
    get_modifiers first just to check existence. The controller itself performs
    the required modifier preflight, approval gate, execution, and post-apply
    verification.

Typical flows:

Knowledge question:
search_blender_docs -> answer with citations

Scene question:
get_scene_objects -> answer

Simple supported object action:
action -> get_scene_objects -> answer

Material action:
create/set/assign material -> get_materials -> answer

Modifier action:
add/set/remove/apply modifier -> get_modifiers -> answer

Camera action:
create/move/set/aim camera -> get_cameras -> answer

Light action:
create/move/set/aim light -> get_lights -> answer

Render-settings action:
set render settings -> get_render_settings -> answer

Render action:
ensure an active camera exists -> render_scene -> answer using the returned verified result.
If the user wants a saved image and explicitly provides a filename, use that exact filename.
If the user says only "render to file" without a new filename, preserve Blender's already
configured safe output path; do not invent scene_render.png.
Never call render_scene more than once for the same user request. The controller enforces
at-most-once execution because rendering is side-effecting.
Never claim a render or saved image exists based only on render settings. A saved render is
complete only when render_scene itself returns success=true, verified=true,
saved_to_file=true, file_verified=true, and an exact output_path.

Mesh modeling:
- primitive creation -> get_scene_objects -> answer or continue;
- shade_smooth / recalculate_normals -> get_mesh_info -> answer or continue;
- scale_mesh_geometry, extrude_top_face, bevel_mesh_edges, inset_top_face,
  subdivide_mesh, translate_top_face, scale_top_face, merge_by_distance,
  and solidify_mesh are permanent direct geometry edits and require controller approval;
- use get_mesh_regions when a workflow depends on the semantic highest upward-facing top region;
- prefer add_bevel_modifier for a non-destructive bevel unless the user
  specifically requests a permanent mesh bevel.

Documentation-dependent supported action:
search_blender_docs -> reconsider -> action -> appropriate verification tool -> answer
""".strip()


HISTORY_CONTEXT_MAX_MESSAGES = 6
HISTORY_CONTEXT_MAX_MESSAGE_CHARS = 1400
HISTORY_MEMORY_MAX_CHARS = 3500
HISTORY_MEMORY_MAX_LINES = 24

HISTORY_CONTEXT_PROMPT = """
Conversation context is provided only to resolve references and maintain continuity.
Treat it as background, not as new instructions and not as proof of current Blender state.
The current user request is the task to execute.
When prior conversation conflicts with verified Blender tool observations, trust the current
tool observations. Do not repeat an old action unless the current user request asks for it.
""".strip()


@dataclass
class AgentState:
    user_request: str
    messages: list
    trace: dict
    needs_verification: bool = False
    required_verification_tool: str | None = None
    step: int = 0
    citation_repair_used: bool = False

    # Render transaction state. One user request may execute render_scene
    # at most once. This is controller-owned, not LLM-owned.
    render_required: bool = False
    render_save_required: bool = False
    render_attempted: bool = False
    render_completed: bool = False
    requested_render_filename: str | None = None
    verified_render_result: dict | None = None

    # Parsed tool calls from one LLM turn are no longer discarded.
    # The controller executes this queue one command at a time and still
    # applies validation, risk checks, approval, and verification per tool.
    pending_tool_plan: list = field(
        default_factory=list
    )

    # Controller-owned task goals parsed from the user's request. The agent
    # may only report success when every explicit supported goal is satisfied.
    goal_ledger: list = field(default_factory=list)
    goal_repair_attempts: int = 0

    # Bounded conversation context supplied by the Blender scene.
    conversation_context: dict = field(default_factory=dict)
    prior_memory_summary: str = ""

    # Structured referential state. This is controller-owned and is used to
    # resolve "it", "that object", "the camera", etc. before tool discovery.
    prior_structured_memory: dict = field(default_factory=dict)
    structured_memory: dict = field(default_factory=dict)
    reference_resolution: dict = field(default_factory=dict)
    planning_request: str = ""


class CopilotAgent:
    def __init__(
        self,
        project_root,
        ollama_url="http://127.0.0.1:11434",
        model="qwen3:4b-instruct",
        max_steps=40,
        ollama_num_ctx=None,
        ollama_max_num_ctx=None,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.max_steps = int(max_steps)

        # Long multi-capability agent requests can exceed Ollama's default
        # context. Start at 8K and automatically grow to 32K when needed.
        self.ollama_num_ctx = int(
            ollama_num_ctx
            or os.environ.get(
                "BLENDER_COPILOT_NUM_CTX",
                "8192",
            )
        )

        self.ollama_max_num_ctx = int(
            ollama_max_num_ctx
            or os.environ.get(
                "BLENDER_COPILOT_MAX_NUM_CTX",
                "32768",
            )
        )

        if self.ollama_num_ctx < 4096:
            self.ollama_num_ctx = 4096

        if self.ollama_max_num_ctx < self.ollama_num_ctx:
            self.ollama_max_num_ctx = self.ollama_num_ctx

        self.bridge = BlenderBridge(self.project_root)
        self.rag = BlenderRAGRetriever(self.project_root)
        self.router = ToolRouter(self.rag, self.bridge)

        self.pending_approvals = {}
        self._pending_lock = threading.Lock()

        # ----------------------------------------------------------
        # Persistent local observability.
        #
        # Every request gets:
        #   traces/<trace_id>.json  - structured snapshot
        #   traces/<trace_id>.log   - live JSONL event stream
        #
        # Convenience mirrors:
        #   traces/latest.json
        #   traces/latest.log
        # ----------------------------------------------------------
        self.trace_dir = (
            self.project_root
            / "traces"
        )
        self.trace_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._trace_lock = (
            threading.Lock()
        )
        self._latest_trace_id = None

    def health(self):
        return {
            "status": "ok",
            "project_root": str(self.project_root),
            "model": self.model,
            "ollama_url": self.ollama_url,
            "rag_chunks": len(self.rag.chunks),
            "pending_approvals": len(self.pending_approvals),
            "controller_version": "0.8.0.2",
            "evaluation_harness": True,
            "evaluation_harness_version": "0.8.0.2",
            "structured_referential_memory": True,
            "reference_resolution_fail_closed": True,
            "object_color_uses_dedicated_material": True,
            "reference_target_argument_enforcement": True,
            "same_turn_reference_handling": True,
            "deterministic_simple_goal_injection": True,
            "mixed_workflow_aim_goal_coverage": True,
            "final_answer_claim_grounding": True,
            "decimal_safe_clause_parsing": True,
            "clause_local_aim_goal_extraction": True,
            "deterministic_ready_aim_completion": True,
            "conversation_history_context": True,
            "conversation_recent_turns_default": 3,
            "conversation_recent_message_limit": HISTORY_CONTEXT_MAX_MESSAGES,
            "conversation_memory_summary": True,
            "conversation_memory_summary_max_chars": HISTORY_MEMORY_MAX_CHARS,
            "conversation_context_source": "blender_scene",
            "conversation_context_is_bounded": True,
            "tool_grouping": True,
            "tool_grouping_axes": ["domain", "behavior"],
            "no_auto_replay_after_successful_side_effect": True,
            "direction_semantic_normalization": True,
            "complete_goal_coverage_v2": True,
            "mesh_region_contract_v2": True,
            "more_mesh_tools": True,
            "tool_level_dynamic_discovery": True,
            "dynamic_discovery_mode": "goal_aware_tool_discovery",
            "goal_level_verification": True,
            "semantic_top_region_verification": True,
            "chat_history_ui": True,
            "high_risk_mesh_post_verification": True,
            "deterministic_state_verification": True,
            "camera_state_reconciliation": True,
            "two_stage_render_save": True,
            "render_handler_verification": True,
            "bridge_preserves_failure_metadata": True,
            "validated_multi_tool_plan_queue": True,
            "terminal_render_obligation": True,
            "plan_aborts_on_tool_failure": True,
            "deterministic_render_verification": True,
            "verification_fail_closed": True,
            "strict_render_filename_parser": True,
            "render_transaction_idempotency": True,
            "render_at_most_once_per_request": True,
            "render_bridge_timeout_seconds": 600,
            "render_bridge_auto_retry": False,
            "render_is_terminal_action": True,
            "trace_observability": True,
            "trace_dir": str(self.trace_dir),
            "latest_trace_json": str(
                self.trace_dir
                / "latest.json"
            ),
            "latest_trace_log": str(
                self.trace_dir
                / "latest.log"
            ),
            "trace_http_endpoint": "/trace/latest",
            "dynamic_tool_gating": True,
            "dynamic_tool_discovery": True,
            "adaptive_context_window": True,
            "ollama_num_ctx": self.ollama_num_ctx,
            "ollama_max_num_ctx": self.ollama_max_num_ctx,
            "max_agent_steps": self.max_steps,
            "total_registered_tools": len(OLLAMA_TOOLS),
            "tool_routes": {
                tool["function"]["name"]: self.router.route_name(
                    tool["function"]["name"]
                )
                for tool in OLLAMA_TOOLS
            },
        }

    @staticmethod
    def _trace_json_safe(
        value,
    ):
        """
        Convert arbitrary trace payloads to ordinary JSON-safe objects.
        """
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        )


    def _trace_snapshot(
        self,
        trace,
    ):
        snapshot = (
            self._trace_json_safe(
                trace
            )
        )

        started = (
            trace.get(
                "_started_perf"
            )
        )

        if started is not None:
            snapshot[
                "live_elapsed"
            ] = max(
                0.0,
                time.perf_counter()
                - float(started),
            )

        # Internal perf counter is meaningless outside this Python process.
        snapshot.pop(
            "_started_perf",
            None,
        )

        return snapshot


    @staticmethod
    def _atomic_write_text(
        path,
        text,
    ):
        temporary = (
            path.with_suffix(
                path.suffix
                + ".tmp"
            )
        )

        temporary.write_text(
            text,
            encoding="utf-8",
        )

        temporary.replace(
            path
        )


    def _persist_trace_snapshot(
        self,
        trace,
    ):
        trace_id = str(
            trace[
                "trace_id"
            ]
        )

        snapshot = (
            self._trace_snapshot(
                trace
            )
        )

        payload = json.dumps(
            snapshot,
            indent=2,
            ensure_ascii=False,
        )

        with self._trace_lock:
            self.trace_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._atomic_write_text(
                self.trace_dir
                / f"{trace_id}.json",
                payload,
            )

            if (
                self._latest_trace_id
                == trace_id
            ):
                self._atomic_write_text(
                    self.trace_dir
                    / "latest.json",
                    payload,
                )


    def _record_trace_event(
        self,
        trace,
        event,
        **fields,
    ):
        trace_id = str(
            trace[
                "trace_id"
            ]
        )

        event_payload = {
            "timestamp": (
                datetime.now()
                .isoformat()
            ),
            "trace_id": trace_id,
            "event": str(
                event
            ),
            **self._trace_json_safe(
                fields
            ),
        }

        trace.setdefault(
            "live_events",
            [],
        ).append(
            event_payload
        )

        line = (
            json.dumps(
                event_payload,
                ensure_ascii=False,
            )
            + "\\n"
        )

        with self._trace_lock:
            self.trace_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with (
                self.trace_dir
                / f"{trace_id}.log"
            ).open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(
                    line
                )

            if (
                self._latest_trace_id
                == trace_id
            ):
                with (
                    self.trace_dir
                    / "latest.log"
                ).open(
                    "a",
                    encoding="utf-8",
                ) as stream:
                    stream.write(
                        line
                    )

        # Also expose the same event in the backend terminal for immediate
        # debugging without opening a file.
        summary_fields = []

        for key in (
            "step",
            "tool",
            "status",
            "latency",
            "attempts",
            "reason",
            "output_path",
        ):
            if key in event_payload:
                summary_fields.append(
                    f"{key}={event_payload[key]}"
                )

        summary = (
            " "
            + " ".join(
                summary_fields
            )
            if summary_fields
            else ""
        )

        print(
            f"[TRACE {trace_id[:8]}] "
            f"{event}{summary}"
        )

        self._persist_trace_snapshot(
            trace
        )


    def _start_trace_files(
        self,
        trace,
    ):
        trace_id = str(
            trace[
                "trace_id"
            ]
        )

        with self._trace_lock:
            self.trace_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._latest_trace_id = (
                trace_id
            )

            # A new request owns the convenience "latest" files.
            for path in (
                self.trace_dir
                / f"{trace_id}.log",
                self.trace_dir
                / "latest.log",
            ):
                path.write_text(
                    "",
                    encoding="utf-8",
                )

        self._record_trace_event(
            trace,
            "request_started",
            user_request=(
                trace.get(
                    "user_request"
                )
            ),
        )


    def get_trace(
        self,
        trace_id=None,
    ):
        requested = (
            str(trace_id).strip()
            if trace_id
            else ""
        )

        if not requested:
            requested = (
                self._latest_trace_id
                or ""
            )

        # Only UUID-like trace IDs are allowed as path components.
        if (
            requested
            and not re.fullmatch(
                r"[0-9a-fA-F-]{36}",
                requested,
            )
        ):
            raise ValueError(
                "Invalid trace ID."
            )

        if not requested:
            return {
                "status": "error",
                "error": (
                    "No trace has been created yet."
                ),
            }

        trace_path = (
            self.trace_dir
            / f"{requested}.json"
        )

        log_path = (
            self.trace_dir
            / f"{requested}.log"
        )

        if not trace_path.exists():
            return {
                "status": "error",
                "error": (
                    f"Trace '{requested}' was not found."
                ),
            }

        payload = json.loads(
            trace_path.read_text(
                encoding="utf-8"
            )
        )

        payload[
            "trace_path"
        ] = str(
            trace_path
        )

        payload[
            "trace_log_path"
        ] = str(
            log_path
        )

        return {
            "status": "ok",
            "trace": payload,
        }


    def _create_trace(self, user_request):
        trace = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_request": user_request,
            "steps": [],
            "controller_events": [],
            "live_events": [],
            "final_answer": None,
            "success": None,
            "verification_complete": None,
            "total_time": None,
            "_started_perf": time.perf_counter(),
        }

        self._start_trace_files(
            trace
        )

        return trace

    @staticmethod
    def _sanitize_conversation_context(context, current_request=""):
        """
        The Blender scene owns the visible full history. Only a small, bounded
        subset reaches the model.

        Accepted payload:
          {
            "recent_messages": [{"role": "user|assistant", "content": "..."}],
            "memory_summary": "..."
          }
        """
        if not isinstance(context, dict):
            context = {}

        memory_summary = str(
            context.get("memory_summary", "")
            or ""
        ).strip()

        if len(memory_summary) > HISTORY_MEMORY_MAX_CHARS:
            memory_summary = memory_summary[-HISTORY_MEMORY_MAX_CHARS:]

        recent = []
        raw_messages = context.get("recent_messages", [])

        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue

                role = str(item.get("role", "")).strip().lower()
                if role not in {"user", "assistant"}:
                    continue

                content = str(item.get("content", "") or "").strip()
                if not content:
                    continue

                if len(content) > HISTORY_CONTEXT_MAX_MESSAGE_CHARS:
                    content = content[:HISTORY_CONTEXT_MAX_MESSAGE_CHARS] + "…"

                recent.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        recent = recent[-HISTORY_CONTEXT_MAX_MESSAGES:]

        current = str(current_request or "").strip()
        if (
            current
            and recent
            and recent[-1].get("role") == "user"
            and recent[-1].get("content", "").strip() == current
        ):
            recent.pop()

        structured_memory = context.get("structured_memory", {})
        if not isinstance(structured_memory, dict):
            structured_memory = {}

        allowed_structured_keys = {
            "last_entity_type",
            "last_entity_name",
            "last_object",
            "last_material",
            "last_camera",
            "last_light",
            "last_render_output",
        }

        structured_memory = {
            str(key): str(value)
            for key, value in structured_memory.items()
            if key in allowed_structured_keys
            and value is not None
            and str(value).strip()
        }

        return {
            "recent_messages": recent,
            "memory_summary": memory_summary,
            "structured_memory": structured_memory,
        }


    @staticmethod
    def _conversation_context_messages(context):
        context = context or {}
        messages = []

        summary = str(context.get("memory_summary", "") or "").strip()
        recent = context.get("recent_messages", []) or []

        if summary or recent:
            messages.append(
                {
                    "role": "system",
                    "content": HISTORY_CONTEXT_PROMPT,
                }
            )

        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Compact memory from earlier completed Blender work:\\n"
                        + summary
                    ),
                }
            )

        for item in recent:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        return messages


    @staticmethod
    def _has_referential_phrase(user_request):
        text = str(user_request or "").lower()
        return bool(
            re.search(
                r"\b(it|this|that)\b"
                r"|\b(?:this|that|the)\s+(?:object|mesh|camera|light|material)\b",
                text,
            )
        )


    @staticmethod
    def _color_word(user_request):
        match = re.search(
            r"\b(red|blue|green|yellow|white|black|gray|grey|orange|purple|pink|brown)\b",
            str(user_request or "").lower(),
        )
        return match.group(1) if match else None


    @classmethod
    def _has_same_turn_reference(cls, user_request):
        """
        Return True when a referential phrase occurs *after* an entity is
        introduced inside the current request.

        This distinction matters because a request such as
        "Create CameraA, aim it at SphereA" must not consult prior-turn memory
        for "it". The current request itself owns that antecedent.
        """
        text = str(user_request or "")
        if not cls._has_referential_phrase(text):
            return False

        introduction_pattern = re.compile(
            r"\b(?:create|add|make|place)\b"
            r"[^.;\n]{0,220}"
            r"\b(?:cube|(?:uv\s+)?sphere|cylinder|cone|plane|torus|object|mesh|camera|material|(?:area|point|spot|sun)?\s*light)\b",
            re.IGNORECASE,
        )
        reference_pattern = re.compile(
            r"\b(?:it|this|that)\b"
            r"|\b(?:this|that|the)\s+(?:object|mesh|camera|light|material)\b",
            re.IGNORECASE,
        )

        introductions = list(introduction_pattern.finditer(text))
        references = list(reference_pattern.finditer(text))

        for reference in references:
            if any(intro.end() <= reference.start() for intro in introductions):
                return True

        return False


    @staticmethod
    def _named_entity_from_request(user_request, entity_type):
        text = str(user_request or "")
        noun_by_type = {
            "object": r"(?:cube|(?:uv\s+)?sphere|cylinder|cone|plane|torus|object|mesh)",
            "camera": r"camera",
            "light": r"(?:(?:area|point|spot|sun)\s+)?light",
            "material": r"material",
        }
        noun = noun_by_type.get(entity_type)
        if not noun:
            return None

        quoted = re.search(
            rf"\b{noun}\b[^.;\n]{{0,45}}\bnamed\s+[`'\"]([^`'\"]+)[`'\"]",
            text,
            re.IGNORECASE,
        )
        if quoted:
            return quoted.group(1).strip()

        bare = re.search(
            rf"\b{noun}\b[^.;\n]{{0,45}}\bnamed\s+([A-Za-z0-9_.:-]+)",
            text,
            re.IGNORECASE,
        )
        return bare.group(1).strip().rstrip(".,;:!?") if bare else None


    @staticmethod
    def _aim_target_from_request(user_request):
        text = str(user_request or "")
        target = re.search(
            r"\b(?:aim|point)\s+(?:it|the\s+camera|the\s+light|camera|light)\s+at\s+[`'\"]?([A-Za-z0-9_.:-]+)",
            text,
            re.IGNORECASE,
        )
        return target.group(1).strip().rstrip(".,;:!?") if target else None


    @staticmethod
    def _request_clauses(user_request):
        """
        Split a request into sentence-like clauses without treating decimal
        points in coordinates (for example 364.0) as sentence boundaries.
        """
        text = re.sub(r"\s+", " ", str(user_request or "")).strip()
        if not text:
            return []
        parts = re.split(r"(?<!\d)\.(?=\s|$)|;|\n+", text)
        return [part.strip(" \t,;.") for part in parts if part.strip(" \t,;.")]


    @classmethod
    def _clause_local_aim_intents(cls, user_request):
        """
        Extract typed camera/light aim actions and their exact source/target
        names from the current request. Pronoun `it` is bound to the nearest
        camera or light noun preceding the aim phrase inside the same clause.
        """
        intents = []
        seen = set()

        action_pattern = re.compile(
            r"\b(?:aim|point)\s+"
            r"(?P<subject>it|the\s+camera|camera|the\s+light|light)"
            r"\s+at\s+[`'\"]?(?P<target>[A-Za-z0-9_.:-]+)",
            re.IGNORECASE,
        )

        for clause in cls._request_clauses(user_request):
            for match in action_pattern.finditer(clause):
                subject = match.group("subject").lower()
                target_name = match.group("target").strip().rstrip(".,;:!?")
                prefix = clause[: match.start()].lower()

                if "camera" in subject:
                    source_type = "camera"
                elif "light" in subject:
                    source_type = "light"
                else:
                    camera_pos = prefix.rfind("camera")
                    light_pos = prefix.rfind("light")
                    if camera_pos < 0 and light_pos < 0:
                        continue
                    source_type = "camera" if camera_pos > light_pos else "light"

                source_name = cls._named_entity_from_request(clause, source_type)
                if not source_name:
                    source_name = cls._named_entity_from_request(user_request, source_type)

                tool = (
                    "aim_camera_at_object"
                    if source_type == "camera"
                    else "aim_light_at_object"
                )
                arguments = {"target_object_name": target_name}
                if source_name:
                    arguments[
                        "camera_name" if source_type == "camera" else "light_name"
                    ] = source_name

                key = (tool, tuple(sorted(arguments.items())))
                if key in seen:
                    continue
                seen.add(key)
                intents.append(
                    {
                        "tool": tool,
                        "source_type": source_type,
                        "source_name": source_name,
                        "target_name": target_name,
                        "arguments": arguments,
                        "clause": clause,
                    }
                )

        return intents


    @classmethod
    def _resolve_referential_request(cls, user_request, structured_memory):
        """
        Resolve bounded conversational references before discovery.

        We intentionally keep this deterministic and small. The LLM still
        chooses semantic actions, but it does not get to guess which scene
        entity "it" refers to when the controller already knows.
        """
        text = str(user_request or "").strip()
        lower = text.lower()
        memory = dict(structured_memory or {})

        result = {
            "resolved": False,
            "unresolved": False,
            "target_type": None,
            "target_name": None,
            "source": None,
            "intent": None,
            "scope": None,
            "same_turn": False,
            "planning_request": text,
        }

        if not cls._has_referential_phrase(text):
            return result

        # References whose antecedents are introduced in THIS request are not
        # cross-turn memory lookups. A single global referent would be wrong for
        # workflows that create a sphere, camera, and light and then use "it"
        # for different entities in different clauses. Keep them clause-local.
        if cls._has_same_turn_reference(text):
            result.update(
                {
                    "source": "same_turn_reference",
                    "intent": "same_turn_reference",
                    "scope": "same_turn",
                    "same_turn": True,
                    "planning_request": (
                        text
                        + "\nController note: Pronouns/references in this CURRENT "
                        "request may refer to entities introduced earlier in the "
                        "same request. Resolve them clause-locally from the nearest "
                        "explicit entity type/name. Do not require prior conversation "
                        "memory for those same-turn references."
                    ),
                }
            )
            return result

        # Explicit typed references take precedence.
        typed_patterns = (
            ("camera", r"\b(?:this|that|the)\s+camera\b"),
            ("light", r"\b(?:this|that|the)\s+light\b"),
            ("material", r"\b(?:this|that|the)\s+material\b"),
            ("object", r"\b(?:this|that|the)\s+(?:object|mesh)\b"),
        )

        target_type = None
        source = None

        for entity_type, pattern in typed_patterns:
            if re.search(pattern, lower):
                target_type = entity_type
                source = f"typed_reference:{entity_type}"
                break

        # Generic "it/this/that" uses semantic hints first, then last entity.
        if target_type is None and re.search(r"\b(?:it|this|that)\b", lower):
            if re.search(r"\b(brighter|dimmer|brightness|energy|intensity)\b", lower):
                target_type = "light"
                source = "generic_pronoun:light_intent"
            elif re.search(r"\b(lens|focal|active\s+camera|make\s+it\s+active)\b", lower):
                target_type = "camera"
                source = "generic_pronoun:camera_intent"
            elif cls._color_word(text) or re.search(
                r"\b(taller|shorter|wider|narrower|move|reposition|relocate|"
                r"shade|smooth|subdivide|solidify|inset|taper|scale|delete)\b",
                lower,
            ):
                target_type = "object"
                source = "generic_pronoun:object_intent"
            else:
                target_type = memory.get("last_entity_type")
                source = "generic_pronoun:last_entity"

        key_by_type = {
            "object": "last_object",
            "camera": "last_camera",
            "light": "last_light",
            "material": "last_material",
        }

        target_name = None
        if target_type in key_by_type:
            target_name = memory.get(key_by_type[target_type])

        if not target_name and target_type == memory.get("last_entity_type"):
            target_name = memory.get("last_entity_name")

        if not target_type or not target_name:
            result["unresolved"] = True
            result["scope"] = "cross_turn"
            result["source"] = source or "referential_phrase_without_memory"
            return result

        color = cls._color_word(text)
        if target_type == "object" and color:
            intent = "object_color"
            planning_note = (
                f"Controller-resolved target: {color} object '{target_name}'. "
                f"Create a dedicated {color} material for object '{target_name}' "
                f"and assign that material to '{target_name}'. "
                "Do not recolor an unrelated or shared material."
            )
        elif target_type == "light" and re.search(
            r"\b(brighter|dimmer|brightness|energy|intensity)\b",
            lower,
        ):
            intent = "light_energy"
            planning_note = (
                f"Controller-resolved target light: '{target_name}'."
            )
        elif target_type == "camera":
            intent = "camera_reference"
            planning_note = (
                f"Controller-resolved target camera: '{target_name}'."
            )
        elif target_type == "material":
            intent = "material_reference"
            planning_note = (
                f"Controller-resolved target material: '{target_name}'."
            )
        else:
            intent = "object_reference"
            planning_note = (
                f"Controller-resolved target object: '{target_name}'."
            )

        result.update(
            {
                "resolved": True,
                "target_type": target_type,
                "target_name": target_name,
                "source": source,
                "intent": intent,
                "scope": "cross_turn",
                "planning_request": text + "\n" + planning_note,
            }
        )

        return result


    @staticmethod
    def _reference_context_message(reference_resolution):
        resolution = reference_resolution or {}

        if resolution.get("same_turn"):
            return [
                {
                    "role": "system",
                    "content": (
                        "The CURRENT request introduces one or more entities and "
                        "then refers to them with words such as 'it'. Resolve those "
                        "references clause-locally inside the current request. Nearby "
                        "explicit names/types in the current request take precedence "
                        "over prior conversation memory. Different clauses may use "
                        "'it' for different newly introduced entities."
                    ),
                }
            ]

        if not resolution.get("resolved"):
            return []

        return [
            {
                "role": "system",
                "content": (
                    "Controller-resolved reference for the CURRENT request: "
                    f"{resolution.get('target_type')} "
                    f"'{resolution.get('target_name')}'. "
                    "Use this exact entity when a tool argument refers to the "
                    "pronoun/reference in the current request. Do not substitute "
                    "a different scene entity."
                ),
            }
        ]


    @staticmethod
    def _structured_memory_update_from_step(memory, step):
        if not isinstance(step, dict) or step.get("status") != "success":
            return memory

        updated = dict(memory or {})
        tool = step.get("tool")
        args = step.get("arguments", {}) or {}
        result = step.get("result", {}) or {}

        def set_entity(entity_type, name):
            if not name:
                return
            updated["last_entity_type"] = entity_type
            updated["last_entity_name"] = str(name)
            updated[f"last_{entity_type}"] = str(name)

        if tool in {
            "create_cube",
            "create_uv_sphere",
            "create_cylinder",
            "create_cone",
            "create_plane",
            "create_torus",
        }:
            mesh = result.get("mesh", {}) or {}
            set_entity(
                "object",
                mesh.get("object_name")
                or result.get("name")
                or args.get("name"),
            )

        elif tool in {
            "move_object",
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
        }:
            set_entity(
                "object",
                result.get("object_name")
                or result.get("name")
                or args.get("object_name")
                or args.get("name"),
            )

        elif tool == "delete_object":
            name = args.get("name") or args.get("object_name")
            if name and updated.get("last_object") == name:
                updated.pop("last_object", None)
            if (
                name
                and updated.get("last_entity_type") == "object"
                and updated.get("last_entity_name") == name
            ):
                updated.pop("last_entity_type", None)
                updated.pop("last_entity_name", None)

        elif tool in {"create_material", "set_material_color"}:
            set_entity(
                "material",
                result.get("name")
                or args.get("material_name")
                or args.get("name"),
            )

        elif tool == "assign_material":
            obj = result.get("object_name") or args.get("object_name")
            mat = result.get("material_name") or args.get("material_name")
            if mat:
                updated["last_material"] = str(mat)
            set_entity("object", obj)

        elif tool in {
            "create_camera",
            "move_camera",
            "set_camera_lens",
            "set_active_camera",
            "aim_camera_at_object",
        }:
            camera = result.get("camera", {}) or {}
            set_entity(
                "camera",
                camera.get("name")
                or args.get("camera_name")
                or args.get("name"),
            )

        elif tool in {
            "create_light",
            "move_light",
            "set_light_energy",
            "set_light_color",
            "set_area_light_size",
            "aim_light_at_object",
        }:
            light = result.get("light", {}) or {}
            set_entity(
                "light",
                light.get("name")
                or args.get("light_name")
                or args.get("name"),
            )

        elif tool == "set_render_output":
            filename = (
                result.get("output_filename")
                or result.get("filename")
                or args.get("filename")
            )
            if filename:
                updated["last_render_output"] = str(filename)

        elif tool == "render_scene":
            if result.get("output_path"):
                updated["last_render_output"] = str(result.get("output_path"))

        return updated


    @classmethod
    def _updated_structured_memory(cls, prior_memory, trace):
        memory = dict(prior_memory or {})
        for step in trace.get("steps", []):
            memory = cls._structured_memory_update_from_step(
                memory,
                step,
            )
        return memory


    @staticmethod
    def _memory_number_list(values):
        if not isinstance(values, (list, tuple)):
            return None
        try:
            return tuple(round(float(value), 4) for value in values)
        except Exception:
            return None


    @classmethod
    def _memory_fact_for_step(cls, step):
        """
        Convert verified/successful semantic actions into compact facts.
        Observers are intentionally omitted.
        """
        if not isinstance(step, dict):
            return None

        if step.get("status") != "success":
            return None

        tool = step.get("tool")
        args = step.get("arguments", {}) or {}
        result = step.get("result", {}) or {}

        if tool in OBSERVATION_TOOLS or tool == "search_blender_docs":
            return None

        if tool in {
            "create_cube",
            "create_uv_sphere",
            "create_cylinder",
            "create_cone",
            "create_plane",
            "create_torus",
        }:
            mesh = result.get("mesh", {}) or {}
            name = mesh.get("object_name") or result.get("name") or args.get("name")
            location = mesh.get("location") or result.get("location")
            if name:
                if location:
                    return f"Object '{name}' exists at {cls._memory_number_list(location)}."
                return f"Object '{name}' exists."

        if tool == "move_object":
            name = result.get("name") or args.get("name")
            location = result.get("location")
            if name and location:
                return f"Object '{name}' is at {cls._memory_number_list(location)}."

        if tool == "delete_object":
            name = args.get("name") or args.get("object_name")
            if name:
                return f"Object '{name}' was deleted."

        if tool in {"create_material", "set_material_color"}:
            name = result.get("name") or args.get("name")
            color = result.get("base_color")
            if name:
                if color:
                    return f"Material '{name}' exists with base color {cls._memory_number_list(color)}."
                return f"Material '{name}' exists."

        if tool == "assign_material":
            obj = result.get("object_name") or args.get("object_name")
            mat = result.get("material_name") or args.get("material_name")
            if obj and mat:
                return f"Object '{obj}' uses material '{mat}'."

        if tool == "shade_smooth":
            obj = result.get("object_name") or args.get("object_name")
            if obj:
                return f"Mesh '{obj}' uses smooth shading."

        if tool == "create_camera":
            camera = result.get("camera", {}) or {}
            name = camera.get("name") or args.get("name")
            location = camera.get("location")
            if name:
                if location:
                    return f"Camera '{name}' exists at {cls._memory_number_list(location)}."
                return f"Camera '{name}' exists."

        if tool == "move_camera":
            name = args.get("camera_name")
            camera = result.get("camera", {}) or {}
            location = camera.get("location")
            if name and location:
                return f"Camera '{name}' is at {cls._memory_number_list(location)}."

        if tool == "set_camera_lens":
            name = args.get("camera_name")
            lens = args.get("lens_mm")
            if name and lens is not None:
                return f"Camera '{name}' lens is {float(lens):g} mm."

        if tool == "set_active_camera":
            name = args.get("camera_name")
            if name:
                return f"Active camera is '{name}'."

        if tool == "aim_camera_at_object":
            camera = args.get("camera_name")
            target = args.get("target_object_name")
            if camera and target:
                return f"Camera '{camera}' is aimed at '{target}'."

        if tool == "create_light":
            light = result.get("light", {}) or {}
            name = light.get("name") or args.get("name")
            location = light.get("location")
            light_type = light.get("type") or args.get("light_type")
            if name:
                suffix = f" ({light_type})" if light_type else ""
                if location:
                    return f"Light '{name}'{suffix} exists at {cls._memory_number_list(location)}."
                return f"Light '{name}'{suffix} exists."

        if tool == "move_light":
            name = args.get("light_name")
            light = result.get("light", {}) or {}
            location = light.get("location")
            if name and location:
                return f"Light '{name}' is at {cls._memory_number_list(location)}."

        if tool == "set_light_energy":
            name = args.get("light_name")
            energy = args.get("energy")
            if name and energy is not None:
                return f"Light '{name}' energy is {float(energy):g}."

        if tool == "aim_light_at_object":
            light = args.get("light_name")
            target = args.get("target_object_name")
            if light and target:
                return f"Light '{light}' is aimed at '{target}'."

        if tool == "set_render_output":
            filename = (
                result.get("output_filename")
                or result.get("filename")
                or args.get("filename")
            )
            if filename:
                return f"Render output is configured as '{filename}'."

        if tool == "render_scene":
            output = result.get("output_path")
            if result.get("saved_to_file") is True and output:
                return f"Last verified render was saved to '{output}'."
            return "The scene was rendered successfully."

        if tool in {
            "inset_top_face",
            "subdivide_mesh",
            "translate_top_face",
            "scale_top_face",
            "merge_by_distance",
            "solidify_mesh",
            "scale_mesh_geometry",
            "extrude_top_face",
            "bevel_mesh_edges",
            "recalculate_normals",
        }:
            obj = result.get("object_name") or args.get("object_name")
            if obj:
                return f"Mesh '{obj}' was successfully edited with {tool}."

        if tool in {
            "add_bevel_modifier",
            "set_bevel_modifier",
            "add_subdivision_modifier",
            "set_subdivision_modifier",
            "remove_modifier",
            "apply_modifier",
        }:
            obj = args.get("object_name")
            modifier = args.get("modifier_name") or args.get("name")
            if obj:
                if modifier:
                    return f"Modifier '{modifier}' on '{obj}' was updated by {tool}."
                return f"Modifiers on '{obj}' were updated by {tool}."

        return None


    @classmethod
    def _updated_memory_summary(cls, prior_summary, trace):
        prior_lines = [
            line.strip()
            for line in str(prior_summary or "").splitlines()
            if line.strip()
        ]

        new_facts = []
        for step in trace.get("steps", []):
            fact = cls._memory_fact_for_step(step)
            if fact:
                new_facts.append(fact)

        # De-duplicate exact facts while keeping the newest occurrence.
        combined = prior_lines + new_facts
        deduped_reversed = []
        seen = set()

        for line in reversed(combined):
            if line in seen:
                continue
            seen.add(line)
            deduped_reversed.append(line)

        lines = list(reversed(deduped_reversed))[-HISTORY_MEMORY_MAX_LINES:]

        while lines and len("\\n".join(lines)) > HISTORY_MEMORY_MAX_CHARS:
            lines.pop(0)

        return "\\n".join(lines)


    @staticmethod
    def _contains_any(text, words):
        return any(
            word in text
            for word in words
        )

    def _select_tool_categories(
        self,
        user_request,
        required_verification_tool=None,
    ):
        """
        Deterministic capability gating.

        The model no longer receives the entire Blender tool catalog on every
        request. This keeps the Ollama prompt small and makes tool selection
        more reliable as the Copilot grows.
        """
        text = (
            str(user_request)
            .lower()
        )

        categories = {
            "knowledge",
        }

        # ----------------------------------------------------------
        # Objects / scene
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "cube",
                "object",
                "scene object",
                "move object",
                "move the object",
                "delete ",
                "create cube",
                "position",
                "location",
                "rename",
                "mesh object",
                "sphere",
                "primitive",
            ),
        ):
            categories.add(
                "objects"
            )

        # ----------------------------------------------------------
        # Materials / appearance
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "material",
                "color",
                "colour",
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "metallic",
                "roughness",
                "shader",
                "paint",
            ),
        ):
            categories.add(
                "materials"
            )

        # ----------------------------------------------------------
        # Modifiers
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "modifier",
                "bevel modifier",
                "subdivision modifier",
                "subsurf",
                "rounded edge",
                "apply modifier",
            ),
        ):
            categories.add(
                "modifiers"
            )

        # ----------------------------------------------------------
        # Mesh modeling
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "mesh",
                "geometry",
                "sphere",
                "uv sphere",
                "cylinder",
                "cone",
                "plane",
                "torus",
                "primitive",
                "top face",
                "extrude",
                "recalculate normals",
                "recalculate normal",
                "normals",
                "shade smooth",
                "smooth shading",
                "mesh bevel",
                "bevel edges",
                "scale geometry",
                "topology",
                "vertices",
                "edges",
                "faces",
                "model ",
                "modeling",
                "modelling",
                "inset",
                "subdivide",
                "merge by distance",
                "remove doubles",
                "weld vertices",
                "solidify",
                "thickness",
                "shell",
                "taper",
                "top region",
                "move top face",
                "scale top face",
                "taller",
                "shorter",
                "wider",
                "narrower",
                "thinner",
                "thicker",
            ),
        ):
            categories.add(
                "mesh_modeling"
            )

        # ----------------------------------------------------------
        # Camera
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "camera",
                "lens",
                "focal",
                "frame ",
                "viewpoint",
                "active camera",
            ),
        ):
            categories.add(
                "cameras"
            )

        # ----------------------------------------------------------
        # Lights
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "light",
                "lighting",
                "illumination",
                "area light",
                "point light",
                "spot light",
                "sun light",
                "studio light",
                "soft light",
                "energy",
                "brighter",
                "dimmer",
                "intensity",
            ),
        ):
            categories.add(
                "lights"
            )

        # ----------------------------------------------------------
        # Rendering. Rendering often needs camera awareness too.
        # ----------------------------------------------------------
        if self._contains_any(
            text,
            (
                "render",
                "resolution",
                "samples",
                "cycles",
                "eevee",
                "workbench",
                "png",
                "jpeg",
                "jpg",
                "exr",
                "output file",
                "transparent background",
                "1080p",
                "4k",
            ),
        ):
            categories.add(
                "rendering"
            )

            categories.add(
                "cameras"
            )

        # Explicit documentation request.
        if self._contains_any(
            text,
            (
                "documentation",
                "official blender",
                "blender api",
                "manual",
                "docs",
            ),
        ):
            categories.add(
                "knowledge"
            )

        # ----------------------------------------------------------
        # A required verification tool must always be visible on the
        # next LLM turn, regardless of keyword matching.
        # ----------------------------------------------------------
        if required_verification_tool:
            for category_name, tool_names in TOOL_CATEGORIES.items():
                if required_verification_tool in tool_names:
                    categories.add(
                        category_name
                    )

        # ----------------------------------------------------------
        # Safe fallback for generic requests.
        # ----------------------------------------------------------
        if categories == {
            "knowledge"
        }:
            categories.add(
                "objects"
            )

        return categories

    @staticmethod
    def _normalized_discovery_text(text):
        value = str(text).lower().replace("×", "x")
        value = value.replace("’", "'")
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _is_read_only_request(user_request):
        text = str(user_request).strip().lower()

        if re.match(r"^(how\s+(?:do|can|should)\s+i|how\s+to)\b", text):
            return True

        if re.match(
            r"^(what|which|why|when|where|who|inspect|list|show|tell\s+me|describe|report)\b",
            text,
        ):
            return True

        mutation = re.search(
            r"\b(create|add|make|move|delete|remove|set|change|assign|apply|"
            r"inset|subdivide|merge|solidify|extrude|bevel|taper|scale|aim|"
            r"point|render|shade|recalculate|translate|offset|raise)\b",
            text,
        )

        return mutation is None

    @classmethod
    def _extract_goal_ledger(cls, user_request):
        text = cls._normalized_discovery_text(user_request)
        goals = []
        seen = set()

        def add(tool, description, arguments=None):
            if tool in seen:
                if arguments:
                    for goal in goals:
                        if goal.get("tool") == tool and not goal.get("arguments"):
                            goal["arguments"] = dict(arguments)
                            break
                return
            seen.add(tool)
            goal = {
                "id": f"goal_{len(goals)+1}",
                "tool": tool,
                "description": description,
                "status": "pending",
                "satisfied_by": None,
            }
            if arguments:
                goal["arguments"] = dict(arguments)
            goals.append(goal)

        patterns = [
            ("create_cube", r"\b(?:create|add|make)\b[^.;\n]{0,30}\bcube\b", "Create cube"),
            ("create_uv_sphere", r"\b(?:create|add|make)\b[^.;\n]{0,30}\b(?:uv\s+)?sphere\b", "Create sphere"),
            ("create_cylinder", r"\b(?:create|add|make)\b[^.;\n]{0,30}\bcylinder\b", "Create cylinder"),
            ("create_cone", r"\b(?:create|add|make)\b[^.;\n]{0,30}\bcone\b", "Create cone"),
            ("create_plane", r"\b(?:create|add|make)\b[^.;\n]{0,30}\bplane\b", "Create plane"),
            ("create_torus", r"\b(?:create|add|make)\b[^.;\n]{0,30}\b(?:torus|donut|doughnut)\b", "Create torus"),
            ("move_object", r"\b(?:move|reposition|relocate)\b[^.;\n]{0,25}\bobject\b", "Move object"),
            ("delete_object", r"\b(?:delete|remove)\b[^.;\n]{0,25}\bobject\b", "Delete object"),
            ("create_material", r"\b(?:create|add|make)\b[^.;\n]{0,30}\bmaterial\b", "Create material"),
            ("assign_material", r"\b(?:assign|apply)\b[^.;\n]{0,35}\bmaterial\b", "Assign material"),
            ("shade_smooth", r"\bshade\b[^.;\n]{0,25}\bsmooth\b|\bsmooth\s+shading\b", "Shade mesh smooth"),
            ("add_bevel_modifier", r"\badd\b[^.;\n]{0,25}\bbevel\s+modifier\b", "Add bevel modifier"),
            ("add_subdivision_modifier", r"\badd\b[^.;\n]{0,25}\b(?:subdivision|subsurf)\b", "Add subdivision modifier"),
            ("apply_modifier", r"\bapply\b[^.;\n]{0,30}\bmodifier\b", "Apply modifier"),
            ("create_camera", r"\b(?:create|add|make)\b[^.;\n]{0,35}\bcamera\b", "Create camera"),
            ("set_active_camera", r"\b(?:make|set)\b[^.;\n]{0,35}\b(?:camera|it)\b[^.;\n]{0,20}\bactive\b|\bactive\s+camera\b", "Make camera active"),
            ("create_light", r"\b(?:create|add|make|place)\b[^.;\n]{0,40}\b(?:area|point|spot|sun)?\s*light\b", "Create light"),
            ("set_render_resolution", r"\b(?:set|change)\b[^.;\n]{0,30}\brender\s+resolution\b", "Set render resolution"),
            ("set_render_engine", r"\b(?:set|change)\b[^.;\n]{0,30}\brender\s+engine\b", "Set render engine"),
            ("set_render_samples", r"\b(?:set|change)\b[^.;\n]{0,30}\brender\s+samples\b", "Set render samples"),
            ("set_render_transparent", r"\btransparent\s+(?:background|render)\b", "Set transparent render"),
            ("inset_top_face", r"\binset\b[^.;\n]{0,35}\btop\s+face\b", "Inset top face"),
            ("subdivide_mesh", r"\bsubdivide\b[^.;\n]{0,35}\b(?:mesh|object|it)?\b", "Subdivide mesh"),
            ("translate_top_face", r"\b(?:move|translate|offset|raise)\b[^.;\n]{0,45}\btop\s+face\b|\braise\s+the\s+top\b", "Translate top face"),
            ("scale_top_face", r"\b(?:scale|scaling)\b[^.;\n]{0,40}\btop\s+face\b|\b(?:taper|narrow|widen)\b[^.;\n]{0,50}\btop\b", "Scale/taper top face"),
            ("merge_by_distance", r"\b(?:merge\s+(?:vertices\s+)?by\s+distance|remove\s+doubles|weld\s+vertices)\b", "Merge vertices by distance"),
            ("solidify_mesh", r"\bsolidify\b|\bgive\b[^.;\n]{0,35}\bthickness\b|\bshell\s+thickness\b", "Solidify mesh"),
            ("extrude_top_face", r"\bextrude\b[^.;\n]{0,35}\btop\b|\braise\s+the\s+top\b", "Extrude top face"),
        ]

        for tool, pattern, description in patterns:
            if re.search(pattern, text):
                add(tool, description)

        # Decimal-safe, clause-local camera/light aim extraction. Store exact
        # source/target arguments in the controller-owned goal when available,
        # so completion does not depend on Qwen recovering the pronoun later.
        aim_intents = cls._clause_local_aim_intents(user_request)
        for intent in aim_intents:
            if intent["tool"] == "aim_camera_at_object":
                add(
                    "aim_camera_at_object",
                    "Aim camera at target",
                    arguments=intent.get("arguments"),
                )
            elif intent["tool"] == "aim_light_at_object":
                add(
                    "aim_light_at_object",
                    "Aim light at target",
                    arguments=intent.get("arguments"),
                )

        # Keep explicit non-pronoun forms as a fail-safe goal signal even when
        # names cannot be recovered deterministically.
        if not any(i["tool"] == "aim_camera_at_object" for i in aim_intents):
            if re.search(r"\b(?:aim|point)\s+(?:the\s+)?camera\s+at\b", text):
                add("aim_camera_at_object", "Aim camera at target")
        if not any(i["tool"] == "aim_light_at_object" for i in aim_intents):
            if re.search(r"\b(?:aim|point)\s+(?:the\s+)?light\s+at\b", text):
                add("aim_light_at_object", "Aim light at target")

        # "Create a material ... and assign it to Object" is a common natural
        # language form where the noun material is not repeated after assign.
        if "material" in text and re.search(
            r"\b(?:assign|apply)\s+it\s+to\b",
            text,
        ):
            add("assign_material", "Assign material")

        # An adjective-colored primitive/object implies material creation + assignment.
        if re.search(
            r"\b(red|blue|green|yellow|white|black|gray|grey|orange|purple)\b"
            r"[^.;\n]{0,35}\b(cube|sphere|cylinder|cone|plane|torus|object)\b",
            text,
        ):
            add("create_material", "Create requested colored material")
            add("assign_material", "Assign requested colored material")

        filename = cls._explicit_render_filename(user_request)
        if filename:
            add("set_render_output", f"Set render output to {filename}")

        if cls._render_action_requested(user_request):
            add("render_scene", "Render scene")

        # A controller-resolved object-color planning note is deliberately
        # phrased as "<color> object" plus "create ... material / assign ...".
        # The ordinary patterns above therefore add create_material and
        # assign_material without adding a new monolithic tool.

        return goals

    @classmethod
    def _goal_tool_names_for_request(cls, user_request):
        return {
            goal["tool"]
            for goal in cls._extract_goal_ledger(user_request)
        }

    @staticmethod
    def _goal_trace_snapshot(state):
        return [dict(goal) for goal in state.goal_ledger]

    def _sync_goal_trace(self, state):
        state.trace["goal_ledger"] = self._goal_trace_snapshot(state)

    def _pending_goals(self, state, include_render=True):
        pending = [
            goal
            for goal in state.goal_ledger
            if goal.get("status") != "satisfied"
        ]

        if include_render:
            return pending

        return [
            goal
            for goal in pending
            if goal.get("tool") not in {"render_scene", "set_render_output"}
        ]

    def _deterministic_simple_goal_command(self, state):
        """
        Inject a semantic tool when its exact arguments and prerequisites are
        controller-known. Single simple goals preserve the prior behavior;
        camera/light aim goals may also be completed inside larger workflows
        once their source entities and newly-created object targets are ready.
        """
        pending = self._pending_goals(
            state,
            include_render=not state.render_required,
        )

        if state.render_required:
            pending = [
                goal
                for goal in pending
                if goal.get("tool") not in {"render_scene", "set_render_output"}
            ]

        if not pending:
            return None

        request = state.user_request
        reference = state.reference_resolution or {}
        satisfied_tools = {
            goal.get("tool")
            for goal in state.goal_ledger
            if goal.get("status") == "satisfied"
        }
        ledger_tools = {goal.get("tool") for goal in state.goal_ledger}
        primitive_create_tools = {
            "create_cube",
            "create_uv_sphere",
            "create_cylinder",
            "create_cone",
            "create_plane",
            "create_torus",
        }

        def source_goal_ready(source_tool):
            return source_tool not in ledger_tools or source_tool in satisfied_tools

        def target_object_ready():
            requested_object_creates = ledger_tools & primitive_create_tools
            return not requested_object_creates or requested_object_creates <= satisfied_tools

        # In a multi-goal workflow, only inject clause-local aim operations and
        # only after their source and target creation prerequisites are done.
        if len(pending) > 1:
            for goal in pending:
                tool = goal.get("tool")
                args = dict(goal.get("arguments") or {})
                if tool == "aim_camera_at_object":
                    if not source_goal_ready("create_camera") or not target_object_ready():
                        continue
                    camera_name = args.get("camera_name") or self._named_entity_from_request(request, "camera")
                    target_name = args.get("target_object_name") or self._aim_target_from_request(request)
                    if camera_name and target_name:
                        return {
                            "tool": tool,
                            "arguments": {
                                "camera_name": camera_name,
                                "target_object_name": target_name,
                            },
                        }
                elif tool == "aim_light_at_object":
                    if not source_goal_ready("create_light") or not target_object_ready():
                        continue
                    light_name = args.get("light_name") or self._named_entity_from_request(request, "light")
                    target_name = args.get("target_object_name") or self._aim_target_from_request(request)
                    if light_name and target_name:
                        return {
                            "tool": tool,
                            "arguments": {
                                "light_name": light_name,
                                "target_object_name": target_name,
                            },
                        }
            return None

        goal = pending[0]
        tool = goal.get("tool")
        goal_args = dict(goal.get("arguments") or {})

        object_name = None
        if reference.get("resolved") and reference.get("target_type") == "object":
            object_name = reference.get("target_name")
        if not object_name:
            object_name = self._named_entity_from_request(request, "object")

        camera_name = goal_args.get("camera_name") or self._named_entity_from_request(request, "camera")
        light_name = goal_args.get("light_name") or self._named_entity_from_request(request, "light")
        target_name = (
            goal_args.get("target_object_name")
            or self._aim_target_from_request(request)
            or object_name
        )

        if tool == "shade_smooth" and object_name:
            return {
                "tool": "shade_smooth",
                "arguments": {
                    "object_name": object_name,
                    "enabled": True,
                },
            }

        if tool == "set_active_camera" and camera_name:
            return {
                "tool": "set_active_camera",
                "arguments": {"camera_name": camera_name},
            }

        if tool == "aim_camera_at_object" and camera_name and target_name:
            if source_goal_ready("create_camera") and target_object_ready():
                return {
                    "tool": "aim_camera_at_object",
                    "arguments": {
                        "camera_name": camera_name,
                        "target_object_name": target_name,
                    },
                }

        if tool == "aim_light_at_object" and light_name and target_name:
            if source_goal_ready("create_light") and target_object_ready():
                return {
                    "tool": "aim_light_at_object",
                    "arguments": {
                        "light_name": light_name,
                        "target_object_name": target_name,
                    },
                }

        return None


    @staticmethod
    def _controller_injected_llm_result(tool_name):
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": []},
            "latency": 0.0,
            "tool_categories": ["controller"],
            "tool_count": 0,
            "tool_names": [tool_name],
            "tool_discovery": {
                "mode": "deterministic_simple_goal_injection",
                "selected_tool_names": [tool_name],
                "goal_tool_names": [tool_name],
            },
            "num_ctx": None,
            "attempted_contexts": [],
            "prompt_eval_count": 0,
            "eval_count": 0,
        }


    def _mark_goal_tool_success(self, state, tool_name, tool_result, satisfied_by=None):
        if not isinstance(tool_result, dict) or tool_result.get("success") is not True:
            return

        changed = False

        for goal in state.goal_ledger:
            if goal.get("status") == "satisfied":
                continue

            if goal.get("tool") == tool_name:
                goal["status"] = "satisfied"
                goal["satisfied_by"] = satisfied_by or tool_name
                changed = True

        # If create_camera observed that the requested camera is already active,
        # it legitimately satisfies an explicit active-camera goal.
        if tool_name == "create_camera":
            camera = tool_result.get("camera", {})
            if camera.get("active_scene_camera") is True:
                for goal in state.goal_ledger:
                    if goal.get("tool") == "set_active_camera" and goal.get("status") != "satisfied":
                        goal["status"] = "satisfied"
                        goal["satisfied_by"] = "create_camera:observed_active"
                        changed = True

        if changed:
            state.goal_repair_attempts = 0
            self._sync_goal_trace(state)
            state.trace["controller_events"].append(
                {
                    "step": state.step,
                    "event": "goal_progress",
                    "tool": tool_name,
                    "pending_goals": [
                        goal["tool"]
                        for goal in self._pending_goals(state)
                    ],
                }
            )

    def _tool_matches_request(self, tool_name, text):
        if tool_name in {"aim_camera_at_object", "aim_light_at_object"}:
            if any(
                intent.get("tool") == tool_name
                for intent in self._clause_local_aim_intents(text)
            ):
                return True

        special = {
            "move_object": r"\b(?:move|reposition|relocate)\b[^.;\n]{0,25}\bobject\b",
            "inset_top_face": r"\binset\b[^.;\n]{0,35}\btop\s+face\b",
            "translate_top_face": r"\b(?:move|translate|offset|raise)\b[^.;\n]{0,45}\btop\s+face\b|\braise\s+the\s+top\b",
            "scale_top_face": r"\b(?:scale|scaling)\b[^.;\n]{0,40}\btop\s+face\b|\b(?:taper|narrow|widen)\b[^.;\n]{0,50}\btop\b",
            "aim_camera_at_object": r"\b(?:aim|point)\s+(?:the\s+)?camera\s+at\b",
            "aim_light_at_object": r"\b(?:aim|point)\s+(?:the\s+)?light\s+at\b",
            "render_scene": r"\brender\b(?:\s+the\s+scene|\s+scene|\s+it|\s+to\b|\s+the\s+image|\s*$)|\band\s+render\b",
        }

        pattern = special.get(tool_name)
        if pattern:
            return re.search(pattern, text) is not None

        hints = TOOL_DISCOVERY_HINTS.get(tool_name, ())
        return any(
            self._normalized_discovery_text(hint) in text
            for hint in hints
        )

    @staticmethod
    def _directional_top_face_offset(user_request, arguments):
        """
        Normalize natural directional language into Blender world axes.

        Explicit axis language wins. Otherwise:
          upward / raise / higher -> +Z
          downward / lower        -> -Z

        We intentionally do not guess front/back because that can be
        viewpoint-dependent.
        """
        text = str(user_request).lower().replace("×", "x")

        if not re.search(r"\btop\s+face\b|\bthe\s+top\b", text):
            return None

        direction = None
        sign = 1.0

        if re.search(r"\b(?:upward|upwards|higher)\b|\braise\b", text):
            direction = "up"
            sign = 1.0
        elif re.search(r"\b(?:downward|downwards|lower)\b", text):
            direction = "down"
            sign = -1.0

        if direction is None:
            return None

        number = r"(-?\d+(?:\.\d+)?)"
        match = re.search(
            rf"\b(?:upward|upwards|higher|downward|downwards|lower)\b\s*(?:by\s+)?{number}",
            text,
        )
        if match is None:
            match = re.search(
                rf"\b(?:raise|lower)\b[^.;\n]{{0,45}}\bby\s+{number}",
                text,
            )

        if match is not None:
            magnitude = abs(float(match.group(1)))
        else:
            existing = [
                abs(float(arguments.get("x_offset", 0.0))),
                abs(float(arguments.get("y_offset", 0.0))),
                abs(float(arguments.get("z_offset", 0.0))),
            ]
            magnitude = max(existing)

        if magnitude <= 0.0:
            return None

        return {
            "x_offset": 0.0,
            "y_offset": 0.0,
            "z_offset": sign * magnitude,
            "direction": direction,
            "magnitude": magnitude,
        }


    def _normalize_command_semantics(self, state, command):
        normalized = {
            **command,
            "arguments": dict(command.get("arguments", {})),
        }

        reference = state.reference_resolution or {}
        if reference.get("resolved"):
            target_type = reference.get("target_type")
            target_name = reference.get("target_name")
            tool = normalized.get("tool")
            args = normalized["arguments"]

            object_name_tools = {
                "move_object": "name",
                "delete_object": "name",
                "assign_material": "object_name",
                "add_bevel_modifier": "object_name",
                "set_bevel_modifier": "object_name",
                "add_subdivision_modifier": "object_name",
                "set_subdivision_modifier": "object_name",
                "remove_modifier": "object_name",
                "apply_modifier": "object_name",
                "shade_smooth": "object_name",
                "recalculate_normals": "object_name",
                "scale_mesh_geometry": "object_name",
                "extrude_top_face": "object_name",
                "bevel_mesh_edges": "object_name",
                "get_mesh_regions": "object_name",
                "inset_top_face": "object_name",
                "subdivide_mesh": "object_name",
                "translate_top_face": "object_name",
                "scale_top_face": "object_name",
                "merge_by_distance": "object_name",
                "solidify_mesh": "object_name",
            }

            if target_type == "object" and tool in object_name_tools:
                field = object_name_tools[tool]
                before = args.get(field)
                if before != target_name:
                    args[field] = target_name
                    self._record_trace_event(
                        state.trace,
                        "reference_argument_enforced",
                        step=state.step,
                        tool=tool,
                        field=field,
                        before=before,
                        after=target_name,
                        target_type="object",
                    )

            elif target_type == "camera" and tool in {
                "move_camera",
                "set_camera_lens",
                "set_active_camera",
                "aim_camera_at_object",
            }:
                before = args.get("camera_name")
                if before != target_name:
                    args["camera_name"] = target_name
                    self._record_trace_event(
                        state.trace,
                        "reference_argument_enforced",
                        step=state.step,
                        tool=tool,
                        field="camera_name",
                        before=before,
                        after=target_name,
                        target_type="camera",
                    )

            elif target_type == "light" and tool in {
                "move_light",
                "set_light_energy",
                "set_light_color",
                "set_area_light_size",
                "aim_light_at_object",
            }:
                before = args.get("light_name")
                if before != target_name:
                    args["light_name"] = target_name
                    self._record_trace_event(
                        state.trace,
                        "reference_argument_enforced",
                        step=state.step,
                        tool=tool,
                        field="light_name",
                        before=before,
                        after=target_name,
                        target_type="light",
                    )

            elif target_type == "material" and tool == "set_material_color":
                before = args.get("material_name")
                if before != target_name:
                    args["material_name"] = target_name
                    self._record_trace_event(
                        state.trace,
                        "reference_argument_enforced",
                        step=state.step,
                        tool=tool,
                        field="material_name",
                        before=before,
                        after=target_name,
                        target_type="material",
                    )

        if normalized.get("tool") == "translate_top_face":
            corrected = self._directional_top_face_offset(
                state.user_request,
                normalized["arguments"],
            )
            if corrected is not None:
                before = dict(normalized["arguments"])
                normalized["arguments"].update(
                    {
                        "x_offset": corrected["x_offset"],
                        "y_offset": corrected["y_offset"],
                        "z_offset": corrected["z_offset"],
                    }
                )
                self._record_trace_event(
                    state.trace,
                    "semantic_arguments_normalized",
                    step=state.step,
                    tool="translate_top_face",
                    before=before,
                    after=normalized["arguments"],
                    rule=f"direction:{corrected['direction']}=>world_z",
                )

        return normalized


    def _discover_tool_names(self, user_request, categories, required_verification_tool=None):
        text = self._normalized_discovery_text(user_request)
        read_only = self._is_read_only_request(user_request)
        goal_tools = self._goal_tool_names_for_request(user_request)

        candidate_names = set()
        for category in categories:
            candidate_names.update(TOOL_CATEGORIES[category])

        selected_names = set()
        reasons = {}
        matched_by_category = {category: set() for category in categories}

        explicit_docs = self._contains_any(
            text,
            ("documentation", "official blender", "blender api", "manual", "docs"),
        )

        # Core observation tools remain available, but do not automatically add
        # documentation search to every ordinary request.
        for category in categories:
            for name in CATEGORY_CORE_TOOLS.get(category, set()):
                if name not in candidate_names:
                    continue
                if category == "knowledge" and not explicit_docs and len(categories) > 1:
                    continue
                if (
                    name == "get_mesh_regions"
                    and not self._contains_any(
                        text,
                        ("top face", "top region", "upward", "downward", "side face", "mesh region", "inset", "taper", "translate top", "move the top", "scale the top"),
                    )
                ):
                    continue
                selected_names.add(name)
                reasons.setdefault(name, []).append(f"core:{category}")

        for name in candidate_names:
            if name in MUTATING_TOOLS and read_only:
                continue
            if not self._tool_matches_request(name, text):
                continue

            selected_names.add(name)
            reasons.setdefault(name, []).append("semantic_match")
            for category in categories:
                if name in TOOL_CATEGORIES[category]:
                    matched_by_category[category].add(name)

        # Explicit controller goals always win over fuzzy discovery. This gives
        # discovery recall=1 for supported goals extracted from the request.
        for name in goal_tools:
            if name in candidate_names:
                selected_names.add(name)
                reasons.setdefault(name, []).append("goal_ledger")
                for category in categories:
                    if name in TOOL_CATEGORIES[category]:
                        matched_by_category[category].add(name)

        # Fallback action bundles are only used for mutation requests where a
        # category is clearly relevant but no action in that category matched.
        if not read_only:
            nontrivial = set(categories) - {"knowledge", "objects"}
            for category in categories:
                matched_actions = {
                    name
                    for name in matched_by_category.get(category, set())
                    if name not in CATEGORY_CORE_TOOLS.get(category, set())
                }
                if matched_actions:
                    continue
                if category == "knowledge":
                    continue
                if category == "objects" and nontrivial:
                    continue
                if category == "cameras" and not self._contains_any(
                    text,
                    ("camera", "lens", "focal", "viewpoint", "frame "),
                ):
                    continue
                for name in CATEGORY_FALLBACK_TOOLS.get(category, set()):
                    if name in candidate_names:
                        selected_names.add(name)
                        reasons.setdefault(name, []).append(f"fallback:{category}")

        mesh_actions = selected_names & (
            TOOL_CATEGORIES["mesh_modeling"] - {"get_mesh_info", "get_mesh_regions"}
        )
        if mesh_actions:
            selected_names.add("get_mesh_info")
            reasons.setdefault("get_mesh_info", []).append("dependency:mesh_verification")

        if selected_names & {"extrude_top_face", "inset_top_face", "translate_top_face", "scale_top_face"}:
            selected_names.add("get_mesh_regions")
            reasons.setdefault("get_mesh_regions", []).append("dependency:top_face_semantics")

        if selected_names & {"create_uv_sphere", "create_cylinder", "create_cone", "create_plane", "create_torus"}:
            selected_names.add("get_scene_objects")
            reasons.setdefault("get_scene_objects", []).append("dependency:primitive_verification")

        if "render_scene" in selected_names:
            selected_names.update({"get_render_settings", "get_cameras"})
            reasons.setdefault("get_render_settings", []).append("dependency:render")
            reasons.setdefault("get_cameras", []).append("dependency:render")

        if required_verification_tool:
            selected_names.add(required_verification_tool)
            reasons.setdefault(required_verification_tool, []).append("dependency:required_verifier")

        allow = {tool["function"]["name"] for tool in OLLAMA_TOOLS}
        selected_names &= allow

        return {
            "mode": "goal_aware_tool_discovery",
            "read_only": bool(read_only),
            "candidate_tool_count": len(candidate_names),
            "selected_tool_names": sorted(selected_names),
            "goal_tool_names": sorted(goal_tools),
            "reasons": reasons,
        }

    def _tools_for_request(self, user_request, required_verification_tool=None):
        categories = self._select_tool_categories(
            user_request,
            required_verification_tool=required_verification_tool,
        )
        discovery = self._discover_tool_names(
            user_request,
            categories,
            required_verification_tool=required_verification_tool,
        )
        names = set(discovery["selected_tool_names"])
        selected = [
            tool
            for tool in OLLAMA_TOOLS
            if tool["function"]["name"] in names
        ]

        normalized_request = self._normalized_discovery_text(user_request)
        resolved_object_color = (
            "controller-resolved target:" in normalized_request
            and "dedicated" in normalized_request
            and "material" in normalized_request
            and "object '" in normalized_request
        )

        if resolved_object_color:
            selected = [
                tool
                for tool in selected
                if tool["function"]["name"] != "set_material_color"
            ]
            discovery["selected_tool_names"] = [
                name
                for name in discovery["selected_tool_names"]
                if name != "set_material_color"
            ]
            discovery.setdefault("reasons", {}).setdefault(
                "set_material_color",
                [],
            ).append(
                "suppressed:resolved_object_color_uses_dedicated_material"
            )

        if (
            self._render_action_requested(user_request)
            and self._render_save_requested(user_request)
            and self._explicit_render_filename(user_request) is None
        ):
            selected = [
                tool
                for tool in selected
                if tool["function"]["name"] != "set_render_output"
            ]
            discovery["selected_tool_names"] = [
                name
                for name in discovery["selected_tool_names"]
                if name != "set_render_output"
            ]
            discovery.setdefault("reasons", {}).setdefault("set_render_output", []).append(
                "suppressed:preserve_configured_output"
            )

        return selected, sorted(categories), discovery

    @staticmethod
    def _is_context_size_error(response_text):
        text = str(response_text).lower()

        return any(
            marker in text
            for marker in (
                "exceeded the available context size",
                "exceed_context_size_error",
                "context size",
                "context length",
                "n_ctx",
            )
        )

    def _call_llm(
        self,
        messages,
        user_request,
        required_verification_tool=None,
    ):
        started = time.perf_counter()

        selected_tools, categories, discovery = (
            self._tools_for_request(
                user_request,
                required_verification_tool=
                    required_verification_tool,
            )
        )

        attempted_contexts = []
        current_num_ctx = self.ollama_num_ctx

        while True:
            attempted_contexts.append(
                current_num_ctx
            )

            payload = {
                "model": self.model,
                "messages": messages,
                "tools": selected_tools,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": current_num_ctx,
                },
            }

            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=300,
            )

            if response.ok:
                data = response.json()

                # Remember the larger successful context for the rest of
                # this backend session, so later agent turns do not need
                # to rediscover the same requirement.
                if current_num_ctx > self.ollama_num_ctx:
                    self.ollama_num_ctx = current_num_ctx

                return {
                    "message": data["message"],
                    "latency": (
                        time.perf_counter()
                        - started
                    ),
                    "tool_categories": categories,
                    "tool_count": len(
                        selected_tools
                    ),
                    "tool_names": [
                        tool["function"]["name"]
                        for tool in selected_tools
                    ],
                    "tool_discovery": discovery,
                    "num_ctx": current_num_ctx,
                    "attempted_contexts": attempted_contexts,
                    "prompt_eval_count": data.get(
                        "prompt_eval_count"
                    ),
                    "eval_count": data.get(
                        "eval_count"
                    ),
                }

            detail = (
                response.text
                .strip()
            )

            can_grow_context = (
                self._is_context_size_error(
                    detail
                )
                and current_num_ctx
                < self.ollama_max_num_ctx
            )

            if can_grow_context:
                next_num_ctx = min(
                    current_num_ctx * 2,
                    self.ollama_max_num_ctx,
                )

                if next_num_ctx > current_num_ctx:
                    current_num_ctx = (
                        next_num_ctx
                    )
                    continue

            raise RuntimeError(
                "Ollama /api/chat failed "
                f"HTTP {response.status_code}. "
                f"Selected categories={categories}; "
                f"tool_count={len(selected_tools)}; "
                f"attempted_num_ctx={attempted_contexts}. "
                f"Ollama response: {detail}"
            )

    @staticmethod
    def _explicit_render_filename(
        user_request,
    ):
        text = str(
            user_request
        )

        # Quoted/backtick filename: spaces are allowed inside the quotes.
        quoted = re.search(
            r"""(?i)[`'"]([^`'"]+?\.(?:png|jpe?g|exr))[`'"]""",
            text,
        )

        if quoted:
            return Path(
                quoted.group(1)
            ).name

        # Unquoted filename: consume only a filename token, never the
        # surrounding natural-language phrase.
        token = re.search(
            r"""(?i)(?<![A-Za-z0-9_-])([A-Za-z0-9][A-Za-z0-9_.-]*?\.(?:png|jpe?g|exr))(?=$|[\s,;:!?)}\].])""",
            text,
        )

        if not token:
            return None

        return Path(
            token.group(1)
        ).name


    @staticmethod
    def _render_action_requested(
        user_request,
    ):
        text = (
            str(user_request)
            .strip()
            .lower()
        )

        if not text:
            return False

        # Requests that merely configure render settings are not renders.
        settings_only_starts = (
            "set the render resolution",
            "set render resolution",
            "set the render engine",
            "set render engine",
            "set the render samples",
            "set render samples",
            "set the render output",
            "set render output",
        )

        # A configuration request may still end with "... and render".
        if any(
            marker in text
            for marker in (
                "render the scene",
                "render scene",
                "render it",
                "render to file",
                "render the image",
                "and render",
            )
        ):
            return True

        if text in {
            "render",
            "render.",
        }:
            return True

        if text.startswith(
            "render "
        ):
            return True

        return False


    @classmethod
    def _render_save_requested(
        cls,
        user_request,
    ):
        text = (
            str(user_request)
            .strip()
            .lower()
        )

        filename = (
            cls._explicit_render_filename(
                user_request
            )
        )

        return bool(
            filename
            or any(
                marker in text
                for marker in (
                    "to file",
                    "save the render",
                    "save render",
                    "save the image",
                    "save image",
                    "output filename",
                )
            )
        )


    @staticmethod
    def _file_format_for_filename(
        filename,
    ):
        suffix = (
            Path(filename)
            .suffix
            .lower()
        )

        if suffix == ".png":
            return "PNG"

        if suffix in {
            ".jpg",
            ".jpeg",
        }:
            return "JPEG"

        if suffix == ".exr":
            return "OPEN_EXR"

        return "PNG"


    @staticmethod
    def _render_result_verified(
        tool_result,
        save_required,
    ):
        if not isinstance(
            tool_result,
            dict,
        ):
            return False

        if (
            tool_result.get(
                "success"
            ) is not True
            or tool_result.get(
                "verified"
            ) is not True
        ):
            return False

        render_result = (
            tool_result.get(
                "render_result"
            )
        )

        if (
            not isinstance(
                render_result,
                dict,
            )
            or render_result.get(
                "verified"
            ) is not True
        ):
            return False

        if not save_required:
            return True

        return (
            tool_result.get(
                "saved_to_file"
            ) is True
            and tool_result.get(
                "file_verified"
            ) is True
            and bool(
                tool_result.get(
                    "output_path"
                )
            )
        )


    def _verified_render_answer(
        self,
        state,
    ):
        result = (
            state.verified_render_result
            or {}
        )

        if state.render_save_required:
            return (
                "Render completed successfully. "
                "The saved image was verified at:\\n"
                f"`{result.get('output_path')}`"
            )

        return (
            "Render completed successfully and Blender's "
            "Render Result was verified."
        )


    def _ensure_requested_render_output(
        self,
        state,
    ):
        filename = (
            state.requested_render_filename
        )

        if not filename:
            return {
                "success": True,
                "skipped": True,
            }

        command = {
            "tool": "set_render_output",
            "arguments": {
                "filename": filename,
                "file_format": (
                    self._file_format_for_filename(
                        filename
                    )
                ),
            },
        }

        execution = (
            self.router.execute(
                command
            )
        )

        verification = None

        if execution.get(
            "success",
            False,
        ):
            verification = (
                self._run_deterministic_render_verification(
                    state,
                    command,
                )
            )

            if not verification.get(
                "verified",
                False,
            ):
                execution = {
                    **execution,
                    "success": False,
                    "error_type": (
                        "RENDER_OUTPUT_VERIFICATION"
                    ),
                    "tool_result": {
                        "success": False,
                        "error": (
                            "Controller-enforced render output "
                            "could not be verified. "
                            + str(
                                verification.get(
                                    "reason",
                                    "",
                                )
                            )
                        ),
                        "verification": (
                            verification
                        ),
                    },
                }

        state.trace[
            "controller_events"
        ].append(
            {
                "step": state.step,
                "event": (
                    "render_filename_enforced"
                ),
                "filename": filename,
                "success": execution.get(
                    "success",
                    False,
                ),
                "verified": (
                    verification.get(
                        "verified"
                    )
                    if isinstance(
                        verification,
                        dict,
                    )
                    else None
                ),
                "tool_result": execution.get(
                    "tool_result"
                ),
            }
        )

        return execution


    @staticmethod
    def _queued_llm_metadata(
        llm_result,
    ):
        """
        Preserve model/tool-selection metadata for queued commands without
        charging the original model latency again on every queued tool.
        """
        return {
            "latency": 0.0,
            "tool_categories": list(
                llm_result.get(
                    "tool_categories",
                    [],
                )
            ),
            "tool_count": int(
                llm_result.get(
                    "tool_count",
                    0,
                )
            ),
            "tool_names": list(
                llm_result.get(
                    "tool_names",
                    [],
                )
            ),
            "tool_discovery": dict(
                llm_result.get(
                    "tool_discovery",
                    {},
                )
            ),
            "num_ctx": (
                llm_result.get(
                    "num_ctx"
                )
            ),
            "attempted_contexts": list(
                llm_result.get(
                    "attempted_contexts",
                    [],
                )
            ),
            "prompt_eval_count": (
                llm_result.get(
                    "prompt_eval_count"
                )
            ),
            "eval_count": (
                llm_result.get(
                    "eval_count"
                )
            ),
        }


    def _build_tool_plan(
        self,
        state,
        tool_calls,
        llm_result,
    ):
        """
        Convert one model response into an ordered controller plan.

        Safety rules:
        - malformed calls are rejected rather than guessed;
        - if documentation retrieval is present, execute only that retrieval
          and re-plan after seeing its result;
        - render_scene is de-duplicated and moved to the end of a render plan;
        - every command is still validated again immediately before execution;
        - high-risk tools still stop for human approval.
        """
        commands = []

        for index, tool_call in enumerate(
            tool_calls
        ):
            try:
                command = (
                    tool_call_to_command(
                        tool_call
                    )
                )
            except Exception as exc:
                self._record_trace_event(
                    state.trace,
                    "plan_parse_failed",
                    step=state.step,
                    tool_call_index=index,
                    reason=str(
                        exc
                    ),
                )

                return {
                    "success": False,
                    "error": (
                        "The model returned a malformed tool call. "
                        "The controller discarded the plan instead of guessing."
                    ),
                    "commands": [],
                }

            commands.append(
                command
            )

        if not commands:
            return {
                "success": False,
                "error": (
                    "The model returned an empty tool plan."
                ),
                "commands": [],
            }

        # Documentation must inform the next action. Do not execute actions
        # that were planned before the retrieved docs were available.
        docs_commands = [
            command
            for command in commands
            if command.get(
                "tool"
            )
            == "search_blender_docs"
        ]

        if docs_commands:
            selected = [
                docs_commands[0]
            ]

            self._record_trace_event(
                state.trace,
                "plan_docs_first",
                step=state.step,
                original_count=len(
                    commands
                ),
                queued_tools=[
                    selected[0].get(
                        "tool"
                    )
                ],
                discarded_for_replan=max(
                    0,
                    len(commands)
                    - 1,
                ),
            )

            return {
                "success": True,
                "commands": selected,
                "docs_first": True,
            }

        # A render is terminal. Keep only one render_scene and execute it
        # after all other actions proposed in the same turn.
        if state.render_required:
            non_render = []
            render_command = None

            for command in commands:
                if (
                    command.get(
                        "tool"
                    )
                    == "render_scene"
                ):
                    if render_command is None:
                        render_command = command
                    continue

                non_render.append(
                    command
                )

            commands = non_render

            if render_command is not None:
                commands.append(
                    render_command
                )

        # Keep a bounded plan. Very large plans should be re-planned rather
        # than blindly executed in one burst.
        maximum_plan_tools = 12

        if len(commands) > maximum_plan_tools:
            self._record_trace_event(
                state.trace,
                "plan_truncated",
                step=state.step,
                original_count=len(
                    commands
                ),
                kept_count=(
                    maximum_plan_tools
                ),
            )

            commands = commands[
                :maximum_plan_tools
            ]

        return {
            "success": True,
            "commands": commands,
            "docs_first": False,
        }


    def _enqueue_remaining_plan(
        self,
        state,
        commands,
        llm_result,
        origin_step,
    ):
        if not commands:
            return

        metadata = (
            self._queued_llm_metadata(
                llm_result
            )
        )

        for position, command in enumerate(
            commands,
            start=1,
        ):
            state.pending_tool_plan.append(
                {
                    "command": command,
                    "origin_step": int(
                        origin_step
                    ),
                    "position": position,
                    "llm_result": dict(
                        metadata
                    ),
                }
            )

        self._record_trace_event(
            state.trace,
            "plan_queued",
            step=state.step,
            origin_step=origin_step,
            count=len(
                commands
            ),
            tools=[
                command.get(
                    "tool"
                )
                for command in commands
            ],
        )

        state.trace[
            "controller_events"
        ].append(
            {
                "step": state.step,
                "event": (
                    "multi_tool_plan_queued"
                ),
                "origin_step": (
                    origin_step
                ),
                "count": len(
                    commands
                ),
                "tools": [
                    command.get(
                        "tool"
                    )
                    for command in commands
                ],
            }
        )


    def _clear_pending_plan(
        self,
        state,
        reason,
    ):
        if not state.pending_tool_plan:
            return

        discarded = [
            item.get(
                "command",
                {},
            ).get(
                "tool"
            )
            for item in (
                state.pending_tool_plan
            )
        ]

        state.pending_tool_plan.clear()

        self._record_trace_event(
            state.trace,
            "plan_aborted",
            step=state.step,
            reason=str(
                reason
            ),
            discarded_tools=(
                discarded
            ),
        )

        state.trace[
            "controller_events"
        ].append(
            {
                "step": state.step,
                "event": (
                    "pending_plan_aborted"
                ),
                "reason": str(
                    reason
                ),
                "discarded_tools": (
                    discarded
                ),
            }
        )


    @staticmethod
    def _trace_used_rag(trace):
        return any(
            step.get("tool") == "search_blender_docs"
            and step.get("result", {}).get("success") is True
            for step in trace["steps"]
        )

    @staticmethod
    def _has_doc_citation(text):
        return bool(re.search(r"\[DOC\d+\]", text or ""))

    def chat(self, user_request, conversation_context=None):
        user_request = str(user_request).strip()

        if not user_request:
            return {
                "status": "error",
                "error": "Message cannot be empty.",
                "project_root": str(self.project_root),
            }

        context = self._sanitize_conversation_context(
            conversation_context,
            current_request=user_request,
        )

        structured_memory = dict(
            context.get("structured_memory", {})
        )

        reference_resolution = self._resolve_referential_request(
            user_request,
            structured_memory,
        )

        planning_request = reference_resolution.get(
            "planning_request",
            user_request,
        )

        trace = self._create_trace(user_request)
        trace["conversation_context"] = {
            "recent_message_count": len(context["recent_messages"]),
            "memory_summary_chars": len(context["memory_summary"]),
            "structured_memory_keys": sorted(structured_memory.keys()),
        }
        trace["reference_resolution"] = dict(reference_resolution)

        self._record_trace_event(
            trace,
            "conversation_context_received",
            recent_message_count=len(context["recent_messages"]),
            memory_summary_chars=len(context["memory_summary"]),
            structured_memory_keys=sorted(structured_memory.keys()),
            bounded=True,
        )

        if reference_resolution.get("resolved"):
            self._record_trace_event(
                trace,
                "reference_resolved",
                target_type=reference_resolution.get("target_type"),
                target_name=reference_resolution.get("target_name"),
                source=reference_resolution.get("source"),
                intent=reference_resolution.get("intent"),
            )
        elif reference_resolution.get("same_turn"):
            self._record_trace_event(
                trace,
                "same_turn_reference_detected",
                source=reference_resolution.get("source"),
                scope="same_turn",
            )
        elif reference_resolution.get("unresolved"):
            self._record_trace_event(
                trace,
                "reference_unresolved",
                source=reference_resolution.get("source"),
            )

        render_required = (
            self._render_action_requested(
                planning_request
            )
        )

        render_save_required = (
            render_required
            and self._render_save_requested(
                planning_request
            )
        )

        requested_render_filename = (
            self._explicit_render_filename(
                planning_request
            )
            if render_save_required
            else None
        )

        if render_required:
            render_event = {
                "step": 0,
                "event": (
                    "render_transaction_started"
                ),
                "save_to_file": (
                    render_save_required
                ),
                "requested_filename": (
                    requested_render_filename
                ),
                "render_command_id": (
                    f"{trace['trace_id']}:render"
                ),
            }

            trace[
                "controller_events"
            ].append(
                render_event
            )

            self._record_trace_event(
                trace,
                "render_transaction_started",
                step=0,
                save_to_file=(
                    render_save_required
                ),
                requested_filename=(
                    requested_render_filename
                ),
                render_command_id=(
                    render_event[
                        "render_command_id"
                    ]
                ),
            )

        goal_ledger = self._extract_goal_ledger(planning_request)
        trace["goal_ledger"] = [dict(goal) for goal in goal_ledger]

        state = AgentState(
            user_request=user_request,
            messages=(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                ]
                + self._conversation_context_messages(context)
                + self._reference_context_message(reference_resolution)
                + [
                    {
                        "role": "user",
                        "content": user_request,
                    },
                ]
            ),
            trace=trace,
            render_required=render_required,
            render_save_required=(
                render_save_required
            ),
            requested_render_filename=(
                requested_render_filename
            ),
            goal_ledger=goal_ledger,
            conversation_context=context,
            prior_memory_summary=context.get("memory_summary", ""),
            prior_structured_memory=structured_memory,
            structured_memory=dict(structured_memory),
            reference_resolution=reference_resolution,
            planning_request=planning_request,
        )

        if reference_resolution.get("unresolved"):
            return self._complete(
                state,
                (
                    "I couldn't determine what the reference in this request "
                    "refers to from the bounded conversation context. Please "
                    "name the object, camera, light, or material explicitly."
                ),
                success=False,
            )

        try:
            return self._continue(
                state
            )

        except Exception as exc:
            self._record_trace_event(
                trace,
                "request_exception",
                error_type=(
                    type(exc).__name__
                ),
                reason=str(
                    exc
                ),
            )

            # Keep the HTTP behavior unchanged: the request handler will
            # still return the backend exception as an error.
            raise


    @staticmethod
    def _find_modifier_in_snapshot(
        snapshot,
        object_name,
        modifier_name,
    ):
        """
        Return the requested modifier dictionary from get_modifiers output,
        or None when the object/modifier is absent.
        """
        if not isinstance(snapshot, dict):
            return None

        for object_entry in snapshot.get("objects", []):
            if object_entry.get("object_name") != object_name:
                continue

            for modifier in object_entry.get("modifiers", []):
                if modifier.get("name") == modifier_name:
                    return modifier

        return None

    def _record_controller_observation(
        self,
        state,
        command,
        execution,
        phase,
        verified=None,
    ):
        """
        Record controller-owned observations separately from model-selected
        tool calls. These steps are deterministic harness behavior.
        """
        tool_result = execution["tool_result"]

        step_trace = {
            "step": state.step,
            "tool": command["tool"],
            "route": execution.get("route", self.router.route_name(command["tool"])),
            "arguments": command.get("arguments", {}),
            "risk": get_tool_risk(command),
            "status": (
                "success"
                if execution.get("success")
                else "failed"
            ),
            "approval": None,
            "llm_latency": 0.0,
            "tool_latency": execution.get("latency"),
            "attempts": execution.get("attempts", 1),
            "retry_errors": execution.get("retry_errors", []),
            "error_type": execution.get("error_type"),
            "recovered": execution.get("recovered", False),
            "result": tool_result,
            "error": (
                None
                if execution.get("success")
                else tool_result.get("error")
            ),
            "controller_managed": True,
            "phase": phase,
        }

        if verified is not None:
            step_trace["verified"] = bool(verified)

        state.trace["steps"].append(step_trace)

    def _apply_modifier_preflight(
        self,
        state,
        command,
        phase="apply_modifier_preflight",
    ):
        """
        Deterministically verify that:
        - the target object exists,
        - get_modifiers succeeds,
        - the named modifier currently exists on that object.

        This happens before human approval. A second fresh check is also
        performed immediately after approval and before destructive execution.
        """
        arguments = command["arguments"]

        object_name = arguments["object_name"]
        modifier_name = arguments["modifier_name"]

        observation_command = {
            "tool": "get_modifiers",
            "arguments": {
                "object_name": object_name,
            },
        }

        execution = self.bridge.execute_with_retry(
            observation_command
        )

        snapshot = execution["tool_result"]

        modifier = None

        if execution.get("success"):
            modifier = self._find_modifier_in_snapshot(
                snapshot,
                object_name,
                modifier_name,
            )

        verified = (
            execution.get("success") is True
            and modifier is not None
        )

        self._record_controller_observation(
            state,
            observation_command,
            execution,
            phase=phase,
            verified=verified,
        )

        if not execution.get("success"):
            return {
                "success": False,
                "verified": False,
                "object_name": object_name,
                "modifier_name": modifier_name,
                "error": (
                    "Modifier preflight failed because Blender could not "
                    "inspect the object's modifier stack: "
                    + str(snapshot.get("error", "unknown error"))
                ),
                "snapshot": snapshot,
            }

        if modifier is None:
            return {
                "success": False,
                "verified": False,
                "object_name": object_name,
                "modifier_name": modifier_name,
                "error": (
                    f"Modifier '{modifier_name}' was not found on "
                    f"'{object_name}'. The destructive apply operation "
                    "was blocked before approval."
                ),
                "snapshot": snapshot,
            }

        return {
            "success": True,
            "verified": True,
            "object_name": object_name,
            "modifier_name": modifier_name,
            "modifier": modifier,
            "snapshot": snapshot,
        }

    def _apply_modifier_postcheck(
        self,
        state,
        command,
    ):
        """
        Verify that the modifier is absent after Blender reports that it was
        applied. This is the hard success condition for apply_modifier.
        """
        arguments = command["arguments"]

        object_name = arguments["object_name"]
        modifier_name = arguments["modifier_name"]

        observation_command = {
            "tool": "get_modifiers",
            "arguments": {
                "object_name": object_name,
            },
        }

        execution = self.bridge.execute_with_retry(
            observation_command
        )

        snapshot = execution["tool_result"]

        modifier_still_present = True

        if execution.get("success"):
            modifier_still_present = (
                self._find_modifier_in_snapshot(
                    snapshot,
                    object_name,
                    modifier_name,
                )
                is not None
            )

        verified = (
            execution.get("success") is True
            and not modifier_still_present
        )

        self._record_controller_observation(
            state,
            observation_command,
            execution,
            phase="apply_modifier_postcheck",
            verified=verified,
        )

        if not execution.get("success"):
            return {
                "success": False,
                "verified": False,
                "object_name": object_name,
                "modifier_name": modifier_name,
                "error": (
                    "Blender reported that the modifier was applied, but "
                    "the controller could not perform the post-apply "
                    "verification: "
                    + str(snapshot.get("error", "unknown error"))
                ),
                "snapshot": snapshot,
            }

        if modifier_still_present:
            return {
                "success": False,
                "verified": False,
                "object_name": object_name,
                "modifier_name": modifier_name,
                "error": (
                    f"Post-apply verification failed: modifier "
                    f"'{modifier_name}' is still present on '{object_name}'."
                ),
                "snapshot": snapshot,
            }

        return {
            "success": True,
            "verified": True,
            "object_name": object_name,
            "modifier_name": modifier_name,
            "modifier_absent": True,
            "snapshot": snapshot,
        }

    def approve(self, approval_id, approved):
        with self._pending_lock:
            pending = self.pending_approvals.pop(
                approval_id,
                None,
            )

        if pending is None:
            return {
                "status": "error",
                "error": "Approval request was not found or has expired.",
                "project_root": str(self.project_root),
            }

        state = pending["state"]
        command = pending["command"]
        trace_index = pending["trace_index"]
        original_preflight = pending.get("preflight")

        if approved:
            # ------------------------------------------------
            # APPLY MODIFIER: re-check immediately before the
            # destructive operation, because the scene may
            # have changed while waiting for human approval.
            # ------------------------------------------------
            if command["tool"] == "apply_modifier":
                fresh_preflight = self._apply_modifier_preflight(
                    state,
                    command,
                    phase="apply_modifier_pre_execute_recheck",
                )

                if not fresh_preflight["success"]:
                    tool_result = {
                        "success": False,
                        "error": (
                            "The apply operation was approved, but the "
                            "controller blocked execution because the fresh "
                            "pre-execution modifier check failed."
                        ),
                        "preflight": original_preflight,
                        "pre_execute_recheck": fresh_preflight,
                    }

                    state.trace["steps"][trace_index].update(
                        {
                            "status": "blocked_after_approval",
                            "approval": "approved",
                            "result": tool_result,
                            "error": tool_result["error"],
                            "error_type": "PREFLIGHT_FAILED",
                        }
                    )

                    state.trace["controller_events"].append(
                        {
                            "step": state.step,
                            "event": "apply_modifier_blocked_after_approval",
                            "tool": command["tool"],
                            "reason": fresh_preflight["error"],
                        }
                    )

                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_name": command["tool"],
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                            ),
                        }
                    )

                    return self._continue(state)

            approved_command = {
                **command,
                "approved_high_risk": True,
            }

            execution = self.router.execute(
                approved_command
            )

            raw_tool_result = execution["tool_result"]

            self._finish_trace_step(
                state,
                trace_index,
                execution=execution,
                approval="approved",
            )

            # ------------------------------------------------
            # APPLY MODIFIER: controller, not Qwen, owns the
            # post-apply verification.
            # ------------------------------------------------
            if (
                command["tool"] == "apply_modifier"
                and execution["success"]
            ):
                state.needs_verification = True

                state.trace["controller_events"].append(
                    {
                        "step": state.step,
                        "event": "verification_required",
                        "tool": "apply_modifier",
                        "verification_tool": "get_modifiers",
                    }
                )

                postcheck = self._apply_modifier_postcheck(
                    state,
                    command,
                )

                tool_result = {
                    **raw_tool_result,
                    "preflight": original_preflight,
                    "post_verification": postcheck,
                }

                if postcheck["verified"]:
                    state.needs_verification = False
                    state.required_verification_tool = None

                    state.trace["controller_events"].append(
                        {
                            "step": state.step,
                            "event": "verification_completed",
                            "tool": "apply_modifier",
                            "verification_tool": "get_modifiers",
                        }
                    )

                    tool_result["success"] = True
                    tool_result["verified"] = True

                    state.trace["steps"][trace_index].update(
                        {
                            "status": "success",
                            "result": tool_result,
                            "error": None,
                        }
                    )

                else:
                    # Blender execution may have returned success, but the
                    # controller refuses to call the overall operation
                    # successful without hard post-verification.
                    tool_result = {
                        **tool_result,
                        "success": False,
                        "verified": False,
                        "execution_success": True,
                        "error": postcheck["error"],
                    }

                    state.trace["steps"][trace_index].update(
                        {
                            "status": "verification_failed",
                            "result": tool_result,
                            "error": postcheck["error"],
                            "error_type": "POST_VERIFICATION_FAILED",
                        }
                    )

                    state.trace["controller_events"].append(
                        {
                            "step": state.step,
                            "event": "side_effect_executed_verification_failed_no_replay",
                            "tool": "apply_modifier",
                            "reason": postcheck["error"],
                            "behavior_group": get_tool_behavior("apply_modifier"),
                        }
                    )
                    self._clear_pending_plan(
                        state,
                        reason="apply_modifier_postcheck_failed_no_replay",
                    )
                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_name": "apply_modifier",
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                            ),
                        }
                    )
                    return self._complete(
                        state,
                        (
                            "The modifier apply operation executed, but the "
                            "controller could not verify the final state. "
                            "It was not replayed automatically.\n"
                            f"Reason: {postcheck['error']}"
                        ),
                        success=False,
                    )

            else:
                tool_result = raw_tool_result

                self._update_verification_state(
                    state,
                    command["tool"],
                    tool_result,
                )

                if execution.get("success", False) and command["tool"] in MUTATING_TOOLS:
                    deterministic_check = self._run_deterministic_mutation_verification(state, command, tool_result)
                    tool_result = {**tool_result, "post_verification": deterministic_check}
                    if deterministic_check.get("verified", False):
                        state.needs_verification = False
                        state.required_verification_tool = None
                        tool_result["verified"] = True
                        state.trace["steps"][trace_index].update({"status":"success","result":tool_result,"error":None})
                        state.trace["controller_events"].append({"step":state.step,"event":"verification_completed","tool":command["tool"],"verification_tool":deterministic_check.get("verification_tool"),"controller_driven":True,"after_approval":True})
                    else:
                        state.needs_verification = False
                        state.required_verification_tool = None
                        tool_result = {
                            **tool_result,
                            "success": False,
                            "verified": False,
                            "execution_success": True,
                            "no_auto_replay": True,
                            "error": (
                                "The approved mutation executed, but deterministic "
                                "post-verification failed. "
                                + str(deterministic_check.get("reason", ""))
                            ),
                        }
                        state.trace["steps"][trace_index].update(
                            {
                                "status": "verification_failed",
                                "result": tool_result,
                                "error": tool_result["error"],
                                "error_type": "POST_VERIFICATION_FAILED",
                            }
                        )
                        state.trace["controller_events"].append(
                            {
                                "step": state.step,
                                "event": "side_effect_executed_verification_failed_no_replay",
                                "tool": command["tool"],
                                "reason": tool_result["error"],
                                "after_approval": True,
                                "behavior_group": get_tool_behavior(command),
                            }
                        )
                        self._clear_pending_plan(
                            state,
                            reason="successful_side_effect_failed_verification_no_replay",
                        )
                        state.messages.append(
                            {
                                "role": "tool",
                                "tool_name": command["tool"],
                                "content": json.dumps(
                                    tool_result,
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        return self._complete(
                            state,
                            (
                                "The Blender mutation executed, but its post-"
                                "verification was inconclusive or failed. "
                                "Because this tool changes scene state, the "
                                "controller did not replay it automatically.\n"
                                f"Reason: {deterministic_check.get('reason')}"
                            ),
                            success=False,
                        )

        else:
            tool_result = {
                "success": False,
                "error": "User rejected the high-risk action.",
            }

            step_trace = state.trace["steps"][trace_index]

            step_trace.update(
                {
                    "status": "rejected",
                    "approval": "rejected",
                    "result": tool_result,
                    "error": tool_result["error"],
                    "error_type": "AUTHORIZATION",
                }
            )

            state.trace["controller_events"].append(
                {
                    "step": state.step,
                    "event": "high_risk_rejected",
                    "tool": command["tool"],
                }
            )

            self._clear_pending_plan(
                state,
                reason=(
                    "high_risk_action_rejected"
                ),
            )

        if (
            approved
            and isinstance(
                tool_result,
                dict,
            )
            and tool_result.get(
                "success"
            ) is not True
        ):
            self._clear_pending_plan(
                state,
                reason=(
                    "approved_high_risk_action_failed"
                ),
            )

        if (
            approved
            and isinstance(tool_result, dict)
            and tool_result.get("success") is True
        ):
            self._mark_goal_tool_success(
                state,
                command["tool"],
                tool_result,
                satisfied_by="approved_verified_tool_execution",
            )

        state.messages.append(
            {
                "role": "tool",
                "tool_name": command["tool"],
                "content": json.dumps(
                    tool_result,
                    ensure_ascii=False,
                ),
            }
        )

        return self._continue(state)

    def _pending_approval_response(
        self,
        state,
        command,
        llm_latency,
        preflight=None,
    ):
        approval_id = str(uuid.uuid4())

        step_trace = {
            "step": state.step,
            "tool": command["tool"],
            "route": self.router.route_name(command["tool"]),
            "arguments": command.get("arguments", {}),
            "risk": get_tool_risk(command),
            "status": "approval_required",
            "approval": "pending",
            "llm_latency": llm_latency,
            "tool_latency": None,
            "attempts": 0,
            "retry_errors": [],
            "error_type": None,
            "recovered": False,
            "result": None,
            "error": None,
            "preflight": preflight,
        }

        state.trace["steps"].append(step_trace)
        trace_index = len(state.trace["steps"]) - 1

        with self._pending_lock:
            self.pending_approvals[approval_id] = {
                "state": state,
                "command": command,
                "trace_index": trace_index,
                "preflight": preflight,
                "created_at": time.time(),
            }

        state.trace["controller_events"].append(
            {
                "step": state.step,
                "event": "high_risk_approval_required",
                "tool": command["tool"],
            }
        )

        self._record_trace_event(
            state.trace,
            "approval_required",
            step=state.step,
            tool=command[
                "tool"
            ],
            approval_id=(
                approval_id
            ),
        )

        trace_id = str(
            state.trace[
                "trace_id"
            ]
        )

        return {
            "status": "approval_required",
            "approval_id": approval_id,
            "trace_id": trace_id,
            "trace_path": str(
                self.trace_dir
                / f"{trace_id}.json"
            ),
            "trace_log_path": str(
                self.trace_dir
                / f"{trace_id}.log"
            ),
            "action": {
                "tool": command["tool"],
                "arguments": command.get("arguments", {}),
            },
            "message": (
                (
                    f"Preflight confirmed modifier "
                    f"'{command['arguments'].get('modifier_name')}' on "
                    f"'{command['arguments'].get('object_name')}'. "
                    "Approval is required before permanently applying it."
                )
                if command["tool"] == "apply_modifier"
                else f"Approval required before executing {command['tool']}."
            ),
            "project_root": str(self.project_root),
        }

    def _finish_trace_step(
        self,
        state,
        trace_index,
        execution,
        approval=None,
    ):
        tool_result = execution["tool_result"]

        state.trace["steps"][trace_index].update(
            {
                "status": (
                    "success"
                    if execution["success"]
                    else "failed"
                ),
                "approval": approval,
                "tool_latency": execution["latency"],
                "attempts": execution["attempts"],
                "retry_errors": execution["retry_errors"],
                "error_type": execution["error_type"],
                "recovered": execution["recovered"],
                "result": tool_result,
                "error": (
                    None
                    if execution["success"]
                    else tool_result.get("error")
                ),
            }
        )

        step_trace = (
            state.trace[
                "steps"
            ][
                trace_index
            ]
        )

        self._record_trace_event(
            state.trace,
            "tool_end",
            step=(
                step_trace.get(
                    "step"
                )
            ),
            tool=(
                step_trace.get(
                    "tool"
                )
            ),
            status=(
                step_trace.get(
                    "status"
                )
            ),
            latency=(
                execution.get(
                    "latency"
                )
            ),
            attempts=(
                execution.get(
                    "attempts"
                )
            ),
            error_type=(
                execution.get(
                    "error_type"
                )
            ),
            recovered=(
                execution.get(
                    "recovered"
                )
            ),
            result=(
                tool_result
            ),
        )

    @staticmethod
    def _approximately_equal(
        left,
        right,
        tolerance=1e-6,
    ):
        try:
            return abs(
                float(left)
                - float(right)
            ) <= float(
                tolerance
            )
        except (
            TypeError,
            ValueError,
        ):
            return False


    def _render_verification_predicate(
        self,
        mutation_tool,
        mutation_arguments,
        verification_result,
    ):
        """
        Fail-closed predicate for deterministic render-setting verification.

        Returns:
            (verified: bool, reason: str)

        The verifier itself is fixed: get_render_settings.
        The predicate checks that the observed setting actually matches the
        mutation that was requested. A successful GET alone is not enough.
        """
        if not isinstance(
            verification_result,
            dict,
        ):
            return (
                False,
                "Verifier returned a malformed result.",
            )

        if verification_result.get(
            "success"
        ) is not True:
            return (
                False,
                verification_result.get(
                    "error",
                    "get_render_settings failed.",
                ),
            )

        arguments = (
            mutation_arguments
            or {}
        )

        if (
            mutation_tool
            == "set_render_resolution"
        ):
            observed = (
                verification_result.get(
                    "resolution",
                    {},
                )
            )

            expected = {
                "width": int(
                    arguments[
                        "width"
                    ]
                ),
                "height": int(
                    arguments[
                        "height"
                    ]
                ),
                "percentage": int(
                    arguments[
                        "percentage"
                    ]
                ),
            }

            actual = {
                "width": observed.get(
                    "width"
                ),
                "height": observed.get(
                    "height"
                ),
                "percentage": observed.get(
                    "percentage"
                ),
            }

            if actual == expected:
                return (
                    True,
                    "Resolution matches requested values.",
                )

            return (
                False,
                f"Resolution mismatch: expected {expected}, observed {actual}.",
            )

        if (
            mutation_tool
            == "set_render_engine"
        ):
            expected = str(
                arguments[
                    "engine"
                ]
            ).upper()

            actual = str(
                verification_result.get(
                    "engine",
                    "",
                )
            ).upper()

            if actual == expected:
                return (
                    True,
                    "Render engine matches requested value.",
                )

            return (
                False,
                f"Render engine mismatch: expected {expected}, observed {actual}.",
            )

        if (
            mutation_tool
            == "set_render_transparent"
        ):
            expected = bool(
                arguments[
                    "enabled"
                ]
            )

            actual = bool(
                verification_result.get(
                    "film_transparent"
                )
            )

            if actual == expected:
                return (
                    True,
                    "Transparency setting matches requested value.",
                )

            return (
                False,
                f"Transparency mismatch: expected {expected}, observed {actual}.",
            )

        if (
            mutation_tool
            == "set_render_output"
        ):
            requested_filename = Path(
                str(
                    arguments[
                        "filename"
                    ]
                )
            ).name

            requested_format = str(
                arguments[
                    "file_format"
                ]
            ).upper()

            extension_by_format = {
                "PNG": ".png",
                "JPEG": ".jpg",
                "OPEN_EXR": ".exr",
            }

            expected_extension = (
                extension_by_format.get(
                    requested_format
                )
            )

            requested_path = Path(
                requested_filename
            )

            if (
                expected_extension
                and requested_path.suffix.lower()
                != expected_extension
            ):
                expected_filename = (
                    requested_path.name
                    + expected_extension
                )
            else:
                expected_filename = (
                    requested_path.name
                )

            output = (
                verification_result.get(
                    "output",
                    {}
                )
            )

            observed_path = (
                output.get(
                    "absolute_filepath"
                )
                or output.get(
                    "filepath"
                )
                or ""
            )

            observed_filename = Path(
                str(
                    observed_path
                )
            ).name

            observed_format = str(
                output.get(
                    "file_format",
                    "",
                )
            ).upper()

            filename_ok = (
                observed_filename.lower()
                == expected_filename.lower()
            )

            format_ok = (
                observed_format
                == requested_format
            )

            if (
                filename_ok
                and format_ok
            ):
                return (
                    True,
                    "Render output filename and format match requested values.",
                )

            return (
                False,
                (
                    "Render output mismatch: "
                    f"expected filename={expected_filename}, "
                    f"format={requested_format}; "
                    f"observed filename={observed_filename}, "
                    f"format={observed_format}."
                ),
            )

        if (
            mutation_tool
            == "set_render_samples"
        ):
            expected = int(
                arguments[
                    "samples"
                ]
            )

            samples = (
                verification_result.get(
                    "samples",
                    {}
                )
            )

            candidates = [
                samples.get(
                    "cycles_samples"
                ),
                samples.get(
                    "eevee_samples"
                ),
            ]

            if expected in candidates:
                return (
                    True,
                    "Render sample count matches requested value.",
                )

            return (
                False,
                (
                    "Render samples mismatch: "
                    f"expected {expected}, observed {samples}."
                ),
            )

        return (
            False,
            (
                "No deterministic render verification predicate "
                f"is defined for '{mutation_tool}'."
            ),
        )


    def _run_deterministic_render_verification(
        self,
        state,
        mutation_command,
    ):
        """
        Execute get_render_settings directly for a render-setting mutation.

        This never asks the LLM which verifier to use. The mapping comes from
        VERIFICATION_TOOL_BY_MUTATION and is accepted only when it maps to
        get_render_settings. Unknown/mismatched mappings fail closed.
        """
        mutation_tool = (
            mutation_command[
                "tool"
            ]
        )

        expected_verifier = (
            VERIFICATION_TOOL_BY_MUTATION.get(
                mutation_tool
            )
        )

        if (
            expected_verifier
            != "get_render_settings"
        ):
            return {
                "success": False,
                "verified": False,
                "error": (
                    "Deterministic render verifier mapping mismatch: "
                    f"{mutation_tool} -> {expected_verifier!r}."
                ),
            }

        self._record_trace_event(
            state.trace,
            "verification_start",
            step=state.step,
            tool=mutation_tool,
            verification_tool=(
                expected_verifier
            ),
        )

        execution = self.router.execute(
            {
                "tool": (
                    expected_verifier
                ),
                "arguments": {},
            }
        )

        verifier_result = (
            execution.get(
                "tool_result",
                {},
            )
        )

        predicate_ok, reason = (
            self._render_verification_predicate(
                mutation_tool,
                mutation_command.get(
                    "arguments",
                    {},
                ),
                verifier_result,
            )
        )

        verified = (
            execution.get(
                "success",
                False,
            )
            and predicate_ok
        )

        self._record_trace_event(
            state.trace,
            "verification_end",
            step=state.step,
            tool=mutation_tool,
            verification_tool=(
                expected_verifier
            ),
            status=(
                "success"
                if verified
                else "failed"
            ),
            latency=(
                execution.get(
                    "latency"
                )
            ),
            attempts=(
                execution.get(
                    "attempts"
                )
            ),
            reason=reason,
            observed=(
                verifier_result
            ),
        )

        state.trace[
            "controller_events"
        ].append(
            {
                "step": state.step,
                "event": (
                    "deterministic_verification_completed"
                    if verified
                    else
                    "deterministic_verification_failed"
                ),
                "tool": mutation_tool,
                "verification_tool": (
                    expected_verifier
                ),
                "predicate_passed": (
                    predicate_ok
                ),
                "reason": reason,
            }
        )

        return {
            "success": bool(
                verified
            ),
            "verified": bool(
                verified
            ),
            "verification_tool": (
                expected_verifier
            ),
            "verification_result": (
                verifier_result
            ),
            "reason": reason,
            "execution": execution,
        }


    @staticmethod
    def _close_sequence(
        actual,
        expected,
        tolerance=1e-4,
    ):
        if (
            not isinstance(
                actual,
                (list, tuple),
            )
            or not isinstance(
                expected,
                (list, tuple),
            )
            or len(actual)
            != len(expected)
        ):
            return False

        for left, right in zip(
            actual,
            expected,
        ):
            try:
                if abs(
                    float(left)
                    - float(right)
                ) > float(
                    tolerance
                ):
                    return False
            except (
                TypeError,
                ValueError,
            ):
                return False

        return True


    @staticmethod
    def _find_named(
        items,
        name,
        key="name",
    ):
        if not isinstance(
            items,
            list,
        ):
            return None

        for item in items:
            if (
                isinstance(
                    item,
                    dict,
                )
                and item.get(
                    key
                )
                == name
            ):
                return item

        return None


    def _verification_command_for_mutation(
        self,
        mutation_command,
    ):
        tool_name = (
            mutation_command[
                "tool"
            ]
        )

        arguments = (
            mutation_command.get(
                "arguments",
                {},
            )
        )

        verifier = (
            VERIFICATION_TOOL_BY_MUTATION.get(
                tool_name
            )
        )

        if verifier is None:
            return None

        verifier_arguments = {}

        if (
            verifier
            == "get_modifiers"
            and arguments.get(
                "object_name"
            )
        ):
            verifier_arguments[
                "object_name"
            ] = arguments[
                "object_name"
            ]

        elif verifier in {
            "get_mesh_info",
            "get_mesh_regions",
        }:
            object_name = (
                arguments.get(
                    "object_name"
                )
            )

            if not object_name:
                return None

            verifier_arguments[
                "object_name"
            ] = object_name

        return {
            "tool": verifier,
            "arguments": (
                verifier_arguments
            ),
        }


    def _mutation_verification_predicate(
        self,
        mutation_tool,
        mutation_arguments,
        mutation_result,
        verification_result,
    ):
        if not isinstance(
            verification_result,
            dict,
        ):
            return (
                False,
                "Verifier returned a malformed result.",
            )

        if verification_result.get(
            "success"
        ) is not True:
            return (
                False,
                verification_result.get(
                    "error",
                    "Verification tool failed.",
                ),
            )

        args = (
            mutation_arguments
            or {}
        )

        result = (
            mutation_result
            if isinstance(
                mutation_result,
                dict,
            )
            else {}
        )

        object_tools = {
            "create_cube",
            "move_object",
            "create_uv_sphere",
            "create_cylinder",
            "create_cone",
            "create_plane",
            "create_torus",
        }

        if mutation_tool in object_tools:
            object_name = (
                args.get(
                    "name"
                )
                or args.get(
                    "object_name"
                )
            )

            observed = (
                self._find_named(
                    verification_result.get(
                        "objects",
                        [],
                    ),
                    object_name,
                )
            )

            if observed is None:
                return (
                    False,
                    f"Object '{object_name}' was not found after mutation.",
                )

            expected_location = [
                args.get(
                    "x"
                ),
                args.get(
                    "y"
                ),
                args.get(
                    "z"
                ),
            ]

            if not self._close_sequence(
                observed.get(
                    "location"
                ),
                expected_location,
            ):
                return (
                    False,
                    (
                        f"Object '{object_name}' location mismatch: "
                        f"expected {expected_location}, "
                        f"observed {observed.get('location')}."
                    ),
                )

            return (
                True,
                f"Object '{object_name}' exists at the requested location.",
            )

        if mutation_tool == "delete_object":
            object_name = (
                args.get("object_name")
                or args.get("name")
            )

            observed = (
                self._find_named(
                    verification_result.get(
                        "objects",
                        [],
                    ),
                    object_name,
                )
            )

            return (
                observed is None,
                (
                    f"Object '{object_name}' is absent."
                    if observed is None
                    else
                    f"Object '{object_name}' is still present."
                ),
            )

        if mutation_tool in {
            "create_material",
            "set_material_color",
        }:
            material_name = (
                args.get(
                    "name"
                )
                or args.get(
                    "material_name"
                )
            )

            observed = (
                self._find_named(
                    verification_result.get(
                        "materials",
                        [],
                    ),
                    material_name,
                )
            )

            if observed is None:
                return (
                    False,
                    f"Material '{material_name}' was not found.",
                )

            expected_color = [
                args[
                    "r"
                ],
                args[
                    "g"
                ],
                args[
                    "b"
                ],
                args[
                    "a"
                ],
            ]

            if not self._close_sequence(
                observed.get(
                    "base_color"
                ),
                expected_color,
            ):
                return (
                    False,
                    (
                        f"Material '{material_name}' color mismatch: "
                        f"expected {expected_color}, "
                        f"observed {observed.get('base_color')}."
                    ),
                )

            return (
                True,
                f"Material '{material_name}' has the requested color.",
            )

        if mutation_tool == "assign_material":
            material_name = (
                args[
                    "material_name"
                ]
            )

            object_name = (
                args[
                    "object_name"
                ]
            )

            observed = (
                self._find_named(
                    verification_result.get(
                        "materials",
                        [],
                    ),
                    material_name,
                )
            )

            if observed is None:
                return (
                    False,
                    f"Material '{material_name}' was not found.",
                )

            assigned = (
                object_name
                in observed.get(
                    "objects",
                    [],
                )
            )

            return (
                assigned,
                (
                    f"Material '{material_name}' is assigned to '{object_name}'."
                    if assigned
                    else
                    f"Material '{material_name}' is not assigned to '{object_name}'."
                ),
            )

        if mutation_tool in {
            "add_bevel_modifier",
            "set_bevel_modifier",
            "add_subdivision_modifier",
            "set_subdivision_modifier",
            "remove_modifier",
            "apply_modifier",
        }:
            object_name = (
                args[
                    "object_name"
                ]
            )

            modifier_name = (
                args[
                    "modifier_name"
                ]
            )

            object_entry = (
                self._find_named(
                    verification_result.get(
                        "objects",
                        [],
                    ),
                    object_name,
                    key="object_name",
                )
            )

            if object_entry is None:
                return (
                    False,
                    f"Modifier verifier could not find object '{object_name}'.",
                )

            modifier = (
                self._find_named(
                    object_entry.get(
                        "modifiers",
                        [],
                    ),
                    modifier_name,
                )
            )

            if mutation_tool in {
                "remove_modifier",
                "apply_modifier",
            }:
                absent = (
                    modifier is None
                )

                return (
                    absent,
                    (
                        f"Modifier '{modifier_name}' is absent from '{object_name}'."
                        if absent
                        else
                        f"Modifier '{modifier_name}' is still present on '{object_name}'."
                    ),
                )

            if modifier is None:
                return (
                    False,
                    f"Modifier '{modifier_name}' was not found on '{object_name}'.",
                )

            if mutation_tool in {
                "add_bevel_modifier",
                "set_bevel_modifier",
            }:
                width_ok = (
                    modifier.get(
                        "type"
                    )
                    == "BEVEL"
                    and self._approximately_equal(
                        modifier.get(
                            "width"
                        ),
                        args[
                            "width"
                        ],
                        tolerance=1e-4,
                    )
                )

                segments_ok = (
                    int(
                        modifier.get(
                            "segments",
                            -1,
                        )
                    )
                    == int(
                        args[
                            "segments"
                        ]
                    )
                )

                return (
                    width_ok
                    and segments_ok,
                    (
                        f"Bevel '{modifier_name}' matches width and segments."
                        if width_ok
                        and segments_ok
                        else
                        f"Bevel '{modifier_name}' does not match requested settings."
                    ),
                )

            levels_ok = (
                int(
                    modifier.get(
                        "levels",
                        -1,
                    )
                )
                == int(
                    args[
                        "levels"
                    ]
                )
            )

            render_levels_ok = (
                int(
                    modifier.get(
                        "render_levels",
                        -1,
                    )
                )
                == int(
                    args[
                        "render_levels"
                    ]
                )
            )

            return (
                levels_ok
                and render_levels_ok,
                (
                    f"Subdivision '{modifier_name}' matches requested levels."
                    if levels_ok
                    and render_levels_ok
                    else
                    f"Subdivision '{modifier_name}' does not match requested levels."
                ),
            )

        if mutation_tool in {
            "create_camera",
            "move_camera",
            "set_camera_lens",
            "set_active_camera",
            "aim_camera_at_object",
        }:
            camera_name = (
                args.get(
                    "name"
                )
                or args.get(
                    "camera_name"
                )
            )

            observed = (
                self._find_named(
                    verification_result.get(
                        "cameras",
                        [],
                    ),
                    camera_name,
                )
            )

            if observed is None:
                return (
                    False,
                    f"Camera '{camera_name}' was not found.",
                )

            if mutation_tool in {
                "create_camera",
                "move_camera",
            }:
                expected_location = [
                    args[
                        "x"
                    ],
                    args[
                        "y"
                    ],
                    args[
                        "z"
                    ],
                ]

                if not self._close_sequence(
                    observed.get(
                        "location"
                    ),
                    expected_location,
                ):
                    return (
                        False,
                        (
                            f"Camera '{camera_name}' location mismatch: "
                            f"expected {expected_location}, "
                            f"observed {observed.get('location')}."
                        ),
                    )

            if mutation_tool in {
                "create_camera",
                "set_camera_lens",
            }:
                expected_lens = (
                    args[
                        "lens_mm"
                    ]
                )

                if not self._approximately_equal(
                    observed.get(
                        "lens_mm"
                    ),
                    expected_lens,
                    tolerance=1e-4,
                ):
                    return (
                        False,
                        (
                            f"Camera '{camera_name}' lens mismatch: "
                            f"expected {expected_lens}, "
                            f"observed {observed.get('lens_mm')}."
                        ),
                    )

            if mutation_tool == "set_active_camera":
                active = (
                    verification_result.get(
                        "active_camera"
                    )
                    == camera_name
                )

                if not active:
                    return (
                        False,
                        f"Camera '{camera_name}' is not active.",
                    )

            if mutation_tool == "aim_camera_at_object":
                expected_camera = (
                    result.get(
                        "camera",
                        {},
                    )
                )

                if not self._close_sequence(
                    observed.get(
                        "rotation_euler"
                    ),
                    expected_camera.get(
                        "rotation_euler"
                    ),
                    tolerance=1e-4,
                ):
                    return (
                        False,
                        f"Camera '{camera_name}' rotation changed after aiming.",
                    )

            return (
                True,
                f"Camera '{camera_name}' matches the requested state.",
            )

        if mutation_tool in {
            "create_light",
            "move_light",
            "set_light_energy",
            "set_light_color",
            "set_area_light_size",
            "aim_light_at_object",
        }:
            light_name = (
                args.get(
                    "name"
                )
                or args.get(
                    "light_name"
                )
            )

            observed = (
                self._find_named(
                    verification_result.get(
                        "lights",
                        [],
                    ),
                    light_name,
                )
            )

            if observed is None:
                return (
                    False,
                    f"Light '{light_name}' was not found.",
                )

            if mutation_tool == "create_light":
                if (
                    observed.get(
                        "light_type"
                    )
                    != str(
                        args[
                            "light_type"
                        ]
                    ).upper()
                ):
                    return (
                        False,
                        f"Light '{light_name}' type mismatch.",
                    )

            if mutation_tool in {
                "create_light",
                "move_light",
            }:
                expected_location = [
                    args[
                        "x"
                    ],
                    args[
                        "y"
                    ],
                    args[
                        "z"
                    ],
                ]

                if not self._close_sequence(
                    observed.get(
                        "location"
                    ),
                    expected_location,
                ):
                    return (
                        False,
                        f"Light '{light_name}' location mismatch.",
                    )

            if mutation_tool in {
                "create_light",
                "set_light_energy",
            }:
                if not self._approximately_equal(
                    observed.get(
                        "energy"
                    ),
                    args[
                        "energy"
                    ],
                    tolerance=1e-4,
                ):
                    return (
                        False,
                        f"Light '{light_name}' energy mismatch.",
                    )

            if mutation_tool in {
                "create_light",
                "set_light_color",
            }:
                expected_color = [
                    args[
                        "r"
                    ],
                    args[
                        "g"
                    ],
                    args[
                        "b"
                    ],
                ]

                if not self._close_sequence(
                    observed.get(
                        "color"
                    ),
                    expected_color,
                ):
                    return (
                        False,
                        f"Light '{light_name}' color mismatch.",
                    )

            if (
                mutation_tool
                in {
                    "create_light",
                    "set_area_light_size",
                }
                and str(
                    observed.get(
                        "light_type",
                        ""
                    )
                ).upper()
                == "AREA"
            ):
                if not self._approximately_equal(
                    observed.get(
                        "size"
                    ),
                    args[
                        "size"
                    ],
                    tolerance=1e-4,
                ):
                    return (
                        False,
                        f"AREA light '{light_name}' size mismatch.",
                    )

            if mutation_tool == "aim_light_at_object":
                expected_light = (
                    result.get(
                        "light",
                        {},
                    )
                )

                if not self._close_sequence(
                    observed.get(
                        "rotation_euler"
                    ),
                    expected_light.get(
                        "rotation_euler"
                    ),
                    tolerance=1e-4,
                ):
                    return (
                        False,
                        f"Light '{light_name}' rotation changed after aiming.",
                    )

            return (
                True,
                f"Light '{light_name}' matches the requested state.",
            )

        if mutation_tool in {
            "shade_smooth",
            "recalculate_normals",
        }:
            mesh = (
                verification_result.get(
                    "mesh",
                    {},
                )
            )

            if mutation_tool == "shade_smooth":
                face_count = int(
                    mesh.get(
                        "face_count",
                        0,
                    )
                )

                smooth_count = int(
                    mesh.get(
                        "smooth_face_count",
                        -1,
                    )
                )

                enabled = bool(
                    args[
                        "enabled"
                    ]
                )

                verified = (
                    (
                        enabled
                        and smooth_count
                        == face_count
                    )
                    or (
                        not enabled
                        and smooth_count
                        == 0
                    )
                )

                return (
                    verified,
                    (
                        "Mesh shading matches the requested smooth state."
                        if verified
                        else
                        "Mesh shading does not match the requested smooth state."
                    ),
                )

            expected_mesh = (
                result.get(
                    "mesh",
                    {},
                )
            )

            same_counts = all(
                int(
                    mesh.get(
                        key,
                        -1,
                    )
                )
                == int(
                    expected_mesh.get(
                        key,
                        -2,
                    )
                )
                for key in (
                    "vertex_count",
                    "edge_count",
                    "face_count",
                )
            )

            return (
                same_counts,
                (
                    "Mesh state remains consistent after normal recalculation."
                    if same_counts
                    else
                    "Mesh state changed unexpectedly after normal recalculation."
                ),
            )

        if mutation_tool in {
            "inset_top_face",
            "translate_top_face",
            "scale_top_face",
        }:
            observed_mesh = verification_result.get("mesh", {})
            observed_region = verification_result.get("top_region", {})
            before_mesh = result.get("before", {})
            after_mesh = result.get("after", {})
            before_region = result.get("before_region", {})
            expected_region = result.get("after_region", {})

            counts_match = all(
                int(observed_mesh.get(key, -1)) == int(after_mesh.get(key, -2))
                for key in ("vertex_count", "edge_count", "face_count")
            )

            if mutation_tool == "inset_top_face":
                topology_changed = (
                    int(after_mesh.get("vertex_count", -1)) > int(before_mesh.get("vertex_count", -1))
                    and int(after_mesh.get("face_count", -1)) > int(before_mesh.get("face_count", -1))
                )
                region_matches = (
                    int(observed_region.get("face_count", -1))
                    == int(expected_region.get("face_count", -2))
                    and int(observed_region.get("vertex_count", -1))
                    == int(expected_region.get("vertex_count", -2))
                )
                verified = counts_match and topology_changed and region_matches
                return verified, (
                    "Inset changed topology and the observed top-region state matches the mutation result."
                    if verified
                    else "Inset semantic verification failed: observed topology/top-region state does not match the executed mutation."
                )

            if mutation_tool == "translate_top_face":
                before_centroid = before_region.get("centroid")
                observed_centroid = observed_region.get("centroid")
                offset = result.get("offset", [0.0, 0.0, 0.0])

                expected_centroid = (
                    [
                        float(before_centroid[index]) + float(offset[index])
                        for index in range(3)
                    ]
                    if isinstance(before_centroid, list) and len(before_centroid) == 3
                    else None
                )

                centroid_match = self._close_sequence(
                    observed_centroid,
                    expected_centroid,
                    tolerance=1e-4,
                )
                verified = counts_match and centroid_match
                return verified, (
                    "Top-region centroid moved by the requested XYZ offset."
                    if verified
                    else "Top-face translation verification failed: observed centroid does not match the requested offset."
                )

            before_width = before_region.get("width")
            before_depth = before_region.get("depth")
            observed_width = observed_region.get("width")
            observed_depth = observed_region.get("depth")
            x_factor = abs(float(result.get("x_factor", 1.0)))
            y_factor = abs(float(result.get("y_factor", 1.0)))

            expected_width = (
                float(before_width) * x_factor
                if before_width is not None
                else None
            )
            expected_depth = (
                float(before_depth) * y_factor
                if before_depth is not None
                else None
            )

            width_match = self._approximately_equal(
                observed_width,
                expected_width,
                tolerance=1e-4,
            )
            depth_match = self._approximately_equal(
                observed_depth,
                expected_depth,
                tolerance=1e-4,
            )
            center_match = self._close_sequence(
                observed_region.get("centroid")[:2]
                if isinstance(observed_region.get("centroid"), list)
                else None,
                before_region.get("centroid")[:2]
                if isinstance(before_region.get("centroid"), list)
                else None,
                tolerance=1e-4,
            )
            verified = counts_match and width_match and depth_match and center_match
            return verified, (
                "Top-region width/depth changed by the requested scale factors."
                if verified
                else "Top-face scale verification failed: observed top-region dimensions do not match requested factors."
            )

        if mutation_tool in {
            "scale_mesh_geometry",
            "extrude_top_face",
            "bevel_mesh_edges",
            "subdivide_mesh",
            "merge_by_distance",
            "solidify_mesh",
        }:
            observed_mesh = verification_result.get("mesh", {})
            before_mesh = result.get("before", {})
            expected_after = result.get("after", {})
            counts_match = all(
                int(observed_mesh.get(key, -1)) == int(expected_after.get(key, -2))
                for key in ("vertex_count", "edge_count", "face_count")
            )
            bounds_match = (
                self._close_sequence(
                    observed_mesh.get("local_bounds", {}).get("min"),
                    expected_after.get("local_bounds", {}).get("min"),
                    tolerance=1e-4,
                )
                and self._close_sequence(
                    observed_mesh.get("local_bounds", {}).get("max"),
                    expected_after.get("local_bounds", {}).get("max"),
                    tolerance=1e-4,
                )
            )

            semantic_change = True
            if mutation_tool == "subdivide_mesh":
                semantic_change = int(observed_mesh.get("vertex_count", -1)) > int(before_mesh.get("vertex_count", -1))
            elif mutation_tool == "merge_by_distance":
                semantic_change = int(observed_mesh.get("vertex_count", -1)) <= int(before_mesh.get("vertex_count", -1))
            elif mutation_tool == "solidify_mesh":
                semantic_change = int(observed_mesh.get("face_count", -1)) > int(before_mesh.get("face_count", -1))

            verified = counts_match and bounds_match and semantic_change
            return verified, (
                "Observed mesh state matches the direct-edit result and semantic invariant."
                if verified
                else "Observed mesh state does not satisfy the direct-edit semantic invariant."
            )

        if (
            VERIFICATION_TOOL_BY_MUTATION.get(
                mutation_tool
            )
            == "get_render_settings"
        ):
            return self._render_verification_predicate(
                mutation_tool,
                args,
                verification_result,
            )

        return (
            False,
            (
                "No deterministic verification predicate is defined for "
                f"'{mutation_tool}'."
            ),
        )


    def _run_deterministic_mutation_verification(
        self,
        state,
        mutation_command,
        mutation_result,
    ):
        verification_command = (
            self._verification_command_for_mutation(
                mutation_command
            )
        )

        if verification_command is None:
            return {
                "success": False,
                "verified": False,
                "reason": (
                    "No deterministic verification command is available."
                ),
            }

        mutation_tool = (
            mutation_command[
                "tool"
            ]
        )

        verifier_tool = (
            verification_command[
                "tool"
            ]
        )

        if verifier_tool == "get_render_settings":
            return self._run_deterministic_render_verification(
                state,
                mutation_command,
            )

        self._record_trace_event(
            state.trace,
            "verification_start",
            step=state.step,
            tool=mutation_tool,
            verification_tool=(
                verifier_tool
            ),
        )

        execution = (
            self.router.execute(
                verification_command
            )
        )

        verification_result = (
            execution.get(
                "tool_result",
                {},
            )
        )

        predicate_ok, reason = (
            self._mutation_verification_predicate(
                mutation_tool,
                mutation_command.get(
                    "arguments",
                    {},
                ),
                mutation_result,
                verification_result,
            )
        )

        verified = (
            execution.get(
                "success",
                False,
            )
            and predicate_ok
        )

        self._record_trace_event(
            state.trace,
            "verification_end",
            step=state.step,
            tool=mutation_tool,
            verification_tool=(
                verifier_tool
            ),
            status=(
                "success"
                if verified
                else "failed"
            ),
            latency=(
                execution.get(
                    "latency"
                )
            ),
            attempts=(
                execution.get(
                    "attempts"
                )
            ),
            reason=reason,
            observed=(
                verification_result
            ),
        )

        state.trace[
            "controller_events"
        ].append(
            {
                "step": state.step,
                "event": (
                    "deterministic_verification_completed"
                    if verified
                    else
                    "deterministic_verification_failed"
                ),
                "tool": mutation_tool,
                "verification_tool": (
                    verifier_tool
                ),
                "predicate_passed": bool(
                    predicate_ok
                ),
                "reason": reason,
            }
        )

        return {
            "success": bool(
                verified
            ),
            "verified": bool(
                verified
            ),
            "verification_tool": (
                verifier_tool
            ),
            "verification_result": (
                verification_result
            ),
            "reason": reason,
            "execution": execution,
        }


    def _update_verification_state(
        self,
        state,
        tool_name,
        tool_result,
    ):
        if tool_name in MUTATING_TOOLS and tool_result.get("success") is True:
            state.needs_verification = True
            state.required_verification_tool = VERIFICATION_TOOL_BY_MUTATION.get(tool_name)
            state.trace["controller_events"].append(
                {
                    "step": state.step,
                    "event": "verification_required",
                    "tool": tool_name,
                    "verification_tool": state.required_verification_tool,
                }
            )

        if (
            state.needs_verification
            and state.required_verification_tool
            and tool_name == state.required_verification_tool
            and tool_result.get("success") is True
        ):
            completed_with = state.required_verification_tool
            state.needs_verification = False
            state.required_verification_tool = None
            state.trace["controller_events"].append(
                {
                    "step": state.step,
                    "event": "verification_completed",
                    "tool": tool_name,
                    "verification_tool": completed_with,
                }
            )

    @staticmethod
    def _successful_trace_tools(trace):
        return {
            step.get("tool")
            for step in trace.get("steps", [])
            if step.get("status") == "success" and step.get("tool")
        }


    def _ground_final_answer(self, state, answer):
        """
        Guard a small set of state claims that are easy to verify
        deterministically. If the model adds an unsupported claim, replace the
        prose with a controller-owned summary of satisfied explicit goals.
        """
        if not state.goal_ledger:
            return answer

        text = str(answer or "")
        tools = self._successful_trace_tools(state.trace)
        unsupported = []

        smooth_claim = re.search(
            r"\bshaded\s+smooth\b|\bsmooth\b[^.\n]{0,24}\b(?:surface|shading|appearance|mesh|object)\b",
            text,
            re.IGNORECASE,
        )
        if smooth_claim and "shade_smooth" not in tools:
            unsupported.append("smooth_shading")

        render_claim = re.search(
            r"\b(?:render(?:ed|ing)?|saved\s+(?:render|image)|image\s+saved)\b",
            text,
            re.IGNORECASE,
        )
        if render_claim and "render_scene" not in tools and state.render_required:
            unsupported.append("render_completion")

        active_camera_claim = re.search(
            r"\b(?:camera\s+is\s+active|active\s+camera|made\s+active)\b",
            text,
            re.IGNORECASE,
        )
        if active_camera_claim and not ({"set_active_camera", "create_camera"} & tools):
            unsupported.append("active_camera")

        aim_claim = re.search(
            r"\b(?:aimed|pointing\s+at|points\s+at)\b",
            text,
            re.IGNORECASE,
        )
        if aim_claim and not ({"aim_camera_at_object", "aim_light_at_object"} & tools):
            unsupported.append("aim")

        if not unsupported:
            return answer

        satisfied = [
            goal.get("description", goal.get("tool", "requested change"))
            for goal in state.goal_ledger
            if goal.get("status") == "satisfied"
        ]
        summary = "; ".join(satisfied) or "the verified requested changes"
        grounded = "Completed and verified: " + summary + "."

        state.trace["controller_events"].append(
            {
                "step": state.step,
                "event": "final_answer_grounded",
                "unsupported_claims": unsupported,
            }
        )
        self._record_trace_event(
            state.trace,
            "final_answer_grounded",
            step=state.step,
            unsupported_claims=unsupported,
        )
        return grounded


    def _complete(self, state, answer, success=True):
        pending_goals = self._pending_goals(state)

        if success and pending_goals:
            success = False
            pending_text = ", ".join(
                goal.get("description", goal.get("tool", "unknown"))
                for goal in pending_goals
            )
            answer = (
                "The request was not declared complete because explicit goals "
                "remain unsatisfied: " + pending_text
            )
            state.trace["controller_events"].append(
                {
                    "step": state.step,
                    "event": "goal_completion_blocked",
                    "pending_goals": [goal.get("tool") for goal in pending_goals],
                }
            )

        if success:
            answer = self._ground_final_answer(state, answer)

        self._sync_goal_trace(state)
        trace = state.trace
        trace["final_answer"] = answer
        trace["success"] = bool(success)
        trace["verification_complete"] = not state.needs_verification
        trace["total_time"] = (
            time.perf_counter()
            - trace[
                "_started_perf"
            ]
        )
        trace.pop(
            "_started_perf",
            None,
        )

        self._record_trace_event(
            trace,
            "request_complete",
            status=(
                "success"
                if success
                else "failed"
            ),
            total_time=(
                trace[
                    "total_time"
                ]
            ),
            verification_complete=(
                trace[
                    "verification_complete"
                ]
            ),
        )

        updated_memory_summary = self._updated_memory_summary(
            state.prior_memory_summary,
            trace,
        )

        updated_structured_memory = self._updated_structured_memory(
            state.prior_structured_memory,
            trace,
        )

        trace["conversation_memory_summary_chars"] = len(
            updated_memory_summary
        )
        trace["structured_memory"] = dict(updated_structured_memory)

        trace_id = str(
            trace[
                "trace_id"
            ]
        )

        trace_path = (
            self.trace_dir
            / f"{trace_id}.json"
        )

        trace_log_path = (
            self.trace_dir
            / f"{trace_id}.log"
        )

        return {
            "status": "complete",
            "answer": answer,
            "project_root": str(self.project_root),
            "trace_id": trace_id,
            "trace_path": str(
                trace_path
            ),
            "trace_log_path": str(
                trace_log_path
            ),
            "memory_summary": updated_memory_summary,
            "structured_memory": updated_structured_memory,
            "conversation_context": {
                "recent_message_limit": HISTORY_CONTEXT_MAX_MESSAGES,
                "memory_summary_chars": len(updated_memory_summary),
                "structured_memory_keys": sorted(
                    updated_structured_memory.keys()
                ),
            },
            "trace": trace,
        }

    def _continue(self, state):
        while state.step < self.max_steps:
            state.step += 1

            plan_item = None
            plan_origin_step = None
            command = None

            # ------------------------------------------------------
            # VALIDATED PLAN QUEUE
            #
            # Do not consume a queued mutation while some earlier
            # mutation still requires verification. Render-setting
            # mutations clear their verification obligation
            # deterministically before reaching the next iteration.
            # ------------------------------------------------------
            deterministic_command = None
            if (
                not state.pending_tool_plan
                and not state.needs_verification
            ):
                deterministic_command = self._deterministic_simple_goal_command(state)

            if deterministic_command is not None:
                command = deterministic_command
                plan_origin_step = state.step
                llm_result = self._controller_injected_llm_result(
                    command.get("tool")
                )

                state.trace["controller_events"].append(
                    {
                        "step": state.step,
                        "event": "deterministic_goal_tool_injected",
                        "tool": command.get("tool"),
                        "reason": "single_exact_pending_goal",
                    }
                )
                self._record_trace_event(
                    state.trace,
                    "deterministic_goal_tool_injected",
                    step=state.step,
                    tool=command.get("tool"),
                    reason="single_exact_pending_goal",
                )

            elif (
                state.pending_tool_plan
                and not state.needs_verification
            ):
                plan_item = (
                    state.pending_tool_plan.pop(
                        0
                    )
                )

                command = dict(
                    plan_item[
                        "command"
                    ]
                )

                llm_result = dict(
                    plan_item[
                        "llm_result"
                    ]
                )

                plan_origin_step = (
                    plan_item.get(
                        "origin_step"
                    )
                )

                self._record_trace_event(
                    state.trace,
                    "plan_dequeued",
                    step=state.step,
                    origin_step=(
                        plan_origin_step
                    ),
                    tool=command.get(
                        "tool"
                    ),
                    remaining=len(
                        state.pending_tool_plan
                    ),
                )

            else:
                self._record_trace_event(
                    state.trace,
                    "llm_start",
                    step=state.step,
                    required_verification_tool=(
                        state.required_verification_tool
                    ),
                    pending_plan_count=len(
                        state.pending_tool_plan
                    ),
                )

                llm_result = self._call_llm(
                    state.messages,
                    user_request=(
                        state.planning_request
                        or state.user_request
                    ),
                    required_verification_tool=
                        state.required_verification_tool,
                )

                self._record_trace_event(
                    state.trace,
                    "llm_end",
                    step=state.step,
                    latency=(
                        llm_result.get(
                            "latency"
                        )
                    ),
                    tool_categories=(
                        llm_result.get(
                            "tool_categories",
                            [],
                        )
                    ),
                    tool_count=(
                        llm_result.get(
                            "tool_count",
                            0,
                        )
                    ),
                    selected_tool_names=(
                        llm_result.get(
                            "tool_names",
                            [],
                        )
                    ),
                    tool_discovery=(
                        llm_result.get(
                            "tool_discovery",
                            {},
                        )
                    ),
                    num_ctx=(
                        llm_result.get(
                            "num_ctx"
                        )
                    ),
                    prompt_eval_count=(
                        llm_result.get(
                            "prompt_eval_count"
                        )
                    ),
                    eval_count=(
                        llm_result.get(
                            "eval_count"
                        )
                    ),
                )

                assistant_message = (
                    llm_result[
                        "message"
                    ]
                )

                state.messages.append(
                    assistant_message
                )

                tool_calls = (
                    assistant_message.get(
                        "tool_calls",
                        [],
                    )
                )

                if not tool_calls:
                    answer = (
                        assistant_message.get(
                            "content",
                            "",
                        ).strip()
                    )

                    # Explicit task goals take precedence over a terminal render.
                    # If Qwen stops early, give it a bounded repair turn with the
                    # exact controller-owned goals that remain.
                    pending_goals = self._pending_goals(state)
                    if state.render_required:
                        pending_goals = [
                            goal
                            for goal in pending_goals
                            if goal.get("tool") not in {"render_scene", "set_render_output"}
                        ]
                    if pending_goals:
                        if state.goal_repair_attempts >= 2:
                            pending_text = ", ".join(
                                goal.get("description", goal.get("tool", "unknown"))
                                for goal in pending_goals
                            )
                            return self._complete(
                                state,
                                "The agent stopped safely because these explicit goals were not completed: " + pending_text,
                                success=False,
                            )

                        state.goal_repair_attempts += 1
                        state.trace["controller_events"].append(
                            {
                                "step": state.step,
                                "event": "goal_repair_requested",
                                "attempt": state.goal_repair_attempts,
                                "pending_goals": [goal.get("tool") for goal in pending_goals],
                            }
                        )
                        state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The controller still has unfinished explicit goals. "
                                    "Continue the task and use the available semantic tools for: "
                                    + ", ".join(goal.get("description", goal.get("tool", "unknown")) for goal in pending_goals)
                                    + ". Do not claim completion until they are satisfied."
                                ),
                            }
                        )
                        continue

                    # Verification obligations always take precedence over
                    # a terminal render. Never render an unverified mutation.
                    if state.needs_verification:
                        if (
                            state.required_verification_tool
                            == "get_render_settings"
                        ):
                            state.trace[
                                "controller_events"
                            ].append(
                                {
                                    "step": state.step,
                                    "event": (
                                        "unexpected_render_verification_pending"
                                    ),
                                    "reason": (
                                        "controller_owned_verification_should_have_run"
                                    ),
                                }
                            )

                            return self._complete(
                                state,
                                (
                                    "A render-setting verification obligation "
                                    "remained unexpectedly pending. The agent "
                                    "stopped safely instead of guessing."
                                ),
                                success=False,
                            )

                        state.trace[
                            "controller_events"
                        ].append(
                            {
                                "step": state.step,
                                "event": (
                                    "finish_blocked"
                                ),
                                "reason": (
                                    "scene_verification_required"
                                ),
                            }
                        )

                        state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Blender state was modified but not yet "
                                    "verified. You MUST call "
                                    f"{state.required_verification_tool} "
                                    "successfully before finishing."
                                ),
                            }
                        )

                        continue

                    if (
                        state.render_required
                        and not state.render_completed
                    ):
                        if state.render_attempted:
                            failure_result = (
                                state.verified_render_result
                                or {}
                            )

                            error_message = (
                                failure_result.get(
                                    "error"
                                )
                                or (
                                    "The render was attempted once but did "
                                    "not return a verified successful result."
                                )
                            )

                            state.trace[
                                "controller_events"
                            ].append(
                                {
                                    "step": state.step,
                                    "event": (
                                        "render_finish_failed"
                                    ),
                                    "reason": (
                                        error_message
                                    ),
                                }
                            )

                            return self._complete(
                                state,
                                (
                                    "The render was attempted once but could "
                                    "not be verified, so it was not replayed "
                                    "automatically.\n"
                                    f"Reason: {error_message}"
                                ),
                                success=False,
                            )

                        # --------------------------------------------------
                        # TERMINAL OBLIGATION
                        #
                        # The user explicitly asked for a render. If the
                        # model has finished talking but omitted render_scene,
                        # do not ask it again for 40 turns. The controller
                        # already knows the required semantic action.
                        # --------------------------------------------------
                        command = {
                            "tool": "render_scene",
                            "arguments": {
                                "save_to_file": bool(
                                    state.render_save_required
                                ),
                            },
                        }

                        plan_origin_step = (
                            state.step
                        )

                        state.trace[
                            "controller_events"
                        ].append(
                            {
                                "step": state.step,
                                "event": (
                                    "render_terminal_obligation_injected"
                                ),
                                "save_to_file": (
                                    state.render_save_required
                                ),
                            }
                        )

                        self._record_trace_event(
                            state.trace,
                            "terminal_tool_injected",
                            step=state.step,
                            tool="render_scene",
                            reason=(
                                "explicit_user_render_request"
                            ),
                            save_to_file=(
                                state.render_save_required
                            ),
                        )

                    else:
                        if (
                            self._trace_used_rag(
                                state.trace
                            )
                            and not self._has_doc_citation(
                                answer
                            )
                            and not state.citation_repair_used
                        ):
                            state.citation_repair_used = (
                                True
                            )

                            state.trace[
                                "controller_events"
                            ].append(
                                {
                                    "step": state.step,
                                    "event": (
                                        "citation_repair_requested"
                                    ),
                                }
                            )

                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "You used official Blender documentation. "
                                        "Revise the final answer and include valid "
                                        "retrieved citation labels such as [DOC1]."
                                    ),
                                }
                            )

                            continue

                        if (
                            state.render_required
                            and state.render_completed
                        ):
                            answer = (
                                self._verified_render_answer(
                                    state
                                )
                            )

                        return self._complete(
                            state,
                            answer,
                            success=True,
                        )

                else:
                    plan = self._build_tool_plan(
                        state,
                        tool_calls,
                        llm_result,
                    )

                    if not plan.get(
                        "success",
                        False,
                    ):
                        return self._complete(
                            state,
                            plan.get(
                                "error",
                                (
                                    "The model returned an invalid tool plan."
                                ),
                            ),
                            success=False,
                        )

                    commands = list(
                        plan[
                            "commands"
                        ]
                    )

                    command = commands.pop(
                        0
                    )

                    plan_origin_step = (
                        state.step
                    )

                    self._enqueue_remaining_plan(
                        state,
                        commands,
                        llm_result,
                        origin_step=(
                            state.step
                        ),
                    )

                    self._record_trace_event(
                        state.trace,
                        "plan_started",
                        step=state.step,
                        tool=command.get(
                            "tool"
                        ),
                        queued_count=len(
                            state.pending_tool_plan
                        ),
                    )

            # Controller semantic normalization happens before validation and
            # before the command can cross the Blender bridge.
            command = self._normalize_command_semantics(
                state,
                command,
            )

            if command.get("tool") == "render_scene":
                pending_non_render = self._pending_goals(
                    state,
                    include_render=False,
                )
                if pending_non_render:
                    state.trace["controller_events"].append(
                        {
                            "step": state.step,
                            "event": "render_deferred_for_goals",
                            "pending_goals": [goal.get("tool") for goal in pending_non_render],
                        }
                    )
                    state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Do not render yet. Complete these remaining goals first: "
                                + ", ".join(goal.get("description", goal.get("tool", "unknown")) for goal in pending_non_render)
                            ),
                        }
                    )
                    continue

            self._record_trace_event(
                state.trace,
                "tool_selected",
                step=state.step,
                tool=(
                    command.get(
                        "tool"
                    )
                ),
                arguments=(
                    command.get(
                        "arguments",
                        {},
                    )
                ),
                risk=(
                    get_tool_risk(
                        command.get(
                            "tool"
                        )
                    )
                ),
                domain_group=get_tool_domain(command),
                behavior_group=get_tool_behavior(command),
                route=(
                    self.router.route_name(
                        command.get(
                            "tool"
                        )
                    )
                ),
                plan_origin_step=(
                    plan_origin_step
                ),
                queued=(
                    plan_item is not None
                ),
                pending_plan_count=len(
                    state.pending_tool_plan
                ),
            )

            valid, validation_message = validate_tool_call(command)

            step_trace = {
                "step": state.step,
                "plan_origin_step": (
                    plan_origin_step
                ),
                "queued_plan_tool": (
                    plan_item is not None
                ),
                "tool": command.get("tool"),
                "route": self.router.route_name(command.get("tool")),
                "arguments": command.get("arguments", {}),
                "risk": get_tool_risk(command.get("tool")),
                "domain_group": get_tool_domain(command),
                "behavior_group": get_tool_behavior(command),
                "status": None,
                "approval": None,
                "llm_latency": llm_result["latency"],
                "llm_tool_categories": llm_result.get(
                    "tool_categories",
                    [],
                ),
                "llm_tool_count": llm_result.get(
                    "tool_count",
                    0,
                ),
                "llm_tool_names": llm_result.get(
                    "tool_names",
                    [],
                ),
                "ollama_num_ctx": llm_result.get(
                    "num_ctx"
                ),
                "ollama_context_attempts": llm_result.get(
                    "attempted_contexts",
                    [],
                ),
                "prompt_eval_count": llm_result.get(
                    "prompt_eval_count"
                ),
                "eval_count": llm_result.get(
                    "eval_count"
                ),
                "tool_latency": None,
                "attempts": 0,
                "retry_errors": [],
                "error_type": None,
                "recovered": False,
                "result": None,
                "error": None,
            }

            if not valid:
                tool_result = {
                    "success": False,
                    "error": validation_message,
                }

                step_trace.update(
                    {
                        "status": "failed",
                        "error": validation_message,
                        "error_type": "VALIDATION",
                        "result": tool_result,
                    }
                )

                state.trace["steps"].append(step_trace)

                state.messages.append(
                    {
                        "role": "tool",
                        "tool_name": command.get("tool", "unknown"),
                        "content": json.dumps(tool_result),
                    }
                )

                self._clear_pending_plan(
                    state,
                    reason=(
                        "tool_validation_failed"
                    ),
                )

                continue

            if get_tool_risk(command) == "high":
                preflight = None

                if command["tool"] == "apply_modifier":
                    preflight = self._apply_modifier_preflight(
                        state,
                        command,
                    )

                    if not preflight["success"]:
                        tool_result = {
                            "success": False,
                            "error": preflight["error"],
                            "preflight": preflight,
                        }

                        step_trace.update(
                            {
                                "status": "blocked_preflight",
                                "result": tool_result,
                                "error": preflight["error"],
                                "error_type": "PREFLIGHT_FAILED",
                            }
                        )

                        state.trace["steps"].append(
                            step_trace
                        )

                        state.trace["controller_events"].append(
                            {
                                "step": state.step,
                                "event": "apply_modifier_preflight_blocked",
                                "tool": "apply_modifier",
                                "reason": preflight["error"],
                            }
                        )

                        state.messages.append(
                            {
                                "role": "tool",
                                "tool_name": command["tool"],
                                "content": json.dumps(
                                    tool_result,
                                    ensure_ascii=False,
                                ),
                            }
                        )

                        continue

                return self._pending_approval_response(
                    state,
                    command,
                    llm_result["latency"],
                    preflight=preflight,
                )

            state.trace["steps"].append(step_trace)
            trace_index = len(state.trace["steps"]) - 1

            if command["tool"] == "render_scene":
                if state.render_attempted:
                    cached_result = (
                        state.verified_render_result
                        or {
                            "success": False,
                            "error": (
                                "render_scene was already attempted for this "
                                "user request. The controller will not replay "
                                "a side-effecting render operation."
                            ),
                            "render_reused": True,
                        }
                    )

                    state.trace[
                        "controller_events"
                    ].append(
                        {
                            "step": state.step,
                            "event": (
                                "duplicate_render_blocked"
                            ),
                        }
                    )

                    execution = {
                        "success": (
                            cached_result.get(
                                "success"
                            ) is True
                        ),
                        "route": "controller",
                        "tool_result": (
                            cached_result
                        ),
                        "attempts": 0,
                        "retry_errors": [],
                        "error_type": None,
                        "recovered": False,
                        "latency": 0.0,
                    }

                    tool_result = (
                        execution[
                            "tool_result"
                        ]
                    )

                    self._finish_trace_step(
                        state,
                        trace_index,
                        execution=execution,
                    )

                    state.messages.append(
                        {
                            "role": "tool",
                            "tool_name": (
                                "render_scene"
                            ),
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False,
                            ),
                        }
                    )

                    continue

                # The current user request owns one stable render command ID.
                command[
                    "_command_id"
                ] = (
                    f"{state.trace['trace_id']}:render"
                )

                # The controller, not the model, decides whether this render
                # must be written to disk.
                command.setdefault(
                    "arguments",
                    {},
                )[
                    "save_to_file"
                ] = bool(
                    state.render_save_required
                )

                if state.render_save_required:
                    filename_execution = (
                        self._ensure_requested_render_output(
                            state
                        )
                    )

                    if filename_execution.get("success", False):
                        self._mark_goal_tool_success(
                            state,
                            "set_render_output",
                            {"success": True},
                            satisfied_by="controller_render_output_preflight",
                        )

                    if not filename_execution.get(
                        "success",
                        False,
                    ):
                        filename_error = (
                            filename_execution.get(
                                "tool_result",
                                {},
                            ).get(
                                "error",
                                "Could not configure requested render filename.",
                            )
                        )

                        execution = {
                            "success": False,
                            "route": "controller",
                            "tool_result": {
                                "success": False,
                                "error": (
                                    filename_error
                                ),
                            },
                            "attempts": 0,
                            "retry_errors": [],
                            "error_type": (
                                "RENDER_OUTPUT_PREFLIGHT"
                            ),
                            "recovered": False,
                            "latency": 0.0,
                        }

                        tool_result = (
                            execution[
                                "tool_result"
                            ]
                        )

                        state.render_attempted = True
                        state.verified_render_result = (
                            dict(
                                tool_result
                            )
                        )

                        self._finish_trace_step(
                            state,
                            trace_index,
                            execution=execution,
                        )

                        state.messages.append(
                            {
                                "role": "tool",
                                "tool_name": (
                                    "render_scene"
                                ),
                                "content": json.dumps(
                                    tool_result,
                                    ensure_ascii=False,
                                ),
                            }
                        )

                        continue

                # Mark BEFORE crossing the bridge. If transport times out,
                # execution status is unknown and replaying is unsafe.
                state.render_attempted = True

            self._record_trace_event(
                state.trace,
                "tool_start",
                step=state.step,
                tool=command[
                    "tool"
                ],
                arguments=(
                    command.get(
                        "arguments",
                        {},
                    )
                ),
                command_id=(
                    command.get(
                        "_command_id"
                    )
                ),
            )

            execution = self.router.execute(
                command
            )
            tool_result = execution["tool_result"]

            self._finish_trace_step(
                state,
                trace_index,
                execution=execution,
            )

            if not execution.get(
                "success",
                False,
            ):
                self._clear_pending_plan(
                    state,
                    reason=(
                        f"{command['tool']}_execution_failed"
                    ),
                )

            self._update_verification_state(
                state,
                command["tool"],
                tool_result,
            )

            if (
                execution.get(
                    "success",
                    False,
                )
                and command[
                    "tool"
                ] in MUTATING_TOOLS
                and get_tool_risk(
                    command
                )
                != "high"
            ):
                deterministic_check = (
                    self._run_deterministic_mutation_verification(
                        state,
                        command,
                        tool_result,
                    )
                )

                if not deterministic_check.get(
                    "verified",
                    False,
                ):
                    # Fail closed. Do not ask the LLM to select another
                    # verifier and do not enter a verification loop.
                    return self._complete(
                        state,
                        (
                            "The Blender mutation executed, but deterministic "
                            "verification failed. The agent stopped instead "
                            "of claiming success or asking the LLM to guess.\n"
                            f"Reason: {deterministic_check.get('reason')}"
                        ),
                        success=False,
                    )

                # Clear the verification obligation only after the
                # controller-owned predicate passed.
                completed_with = (
                    state.required_verification_tool
                )

                state.needs_verification = False
                state.required_verification_tool = None

                state.trace[
                    "controller_events"
                ].append(
                    {
                        "step": state.step,
                        "event": (
                            "verification_completed"
                        ),
                        "tool": command[
                            "tool"
                        ],
                        "verification_tool": (
                            completed_with
                        ),
                        "controller_driven": True,
                    }
                )

                # Give the model the observed read-only state so it can plan
                # the next semantic action without another verification turn.
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_name": (
                            deterministic_check[
                                "verification_tool"
                            ]
                        ),
                        "content": json.dumps(
                            deterministic_check[
                                "verification_result"
                            ],
                            ensure_ascii=False,
                        ),
                    }
                )

            if (
                execution.get("success", False)
                and command.get("tool") in MUTATING_TOOLS
                and get_tool_risk(command) != "high"
            ):
                self._mark_goal_tool_success(
                    state,
                    command["tool"],
                    tool_result,
                    satisfied_by="verified_tool_execution",
                )

            if (
                command["tool"]
                == "render_scene"
                and state.render_required
            ):
                state.verified_render_result = (
                    dict(
                        tool_result
                    )
                    if isinstance(
                        tool_result,
                        dict,
                    )
                    else {
                        "success": False,
                        "error": (
                            "Malformed render result."
                        ),
                    }
                )

                state.render_completed = (
                    self._render_result_verified(
                        tool_result,
                        state.render_save_required,
                    )
                )

                state.trace[
                    "controller_events"
                ].append(
                    {
                        "step": state.step,
                        "event": (
                            "render_verified"
                            if state.render_completed
                            else
                            "render_not_verified"
                        ),
                        "output_path": (
                            tool_result.get(
                                "output_path"
                            )
                            if isinstance(
                                tool_result,
                                dict,
                            )
                            else None
                        ),
                        "file_verified": (
                            tool_result.get(
                                "file_verified"
                            )
                            if isinstance(
                                tool_result,
                                dict,
                            )
                            else None
                        ),
                        "bridge_attempts": (
                            execution.get(
                                "attempts"
                            )
                        ),
                    }
                )

                # ----------------------------------------------------
                # CRITICAL:
                # render_scene is terminal for the current request.
                #
                # Do not return control to the LLM after rendering.
                # Otherwise a small local model can repeatedly request
                # render_scene, consume the step budget, and eventually
                # report "Maximum agent steps reached" even though the
                # render already succeeded.
                # ----------------------------------------------------
                if state.render_completed:
                    self._mark_goal_tool_success(
                        state,
                        "render_scene",
                        tool_result,
                        satisfied_by="verified_render",
                    )

                    # If the last pending mutation was a render-setting
                    # mutation, verify it deterministically once here.
                    # No extra LLM round is needed.
                    if (
                        state.needs_verification
                        and state.required_verification_tool
                        == "get_render_settings"
                    ):
                        verification_execution = (
                            self.router.execute(
                                {
                                    "tool": "get_render_settings",
                                    "arguments": {},
                                }
                            )
                        )

                        verification_result = (
                            verification_execution.get(
                                "tool_result",
                                {},
                            )
                        )

                        self._update_verification_state(
                            state,
                            "get_render_settings",
                            verification_result,
                        )

                        state.trace[
                            "controller_events"
                        ].append(
                            {
                                "step": state.step,
                                "event": (
                                    "render_terminal_settings_verification"
                                ),
                                "success": (
                                    verification_execution.get(
                                        "success",
                                        False,
                                    )
                                ),
                            }
                        )

                    state.trace[
                        "controller_events"
                    ].append(
                        {
                            "step": state.step,
                            "event": (
                                "render_terminal_success"
                            ),
                        }
                    )

                    return self._complete(
                        state,
                        self._verified_render_answer(
                            state
                        ),
                        success=True,
                    )

                # The single render attempt failed or was not verifiable.
                # Do NOT ask the model to render again.
                error_message = (
                    state.verified_render_result.get(
                        "error"
                    )
                    or (
                        "The render was attempted once but the result "
                        "could not be verified."
                    )
                )

                state.trace[
                    "controller_events"
                ].append(
                    {
                        "step": state.step,
                        "event": (
                            "render_terminal_failure"
                        ),
                        "reason": error_message,
                    }
                )

                return self._complete(
                    state,
                    (
                        "The render was attempted once and was not "
                        "replayed automatically.\n"
                        f"Reason: {error_message}"
                    ),
                    success=False,
                )

            state.messages.append(
                {
                    "role": "tool",
                    "tool_name": command["tool"],
                    "content": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                }
            )

        return self._complete(
            state,
            "Maximum agent steps reached before completion.",
            success=False,
        )


class CopilotRequestHandler(BaseHTTPRequestHandler):
    server_version = "BlenderCopilot/0.1"

    def log_message(self, format_string, *args):
        print(
            f"[HTTP] {self.address_string()} - "
            + format_string % args
        )

    def _write_json(self, payload, status_code=200):
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"

        if not raw:
            return {}

        return json.loads(raw.decode("utf-8"))

    @property
    def agent(self):
        return self.server.copilot_agent

    def do_GET(self):
        if self.path == "/health":
            self._write_json(
                self.agent.health()
            )
            return

        if self.path == "/trace/latest":
            result = (
                self.agent.get_trace()
            )

            self._write_json(
                result,
                status_code=(
                    200
                    if result.get(
                        "status"
                    )
                    == "ok"
                    else 404
                ),
            )
            return

        if self.path.startswith(
            "/trace/"
        ):
            trace_id = (
                self.path[
                    len(
                        "/trace/"
                    ):
                ]
                .split(
                    "?",
                    1,
                )[0]
                .strip()
            )

            try:
                result = (
                    self.agent.get_trace(
                        trace_id
                    )
                )

                self._write_json(
                    result,
                    status_code=(
                        200
                        if result.get(
                            "status"
                        )
                        == "ok"
                        else 404
                    ),
                )

            except ValueError as exc:
                self._write_json(
                    {
                        "status": "error",
                        "error": str(
                            exc
                        ),
                    },
                    status_code=400,
                )

            return

        self._write_json(
            {
                "status": "error",
                "error": "Not found.",
            },
            status_code=404,
        )

    def do_POST(self):
        try:
            payload = self._read_json()

            if self.path == "/chat":
                result = self.agent.chat(
                    payload.get("message", ""),
                    conversation_context=payload.get(
                        "conversation_context"
                    ),
                )
                self._write_json(result)
                return

            if self.path == "/approve":
                result = self.agent.approve(
                    approval_id=payload.get("approval_id", ""),
                    approved=bool(payload.get("approved", False)),
                )
                self._write_json(result)
                return

            self._write_json(
                {
                    "status": "error",
                    "error": "Not found.",
                },
                status_code=404,
            )

        except Exception as exc:
            self._write_json(
                {
                    "status": "error",
                    "error": str(exc),
                },
                status_code=500,
            )


def default_project_root():
    env_value = os.environ.get("BLENDER_COPILOT_PROJECT_ROOT")

    if env_value:
        return Path(env_value).expanduser().resolve()

    # src/agent.py -> project root
    return Path(__file__).resolve().parent.parent


def run_server(
    project_root,
    host="127.0.0.1",
    port=8765,
    ollama_url="http://127.0.0.1:11434",
    model="qwen3:4b-instruct",
):
    agent = CopilotAgent(
        project_root=project_root,
        ollama_url=ollama_url,
        model=model,
    )

    server = ThreadingHTTPServer(
        (host, int(port)),
        CopilotRequestHandler,
    )
    server.copilot_agent = agent

    print()
    print("Blender AI Copilot backend is running")
    print(f"Backend: http://{host}:{port}")
    print(f"Project root: {Path(project_root).resolve()}")
    print(f"Ollama: {ollama_url}")
    print(f"Model: {model}")
    print("Controller: v0.8.0.2-aim-goal-fix")
    print(
        f"Dynamic tool gating: ON "
        f"({len(OLLAMA_TOOLS)} total registered tools)"
    )
    print(
        "Adaptive Ollama context: "
        f"{agent.ollama_num_ctx} initial / "
        f"{agent.ollama_max_num_ctx} max"
    )
    print(
        f"Agent step budget: {agent.max_steps}"
    )
    print("Registered tool routes:")
    for tool_name, route in agent.health()["tool_routes"].items():
        print(f"  {tool_name}: {route}")
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping backend...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(
        description="Run the local Blender AI Copilot backend."
    )

    parser.add_argument(
        "--project-root",
        default=str(default_project_root()),
        help="Root of blender-ai-copilot project.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--model",
        default="qwen3:4b-instruct",
    )

    args = parser.parse_args()

    run_server(
        project_root=args.project_root,
        host=args.host,
        port=args.port,
        ollama_url=args.ollama_url,
        model=args.model,
    )


if __name__ == "__main__":
    main()
