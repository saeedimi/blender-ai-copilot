"""File bridge between the external copilot backend and Blender."""

from pathlib import Path
import json
import threading
import time
import uuid


RETRY_POLICY = {
    "TRANSIENT": {"retry": True, "max_retries": 2},
    "TIMEOUT": {"retry": True, "max_retries": 2},
    "NOT_FOUND": {"retry": False, "max_retries": 0},
    "VALIDATION": {"retry": False, "max_retries": 0},
    "AUTHORIZATION": {"retry": False, "max_retries": 0},
    "UNKNOWN": {"retry": False, "max_retries": 0},
}


def classify_error(error_message):
    if not error_message:
        return None

    text = str(error_message).lower()

    if any(
        phrase in text
        for phrase in (
            "temporary",
            "temporarily",
            "connection reset",
            "service unavailable",
        )
    ):
        return "TRANSIENT"

    if any(
        phrase in text
        for phrase in (
            "timeout",
            "timed out",
            "did not respond",
        )
    ):
        return "TIMEOUT"

    if any(
        phrase in text
        for phrase in (
            "not found",
            "does not exist",
        )
    ):
        return "NOT_FOUND"

    if any(
        phrase in text
        for phrase in (
            "missing required",
            "unexpected argument",
            "should be",
            "invalid",
        )
    ):
        return "VALIDATION"

    if any(
        phrase in text
        for phrase in (
            "not authorized",
            "permission denied",
            "approval required",
        )
    ):
        return "AUTHORIZATION"

    return "UNKNOWN"


class BlenderBridge:
    """Serialized file-based RPC bridge used by the backend."""

    def __init__(
        self,
        project_root,
        timeout=15.0,
        poll_interval=0.10,
    ):
        self.project_root = Path(project_root).expanduser().resolve()
        self.bridge_dir = self.project_root / "bridge"
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

        self.command_file = self.bridge_dir / "command.json"
        self.result_file = self.bridge_dir / "result.json"

        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)

        # Only one bridge command at a time. Blender is a single interactive app.
        self._lock = threading.Lock()

    def _atomic_write_json(self, path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")

        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        temporary.replace(path)

    def send(
        self,
        command,
        command_id=None,
        timeout=None,
    ):
        with self._lock:
            command_id = str(
                command_id
                or uuid.uuid4()
            )

            payload = {
                "id": command_id,
                "timestamp": time.time(),
                "tool": command["tool"],
                "arguments": command.get("arguments", {}),
                # The Blender extension independently checks this for delete_object.
                "approved_high_risk": bool(
                    command.get("approved_high_risk", False)
                ),
            }

            self._atomic_write_json(self.command_file, payload)

            effective_timeout = (
                float(timeout)
                if timeout is not None
                else self.timeout
            )

            deadline = (
                time.monotonic()
                + effective_timeout
            )

            while time.monotonic() < deadline:
                if self.result_file.exists():
                    try:
                        data = json.loads(
                            self.result_file.read_text(encoding="utf-8")
                        )
                    except (json.JSONDecodeError, OSError):
                        data = None

                    if data and data.get("id") == command_id:
                        result = data.get("result")

                        if not isinstance(result, dict):
                            raise RuntimeError(
                                "Blender returned a malformed result payload."
                            )

                        return result

                time.sleep(self.poll_interval)

            raise TimeoutError(
                "Blender did not respond within "
                f"{effective_timeout:.1f} seconds."
            )

    def execute_with_retry(self, command):
        errors = []
        max_total_attempts = 1

        tool_name = str(
            command.get(
                "tool",
                "",
            )
        )

        render_mode = (
            tool_name
            == "render_scene"
        )

        # The controller may provide a stable bridge ID for a side-effecting
        # operation. For render_scene this is derived from the user request's
        # trace ID, so the same logical render cannot become a new Blender
        # command merely because of transport behavior.
        stable_command_id = (
            command.get(
                "_command_id"
            )
            if render_mode
            else None
        )

        while True:
            attempt = len(errors) + 1

            result = None

            try:
                started = time.perf_counter()

                result = self.send(
                    command,
                    command_id=(
                        stable_command_id
                        if render_mode
                        else None
                    ),
                    timeout=(
                        600.0
                        if render_mode
                        else None
                    ),
                )

                latency = (
                    time.perf_counter()
                    - started
                )

                if result.get("success") is True:
                    return {
                        "success": True,
                        "route": "blender",
                        "tool_result": result,
                        "attempts": attempt,
                        "retry_errors": errors,
                        "error_type": None,
                        "recovered": bool(errors),
                        "latency": latency,
                    }

                error_message = result.get(
                    "error",
                    "Blender tool returned success=False.",
                )

            except Exception as exc:
                latency = (
                    time.perf_counter()
                    - started
                    if "started" in locals()
                    else 0.0
                )
                error_message = str(exc)

            error_type = (
                classify_error(
                    error_message
                )
                or "UNKNOWN"
            )

            if render_mode:
                # CRITICAL:
                # Never automatically replay render_scene. A timeout means
                # execution status may be unknown; Blender may still be
                # rendering. Replaying would create another image.
                policy = {
                    "retry": False,
                    "max_retries": 0,
                }
            else:
                policy = RETRY_POLICY.get(
                    error_type,
                    RETRY_POLICY[
                        "UNKNOWN"
                    ],
                )

            max_total_attempts = (
                1
                + int(
                    policy.get(
                        "max_retries",
                        0,
                    )
                )
            )

            errors.append(
                {
                    "attempt": attempt,
                    "error": error_message,
                    "error_type": error_type,
                }
            )

            if (
                not policy.get("retry", False)
                or attempt >= max_total_attempts
            ):
                if isinstance(
                    result,
                    dict,
                ):
                    failure_payload = dict(
                        result
                    )

                    failure_payload.setdefault(
                        "success",
                        False,
                    )

                    failure_payload.setdefault(
                        "error",
                        error_message,
                    )
                else:
                    failure_payload = {
                        "success": False,
                        "error": error_message,
                    }

                return {
                    "success": False,
                    "route": "blender",
                    "tool_result": (
                        failure_payload
                    ),
                    "attempts": attempt,
                    "retry_errors": errors,
                    "error_type": error_type,
                    "recovered": False,
                    "latency": latency,
                }

            time.sleep(min(0.5 * attempt, 1.0))
