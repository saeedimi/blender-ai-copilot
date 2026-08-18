"""Small standard-library HTTP client for the local copilot backend."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CopilotClientError(RuntimeError):
    pass


def _request_json(
    base_url,
    path,
    payload=None,
    timeout=180,
):
    url = base_url.rstrip("/") + path

    data = None
    method = "GET"

    if payload is not None:
        method = "POST"
        data = json.dumps(payload).encode("utf-8")

    request = Request(
        url=url,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)

    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)

        raise CopilotClientError(
            f"Backend HTTP {exc.code}: {detail}"
        ) from exc

    except URLError as exc:
        raise CopilotClientError(
            f"Could not reach the copilot backend: {exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise CopilotClientError(
            "Copilot backend request timed out."
        ) from exc


def health(base_url, timeout=5):
    return _request_json(
        base_url,
        "/health",
        timeout=timeout,
    )


def chat(
    base_url,
    message,
    conversation_context=None,
    timeout=900,
):
    payload = {
        "message": message,
    }

    if conversation_context:
        payload["conversation_context"] = conversation_context

    return _request_json(
        base_url,
        "/chat",
        payload=payload,
        timeout=timeout,
    )


def approve(
    base_url,
    approval_id,
    approved,
    timeout=180,
):
    return _request_json(
        base_url,
        "/approve",
        payload={
            "approval_id": approval_id,
            "approved": bool(approved),
        },
        timeout=timeout,
    )
