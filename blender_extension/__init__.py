"""Blender AI Copilot Extension."""

import os

import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty

from . import operators
from . import ui


def register_properties():
    bpy.types.Scene.copilot_prompt = StringProperty(
        name="Prompt",
        description="Request for the Blender AI Copilot",
        default="",
    )

    # Kept for compatibility with previous versions and scripting access.
    bpy.types.Scene.copilot_response = StringProperty(
        name="Latest Response",
        default="",
    )

    bpy.types.Scene.copilot_chat_history = CollectionProperty(
        name="Copilot Chat History",
        type=ui.COPILOT_PG_chat_message,
    )

    bpy.types.Scene.copilot_context_enabled = BoolProperty(
        name="Use Conversation Context",
        description=(
            "Send a bounded recent conversation window and compact memory "
            "summary to the local model"
        ),
        default=True,
    )

    bpy.types.Scene.copilot_context_turns = IntProperty(
        name="Recent Turns",
        description="Number of recent request/response turns sent directly to Qwen",
        default=3,
        min=1,
        max=6,
    )

    bpy.types.Scene.copilot_memory_summary = StringProperty(
        name="Compact Conversation Memory",
        description="Bounded persistent memory generated from verified Copilot actions",
        default="",
        options={"HIDDEN"},
    )

    bpy.types.Scene.copilot_reference_memory = StringProperty(
        name="Structured Referential Memory",
        description=(
            "Controller-owned JSON memory for the most recently referenced "
            "object, material, camera, and light"
        ),
        default="{}",
        options={"HIDDEN"},
    )

    bpy.types.Scene.copilot_status = StringProperty(
        name="Status",
        default="Not connected",
    )

    bpy.types.Scene.copilot_backend_url = StringProperty(
        name="Backend URL",
        description="Local Blender AI Copilot backend",
        default="http://127.0.0.1:8765",
    )

    bpy.types.Scene.copilot_project_root = StringProperty(
        name="Project Root",
        description="Root folder containing src/, bridge/, rag/, and notebook/",
        subtype="DIR_PATH",
        default=os.environ.get("BLENDER_COPILOT_PROJECT_ROOT", ""),
    )

    bpy.types.Scene.copilot_busy = BoolProperty(
        name="Busy",
        default=False,
        options={"HIDDEN"},
    )

    bpy.types.Scene.copilot_pending_approval_id = StringProperty(
        name="Pending Approval ID",
        default="",
        options={"HIDDEN"},
    )

    bpy.types.Scene.copilot_pending_action = StringProperty(
        name="Pending Action",
        default="",
        options={"HIDDEN"},
    )


def unregister_properties():
    names = (
        "copilot_prompt",
        "copilot_response",
        "copilot_chat_history",
        "copilot_context_enabled",
        "copilot_context_turns",
        "copilot_memory_summary",
        "copilot_reference_memory",
        "copilot_status",
        "copilot_backend_url",
        "copilot_project_root",
        "copilot_busy",
        "copilot_pending_approval_id",
        "copilot_pending_action",
    )

    for name in names:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)


def register():
    bpy.utils.register_class(ui.COPILOT_PG_chat_message)
    register_properties()

    for cls in (*operators.CLASSES, *ui.CLASSES):
        bpy.utils.register_class(cls)

    operators.register_runtime()
    print("[Blender AI Copilot] Extension registered.")


def unregister():
    operators.unregister_runtime()

    for cls in reversed((*operators.CLASSES, *ui.CLASSES)):
        bpy.utils.unregister_class(cls)

    unregister_properties()
    bpy.utils.unregister_class(ui.COPILOT_PG_chat_message)
    print("[Blender AI Copilot] Extension unregistered.")
