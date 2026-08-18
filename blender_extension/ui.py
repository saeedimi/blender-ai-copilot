import json
"""3D View sidebar UI for Blender AI Copilot."""

import textwrap

import bpy
from bpy.props import EnumProperty, StringProperty


class COPILOT_PG_chat_message(bpy.types.PropertyGroup):
    role: EnumProperty(
        name="Role",
        items=(
            ("USER", "You", "User request"),
            ("ASSISTANT", "Copilot", "Copilot response"),
            ("SYSTEM", "System", "System/status message"),
        ),
        default="ASSISTANT",
    )
    text: StringProperty(name="Text", default="")
    status: StringProperty(name="Status", default="")
    trace_id: StringProperty(name="Trace ID", default="")
    timestamp: StringProperty(name="Timestamp", default="")


def draw_wrapped_text(layout, text, width=44, empty_text=""):
    text = (text or "").strip()

    if not text:
        if empty_text:
            layout.label(text=empty_text)
        return

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()

        if not paragraph:
            layout.separator(factor=0.30)
            continue

        for line in textwrap.wrap(
            paragraph,
            width=width,
            replace_whitespace=False,
            break_long_words=False,
        ):
            layout.label(text=line)


def _draw_message(layout, message):
    box = layout.box()
    header = box.row(align=True)

    if message.role == "USER":
        header.label(text="You", icon="USER")
    elif message.role == "SYSTEM":
        header.label(text="System", icon="INFO")
    else:
        header.label(text="Copilot", icon="OUTLINER_OB_LIGHT")

    if message.status:
        header.label(text=message.status)

    draw_wrapped_text(
        box,
        message.text,
        width=42,
    )

    if message.trace_id:
        trace_row = box.row()
        trace_row.scale_y = 0.75
        trace_row.label(
            text=f"Trace: {message.trace_id[:8]}…"
        )


class VIEW3D_PT_blender_ai_copilot(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_blender_ai_copilot"
    bl_label = "Blender AI Copilot"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Copilot"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        status_box = layout.box()
        status_box.label(text=f"Status: {scene.copilot_status}")

        if not getattr(bpy.app, "online_access", True):
            warning = layout.box()
            warning.alert = True
            warning.label(text="Online Access is disabled")
            warning.label(text="Enable it in Preferences > System")

        prompt_box = layout.box()
        prompt_box.label(text="Message Copilot")
        prompt_box.prop(scene, "copilot_prompt", text="")

        row = prompt_box.row(align=True)
        row.enabled = not scene.copilot_busy
        row.operator("copilot.send", text="Send", icon="PLAY")
        row.operator("copilot.check_backend", text="Check")

        if scene.copilot_busy:
            prompt_box.label(text="Working…")

        if scene.copilot_pending_approval_id:
            approval_box = layout.box()
            approval_box.alert = True
            approval_box.label(text="High-risk action requires approval")
            draw_wrapped_text(
                approval_box,
                scene.copilot_pending_action,
                width=38,
            )
            row = approval_box.row(align=True)
            row.enabled = not scene.copilot_busy
            row.operator("copilot.approve", text="Approve", icon="CHECKMARK")
            row.operator("copilot.reject", text="Reject", icon="CANCEL")

        history_box = layout.box()
        header = history_box.row(align=True)
        header.label(text="Conversation")
        header.operator("copilot.clear_chat", text="", icon="TRASH")

        history_box.label(text="Newest first · scroll for earlier requests")

        if len(scene.copilot_chat_history) == 0:
            history_box.label(text="No messages yet.")
        else:
            # Newest first keeps the current result visible in Blender's narrow
            # sidebar; the panel itself remains vertically scrollable.
            for index in range(len(scene.copilot_chat_history) - 1, -1, -1):
                _draw_message(
                    history_box,
                    scene.copilot_chat_history[index],
                )

        context_box = layout.box()
        context_box.label(text="Conversation Context", icon="INFO")
        context_box.prop(
            scene,
            "copilot_context_enabled",
            text="Use history context",
        )

        row = context_box.row()
        row.enabled = scene.copilot_context_enabled
        row.prop(
            scene,
            "copilot_context_turns",
            text="Recent turns",
        )

        if scene.copilot_context_enabled:
            context_box.label(
                text=(
                    f"Qwen gets last {scene.copilot_context_turns} turns "
                    "+ compact memory"
                )
            )
            if scene.copilot_memory_summary.strip():
                fact_count = len(
                    [
                        line
                        for line in scene.copilot_memory_summary.splitlines()
                        if line.strip()
                    ]
                )
                context_box.label(
                    text=f"Memory facts: {fact_count}"
                )
            else:
                context_box.label(text="Memory facts: 0")

            try:
                reference_memory = json.loads(
                    scene.copilot_reference_memory or "{}"
                )
            except Exception:
                reference_memory = {}

            last_type = reference_memory.get("last_entity_type")
            last_name = reference_memory.get("last_entity_name")
            if last_type and last_name:
                context_box.label(
                    text=f"Last referent: {last_type} · {last_name}"
                )

        settings_box = layout.box()
        settings_box.label(text="Connection")
        settings_box.prop(scene, "copilot_backend_url", text="Backend")
        settings_box.prop(scene, "copilot_project_root", text="Project")
        settings_box.label(text="History + compact memory are stored with the scene")


CLASSES = (
    VIEW3D_PT_blender_ai_copilot,
)
