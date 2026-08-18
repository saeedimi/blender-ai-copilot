"""Routes knowledge tools locally and Blender tools through the file bridge."""

import time


class ToolRouter:
    def __init__(self, rag_retriever, blender_bridge):
        self.rag = rag_retriever
        self.bridge = blender_bridge

        self.local_tools = {
            "search_blender_docs": self._search_blender_docs,
        }

        self.blender_tools = {
            "get_scene_objects",
            "get_materials",
            "create_cube",
            "move_object",
            "delete_object",
            "create_material",
            "set_material_color",
            "assign_material",
            "get_modifiers",
            "add_bevel_modifier",
            "set_bevel_modifier",
            "add_subdivision_modifier",
            "set_subdivision_modifier",
            "remove_modifier",
            "apply_modifier",
            "get_cameras",
            "create_camera",
            "move_camera",
            "set_camera_lens",
            "set_active_camera",
            "aim_camera_at_object",
            "get_lights",
            "create_light",
            "move_light",
            "set_light_energy",
            "set_light_color",
            "set_area_light_size",
            "aim_light_at_object",
            "get_render_settings",
            "set_render_engine",
            "set_render_resolution",
            "set_render_samples",
            "set_render_output",
            "set_render_transparent",
            "render_scene",
            "get_mesh_info",
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
            "get_mesh_regions",
            "inset_top_face",
            "subdivide_mesh",
            "translate_top_face",
            "scale_top_face",
            "merge_by_distance",
            "solidify_mesh",
        }

    def route_name(self, tool_name):
        if tool_name in self.local_tools:
            return "local"

        if tool_name in self.blender_tools:
            return "blender"

        return "unknown"

    def _search_blender_docs(self, arguments):
        return self.rag.search(
            query=arguments["query"],
            top_k=arguments.get("top_k", 5),
        )

    def execute(self, command):
        tool_name = command["tool"]
        route = self.route_name(tool_name)

        if route == "local":
            started = time.perf_counter()

            try:
                result = self.local_tools[tool_name](
                    command.get("arguments", {})
                )

                return {
                    "success": result.get("success") is True,
                    "route": "local",
                    "tool_result": result,
                    "attempts": 1,
                    "retry_errors": [],
                    "error_type": None,
                    "recovered": False,
                    "latency": time.perf_counter() - started,
                }

            except Exception as exc:
                return {
                    "success": False,
                    "route": "local",
                    "tool_result": {
                        "success": False,
                        "error": str(exc),
                    },
                    "attempts": 1,
                    "retry_errors": [],
                    "error_type": "LOCAL_TOOL_ERROR",
                    "recovered": False,
                    "latency": time.perf_counter() - started,
                }

        if route == "blender":
            return self.bridge.execute_with_retry(command)

        return {
            "success": False,
            "route": "unknown",
            "tool_result": {
                "success": False,
                "error": f"Unknown tool route: {tool_name}",
            },
            "attempts": 0,
            "retry_errors": [],
            "error_type": "VALIDATION",
            "recovered": False,
            "latency": 0.0,
        }
