"""Blender operators, async client workers, and the controlled bridge listener."""

from pathlib import Path
import json
import os
import queue
import threading
import time

import bpy
import bmesh
from mathutils import Vector

from . import client


_RESULT_QUEUE = queue.Queue()
_LAST_COMMAND_ID = None
_BACKEND_PROJECT_ROOT = None

# Bounded command-result ledger. If the backend ever presents the same
# command ID again, Blender returns the original result instead of executing
# the side effect again.
_COMMAND_RESULT_CACHE = {}
_COMMAND_RESULT_ORDER = []
_COMMAND_RESULT_CACHE_LIMIT = 128


def _cache_command_result(
    command_id,
    result,
):
    command_id = str(
        command_id
    )

    if command_id not in (
        _COMMAND_RESULT_CACHE
    ):
        _COMMAND_RESULT_ORDER.append(
            command_id
        )

    _COMMAND_RESULT_CACHE[
        command_id
    ] = result

    while (
        len(
            _COMMAND_RESULT_ORDER
        )
        > _COMMAND_RESULT_CACHE_LIMIT
    ):
        oldest = (
            _COMMAND_RESULT_ORDER.pop(
                0
            )
        )

        _COMMAND_RESULT_CACHE.pop(
            oldest,
            None,
        )


def _cached_command_result(
    command_id,
):
    return (
        _COMMAND_RESULT_CACHE.get(
            str(command_id)
        )
    )


def _tag_redraw():
    window_manager = bpy.context.window_manager

    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = window.screen

        if screen is None:
            continue

        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _online_access_allowed():
    # Blender's extension guidance asks network-using add-ons to honor this.
    return bool(getattr(bpy.app, "online_access", True))


def _set_backend_project_root(value):
    global _BACKEND_PROJECT_ROOT

    if value:
        _BACKEND_PROJECT_ROOT = str(value)


def _resolve_project_root(scene=None):
    if scene is not None:
        configured = getattr(
            scene,
            "copilot_project_root",
            "",
        ).strip()

        if configured:
            return Path(
                bpy.path.abspath(configured)
            ).expanduser().resolve()

    if _BACKEND_PROJECT_ROOT:
        return Path(
            _BACKEND_PROJECT_ROOT
        ).expanduser().resolve()

    env_value = os.environ.get(
        "BLENDER_COPILOT_PROJECT_ROOT"
    )

    if env_value:
        return Path(
            env_value
        ).expanduser().resolve()

    return None


def _atomic_write_json(path, payload):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def _tool_get_scene_objects(arguments):
    objects = []

    for obj in bpy.context.scene.objects:
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "location": [
                    float(obj.location.x),
                    float(obj.location.y),
                    float(obj.location.z),
                ],
                "materials": (
                    [
                        material.name
                        for material in getattr(obj.data, "materials", [])
                        if material is not None
                    ]
                    if getattr(obj, "data", None) is not None
                    and hasattr(obj.data, "materials")
                    else []
                ),
            }
        )

    return {
        "success": True,
        "objects": objects,
    }


def _tool_create_cube(arguments):
    name = arguments["name"]

    if name in bpy.data.objects:
        return {
            "success": False,
            "error": (
                f"Object '{name}' already exists. "
                "Choose another name or use move_object."
            ),
        }

    x = float(arguments["x"])
    y = float(arguments["y"])
    z = float(arguments["z"])

    bpy.ops.mesh.primitive_cube_add(
        location=(x, y, z)
    )

    obj = bpy.context.active_object
    obj.name = name

    return {
        "success": True,
        "name": obj.name,
        "type": obj.type,
        "location": [
            float(obj.location.x),
            float(obj.location.y),
            float(obj.location.z),
        ],
    }


def _tool_move_object(arguments):
    name = arguments["name"]
    obj = bpy.data.objects.get(name)

    if obj is None:
        return {
            "success": False,
            "error": f"Object '{name}' was not found.",
        }

    obj.location = (
        float(arguments["x"]),
        float(arguments["y"]),
        float(arguments["z"]),
    )

    bpy.context.view_layer.update()

    return {
        "success": True,
        "name": obj.name,
        "location": [
            float(obj.location.x),
            float(obj.location.y),
            float(obj.location.z),
        ],
    }


def _rgba_from_arguments(arguments):
    return (
        float(arguments["r"]),
        float(arguments["g"]),
        float(arguments["b"]),
        float(arguments["a"]),
    )


def _set_principled_base_color(material, rgba):
    """Keep the datablock display color and Principled BSDF base color aligned."""
    material.diffuse_color = rgba
    material.use_nodes = True

    node_tree = material.node_tree
    if node_tree is None:
        return

    principled = next(
        (
            node
            for node in node_tree.nodes
            if node.type == "BSDF_PRINCIPLED"
        ),
        None,
    )

    if principled is None:
        return

    base_color = principled.inputs.get("Base Color")
    if base_color is not None:
        base_color.default_value = rgba

    alpha_input = principled.inputs.get("Alpha")
    if alpha_input is not None:
        alpha_input.default_value = rgba[3]


def _tool_get_materials(arguments):
    materials = []

    for material in bpy.data.materials:
        users = []

        for obj in bpy.context.scene.objects:
            data = getattr(obj, "data", None)

            if data is None or not hasattr(data, "materials"):
                continue

            if any(
                slot_material == material
                for slot_material in data.materials
            ):
                users.append(obj.name)

        materials.append(
            {
                "name": material.name,
                "base_color": [
                    float(value)
                    for value in material.diffuse_color
                ],
                "use_nodes": bool(material.use_nodes),
                "objects": users,
            }
        )

    return {
        "success": True,
        "materials": materials,
    }


def _tool_create_material(arguments):
    name = arguments["name"]

    if bpy.data.materials.get(name) is not None:
        return {
            "success": False,
            "error": f"Material '{name}' already exists.",
        }

    material = bpy.data.materials.new(name=name)
    rgba = _rgba_from_arguments(arguments)
    _set_principled_base_color(material, rgba)

    return {
        "success": True,
        "name": material.name,
        "base_color": list(rgba),
    }


def _tool_set_material_color(arguments):
    material_name = arguments["material_name"]
    material = bpy.data.materials.get(material_name)

    if material is None:
        return {
            "success": False,
            "error": f"Material '{material_name}' was not found.",
        }

    rgba = _rgba_from_arguments(arguments)
    _set_principled_base_color(material, rgba)

    return {
        "success": True,
        "name": material.name,
        "base_color": list(rgba),
    }


def _tool_assign_material(arguments):
    object_name = arguments["object_name"]
    material_name = arguments["material_name"]

    obj = bpy.data.objects.get(object_name)

    if obj is None:
        return {
            "success": False,
            "error": f"Object '{object_name}' was not found.",
        }

    if obj.type != "MESH":
        return {
            "success": False,
            "error": (
                f"Object '{object_name}' is type {obj.type}, not MESH. "
                "This controlled tool currently assigns materials only to mesh objects."
            ),
        }

    material = bpy.data.materials.get(material_name)

    if material is None:
        return {
            "success": False,
            "error": f"Material '{material_name}' was not found.",
        }

    slots = obj.data.materials

    existing_index = next(
        (
            index
            for index, current in enumerate(slots)
            if current == material
        ),
        None,
    )

    if existing_index is None:
        slots.append(material)
        existing_index = len(slots) - 1

    obj.active_material_index = existing_index

    return {
        "success": True,
        "object_name": obj.name,
        "material_name": material.name,
        "material_slot": existing_index,
    }


def _get_mesh_object(object_name):
    obj = bpy.data.objects.get(object_name)
    if obj is None:
        return None, {"success": False, "error": f"Object '{object_name}' was not found."}
    if obj.type != "MESH":
        return None, {
            "success": False,
            "error": (
                f"Object '{object_name}' is type {obj.type}, not MESH. "
                "Controlled modifier tools currently support mesh objects."
            ),
        }
    return obj, None


def _modifier_to_dict(modifier):
    data = {
        "name": modifier.name,
        "type": modifier.type,
        "show_viewport": bool(modifier.show_viewport),
        "show_render": bool(modifier.show_render),
    }
    if modifier.type == "BEVEL":
        data.update({
            "width": float(modifier.width),
            "segments": int(modifier.segments),
            "limit_method": str(modifier.limit_method),
        })
    elif modifier.type == "SUBSURF":
        data.update({
            "levels": int(modifier.levels),
            "render_levels": int(modifier.render_levels),
            "subdivision_type": str(modifier.subdivision_type),
        })
    return data


def _tool_get_modifiers(arguments):
    object_name = arguments.get("object_name")
    if object_name:
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            return {"success": False, "error": f"Object '{object_name}' was not found."}
        objects = [obj]
    else:
        objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    return {
        "success": True,
        "objects": [
            {
                "object_name": obj.name,
                "object_type": obj.type,
                "modifiers": [_modifier_to_dict(mod) for mod in obj.modifiers],
            }
            for obj in objects
        ],
    }


def _tool_add_bevel_modifier(arguments):
    obj, error = _get_mesh_object(arguments["object_name"])
    if error:
        return error
    name = arguments["modifier_name"]
    if obj.modifiers.get(name) is not None:
        return {"success": False, "error": f"Modifier '{name}' already exists on '{obj.name}'."}
    modifier = obj.modifiers.new(name=name, type="BEVEL")
    modifier.width = float(arguments["width"])
    modifier.segments = int(arguments["segments"])
    return {"success": True, "object_name": obj.name, "modifier": _modifier_to_dict(modifier)}


def _tool_set_bevel_modifier(arguments):
    obj, error = _get_mesh_object(arguments["object_name"])
    if error:
        return error
    modifier = obj.modifiers.get(arguments["modifier_name"])
    if modifier is None:
        return {"success": False, "error": f"Modifier '{arguments['modifier_name']}' was not found on '{obj.name}'."}
    if modifier.type != "BEVEL":
        return {"success": False, "error": f"Modifier '{modifier.name}' is type {modifier.type}, not BEVEL."}
    modifier.width = float(arguments["width"])
    modifier.segments = int(arguments["segments"])
    return {"success": True, "object_name": obj.name, "modifier": _modifier_to_dict(modifier)}


def _tool_add_subdivision_modifier(arguments):
    obj, error = _get_mesh_object(arguments["object_name"])
    if error:
        return error
    name = arguments["modifier_name"]
    if obj.modifiers.get(name) is not None:
        return {"success": False, "error": f"Modifier '{name}' already exists on '{obj.name}'."}
    modifier = obj.modifiers.new(name=name, type="SUBSURF")
    modifier.levels = int(arguments["levels"])
    modifier.render_levels = int(arguments["render_levels"])
    return {"success": True, "object_name": obj.name, "modifier": _modifier_to_dict(modifier)}


def _tool_set_subdivision_modifier(arguments):
    obj, error = _get_mesh_object(arguments["object_name"])
    if error:
        return error
    modifier = obj.modifiers.get(arguments["modifier_name"])
    if modifier is None:
        return {"success": False, "error": f"Modifier '{arguments['modifier_name']}' was not found on '{obj.name}'."}
    if modifier.type != "SUBSURF":
        return {"success": False, "error": f"Modifier '{modifier.name}' is type {modifier.type}, not SUBSURF."}
    modifier.levels = int(arguments["levels"])
    modifier.render_levels = int(arguments["render_levels"])
    return {"success": True, "object_name": obj.name, "modifier": _modifier_to_dict(modifier)}


def _tool_remove_modifier(arguments):
    obj, error = _get_mesh_object(arguments["object_name"])
    if error:
        return error
    name = arguments["modifier_name"]
    modifier = obj.modifiers.get(name)
    if modifier is None:
        return {"success": False, "error": f"Modifier '{name}' was not found on '{obj.name}'."}
    modifier_type = modifier.type
    obj.modifiers.remove(modifier)
    return {"success": True, "object_name": obj.name, "removed_modifier": name, "removed_type": modifier_type}


def _tool_apply_modifier(arguments, approved_high_risk=False):
    if not approved_high_risk:
        return {
            "success": False,
            "error": "Apply modifier was blocked because the backend did not include high-risk approval.",
        }

    obj, error = _get_mesh_object(arguments["object_name"])
    if error:
        return error
    name = arguments["modifier_name"]
    if obj.modifiers.get(name) is None:
        return {"success": False, "error": f"Modifier '{name}' was not found on '{obj.name}'."}

    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)

    try:
        for selected in list(bpy.context.selected_objects):
            selected.select_set(False)
        obj.select_set(True)
        view_layer.objects.active = obj
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        result = bpy.ops.object.modifier_apply(modifier=name)
        if "FINISHED" not in result:
            return {"success": False, "error": f"Blender did not finish applying modifier '{name}'."}
        return {"success": True, "object_name": obj.name, "applied_modifier": name}
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to apply modifier '{name}': {type(exc).__name__}: {exc}",
        }
    finally:
        try:
            for selected in list(bpy.context.selected_objects):
                selected.select_set(False)
            for selected in previous_selected:
                if selected.name in bpy.data.objects:
                    selected.select_set(True)
            if previous_active is not None and previous_active.name in bpy.data.objects:
                view_layer.objects.active = previous_active
        except Exception:
            pass


def _vector3(values):
    return [float(values[0]), float(values[1]), float(values[2])]


def _rotation3(obj):
    return [float(obj.rotation_euler.x), float(obj.rotation_euler.y), float(obj.rotation_euler.z)]


def _get_camera_object(camera_name):
    obj = bpy.data.objects.get(camera_name)
    if obj is None:
        return None, {"success": False, "error": f"Camera object '{camera_name}' was not found."}
    if obj.type != "CAMERA":
        return None, {"success": False, "error": f"Object '{camera_name}' is type {obj.type}, not CAMERA."}
    return obj, None


def _camera_to_dict(obj):
    data = obj.data
    return {
        "name": obj.name,
        "location": _vector3(obj.location),
        "rotation_euler": _rotation3(obj),
        "camera_type": str(data.type),
        "lens_mm": float(data.lens),
        "clip_start": float(data.clip_start),
        "clip_end": float(data.clip_end),
        "active_scene_camera": bool(bpy.context.scene.camera == obj),
    }


def _tool_get_cameras(arguments):
    cameras = [_camera_to_dict(obj) for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    return {
        "success": True,
        "cameras": cameras,
        "active_camera": bpy.context.scene.camera.name if bpy.context.scene.camera else None,
    }


def _tool_create_camera(arguments):
    name = arguments["name"]

    existing = bpy.data.objects.get(
        name
    )

    if existing is not None:
        if existing.type != "CAMERA":
            return {
                "success": False,
                "error": (
                    f"Object '{name}' already exists and is type "
                    f"{existing.type}, not CAMERA."
                ),
            }

        # Same-name CAMERA: reconcile it to the requested state.
        # This prevents the agent from silently using a stale default camera.
        existing.data.type = "PERSP"
        existing.data.lens = float(
            arguments["lens_mm"]
        )
        existing.location = (
            float(arguments["x"]),
            float(arguments["y"]),
            float(arguments["z"]),
        )

        bpy.context.view_layer.update()

        return {
            "success": True,
            "created": False,
            "reused_existing": True,
            "camera": _camera_to_dict(
                existing
            ),
        }

    data = bpy.data.cameras.new(
        name=f"{name}_Data"
    )
    data.type = "PERSP"
    data.lens = float(
        arguments["lens_mm"]
    )

    obj = bpy.data.objects.new(
        name,
        data,
    )

    bpy.context.collection.objects.link(
        obj
    )

    obj.location = (
        float(arguments["x"]),
        float(arguments["y"]),
        float(arguments["z"]),
    )

    bpy.context.view_layer.update()

    return {
        "success": True,
        "created": True,
        "reused_existing": False,
        "camera": _camera_to_dict(
            obj
        ),
    }


def _tool_move_camera(arguments):
    obj, error = _get_camera_object(arguments["camera_name"])
    if error:
        return error
    obj.location = (float(arguments["x"]), float(arguments["y"]), float(arguments["z"]))
    bpy.context.view_layer.update()
    return {"success": True, "camera": _camera_to_dict(obj)}


def _tool_set_camera_lens(arguments):
    obj, error = _get_camera_object(arguments["camera_name"])
    if error:
        return error
    obj.data.type = "PERSP"
    obj.data.lens = float(arguments["lens_mm"])
    return {"success": True, "camera": _camera_to_dict(obj)}


def _tool_set_active_camera(arguments):
    obj, error = _get_camera_object(arguments["camera_name"])
    if error:
        return error
    bpy.context.scene.camera = obj
    return {"success": True, "active_camera": obj.name, "camera": _camera_to_dict(obj)}


def _aim_object_negative_z_at_point(obj, target_location):
    direction = Vector(target_location) - obj.location
    if direction.length < 1e-8:
        return {"success": False, "error": f"Cannot aim '{obj.name}' because it is at the same location as the target."}
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    return None


def _tool_aim_camera_at_object(arguments):
    camera, error = _get_camera_object(arguments["camera_name"])
    if error:
        return error
    target = bpy.data.objects.get(arguments["target_object_name"])
    if target is None:
        return {"success": False, "error": f"Target object '{arguments['target_object_name']}' was not found."}
    aim_error = _aim_object_negative_z_at_point(camera, target.matrix_world.translation)
    if aim_error:
        return aim_error
    return {"success": True, "camera": _camera_to_dict(camera), "target_object_name": target.name}


def _get_light_object(light_name):
    obj = bpy.data.objects.get(light_name)
    if obj is None:
        return None, {"success": False, "error": f"Light object '{light_name}' was not found."}
    if obj.type != "LIGHT":
        return None, {"success": False, "error": f"Object '{light_name}' is type {obj.type}, not LIGHT."}
    return obj, None


def _light_to_dict(obj):
    light = obj.data
    result = {
        "name": obj.name,
        "light_type": str(light.type),
        "location": _vector3(obj.location),
        "rotation_euler": _rotation3(obj),
        "energy": float(light.energy),
        "color": [float(light.color[0]), float(light.color[1]), float(light.color[2])],
    }
    if light.type == "AREA":
        result["size"] = float(light.size)
        result["shape"] = str(light.shape)
    if hasattr(light, "shadow_soft_size"):
        result["shadow_soft_size"] = float(light.shadow_soft_size)
    if light.type == "SPOT":
        result["spot_size"] = float(light.spot_size)
        result["spot_blend"] = float(light.spot_blend)
    return result


def _tool_get_lights(arguments):
    return {
        "success": True,
        "lights": [_light_to_dict(obj) for obj in bpy.context.scene.objects if obj.type == "LIGHT"],
    }


def _tool_create_light(arguments):
    name = arguments["name"]
    if bpy.data.objects.get(name) is not None:
        return {"success": False, "error": f"Object '{name}' already exists."}

    light_type = str(arguments["light_type"]).upper()
    data = bpy.data.lights.new(name=f"{name}_Data", type=light_type)
    data.energy = float(arguments["energy"])
    data.color = (float(arguments["r"]), float(arguments["g"]), float(arguments["b"]))

    if light_type == "AREA":
        data.size = float(arguments["size"])
    elif hasattr(data, "shadow_soft_size"):
        data.shadow_soft_size = float(arguments["size"])

    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = (float(arguments["x"]), float(arguments["y"]), float(arguments["z"]))
    bpy.context.view_layer.update()
    return {"success": True, "light": _light_to_dict(obj)}


def _tool_move_light(arguments):
    obj, error = _get_light_object(arguments["light_name"])
    if error:
        return error
    obj.location = (float(arguments["x"]), float(arguments["y"]), float(arguments["z"]))
    bpy.context.view_layer.update()
    return {"success": True, "light": _light_to_dict(obj)}


def _tool_set_light_energy(arguments):
    obj, error = _get_light_object(arguments["light_name"])
    if error:
        return error
    obj.data.energy = float(arguments["energy"])
    return {"success": True, "light": _light_to_dict(obj)}


def _tool_set_light_color(arguments):
    obj, error = _get_light_object(arguments["light_name"])
    if error:
        return error
    obj.data.color = (float(arguments["r"]), float(arguments["g"]), float(arguments["b"]))
    return {"success": True, "light": _light_to_dict(obj)}


def _tool_set_area_light_size(arguments):
    obj, error = _get_light_object(arguments["light_name"])
    if error:
        return error
    if obj.data.type != "AREA":
        return {"success": False, "error": f"Light '{obj.name}' is type {obj.data.type}, not AREA."}
    obj.data.size = float(arguments["size"])
    return {"success": True, "light": _light_to_dict(obj)}


def _tool_aim_light_at_object(arguments):
    obj, error = _get_light_object(arguments["light_name"])
    if error:
        return error
    if obj.data.type == "POINT":
        return {
            "success": False,
            "error": "POINT lights emit in all directions and do not need aiming. Use AREA, SPOT, or SUN.",
        }
    target = bpy.data.objects.get(arguments["target_object_name"])
    if target is None:
        return {"success": False, "error": f"Target object '{arguments['target_object_name']}' was not found."}
    aim_error = _aim_object_negative_z_at_point(obj, target.matrix_world.translation)
    if aim_error:
        return aim_error
    return {"success": True, "light": _light_to_dict(obj), "target_object_name": target.name}


def _safe_render_directory():
    project_root = _resolve_project_root(
        bpy.context.scene
    )

    if project_root is None:
        return None

    render_dir = (
        Path(project_root)
        / "renders"
    )

    render_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return render_dir


def _render_samples_info(scene):
    result = {
        "cycles_samples": None,
        "eevee_samples": None,
        "sample_property": None,
    }

    if hasattr(scene, "cycles") and hasattr(scene.cycles, "samples"):
        try:
            result["cycles_samples"] = int(
                scene.cycles.samples
            )
        except Exception:
            pass

    # Blender versions expose EEVEE sampling differently.
    # Handle known RNA layouts without assuming a single version.
    eevee = getattr(
        scene,
        "eevee",
        None,
    )

    if eevee is not None:
        for property_name in (
            "taa_render_samples",
            "render_samples",
        ):
            if hasattr(
                eevee,
                property_name,
            ):
                try:
                    result["eevee_samples"] = int(
                        getattr(
                            eevee,
                            property_name,
                        )
                    )
                    result["sample_property"] = (
                        f"scene.eevee.{property_name}"
                    )
                    break
                except Exception:
                    pass

    # Some newer versions may expose an engine-owned sampling property
    # elsewhere. We report None rather than inventing a setting.
    return result


def _current_render_output(scene):
    filepath = scene.render.filepath

    return {
        "filepath": filepath,
        "absolute_filepath": (
            bpy.path.abspath(
                filepath
            )
            if filepath
            else ""
        ),
        "file_format": str(
            scene.render.image_settings.file_format
        ),
    }


def _tool_get_render_settings(arguments):
    scene = bpy.context.scene

    samples = _render_samples_info(
        scene
    )

    return {
        "success": True,
        "engine": str(
            scene.render.engine
        ),
        "active_camera": (
            scene.camera.name
            if scene.camera is not None
            else None
        ),
        "resolution": {
            "width": int(
                scene.render.resolution_x
            ),
            "height": int(
                scene.render.resolution_y
            ),
            "percentage": int(
                scene.render.resolution_percentage
            ),
        },
        "samples": samples,
        "output": _current_render_output(
            scene
        ),
        "film_transparent": bool(
            scene.render.film_transparent
        ),
    }


def _tool_set_render_engine(arguments):
    scene = bpy.context.scene
    requested = str(
        arguments["engine"]
    ).upper()

    # Validate against what this exact Blender installation exposes.
    engine_property = (
        scene.bl_rna
        .properties["render"]
        .fixed_type
        .properties["engine"]
    )

    available = {
        item.identifier
        for item in engine_property.enum_items
    }

    if requested not in available:
        return {
            "success": False,
            "error": (
                f"Render engine '{requested}' is not available in this "
                f"Blender installation. Available engines: {sorted(available)}"
            ),
        }

    scene.render.engine = requested

    return {
        "success": True,
        "engine": str(
            scene.render.engine
        ),
    }


def _tool_set_render_resolution(arguments):
    scene = bpy.context.scene

    scene.render.resolution_x = int(
        arguments["width"]
    )

    scene.render.resolution_y = int(
        arguments["height"]
    )

    scene.render.resolution_percentage = int(
        arguments["percentage"]
    )

    return {
        "success": True,
        "resolution": {
            "width": int(
                scene.render.resolution_x
            ),
            "height": int(
                scene.render.resolution_y
            ),
            "percentage": int(
                scene.render.resolution_percentage
            ),
        },
    }


def _tool_set_render_samples(arguments):
    scene = bpy.context.scene
    samples = int(
        arguments["samples"]
    )

    engine = str(
        scene.render.engine
    )

    if (
        engine == "CYCLES"
        and hasattr(scene, "cycles")
        and hasattr(scene.cycles, "samples")
    ):
        scene.cycles.samples = samples

        return {
            "success": True,
            "engine": engine,
            "samples": samples,
            "sample_property": "scene.cycles.samples",
        }

    eevee = getattr(
        scene,
        "eevee",
        None,
    )

    if (
        engine in {
            "BLENDER_EEVEE",
            "BLENDER_EEVEE_NEXT",
        }
        and eevee is not None
    ):
        for property_name in (
            "taa_render_samples",
            "render_samples",
        ):
            if hasattr(
                eevee,
                property_name,
            ):
                setattr(
                    eevee,
                    property_name,
                    samples,
                )

                return {
                    "success": True,
                    "engine": engine,
                    "samples": samples,
                    "sample_property": (
                        f"scene.eevee.{property_name}"
                    ),
                }

    return {
        "success": False,
        "error": (
            f"This Blender build does not expose a supported programmable "
            f"sample property for render engine '{engine}'. "
            "Use get_render_settings to inspect current support."
        ),
    }


def _normalized_output_extension(
    file_format,
):
    return {
        "PNG": ".png",
        "JPEG": ".jpg",
        "OPEN_EXR": ".exr",
    }[
        file_format
    ]


def _tool_set_render_output(arguments):
    scene = bpy.context.scene

    render_dir = (
        _safe_render_directory()
    )

    if render_dir is None:
        return {
            "success": False,
            "error": (
                "Project root is unavailable, so a safe render "
                "output directory cannot be created."
            ),
        }

    filename = (
        str(
            arguments[
                "filename"
            ]
        )
        .strip()
    )

    file_format = str(
        arguments[
            "file_format"
        ]
    ).upper()

    # Defense in depth; backend schema already validates this.
    filename = Path(
        filename
    ).name

    extension = (
        _normalized_output_extension(
            file_format
        )
    )

    if filename.lower().endswith(
        extension.lower()
    ):
        stem = filename[
            :-len(extension)
        ]
    else:
        stem = filename

    stem = stem.strip()

    if not stem:
        return {
            "success": False,
            "error": "Render filename is empty after normalization.",
        }

    output_path = (
        render_dir
        / f"{stem}{extension}"
    )

    scene.render.image_settings.file_format = (
        file_format
    )

    scene.render.filepath = str(
        output_path
    )

    return {
        "success": True,
        "output": _current_render_output(
            scene
        ),
        "safe_render_directory": str(
            render_dir
        ),
    }


def _tool_set_render_transparent(arguments):
    scene = bpy.context.scene

    scene.render.film_transparent = bool(
        arguments["enabled"]
    )

    return {
        "success": True,
        "film_transparent": bool(
            scene.render.film_transparent
        ),
    }


def _tool_render_scene(arguments):
    scene = bpy.context.scene

    if scene.camera is None:
        return {
            "success": False,
            "error": (
                "No active scene camera is configured. "
                "Use set_active_camera before rendering."
            ),
        }

    save_to_file = bool(
        arguments["save_to_file"]
    )

    output_path = None

    if save_to_file:
        render_dir = (
            _safe_render_directory()
        )

        if render_dir is None:
            return {
                "success": False,
                "error": (
                    "Project root is unavailable, so saved rendering is blocked."
                ),
            }

        configured_path = Path(
            bpy.path.abspath(
                scene.render.filepath
            )
        ).expanduser()

        try:
            configured_resolved = (
                configured_path.resolve()
            )

            render_dir_resolved = (
                render_dir.resolve()
            )

            configured_resolved.relative_to(
                render_dir_resolved
            )

        except Exception:
            return {
                "success": False,
                "error": (
                    "The configured render output is outside the project's "
                    "safe renders/ directory. Use set_render_output first."
                ),
            }

        if configured_path.name in {
            "",
            ".",
            "..",
        }:
            return {
                "success": False,
                "error": (
                    "No safe render filename is configured. "
                    "Use set_render_output first."
                ),
            }

        # Avoid silent overwrite. If a file exists, create a unique suffix.
        if configured_path.exists():
            stem = configured_path.stem
            suffix = configured_path.suffix

            counter = 1

            candidate = (
                configured_path.parent
                / f"{stem}_{counter}{suffix}"
            )

            while candidate.exists():
                counter += 1
                candidate = (
                    configured_path.parent
                    / f"{stem}_{counter}{suffix}"
                )

            configured_path = candidate
            scene.render.filepath = str(
                configured_path
            )

        output_path = configured_path

    started = time.perf_counter()

    # Stage A: render exactly once.
    #
    # Blender's Render Result image can report a zero size even after a
    # completed render in some contexts, so Image.size is diagnostic only.
    # The controller now verifies render completion from Blender's lifecycle
    # handlers plus the operator result.
    render_state = {
        "complete": False,
        "cancelled": False,
    }

    def _copilot_render_complete(
        *_args,
    ):
        render_state[
            "complete"
        ] = True

    def _copilot_render_cancel(
        *_args,
    ):
        render_state[
            "cancelled"
        ] = True

    complete_handlers = (
        bpy.app.handlers.render_complete
    )
    cancel_handlers = (
        bpy.app.handlers.render_cancel
    )

    complete_handlers.append(
        _copilot_render_complete
    )
    cancel_handlers.append(
        _copilot_render_cancel
    )

    result = None
    render_exception = None

    try:
        result = bpy.ops.render.render(
            write_still=False
        )

    except Exception as exc:
        render_exception = exc

    finally:
        if (
            _copilot_render_complete
            in complete_handlers
        ):
            complete_handlers.remove(
                _copilot_render_complete
            )

        if (
            _copilot_render_cancel
            in cancel_handlers
        ):
            cancel_handlers.remove(
                _copilot_render_cancel
            )

    latency = (
        time.perf_counter()
        - started
    )

    if render_exception is not None:
        return {
            "success": False,
            "failure_stage": "render",
            "render_complete_handler_fired": bool(
                render_state[
                    "complete"
                ]
            ),
            "render_cancel_handler_fired": bool(
                render_state[
                    "cancelled"
                ]
            ),
            "render_latency": latency,
            "error": (
                f"Render failed before saving: "
                f"{type(render_exception).__name__}: "
                f"{render_exception}"
            ),
        }

    operator_finished = (
        isinstance(
            result,
            set,
        )
        and "FINISHED"
        in result
    )

    if not operator_finished:
        return {
            "success": False,
            "failure_stage": "render",
            "render_operator_result": (
                sorted(
                    result
                )
                if isinstance(
                    result,
                    set,
                )
                else str(
                    result
                )
            ),
            "render_complete_handler_fired": bool(
                render_state[
                    "complete"
                ]
            ),
            "render_cancel_handler_fired": bool(
                render_state[
                    "cancelled"
                ]
            ),
            "render_latency": latency,
            "error": (
                "Blender render operator did not return FINISHED."
            ),
        }

    render_stage_verified = (
        render_state[
            "complete"
        ]
        and not render_state[
            "cancelled"
        ]
    )

    if not render_stage_verified:
        return {
            "success": False,
            "failure_stage": "render_verification",
            "render_operator_result": sorted(
                result
            ),
            "render_complete_handler_fired": bool(
                render_state[
                    "complete"
                ]
            ),
            "render_cancel_handler_fired": bool(
                render_state[
                    "cancelled"
                ]
            ),
            "render_latency": latency,
            "error": (
                "Blender returned FINISHED, but the render lifecycle "
                "could not be verified as completed."
            ),
        }

    render_result = bpy.data.images.get(
        "Render Result"
    )

    render_result_present = (
        render_result is not None
    )

    render_result_width = (
        int(
            render_result.size[0]
        )
        if render_result_present
        else 0
    )

    render_result_height = (
        int(
            render_result.size[1]
        )
        if render_result_present
        else 0
    )

    file_verified = None
    save_latency = None

    # Stage B: save the already-rendered image without rendering again.
    if save_to_file:
        if render_result is None:
            return {
                "success": False,
                "verified": True,
                "failure_stage": "render_result_lookup",
                "render_complete_handler_fired": True,
                "render_cancel_handler_fired": False,
                "render_latency": latency,
                "error": (
                    "The render completed, but Blender's 'Render Result' "
                    "image was not available for saving."
                ),
            }

        save_started = time.perf_counter()

        try:
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            render_result.save_render(
                filepath=str(
                    output_path
                ),
                scene=scene,
            )

        except Exception as exc:
            return {
                "success": False,
                "verified": True,
                "failure_stage": "save",
                "render_complete_handler_fired": True,
                "render_cancel_handler_fired": False,
                "render_result_present": bool(
                    render_result_present
                ),
                "render_result_width": (
                    render_result_width
                ),
                "render_result_height": (
                    render_result_height
                ),
                "error": (
                    "Render completed successfully, but saving "
                    f"failed: {type(exc).__name__}: {exc}"
                ),
                "render_latency": latency,
                "output_path": (
                    str(output_path)
                    if output_path
                    else None
                ),
            }

        save_latency = (
            time.perf_counter()
            - save_started
        )

        file_verified = bool(
            output_path
            and output_path.exists()
            and output_path.is_file()
            and output_path.stat().st_size > 0
        )

        if not file_verified:
            return {
                "success": False,
                "verified": True,
                "failure_stage": "save_verification",
                "render_complete_handler_fired": True,
                "render_cancel_handler_fired": False,
                "render_result_present": bool(
                    render_result_present
                ),
                "render_result_width": (
                    render_result_width
                ),
                "render_result_height": (
                    render_result_height
                ),
                "error": (
                    "Render completed and save_render returned, but the "
                    "requested output file could not be verified."
                ),
                "render_latency": latency,
                "save_latency": save_latency,
                "output_path": (
                    str(output_path)
                    if output_path
                    else None
                ),
            }

    return {
        "success": True,
        "verified": True,
        "engine": str(
            scene.render.engine
        ),
        "active_camera": (
            scene.camera.name
            if scene.camera is not None
            else None
        ),
        "render_result": {
            "name": (
                render_result.name
                if render_result is not None
                else None
            ),
            "width": (
                render_result_width
            ),
            "height": (
                render_result_height
            ),
            "present": bool(
                render_result_present
            ),
            "verified": True,
            "verification_method": (
                "render_complete_handler"
            ),
        },
        "saved_to_file": save_to_file,
        "output_path": (
            str(output_path)
            if output_path is not None
            else None
        ),
        "file_verified": file_verified,
        "render_latency": latency,
        "save_latency": save_latency,
        "render_stage_verified": True,
        "render_complete_handler_fired": True,
        "render_cancel_handler_fired": False,
        "render_operator_result": sorted(
            result
        ),
        "save_stage_verified": (
            file_verified
            if save_to_file
            else None
        ),
    }


def _get_mesh_object_for_modeling(object_name):
    obj = bpy.data.objects.get(object_name)

    if obj is None:
        return None, {
            "success": False,
            "error": f"Object '{object_name}' was not found.",
        }

    if obj.type != "MESH":
        return None, {
            "success": False,
            "error": f"Object '{object_name}' is type {obj.type}, not MESH.",
        }

    return obj, None


def _mesh_info_dict(obj):
    mesh = obj.data

    if len(mesh.vertices):
        xs = [float(vertex.co.x) for vertex in mesh.vertices]
        ys = [float(vertex.co.y) for vertex in mesh.vertices]
        zs = [float(vertex.co.z) for vertex in mesh.vertices]
        local_bounds = {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
        }
    else:
        local_bounds = {
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
        }

    smooth_faces = sum(
        1
        for polygon in mesh.polygons
        if polygon.use_smooth
    )

    return {
        "object_name": obj.name,
        "location": [
            float(obj.location.x),
            float(obj.location.y),
            float(obj.location.z),
        ],
        "dimensions": [
            float(obj.dimensions.x),
            float(obj.dimensions.y),
            float(obj.dimensions.z),
        ],
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "face_count": len(mesh.polygons),
        "smooth_face_count": smooth_faces,
        "flat_face_count": len(mesh.polygons) - smooth_faces,
        "local_bounds": local_bounds,
        "materials": [
            material.name
            for material in mesh.materials
            if material is not None
        ],
        "modifiers": [
            {
                "name": modifier.name,
                "type": modifier.type,
            }
            for modifier in obj.modifiers
        ],
    }


def _tool_get_mesh_info(arguments):
    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )
    if error:
        return error

    bpy.context.view_layer.update()

    return {
        "success": True,
        "mesh": _mesh_info_dict(obj),
    }


def _ensure_object_name_available(name):
    if bpy.data.objects.get(name) is not None:
        return {
            "success": False,
            "error": f"Object '{name}' already exists.",
        }
    return None


def _finalize_created_mesh_object(name):
    obj = bpy.context.active_object

    if obj is None or obj.type != "MESH":
        return None, {
            "success": False,
            "error": "Blender primitive operator did not produce an active mesh object.",
        }

    obj.name = name
    bpy.context.view_layer.update()
    return obj, None


def _tool_create_uv_sphere(arguments):
    name = arguments["name"]
    error = _ensure_object_name_available(name)
    if error:
        return error

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=int(arguments["segments"]),
        ring_count=int(arguments["ring_count"]),
        radius=float(arguments["radius"]),
        location=(
            float(arguments["x"]),
            float(arguments["y"]),
            float(arguments["z"]),
        ),
    )

    obj, error = _finalize_created_mesh_object(name)
    if error:
        return error

    return {
        "success": True,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_create_cylinder(arguments):
    name = arguments["name"]
    error = _ensure_object_name_available(name)
    if error:
        return error

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=int(arguments["vertices"]),
        radius=float(arguments["radius"]),
        depth=float(arguments["depth"]),
        location=(
            float(arguments["x"]),
            float(arguments["y"]),
            float(arguments["z"]),
        ),
    )

    obj, error = _finalize_created_mesh_object(name)
    if error:
        return error

    return {
        "success": True,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_create_cone(arguments):
    name = arguments["name"]
    error = _ensure_object_name_available(name)
    if error:
        return error

    bpy.ops.mesh.primitive_cone_add(
        vertices=int(arguments["vertices"]),
        radius1=float(arguments["radius1"]),
        radius2=float(arguments["radius2"]),
        depth=float(arguments["depth"]),
        location=(
            float(arguments["x"]),
            float(arguments["y"]),
            float(arguments["z"]),
        ),
    )

    obj, error = _finalize_created_mesh_object(name)
    if error:
        return error

    return {
        "success": True,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_create_plane(arguments):
    name = arguments["name"]
    error = _ensure_object_name_available(name)
    if error:
        return error

    bpy.ops.mesh.primitive_plane_add(
        size=float(arguments["size"]),
        location=(
            float(arguments["x"]),
            float(arguments["y"]),
            float(arguments["z"]),
        ),
    )

    obj, error = _finalize_created_mesh_object(name)
    if error:
        return error

    return {
        "success": True,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_create_torus(arguments):
    name = arguments["name"]
    error = _ensure_object_name_available(name)
    if error:
        return error

    bpy.ops.mesh.primitive_torus_add(
        major_segments=int(arguments["major_segments"]),
        minor_segments=int(arguments["minor_segments"]),
        major_radius=float(arguments["major_radius"]),
        minor_radius=float(arguments["minor_radius"]),
        location=(
            float(arguments["x"]),
            float(arguments["y"]),
            float(arguments["z"]),
        ),
    )

    obj, error = _finalize_created_mesh_object(name)
    if error:
        return error

    return {
        "success": True,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_shade_smooth(arguments):
    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )
    if error:
        return error

    enabled = bool(arguments["enabled"])

    for polygon in obj.data.polygons:
        polygon.use_smooth = enabled

    obj.data.update()

    return {
        "success": True,
        "object_name": obj.name,
        "smooth_enabled": enabled,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_recalculate_normals(arguments):
    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )
    if error:
        return error

    mesh = obj.data
    bm = bmesh.new()

    try:
        bm.from_mesh(mesh)

        if len(bm.faces) == 0:
            return {
                "success": False,
                "error": f"Mesh '{obj.name}' has no faces to recalculate.",
            }

        bmesh.ops.recalc_face_normals(
            bm,
            faces=list(bm.faces),
        )

        bm.to_mesh(mesh)
        mesh.update()

    finally:
        bm.free()

    return {
        "success": True,
        "object_name": obj.name,
        "normals_recalculated": True,
        "mesh": _mesh_info_dict(obj),
    }


def _tool_scale_mesh_geometry(
    arguments,
    approved_high_risk=False,
):
    if not approved_high_risk:
        return {
            "success": False,
            "error": (
                "Direct mesh geometry scaling was blocked because "
                "the backend did not include high-risk approval."
            ),
        }

    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )
    if error:
        return error

    before = _mesh_info_dict(obj)

    x_factor = float(arguments["x_factor"])
    y_factor = float(arguments["y_factor"])
    z_factor = float(arguments["z_factor"])

    for vertex in obj.data.vertices:
        vertex.co.x *= x_factor
        vertex.co.y *= y_factor
        vertex.co.z *= z_factor

    obj.data.update()
    bpy.context.view_layer.update()

    after = _mesh_info_dict(obj)

    return {
        "success": True,
        "object_name": obj.name,
        "scale_factors": [x_factor, y_factor, z_factor],
        "before": before,
        "after": after,
    }


def _highest_upward_faces(bm):
    bm.normal_update()

    upward_faces = [
        face
        for face in bm.faces
        if face.normal.z > 0.5
    ]

    if not upward_faces:
        return []

    max_center_z = max(
        float(face.calc_center_median().z)
        for face in upward_faces
    )

    all_z = [
        float(vertex.co.z)
        for vertex in bm.verts
    ]

    z_span = (
        max(all_z) - min(all_z)
        if all_z
        else 1.0
    )

    tolerance = max(
        1e-6,
        abs(z_span) * 1e-5,
    )

    return [
        face
        for face in upward_faces
        if abs(
            float(face.calc_center_median().z)
            - max_center_z
        )
        <= tolerance
    ]


def _tool_extrude_top_face(
    arguments,
    approved_high_risk=False,
):
    if not approved_high_risk:
        return {
            "success": False,
            "error": (
                "Direct face extrusion was blocked because the "
                "backend did not include high-risk approval."
            ),
        }

    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )
    if error:
        return error

    distance = float(arguments["distance"])
    mesh = obj.data
    before = _mesh_info_dict(obj)

    bm = bmesh.new()
    top_face_count = 0

    try:
        bm.from_mesh(mesh)

        top_faces = _highest_upward_faces(bm)
        top_face_count = len(top_faces)

        if not top_faces:
            return {
                "success": False,
                "error": (
                    f"Could not identify an upward-facing top face "
                    f"on mesh '{obj.name}'."
                ),
            }

        result = bmesh.ops.extrude_face_region(
            bm,
            geom=list(top_faces),
            use_keep_orig=False,
            use_select_history=False,
        )

        extruded_vertices = [
            element
            for element in result["geom"]
            if isinstance(
                element,
                bmesh.types.BMVert,
            )
        ]

        if not extruded_vertices:
            return {
                "success": False,
                "error": "Blender returned no extruded vertices.",
            }

        bmesh.ops.translate(
            bm,
            verts=extruded_vertices,
            vec=Vector((0.0, 0.0, distance)),
        )

        bmesh.ops.recalc_face_normals(
            bm,
            faces=list(bm.faces),
        )

        bm.to_mesh(mesh)
        mesh.update()

    finally:
        bm.free()

    bpy.context.view_layer.update()
    after = _mesh_info_dict(obj)

    return {
        "success": True,
        "object_name": obj.name,
        "distance": distance,
        "top_face_region_count": top_face_count,
        "before": before,
        "after": after,
        "geometry_changed": (
            before["vertex_count"] != after["vertex_count"]
            or before["edge_count"] != after["edge_count"]
            or before["face_count"] != after["face_count"]
        ),
    }


def _restore_object_mode_and_selection(
    obj,
    previous_active,
    previous_selected,
    previous_mode,
):
    try:
        if (
            bpy.context.object is not None
            and bpy.context.object.mode != "OBJECT"
        ):
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    try:
        for selected in list(bpy.context.selected_objects):
            selected.select_set(False)

        for selected in previous_selected:
            if selected.name in bpy.data.objects:
                selected.select_set(True)

        if (
            previous_active is not None
            and previous_active.name in bpy.data.objects
        ):
            bpy.context.view_layer.objects.active = previous_active

        if (
            previous_mode == "EDIT"
            and previous_active is obj
            and obj.name in bpy.data.objects
        ):
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")

    except Exception:
        pass


def _tool_bevel_mesh_edges(
    arguments,
    approved_high_risk=False,
):
    if not approved_high_risk:
        return {
            "success": False,
            "error": (
                "Direct mesh bevel was blocked because the backend "
                "did not include high-risk approval."
            ),
        }

    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )
    if error:
        return error

    width = float(arguments["width"])
    segments = int(arguments["segments"])
    before = _mesh_info_dict(obj)

    view_layer = bpy.context.view_layer
    previous_active = view_layer.objects.active
    previous_selected = list(bpy.context.selected_objects)
    previous_mode = (
        previous_active.mode
        if previous_active is not None
        else "OBJECT"
    )

    try:
        if (
            bpy.context.object is not None
            and bpy.context.object.mode != "OBJECT"
        ):
            bpy.ops.object.mode_set(mode="OBJECT")

        for selected in list(bpy.context.selected_objects):
            selected.select_set(False)

        obj.select_set(True)
        view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")

        try:
            result = bpy.ops.mesh.bevel(
                offset=width,
                segments=segments,
                affect="EDGES",
            )
        except (TypeError, ValueError):
            result = bpy.ops.mesh.bevel(
                offset=width,
                segments=segments,
            )

        if "FINISHED" not in result:
            return {
                "success": False,
                "error": (
                    f"Blender mesh bevel returned {sorted(result)} "
                    "instead of FINISHED."
                ),
            }

        bpy.ops.object.mode_set(mode="OBJECT")
        obj.data.update()
        bpy.context.view_layer.update()

    except Exception as exc:
        return {
            "success": False,
            "error": (
                f"Direct mesh bevel failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }

    finally:
        _restore_object_mode_and_selection(
            obj,
            previous_active,
            previous_selected,
            previous_mode,
        )

    after = _mesh_info_dict(obj)

    return {
        "success": True,
        "object_name": obj.name,
        "width": width,
        "segments": segments,
        "before": before,
        "after": after,
        "geometry_changed": (
            before["vertex_count"] != after["vertex_count"]
            or before["edge_count"] != after["edge_count"]
            or before["face_count"] != after["face_count"]
        ),
    }


def _mesh_counts_and_bounds_match(left, right, tolerance=1e-5):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    for key in ("vertex_count", "edge_count", "face_count"):
        if int(left.get(key, -1)) != int(right.get(key, -2)):
            return False
    for bound_name in ("min", "max"):
        lv=left.get("local_bounds",{}).get(bound_name,[]); rv=right.get("local_bounds",{}).get(bound_name,[])
        if len(lv)!=3 or len(rv)!=3: return False
        if any(abs(float(a)-float(b))>tolerance for a,b in zip(lv,rv)): return False
    return True


def _unique_vertices_from_faces(faces):
    seen=set(); vertices=[]
    for face in faces:
        for vertex in face.verts:
            key=id(vertex)
            if key not in seen:
                seen.add(key); vertices.append(vertex)
    return vertices


def _mesh_top_region_metrics(obj):
    mesh = obj.data
    bpy.context.view_layer.update()

    rows = []

    for poly in mesh.polygons:
        coords = [
            mesh.vertices[index].co
            for index in poly.vertices
        ]

        center_z = (
            sum(float(coord.z) for coord in coords) / len(coords)
            if coords
            else 0.0
        )

        rows.append(
            {
                "index": int(poly.index),
                "normal_z": float(poly.normal.z),
                "center_z": float(center_z),
                "vertex_indices": [
                    int(index)
                    for index in poly.vertices
                ],
            }
        )

    upward = [
        row
        for row in rows
        if row["normal_z"] > 0.5
    ]

    downward = [
        row
        for row in rows
        if row["normal_z"] < -0.5
    ]

    side = [
        row
        for row in rows
        if abs(row["normal_z"]) <= 0.5
    ]

    top = []

    if upward:
        highest = max(
            row["center_z"]
            for row in upward
        )

        z_values = [
            float(vertex.co.z)
            for vertex in mesh.vertices
        ]

        span = (
            max(z_values) - min(z_values)
            if z_values
            else 1.0
        )

        tolerance = max(
            1e-6,
            abs(span) * 1e-5,
        )

        top = [
            row
            for row in upward
            if abs(row["center_z"] - highest) <= tolerance
        ]

    vertex_indices = sorted(
        {
            vertex_index
            for row in top
            for vertex_index in row["vertex_indices"]
        }
    )

    coords = [
        mesh.vertices[index].co
        for index in vertex_indices
    ]

    if coords:
        min_x = min(float(coord.x) for coord in coords)
        max_x = max(float(coord.x) for coord in coords)
        min_y = min(float(coord.y) for coord in coords)
        max_y = max(float(coord.y) for coord in coords)
        min_z = min(float(coord.z) for coord in coords)
        max_z = max(float(coord.z) for coord in coords)

        centroid = [
            sum(float(coord.x) for coord in coords) / len(coords),
            sum(float(coord.y) for coord in coords) / len(coords),
            sum(float(coord.z) for coord in coords) / len(coords),
        ]

        bounds = {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
        }

        width = max_x - min_x
        depth = max_y - min_y
        height = max_z - min_z

    else:
        centroid = None
        bounds = {
            "min": None,
            "max": None,
        }
        width = None
        depth = None
        height = None

    return {
        "face_count": len(top),
        "face_indices": [
            int(row["index"])
            for row in top[:64]
        ],
        "vertex_count": len(vertex_indices),
        "vertex_indices": vertex_indices[:256],
        "centroid": centroid,
        "bounds": bounds,
        "width": width,
        "depth": depth,
        "height": height,
        "center_z": (
            float(centroid[2])
            if centroid is not None
            else None
        ),
        "upward_face_count": len(upward),
        "downward_face_count": len(downward),
        "side_face_count": len(side),
    }


def _tool_get_mesh_regions(arguments):
    obj, error = _get_mesh_object_for_modeling(
        arguments["object_name"]
    )

    if error:
        return error

    region = _mesh_top_region_metrics(obj)
    mesh = _mesh_info_dict(obj)

    return {
        "success": True,
        "object_name": obj.name,
        "mesh": mesh,
        "face_count": int(mesh["face_count"]),
        "upward_face_count": int(region["upward_face_count"]),
        "downward_face_count": int(region["downward_face_count"]),
        "side_face_count": int(region["side_face_count"]),
        "top_face_count": int(region["face_count"]),
        "top_face_indices": list(region["face_indices"]),
        "top_center_z": region["center_z"],
        "top_region": region,
    }


def _commit_bmesh_modeling(obj,bm):
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces)); bm.to_mesh(obj.data); obj.data.update(); bpy.context.view_layer.update()


def _direct_mesh_success_payload(obj,before,**extra):
    after=_mesh_info_dict(obj)
    return {"success":True,"object_name":obj.name,"before":before,"after":after,"geometry_changed":not _mesh_counts_and_bounds_match(before,after),**extra}


def _tool_inset_top_face(arguments,approved_high_risk=False):
    if not approved_high_risk: return {"success":False,"error":"Top-face inset was blocked because the backend did not include high-risk approval."}
    obj,error=_get_mesh_object_for_modeling(arguments["object_name"])
    if error: return error
    thickness=float(arguments["thickness"]); depth=float(arguments["depth"]); before=_mesh_info_dict(obj); before_region=_mesh_top_region_metrics(obj); bm=bmesh.new(); top_faces=[]
    try:
        bm.from_mesh(obj.data); top_faces=_highest_upward_faces(bm)
        if not top_faces: return {"success":False,"error":f"Could not identify a highest upward-facing face region on '{obj.name}'."}
        bmesh.ops.inset_region(bm,faces=list(top_faces),faces_exclude=[],use_boundary=True,use_even_offset=True,use_interpolate=True,use_relative_offset=False,use_edge_rail=False,thickness=thickness,depth=depth,use_outset=False)
        _commit_bmesh_modeling(obj,bm)
    except Exception as exc: return {"success":False,"error":f"Top-face inset failed: {type(exc).__name__}: {exc}"}
    finally: bm.free()
    after_region=_mesh_top_region_metrics(obj)
    return _direct_mesh_success_payload(obj,before,thickness=thickness,depth=depth,top_face_region_count=len(top_faces),before_region=before_region,after_region=after_region)


def _tool_subdivide_mesh(arguments,approved_high_risk=False):
    if not approved_high_risk: return {"success":False,"error":"Mesh subdivision was blocked because the backend did not include high-risk approval."}
    obj,error=_get_mesh_object_for_modeling(arguments["object_name"])
    if error: return error
    cuts=int(arguments["cuts"]); before=_mesh_info_dict(obj); bm=bmesh.new()
    try:
        bm.from_mesh(obj.data); edges=list(bm.edges)
        if not edges: return {"success":False,"error":f"Mesh '{obj.name}' has no edges to subdivide."}
        bmesh.ops.subdivide_edges(bm,edges=edges,cuts=cuts,use_grid_fill=True); _commit_bmesh_modeling(obj,bm)
    except Exception as exc: return {"success":False,"error":f"Mesh subdivision failed: {type(exc).__name__}: {exc}"}
    finally: bm.free()
    return _direct_mesh_success_payload(obj,before,cuts=cuts)


def _tool_translate_top_face(arguments,approved_high_risk=False):
    if not approved_high_risk: return {"success":False,"error":"Top-face translation was blocked because the backend did not include high-risk approval."}
    obj,error=_get_mesh_object_for_modeling(arguments["object_name"])
    if error: return error
    offset=Vector((float(arguments["x_offset"]),float(arguments["y_offset"]),float(arguments["z_offset"]))); before=_mesh_info_dict(obj); before_region=_mesh_top_region_metrics(obj); bm=bmesh.new(); top_faces=[]
    try:
        bm.from_mesh(obj.data); top_faces=_highest_upward_faces(bm)
        if not top_faces: return {"success":False,"error":f"Could not identify a highest upward-facing face region on '{obj.name}'."}
        verts=_unique_vertices_from_faces(top_faces); bmesh.ops.translate(bm,verts=verts,vec=offset); _commit_bmesh_modeling(obj,bm)
    except Exception as exc: return {"success":False,"error":f"Top-face translation failed: {type(exc).__name__}: {exc}"}
    finally: bm.free()
    after_region=_mesh_top_region_metrics(obj)
    return _direct_mesh_success_payload(obj,before,offset=[float(offset.x),float(offset.y),float(offset.z)],top_face_region_count=len(top_faces),before_region=before_region,after_region=after_region)


def _tool_scale_top_face(arguments,approved_high_risk=False):
    if not approved_high_risk: return {"success":False,"error":"Top-face scaling was blocked because the backend did not include high-risk approval."}
    obj,error=_get_mesh_object_for_modeling(arguments["object_name"])
    if error: return error
    xf=float(arguments["x_factor"]); yf=float(arguments["y_factor"]); before=_mesh_info_dict(obj); before_region=_mesh_top_region_metrics(obj); bm=bmesh.new(); top_faces=[]
    try:
        bm.from_mesh(obj.data); top_faces=_highest_upward_faces(bm)
        if not top_faces: return {"success":False,"error":f"Could not identify a highest upward-facing face region on '{obj.name}'."}
        verts=_unique_vertices_from_faces(top_faces)
        if not verts: return {"success":False,"error":"Top-face region contained no vertices."}
        center=Vector((0.0,0.0,0.0))
        for v in verts: center += v.co
        center /= len(verts)
        for v in verts:
            v.co.x=center.x+(v.co.x-center.x)*xf; v.co.y=center.y+(v.co.y-center.y)*yf
        _commit_bmesh_modeling(obj,bm)
    except Exception as exc: return {"success":False,"error":f"Top-face scaling failed: {type(exc).__name__}: {exc}"}
    finally: bm.free()
    after_region=_mesh_top_region_metrics(obj)
    return _direct_mesh_success_payload(obj,before,x_factor=xf,y_factor=yf,top_face_region_count=len(top_faces),before_region=before_region,after_region=after_region)


def _tool_merge_by_distance(arguments,approved_high_risk=False):
    if not approved_high_risk: return {"success":False,"error":"Merge by distance was blocked because the backend did not include high-risk approval."}
    obj,error=_get_mesh_object_for_modeling(arguments["object_name"])
    if error: return error
    dist=float(arguments["distance"]); before=_mesh_info_dict(obj); before_count=int(before["vertex_count"]); bm=bmesh.new()
    try:
        bm.from_mesh(obj.data); bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=dist); _commit_bmesh_modeling(obj,bm)
    except Exception as exc: return {"success":False,"error":f"Merge by distance failed: {type(exc).__name__}: {exc}"}
    finally: bm.free()
    payload=_direct_mesh_success_payload(obj,before,distance=dist); payload["merged_vertices"]=max(0,before_count-int(payload["after"]["vertex_count"])); return payload


def _tool_solidify_mesh(arguments,approved_high_risk=False):
    if not approved_high_risk: return {"success":False,"error":"Mesh solidification was blocked because the backend did not include high-risk approval."}
    obj,error=_get_mesh_object_for_modeling(arguments["object_name"])
    if error: return error
    thickness=float(arguments["thickness"]); before=_mesh_info_dict(obj); bm=bmesh.new()
    try:
        bm.from_mesh(obj.data); faces=list(bm.faces)
        if not faces: return {"success":False,"error":f"Mesh '{obj.name}' has no faces to solidify."}
        bmesh.ops.solidify(bm,geom=faces,thickness=thickness); _commit_bmesh_modeling(obj,bm)
    except Exception as exc: return {"success":False,"error":f"Mesh solidification failed: {type(exc).__name__}: {exc}"}
    finally: bm.free()
    return _direct_mesh_success_payload(obj,before,thickness=thickness)


def _tool_delete_object(arguments, approved_high_risk=False):
    if not approved_high_risk:
        return {
            "success": False,
            "error": (
                "Delete command was blocked because the backend "
                "did not include high-risk approval."
            ),
        }

    name = arguments["name"]
    obj = bpy.data.objects.get(name)

    if obj is None:
        return {
            "success": False,
            "error": f"Object '{name}' was not found.",
        }

    bpy.data.objects.remove(
        obj,
        do_unlink=True,
    )

    return {
        "success": True,
        "deleted": name,
    }


def _execute_bridge_command(command):
    tool_name = command.get("tool")
    arguments = command.get("arguments", {})

    if tool_name == "get_scene_objects":
        return _tool_get_scene_objects(arguments)

    if tool_name == "get_materials":
        return _tool_get_materials(arguments)

    if tool_name == "create_material":
        return _tool_create_material(arguments)

    if tool_name == "set_material_color":
        return _tool_set_material_color(arguments)

    if tool_name == "assign_material":
        return _tool_assign_material(arguments)

    if tool_name == "get_modifiers":
        return _tool_get_modifiers(arguments)

    if tool_name == "add_bevel_modifier":
        return _tool_add_bevel_modifier(arguments)

    if tool_name == "set_bevel_modifier":
        return _tool_set_bevel_modifier(arguments)

    if tool_name == "add_subdivision_modifier":
        return _tool_add_subdivision_modifier(arguments)

    if tool_name == "set_subdivision_modifier":
        return _tool_set_subdivision_modifier(arguments)

    if tool_name == "remove_modifier":
        return _tool_remove_modifier(arguments)

    if tool_name == "apply_modifier":
        return _tool_apply_modifier(
            arguments,
            approved_high_risk=bool(command.get("approved_high_risk", False)),
        )

    if tool_name == "get_cameras":
        return _tool_get_cameras(arguments)
    if tool_name == "create_camera":
        return _tool_create_camera(arguments)
    if tool_name == "move_camera":
        return _tool_move_camera(arguments)
    if tool_name == "set_camera_lens":
        return _tool_set_camera_lens(arguments)
    if tool_name == "set_active_camera":
        return _tool_set_active_camera(arguments)
    if tool_name == "aim_camera_at_object":
        return _tool_aim_camera_at_object(arguments)

    if tool_name == "get_lights":
        return _tool_get_lights(arguments)
    if tool_name == "create_light":
        return _tool_create_light(arguments)
    if tool_name == "move_light":
        return _tool_move_light(arguments)
    if tool_name == "set_light_energy":
        return _tool_set_light_energy(arguments)
    if tool_name == "set_light_color":
        return _tool_set_light_color(arguments)
    if tool_name == "set_area_light_size":
        return _tool_set_area_light_size(arguments)
    if tool_name == "aim_light_at_object":
        return _tool_aim_light_at_object(arguments)

    if tool_name == "get_render_settings":
        return _tool_get_render_settings(arguments)

    if tool_name == "set_render_engine":
        return _tool_set_render_engine(arguments)

    if tool_name == "set_render_resolution":
        return _tool_set_render_resolution(arguments)

    if tool_name == "set_render_samples":
        return _tool_set_render_samples(arguments)

    if tool_name == "set_render_output":
        return _tool_set_render_output(arguments)

    if tool_name == "set_render_transparent":
        return _tool_set_render_transparent(arguments)

    if tool_name == "render_scene":
        return _tool_render_scene(arguments)

    if tool_name == "get_mesh_info":
        return _tool_get_mesh_info(arguments)

    if tool_name == "create_uv_sphere":
        return _tool_create_uv_sphere(arguments)

    if tool_name == "create_cylinder":
        return _tool_create_cylinder(arguments)

    if tool_name == "create_cone":
        return _tool_create_cone(arguments)

    if tool_name == "create_plane":
        return _tool_create_plane(arguments)

    if tool_name == "create_torus":
        return _tool_create_torus(arguments)

    if tool_name == "shade_smooth":
        return _tool_shade_smooth(arguments)

    if tool_name == "recalculate_normals":
        return _tool_recalculate_normals(arguments)

    if tool_name == "scale_mesh_geometry":
        return _tool_scale_mesh_geometry(
            arguments,
            approved_high_risk=bool(
                command.get("approved_high_risk", False)
            ),
        )

    if tool_name == "extrude_top_face":
        return _tool_extrude_top_face(
            arguments,
            approved_high_risk=bool(
                command.get("approved_high_risk", False)
            ),
        )

    if tool_name == "bevel_mesh_edges":
        return _tool_bevel_mesh_edges(
            arguments,
            approved_high_risk=bool(
                command.get("approved_high_risk", False)
            ),
        )

    if tool_name == "get_mesh_regions":
        return _tool_get_mesh_regions(arguments)

    if tool_name == "inset_top_face":
        return _tool_inset_top_face(arguments, approved_high_risk=bool(command.get("approved_high_risk", False)))

    if tool_name == "subdivide_mesh":
        return _tool_subdivide_mesh(arguments, approved_high_risk=bool(command.get("approved_high_risk", False)))

    if tool_name == "translate_top_face":
        return _tool_translate_top_face(arguments, approved_high_risk=bool(command.get("approved_high_risk", False)))

    if tool_name == "scale_top_face":
        return _tool_scale_top_face(arguments, approved_high_risk=bool(command.get("approved_high_risk", False)))

    if tool_name == "merge_by_distance":
        return _tool_merge_by_distance(arguments, approved_high_risk=bool(command.get("approved_high_risk", False)))

    if tool_name == "solidify_mesh":
        return _tool_solidify_mesh(arguments, approved_high_risk=bool(command.get("approved_high_risk", False)))

    if tool_name == "create_cube":
        return _tool_create_cube(arguments)

    if tool_name == "move_object":
        return _tool_move_object(arguments)

    if tool_name == "delete_object":
        return _tool_delete_object(
            arguments,
            approved_high_risk=bool(
                command.get(
                    "approved_high_risk",
                    False,
                )
            ),
        )

    return {
        "success": False,
        "error": f"Unknown Blender bridge tool: {tool_name}",
    }


def bridge_listener():
    """
    Runs on Blender's main thread via bpy.app.timers.

    This is why the UI HTTP request itself runs on a background thread:
    Blender's main thread must stay free to execute controlled bpy actions.
    """
    global _LAST_COMMAND_ID

    scene = bpy.context.scene
    project_root = _resolve_project_root(scene)

    if project_root is None:
        return 0.25

    bridge_dir = project_root / "bridge"
    command_file = bridge_dir / "command.json"
    result_file = bridge_dir / "result.json"

    if not command_file.exists():
        return 0.25

    try:
        command = json.loads(
            command_file.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return 0.10

    command_id = command.get("id")

    if not command_id:
        return 0.25

    cached_result = (
        _cached_command_result(
            command_id
        )
    )

    if cached_result is not None:
        payload = {
            "id": command_id,
            "timestamp": time.time(),
            "result": cached_result,
            "reused_cached_result": True,
        }

        bridge_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            _atomic_write_json(
                result_file,
                payload,
            )
            _LAST_COMMAND_ID = (
                command_id
            )
        except OSError as exc:
            print(
                "[Blender AI Copilot] "
                "Could not rewrite cached bridge result: "
                f"{exc}"
            )

        return 0.25

    if command_id == _LAST_COMMAND_ID:
        return 0.25

    # Do not replay abandoned commands from an old backend session.
    timestamp = float(
        command.get(
            "timestamp",
            0.0,
        )
        or 0.0
    )

    if timestamp and time.time() - timestamp > 120:
        _LAST_COMMAND_ID = command_id
        return 0.25

    try:
        result = _execute_bridge_command(command)

    except Exception as exc:
        result = {
            "success": False,
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    _cache_command_result(
        command_id,
        result,
    )

    payload = {
        "id": command_id,
        "timestamp": time.time(),
        "result": result,
    }

    bridge_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        _atomic_write_json(
            result_file,
            payload,
        )
        _LAST_COMMAND_ID = command_id

    except OSError as exc:
        print(
            "[Blender AI Copilot] "
            f"Could not write bridge result: {exc}"
        )

    return 0.25


def _health_worker(base_url):
    global _BACKEND_PROJECT_ROOT

    try:
        result = client.health(
            base_url,
            timeout=5,
        )

        project_root = result.get(
            "project_root"
        )

        if project_root:
            _set_backend_project_root(
                project_root
            )

        _RESULT_QUEUE.put(
            ("health", result)
        )

    except Exception as exc:
        _RESULT_QUEUE.put(
            (
                "error",
                {
                    "error": str(exc),
                    "operation": "health",
                },
            )
        )


def _build_conversation_context(scene):
    """
    Full conversation remains in scene.copilot_chat_history for the user.
    Only the last N user/assistant request-result messages plus the compact
    backend memory summary are sent to Qwen.
    """
    if not getattr(scene, "copilot_context_enabled", True):
        return {
            "recent_messages": [],
            "memory_summary": "",
            "structured_memory": {},
        }

    turns = max(
        1,
        min(
            int(getattr(scene, "copilot_context_turns", 3)),
            6,
        ),
    )

    filtered = []

    for entry in scene.copilot_chat_history:
        if entry.role == "USER" and entry.status == "Request":
            filtered.append(
                {
                    "role": "user",
                    "content": str(entry.text or "").strip(),
                }
            )
        elif entry.role == "ASSISTANT" and entry.status == "Complete":
            filtered.append(
                {
                    "role": "assistant",
                    "content": str(entry.text or "").strip(),
                }
            )

    recent = [
        item
        for item in filtered[-(turns * 2):]
        if item.get("content")
    ]

    try:
        structured_memory = json.loads(
            str(
                getattr(scene, "copilot_reference_memory", "{}")
                or "{}"
            )
        )
        if not isinstance(structured_memory, dict):
            structured_memory = {}
    except Exception:
        structured_memory = {}

    return {
        "recent_messages": recent,
        "memory_summary": str(
            getattr(scene, "copilot_memory_summary", "")
            or ""
        ).strip(),
        "structured_memory": structured_memory,
    }


def _chat_worker(
    base_url,
    prompt,
    conversation_context,
):
    global _BACKEND_PROJECT_ROOT

    try:
        # Learn the backend's project root before the chat request. This lets
        # the bridge timer service an action even if the Scene field is blank.
        if not _BACKEND_PROJECT_ROOT:
            health_result = client.health(
                base_url,
                timeout=5,
            )

            project_root = health_result.get(
                "project_root"
            )

            if project_root:
                _set_backend_project_root(
                    project_root
                )

        result = client.chat(
            base_url,
            prompt,
            conversation_context=conversation_context,
            timeout=900,
        )

        project_root = result.get(
            "project_root"
        )

        if project_root:
            _set_backend_project_root(
                project_root
            )

        _RESULT_QUEUE.put(
            ("chat", result)
        )

    except Exception as exc:
        _RESULT_QUEUE.put(
            (
                "error",
                {
                    "error": str(exc),
                    "operation": "chat",
                },
            )
        )


def _approval_worker(
    base_url,
    approval_id,
    approved,
):
    try:
        result = client.approve(
            base_url,
            approval_id,
            approved,
            timeout=180,
        )

        _RESULT_QUEUE.put(
            ("approval", result)
        )

    except Exception as exc:
        _RESULT_QUEUE.put(
            (
                "error",
                {
                    "error": str(exc),
                    "operation": "approval",
                },
            )
        )


def _format_action(action):
    if not action:
        return ""

    return (
        f"{action.get('tool')} "
        f"{json.dumps(action.get('arguments', {}), ensure_ascii=False)}"
    )


def _append_chat_message(
    scene,
    role,
    text,
    status="",
    trace_id="",
):
    text = str(text or "").strip()
    if not text:
        return

    entry = scene.copilot_chat_history.add()
    entry.role = role
    entry.text = text
    entry.status = str(status or "")
    entry.trace_id = str(trace_id or "")
    entry.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Keep the .blend scene history bounded. 120 entries is roughly 60 turns.
    while len(scene.copilot_chat_history) > 120:
        scene.copilot_chat_history.remove(0)


def _apply_backend_result(scene, result):
    memory_summary = result.get("memory_summary")
    if isinstance(memory_summary, str):
        scene.copilot_memory_summary = memory_summary

    structured_memory = result.get("structured_memory")
    if isinstance(structured_memory, dict):
        scene.copilot_reference_memory = json.dumps(
            structured_memory,
            ensure_ascii=False,
            sort_keys=True,
        )

    project_root = result.get(
        "project_root"
    )

    if project_root:
        _set_backend_project_root(
            project_root
        )

        if not scene.copilot_project_root.strip():
            scene.copilot_project_root = project_root

    status = result.get(
        "status"
    )

    if status == "approval_required":
        scene.copilot_status = "Approval required"
        scene.copilot_pending_approval_id = result.get(
            "approval_id",
            "",
        )
        scene.copilot_pending_action = _format_action(
            result.get("action")
        )
        scene.copilot_response = result.get(
            "message",
            "Approval required.",
        )
        approval_text = scene.copilot_response
        if scene.copilot_pending_action:
            approval_text += "\n" + scene.copilot_pending_action
        _append_chat_message(
            scene,
            "ASSISTANT",
            approval_text,
            status="Approval required",
            trace_id=result.get("trace_id", ""),
        )
        scene.copilot_busy = False
        return

    if status == "complete":
        scene.copilot_status = "Ready"
        scene.copilot_response = result.get(
            "answer",
            "",
        )
        scene.copilot_pending_approval_id = ""
        scene.copilot_pending_action = ""
        _append_chat_message(
            scene,
            "ASSISTANT",
            scene.copilot_response,
            status="Complete",
            trace_id=result.get("trace_id", ""),
        )
        scene.copilot_busy = False
        return

    if status == "error":
        scene.copilot_status = "Error"
        scene.copilot_response = result.get(
            "error",
            "Unknown backend error.",
        )
        _append_chat_message(
            scene,
            "ASSISTANT",
            scene.copilot_response,
            status="Error",
            trace_id=result.get("trace_id", ""),
        )
        scene.copilot_busy = False
        return

    # Health response.
    if result.get("status") == "ok":
        scene.copilot_status = (
            f"Backend connected · {result.get('rag_chunks', '?')} RAG chunks"
        )
        scene.copilot_busy = False


def poll_client_results():
    scene = bpy.context.scene

    if scene is None:
        return 0.20

    while True:
        try:
            kind, result = _RESULT_QUEUE.get_nowait()
        except queue.Empty:
            break

        if kind == "error":
            scene.copilot_status = "Error"
            scene.copilot_response = result.get(
                "error",
                "Unknown client error.",
            )
            _append_chat_message(
                scene,
                "ASSISTANT",
                scene.copilot_response,
                status="Connection error",
            )
            scene.copilot_busy = False
        else:
            _apply_backend_result(
                scene,
                result,
            )

    _tag_redraw()
    return 0.20


class COPILOT_OT_check_backend(bpy.types.Operator):
    bl_idname = "copilot.check_backend"
    bl_label = "Check Backend"
    bl_description = "Check the local Blender AI Copilot backend"

    def execute(self, context):
        scene = context.scene

        if not _online_access_allowed():
            self.report(
                {"ERROR"},
                "Blender online access is disabled in Preferences > System.",
            )
            return {"CANCELLED"}

        if scene.copilot_busy:
            self.report(
                {"WARNING"},
                "Copilot is already busy.",
            )
            return {"CANCELLED"}

        scene.copilot_busy = True
        scene.copilot_status = "Checking backend..."

        threading.Thread(
            target=_health_worker,
            args=(scene.copilot_backend_url,),
            daemon=True,
        ).start()

        return {"FINISHED"}


class COPILOT_OT_send(bpy.types.Operator):
    bl_idname = "copilot.send"
    bl_label = "Send"
    bl_description = "Send the prompt to the local Blender AI Copilot"

    def execute(self, context):
        scene = context.scene
        prompt = scene.copilot_prompt.strip()

        if not _online_access_allowed():
            self.report(
                {"ERROR"},
                "Blender online access is disabled in Preferences > System.",
            )
            return {"CANCELLED"}

        if not prompt:
            self.report(
                {"WARNING"},
                "Enter a prompt first.",
            )
            return {"CANCELLED"}

        if scene.copilot_busy:
            self.report(
                {"WARNING"},
                "Copilot is already processing a request.",
            )
            return {"CANCELLED"}

        conversation_context = _build_conversation_context(
            scene
        )

        _append_chat_message(
            scene,
            "USER",
            prompt,
            status="Request",
        )
        scene.copilot_prompt = ""
        scene.copilot_busy = True
        scene.copilot_status = "Thinking..."
        scene.copilot_response = ""

        threading.Thread(
            target=_chat_worker,
            args=(
                scene.copilot_backend_url,
                prompt,
                conversation_context,
            ),
            daemon=True,
        ).start()

        return {"FINISHED"}


class COPILOT_OT_approve(bpy.types.Operator):
    bl_idname = "copilot.approve"
    bl_label = "Approve"
    bl_description = "Approve the pending high-risk copilot action"

    def execute(self, context):
        scene = context.scene
        approval_id = scene.copilot_pending_approval_id

        if not approval_id:
            self.report(
                {"WARNING"},
                "There is no pending approval.",
            )
            return {"CANCELLED"}

        _append_chat_message(
            scene,
            "USER",
            f"Approved: {scene.copilot_pending_action}",
            status="Approval",
        )
        scene.copilot_busy = True
        scene.copilot_status = "Executing approved action..."

        threading.Thread(
            target=_approval_worker,
            args=(
                scene.copilot_backend_url,
                approval_id,
                True,
            ),
            daemon=True,
        ).start()

        return {"FINISHED"}


class COPILOT_OT_reject(bpy.types.Operator):
    bl_idname = "copilot.reject"
    bl_label = "Reject"
    bl_description = "Reject the pending high-risk copilot action"

    def execute(self, context):
        scene = context.scene
        approval_id = scene.copilot_pending_approval_id

        if not approval_id:
            self.report(
                {"WARNING"},
                "There is no pending approval.",
            )
            return {"CANCELLED"}

        _append_chat_message(
            scene,
            "USER",
            f"Rejected: {scene.copilot_pending_action}",
            status="Approval",
        )
        scene.copilot_busy = True
        scene.copilot_status = "Rejecting action..."

        threading.Thread(
            target=_approval_worker,
            args=(
                scene.copilot_backend_url,
                approval_id,
                False,
            ),
            daemon=True,
        ).start()

        return {"FINISHED"}


class COPILOT_OT_clear_chat(bpy.types.Operator):
    bl_idname = "copilot.clear_chat"
    bl_label = "Clear Chat"
    bl_description = "Clear Copilot conversation history stored in this scene"

    def execute(self, context):
        scene = context.scene
        scene.copilot_chat_history.clear()
        scene.copilot_memory_summary = ""
        scene.copilot_reference_memory = "{}"
        scene.copilot_response = ""
        self.report({"INFO"}, "Copilot chat history cleared.")
        return {"FINISHED"}


CLASSES = (
    COPILOT_OT_check_backend,
    COPILOT_OT_send,
    COPILOT_OT_approve,
    COPILOT_OT_reject,
    COPILOT_OT_clear_chat,
)


def register_runtime():
    if not bpy.app.timers.is_registered(
        poll_client_results
    ):
        bpy.app.timers.register(
            poll_client_results,
            first_interval=0.20,
            persistent=True,
        )

    if not bpy.app.timers.is_registered(
        bridge_listener
    ):
        bpy.app.timers.register(
            bridge_listener,
            first_interval=0.25,
            persistent=True,
        )


def unregister_runtime():
    for callback in (
        poll_client_results,
        bridge_listener,
    ):
        if bpy.app.timers.is_registered(
            callback
        ):
            try:
                bpy.app.timers.unregister(
                    callback
                )
            except ValueError:
                pass
