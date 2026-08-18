"""Tool schemas, validation, risk policy, and Ollama tool conversion."""

import json


TOOL_SCHEMAS = [
    {
        "name": "search_blender_docs",
        "description": (
            "Search official Blender Manual and Blender Python API documentation. "
            "Use this for precise Blender behavior, API semantics, modes/context, "
            "modeling concepts, materials, cameras, rendering, modifiers, or when "
            "Blender-specific knowledge is uncertain. This tool does not modify Blender."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused Blender documentation question.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of diverse documentation passages to return, normally 3 to 5.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_materials",
        "description": (
            "Inspect Blender materials and which objects use them. "
            "Use this to answer material questions and verify material changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_material",
        "description": (
            "Create a Blender material with an RGBA base color. "
            "Color values must be between 0 and 1."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "r": {"type": "number"},
                "g": {"type": "number"},
                "b": {"type": "number"},
                "a": {"type": "number"},
            },
            "required": ["name", "r", "g", "b", "a"],
        },
    },
    {
        "name": "set_material_color",
        "description": (
            "Change the RGBA base color of an existing Blender material. "
            "Color values must be between 0 and 1."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "material_name": {"type": "string"},
                "r": {"type": "number"},
                "g": {"type": "number"},
                "b": {"type": "number"},
                "a": {"type": "number"},
            },
            "required": ["material_name", "r", "g", "b", "a"],
        },
    },
    {
        "name": "assign_material",
        "description": (
            "Assign an existing Blender material to an existing mesh object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "material_name": {"type": "string"},
            },
            "required": ["object_name", "material_name"],
        },
    },
    {
        "name": "get_modifiers",
        "description": (
            "Inspect modifiers on Blender objects. Optionally provide object_name "
            "to inspect one object; omit it to inspect all mesh objects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Optional object name to inspect.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "add_bevel_modifier",
        "description": (
            "Add a Bevel modifier to a mesh object. width is in Blender units "
            "and segments controls bevel smoothness."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "modifier_name": {"type": "string"},
                "width": {"type": "number"},
                "segments": {"type": "integer"},
            },
            "required": ["object_name", "modifier_name", "width", "segments"],
        },
    },
    {
        "name": "set_bevel_modifier",
        "description": "Change width and segments of an existing Bevel modifier.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "modifier_name": {"type": "string"},
                "width": {"type": "number"},
                "segments": {"type": "integer"},
            },
            "required": ["object_name", "modifier_name", "width", "segments"],
        },
    },
    {
        "name": "add_subdivision_modifier",
        "description": "Add a Subdivision Surface modifier to a mesh object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "modifier_name": {"type": "string"},
                "levels": {"type": "integer"},
                "render_levels": {"type": "integer"},
            },
            "required": ["object_name", "modifier_name", "levels", "render_levels"],
        },
    },
    {
        "name": "set_subdivision_modifier",
        "description": (
            "Change viewport and render levels of an existing Subdivision Surface modifier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "modifier_name": {"type": "string"},
                "levels": {"type": "integer"},
                "render_levels": {"type": "integer"},
            },
            "required": ["object_name", "modifier_name", "levels", "render_levels"],
        },
    },
    {
        "name": "remove_modifier",
        "description": (
            "Remove an unapplied modifier from an object without baking it into geometry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "modifier_name": {"type": "string"},
            },
            "required": ["object_name", "modifier_name"],
        },
    },
    {
        "name": "apply_modifier",
        "description": (
            "Apply a named modifier permanently to a named object's geometry. "
            "Use this tool directly when the user explicitly asks to apply an "
            "existing modifier. The controller automatically performs a fresh "
            "get_modifiers preflight, human approval, execution, and post-apply "
            "verification. This is high-risk and requires explicit human approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "modifier_name": {"type": "string"},
            },
            "required": ["object_name", "modifier_name"],
        },
    },
    {
        "name": "get_cameras",
        "description": (
            "Inspect camera objects in the current Blender scene, including location, "
            "rotation, lens, clipping range, and active-camera state."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "create_camera",
        "description": (
            "Create a perspective camera at an exact location with a lens in millimeters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "lens_mm": {"type": "number"},
            },
            "required": ["name", "x", "y", "z", "lens_mm"],
        },
    },
    {
        "name": "move_camera",
        "description": "Move an existing camera to an exact x, y, z location.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["camera_name", "x", "y", "z"],
        },
    },
    {
        "name": "set_camera_lens",
        "description": "Set an existing perspective camera lens in millimeters.",
        "parameters": {
            "type": "object",
            "properties": {
                "camera_name": {"type": "string"},
                "lens_mm": {"type": "number"},
            },
            "required": ["camera_name", "lens_mm"],
        },
    },
    {
        "name": "set_active_camera",
        "description": "Set an existing camera as the active scene camera used for rendering.",
        "parameters": {
            "type": "object",
            "properties": {"camera_name": {"type": "string"}},
            "required": ["camera_name"],
        },
    },
    {
        "name": "aim_camera_at_object",
        "description": (
            "Rotate an existing camera so its forward -Z axis points at a target object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera_name": {"type": "string"},
                "target_object_name": {"type": "string"},
            },
            "required": ["camera_name", "target_object_name"],
        },
    },
    {
        "name": "get_lights",
        "description": (
            "Inspect light objects in the current Blender scene, including type, "
            "location, rotation, energy, color, and relevant size properties."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "create_light",
        "description": (
            "Create a POINT, SUN, SPOT, or AREA light. RGB values are 0 to 1. "
            "The size parameter controls AREA size or soft-shadow size where supported."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "light_type": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "energy": {"type": "number"},
                "r": {"type": "number"},
                "g": {"type": "number"},
                "b": {"type": "number"},
                "size": {"type": "number"},
            },
            "required": ["name", "light_type", "x", "y", "z", "energy", "r", "g", "b", "size"],
        },
    },
    {
        "name": "move_light",
        "description": "Move an existing light to an exact x, y, z location.",
        "parameters": {
            "type": "object",
            "properties": {
                "light_name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["light_name", "x", "y", "z"],
        },
    },
    {
        "name": "set_light_energy",
        "description": "Set the energy/power of an existing light.",
        "parameters": {
            "type": "object",
            "properties": {
                "light_name": {"type": "string"},
                "energy": {"type": "number"},
            },
            "required": ["light_name", "energy"],
        },
    },
    {
        "name": "set_light_color",
        "description": "Set RGB color of an existing light; each channel is 0 to 1.",
        "parameters": {
            "type": "object",
            "properties": {
                "light_name": {"type": "string"},
                "r": {"type": "number"},
                "g": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["light_name", "r", "g", "b"],
        },
    },
    {
        "name": "set_area_light_size",
        "description": "Set size of an existing AREA light.",
        "parameters": {
            "type": "object",
            "properties": {
                "light_name": {"type": "string"},
                "size": {"type": "number"},
            },
            "required": ["light_name", "size"],
        },
    },
    {
        "name": "aim_light_at_object",
        "description": (
            "Rotate an AREA, SPOT, or SUN light so its forward -Z axis points at a target object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "light_name": {"type": "string"},
                "target_object_name": {"type": "string"},
            },
            "required": ["light_name", "target_object_name"],
        },
    },
    {
        "name": "get_render_settings",
        "description": (
            "Inspect the current Blender render engine, active camera, resolution, "
            "output format/path, transparency setting, and available sampling values."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "set_render_engine",
        "description": (
            "Set Blender's render engine. Supported values are CYCLES, "
            "BLENDER_EEVEE_NEXT, and BLENDER_WORKBENCH."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "engine": {"type": "string"},
            },
            "required": ["engine"],
        },
    },
    {
        "name": "set_render_resolution",
        "description": (
            "Set final render width, height, and percentage scale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "percentage": {"type": "integer"},
            },
            "required": ["width", "height", "percentage"],
        },
    },
    {
        "name": "set_render_samples",
        "description": (
            "Set final render sample count for the active engine when the installed "
            "Blender version exposes a supported sample property. Cycles is supported; "
            "EEVEE is handled when its RNA sample property is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "samples": {"type": "integer"},
            },
            "required": ["samples"],
        },
    },
    {
        "name": "set_render_output",
        "description": (
            "Set a safe render output filename and image format. The filename must "
            "contain only a base filename, not directories. Output is constrained "
            "to the project's renders/ directory. Supported formats: PNG, JPEG, OPEN_EXR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "file_format": {"type": "string"},
            },
            "required": ["filename", "file_format"],
        },
    },
    {
        "name": "set_render_transparent",
        "description": (
            "Enable or disable transparent film/background for rendering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
            },
            "required": ["enabled"],
        },
    },
    {
        "name": "render_scene",
        "description": (
            "Render the active Blender scene using the active scene camera. "
            "If save_to_file is true, save to the safe project renders/ output path "
            "configured by set_render_output. The tool verifies Render Result and, "
            "when saving, verifies that the output file exists."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "save_to_file": {"type": "boolean"},
            },
            "required": ["save_to_file"],
        },
    },
    {
        "name": "get_mesh_info",
        "description": (
            "Inspect one mesh object, including vertex/edge/face counts, object "
            "dimensions, local bounds, smooth-face count, materials, and modifiers. "
            "Use this to inspect and verify mesh-modeling operations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"}
            },
            "required": ["object_name"]
        }
    },
    {
        "name": "create_uv_sphere",
        "description": "Create a UV sphere mesh with controlled radius and topology.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "radius": {"type": "number"},
                "segments": {"type": "integer"},
                "ring_count": {"type": "integer"}
            },
            "required": ["name", "x", "y", "z", "radius", "segments", "ring_count"]
        }
    },
    {
        "name": "create_cylinder",
        "description": "Create a cylinder mesh with controlled radius, depth, and radial vertex count.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "radius": {"type": "number"},
                "depth": {"type": "number"},
                "vertices": {"type": "integer"}
            },
            "required": ["name", "x", "y", "z", "radius", "depth", "vertices"]
        }
    },
    {
        "name": "create_cone",
        "description": (
            "Create a cone or truncated cone mesh. radius1 is the bottom radius "
            "and radius2 is the top radius; radius2=0 creates a pointed cone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "radius1": {"type": "number"},
                "radius2": {"type": "number"},
                "depth": {"type": "number"},
                "vertices": {"type": "integer"}
            },
            "required": ["name", "x", "y", "z", "radius1", "radius2", "depth", "vertices"]
        }
    },
    {
        "name": "create_plane",
        "description": "Create a square plane mesh with a controlled side length.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "size": {"type": "number"}
            },
            "required": ["name", "x", "y", "z", "size"]
        }
    },
    {
        "name": "create_torus",
        "description": "Create a torus mesh with controlled major/minor radius and segment counts.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "major_radius": {"type": "number"},
                "minor_radius": {"type": "number"},
                "major_segments": {"type": "integer"},
                "minor_segments": {"type": "integer"}
            },
            "required": [
                "name", "x", "y", "z",
                "major_radius", "minor_radius",
                "major_segments", "minor_segments"
            ]
        }
    },
    {
        "name": "shade_smooth",
        "description": "Enable or disable smooth shading for every polygon of an existing mesh object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "enabled": {"type": "boolean"}
            },
            "required": ["object_name", "enabled"]
        }
    },
    {
        "name": "recalculate_normals",
        "description": "Recalculate all mesh face normals consistently outward using Blender BMesh.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"}
            },
            "required": ["object_name"]
        }
    },
    {
        "name": "scale_mesh_geometry",
        "description": (
            "Permanently scale mesh vertex coordinates around the object's local origin "
            "without changing the object's transform. Destructive; requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "x_factor": {"type": "number"},
                "y_factor": {"type": "number"},
                "z_factor": {"type": "number"}
            },
            "required": ["object_name", "x_factor", "y_factor", "z_factor"]
        }
    },
    {
        "name": "extrude_top_face",
        "description": (
            "Permanently extrude the highest upward-facing face region along local +Z. "
            "Best for box/cylinder-style meshes. Destructive; requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "distance": {"type": "number"}
            },
            "required": ["object_name", "distance"]
        }
    },
    {
        "name": "bevel_mesh_edges",
        "description": (
            "Permanently bevel all mesh edges. Prefer add_bevel_modifier for a "
            "non-destructive workflow. Direct geometry edit; requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "width": {"type": "number"},
                "segments": {"type": "integer"}
            },
            "required": ["object_name", "width", "segments"]
        }
    },
    {
        "name": "get_mesh_regions",
        "description": (
            "Inspect semantic face regions of one mesh: upward, downward, side, "
            "and highest upward-facing top faces. Use this before or after "
            "top-face modeling operations when region structure matters."
        ),
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}}, "required": ["object_name"]}
    },
    {
        "name": "inset_top_face",
        "description": (
            "Permanently inset the highest upward-facing face region of a mesh. "
            "thickness controls the inset amount and depth offsets the inset along its normal. "
            "Direct geometry edit; requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {"object_name": {"type": "string"}, "thickness": {"type": "number"}, "depth": {"type": "number"}},
            "required": ["object_name", "thickness", "depth"]
        }
    },
    {
        "name": "subdivide_mesh",
        "description": (
            "Permanently subdivide all edges of a mesh with a controlled number of cuts. "
            "Useful for adding topology before further modeling. Direct geometry edit; requires approval."
        ),
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}, "cuts": {"type": "integer"}}, "required": ["object_name", "cuts"]}
    },
    {
        "name": "translate_top_face",
        "description": (
            "Permanently translate the highest upward-facing face region by a local x/y/z offset. "
            "Useful for leaning or shaping upper geometry. Direct geometry edit; requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {"object_name": {"type": "string"}, "x_offset": {"type": "number"}, "y_offset": {"type": "number"}, "z_offset": {"type": "number"}},
            "required": ["object_name", "x_offset", "y_offset", "z_offset"]
        }
    },
    {
        "name": "scale_top_face",
        "description": (
            "Permanently scale the highest upward-facing face region around its local centroid in X and Y, preserving Z. "
            "Useful for tapering or widening the top of a mesh. Direct geometry edit; requires approval."
        ),
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}, "x_factor": {"type": "number"}, "y_factor": {"type": "number"}}, "required": ["object_name", "x_factor", "y_factor"]}
    },
    {
        "name": "merge_by_distance",
        "description": (
            "Permanently merge vertices within a distance threshold (remove doubles / merge by distance). "
            "Direct geometry edit; requires approval."
        ),
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}, "distance": {"type": "number"}}, "required": ["object_name", "distance"]}
    },
    {
        "name": "solidify_mesh",
        "description": (
            "Permanently give a mesh shell thickness using Blender BMesh solidification. "
            "Prefer a non-destructive modifier when available. Direct geometry edit; requires approval."
        ),
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string"}, "thickness": {"type": "number"}}, "required": ["object_name", "thickness"]}
    },
    {
        "name": "get_scene_objects",
        "description": (
            "Inspect all objects currently in the Blender scene, including names, "
            "types, and locations. Use this for scene-state questions and verification."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_cube",
        "description": "Create a cube in Blender at an exact x, y, z location.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["name", "x", "y", "z"],
        },
    },
    {
        "name": "move_object",
        "description": "Move an existing Blender object to an exact x, y, z location.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
            },
            "required": ["name", "x", "y", "z"],
        },
    },
    {
        "name": "delete_object",
        "description": "Delete an existing Blender object. This is a high-risk action requiring approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    },
]


SCHEMA_BY_NAME = {schema["name"]: schema for schema in TOOL_SCHEMAS}

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": schema,
    }
    for schema in TOOL_SCHEMAS
]


TOOL_CATEGORIES = {
    "knowledge": {
        "search_blender_docs",
    },

    "objects": {
        "get_scene_objects",
        "create_cube",
        "move_object",
        "delete_object",
    },

    "materials": {
        "get_materials",
        "create_material",
        "set_material_color",
        "assign_material",
    },

    "modifiers": {
        "get_modifiers",
        "add_bevel_modifier",
        "set_bevel_modifier",
        "add_subdivision_modifier",
        "set_subdivision_modifier",
        "remove_modifier",
        "apply_modifier",
    },

    "cameras": {
        "get_cameras",
        "create_camera",
        "move_camera",
        "set_camera_lens",
        "set_active_camera",
        "aim_camera_at_object",
    },

    "lights": {
        "get_lights",
        "create_light",
        "move_light",
        "set_light_energy",
        "set_light_color",
        "set_area_light_size",
        "aim_light_at_object",
    },

    "rendering": {
        "get_render_settings",
        "set_render_engine",
        "set_render_resolution",
        "set_render_samples",
        "set_render_output",
        "set_render_transparent",
        "render_scene",
    },

    "mesh_modeling": {
        "get_mesh_info",
        "get_mesh_regions",
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
    },
}

CATEGORY_CORE_TOOLS = {
    "knowledge": {"search_blender_docs"},
    "objects": {"get_scene_objects"},
    "materials": {"get_materials"},
    "modifiers": {"get_modifiers"},
    "cameras": {"get_cameras"},
    "lights": {"get_lights"},
    "rendering": {"get_render_settings"},
    "mesh_modeling": {"get_mesh_info", "get_mesh_regions"},
}

CATEGORY_FALLBACK_TOOLS = {
    "knowledge": {"search_blender_docs"},
    "objects": {"create_cube", "move_object"},
    "materials": {"create_material", "set_material_color", "assign_material"},
    "modifiers": {"add_bevel_modifier", "set_bevel_modifier", "add_subdivision_modifier", "set_subdivision_modifier"},
    "cameras": {"create_camera", "move_camera", "set_camera_lens", "set_active_camera", "aim_camera_at_object"},
    "lights": {"create_light", "move_light", "set_light_energy", "set_light_color", "set_area_light_size", "aim_light_at_object"},
    "rendering": {"set_render_resolution", "set_render_output", "render_scene"},
    "mesh_modeling": {
        "create_uv_sphere", "create_cylinder", "create_plane", "shade_smooth",
        "extrude_top_face", "inset_top_face", "subdivide_mesh",
        "translate_top_face", "scale_top_face", "merge_by_distance", "solidify_mesh",
    },
}

TOOL_DISCOVERY_HINTS = {
    "get_scene_objects": ("inspect scene", "list objects", "what objects", "scene objects"),
    "create_cube": ("cube", "box", "block", "rectangular"),
    "move_object": ("move object", "move the object", "reposition object", "relocate object"),
    "delete_object": ("delete object", "delete the", "remove object"),
    "get_materials": ("inspect material", "list materials"),
    "create_material": ("create material", "new material", "material named"),
    "set_material_color": ("set material color", "change material color", "make it red", "make it blue", "make it green", "make it white", "make it black", "colour", "color"),
    "assign_material": ("assign material", "apply material", "material to"),
    "add_bevel_modifier": ("add bevel modifier", "bevel modifier", "rounded edges"),
    "set_bevel_modifier": ("change bevel", "set bevel", "bevel width", "bevel segments"),
    "add_subdivision_modifier": ("add subdivision", "subdivision modifier", "subsurf"),
    "set_subdivision_modifier": ("set subdivision", "change subdivision", "subdivision levels"),
    "remove_modifier": ("remove modifier", "delete modifier"),
    "apply_modifier": ("apply modifier", "apply the modifier"),
    "create_camera": ("create camera", "add camera", "camera at"),
    "move_camera": ("move camera", "reposition camera"),
    "set_camera_lens": ("camera lens", "lens", "focal length"),
    "set_active_camera": ("active camera", "make it active", "make the camera active"),
    "aim_camera_at_object": ("aim camera", "aim the camera", "aim it at", "point camera", "look at", "frame the"),
    "create_light": ("create light", "add light", "area light", "point light", "spot light", "sun light"),
    "move_light": ("move light", "reposition light"),
    "set_light_energy": ("light energy", "energy", "power"),
    "set_light_color": ("light color", "light colour"),
    "set_area_light_size": ("area light size", "soft light"),
    "aim_light_at_object": ("aim light", "point light at", "aim the light"),
    "set_render_engine": ("render engine", "cycles", "eevee", "workbench"),
    "set_render_resolution": ("render resolution", "resolution", "1080p", "4k", "800x600"),
    "set_render_samples": ("render samples", "samples"),
    "set_render_output": ("output filename", "output file", ".png", ".jpg", ".jpeg", ".exr"),
    "set_render_transparent": ("transparent background", "transparent render"),
    "render_scene": ("render scene", "render the scene", "render to file", "render"),
    "get_mesh_info": ("inspect mesh", "mesh info", "topology", "vertex count", "face count"),
    "get_mesh_regions": ("top face", "top region", "upward face", "side faces", "bottom face", "mesh regions"),
    "create_uv_sphere": ("uv sphere", "sphere"),
    "create_cylinder": ("cylinder",),
    "create_cone": ("cone", "truncated cone"),
    "create_plane": ("plane", "flat plane"),
    "create_torus": ("torus", "donut", "doughnut"),
    "shade_smooth": ("shade smooth", "smooth shading"),
    "recalculate_normals": ("recalculate normals", "fix normals", "face normals"),
    "scale_mesh_geometry": ("scale geometry", "scale mesh geometry"),
    "extrude_top_face": ("extrude top face", "extrude the top", "raise the top"),
    "bevel_mesh_edges": ("bevel mesh edges", "permanent bevel", "direct bevel"),
    "inset_top_face": ("inset top face", "inset the top face", "inset its top face", "inset the top", "inset face"),
    "subdivide_mesh": ("subdivide mesh", "subdivide the mesh", "add topology", "add cuts"),
    "translate_top_face": ("move top face", "move the top face", "move its top face", "translate top face", "offset the top", "raise the top", "lean the top"),
    "scale_top_face": ("scale top face", "scale the top face", "scale its top face", "scaling its top face", "taper", "narrow the top", "widen the top"),
    "merge_by_distance": ("merge by distance", "remove doubles", "weld vertices", "merge vertices"),
    "solidify_mesh": ("solidify", "give thickness", "mesh thickness", "shell thickness"),
}


OLLAMA_TOOL_BY_NAME = {
    tool["function"]["name"]: tool
    for tool in OLLAMA_TOOLS
}


TOOL_RISK = {
    "search_blender_docs": "low",
    "get_scene_objects": "low",
    "get_materials": "low",
    "create_material": "low",
    "set_material_color": "medium",
    "assign_material": "medium",
    "get_modifiers": "low",
    "add_bevel_modifier": "medium",
    "set_bevel_modifier": "medium",
    "add_subdivision_modifier": "medium",
    "set_subdivision_modifier": "medium",
    "remove_modifier": "medium",
    "apply_modifier": "high",
    "get_cameras": "low",
    "create_camera": "low",
    "move_camera": "medium",
    "set_camera_lens": "medium",
    "set_active_camera": "medium",
    "aim_camera_at_object": "medium",
    "get_lights": "low",
    "create_light": "low",
    "move_light": "medium",
    "set_light_energy": "medium",
    "set_light_color": "medium",
    "set_area_light_size": "medium",
    "aim_light_at_object": "medium",
    "get_render_settings": "low",
    "set_render_engine": "medium",
    "set_render_resolution": "medium",
    "set_render_samples": "medium",
    "set_render_output": "medium",
    "set_render_transparent": "medium",
    "render_scene": "medium",
    "get_mesh_info": "low",
    "create_uv_sphere": "low",
    "create_cylinder": "low",
    "create_cone": "low",
    "create_plane": "low",
    "create_torus": "low",
    "shade_smooth": "medium",
    "recalculate_normals": "medium",
    "scale_mesh_geometry": "high",
    "extrude_top_face": "high",
    "bevel_mesh_edges": "high",
    "get_mesh_regions": "low",
    "inset_top_face": "high",
    "subdivide_mesh": "high",
    "translate_top_face": "high",
    "scale_top_face": "high",
    "merge_by_distance": "high",
    "solidify_mesh": "high",
    "create_cube": "low",
    "move_object": "medium",
    "delete_object": "high",
}

MUTATING_TOOLS = {
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

OBSERVATION_TOOLS = {
    "get_scene_objects",
    "get_materials",
    "get_modifiers",
    "get_cameras",
    "get_lights",
    "get_render_settings",
    "get_mesh_info",
    "get_mesh_regions",
}

# Reliability-oriented grouping.
#
# TOOL_CATEGORIES remains the domain/capability index used by dynamic
# discovery. TOOL_BEHAVIOR_GROUPS is orthogonal: it tells the controller
# how a tool behaves operationally, especially whether automatic replay is
# ever safe after a side effect has already executed.
TOOL_DOMAIN_BY_NAME = {
    tool_name: category
    for category, tool_names in TOOL_CATEGORIES.items()
    for tool_name in tool_names
}

TOOL_BEHAVIOR_GROUPS = {
    "knowledge_read": {
        "search_blender_docs",
    },
    "scene_read": set(OBSERVATION_TOOLS),
    "state_mutation": {
        "create_cube",
        "move_object",
        "create_material",
        "set_material_color",
        "assign_material",
        "add_bevel_modifier",
        "set_bevel_modifier",
        "add_subdivision_modifier",
        "set_subdivision_modifier",
        "remove_modifier",
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
        "create_uv_sphere",
        "create_cylinder",
        "create_cone",
        "create_plane",
        "create_torus",
        "shade_smooth",
        "recalculate_normals",
    },
    "destructive_geometry": {
        "delete_object",
        "apply_modifier",
        "scale_mesh_geometry",
        "extrude_top_face",
        "bevel_mesh_edges",
        "inset_top_face",
        "subdivide_mesh",
        "translate_top_face",
        "scale_top_face",
        "merge_by_distance",
        "solidify_mesh",
    },
    "terminal_side_effect": {
        "render_scene",
    },
}

TOOL_BEHAVIOR_BY_NAME = {
    tool_name: behavior
    for behavior, tool_names in TOOL_BEHAVIOR_GROUPS.items()
    for tool_name in tool_names
}

NO_AUTO_REPLAY_TOOLS = (
    set(MUTATING_TOOLS)
    | TOOL_BEHAVIOR_GROUPS["terminal_side_effect"]
)

def get_tool_domain(tool_or_command):
    if isinstance(tool_or_command, dict):
        tool_name = tool_or_command.get("tool")
    else:
        tool_name = str(tool_or_command)
    return TOOL_DOMAIN_BY_NAME.get(tool_name, "unknown")

def get_tool_behavior(tool_or_command):
    if isinstance(tool_or_command, dict):
        tool_name = tool_or_command.get("tool")
    else:
        tool_name = str(tool_or_command)
    return TOOL_BEHAVIOR_BY_NAME.get(tool_name, "unknown")

VERIFICATION_TOOL_BY_MUTATION = {
    "create_cube": "get_scene_objects",
    "move_object": "get_scene_objects",
    "delete_object": "get_scene_objects",
    "create_material": "get_materials",
    "set_material_color": "get_materials",
    "assign_material": "get_materials",
    "add_bevel_modifier": "get_modifiers",
    "set_bevel_modifier": "get_modifiers",
    "add_subdivision_modifier": "get_modifiers",
    "set_subdivision_modifier": "get_modifiers",
    "remove_modifier": "get_modifiers",
    "apply_modifier": "get_modifiers",
    "create_camera": "get_cameras",
    "move_camera": "get_cameras",
    "set_camera_lens": "get_cameras",
    "set_active_camera": "get_cameras",
    "aim_camera_at_object": "get_cameras",
    "create_light": "get_lights",
    "move_light": "get_lights",
    "set_light_energy": "get_lights",
    "set_light_color": "get_lights",
    "set_area_light_size": "get_lights",
    "aim_light_at_object": "get_lights",
    "set_render_engine": "get_render_settings",
    "set_render_resolution": "get_render_settings",
    "set_render_samples": "get_render_settings",
    "set_render_output": "get_render_settings",
    "set_render_transparent": "get_render_settings",
    "create_uv_sphere": "get_scene_objects",
    "create_cylinder": "get_scene_objects",
    "create_cone": "get_scene_objects",
    "create_plane": "get_scene_objects",
    "create_torus": "get_scene_objects",
    "shade_smooth": "get_mesh_info",
    "recalculate_normals": "get_mesh_info",
    "scale_mesh_geometry": "get_mesh_info",
    "extrude_top_face": "get_mesh_info",
    "bevel_mesh_edges": "get_mesh_info",
    "inset_top_face": "get_mesh_regions",
    "subdivide_mesh": "get_mesh_info",
    "translate_top_face": "get_mesh_regions",
    "scale_top_face": "get_mesh_regions",
    "merge_by_distance": "get_mesh_info",
    "solidify_mesh": "get_mesh_info",
}


def normalize_tool_arguments(raw_arguments):
    if raw_arguments is None:
        return {}

    if isinstance(raw_arguments, dict):
        return raw_arguments

    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {"__invalid_arguments__": raw_arguments}


def tool_call_to_command(tool_call):
    function = tool_call["function"]
    return {
        "tool": function["name"],
        "arguments": normalize_tool_arguments(function.get("arguments", {})),
    }


def _matches_type(value, expected_type):
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)

    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "boolean":
        return isinstance(value, bool)

    return True


def validate_tool_call(command):
    if not isinstance(command, dict):
        return False, "Tool call must be a dictionary."

    tool_name = command.get("tool")

    if not tool_name:
        return False, "Missing tool name."

    if tool_name not in SCHEMA_BY_NAME:
        return False, f"Unknown tool: {tool_name}"

    arguments = command.get("arguments", {})

    if not isinstance(arguments, dict):
        return False, "Tool arguments must be a dictionary."

    schema = SCHEMA_BY_NAME[tool_name]["parameters"]
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for argument_name in required:
        if argument_name not in arguments:
            return False, f"Missing required argument: {argument_name}"

    for argument_name, value in arguments.items():
        if argument_name not in properties:
            return False, f"Unexpected argument: {argument_name}"

        expected_type = properties[argument_name].get("type")

        if not _matches_type(value, expected_type):
            return (
                False,
                f"Argument '{argument_name}' should be {expected_type}, "
                f"got {type(value).__name__}.",
            )

    if tool_name == "search_blender_docs":
        top_k = arguments.get("top_k", 5)
        if not 1 <= top_k <= 8:
            return False, "search_blender_docs top_k must be between 1 and 8."

    if tool_name in {"create_material", "set_material_color"}:
        for channel in ("r", "g", "b", "a"):
            value = float(arguments[channel])
            if not 0.0 <= value <= 1.0:
                return (
                    False,
                    f"Material color channel '{channel}' must be between 0 and 1.",
                )

    if tool_name in {"add_bevel_modifier", "set_bevel_modifier"}:
        if float(arguments["width"]) < 0.0:
            return False, "Bevel width must be non-negative."
        if not 1 <= int(arguments["segments"]) <= 64:
            return False, "Bevel segments must be between 1 and 64."

    if tool_name in {"add_subdivision_modifier", "set_subdivision_modifier"}:
        levels = int(arguments["levels"])
        render_levels = int(arguments["render_levels"])
        if not 0 <= levels <= 6:
            return False, "Subdivision levels must be between 0 and 6."
        if not 0 <= render_levels <= 6:
            return False, "Subdivision render_levels must be between 0 and 6."

    if tool_name in {"create_camera", "set_camera_lens"}:
        lens_mm = float(arguments["lens_mm"])
        if not 1.0 <= lens_mm <= 500.0:
            return False, "Camera lens_mm must be between 1 and 500."

    if tool_name == "create_light":
        light_type = str(arguments["light_type"]).upper()
        if light_type not in {"POINT", "SUN", "SPOT", "AREA"}:
            return False, "light_type must be one of POINT, SUN, SPOT, AREA."
        if float(arguments["energy"]) < 0.0:
            return False, "Light energy must be non-negative."
        if float(arguments["size"]) <= 0.0:
            return False, "Light size must be greater than 0."
        for channel in ("r", "g", "b"):
            if not 0.0 <= float(arguments[channel]) <= 1.0:
                return False, f"Light color channel '{channel}' must be between 0 and 1."

    if tool_name == "set_light_energy" and float(arguments["energy"]) < 0.0:
        return False, "Light energy must be non-negative."

    if tool_name == "set_light_color":
        for channel in ("r", "g", "b"):
            if not 0.0 <= float(arguments[channel]) <= 1.0:
                return False, f"Light color channel '{channel}' must be between 0 and 1."

    if tool_name == "set_area_light_size" and float(arguments["size"]) <= 0.0:
        return False, "AREA light size must be greater than 0."

    if tool_name == "set_render_engine":
        engine = str(arguments["engine"]).upper()
        if engine not in {
            "CYCLES",
            "BLENDER_EEVEE_NEXT",
            "BLENDER_WORKBENCH",
        }:
            return (
                False,
                "Render engine must be CYCLES, BLENDER_EEVEE_NEXT, or BLENDER_WORKBENCH.",
            )

    if tool_name == "set_render_resolution":
        width = int(arguments["width"])
        height = int(arguments["height"])
        percentage = int(arguments["percentage"])

        if not 4 <= width <= 16384:
            return False, "Render width must be between 4 and 16384 pixels."

        if not 4 <= height <= 16384:
            return False, "Render height must be between 4 and 16384 pixels."

        if not 1 <= percentage <= 100:
            return False, "Render percentage must be between 1 and 100."

    if tool_name == "set_render_samples":
        samples = int(arguments["samples"])
        if not 1 <= samples <= 4096:
            return False, "Render samples must be between 1 and 4096."

    if tool_name == "set_render_output":
        filename = str(arguments["filename"]).strip()
        file_format = str(arguments["file_format"]).upper()

        if not filename:
            return False, "Render filename cannot be empty."

        if "/" in filename or "\\" in filename:
            return False, "Render filename must not contain directory separators."

        if filename in {".", ".."}:
            return False, "Invalid render filename."

        if file_format not in {"PNG", "JPEG", "OPEN_EXR"}:
            return False, "file_format must be PNG, JPEG, or OPEN_EXR."

    if tool_name == "create_uv_sphere":
        if float(arguments["radius"]) <= 0.0:
            return False, "Sphere radius must be greater than 0."
        if not 3 <= int(arguments["segments"]) <= 256:
            return False, "Sphere segments must be between 3 and 256."
        if not 3 <= int(arguments["ring_count"]) <= 256:
            return False, "Sphere ring_count must be between 3 and 256."

    if tool_name == "create_cylinder":
        if float(arguments["radius"]) <= 0.0:
            return False, "Cylinder radius must be greater than 0."
        if float(arguments["depth"]) <= 0.0:
            return False, "Cylinder depth must be greater than 0."
        if not 3 <= int(arguments["vertices"]) <= 256:
            return False, "Cylinder vertices must be between 3 and 256."

    if tool_name == "create_cone":
        radius1 = float(arguments["radius1"])
        radius2 = float(arguments["radius2"])
        if radius1 < 0.0 or radius2 < 0.0:
            return False, "Cone radii must be non-negative."
        if radius1 == 0.0 and radius2 == 0.0:
            return False, "At least one cone radius must be greater than 0."
        if float(arguments["depth"]) <= 0.0:
            return False, "Cone depth must be greater than 0."
        if not 3 <= int(arguments["vertices"]) <= 256:
            return False, "Cone vertices must be between 3 and 256."

    if tool_name == "create_plane" and float(arguments["size"]) <= 0.0:
        return False, "Plane size must be greater than 0."

    if tool_name == "create_torus":
        major_radius = float(arguments["major_radius"])
        minor_radius = float(arguments["minor_radius"])
        if major_radius <= 0.0:
            return False, "Torus major_radius must be greater than 0."
        if minor_radius <= 0.0:
            return False, "Torus minor_radius must be greater than 0."
        if minor_radius >= major_radius:
            return False, "Torus minor_radius must be smaller than major_radius."
        if not 3 <= int(arguments["major_segments"]) <= 256:
            return False, "Torus major_segments must be between 3 and 256."
        if not 3 <= int(arguments["minor_segments"]) <= 128:
            return False, "Torus minor_segments must be between 3 and 128."

    if tool_name == "scale_mesh_geometry":
        for axis in ("x_factor", "y_factor", "z_factor"):
            value = float(arguments[axis])
            if not 0.01 <= abs(value) <= 100.0:
                return False, f"{axis} absolute value must be between 0.01 and 100."

    if tool_name == "extrude_top_face":
        if abs(float(arguments["distance"])) > 10000.0:
            return False, "Extrude distance is outside the allowed range."

    if tool_name == "bevel_mesh_edges":
        if float(arguments["width"]) <= 0.0:
            return False, "Mesh bevel width must be greater than 0."
        if not 1 <= int(arguments["segments"]) <= 64:
            return False, "Mesh bevel segments must be between 1 and 64."

    if tool_name == "inset_top_face":
        if not 0.0 < float(arguments["thickness"]) <= 1000.0:
            return False, "Inset thickness must be greater than 0 and at most 1000."
        if abs(float(arguments["depth"])) > 1000.0:
            return False, "Inset depth absolute value must be at most 1000."

    if tool_name == "subdivide_mesh":
        if not 1 <= int(arguments["cuts"]) <= 10:
            return False, "Subdivision cuts must be between 1 and 10."

    if tool_name == "translate_top_face":
        for axis in ("x_offset", "y_offset", "z_offset"):
            if abs(float(arguments[axis])) > 10000.0:
                return False, f"{axis} absolute value must be at most 10000."

    if tool_name == "scale_top_face":
        for axis in ("x_factor", "y_factor"):
            value = float(arguments[axis])
            if not 0.01 <= abs(value) <= 100.0:
                return False, f"{axis} absolute value must be between 0.01 and 100."

    if tool_name == "merge_by_distance":
        if not 0.0 < float(arguments["distance"]) <= 100.0:
            return False, "Merge distance must be greater than 0 and at most 100."

    if tool_name == "solidify_mesh":
        thickness = float(arguments["thickness"])
        if thickness == 0.0 or abs(thickness) > 1000.0:
            return False, "Solidify thickness must be non-zero with absolute value at most 1000."

    return True, "Valid tool call."


def get_tool_risk(command_or_name):
    if isinstance(command_or_name, str):
        tool_name = command_or_name
    else:
        tool_name = command_or_name["tool"]

    return TOOL_RISK.get(tool_name, "high")
