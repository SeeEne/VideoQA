# Filename: generate_blender_stimuli_v3.py
# Purpose: Generate 3D stimuli, varying shape size systematically.
# How to Run: blender --background --python generate_blender_stimuli_v3.py

import bpy
import os
import math
import numpy as np # For arange
import os
import itertools
import collections

# --- CONFIGURATION ---

# Output Settings
BASE_OUTPUT_DIR = "3D_shape_sweep" # Main output folder
RESOLUTION_X = 640
RESOLUTION_Y = 480
FILE_FORMAT = 'PNG'

# Rendering Settings
RENDER_ENGINE = 'CYCLES'
RENDER_SAMPLES = 128 # Increase if needed for larger renders
USE_GPU = True

# Scene Setup
TABLE_SIZE = 6.0
TABLE_THICKNESS = 0.5
TABLE_COLOR = (0.3, 0.15, 0.05, 1.0)
BACKGROUND_COLOR = (0.8, 0.8, 0.8)

# Object Settings
# --- Parameter Sweep ---
SHAPE_SIZES_TO_TEST = np.arange(0.1, 1.25, 0.05) # From 0.6 to 1.2 inclusive, step 0.05
# --- End Parameter Sweep ---
SHAPE_THICKNESS = 0.5 # Keep thickness constant for simplicity
USE_UNIQUE_COLORS = True
SHAPE_COLORS = {
    "square":   (0.8, 0.1, 0.1, 1.0), # Red
    "circle":   (0.8, 0.1, 0.1, 1.0), # same color
    "triangle": (0.8, 0.1, 0.1, 1.0)  # same color
}
# Z offset depends only on fixed thickness now
SHAPE_Z_OFFSET = TABLE_THICKNESS / 2 + SHAPE_THICKNESS / 2

# Camera Settings (Check if needs adjustment for largest size)
CAMERA_LOCATION = (0, -TABLE_SIZE * 1.1, TABLE_SIZE * 1.1) # Might need to pull back Y or increase Z
CAMERA_ROTATION_EULER = (math.radians(45), 0, 0)

# Lighting Settings
LIGHT_TYPE = 'SUN'
LIGHT_LOCATION = (-TABLE_SIZE / 2, -TABLE_SIZE, TABLE_SIZE * 1.5)
LIGHT_ROTATION_EULER = (math.radians(45), math.radians(-30), 0)
LIGHT_ENERGY = 5

# --- Available Shapes ---
SHAPES = ["square", "circle", "triangle"]

# --- Configurations to Generate ---
# Using all permutations for more data
BASE_CONFIGS = list(itertools.permutations(SHAPES)) # [('s','c','t'), ('s','t','c'), ...] - 6 configs

# --- HELPER FUNCTIONS (configure_rendering, clear_scene, setup_environment, create_material, create_shape_object, render_scene - largely unchanged, ensure setup_environment is called only once if static elements don't change) ---
# NOTE: Make sure helper functions use passed parameters (like size, location) correctly.
# The functions from generate_blender_stimuli_v2.py should mostly work.
# We will redefine create_shape_object slightly to pass size directly.

def configure_rendering(engine, samples, use_gpu):
    """Sets up basic render settings."""
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = RESOLUTION_X
    bpy.context.scene.render.resolution_y = RESOLUTION_Y
    bpy.context.scene.render.image_settings.file_format = FILE_FORMAT
    bpy.context.scene.render.film_transparent = False

    if engine == 'CYCLES':
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = True
        if use_gpu:
            bpy.context.scene.cycles.device = 'GPU'
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'CUDA' # Or 'OPTIX' or 'HIP' or 'METAL'
                prefs.get_devices()
                active_gpus = [d.name for d in prefs.devices if d.use]
                if not active_gpus: # Try activating if none are automatically
                    success = False
                    for device in prefs.devices:
                        if device.type == prefs.compute_device_type:
                            device.use = True
                            success = True
                        else:
                            device.use = False
                    if success:
                        active_gpus = [d.name for d in prefs.devices if d.use]
                        print(f"Activated GPU device type: {prefs.compute_device_type}, Devices: {active_gpus}")
                    else:
                        print("Warning: No compatible GPU device found or activated for Cycles. Falling back to CPU.")
                        bpy.context.scene.cycles.device = 'CPU'

                elif active_gpus :
                    print(f"Using active GPU devices: {active_gpus}")

            except Exception as e:
                print(f"Could not set GPU preferences (might be okay, check render device): {e}")
                bpy.context.scene.cycles.device = 'CPU' # Fallback safely
        else:
            bpy.context.scene.cycles.device = 'CPU'

    elif engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_render_samples = samples
        bpy.context.scene.eevee.use_gtao = True


def clear_scene_dynamic_objects(object_list):
    """Deletes objects passed in the list."""
    bpy.ops.object.select_all(action='DESELECT')
    objects_to_delete = []
    for obj_ref in object_list:
        if obj_ref and obj_ref.name in bpy.data.objects:
            objects_to_delete.append(bpy.data.objects[obj_ref.name])

    if objects_to_delete:
        for obj in objects_to_delete:
            obj.select_set(True)
        bpy.ops.object.delete()

def clear_scene():
    """Deletes all mesh objects, lights, and cameras from the current scene."""
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    # Select relevant objects
    for obj in bpy.data.objects:
        if obj.type in ['MESH', 'LIGHT', 'CAMERA']:
            obj.select_set(True)
    # Delete selected
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    # Clear materials not linked (optional cleanup)
    # for material in bpy.data.materials:
    #     if not material.users:
    #         bpy.data.materials.remove(material)

def setup_static_environment():
    """Creates the table, camera, and light ONCE."""
    # Check if already set up (simple check)
    if "Table" in bpy.data.objects:
        print("Static environment likely already exists.")
        return

    print("Setting up static environment (Table, Camera, Light)...")
    # --- Table ---
    bpy.ops.mesh.primitive_plane_add(size=TABLE_SIZE, location=(0, 0, 0))
    table = bpy.context.object
    table.name = "Table"
    table.modifiers.new(name='Solidify', type='SOLIDIFY')
    table.modifiers['Solidify'].thickness = TABLE_THICKNESS
    mat_table = bpy.data.materials.get("TableMaterial")
    if mat_table is None:
        mat_table = create_material("TableMaterial", TABLE_COLOR)
    if table.data.materials: table.data.materials.clear()
    table.data.materials.append(mat_table)

    # --- Camera ---
    bpy.ops.object.camera_add(location=CAMERA_LOCATION, rotation=CAMERA_ROTATION_EULER)
    camera = bpy.context.object
    camera.name = "SceneCamera"
    bpy.context.scene.camera = camera

    # --- Light ---
    bpy.ops.object.light_add(type=LIGHT_TYPE, location=LIGHT_LOCATION, rotation=LIGHT_ROTATION_EULER)
    light = bpy.context.object
    light.name = "SceneLight"
    if hasattr(light.data, 'energy'): light.data.energy = LIGHT_ENERGY

    # --- World Background ---
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (*BACKGROUND_COLOR, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0

def create_material(name, color_rgba):
    """Creates a simple Principled BSDF material."""
    # Reuse existing material if available
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf is None:
        principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')

    principled_bsdf.inputs['Base Color'].default_value = color_rgba
    principled_bsdf.inputs['Roughness'].default_value = 0.7
    principled_bsdf.inputs['Metallic'].default_value = 0.1
    mat_output = mat.node_tree.nodes.get('Material Output')
    if mat_output:
        # Ensure link doesn't already exist (can happen with node reuse)
        link_exists = False
        for link in mat_output.inputs['Surface'].links:
            if link.from_node == principled_bsdf:
                link_exists = True
                break
        if not link_exists:
            mat.node_tree.links.new(principled_bsdf.outputs['BSDF'], mat_output.inputs['Surface'])
    return mat

# Modified to accept current_size directly
def create_shape_object(shape_type, obj_name, location, current_size, thickness, material):
    """Creates a specific shape mesh object with given size."""
    obj = None
    if shape_type == "square":
        # Cube size param is overall, so use current_size directly
        bpy.ops.mesh.primitive_cube_add(size=current_size, location=location)
        obj = bpy.context.object
        # Adjust Z scale to match fixed thickness
        obj.scale.z = thickness / current_size
    elif shape_type == "circle":
        # Cylinder radius is size/2, depth is thickness
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=current_size/2, depth=thickness, location=location)
        obj = bpy.context.object
    elif shape_type == "triangle":
        # Cone radius1 is size/2, depth is thickness
        bpy.ops.mesh.primitive_cone_add(vertices=3, radius1=current_size/2, depth=thickness, location=location)
        obj = bpy.context.object

    if obj:
        obj.name = obj_name
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        # Check if mesh has polygons before shading smooth (cone might not initially?)
        if obj.data.polygons:
            try:
                bpy.ops.object.shade_smooth()
            except Exception as e:
                print(f"Could not shade smooth {obj_name}: {e}") # Handle potential errors

        if material:
            if obj.data.materials:
                obj.data.materials[0] = material
            else:
                obj.data.materials.append(material)
    else:
        print(f"Warning: Could not create shape type '{shape_type}'")
    return obj

def render_scene(filepath):
    """Configures output path and renders the scene."""
    bpy.context.scene.render.filepath = filepath
    print(f"Rendering to {filepath}...")
    try:
        bpy.ops.render.render(write_still=True)
        print("Rendering finished.")
    except Exception as e:
        print(f"ERROR during rendering: {e}")


# --- MAIN SCRIPT LOGIC ---

if __name__ == "__main__":
    print("--- Blender Stimulus Generation Script (v3 - Size Sweep) ---")

    # 1. Initial Setup (Done once)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    base_output_path_full = os.path.join(script_dir, BASE_OUTPUT_DIR) # Relative to script if possible
    print(f"Base output directory: {base_output_path_full}")

    # Ensure Blender is in a state ready for object manipulation
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    clear_scene() # Clear any previous mesh/light/camera objects
    configure_rendering(RENDER_ENGINE, RENDER_SAMPLES, USE_GPU)
    setup_static_environment() # Create table, camera, light

    # Create materials once
    materials = {}
    if USE_UNIQUE_COLORS:
        for shape_name, color in SHAPE_COLORS.items():
            materials[shape_name] = create_material(f"{shape_name}Material", color)
    else:
        SHAPE_COLOR = (0.8, 0.1, 0.1, 1.0)
        mat_shape = create_material("ShapeMaterial", SHAPE_COLOR)
        for shape_name in SHAPES:
            materials[shape_name] = mat_shape

    rendered_configs_by_size = collections.defaultdict(set)
    objects_in_scene = []

    # --- Outer loop for Shape Size ---
    for current_shape_size in SHAPE_SIZES_TO_TEST:
        size_str = f"{current_shape_size:.2f}"
        print(f"\n--- Processing Size: {size_str} ---")

        # Create size-specific output directory
        current_output_dir = os.path.join(base_output_path_full, f"size_{size_str}")
        os.makedirs(current_output_dir, exist_ok=True)

        # Calculate positions based on current size to maintain relative layout
        spacing_factor = current_shape_size * 1.8 # Adjust multiplier for spacing
        current_positions = [
            (0,                spacing_factor * math.sqrt(3)/2, SHAPE_Z_OFFSET),
            (-spacing_factor, -spacing_factor * math.sqrt(3)/6, SHAPE_Z_OFFSET),
            ( spacing_factor, -spacing_factor * math.sqrt(3)/6, SHAPE_Z_OFFSET),
        ]

        # Generate and Render Base Configs and Swaps for this size
        for base_config in BASE_CONFIGS:
            configs_to_render_from_base = {tuple(base_config)}

            # Generate all single swaps from this base
            for i in range(len(base_config)):
                original_shape = base_config[i]
                shapes_to_swap_in = [s for s in SHAPES if s != original_shape]
                for new_shape in shapes_to_swap_in:
                    mod_config = list(base_config)
                    mod_config[i] = new_shape
                    configs_to_render_from_base.add(tuple(mod_config))

            # Render all unique configs for this size derived from this base
            for config_to_render in configs_to_render_from_base:
                if config_to_render not in rendered_configs_by_size[size_str]:
                    print(f"  Processing config: {config_to_render}")

                    # Clear previous dynamic shape objects
                    clear_scene_dynamic_objects(objects_in_scene)
                    objects_in_scene.clear()

                    # Create objects for the current configuration and size
                    for i, shape_type in enumerate(config_to_render):
                        obj_name = f"Shape_{i+1}_{shape_type}"
                        location = current_positions[i]
                        material = materials[shape_type]
                        shape_obj = create_shape_object(shape_type, obj_name, location, current_shape_size, SHAPE_THICKNESS, material)
                        if shape_obj:
                            objects_in_scene.append(shape_obj)

                    # Define filename and render
                    filename = f"config_S{size_str}__{'_'.join(config_to_render)}.png"
                    filepath_abs = os.path.join(current_output_dir, filename)

                    render_scene(filepath_abs)
                    rendered_configs_by_size[size_str].add(config_to_render)

    # Final cleanup of last set of objects
    clear_scene_dynamic_objects(objects_in_scene)

    total_rendered = sum(len(v) for v in rendered_configs_by_size.values())
    print(f"\n--- Generation Complete ---")
    print(f"Rendered {total_rendered} unique configurations across {len(SHAPE_SIZES_TO_TEST)} sizes to '{base_output_path_full}'")