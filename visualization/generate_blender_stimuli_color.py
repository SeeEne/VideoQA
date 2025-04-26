# Filename: generate_blender_stimuli_v4_color_swap.py
# Purpose: Generate 3D stimuli, varying shape size systematically
#          and swapping the color of one object at a time for a fixed shape configuration.
# How to Run: blender --background --python generate_blender_stimuli_v4_color_swap.py

import bpy
import os
import math
import numpy as np # For arange
import collections # For defaultdict

# --- CONFIGURATION ---

# Output Settings
BASE_OUTPUT_DIR = "test_images/3D_color_swap_sweep" # Main output folder
RESOLUTION_X = 640
RESOLUTION_Y = 480
FILE_FORMAT = 'PNG'

# Rendering Settings
RENDER_ENGINE = 'CYCLES'
RENDER_SAMPLES = 128 # Increase if needed for larger renders
USE_GPU = True # Set to False if you don't have a compatible GPU or encounter issues

# Scene Setup
TABLE_SIZE = 6.0
TABLE_THICKNESS = 0.5
TABLE_COLOR = (0.3, 0.15, 0.05, 1.0) # Brown
BACKGROUND_COLOR = (0.8, 0.8, 0.8) # Light grey

# Object Settings
# --- Parameter Sweep ---
SHAPE_SIZES_TO_TEST = np.arange(0.1, 1.25, 0.05) # From 0.1 to 1.2 inclusive, step 0.05
# --- End Parameter Sweep ---

FIXED_SHAPE_CONFIG = ["square", "circle", "triangle"] # The fixed order of shapes
SHAPE_THICKNESS = 0.5 # Keep thickness constant

# --- Color Definitions ---
# Default colors for each shape when NOT being swapped
DEFAULT_SHAPE_COLORS = {
    "square":   (0.8, 0.1, 0.1, 1.0), # Default Red
    "circle":   (0.8, 0.1, 0.1, 1.0), # Default to Red (same as square for consistency)
    "triangle": (0.8, 0.1, 0.1, 1.0)  # Default to Red (same as square for consistency)
}

# Colors to cycle through when swapping ONE object's color
# Includes the default colors as well to generate all combinations
SWAP_COLORS_RGBA = [
    (0.1, 0.1, 0.8, 1.0), # Blue
    (0.1, 0.8, 0.1, 1.0), # Green
    (0.8, 0.8, 0.1, 1.0), # Yellow
    (0.8, 0.1, 0.8, 1.0), # Magenta
    (0.1, 0.8, 0.8, 1.0), # Cyan
    (0.5, 0.5, 0.5, 1.0), # Grey
]
# Optional: Assign names for clearer filenames/logs (match order in SWAP_COLORS_RGBA)
SWAP_COLOR_NAMES = [
    "Blue", "Green", "Yellow", "Magenta", "Cyan", "Grey"
]
if len(SWAP_COLOR_NAMES) != len(SWAP_COLORS_RGBA):
    print("Warning: SWAP_COLOR_NAMES length doesn't match SWAP_COLORS_RGBA length. Using indices for filenames.")
    SWAP_COLOR_NAMES = [str(i) for i in range(len(SWAP_COLORS_RGBA))]


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


# --- HELPER FUNCTIONS (configure_rendering, clear_scene, setup_environment, create_material, create_shape_object, render_scene - largely unchanged, ensure setup_environment is called only once if static elements don't change) ---
# NOTE: Make sure helper functions use passed parameters (like size, location) correctly.

def configure_rendering(engine, samples, use_gpu):
    """Sets up basic render settings."""
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = RESOLUTION_X
    bpy.context.scene.render.resolution_y = RESOLUTION_Y
    bpy.context.scene.render.image_settings.file_format = FILE_FORMAT
    bpy.context.scene.render.film_transparent = False # Background color is used

    if engine == 'CYCLES':
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = True
        if use_gpu:
            bpy.context.scene.cycles.device = 'GPU'
            try:
                # Attempt to configure GPU rendering device (CUDA/OPTIX/etc.)
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'CUDA' # Or 'OPTIX', 'HIP', 'METAL' - try CUDA first
                prefs.get_devices() # Refresh device list

                # Ensure a device of the selected type is actually enabled
                activated = False
                for device in prefs.devices:
                    if device.type == prefs.compute_device_type:
                        device.use = True
                        activated = True
                        print(f"Activated GPU device: {device.name}")
                    else:
                        device.use = False # Explicitly disable others for clarity

                if not activated:
                    # Fallback or try other types if needed
                    print(f"Warning: No {prefs.compute_device_type} device found or activated. Falling back to CPU.")
                    bpy.context.scene.cycles.device = 'CPU'
                else:
                    print(f"Using GPU compute type: {prefs.compute_device_type}")

            except Exception as e:
                print(f"Error setting GPU preferences: {e}. Falling back to CPU.")
                bpy.context.scene.cycles.device = 'CPU' # Fallback safely
        else:
            bpy.context.scene.cycles.device = 'CPU'
            print("Using CPU for rendering.")

    elif engine == 'BLENDER_EEVEE':
        # Eevee specific settings (if you switch)
        bpy.context.scene.eevee.taa_render_samples = samples
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.use_bloom = True
        print("Using Eevee render engine.")


def clear_scene_dynamic_objects(object_list):
    """Deletes objects passed in the list from the current scene."""
    if not object_list:
        return
    bpy.ops.object.select_all(action='DESELECT')
    objects_to_delete = []
    for obj_ref in object_list:
        # Check if the object still exists in Blender's data
        if obj_ref and obj_ref.name in bpy.data.objects:
            objects_to_delete.append(bpy.data.objects[obj_ref.name])

    if objects_to_delete:
        # Select all objects to delete at once
        for obj in objects_to_delete:
            obj.select_set(True)
        # Delete selected objects
        bpy.ops.object.delete()
        # print(f"Cleared {len(objects_to_delete)} dynamic objects.") # Optional: for debugging
    # else:
    #     print("No dynamic objects needed clearing.") # Optional: for debugging


def clear_scene():
    """Deletes all mesh objects, lights, and cameras from the current scene."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT') # Ensure Object mode

    bpy.ops.object.select_all(action='DESELECT')
    # Select objects by type
    for obj in bpy.data.objects:
        if obj.type in ['MESH', 'LIGHT', 'CAMERA']:
            obj.select_set(True)
    # Delete selected objects
    if bpy.context.selected_objects:
        bpy.ops.object.delete()
    print("Cleared initial scene (Meshes, Lights, Cameras).")


def setup_static_environment():
    """Creates the table, camera, and light ONCE."""
    # Check if already set up (simple check by name)
    if "Table" in bpy.data.objects and "SceneCamera" in bpy.data.objects and "SceneLight" in bpy.data.objects:
        print("Static environment (Table, Camera, Light) already exists.")
        return

    print("Setting up static environment (Table, Camera, Light)...")

    # --- Table ---
    bpy.ops.mesh.primitive_plane_add(size=TABLE_SIZE, location=(0, 0, 0))
    table = bpy.context.object
    table.name = "Table"
    # Add thickness using Solidify modifier
    mod = table.modifiers.new(name='Solidify', type='SOLIDIFY')
    mod.thickness = TABLE_THICKNESS
    mod.offset = 0 # Center the thickness around the original plane
    # Create and assign table material
    mat_table = create_material("TableMaterial", TABLE_COLOR) # Reuse or create
    if table.data.materials: table.data.materials.clear()
    table.data.materials.append(mat_table)

    # --- Camera ---
    bpy.ops.object.camera_add(location=CAMERA_LOCATION, rotation=CAMERA_ROTATION_EULER)
    camera = bpy.context.object
    camera.name = "SceneCamera"
    bpy.context.scene.camera = camera # Set as active camera

    # --- Light ---
    bpy.ops.object.light_add(type=LIGHT_TYPE, location=LIGHT_LOCATION, rotation=LIGHT_ROTATION_EULER)
    light = bpy.context.object
    light.name = "SceneLight"
    if hasattr(light.data, 'energy'): light.data.energy = LIGHT_ENERGY
    if hasattr(light.data, 'angle'): light.data.angle = math.radians(10) # Make Sun light slightly softer if needed
    if hasattr(light.data, 'use_shadow'): light.data.use_shadow = True

    # --- World Background ---
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    # Check if Background node exists, otherwise create basic setup
    bg_node = world.node_tree.nodes.get("Background")
    if not bg_node:
        world.node_tree.nodes.clear() # Start fresh if needed
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
        out_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        world.node_tree.links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])

    # Set background color and strength
    if bg_node:
        bg_node.inputs["Color"].default_value = (*BACKGROUND_COLOR, 1.0) # BG needs alpha too
        bg_node.inputs["Strength"].default_value = 1.0
    else:
        print("Warning: Could not set world background color.")

    print("Static environment setup complete.")


def create_material(name, color_rgba):
    """Creates or retrieves a simple Principled BSDF material."""
    # Reuse existing material if available to avoid duplicates
    if name in bpy.data.materials:
        mat = bpy.data.materials[name]
        # Ensure the color is up-to-date if reusing (might not be needed if names are unique per color)
        if mat.use_nodes and mat.node_tree.nodes.get('Principled BSDF'):
            mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = color_rgba
        return mat

    # Create new material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear() # Start with a clean node tree

    # Create Principled BSDF shader node
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = (0, 0)
    principled_bsdf.inputs['Base Color'].default_value = color_rgba
    principled_bsdf.inputs['Roughness'].default_value = 0.7 # Adjust for desired look
    principled_bsdf.inputs['Metallic'].default_value = 0.1 # Slightly metallic look

    # Create Material Output node
    mat_output = nodes.new(type='ShaderNodeOutputMaterial')
    mat_output.location = (300, 0)

    # Link BSDF to Output
    links.new(principled_bsdf.outputs['BSDF'], mat_output.inputs['Surface'])

    return mat


# Modified to accept current_size directly
def create_shape_object(shape_type, obj_name, location, current_size, thickness, material):
    """Creates a specific shape mesh object with given size, thickness, and material."""
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
        obj.rotation_euler = (0, 0, 0) # Ensure no default rotation

    else:
        print(f"Warning: Unknown shape type '{shape_type}' requested.")
        return None # Return None if shape type is invalid

    # Common operations for valid shapes
    if obj:
        obj.name = obj_name
        # Apply material
        if material:
            if not obj.data.materials:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material # Replace default material

        # Apply Shade Smooth for curved surfaces (like circle)
        if shape_type in ["circle"]: # Add other shapes if needed
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            # Check if the object has polygons before trying to shade smooth
            if obj.data and obj.data.polygons:
                try:
                    bpy.ops.object.shade_smooth()
                except Exception as e:
                    print(f"Could not shade smooth {obj_name}: {e}") # Handle potential errors
            else:
                print(f"Skipping shade smooth for {obj_name} as it has no polygons.") # Should not happen for primitives

    return obj


def render_scene(filepath):
    """Configures output path and renders the scene."""
    abs_filepath = os.path.abspath(filepath)
    bpy.context.scene.render.filepath = abs_filepath
    print(f"Rendering to {abs_filepath}...")
    try:
        bpy.ops.render.render(write_still=True)
        print("Rendering finished.")
    except Exception as e:
        print(f"ERROR during rendering: {e}")
        # Consider adding more robust error handling if needed

# --- MAIN SCRIPT LOGIC ---

if __name__ == "__main__":
    print("--- Blender Stimulus Generation Script (v4 - Color Swap Sweep) ---")

    # 1. Initial Setup (Done once)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    base_output_path_full = os.path.join(script_dir, BASE_OUTPUT_DIR) # Relative to script if possible
    print(f"Base output directory: {base_output_path_full}")

    # Ensure Blender is in Object mode before starting
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    clear_scene() # Clear any previous mesh/light/camera objects from default file
    configure_rendering(RENDER_ENGINE, RENDER_SAMPLES, USE_GPU)
    setup_static_environment() # Create table, camera, light

    # 2. Pre-create all needed materials to avoid creating them in the loop
    print("Pre-creating materials...")
    default_materials = {}
    for shape_name, color in DEFAULT_SHAPE_COLORS.items():
        mat_name = f"Default_{shape_name}"
        default_materials[shape_name] = create_material(mat_name, color)

    swap_materials = {}
    for i, color in enumerate(SWAP_COLORS_RGBA):
        color_name = SWAP_COLOR_NAMES[i]
        mat_name = f"SwapColor_{i}_{color_name}"
        swap_materials[i] = create_material(mat_name, color) # Store by index
    print(f"Created {len(default_materials)} default and {len(swap_materials)} swap materials.")


    # 3. Main Loop: Iterate through sizes, then positions to swap color, then colors
    objects_in_scene = [] # Keep track of dynamically created shape objects
    total_rendered_count = 0

    # --- Outer loop for Shape Size ---
    for current_shape_size in SHAPE_SIZES_TO_TEST:
        size_str = f"{current_shape_size:.2f}" # Format size for directory/filename
        print(f"\n--- Processing Size: {size_str} ---")

        # Create size-specific output directory
        current_output_dir = os.path.join(base_output_path_full, f"size_{size_str}")
        os.makedirs(current_output_dir, exist_ok=True)

        # Calculate positions based on current size and spacing
        # Place them in a triangular arrangement, adjust spacing based on size
        # spacing_factor = current_shape_size * 2.0 # Adjust multiplier for desired spacing
        # center_y_offset = spacing_factor * math.sqrt(3)/6 # Adjust Y for equilateral triangle center
        # current_positions = [
        #      (0,               center_y_offset + spacing_factor * math.sqrt(3)/3, SHAPE_Z_OFFSET), # Top point
        #      (-spacing_factor/2, center_y_offset - spacing_factor * math.sqrt(3)/6, SHAPE_Z_OFFSET), # Bottom-left
        #      ( spacing_factor/2, center_y_offset - spacing_factor * math.sqrt(3)/6, SHAPE_Z_OFFSET), # Bottom-right
        # ]
        spacing_factor = current_shape_size * 1.8 # Adjust multiplier for spacing
        current_positions = [
            (0,                spacing_factor * math.sqrt(3)/2, SHAPE_Z_OFFSET),
            (-spacing_factor, -spacing_factor * math.sqrt(3)/6, SHAPE_Z_OFFSET),
            ( spacing_factor, -spacing_factor * math.sqrt(3)/6, SHAPE_Z_OFFSET),
        ]
        # Ensure positions match FIXED_SHAPE_CONFIG order:
        # Pos 0 -> square, Pos 1 -> circle, Pos 2 -> triangle

        # --- Loop through which object position gets the swapped color ---
        for swap_pos_idx in range(len(FIXED_SHAPE_CONFIG)):
            shape_being_swapped = FIXED_SHAPE_CONFIG[swap_pos_idx]
            print(f"  Swapping color for shape at position {swap_pos_idx} ({shape_being_swapped})")

            # --- Loop through the available swap colors ---
            for swap_color_idx, swap_color_rgba in enumerate(SWAP_COLORS_RGBA):
                swap_color_name = SWAP_COLOR_NAMES[swap_color_idx]
                print(f"    Using swap color: {swap_color_name}")

                # Clear previous dynamic shape objects before creating new ones
                clear_scene_dynamic_objects(objects_in_scene)
                objects_in_scene.clear() # Reset the list

                # Create the three shapes for this specific configuration
                current_config_shapes = [] # Keep track of shape types for filename
                for obj_idx in range(len(FIXED_SHAPE_CONFIG)):
                    shape_type = FIXED_SHAPE_CONFIG[obj_idx]
                    location = current_positions[obj_idx]
                    obj_name = f"Shape_{obj_idx+1}_{shape_type}"

                    # Determine the material: Use swap color if it's the swap position, else default
                    if obj_idx == swap_pos_idx:
                        material_to_use = swap_materials[swap_color_idx]
                        current_config_shapes.append(f"{shape_type}({swap_color_name})") # Indicate swapped color in config list
                    else:
                        material_to_use = default_materials[shape_type]
                        # Get default color name for clarity in filename (optional)
                        # default_color_name = next((name for name, color in DEFAULT_SHAPE_COLORS.items() if color == material_to_use.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value), "Default")
                        current_config_shapes.append(shape_type) # Just use shape type for non-swapped


                    # Create the shape object
                    shape_obj = create_shape_object(shape_type, obj_name, location, current_shape_size, SHAPE_THICKNESS, material_to_use)
                    if shape_obj:
                        objects_in_scene.append(shape_obj) # Add to list for later cleanup

                # Define filename reflecting size, swapped position, and color used
                # Format: size_X.XX_swapPos_Y_swapColor_Z_config_shape1_shape2_shape3.png
                base_filename = f"size_{size_str}_swapPos_{swap_pos_idx}_swapColor_{swap_color_name}"
                # Optional: Add shape config for extra clarity, though it's fixed
                # config_str = '_'.join(FIXED_SHAPE_CONFIG)
                # filename = f"{base_filename}_config_{config_str}.{FILE_FORMAT.lower()}"
                filename = f"{base_filename}.{FILE_FORMAT.lower()}" # Simpler filename
                filepath_abs = os.path.join(current_output_dir, filename)

                # Render the scene with the current setup
                render_scene(filepath_abs)
                total_rendered_count += 1

    # Final cleanup of the last set of dynamic objects
    clear_scene_dynamic_objects(objects_in_scene)

    print(f"\n--- Generation Complete ---")
    print(f"Rendered {total_rendered_count} images across {len(SHAPE_SIZES_TO_TEST)} sizes.")
    print(f"Output saved in '{base_output_path_full}'")