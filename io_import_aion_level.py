import bpy
import struct
import math
import os
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from mathutils import Vector, Euler, Matrix
from bpy_extras.io_utils import ImportHelper
from bpy.props import (StringProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty)
from bpy.types import (Operator,
                       Panel,
                       PropertyGroup,
                       AddonPreferences)
import time
import bmesh
from collections import defaultdict
from math import radians
import numpy as np
import logging

blend_logger = logging.getLogger("H32HeightmapImporter")
blend_logger.setLevel(logging.INFO)

bl_info = {
    "name": "Aion Map Importer",
    "author": "Angry Catster",
    "version": (1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Tools > Aion Importer",
    "description": "Import Aion game maps (heightmaps, vegetation, brushes, and mission objects) from .pak archives",
    "category": "Import",
}

class VegetationObject:
    def __init__(self):
        self.position = Vector((0, 0, 0))
        self.object_id = 0
        self.scale = 1.0
        self.heading = 0
        self.cgf_path = ""
        self.short_name = ""
    def __str__(self):
        return f"ID: {self.object_id}, Pos: {self.position}, Scale: {self.scale}, Heading: {self.heading}, Model: {self.short_name}"

class MissionObject:
    def __init__(self):
        self.name = ""
        self.model_path = ""
        self.position = Vector((0, 0, 0))
        self.rotation = Euler((0, 0, 0))
        self.scale = Vector((1, 1, 1))
    def __str__(self):
        return f"{self.name} - {self.model_path} at {self.position}"

class AionImporterPreferences(AddonPreferences):
    bl_idname = __name__
    models_root: StringProperty(
        name="Models Root Directory",
        subtype='DIR_PATH',
        default="D:\\AionResources",
        description="Root directory where CGF models are stored"
    )
    def draw(self, context):
        layout = self.layout
        layout.label(text="Set the root directory for game assets:")
        layout.prop(self, "models_root")

class AionImporterProperties(PropertyGroup):
    remove_nodraw: BoolProperty(
        name="Remove NoDraw Geometry",
        default=True,
        description="Remove faces with materials containing '(NoDraw)'"
    )
    import_water_plane: BoolProperty(
        name="Import Water Plane",
        description="Import a water plane based on leveldata.xml settings",
        default=True
    )
    enable_smooth_blending: BoolProperty(
        name="Material Blending",
        description="Enable smooth transitions between materials on heightmap",
        default=True
    )
    blend_radius: FloatProperty(
        name="Blend Radius",
        description="Radius for material blending in grid units",
        default=2.0,
        min=0.5,
        max=10.0
    )

def extract_all_paks_to_temp(folder_path, temp_dir):
    """Extracts contents of relevant .pak files from folder_path to temp_dir"""
    extracted_paks = []
    skipped_paks = ["terrainlm.pak", "levellm.pak"]
    for filename in os.listdir(folder_path):
        if (filename.lower().endswith('.pak') or filename.lower().endswith('.zip')) and filename.lower() not in skipped_paks:
            pak_path = os.path.join(folder_path, filename)
            try:
                with zipfile.ZipFile(pak_path, 'r') as pak:
                    pak.extractall(path=temp_dir)
                    extracted_paks.append(filename)
                    print(f"Extracted: {filename}")
            except zipfile.BadZipFile:
                print(f"Warning: {pak_path} is not a valid zip file or is corrupted.")
            except Exception as e:
                print(f"Error extracting {pak_path}: {e}")
    return extracted_paks

class H32HeightmapImporter:
    @staticmethod
    def create_terrain_blend_material(surface_types, material_name="TerrainBlend"):
        """Create a multi-material blending shader using vertex colors and mix nodes"""
        blend_logger.info(f"Creating terrain blend material: {material_name}")
        mat = bpy.data.materials.new(material_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
        # Create output node
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (1200, 0)
        # Create main BSDF
        main_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        main_bsdf.location = (1000, 0)
        main_bsdf.inputs['Roughness'].default_value = 1.0
        main_bsdf.inputs['Specular'].default_value = 0.0
        links.new(main_bsdf.outputs[0], output_node.inputs[0])
        # Create UV input
        uv_node = nodes.new('ShaderNodeUVMap')
        uv_node.uv_map = "ScaledUV"
        uv_node.location = (-800, 0)
        # Create vertex color nodes for blend weights (8 layers for 32 materials)
        vertex_color_nodes = []
        for i in range(8):  # 8 layers for 32 materials (4 per layer)
            vc_node = nodes.new('ShaderNodeVertexColor')
            vc_node.layer_name = f"BlendWeights_{i}"
            vc_node.location = (-600, i * -200)
            vertex_color_nodes.append(vc_node)
        # Create texture nodes
        texture_nodes = []
        current_x = -400
        current_y = 200
        # Collect active materials (those with valid textures)
        active_materials = []
        for mat_id, (tex_path, scale_x, scale_y) in enumerate(surface_types):
            if mat_id >= 32 or not tex_path:
                continue
            if os.path.exists(tex_path):
                active_materials.append((mat_id, tex_path, scale_x, scale_y))
                blend_logger.info(f"Adding material {mat_id}: {os.path.basename(tex_path)} (scale: {scale_x}x{scale_y})")
            else:
                blend_logger.warning(f"Texture not found for material {mat_id}: {tex_path}")
        blend_logger.info(f"Processing {len(active_materials)} active materials for blending")
        if not active_materials:
            blend_logger.warning("No active materials found, creating default material")
            # Create a default material
            links.new(uv_node.outputs[0], main_bsdf.inputs[0])
            return mat
        # Create texture nodes
        for mat_id, tex_path, scale_x, scale_y in active_materials:
            # Texture coordinate scaling
            mapping_node = nodes.new('ShaderNodeMapping')
            mapping_node.location = (current_x - 200, current_y)
            # Apply scaling like in the first script (scale * 4)
            mapping_node.inputs['Scale'].default_value = (scale_x * 4, scale_y * 4, 1.0)
            links.new(uv_node.outputs[0], mapping_node.inputs[0])
            # Texture node
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (current_x, current_y)
            try:
                tex_node.image = bpy.data.images.load(tex_path)
                # Pack the image data into the .blend file
                if tex_node.image:
                     tex_node.image.pack()
                     blend_logger.info(f"Packed texture: {tex_path}")
            except Exception as e:
                 blend_logger.error(f"Failed to load or pack texture {tex_path}: {e}")
            links.new(mapping_node.outputs[0], tex_node.inputs[0])
            texture_nodes.append((mat_id, tex_node))
            current_y -= 300
            if (len(texture_nodes)) % 4 == 0:
                current_x += 200
                current_y = 200
        if len(texture_nodes) == 1:
            links.new(texture_nodes[0][1].outputs[0], main_bsdf.inputs[0])
        else:
            # Sort texture nodes by material ID to ensure correct mapping
            texture_nodes.sort(key=lambda x: x[0])
            # Create a list of (color_output, blend_factor_socket) for each active material
            material_data = []
            for mat_id, tex_node in texture_nodes:
                vc_layer_idx = mat_id // 4
                channel_idx = mat_id % 4
                if vc_layer_idx >= len(vertex_color_nodes):
                     blend_logger.warning(f"Vertex color layer index {vc_layer_idx} out of range for material {mat_id}")
                     continue
                vc_node = vertex_color_nodes[vc_layer_idx]
                # Extract the correct channel for the blend weight
                if channel_idx == 0: # Red
                    separate_rgb = nodes.new('ShaderNodeSeparateRGB')
                    separate_rgb.location = (vc_node.location.x + 200, vc_node.location.y)
                    links.new(vc_node.outputs['Color'], separate_rgb.inputs['Image'])
                    blend_factor_socket = separate_rgb.outputs['R']
                elif channel_idx == 1: # Green
                    separate_rgb = nodes.new('ShaderNodeSeparateRGB')
                    separate_rgb.location = (vc_node.location.x + 200, vc_node.location.y - 100)
                    links.new(vc_node.outputs['Color'], separate_rgb.inputs['Image'])
                    blend_factor_socket = separate_rgb.outputs['G']
                elif channel_idx == 2: # Blue
                    separate_rgb = nodes.new('ShaderNodeSeparateRGB')
                    separate_rgb.location = (vc_node.location.x + 200, vc_node.location.y - 200)
                    links.new(vc_node.outputs['Color'], separate_rgb.inputs['Image'])
                    blend_factor_socket = separate_rgb.outputs['B']
                elif channel_idx == 3: # Alpha
                    blend_factor_socket = vc_node.outputs['Alpha']
                else:
                    blend_logger.error(f"Invalid channel index {channel_idx} for material {mat_id}")
                    continue
                material_data.append((tex_node.outputs['Color'], blend_factor_socket, mat_id))
            current_blend_data = material_data[:]
            mix_x_start = 200
            mix_y_start = 0
            mix_y_offset_step = -150
            iteration = 0
            while len(current_blend_data) > 1:
                next_blend_data = []
                mix_y_current = mix_y_start + (iteration * mix_y_offset_step * (len(current_blend_data) // 2 + 1))
                for i in range(0, len(current_blend_data), 2):
                    if i + 1 < len(current_blend_data):
                        # We have a pair to blend
                        color_out_1, weight_out_1, mat_id_1 = current_blend_data[i]
                        color_out_2, weight_out_2, mat_id_2 = current_blend_data[i+1]
                        # Create a math node to calculate the denominator (weight1 + weight2)
                        math_add = nodes.new('ShaderNodeMath')
                        math_add.operation = 'ADD'
                        math_add.location = (mix_x_start + iteration * 200, mix_y_current + i * 50)
                        links.new(weight_out_1, math_add.inputs[0])
                        links.new(weight_out_2, math_add.inputs[1])
                        # Create a math node to calculate the blend factor: weight2 / (weight1 + weight2)
                        # Handle potential division by zero by using a small epsilon or Clamp
                        math_divide = nodes.new('ShaderNodeMath')
                        math_divide.operation = 'DIVIDE'
                        math_divide.location = (mix_x_start + iteration * 200 + 200, mix_y_current + i * 50)
                        links.new(weight_out_2, math_divide.inputs[0]) # Numerator
                        links.new(math_add.outputs[0], math_divide.inputs[1]) # Denominator
                        # Optional: Clamp the blend factor to [0, 1] to prevent artifacts if sum is zero
                        # This handles cases where both weights might be zero or very small
                        math_clamp = nodes.new('ShaderNodeClamp')
                        math_clamp.location = (mix_x_start + iteration * 200 + 400, mix_y_current + i * 50)
                        links.new(math_divide.outputs[0], math_clamp.inputs['Value'])
                        # Create the mix node
                        mix_shader = nodes.new('ShaderNodeMixRGB')
                        mix_shader.location = (mix_x_start + iteration * 200 + 600, mix_y_current + i * 50)
                        mix_shader.blend_type = 'MIX'
                        mix_shader.use_clamp = True # Clamp output to [0,1]
                        # Connect colors
                        links.new(color_out_1, mix_shader.inputs['Color1'])
                        links.new(color_out_2, mix_shader.inputs['Color2'])
                        # Connect the clamped blend factor
                        links.new(math_clamp.outputs[0], mix_shader.inputs['Fac'])
                        # The output of this mix becomes an element for the next iteration
                        next_blend_data.append((mix_shader.outputs['Color'], math_add.outputs[0], f"{mat_id_1}_{mat_id_2}"))
                    else:
                        # Odd number of elements, carry the last one forward
                        next_blend_data.append(current_blend_data[i])
                current_blend_data = next_blend_data
                iteration += 1
            # Connect the final result
            if current_blend_data:
                final_color_output = current_blend_data[0][0]
                links.new(final_color_output, main_bsdf.inputs['Base Color'])
            else:
                blend_logger.error("Error in pairwise blending logic: no final color output.")
                links.new(material_data[0][0], main_bsdf.inputs['Base Color'])
        blend_logger.info("Terrain blend material created successfully")
        return mat

    @staticmethod
    def calculate_blend_weights(vertices, materials, width, blend_radius=2.0):
        """Calculate smooth blend weights based on material boundaries"""
        blend_logger.info(f"Calculating blend weights with radius {blend_radius}")
        vertex_count = len(vertices)
        # Create blend weight arrays for each vertex (up to 32 materials in 8 RGBA layers)
        blend_weights = np.zeros((vertex_count, 32), dtype=np.float32)
        # First pass: Set primary material weights (like in the first script)
        for i, mat_id in enumerate(materials):
            if mat_id < 32:  # Limit to 32 materials
                blend_weights[i, mat_id] = 1.0
        # Second pass: Create smooth transitions
        blend_logger.info("Creating smooth material transitions...")
        for y in range(width):
            for x in range(width):
                vertex_idx = y * width + x
                current_mat = materials[vertex_idx]
                if current_mat >= 32:  # Skip invalid materials
                    continue
                # Sample neighboring vertices within blend radius
                neighbor_materials = defaultdict(float)
                radius_int = int(math.ceil(blend_radius))
                for dy in range(-radius_int, radius_int + 1):
                    for dx in range(-radius_int, radius_int + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < width:
                            neighbor_idx = ny * width + nx
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance <= blend_radius and neighbor_idx != vertex_idx:
                                neighbor_mat = materials[neighbor_idx]
                                if neighbor_mat < 32 and neighbor_mat != current_mat:
                                    weight = 1.0 - (distance / blend_radius)
                                    neighbor_materials[neighbor_mat] += weight
                # Apply blending weights
                if neighbor_materials:
                    # Reduce weight of current material
                    blend_amount = 0.3  # Blend strength
                    total_neighbor_weight = sum(neighbor_materials.values()) * blend_amount
                    # Ensure we don't exceed 1.0 total weight
                    if total_neighbor_weight > 0.9:
                        total_neighbor_weight = 0.9
                    # Reduce current material weight
                    blend_weights[vertex_idx, current_mat] *= (1.0 - total_neighbor_weight)
                    # Distribute remaining weight to neighboring materials
                    total_raw_weight = sum(neighbor_materials.values())
                    if total_raw_weight > 0:  # Avoid division by zero
                        for neighbor_mat, raw_weight in neighbor_materials.items():
                            weight = (raw_weight / total_raw_weight) * total_neighbor_weight
                            blend_weights[vertex_idx, neighbor_mat] += weight
        # Normalize weights to sum to 1 (important for correct blending)
        # This step ensures that even if the initial calculation doesn't perfectly sum to 1,
        # the final blend will be correct.
        weight_sums = np.sum(blend_weights, axis=1, keepdims=True)
        # Avoid division by zero for vertices that might have zero total weight (e.g., cutouts handled elsewhere)
        weight_sums_safe = np.where(weight_sums > 0, weight_sums, 1.0)
        blend_weights = blend_weights / weight_sums_safe
        blend_logger.info("Blend weight calculation completed")
        return blend_weights

    @staticmethod
    def apply_vertex_colors(mesh, blend_weights, width):
        """Apply blend weights as vertex colors (8 RGBA layers for 32 materials)"""
        blend_logger.info("Applying vertex colors for material blending")
        # Create vertex color layers
        color_layers = []
        for i in range(8):  # 8 layers for 32 materials (4 per layer)
            layer_name = f"BlendWeights_{i}"
            if layer_name in mesh.vertex_colors:
                color_layer = mesh.vertex_colors[layer_name]
            else:
                color_layer = mesh.vertex_colors.new(name=layer_name)
            color_layers.append(color_layer)
        # Apply colors to vertices through loops
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vertex_idx = mesh.loops[loop_idx].vertex_index
                # Pack 32 material weights into 8 RGBA channels
                for layer_idx in range(8):
                    base_mat_idx = layer_idx * 4
                    # Ensure indices are within bounds before accessing
                    r = blend_weights[vertex_idx, base_mat_idx] if base_mat_idx < 32 else 0.0
                    g = blend_weights[vertex_idx, base_mat_idx + 1] if base_mat_idx + 1 < 32 else 0.0
                    b = blend_weights[vertex_idx, base_mat_idx + 2] if base_mat_idx + 2 < 32 else 0.0
                    a = blend_weights[vertex_idx, base_mat_idx + 3] if base_mat_idx + 3 < 32 else 0.0
                    color_layers[layer_idx].data[loop_idx].color = (r, g, b, a) # RGBA order
        blend_logger.info("Vertex colors applied successfully")

    # --- Main Import Method ---
    @staticmethod
    def load_h32_heightmap(h32_path, leveldata_xml_path, enable_smooth_blending=True, blend_radius=2.0, import_water_plane=True):
        """Load heightmap with optional smooth material blending and water plane"""
        # Use the provided paths directly (from temp extraction)
        map_texture_path = os.path.join(os.path.dirname(leveldata_xml_path), "detail")

        if not os.path.exists(leveldata_xml_path):
            raise Exception(f"leveldata.xml not found at expected location: {leveldata_xml_path}")
        if not os.path.exists(map_texture_path):
            raise Exception(f"Textures directory not found at expected location: {map_texture_path}")

        blend_logger.info(f"Loading H32 heightmap: {h32_path}")
        blend_logger.info(f"Smooth blending: {enable_smooth_blending}, Blend radius: {blend_radius}, Import water: {import_water_plane}")

        # Parse surface types with scale values
        tree = ET.parse(leveldata_xml_path)
        surface_types = []
        for st in tree.findall('.//SurfaceTypes/SurfaceType'):
            tex_path = st.get('DetailTexture', '')
            scale_x = float(st.get('DetailScaleX', 1.0))
            scale_y = float(st.get('DetailScaleY', 1.0))
            if tex_path:
                # Use the detail textures directory - resolve full path
                full_tex_path = os.path.join(map_texture_path, os.path.basename(tex_path.replace('\\', os.sep)))
                if os.path.exists(full_tex_path):
                    surface_types.append((full_tex_path, scale_x, scale_y))
                    blend_logger.info(f"Found texture: {os.path.basename(full_tex_path)} (scale: {scale_x}x{scale_y})")
                else:
                    surface_types.append((None, 1.0, 1.0))
                    blend_logger.warning(f"Texture not found: {full_tex_path}")
            else:
                surface_types.append((None, 1.0, 1.0))
        blend_logger.info(f"Loaded {len(surface_types)} surface types")

        # Read heightmap
        blend_logger.info("Reading heightmap data")
        with open(h32_path, 'rb') as f:
            data = f.read()
        vertex_count = len(data) // 3
        width = int(math.sqrt(vertex_count))
        blend_logger.info(f"Heightmap dimensions: {width}x{width} ({vertex_count} vertices)")

        # Parse vertices and materials
        vertices = []
        materials = []
        cutout_indices = set()
        blend_logger.info("Parsing vertex data")
        for i in range(vertex_count):
            offset = i * 3
            height = struct.unpack_from('<H', data, offset)[0]
            mat = data[offset + 2]
            if mat == 0x3F:
                cutout_indices.add(i)
            x = (i % width) * 2
            y = (i // width) * 2
            z = height / 32.0
            vertices.append(Vector((y, x, z)))
            materials.append(mat)
        blend_logger.info(f"Found {len(cutout_indices)} cutout vertices")

        # Create mesh
        blend_logger.info("Creating Blender mesh")
        mesh_name = "Heightmap" if not enable_smooth_blending else "HeightmapBlended"
        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(mesh_name, mesh)
        bpy.context.collection.objects.link(obj)

        # --- Material Assignment Logic ---
        if enable_smooth_blending:
            blend_logger.info("Generating smooth material transitions")
            # Create faces and UVs (simple, no per-material scaling in shader)
            faces = []
            uvs = []
            face_materials = []
            blend_logger.info("Creating faces and UV coordinates for blending")
            for y in range(width - 1):
                for x in range(width - 1):
                    v1 = y * width + x
                    v2 = y * width + (x + 1)
                    v3 = (y + 1) * width + (x + 1)
                    v4 = (y + 1) * width + x
                    if v1 in cutout_indices or v2 in cutout_indices or v3 in cutout_indices or v4 in cutout_indices:
                        continue
                    faces.append((v1, v2, v3, v4))
                    face_materials.append(materials[v1])
                    # Create UVs with base coordinates (scaling will be applied in the shader via material nodes)
                    uvs.extend([
                        (x, y),
                        ((x + 1), y),
                        ((x + 1), (y + 1)),
                        (x, (y + 1))
                    ])
            blend_logger.info(f"Created {len(faces)} faces for blending")

            # Build mesh
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            # Create UV layer
            uv_layer = mesh.uv_layers.new(name="ScaledUV")
            for i, poly in enumerate(mesh.polygons):
                for j, loop_idx in enumerate(poly.loop_indices):
                    if i * 4 + j < len(uvs):
                        uv_layer.data[loop_idx].uv = uvs[i * 4 + j]

            # Calculate blend weights
            blend_weights = H32HeightmapImporter.calculate_blend_weights(
                vertices, materials, width, blend_radius
            )
            # Apply vertex colors
            H32HeightmapImporter.apply_vertex_colors(mesh, blend_weights, width)
            # Create and assign blend material
            blend_material = H32HeightmapImporter.create_terrain_blend_material(
                surface_types, "TerrainBlend"
            )
            mesh.materials.append(blend_material)
            # Assign material to all faces (single material slot for blended shader)
            for poly in mesh.polygons:
                poly.material_index = 0

        else:
            # --- Legacy Path ---
            blend_logger.info("Using legacy per-face material assignment")
            # Create faces, UVs, and gather material info per face
            faces = []
            uvs = [] # This will be populated per-face now
            material_indices = [] # Per face material ID
            mat_to_faces = defaultdict(list) # For final assignment
            blend_logger.info("Creating faces and UV coordinates for legacy mode")
            for y in range(width - 1):
                for x in range(width - 1):
                    v1 = y * width + x
                    v2 = y * width + (x + 1)
                    v3 = (y + 1) * width + (x + 1)
                    v4 = (y + 1) * width + x
                    if v1 in cutout_indices or v2 in cutout_indices or v3 in cutout_indices or v4 in cutout_indices:
                        continue
                    face_idx = len(faces)
                    mat_id = materials[v1]
                    faces.append((v1, v2, v3, v4))
                    material_indices.append(mat_id)
                    mat_to_faces[mat_id].append(face_idx)

                    # UVs with scaling based on THIS face's material
                    scale_x = surface_types[mat_id][1] if mat_id < len(surface_types) else 1.0
                    scale_y = surface_types[mat_id][2] if mat_id < len(surface_types) else 1.0
                    # Apply scaling - larger scale means more tiling
                    u_x = x * scale_x * 4
                    u_y = y * scale_y * 4
                    u_x2 = (x + 1) * scale_x * 4
                    u_y2 = (y + 1) * scale_y * 4
                    uvs.extend([
                        (u_x, u_y),
                        (u_x2, u_y),
                        (u_x2, u_y2),
                        (u_x, u_y2)
                    ])
            blend_logger.info(f"Created {len(faces)} faces for legacy mode")

            # Build mesh
            mesh.from_pydata(vertices, [], faces)
            mesh.update()

            # Create UV layer
            uv_layer = mesh.uv_layers.new(name="ScaledUV")
            for i, poly in enumerate(mesh.polygons):
                for j, loop_idx in enumerate(poly.loop_indices):
                    if i * 4 + j < len(uvs):
                        uv_layer.data[loop_idx].uv = uvs[i * 4 + j]

            # Create materials (grouped by texture) - Legacy logic
            texture_materials = {}
            for mat_id, (tex_path, scale_x, scale_y) in enumerate(surface_types):
                if mat_id >= 64 or not tex_path:
                    continue
                if tex_path not in texture_materials:
                    mat = bpy.data.materials.new(f"mat_{mat_id}")
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    # Clear default nodes
                    for node in nodes:
                        nodes.remove(node)
                    # Create nodes (Legacy structure)
                    output = nodes.new('ShaderNodeOutputMaterial')
                    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                    tex_node = nodes.new('ShaderNodeTexImage')
                    uv_node = nodes.new('ShaderNodeUVMap')
                    # Set UV map
                    uv_node.uv_map = "ScaledUV"
                    bsdf.inputs['Roughness'].default_value = 1.0
                    bsdf.inputs['Specular'].default_value = 0.0
                    # Connect nodes
                    links.new(uv_node.outputs[0], tex_node.inputs[0])
                    links.new(tex_node.outputs[0], bsdf.inputs[0])
                    links.new(bsdf.outputs[0], output.inputs[0])
                    # Position nodes
                    uv_node.location = (-300, 0)
                    tex_node.location = (-100, 0)
                    bsdf.location = (100, 0)
                    output.location = (300, 0)
                    try:
                        # Load the image from the resolved path
                        tex_node.image = bpy.data.images.load(tex_path)
                        # Pack the image data into the .blend file
                        if tex_node.image:
                             tex_node.image.pack()
                             blend_logger.info(f"Packed texture: {tex_path}")
                    except Exception as e:
                         blend_logger.error(f"Failed to load or pack texture {tex_path}: {e}")
                    texture_materials[tex_path] = mat
            # Assign materials to mesh
            for tex_path, mat in texture_materials.items():
                mesh.materials.append(mat)
            # Assign material indices to polygons
            for mat_id, face_indices in mat_to_faces.items():
                if mat_id >= len(surface_types) or not surface_types[mat_id][0]:
                    continue
                tex_path = surface_types[mat_id][0]
                if tex_path not in texture_materials:
                    continue
                mat_index = list(texture_materials.keys()).index(tex_path)
                for face_idx in face_indices:
                    # Ensure face index is valid (should be, but good check)
                    if face_idx < len(mesh.polygons):
                        mesh.polygons[face_idx].material_index = mat_index

        # Enable smooth shading
        mesh.shade_smooth()
        # Select the object
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        # Set viewport shading
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
        blend_logger.info("Heightmap mesh loading completed successfully")

        # --- Water Plane Creation ---
        if import_water_plane:
            blend_logger.info("Creating water plane...")
            try:
                # Parse LevelInfo for water settings
                level_info = tree.find('LevelInfo')
                if level_info is not None:
                    heightmap_unit_size = float(level_info.get('HeightmapUnitSize', 2.0))
                    heightmap_x_size = float(level_info.get('HeightmapXSize', 1536.0))
                    # heightmap_y_size = float(level_info.get('HeightmapYSize', heightmap_x_size))
                    water_level_str = level_info.get('WaterLevel')
                    if water_level_str is not None:
                        water_level = float(water_level_str)
                        # Calculate plane dimensions and position
                        plane_size_x = heightmap_unit_size * heightmap_x_size
                        # plane_size_y = heightmap_unit_size * heightmap_y_size
                        plane_size_y = plane_size_x
                        center_x = plane_size_x / 2.0
                        center_y = plane_size_y / 2.0
                        # Create the plane
                        bpy.ops.mesh.primitive_plane_add(size=plane_size_x, enter_editmode=False, align='WORLD', location=(center_x, center_y, water_level))
                        water_obj = bpy.context.active_object
                        water_obj.name = "WaterPlane"
                        # Create water material
                        water_mat = bpy.data.materials.new(name="WaterMaterial")
                        water_mat.use_nodes = True
                        nodes = water_mat.node_tree.nodes
                        links = water_mat.node_tree.links
                        # Clear default nodes
                        for node in nodes:
                            nodes.remove(node)
                        # Create nodes for water shader
                        output_node = nodes.new(type='ShaderNodeOutputMaterial')
                        glossy_node = nodes.new(type='ShaderNodeBsdfGlossy')
                        bump_node = nodes.new(type='ShaderNodeBump')
                        noise_node = nodes.new(type='ShaderNodeTexNoise')
                        mapping_node = nodes.new(type='ShaderNodeMapping')
                        texcoord_node = nodes.new(type='ShaderNodeTexCoord')
                        # Set properties
                        glossy_node.inputs['Roughness'].default_value = 0.0
                        noise_node.inputs['Scale'].default_value = 108.0
                        noise_node.inputs['Detail'].default_value = 16.0
                        bump_node.inputs['Strength'].default_value = 1.0
                        # Position nodes
                        texcoord_node.location = (-800, 0)
                        mapping_node.location = (-600, 0)
                        noise_node.location = (-400, 0)
                        bump_node.location = (-200, 0)
                        glossy_node.location = (0, 0)
                        output_node.location = (200, 0)
                        # Connect nodes
                        links.new(texcoord_node.outputs['Generated'], mapping_node.inputs['Vector'])
                        links.new(mapping_node.outputs['Vector'], noise_node.inputs['Vector'])
                        links.new(noise_node.outputs['Color'], bump_node.inputs['Height'])
                        links.new(bump_node.outputs['Normal'], glossy_node.inputs['Normal'])
                        links.new(glossy_node.outputs['BSDF'], output_node.inputs['Surface'])
                        # Assign material
                        water_obj.data.materials.append(water_mat)
                        blend_logger.info(f"Water plane created: Size {plane_size_x}x{plane_size_y}, Z={water_level}")
                    else:
                        blend_logger.info("WaterLevel not found in LevelInfo, skipping water plane creation.")
                else:
                    blend_logger.warning("LevelInfo not found in leveldata.xml, skipping water plane creation.")
            except Exception as e:
                blend_logger.error(f"Failed to create water plane: {e}")
        blend_logger.info("Full heightmap import process completed successfully")
        return obj
        
# --- Internal Operators (Do the actual import using extracted temp files) ---
class AION_OT_ImportHeightmapInternal(Operator):
    """Load a heightmap from an H32 file (internal use)"""
    bl_idname = "aion.import_heightmap_internal"
    bl_label = "Import Heightmap (Internal)"
    bl_options = {'REGISTER', 'UNDO'}
    h32_path: StringProperty()
    leveldata_xml_path: StringProperty()
    enable_smooth_blending: BoolProperty(default=True)
    blend_radius: FloatProperty(default=2.0)
    import_water_plane: BoolProperty(default=True)

    def execute(self, context):
        if not all([self.h32_path, self.leveldata_xml_path]):
             self.report({'ERROR'}, "Missing required file paths for heightmap import.")
             return {'CANCELLED'}
        try:
            H32HeightmapImporter.load_h32_heightmap(
                self.h32_path,
                self.leveldata_xml_path,
                self.enable_smooth_blending,
                self.blend_radius,
                self.import_water_plane
            )
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

class AION_OT_RemoveNoDraw(Operator):
    """Remove all geometry with NoDraw materials"""
    bl_idname = "aion.remove_nodraw"
    bl_label = "Remove NoDraw Geometry"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        # Store the current active object and mode
        current_active = bpy.context.active_object
        current_mode = bpy.context.object.mode if current_active else None
        # Force object mode for all objects
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        # Iterate through all objects in the scene
        for obj in bpy.context.scene.objects:
            # Check if the object is a mesh
            if obj.type == 'MESH':
                # Get the mesh data
                mesh = obj.data
                # Check if the mesh has any materials
                if not mesh.materials:
                    continue
                # Create a list of material indices that contain "default(NoDraw)"
                nodraw_material_indices = []
                for i, mat in enumerate(mesh.materials):
                    if mat and "(NoDraw)" in mat.name:
                        nodraw_material_indices.append(i)
                # If no NoDraw materials found, skip this object
                if not nodraw_material_indices:
                    continue
                # Make sure the object is selected and active (required for some operations)
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                # Create a bmesh from the mesh
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bm.faces.ensure_lookup_table()
                # Collect faces to remove
                faces_to_remove = []
                for face in bm.faces:
                    if face.material_index in nodraw_material_indices:
                        faces_to_remove.append(face)
                # Remove the faces
                if faces_to_remove:
                    bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
                    # Remove any now-orphaned vertices
                    verts_to_remove = [v for v in bm.verts if not v.link_faces]
                    if verts_to_remove:
                        bmesh.ops.delete(bm, geom=verts_to_remove, context='VERTS')
                # Update the original mesh
                bm.to_mesh(mesh)
                bm.free()
                # Deselect the object
                obj.select_set(False)
        # Restore the original active object and mode
        if current_active:
            bpy.context.view_layer.objects.active = current_active
            if current_mode and current_mode != 'OBJECT':
                try:
                    bpy.ops.object.mode_set(mode=current_mode)
                except:
                    pass
        self.report({'INFO'}, "Removed NoDraw geometry")
        return {'FINISHED'}

class AION_OT_ImportVegetationInternal(Operator):
    """Import vegetation from objects.lst (internal use)"""
    bl_idname = "aion.import_vegetation_internal"
    bl_label = "Import Vegetation (Internal)"
    bl_options = {'REGISTER', 'UNDO'}
    lst_path: StringProperty() # objects.lst path
    leveldata_xml_path: StringProperty()
    def execute(self, context):
        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        if not all([self.lst_path, self.leveldata_xml_path]):
             self.report({'ERROR'}, "Missing required file paths for vegetation import.")
             return {'CANCELLED'}
        def get_or_create_collection(name, parent=None):
            """Get or create a collection with the given name"""
            if name in bpy.data.collections:
                return bpy.data.collections[name]
            new_col = bpy.data.collections.new(name)
            if parent:
                parent.children.link(new_col)
            else:
                bpy.context.scene.collection.children.link(new_col)
            return new_col
        def parse_leveldata_xml(xml_path):
            """Parse leveldata.xml and return a dictionary mapping object IDs to CGF paths and map size"""
            tree = ET.parse(xml_path)
            root = tree.getroot()
            object_map = {}
            map_size = 1536
            # Get map size from LevelInfo
            level_info = root.find('LevelInfo')
            if level_info is not None:
                map_size = int(level_info.get('HeightmapXSize', 1536))
            vegetation = root.find('Vegetation')
            if vegetation is not None:
                for i, obj in enumerate(vegetation.findall('Object')):
                    cgf_path = obj.get('FileName', '').replace('\\', '/')
                    short_name = cgf_path[-57:] if len(cgf_path) > 57 else cgf_path
                    object_map[i] = {
                        'path': cgf_path,
                        'short_name': short_name
                    }
            return object_map, map_size
        def parse_objects_lst(lst_path, map_size, object_map):
            """Parse objects.lst and return a list of VegetationObject"""
            items = []
            with open(lst_path, 'rb') as f:
                header = struct.unpack('<I', f.read(4))[0]
                if header != 0x10:
                    raise ValueError("objects.lst: expected 0x10 header")
                magic = 32768.0 / map_size
                while True:
                    data = f.read(16)
                    if not data or len(data) < 16:
                        break
                    x_pos, y_pos, z_pos, object_id, unk123, scale, heading = struct.unpack('<HHHBBfI', data)
                    item = VegetationObject()
                    item.position = Vector((
                        x_pos / magic,
                        y_pos / magic,
                        z_pos / magic
                    ))
                    item.object_id = object_id
                    item.scale = scale
                    item.heading = heading * 360 / 255
                    if object_id in object_map:
                        item.cgf_path = object_map[object_id]['path']
                        item.short_name = object_map[object_id]['short_name']
                    items.append(item)
            return items
        def load_vegetation_models(object_items):
            """Vegetation loading with instancing"""
            # Create collections
            main_col = get_or_create_collection("Vegetation")
            mesh_col = get_or_create_collection("Vegetation_Meshes", main_col)
            instances_col = get_or_create_collection("Vegetation_Instances", main_col)
            # Dictionary to track loaded meshes: {cgf_path: (mesh_data, count)}
            mesh_library = {}
            # First pass: Load all unique meshes
            unique_paths = {obj.cgf_path for obj in object_items if obj.cgf_path}
            for path in unique_paths:
                cgf_path = os.path.join(prefs.models_root, path)
                if not os.path.exists(cgf_path):
                    self.report({'WARNING'}, f"Model not found: {cgf_path}")
                    continue
                try:
                    # Store objects before import
                    objects_before = set(bpy.data.objects)
                    # Import the CGF file
                    bpy.ops.import_scene.cgf(
                        filepath=cgf_path,
                        convert_dds_to_png=False,
                        reuse_materials=False,
                        reuse_images=True,
                        import_skeleton=False,
                        skeleton_auto_connect=False,
                        import_animations=False
                    )
                    # Get imported objects
                    imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
                    if not imported_objects:
                        self.report({'WARNING'}, f"No objects imported from {cgf_path}")
                        continue
                    # Process imported objects
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.hide_set(False)
                        obj.hide_viewport = False
                        obj.hide_render = False
                        obj.select_set(True)
                    # Handle multiple meshes by joining them
                    if len(imported_objects) > 1:
                        first_obj_matrix = imported_objects[0].matrix_world.copy()
                        # Apply transforms to mesh data
                        for obj in imported_objects:
                            obj.data.transform(obj.matrix_world)
                            obj.matrix_world.identity()
                        # Join meshes
                        bpy.context.view_layer.objects.active = imported_objects[0]
                        bpy.ops.object.join()
                        imported_objects = [bpy.context.active_object]
                        imported_objects[0].matrix_world = first_obj_matrix
                    # Prepare the mesh object
                    imported_obj = imported_objects[0]
                    short_name = path[-57:] if len(path) > 57 else path
                    imported_obj.name = short_name
                    imported_obj.data.name = short_name
                    # Move to mesh collection and hide
                    for col in imported_obj.users_collection:
                        col.objects.unlink(imported_obj)
                    mesh_col.objects.link(imported_obj)
                    imported_obj.hide_viewport = True
                    imported_obj.hide_render = True
                    # Store in library
                    mesh_library[path] = imported_obj.data
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to load {cgf_path}: {str(e)}")
                    # Clean up any partially imported objects
                    for obj in bpy.data.objects:
                        if obj not in objects_before:
                            bpy.data.objects.remove(obj, do_unlink=True)
                    continue
            # Second pass: Create instances
            for obj in object_items:
                if not obj.cgf_path or obj.cgf_path not in mesh_library:
                    continue
                mesh_data = mesh_library[obj.cgf_path]
                new_obj = bpy.data.objects.new(obj.short_name, mesh_data)
                # Set transform
                new_obj.location = obj.position
                new_obj.rotation_euler = Euler((0, 0, radians(obj.heading)), 'XYZ')
                new_obj.scale = (obj.scale, obj.scale, obj.scale)
                instances_col.objects.link(new_obj)
            # Report statistics
            unique_count = len(mesh_library)
            instance_count = sum(1 for obj in object_items if obj.cgf_path and obj.cgf_path in mesh_library)
            self.report({'INFO'}, f"Created {instance_count} instances of {unique_count} unique vegetation models")
        # Parse leveldata.xml (now in temp dir)
        try:
            object_map, map_size = parse_leveldata_xml(self.leveldata_xml_path)
            self.report({'INFO'}, f"Using map size from leveldata.xml: {map_size}")
        except Exception as e:
            self.report({'ERROR'}, f"Error parsing leveldata.xml: {str(e)}")
            return {'CANCELLED'}
        # Parse objects.lst
        try:
            vegetation_items = parse_objects_lst(self.lst_path, map_size, object_map)
        except Exception as e:
            self.report({'ERROR'}, f"Error parsing objects.lst: {str(e)}")
            return {'CANCELLED'}
        # Load vegetation models with instancing
        load_vegetation_models(vegetation_items)
        # Remove NoDraw geometry if enabled
        if props.remove_nodraw:
            bpy.ops.aion.remove_nodraw()
        return {'FINISHED'}

class AION_OT_ImportBrushesInternal(Operator):
    """Import brushes from brush.lst (internal use)"""
    bl_idname = "aion.import_brushes_internal"
    bl_label = "Import Brushes (Internal)"
    bl_options = {'REGISTER', 'UNDO'}
    lst_path: StringProperty()
    leveldata_xml_path: StringProperty()
    def execute(self, context):
        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        if not self.lst_path:
             self.report({'ERROR'}, "Missing brush.lst path for brush import.")
             return {'CANCELLED'}
        def get_or_create_collection(name, parent=None):
            """Get or create a collection with the given name"""
            if name in bpy.data.collections:
                return bpy.data.collections[name]
            new_col = bpy.data.collections.new(name)
            if parent:
                parent.children.link(new_col)
            else:
                bpy.context.scene.collection.children.link(new_col)
            return new_col
        def parse_brush_lst(filepath):
            """Parse brush.lst and return brush info and entries"""
            brush_info = []
            entries = []
            with open(filepath, 'rb') as f:
                # Read signature
                signature = f.read(3)
                if signature != b'CRY':
                    raise ValueError("Wrong signature")
                f.read(4)  # skip dw1
                meshDataBlockSz = struct.unpack('<i', f.read(4))[0]
                if meshDataBlockSz < 16 or meshDataBlockSz > 19:
                    raise ValueError("Unexpected block size")
                titlesCount = struct.unpack('<i', f.read(4))[0]
                # Read titles
                for _ in range(titlesCount):
                    nameLen = struct.unpack('<i', f.read(4))[0]
                    f.read(nameLen - 4)
                # Read mesh info
                meshInfoCount = struct.unpack('<i', f.read(4))[0]
                for i in range(meshInfoCount):
                    f.read(4)  # skip
                    filename_bytes = f.read(128)
                    filename = filename_bytes.decode('utf-8').strip('\x00').lower().replace('\\', '/')
                    f.read(4)  # skip
                    # Read bounding box
                    x1, y1, z1, x2, y2, z2 = struct.unpack('<6f', f.read(24))
                    brush_info.append({
                        'filename': filename,
                        'bbox': ((x1, y1, z1), (x2, y2, z2))
                    })
                # Read mesh data
                meshDataCount = struct.unpack('<i', f.read(4))[0]
                for i in range(meshDataCount):
                    f.read(4)  # skip
                    f.read(4)  # skip
                    meshIdx = struct.unpack('<i', f.read(4))[0]
                    if meshIdx < 0 or meshIdx >= len(brush_info):
                        print(f"Invalid mesh index {meshIdx}, skipping")
                        continue
                    f.read(4)  # skip
                    f.read(4)  # skip
                    f.read(4)  # skip
                    # Read 3x4 matrix
                    matrix = struct.unpack('<12f', f.read(48))
                    # Rotation matrix
                    rot_matrix = Matrix((
                        (matrix[0 * 4 + 0], matrix[0 * 4 + 1], matrix[0 * 4 + 2], matrix[0 * 4 + 3]),
                        (matrix[1 * 4 + 0], matrix[1 * 4 + 1], matrix[1 * 4 + 2], matrix[1 * 4 + 3]),
                        (matrix[2 * 4 + 0], matrix[2 * 4 + 1], matrix[2 * 4 + 2], matrix[2 * 4 + 3]),
                        (0, 0, 0, 1)
                    ))
                    f.read(1)  # skip
                    f.read(1)  # skip
                    f.read(1)  # skip
                    f.read(1)  # skip
                    f.read(4)  # skip
                    f.read(4)  # skip
                    f.read(4)  # skip
                    eventType = struct.unpack('<i', f.read(4))[0]
                    if eventType < 0 or eventType > 4:
                        print(f"Ignoring unknown event: {eventType}")
                    f.read(4)  # skip (always 3?)
                    # Read remaining bytes based on meshDataBlockSz
                    f.read(4 * (meshDataBlockSz - 16))
                    # Get the filename and create short name
                    filename = brush_info[meshIdx]['filename']
                    short_name = filename[-57:] if len(filename) > 57 else filename
                    entries.append({
                        'meshIdx': meshIdx,
                        'rotationMatrix': rot_matrix,
                        'eventType': eventType,
                        'filename': filename,
                        'short_name': short_name
                    })
            return brush_info, entries
        def load_brush_models(brush_info, entries):
            """Load brush models with instancing"""
            # Create collections
            main_col = get_or_create_collection("Brushes")
            mesh_col = get_or_create_collection("Brush_Meshes", main_col)
            instances_col = get_or_create_collection("Brush_Instances", main_col)
            # Dictionary to track loaded meshes: {filename: mesh_data}
            mesh_library = {}
            stats = {'success': 0, 'missing': 0, 'failed': 0}
            # First pass: Load all unique meshes
            unique_files = {brush_info[entry['meshIdx']]['filename'] for entry in entries}
            for filename in unique_files:
                cgf_path = os.path.join(prefs.models_root, filename)
                if not os.path.exists(cgf_path):
                    self.report({'WARNING'}, f"Model not found: {cgf_path}")
                    stats['missing'] += 1
                    continue
                try:
                    # Store objects before import
                    objects_before = set(bpy.data.objects)
                    # Import the CGF file
                    bpy.ops.import_scene.cgf(
                        filepath=cgf_path,
                        convert_dds_to_png=False,
                        reuse_materials=False,
                        reuse_images=True,
                        import_skeleton=False,
                        skeleton_auto_connect=False,
                        import_animations=False
                    )
                    # Get imported objects
                    imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
                    if not imported_objects:
                        self.report({'WARNING'}, f"No objects imported from {cgf_path}")
                        stats['failed'] += 1
                        continue
                    # Process imported objects
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.hide_set(False)
                        obj.hide_viewport = False
                        obj.hide_render = False
                        obj.select_set(True)
                    temp=False
                    # Handle multiple meshes by joining them
                    if len(imported_objects) > 1:
                        temp=True
                        first_obj_matrix = imported_objects[0].matrix_world.copy()
                        # Apply transforms to mesh data
                        for obj in imported_objects:
                            obj.data.transform(obj.matrix_world)
                            obj.matrix_world.identity()
                        # Join meshes
                        bpy.context.view_layer.objects.active = imported_objects[0]
                        bpy.ops.object.join()
                        imported_objects = [bpy.context.active_object]
                        imported_objects[0].matrix_world = first_obj_matrix
                    # Prepare the mesh object
                    imported_obj = imported_objects[0]
                    if (temp==False):
                        imported_obj.data.transform(imported_obj.matrix_world)
                    short_name = os.path.basename(filename)
                    imported_obj.name = short_name
                    imported_obj.data.name = short_name
                    # Move to mesh collection and hide
                    for col in imported_obj.users_collection:
                        col.objects.unlink(imported_obj)
                    mesh_col.objects.link(imported_obj)
                    imported_obj.hide_viewport = True
                    imported_obj.hide_render = True
                    # Store in library
                    mesh_library[filename] = imported_obj.data
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to load {cgf_path}: {str(e)}")
                    stats['failed'] += 1
                    # Clean up any partially imported objects
                    for obj in bpy.data.objects:
                        if obj not in objects_before:
                            bpy.data.objects.remove(obj, do_unlink=True)
                    continue
            # Second pass: Create instances
            for entry in entries:
                filename = brush_info[entry['meshIdx']]['filename']
                if filename not in mesh_library:
                    continue
                if entry['eventType'] > 0:
                    continue # Silently skip for now. Event meshes (Christmas, Halloween and such)
                mesh_data = mesh_library[filename]
                new_obj = bpy.data.objects.new(entry['short_name'], mesh_data)
                new_obj.matrix_world = entry['rotationMatrix']
                instances_col.objects.link(new_obj)
                stats['success'] += 1
            # Report statistics
            unique_count = len(mesh_library)
            self.report({'INFO'}, f"Successfully placed {stats['success']} instances of {unique_count} unique brush models")
        # Parse brush.lst
        try:
            brush_info, entries = parse_brush_lst(self.lst_path)
        except Exception as e:
            self.report({'ERROR'}, f"Error parsing brush.lst: {str(e)}")
            return {'CANCELLED'}
        # Load brush models with instancing
        load_brush_models(brush_info, entries)
        # Remove NoDraw geometry if enabled
        if props.remove_nodraw:
            bpy.ops.aion.remove_nodraw()
        return {'FINISHED'}

class AION_OT_ImportMissionObjectsInternal(Operator):
    """Import mission objects from mission XML (internal use)"""
    bl_idname = "aion.import_mission_objects_internal"
    bl_label = "Import Mission Objects (Internal)"
    bl_options = {'REGISTER', 'UNDO'}
    xml_path: StringProperty()
    def execute(self, context):
        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        if not self.xml_path:
             self.report({'ERROR'}, "Missing mission XML path for mission object import.")
             return {'CANCELLED'}
        def get_or_create_collection(name, parent=None):
            """Get or create a collection with the given name"""
            if name in bpy.data.collections:
                return bpy.data.collections[name]
            new_col = bpy.data.collections.new(name)
            if parent:
                parent.children.link(new_col)
            else:
                bpy.context.scene.collection.children.link(new_col)
            return new_col
        def parse_mission_xml(xml_path):
            """Parse mission XML and return a list of MissionObjects with model paths"""
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = []
            # Find all Entity elements in Objects section
            objects_section = root.find('Objects')
            if objects_section is None:
                return objects
            for entity in objects_section.findall('Entity'):
                # Check for model paths in Properties
                properties = entity.find('Properties')
                if properties is None:
                    continue
                # Check both possible model path attributes
                model_path = properties.get('fileLadderCGF', '') or properties.get('object_Model', '')
                if not model_path:
                    continue
                # Create new mission object
                obj = MissionObject()
                obj.name = entity.get('Name', '')
                obj.model_path = model_path.replace('\\', '/')
                # Parse position (format: "x,y,z")
                pos_str = entity.get('Pos', '0,0,0')
                try:
                    x, y, z = map(float, pos_str.split(','))
                    obj.position = Vector((x, y, z))
                except:
                    self.report({'WARNING'}, f"Invalid position for {obj.name}: {pos_str}")
                    continue
                # Parse rotation (format: "x,y,z" in degrees)
                angles_str = entity.get('Angles', '0,0,0')
                try:
                    x_rot, y_rot, z_rot = map(float, angles_str.split(','))
                    # Convert to radians and create Euler rotation
                    obj.rotation = Euler((
                        radians(x_rot),
                        radians(y_rot),
                        radians(z_rot)
                    ), 'XYZ')
                except:
                    self.report({'WARNING'}, f"Invalid rotation for {obj.name}: {angles_str}")
                    continue
                # Parse scale (can be single value or "x,y,z")
                scale_str = entity.get('Scale', '1')
                try:
                    if ',' in scale_str:
                        x_scale, y_scale, z_scale = map(float, scale_str.split(','))
                        obj.scale = Vector((x_scale, y_scale, z_scale))
                    else:
                        scale = float(scale_str)
                        obj.scale = Vector((scale, scale, scale))
                except:
                    self.report({'WARNING'}, f"Invalid scale for {obj.name}: {scale_str}")
                    continue
                objects.append(obj)
            return objects
        def load_mission_objects(mission_objects):
            """Load mission objects with instancing"""
            # Create collections
            main_col = get_or_create_collection("MissionObjects")
            mesh_col = get_or_create_collection("Mission_Meshes", main_col)
            instances_col = get_or_create_collection("Mission_Instances", main_col)
            # Dictionary to track loaded meshes: {model_path: mesh_data}
            mesh_library = {}
            stats = {'success': 0, 'missing': 0, 'failed': 0}
            # First pass: Load all unique meshes
            unique_paths = {obj.model_path for obj in mission_objects}
            for path in unique_paths:
                cgf_path = os.path.join(prefs.models_root, path)
                if not os.path.exists(cgf_path):
                    self.report({'WARNING'}, f"Model not found: {cgf_path}")
                    stats['missing'] += 1
                    continue
                try:
                    # Store objects before import
                    objects_before = set(bpy.data.objects)
                    # Import the CGF file (without animations)
                    bpy.ops.import_scene.cgf(
                        filepath=cgf_path,
                        convert_dds_to_png=False,
                        reuse_materials=False,
                        reuse_images=True,
                        import_skeleton=False,
                        skeleton_auto_connect=False,
                        import_animations=False
                    )
                    # Get imported objects
                    imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
                    if not imported_objects:
                        self.report({'WARNING'}, f"No objects imported from {cgf_path}")
                        stats['failed'] += 1
                        continue
                    # Process imported objects
                    bpy.ops.object.select_all(action='DESELECT')
                    for obj in imported_objects:
                        obj.hide_set(False)
                        obj.hide_viewport = False
                        obj.hide_render = False
                        obj.select_set(True)
                    # Handle multiple meshes by joining them
                    if len(imported_objects) > 1:
                        first_obj_matrix = imported_objects[0].matrix_world.copy()
                        # Apply transforms to mesh data
                        for obj in imported_objects:
                            obj.data.transform(obj.matrix_world)
                            obj.matrix_world.identity()
                        # Join meshes
                        bpy.context.view_layer.objects.active = imported_objects[0]
                        bpy.ops.object.join()
                        imported_objects = [bpy.context.active_object]
                        imported_objects[0].matrix_world = first_obj_matrix
                    # Prepare the mesh object
                    imported_obj = imported_objects[0]
                    short_name = os.path.basename(path)
                    imported_obj.name = short_name
                    imported_obj.data.name = short_name
                    # Move to mesh collection and hide
                    for col in imported_obj.users_collection:
                        col.objects.unlink(imported_obj)
                    mesh_col.objects.link(imported_obj)
                    imported_obj.hide_viewport = True
                    imported_obj.hide_render = True
                    # Store in library
                    mesh_library[path] = imported_obj.data
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to load {cgf_path}: {str(e)}")
                    stats['failed'] += 1
                    # Clean up any partially imported objects
                    for obj in bpy.data.objects:
                        if obj not in objects_before:
                            bpy.data.objects.remove(obj, do_unlink=True)
                    continue
            # Second pass: Create instances
            for obj in mission_objects:
                if obj.model_path not in mesh_library:
                    continue
                mesh_data = mesh_library[obj.model_path]
                new_obj = bpy.data.objects.new(obj.name, mesh_data)
                # Set transform
                new_obj.location = obj.position
                new_obj.rotation_euler = obj.rotation
                new_obj.scale = obj.scale
                instances_col.objects.link(new_obj)
                stats['success'] += 1
            # Report statistics
            unique_count = len(mesh_library)
            self.report({'INFO'}, f"Successfully placed {stats['success']} instances of {unique_count} unique mission objects")
        # Parse mission XML
        try:
            mission_objects = parse_mission_xml(self.xml_path)
        except Exception as e:
            self.report({'ERROR'}, f"Error parsing mission XML: {str(e)}")
            return {'CANCELLED'}
        if not mission_objects:
            self.report({'WARNING'}, "No objects with valid model paths found")
            return {'CANCELLED'}
        # Load mission objects with instancing
        load_mission_objects(mission_objects)
        # Remove NoDraw geometry if enabled
        if props.remove_nodraw:
            bpy.ops.aion.remove_nodraw()
        return {'FINISHED'}

class AION_OT_ImportHeightmap(Operator, ImportHelper):
    """Load a heightmap from an H32 file (select .h32 or folder containing .pak)"""
    bl_idname = "aion.import_heightmap"
    bl_label = "Import Heightmap"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    enable_smooth_blending: BoolProperty(
        name="Material Blending",
        description="Enable smooth transitions between materials",
        default=True
    )
    blend_radius: FloatProperty(
        name="Blend Radius",
        description="Radius for material blending in grid units",
        default=2.0,
        min=0.5,
        max=10.0
    )
    import_water_plane: BoolProperty(
        name="Import Water Plane",
        description="Import a water plane based on leveldata.xml settings",
        default=True
    )

    def execute(self, context):
        selected_path = self.filepath
        folder_path = selected_path if os.path.isdir(selected_path) else os.path.dirname(selected_path)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Selected path is not valid.")
            return {'CANCELLED'}
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.report({'INFO'}, f"Extracting all .pak files to temporary directory: {temp_dir}")
                extract_all_paks_to_temp(folder_path, temp_dir)
                # Find required files in the temp directory structure
                h32_path = os.path.join(temp_dir, "terrain", "land_map.h32")
                leveldata_xml_path = os.path.join(temp_dir, "leveldata.xml")
                if not os.path.exists(h32_path) or not os.path.exists(leveldata_xml_path):
                    missing = []
                    if not os.path.exists(h32_path): missing.append("terrain/land_map.h32")
                    if not os.path.exists(leveldata_xml_path): missing.append("leveldata.xml")
                    self.report({'ERROR'}, f"Missing essential files after extraction: {', '.join(missing)}")
                    return {'CANCELLED'}
                # Call internal operator with blending and water options
                result = bpy.ops.aion.import_heightmap_internal(
                    'EXEC_DEFAULT',
                    h32_path=h32_path,
                    leveldata_xml_path=leveldata_xml_path,
                    enable_smooth_blending=self.enable_smooth_blending, # Pass from this operator
                    blend_radius=self.blend_radius,                   # Pass from this operator
                    import_water_plane=self.import_water_plane      # Pass from this operator
                )
                if 'FINISHED' in result:
                    return {'FINISHED'}
                else:
                    return {'CANCELLED'}
            except Exception as e:
                self.report({'ERROR'}, f"Heightmap import failed: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

class AION_OT_ImportVegetation(Operator, ImportHelper):
    """Import vegetation from objects.lst"""
    bl_idname = "aion.import_vegetation"
    bl_label = "Import Vegetation"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})
    def execute(self, context):
        selected_path = self.filepath
        folder_path = selected_path if os.path.isdir(selected_path) else os.path.dirname(selected_path)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Selected path is not valid.")
            return {'CANCELLED'}
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.report({'INFO'}, f"Extracting all .pak files to temporary directory: {temp_dir}")
                extract_all_paks_to_temp(folder_path, temp_dir)
                lst_path = os.path.join(temp_dir, "objects.lst")
                leveldata_xml_path = os.path.join(temp_dir, "leveldata.xml")
                if not os.path.exists(lst_path) or not os.path.exists(leveldata_xml_path):
                    missing = []
                    if not os.path.exists(lst_path): missing.append("objects.lst")
                    if not os.path.exists(leveldata_xml_path): missing.append("leveldata.xml")
                    self.report({'ERROR'}, f"Missing essential files after extraction: {', '.join(missing)}")
                    return {'CANCELLED'}
                # Call internal operator
                result = bpy.ops.aion.import_vegetation_internal(
                    'EXEC_DEFAULT',
                    lst_path=lst_path,
                    leveldata_xml_path=leveldata_xml_path
                )
                if 'FINISHED' in result:
                    return {'FINISHED'}
                else:
                    return {'CANCELLED'}
            except Exception as e:
                self.report({'ERROR'}, f"Vegetation import failed: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

class AION_OT_ImportBrushes(Operator, ImportHelper):
    """Import brushes from brush.lst"""
    bl_idname = "aion.import_brushes"
    bl_label = "Import Brushes"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})
    def execute(self, context):
        selected_path = self.filepath
        folder_path = selected_path if os.path.isdir(selected_path) else os.path.dirname(selected_path)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Selected path is not valid.")
            return {'CANCELLED'}
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.report({'INFO'}, f"Extracting all .pak files to temporary directory: {temp_dir}")
                extract_all_paks_to_temp(folder_path, temp_dir)
                lst_path = os.path.join(temp_dir, "brush.lst")
                leveldata_xml_path = os.path.join(temp_dir, "leveldata.xml") # Pass for consistency
                if not os.path.exists(lst_path):
                    self.report({'ERROR'}, "Missing brush.lst after extraction")
                    return {'CANCELLED'}
                # Call internal operator
                result = bpy.ops.aion.import_brushes_internal(
                    'EXEC_DEFAULT',
                    lst_path=lst_path,
                    leveldata_xml_path=leveldata_xml_path # Pass if needed
                )
                if 'FINISHED' in result:
                    return {'FINISHED'}
                else:
                    return {'CANCELLED'}
            except Exception as e:
                self.report({'ERROR'}, f"Brushes import failed: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

class AION_OT_ImportMissionObjects(Operator, ImportHelper):
    """Import mission objects from mission XML"""
    bl_idname = "aion.import_mission_objects"
    bl_label = "Import Mission Objects"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})
    def execute(self, context):
        selected_path = self.filepath
        folder_path = selected_path if os.path.isdir(selected_path) else os.path.dirname(selected_path)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Selected path is not valid.")
            return {'CANCELLED'}
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.report({'INFO'}, f"Extracting all .pak files to temporary directory: {temp_dir}")
                extract_all_paks_to_temp(folder_path, temp_dir)
                # Find mission XML file (look for one starting with "mission_")
                mission_xml_path = None
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().startswith("mission_") and file.lower().endswith(".xml"):
                            mission_xml_path = os.path.join(root, file)
                            break
                    if mission_xml_path:
                        break
                if not mission_xml_path:
                    self.report({'ERROR'}, "Missing mission XML file after extraction.")
                    return {'CANCELLED'}
                # Call internal operator
                result = bpy.ops.aion.import_mission_objects_internal(
                    'EXEC_DEFAULT',
                    xml_path=mission_xml_path
                )
                if 'FINISHED' in result:
                    return {'FINISHED'}
                else:
                    return {'CANCELLED'}
            except Exception as e:
                self.report({'ERROR'}, f"Mission Objects import failed: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

# --- Folder Selection Operator for Full Map Import ---
class AION_OT_ImportMapFolder(Operator, ImportHelper):
    """Select the folder containing the map's .pak files to import the full map"""
    bl_idname = "aion.import_map_folder"
    bl_label = "Import Full Map from Folder"
    bl_options = {'REGISTER', 'UNDO'}
    # Configure ImportHelper to select folders
    use_filter_folder = True
    filter_glob: StringProperty(default="", options={'HIDDEN'})
    def execute(self, context):
        folder_path = self.filepath
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Selected path is not a directory.")
            return {'CANCELLED'}
        # Create a temporary directory for extracted files
        with tempfile.TemporaryDirectory() as temp_dir:
            self.report({'INFO'}, f"Extracting all .pak files to temporary directory: {temp_dir}")
            try:
                # --- 1. Extract all relevant .pak files ---
                extract_all_paks_to_temp(folder_path, temp_dir)
                # --- 2. Find required files in the temp directory structure ---
                h32_path = os.path.join(temp_dir, "terrain", "land_map.h32")
                leveldata_xml_path = os.path.join(temp_dir, "leveldata.xml")
                objects_lst_path = os.path.join(temp_dir, "objects.lst")
                brush_lst_path = os.path.join(temp_dir, "brush.lst")
                # Mission Object file (find one starting with "mission_")
                mission_xml_path = None
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().startswith("mission_") and file.lower().endswith(".xml"):
                            mission_xml_path = os.path.join(root, file)
                            break
                    if mission_xml_path:
                        break
                # --- 3. Validate essential files ---
                missing_files = []
                if not os.path.exists(h32_path):
                    missing_files.append("terrain/land_map.h32")
                if not os.path.exists(leveldata_xml_path):
                    missing_files.append("leveldata.xml")
                if missing_files:
                    error_msg = f"Missing essential files after extraction: {', '.join(missing_files)}"
                    self.report({'ERROR'}, error_msg)
                    return {'CANCELLED'}

                # --- Get Properties from Scene ---
                props = context.scene.aion_importer_props
                enable_smooth_blending = props.enable_smooth_blending
                blend_radius = props.blend_radius
                import_water_plane = props.import_water_plane
                # --- 4. Call individual import operators ---
                # Import Heightmap
                if os.path.exists(h32_path) and os.path.exists(leveldata_xml_path):
                    try:
                        result = bpy.ops.aion.import_heightmap_internal(
                            'EXEC_DEFAULT',
                            h32_path=h32_path,
                            leveldata_xml_path=leveldata_xml_path,
                            enable_smooth_blending=enable_smooth_blending, # Pass blending options
                            blend_radius=blend_radius,                   # Pass blending options
                            import_water_plane=import_water_plane      # Pass water option
                        )
                        # Check result if needed, but don't stop on failure
                        if 'FINISHED' not in result:
                             self.report({'WARNING'}, "Heightmap import reported issue.")
                    except Exception as e:
                         self.report({'ERROR'}, f"Heightmap import failed: {e}")
                         # Continue to next step

                # Import Vegetation
                if os.path.exists(objects_lst_path) and os.path.exists(leveldata_xml_path):
                    try:
                        result = bpy.ops.aion.import_vegetation_internal(
                            'EXEC_DEFAULT',
                            lst_path=objects_lst_path,
                            leveldata_xml_path=leveldata_xml_path
                        )
                        if 'FINISHED' not in result:
                             self.report({'WARNING'}, "Vegetation import reported issue.")
                    except Exception as e:
                         self.report({'ERROR'}, f"Vegetation import failed: {e}")
                         # Continue to next step

                # Import Brushes
                if os.path.exists(brush_lst_path):
                    try:
                        result = bpy.ops.aion.import_brushes_internal(
                            'EXEC_DEFAULT',
                            lst_path=brush_lst_path,
                            leveldata_xml_path=leveldata_xml_path # Pass for consistency
                        )
                        if 'FINISHED' not in result:
                             self.report({'WARNING'}, "Brushes import reported issue.")
                    except Exception as e:
                         self.report({'ERROR'}, f"Brushes import failed: {e}")
                         # Continue to next step

                # Import Mission Objects
                if mission_xml_path and os.path.exists(mission_xml_path):
                    try:
                        result = bpy.ops.aion.import_mission_objects_internal(
                            'EXEC_DEFAULT',
                            xml_path=mission_xml_path
                        )
                        if 'FINISHED' not in result:
                             self.report({'WARNING'}, "Mission Objects import reported issue.")
                    except Exception as e:
                         self.report({'ERROR'}, f"Mission Objects import failed: {e}")
                self.report({'INFO'}, "Map import process completed (check for warnings/errors above).")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"An error occurred during import: {e}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

class AION_PT_ImporterPanel(Panel):
    """Creates a Panel in the 3D Viewport"""
    bl_label = "Aion Map Importer"
    bl_idname = "AION_PT_importer_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Aion Importer"
    def draw(self, context):
        layout = self.layout
        props = context.scene.aion_importer_props
        box = layout.box()
        box.label(text="Import Full Map", icon='FOLDER_REDIRECT')
        box.prop(props, "import_water_plane")
        box.prop(props, "enable_smooth_blending")
        if props.enable_smooth_blending:
            box.prop(props, "blend_radius")
        box.operator(AION_OT_ImportMapFolder.bl_idname)
        # Heightmap section
        box = layout.box()
        box.label(text="Heightmap Import", icon='MOD_LATTICE')
        box.prop(props, "import_water_plane")
        box.prop(props, "enable_smooth_blending")
        if props.enable_smooth_blending:
            row = box.row()
            row.prop(props, "blend_radius")
        op = box.operator(AION_OT_ImportHeightmap.bl_idname)
        # Pass properties to the operator (they are defined there too)
        op.import_water_plane = props.import_water_plane
        op.enable_smooth_blending = props.enable_smooth_blending
        op.blend_radius = props.blend_radius
        # Brush section
        box = layout.box()
        box.label(text="Brush Import", icon='BRUSH_DATA')
        box.prop(props, "remove_nodraw")
        box.operator(AION_OT_ImportBrushes.bl_idname)
        # Vegetation/Object section
        box = layout.box()
        box.label(text="Vegetation Import", icon='IMAGE')
        box.prop(props, "remove_nodraw")
        box.operator(AION_OT_ImportVegetation.bl_idname)
        # Mission Objects section
        box = layout.box()
        box.label(text="Mission object Import", icon='OBJECT_DATA')
        box.prop(props, "remove_nodraw")
        box.operator(AION_OT_ImportMissionObjects.bl_idname)
        # Utility section
        box = layout.box()
        box.label(text="Utilities", icon='TOOL_SETTINGS')
        box.operator(AION_OT_RemoveNoDraw.bl_idname)

classes = (
    AionImporterPreferences,
    AionImporterProperties,
    # Internal operators (do the core import logic)
    AION_OT_ImportHeightmapInternal,
    AION_OT_RemoveNoDraw,
    AION_OT_ImportVegetationInternal,
    AION_OT_ImportBrushesInternal,
    AION_OT_ImportMissionObjectsInternal,
    # ImportHelper operators (select file/folder, extract, call internal)
    AION_OT_ImportHeightmap,
    AION_OT_ImportVegetation,
    AION_OT_ImportBrushes,
    AION_OT_ImportMissionObjects,
    # Folder selector for full map
    AION_OT_ImportMapFolder,
    # Updated panel
    AION_PT_ImporterPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.aion_importer_props = PointerProperty(type=AionImporterProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.aion_importer_props

if __name__ == "__main__":
    register()