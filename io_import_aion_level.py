import bpy
import struct
import math
import os
import tempfile
import zipfile
import xml.etree.ElementTree as ET
import io
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
    "version": (1, 1),
    "blender": (3, 6, 0),
    "location": "View3D > Tools > Aion Importer",
    "description": "Import Aion game maps (heightmaps, vegetation, brushes, mission objects and lights)",
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

class LightObject:
    def __init__(self):
        self.name = ""
        self.position = Vector((0, 0, 0))
        self.color = (1.0, 1.0, 1.0)
        self.outer_radius = 3.0
        self.inner_radius = 0.0
        self.diffuse_multiplier = 1.0
        
    def __str__(self):
        return f"Light {self.name} at {self.position}, color {self.color}, radius {self.outer_radius}m"

class PakFileManager:
    """Manages reading files from archives"""
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.pak_files = []
        self.file_index = {}
        self._load_paks()
    
    def _load_paks(self):
        if not os.path.exists(self.folder_path):
            return
            
        for root, dirs, files in os.walk(self.folder_path):
            for filename in files:
                if filename.lower().endswith('.pak') or filename.lower().endswith('.zip'):
                    pak_path = os.path.join(root, filename)
                    try:
                        pak = zipfile.ZipFile(pak_path, 'r')
                        self.pak_files.append(pak)
                        
                        rel_pak_dir = os.path.dirname(os.path.relpath(pak_path, self.folder_path))
                        
                        for name in pak.namelist():
                            if rel_pak_dir and rel_pak_dir != '.':
                                rel_pak_dir_norm = rel_pak_dir.replace('\\', '/')
                                full_name = rel_pak_dir_norm + '/' + name.replace('\\', '/')
                            else:
                                full_name = name.replace('\\', '/')
                            
                            norm_name = full_name.lower()
                            self.file_index[norm_name] = (pak, name, pak_path)
                    except (zipfile.BadZipFile, Exception) as e:
                        print(f"Warning: Could not open {pak_path}: {e}")
                        continue
    
    def read_file(self, relative_path):
        if not relative_path:
            return None
        norm_path = relative_path.lower().replace('\\', '/')
        if norm_path in self.file_index:
            pak, name, pak_path = self.file_index[norm_path]
            try:
                return pak.read(name)
            except Exception as e:
                print(f"Error reading {name} from {pak_path}: {e}")
                return None
        return None
    
    def file_exists(self, relative_path):
        norm_path = relative_path.lower().replace('\\', '/')
        return norm_path in self.file_index
    
    def extract_file(self, norm_path, temp_dir):
        if norm_path in self.file_index:
            pak, internal_name, pak_path = self.file_index[norm_path]
            try:
                data = pak.read(internal_name)
                out_path = os.path.join(temp_dir, norm_path.replace('/', os.sep))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'wb') as f:
                    f.write(data)
                return out_path
            except Exception as e:
                print(f"Error extracting {norm_path}: {e}")
                return None
        return None
    
    def get_file_list(self):
        return list(self.file_index.keys())
    
    def close(self):
        for pak in self.pak_files:
            try:
                pak.close()
            except:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

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
    light_base_power: FloatProperty(
        name="Light Base Power",
        default=200.0,
        min=0.0,
        max=10000.0,
        description="Base power in Watts for lights. Final power = Base Power × DiffuseMultiplier"
    )
    light_radius_multiplier: FloatProperty(
        name="Light Radius Multiplier",
        default=0.01,
        min=0.01,
        max=10.0,
        description="Multiplier for light radius. Smaller values - sharper shadows and more focused light"
    )

def get_or_create_collection(name, parent=None):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    new_col = bpy.data.collections.new(name)
    if parent:
        parent.children.link(new_col)
    else:
        bpy.context.scene.collection.children.link(new_col)
    return new_col

def parse_leveldata_xml(file_obj):
    tree = ET.parse(file_obj)
    root = tree.getroot()
    
    surface_types = []
    for st in tree.findall('.//SurfaceTypes/SurfaceType'):
        tex_path = st.get('DetailTexture', '')
        scale_x = float(st.get('DetailScaleX', 1.0))
        scale_y = float(st.get('DetailScaleY', 1.0))
        proj_axis = st.get('ProjAxis', 'Z').upper()
        surface_types.append({
            'path': tex_path,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'proj_axis': proj_axis
        })
    
    object_map = {}
    vegetation = root.find('Vegetation')
    if vegetation is not None:
        for i, obj in enumerate(vegetation.findall('Object')):
            cgf_path = obj.get('FileName', '').replace('\\', '/')
            short_name = cgf_path[-57:] if len(cgf_path) > 57 else cgf_path
            object_map[i] = {
                'path': cgf_path,
                'short_name': short_name
            }
    
    map_size = 1536
    level_info = root.find('LevelInfo')
    if level_info is not None:
        map_size = int(level_info.get('HeightmapXSize', 1536))
    
    return surface_types, object_map, map_size

def parse_objects_lst(file_obj, map_size, object_map):
    items = []
    header = struct.unpack('<I', file_obj.read(4))[0]
    if header != 0x10:
        raise ValueError("objects.lst: expected 0x10 header")
    magic = 32768.0 / map_size
    while True:
        data = file_obj.read(16)
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

def parse_brush_lst(file_obj):
    brush_info = []
    entries = []
    
    signature = file_obj.read(3)
    if signature != b'CRY':
        raise ValueError("Wrong signature")
    file_obj.read(4)
    meshDataBlockSz = struct.unpack('<i', file_obj.read(4))[0]
    if meshDataBlockSz < 16 or meshDataBlockSz > 19:
        raise ValueError("Unexpected block size")
    titlesCount = struct.unpack('<i', file_obj.read(4))[0]
    
    for _ in range(titlesCount):
        nameLen = struct.unpack('<i', file_obj.read(4))[0]
        file_obj.read(nameLen - 4)
        
    meshInfoCount = struct.unpack('<i', file_obj.read(4))[0]
    for i in range(meshInfoCount):
        file_obj.read(4)
        filename_bytes = file_obj.read(128)
        filename = filename_bytes.decode('utf-8').strip('\x00').lower().replace('\\', '/')
        file_obj.read(4)
        x1, y1, z1, x2, y2, z2 = struct.unpack('<6f', file_obj.read(24))
        brush_info.append({
            'filename': filename,
            'bbox': ((x1, y1, z1), (x2, y2, z2))
        })
        
    meshDataCount = struct.unpack('<i', file_obj.read(4))[0]
    for i in range(meshDataCount):
        file_obj.read(4)
        file_obj.read(4)
        meshIdx = struct.unpack('<i', file_obj.read(4))[0]
        if meshIdx < 0 or meshIdx >= len(brush_info):
            print(f"Invalid mesh index {meshIdx}, skipping")
            continue
        file_obj.read(4)
        file_obj.read(4)
        file_obj.read(4)
        
        matrix = struct.unpack('<12f', file_obj.read(48))
        rot_matrix = Matrix((
            (matrix[0 * 4 + 0], matrix[0 * 4 + 1], matrix[0 * 4 + 2], matrix[0 * 4 + 3]),
            (matrix[1 * 4 + 0], matrix[1 * 4 + 1], matrix[1 * 4 + 2], matrix[1 * 4 + 3]),
            (matrix[2 * 4 + 0], matrix[2 * 4 + 1], matrix[2 * 4 + 2], matrix[2 * 4 + 3]),
            (0, 0, 0, 1)
        ))
        
        file_obj.read(16)
        eventType = struct.unpack('<i', file_obj.read(4))[0]
        if eventType < 0 or eventType > 4:
            print(f"Ignoring unknown event: {eventType}")
        file_obj.read(4)
        file_obj.read(4 * (meshDataBlockSz - 16))
        
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

def parse_mission_xml(file_obj):
    tree = ET.parse(file_obj)
    root = tree.getroot()
    objects = []
    
    objects_section = root.find('Objects')
    if objects_section is None:
        return objects
        
    for entity in objects_section.findall('Entity'):
        properties = entity.find('Properties')
        if properties is None:
            continue
            
        model_path = properties.get('fileLadderCGF', '') or properties.get('object_Model', '')
        if not model_path:
            continue
            
        obj = MissionObject()
        obj.name = entity.get('Name', '')
        obj.model_path = model_path.replace('\\', '/')
        
        pos_str = entity.get('Pos', '0,0,0')
        try:
            x, y, z = map(float, pos_str.split(','))
            obj.position = Vector((x, y, z))
        except:
            continue
            
        angles_str = entity.get('Angles', '0,0,0')
        try:
            x_rot, y_rot, z_rot = map(float, angles_str.split(','))
            obj.rotation = Euler((radians(x_rot), radians(y_rot), radians(z_rot)), 'XYZ')
        except:
            continue
            
        scale_str = entity.get('Scale', '1')
        try:
            if ',' in scale_str:
                x_scale, y_scale, z_scale = map(float, scale_str.split(','))
                obj.scale = Vector((x_scale, y_scale, z_scale))
            else:
                scale = float(scale_str)
                obj.scale = Vector((scale, scale, scale))
        except:
            continue
            
        objects.append(obj)
    return objects

def parse_lights_xml(file_obj):
    """Parse DeferredLight entities from mission XML"""
    tree = ET.parse(file_obj)
    root = tree.getroot()
    lights = []
    
    objects_section = root.find('Objects')
    if objects_section is None:
        return lights
        
    for entity in objects_section.findall('Entity'):
        entity_class = entity.get('EntityClass', '')
        if entity_class != 'DeferredLight':
            continue
            
        light = LightObject()
        light.name = entity.get('Name', 'DeferredLight')
        
        # Parse position
        pos_str = entity.get('Pos', '0,0,0')
        try:
            x, y, z = map(float, pos_str.split(','))
            light.position = Vector((x, y, z))
        except:
            continue
            
        # Parse properties
        props = entity.find('Properties')
        if props is not None:
            # Color (RGB)
            clr_diffuse = props.get('clrDiffuse', '1,1,1')
            try:
                parts = clr_diffuse.split(',')
                r = float(parts[0])
                g = float(parts[1])
                b = float(parts[2])
                light.color = (r, g, b)
            except:
                light.color = (1.0, 1.0, 1.0)
                
            # Multiplier
            try:
                light.diffuse_multiplier = float(props.get('DiffuseMultiplier', '1'))
            except:
                light.diffuse_multiplier = 1.0
                
            # Radius
            try:
                light.outer_radius = float(props.get('OuterRadius', '3'))
            except:
                light.outer_radius = 3.0
                
            try:
                light.inner_radius = float(props.get('InnerRadius', '0'))
            except:
                light.inner_radius = 0.0
        
        lights.append(light)
        
    return lights

def import_cgf_models_batch(cgf_paths, prefs, context):
    """Import all CGF models and return mesh_library dict {path: mesh_data}"""
    mesh_library = {}
    if not cgf_paths:
        return mesh_library
        
    mesh_col = get_or_create_collection("Imported_Meshes")
    valid_paths = [p for p in cgf_paths if p]
    total = len(valid_paths)
    
    for idx, path in enumerate(valid_paths):
        cgf_full_path = os.path.join(prefs.models_root, path)
        if not os.path.exists(cgf_full_path):
            continue
            
        try:
            objects_before = set(bpy.data.objects)
            
            bpy.ops.import_scene.cgf(
                filepath=cgf_full_path,
                convert_dds_to_png=False,
                reuse_materials=False,
                reuse_images=True,
                import_skeleton=False,
                skeleton_auto_connect=False,
                import_animations=False
            )
            
            imported_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
            if not imported_objects:
                continue
                
            bpy.ops.object.select_all(action='DESELECT')
            for obj in imported_objects:
                obj.hide_set(False)
                obj.hide_viewport = False
                obj.hide_render = False
                obj.select_set(True)
            
            if len(imported_objects) > 1:
                first_obj_matrix = imported_objects[0].matrix_world.copy()
                for obj in imported_objects:
                    obj.data.transform(obj.matrix_world)
                    obj.matrix_world.identity()
                bpy.context.view_layer.objects.active = imported_objects[0]
                bpy.ops.object.join()
                imported_objects = [bpy.context.active_object]
                imported_objects[0].matrix_world = first_obj_matrix
                imported_obj = imported_objects[0]
            else:
                imported_obj = imported_objects[0]
                imported_obj.data.transform(imported_obj.matrix_world)
            
            short_name = os.path.basename(path)
            imported_obj.name = short_name
            imported_obj.data.name = short_name
            
            for col in imported_obj.users_collection:
                col.objects.unlink(imported_obj)
            mesh_col.objects.link(imported_obj)
            imported_obj.hide_viewport = True
            imported_obj.hide_render = True
            
            mesh_library[path] = imported_obj.data
            
            if idx % 10 == 0:
                print(f"Imported {idx}/{total} models...")
                
        except Exception as e:
            print(f"Failed to load {cgf_full_path}: {str(e)}")
            for obj in bpy.data.objects:
                if obj not in objects_before:
                    try:
                        bpy.data.objects.remove(obj, do_unlink=True)
                    except:
                        pass
            continue
            
    return mesh_library

def create_vegetation_instances(vegetation_items, mesh_library, context):
    if not vegetation_items:
        return 0
        
    main_col = get_or_create_collection("Vegetation")
    instances_col = get_or_create_collection("Vegetation_Instances", main_col)
    
    count = 0
    for item in vegetation_items:
        if not item.cgf_path or item.cgf_path not in mesh_library:
            continue
            
        mesh_data = mesh_library[item.cgf_path]
        new_obj = bpy.data.objects.new(item.short_name, mesh_data)
        new_obj.location = item.position
        new_obj.rotation_euler = Euler((0, 0, radians(item.heading)), 'XYZ')
        new_obj.scale = (item.scale, item.scale, item.scale)
        instances_col.objects.link(new_obj)
        count += 1
        
    return count

def create_brush_instances(brush_entries, brush_info, mesh_library, context):
    if not brush_entries:
        return 0
        
    main_col = get_or_create_collection("Brushes")
    instances_col = get_or_create_collection("Brush_Instances", main_col)
    
    count = 0
    for entry in brush_entries:
        if entry['eventType'] > 0:
            continue
            
        filename = brush_info[entry['meshIdx']]['filename']
        if filename not in mesh_library:
            continue
            
        mesh_data = mesh_library[filename]
        new_obj = bpy.data.objects.new(entry['short_name'], mesh_data)
        new_obj.matrix_world = entry['rotationMatrix']
        instances_col.objects.link(new_obj)
        count += 1
        
    return count

def create_mission_instances(mission_objects, mesh_library, context):
    if not mission_objects:
        return 0
        
    main_col = get_or_create_collection("MissionObjects")
    instances_col = get_or_create_collection("Mission_Instances", main_col)
    
    count = 0
    for obj in mission_objects:
        if obj.model_path not in mesh_library:
            continue
            
        mesh_data = mesh_library[obj.model_path]
        new_obj = bpy.data.objects.new(obj.name, mesh_data)
        new_obj.location = obj.position
        new_obj.rotation_euler = obj.rotation
        new_obj.scale = obj.scale
        instances_col.objects.link(new_obj)
        count += 1
        
    return count

def create_light_instances(light_objects, context, base_power=100.0, radius_multiplier=1.0):
    """Create Blender light objects from parsed light data"""
    if not light_objects:
        return 0
        
    main_col = get_or_create_collection("Lights")
    
    count = 0
    for light_obj in light_objects:
        # Create light data
        light_data = bpy.data.lights.new(name=light_obj.name, type='POINT')
        light_data.color = light_obj.color
        light_data.energy = base_power * light_obj.diffuse_multiplier
        
        # Set radius (shadow_soft_size) based on OuterRadius and multiplier
        light_data.shadow_soft_size = light_obj.outer_radius * radius_multiplier
        
        # Create object
        light_ob = bpy.data.objects.new(light_obj.name, light_data)
        light_ob.location = light_obj.position
        main_col.objects.link(light_ob)
        count += 1
        
    return count

class H32HeightmapImporter:
    @staticmethod
    def create_terrain_blend_material(surface_types, material_name="TerrainBlend"):
        """Create a multi-material blending shader using vertex colors and mix nodes"""
        blend_logger.info(f"Creating terrain blend material: {material_name}")
        mat = bpy.data.materials.new(material_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        for node in nodes:
            nodes.remove(node)
            
        output_node = nodes.new('ShaderNodeOutputMaterial')
        output_node.location = (1400, 0)
        
        main_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        main_bsdf.location = (1200, 0)
        main_bsdf.inputs['Roughness'].default_value = 1.0
        main_bsdf.inputs['Specular'].default_value = 0.0
        links.new(main_bsdf.outputs[0], output_node.inputs[0])
        
        # Use world Position instead of UV for projection mapping
        geo_node = nodes.new('ShaderNodeNewGeometry')
        geo_node.location = (-1000, 300)
        
        # Shared Separate XYZ for position components
        sep_xyz = nodes.new('ShaderNodeSeparateXYZ')
        sep_xyz.location = (-800, 300)
        links.new(geo_node.outputs['Position'], sep_xyz.inputs[0])
        
        # Vertex color nodes for blending weights
        vertex_color_nodes = []
        separate_rgb_nodes = []
        for i in range(8):
            vc_node = nodes.new('ShaderNodeVertexColor')
            vc_node.layer_name = f"BlendWeights_{i}"
            vc_node.location = (-600, 300 + i * -150)
            vertex_color_nodes.append(vc_node)
            
            sep_node = nodes.new('ShaderNodeSeparateRGB')
            sep_node.location = (-400, 300 + i * -150)
            links.new(vc_node.outputs['Color'], sep_node.inputs['Image'])
            separate_rgb_nodes.append(sep_node)
        
        texture_nodes = []
        current_x = 0
        current_y = 400
        
        active_materials = []
        for mat_id, surf in enumerate(surface_types):
            if mat_id >= 32:
                continue
            tex_path = surf['path']
            scale_x = surf['scale_x']
            scale_y = surf['scale_y']
            proj_axis = surf.get('proj_axis', 'Z')
            
            if tex_path and os.path.exists(tex_path):
                active_materials.append((mat_id, tex_path, scale_x, scale_y, proj_axis))
                blend_logger.info(f"Adding material {mat_id}: {os.path.basename(tex_path)} [{proj_axis}]")
            else:
                if tex_path:
                    blend_logger.warning(f"Texture not found for material {mat_id}: {tex_path}")
        
        blend_logger.info(f"Processing {len(active_materials)} active materials")
        
        if not active_materials:
            blend_logger.warning("No active materials found")
            links.new(geo_node.outputs['Position'], main_bsdf.inputs[0])
            return mat
        
        # Create texture nodes with projection handling
        for mat_id, tex_path, scale_x, scale_y, proj_axis in active_materials:
            # Create Combine XYZ to select projection plane
            combine_node = nodes.new('ShaderNodeCombineXYZ')
            combine_node.location = (current_x - 400, current_y)

            if proj_axis == 'X':
                links.new(sep_xyz.outputs['Y'], combine_node.inputs['X'])
                links.new(sep_xyz.outputs['Z'], combine_node.inputs['Y'])
            elif proj_axis == 'Y':
                links.new(sep_xyz.outputs['X'], combine_node.inputs['X'])
                links.new(sep_xyz.outputs['Z'], combine_node.inputs['Y'])
            else:  # Z (default top-down)
                links.new(sep_xyz.outputs['Y'], combine_node.inputs['X'])
                links.new(sep_xyz.outputs['X'], combine_node.inputs['Y'])
            
            # Mapping node for tiling scale
            mapping_node = nodes.new('ShaderNodeMapping')
            mapping_node.location = (current_x - 200, current_y)
            mapping_node.inputs['Scale'].default_value = (-scale_x, scale_y, 1.0)
            links.new(combine_node.outputs[0], mapping_node.inputs[0])
            
            # Image texture node
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = (current_x, current_y)
            try:
                tex_node.image = bpy.data.images.load(tex_path)
                if tex_node.image:
                    tex_node.image.pack()
                    blend_logger.info(f"Packed texture: {os.path.basename(tex_path)}")
            except Exception as e:
                blend_logger.error(f"Failed to load texture {tex_path}: {e}")
            
            links.new(mapping_node.outputs[0], tex_node.inputs[0])
            texture_nodes.append((mat_id, tex_node))
            
            current_y -= 300
            if (len(texture_nodes)) % 4 == 0:
                current_x += 400
                current_y = 400
        
        if len(texture_nodes) == 1:
            links.new(texture_nodes[0][1].outputs[0], main_bsdf.inputs['Base Color'])
        else:
            texture_nodes.sort(key=lambda x: x[0])
            material_data = []
            for mat_id, tex_node in texture_nodes:
                vc_layer_idx = mat_id // 4
                channel_idx = mat_id % 4
                
                if vc_layer_idx >= len(separate_rgb_nodes):
                    continue
                
                sep_node = separate_rgb_nodes[vc_layer_idx]
                
                if channel_idx == 0:
                    socket = sep_node.outputs['R']
                elif channel_idx == 1:
                    socket = sep_node.outputs['G']
                elif channel_idx == 2:
                    socket = sep_node.outputs['B']
                else:
                    socket = vertex_color_nodes[vc_layer_idx].outputs['Alpha']
                
                material_data.append((tex_node.outputs['Color'], socket, mat_id))
            
            # Pairwise mixing
            current_blend_data = material_data[:]
            mix_x_start = 400
            mix_y_start = 0
            mix_y_offset_step = -150
            iteration = 0
            
            while len(current_blend_data) > 1:
                next_blend_data = []
                mix_y_current = mix_y_start + (iteration * mix_y_offset_step * (len(current_blend_data) // 2 + 1))
                
                for i in range(0, len(current_blend_data), 2):
                    if i + 1 < len(current_blend_data):
                        color_out_1, weight_out_1, mat_id_1 = current_blend_data[i]
                        color_out_2, weight_out_2, mat_id_2 = current_blend_data[i+1]
                        
                        math_add = nodes.new('ShaderNodeMath')
                        math_add.operation = 'ADD'
                        math_add.location = (mix_x_start + iteration * 200, mix_y_current + i * 50)
                        links.new(weight_out_1, math_add.inputs[0])
                        links.new(weight_out_2, math_add.inputs[1])
                        
                        math_divide = nodes.new('ShaderNodeMath')
                        math_divide.operation = 'DIVIDE'
                        math_divide.location = (mix_x_start + iteration * 200 + 200, mix_y_current + i * 50)
                        links.new(weight_out_2, math_divide.inputs[0])
                        links.new(math_add.outputs[0], math_divide.inputs[1])
                        
                        math_clamp = nodes.new('ShaderNodeClamp')
                        math_clamp.location = (mix_x_start + iteration * 200 + 400, mix_y_current + i * 50)
                        links.new(math_divide.outputs[0], math_clamp.inputs['Value'])
                        
                        mix_shader = nodes.new('ShaderNodeMixRGB')
                        mix_shader.location = (mix_x_start + iteration * 200 + 600, mix_y_current + i * 50)
                        mix_shader.blend_type = 'MIX'
                        mix_shader.use_clamp = True
                        
                        links.new(color_out_1, mix_shader.inputs['Color1'])
                        links.new(color_out_2, mix_shader.inputs['Color2'])
                        links.new(math_clamp.outputs[0], mix_shader.inputs['Fac'])
                        
                        next_blend_data.append((mix_shader.outputs['Color'], math_add.outputs[0], f"{mat_id_1}_{mat_id_2}"))
                    else:
                        next_blend_data.append(current_blend_data[i])
                
                current_blend_data = next_blend_data
                iteration += 1
            
            if current_blend_data:
                links.new(current_blend_data[0][0], main_bsdf.inputs['Base Color'])
                
        blend_logger.info("Terrain blend material created successfully")
        return mat

    @staticmethod
    def calculate_blend_weights(vertices, materials, width, blend_radius=2.0):
        """Calculate smooth blend weights based on material boundaries"""
        blend_logger.info(f"Calculating blend weights with radius {blend_radius}")
        vertex_count = len(vertices)
        materials = np.asarray(materials, dtype=np.int32)
        
        # Reshape to 2D grid for spatial operations
        materials_2d = materials.reshape(width, width)
        
        # Initialize weights array (32 materials, H, W)
        raw_weights = np.zeros((32, width, width), dtype=np.float32)
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:width, 0:width]
        
        # Pre-calculate which offsets are within circular radius
        radius_int = int(np.ceil(blend_radius))
        offsets = []
        for dy in range(-radius_int, radius_int + 1):
            for dx in range(-radius_int, radius_int + 1):
                if dx == 0 and dy == 0:
                    continue
                dist = np.sqrt(dx*dx + dy*dy)
                if dist <= blend_radius:
                    weight = 1.0 - (dist / blend_radius)
                    offsets.append((dy, dx, weight))
        
        # For each neighbor offset, roll the material array and accumulate weights
        valid_mask = materials_2d < 32  # Only process valid material indices
        
        for dy, dx, weight in offsets:
            # Roll materials to get neighbor values
            shifted = np.roll(materials_2d, (dy, dx), axis=(0, 1))
            
            # Mask where: 1) we have valid current material, 2) neighbor is different, 3) neighbor is valid
            mask = valid_mask & (shifted != materials_2d) & (shifted < 32)
            
            if not np.any(mask):
                continue
            
            # Get coordinates and material IDs where mask is True
            y_coords = y_grid[mask]
            x_coords = x_grid[mask]
            mat_ids = shifted[mask]
            
            # Accumulate weights: raw_weights[mat_id, y, x] += weight
            np.add.at(raw_weights, (mat_ids, y_coords, x_coords), weight)
        
        # Reshape to (vertex_count, 32) for further processing
        raw_weights_2d = raw_weights.reshape(32, -1).T  # Shape: (vertex_count, 32)
        
        # Apply normalization logic
        blend_weights = np.zeros((vertex_count, 32), dtype=np.float32)
        
        # Only process vertices with valid materials (<32)
        valid_verts = materials < 32
        
        if not np.any(valid_verts):
            return blend_weights
        
        current_mats = materials[valid_verts]
        vertex_indices = np.where(valid_verts)[0]
        
        # Get raw neighbor weights for valid vertices
        valid_raw = raw_weights_2d[valid_verts]  # Shape: (n_valid, 32)
        
        # Sum of neighbor contributions per vertex
        total_neighbor_raw = valid_raw.sum(axis=1)
        
        # Apply blend_amount (0.3) and clamp to 0.9
        total_neighbor_weight = total_neighbor_raw * 0.3
        total_neighbor_weight = np.clip(total_neighbor_weight, 0, 0.9)
        
        # Normalize neighbor contributions
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = valid_raw / total_neighbor_raw[:, np.newaxis]
            normalized = np.where(total_neighbor_raw[:, np.newaxis] > 0, normalized, 0)
        
        # Scale by total_neighbor_weight
        neighbor_contributions = normalized * total_neighbor_weight[:, np.newaxis]
        
        # Assign to output array
        blend_weights[valid_verts] = neighbor_contributions
        
        # Set current material weight
        current_weights = 1.0 - total_neighbor_weight
        blend_weights[vertex_indices, current_mats] = current_weights
        
        blend_logger.info("Blend weight calculation completed (vectorized)")
        return blend_weights

    @staticmethod
    def apply_vertex_colors(mesh, blend_weights, width):
        """Apply blend weights as vertex colors (8 RGBA layers for 32 materials)"""
        blend_logger.info("Applying vertex colors for material blending")
        
        if not isinstance(blend_weights, np.ndarray):
            blend_weights = np.array(blend_weights, dtype=np.float32)
        
        loop_count = len(mesh.loops)
        
        # Get vertex indices for each loop
        loop_vertex_indices = np.zeros(loop_count, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vertex_indices)
        
        for layer_idx in range(8):
            base_idx = layer_idx * 4
            
            # Extract weights for this layer's 4 materials (loop_count, 4)
            # Ensure we don't go out of bounds if <32 materials
            end_idx = min(base_idx + 4, blend_weights.shape[1])
            layer_weights = np.zeros((loop_count, 4), dtype=np.float32)
            
            if base_idx < blend_weights.shape[1]:
                w_slice = blend_weights[loop_vertex_indices, base_idx:end_idx]
                layer_weights[:, :end_idx-base_idx] = w_slice
            
            # Create RGBA array (loop_count, 4)
            colors = layer_weights
            
            # Flatten to [r,g,b,a, r,g,b,a, ...] for foreach_set
            colors_flat = colors.flatten()
            
            # Get or create vertex color layer
            layer_name = f"BlendWeights_{layer_idx}"
            if layer_name not in mesh.vertex_colors:
                vcol = mesh.vertex_colors.new(name=layer_name)
            else:
                vcol = mesh.vertex_colors[layer_name]
            
            # Set all colors at once
            vcol.data.foreach_set("color", colors_flat)
        
        blend_logger.info("Vertex colors applied successfully")

    @staticmethod
    def load_h32_heightmap(h32_data, leveldata_file_obj, enable_smooth_blending=True, 
                          blend_radius=2.0, import_water_plane=True, models_root="", pak_manager=None):
        
        blend_logger.info(f"Loading H32 heightmap: {len(h32_data)} bytes")
        blend_logger.info(f"Smooth blending: {enable_smooth_blending}, Blend radius: {blend_radius}")

        tree = ET.parse(leveldata_file_obj)
        
        with tempfile.TemporaryDirectory() as temp_tex_dir:
            def resolve_texture(tex_path):
                if not tex_path:
                    return None
                    
                tex_rel = tex_path.replace('\\', '/')
                
                if models_root:
                    full_path = os.path.join(models_root, tex_rel.replace('/', os.sep))
                    if os.path.exists(full_path):
                        return full_path
                    full_path_lower = os.path.join(models_root, tex_rel.lower().replace('/', os.sep))
                    if os.path.exists(full_path_lower):
                        return full_path_lower
                
                if pak_manager:
                    norm_path = tex_rel.lower()
                    if pak_manager.file_exists(norm_path):
                        extracted = pak_manager.extract_file(norm_path, temp_tex_dir)
                        if extracted:
                            return extracted
                return None

            surface_types = []
            for st in tree.findall('.//SurfaceTypes/SurfaceType'):
                tex_path = st.get('DetailTexture', '')
                scale_x = float(st.get('DetailScaleX', 1.0))
                scale_y = float(st.get('DetailScaleY', 1.0))
                proj_axis = st.get('ProjAxis', 'Z').upper()
                
                resolved_path = resolve_texture(tex_path)
                if resolved_path:
                    blend_logger.info(f"Found texture: {os.path.basename(tex_path)} (Proj: {proj_axis})")
                elif tex_path:
                    blend_logger.warning(f"Texture not found: {tex_path}")
                
                surface_types.append({
                    'path': resolved_path,
                    'scale_x': scale_x,
                    'scale_y': scale_y,
                    'proj_axis': proj_axis
                })

            # Parse heights and materials using NumPy
            dt = np.dtype([('height', np.uint16), ('mat', np.uint8)])
            dt = dt.newbyteorder('<')
            
            # Ensure data size matches expected vertex count
            vertex_count_expected = len(h32_data) // 3
            valid_bytes = vertex_count_expected * 3
            
            arr = np.frombuffer(h32_data[:valid_bytes], dtype=dt)
            heights = arr['height'].astype(np.float32) / 32.0
            materials = arr['mat'].astype(np.int32)
            
            width = int(np.sqrt(vertex_count_expected))
            blend_logger.info(f"Heightmap dimensions: {width}x{width}")
            
            indices = np.arange(vertex_count_expected)
            x_coords = (indices % width) * 2.0
            y_coords = (indices // width) * 2.0
            
            vertices = list(map(Vector, zip(y_coords, x_coords, heights)))
            
            # Identify cutout vertices (mat == 0x3F)
            cutout_indices = set(np.where(materials == 0x3F)[0].tolist())

            mesh_name = "Heightmap" if not enable_smooth_blending else "HeightmapBlended"
            mesh = bpy.data.meshes.new(mesh_name)
            obj = bpy.data.objects.new(mesh_name, mesh)
            bpy.context.collection.objects.link(obj)

            if enable_smooth_blending:
                blend_logger.info("Generating faces with smooth blending")
                
                y_idx, x_idx = np.mgrid[0:width-1, 0:width-1]
                v1 = y_idx * width + x_idx
                v2 = y_idx * width + (x_idx + 1)
                v3 = (y_idx + 1) * width + (x_idx + 1)
                v4 = (y_idx + 1) * width + x_idx
                
                # Stack into face array
                faces_array = np.stack([v1, v2, v3, v4], axis=-1).reshape(-1, 4)
               
                if cutout_indices:
                    face_starts = faces_array[:, 0]
                    valid_faces_mask = ~np.isin(face_starts, list(cutout_indices))
                    faces_array = faces_array[valid_faces_mask]
                
                faces = [tuple(f) for f in faces_array]
                
                uvs_flat = np.zeros((len(faces) * 4, 2), dtype=np.float32)
                x_face = x_idx.flatten()
                y_face = y_idx.flatten()
                
                if cutout_indices:
                    x_face = x_face[valid_faces_mask]
                    y_face = y_face[valid_faces_mask]
                
                uvs_flat[0::4] = np.column_stack([x_face, y_face])
                uvs_flat[1::4] = np.column_stack([x_face + 1, y_face])
                uvs_flat[2::4] = np.column_stack([x_face + 1, y_face + 1])
                uvs_flat[3::4] = np.column_stack([x_face, y_face + 1])
                
                mesh.from_pydata(vertices, [], faces)
                mesh.update()

                uv_layer = mesh.uv_layers.new(name="ScaledUV")
                uv_layer.data.foreach_set("uv", uvs_flat.flatten())

                # Calculate and apply blend weights
                blend_weights = H32HeightmapImporter.calculate_blend_weights(
                    vertices, materials, width, blend_radius
                )
                H32HeightmapImporter.apply_vertex_colors(mesh, blend_weights, width)
                
                blend_material = H32HeightmapImporter.create_terrain_blend_material(
                    surface_types, "TerrainBlend"
                )
                mesh.materials.append(blend_material)
                for poly in mesh.polygons:
                    poly.material_index = 0
            else:
                # Non-smoothed mode
                y_idx, x_idx = np.mgrid[0:width-1, 0:width-1]
                v1 = y_idx * width + x_idx
                v2 = y_idx * width + (x_idx + 1)
                v3 = (y_idx + 1) * width + (x_idx + 1)
                v4 = (y_idx + 1) * width + x_idx
                
                faces_array = np.stack([v1, v2, v3, v4], axis=-1).reshape(-1, 4)
                
                if cutout_indices:
                    face_starts = faces_array[:, 0]
                    valid_faces_mask = ~np.isin(face_starts, list(cutout_indices))
                    faces_array = faces_array[valid_faces_mask]
                    # Store material indices per face before filtering
                    face_mats = materials[v1.flatten()][valid_faces_mask]
                else:
                    face_mats = materials[v1.flatten()]
                
                faces = [tuple(f) for f in faces_array]
                material_indices = face_mats.tolist()
                
                # UV generation with scaling per material
                uvs_list = []
                mat_to_faces = {}
                
                # Pre-calculate scales for all materials
                scales_x = np.array([s['scale_x'] * 4 if s['path'] else 1.0 for s in surface_types] + [1.0] * (256 - len(surface_types)))
                scales_y = np.array([s['scale_y'] * 4 if s['path'] else 1.0 for s in surface_types] + [1.0] * (256 - len(surface_types)))
                
                x_face = x_idx.flatten()
                y_face = y_idx.flatten()
                if cutout_indices:
                    x_face = x_face[valid_faces_mask]
                    y_face = y_face[valid_faces_mask]
                
                sx = scales_x[face_mats]
                sy = scales_y[face_mats]
                
                uvs_flat = np.zeros((len(faces) * 4, 2), dtype=np.float32)
                uvs_flat[0::4, 0] = x_face * sx
                uvs_flat[0::4, 1] = y_face * sy
                uvs_flat[1::4, 0] = (x_face + 1) * sx
                uvs_flat[1::4, 1] = y_face * sy
                uvs_flat[2::4, 0] = (x_face + 1) * sx
                uvs_flat[2::4, 1] = (y_face + 1) * sy
                uvs_flat[3::4, 0] = x_face * sx
                uvs_flat[3::4, 1] = (y_face + 1) * sy
                
                mesh.from_pydata(vertices, [], faces)
                mesh.update()

                uv_layer = mesh.uv_layers.new(name="ScaledUV")
                uv_layer.data.foreach_set("uv", uvs_flat.flatten())

                # Build material mapping
                texture_materials = {}
                valid_mat_ids = set(material_indices)
                
                for mat_id in valid_mat_ids:
                    if mat_id >= len(surface_types) or mat_id >= 64:
                        continue
                    tex_path = surface_types[mat_id]['path']
                    if not tex_path or tex_path in texture_materials:
                        continue
                    
                    mat = bpy.data.materials.new(f"mat_{mat_id}")
                    mat.use_nodes = True
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    
                    for node in list(nodes):
                        nodes.remove(node)
                        
                    output = nodes.new('ShaderNodeOutputMaterial')
                    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
                    tex_node = nodes.new('ShaderNodeTexImage')
                    uv_node = nodes.new('ShaderNodeUVMap')
                    
                    uv_node.uv_map = "ScaledUV"
                    bsdf.inputs['Roughness'].default_value = 1.0
                    bsdf.inputs['Specular'].default_value = 0.0
                    
                    links.new(uv_node.outputs[0], tex_node.inputs[0])
                    links.new(tex_node.outputs[0], bsdf.inputs[0])
                    links.new(bsdf.outputs[0], output.inputs[0])
                    
                    uv_node.location = (-300, 0)
                    tex_node.location = (-100, 0)
                    bsdf.location = (100, 0)
                    output.location = (300, 0)
                    
                    try:
                        tex_node.image = bpy.data.images.load(tex_path)
                        if tex_node.image:
                            tex_node.image.pack()
                    except Exception as e:
                        blend_logger.error(f"Failed to load texture {tex_path}: {e}")
                    texture_materials[tex_path] = mat
                
                for mat in texture_materials.values():
                    mesh.materials.append(mat)
                
                # Assign material indices to faces
                tex_paths = list(texture_materials.keys())
                for i, face in enumerate(mesh.polygons):
                    mat_idx = material_indices[i]
                    if mat_idx < len(surface_types):
                        tex_path = surface_types[mat_idx]['path']
                        if tex_path in texture_materials:
                            face.material_index = tex_paths.index(tex_path)

            mesh.shade_smooth()
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
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

class AION_OT_RemoveNoDraw(Operator):
    bl_idname = "aion.remove_nodraw"
    bl_label = "Remove NoDraw Geometry"
    bl_description="Remove faces with materials containing '(NoDraw)'"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        current_active = bpy.context.active_object
        current_mode = bpy.context.object.mode if current_active else None
        
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                mesh = obj.data
                if not mesh.materials:
                    continue
                    
                nodraw_material_indices = []
                for i, mat in enumerate(mesh.materials):
                    if mat and "(NoDraw)" in mat.name:
                        nodraw_material_indices.append(i)
                        
                if not nodraw_material_indices:
                    continue
                    
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bm.faces.ensure_lookup_table()
                
                faces_to_remove = []
                for face in bm.faces:
                    if face.material_index in nodraw_material_indices:
                        faces_to_remove.append(face)
                        
                if faces_to_remove:
                    bmesh.ops.delete(bm, geom=faces_to_remove, context='FACES')
                    verts_to_remove = [v for v in bm.verts if not v.link_faces]
                    if verts_to_remove:
                        bmesh.ops.delete(bm, geom=verts_to_remove, context='VERTS')
                        
                bm.to_mesh(mesh)
                bm.free()
                obj.select_set(False)
                
        if current_active:
            bpy.context.view_layer.objects.active = current_active
            if current_mode and current_mode != 'OBJECT':
                try:
                    bpy.ops.object.mode_set(mode=current_mode)
                except:
                    pass
                    
        self.report({'INFO'}, "Removed NoDraw geometry")
        return {'FINISHED'}

class AION_OT_ImportMapFolder(Operator, ImportHelper):
    bl_idname = "aion.import_map_folder"
    bl_label = "Import Full Map from Folder"
    bl_description = "Import Full Map from Folder"
    bl_options = {'REGISTER', 'UNDO'}
    use_filter_folder = True
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        folder_path = self.filepath
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Selected path is not a directory.")
            return {'CANCELLED'}

        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        
        total_start = time.time()
        timers = {}

        with PakFileManager(prefs.models_root) as tex_pak_manager:
            with PakFileManager(folder_path) as map_pak_manager:
                # Parse leveldata
                t_start = time.time()
                leveldata_bytes = map_pak_manager.read_file('leveldata.xml')
                if not leveldata_bytes:
                    self.report({'ERROR'}, "leveldata.xml not found")
                    return {'CANCELLED'}
                
                h32_bytes = map_pak_manager.read_file('terrain/land_map.h32')
                if not h32_bytes:
                    self.report({'ERROR'}, "terrain/land_map.h32 not found")
                    return {'CANCELLED'}
                
                leveldata_io = io.BytesIO(leveldata_bytes)
                surface_types, object_map, map_size = parse_leveldata_xml(leveldata_io)
                timers['parse_leveldata'] = time.time() - t_start

                # Parse vegetation
                t_start = time.time()
                vegetation_items = []
                objects_lst_bytes = map_pak_manager.read_file('objects.lst')
                if objects_lst_bytes:
                    try:
                        vegetation_items = parse_objects_lst(
                            io.BytesIO(objects_lst_bytes), map_size, object_map
                        )
                    except Exception as e:
                        print(f"Failed to parse objects.lst: {e}")
                timers['parse_vegetation'] = time.time() - t_start

                # Parse brushes
                t_start = time.time()
                brush_entries = []
                brush_info = []
                brush_lst_bytes = map_pak_manager.read_file('brush.lst')
                if brush_lst_bytes:
                    try:
                        brush_info, brush_entries = parse_brush_lst(io.BytesIO(brush_lst_bytes))
                    except Exception as e:
                        print(f"Failed to parse brush.lst: {e}")
                timers['parse_brushes'] = time.time() - t_start

                # Parse mission objects and lights
                t_start = time.time()
                mission_objects = []
                light_objects = []
                mission_file = None
                for fname in map_pak_manager.get_file_list():
                    if fname.startswith('mission_') and fname.endswith('.xml'):
                        mission_file = fname
                        break
                        
                if mission_file:
                    mission_bytes = map_pak_manager.read_file(mission_file)
                    if mission_bytes:
                        try:
                            mission_objects = parse_mission_xml(io.BytesIO(mission_bytes))
                            light_objects = parse_lights_xml(io.BytesIO(mission_bytes))
                        except Exception as e:
                            print(f"Failed to parse mission XML: {e}")
                timers['parse_mission'] = time.time() - t_start

                # Collect CGF paths
                all_cgf_paths = set()
                for item in vegetation_items:
                    if item.cgf_path:
                        all_cgf_paths.add(item.cgf_path)
                for entry in brush_entries:
                    if entry['meshIdx'] < len(brush_info):
                        all_cgf_paths.add(brush_info[entry['meshIdx']]['filename'])
                for obj in mission_objects:
                    if obj.model_path:
                        all_cgf_paths.add(obj.model_path)

                # Import all meshes
                t_start = time.time()
                print(f"[Timer] Starting import of {len(all_cgf_paths)} unique CGF models...")
                mesh_library = import_cgf_models_batch(all_cgf_paths, prefs, context)
                timers['import_cgf'] = time.time() - t_start
                print(f"[Timer] CGF Import took: {timers['import_cgf']:.2f}s")

                # Create instances
                t_start = time.time()
                veg_count = create_vegetation_instances(vegetation_items, mesh_library, context)
                timers['instances_veg'] = time.time() - t_start
                print(f"[Timer] Vegetation instances ({veg_count}): {timers['instances_veg']:.2f}s")

                t_start = time.time()
                brush_count = create_brush_instances(brush_entries, brush_info, mesh_library, context)
                timers['instances_brush'] = time.time() - t_start
                print(f"[Timer] Brush instances ({brush_count}): {timers['instances_brush']:.2f}s")

                t_start = time.time()
                mission_count = create_mission_instances(mission_objects, mesh_library, context)
                timers['instances_mission'] = time.time() - t_start
                print(f"[Timer] Mission instances ({mission_count}): {timers['instances_mission']:.2f}s")

                # Create lights
                light_count = 0
                t_start = time.time()
                light_count = create_light_instances(
                    light_objects, 
                    context, 
                    base_power=props.light_base_power,
                    radius_multiplier=props.light_radius_multiplier
                )
                timers['lights'] = time.time() - t_start
                print(f"[Timer] Lights ({light_count}): {timers['lights']:.2f}s")

                # Import heightmap
                t_start = time.time()
                print("[Timer] Starting heightmap import...")
                try:
                    H32HeightmapImporter.load_h32_heightmap(
                        h32_bytes,
                        io.BytesIO(leveldata_bytes),
                        props.enable_smooth_blending,
                        props.blend_radius,
                        props.import_water_plane,
                        prefs.models_root,
                        tex_pak_manager
                    )
                except Exception as e:
                    self.report({'ERROR'}, f"Heightmap import failed: {e}")
                timers['heightmap'] = time.time() - t_start
                print(f"[Timer] Heightmap import: {timers['heightmap']:.2f}s")

                # Remove NoDraw
                if props.remove_nodraw:
                    t_start = time.time()
                    bpy.ops.aion.remove_nodraw()
                    timers['nodraw'] = time.time() - t_start
                    print(f"[Timer] NoDraw removal: {timers['nodraw']:.2f}s")

        total_time = time.time() - total_start
        print(f"\n[Timer] ===== TOTAL IMPORT TIME: {total_time:.2f}s =====")
        print(f"[Timer] Breakdown:")
        for key, val in timers.items():
            print(f"  {key}: {val:.2f}s")
            
        self.report({'INFO'}, f"Import completed in {total_time:.1f}s | Veg:{veg_count} Brush:{brush_count} Mission:{mission_count} Lights:{light_count}")
        return {'FINISHED'}

class AION_OT_ImportHeightmap(Operator, ImportHelper):
    bl_idname = "aion.import_heightmap"
    bl_label = "Import Heightmap"
    bl_description ="Import Aion terrain from a folder"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    enable_smooth_blending: BoolProperty(
        name="Material Blending",
        default=True
    )
    blend_radius: FloatProperty(
        name="Blend Radius",
        default=2.0,
        min=0.5,
        max=10.0
    )
    import_water_plane: BoolProperty(
        name="Import Water Plane",
        default=True
    )

    def execute(self, context):
        folder_path = self.filepath if os.path.isdir(self.filepath) else os.path.dirname(self.filepath)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Invalid directory")
            return {'CANCELLED'}

        prefs = context.preferences.addons[__name__].preferences
        start = time.time()
        
        with PakFileManager(prefs.models_root) as tex_pak_manager:
            with PakFileManager(folder_path) as pak_manager:
                leveldata_bytes = pak_manager.read_file('leveldata.xml')
                h32_bytes = pak_manager.read_file('terrain/land_map.h32')
                
                if not leveldata_bytes or not h32_bytes:
                    missing = []
                    if not leveldata_bytes: missing.append("leveldata.xml")
                    if not h32_bytes: missing.append("terrain/land_map.h32")
                    self.report({'ERROR'}, f"Missing: {', '.join(missing)}")
                    return {'CANCELLED'}

                try:
                    H32HeightmapImporter.load_h32_heightmap(
                        h32_bytes,
                        io.BytesIO(leveldata_bytes),
                        self.enable_smooth_blending,
                        self.blend_radius,
                        self.import_water_plane,
                        prefs.models_root,
                        tex_pak_manager
                    )
                    print(f"[Timer] Heightmap import took: {time.time()-start:.2f}s")
                    return {'FINISHED'}
                except Exception as e:
                    self.report({'ERROR'}, str(e))
                    return {'CANCELLED'}

class AION_OT_ImportVegetation(Operator, ImportHelper):
    bl_idname = "aion.import_vegetation"
    bl_label = "Import Vegetation"
    bl_description ="Import Aion vegetation from a folder"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        folder_path = self.filepath if os.path.isdir(self.filepath) else os.path.dirname(self.filepath)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Invalid directory")
            return {'CANCELLED'}

        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        start = time.time()
        
        with PakFileManager(folder_path) as pak_manager:
            leveldata_bytes = pak_manager.read_file('leveldata.xml')
            objects_lst_bytes = pak_manager.read_file('objects.lst')
            
            if not leveldata_bytes or not objects_lst_bytes:
                missing = []
                if not leveldata_bytes: missing.append("leveldata.xml")
                if not objects_lst_bytes: missing.append("objects.lst")
                self.report({'ERROR'}, f"Missing: {', '.join(missing)}")
                return {'CANCELLED'}

            try:
                _, object_map, map_size = parse_leveldata_xml(io.BytesIO(leveldata_bytes))
                vegetation_items = parse_objects_lst(
                    io.BytesIO(objects_lst_bytes), map_size, object_map
                )
                
                unique_paths = {item.cgf_path for item in vegetation_items if item.cgf_path}
                mesh_library = import_cgf_models_batch(unique_paths, prefs, context)
                count = create_vegetation_instances(vegetation_items, mesh_library, context)
                
                if props.remove_nodraw:
                    bpy.ops.aion.remove_nodraw()
                
                elapsed = time.time() - start
                print(f"[Timer] Vegetation import: {elapsed:.2f}s ({count} instances)")
                self.report({'INFO'}, f"Imported {count} vegetation instances in {elapsed:.1f}s")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

class AION_OT_ImportBrushes(Operator, ImportHelper):
    bl_idname = "aion.import_brushes"
    bl_label = "Import Brushes"
    bl_description ="Import Aion static objects from a folder"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        folder_path = self.filepath if os.path.isdir(self.filepath) else os.path.dirname(self.filepath)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Invalid directory")
            return {'CANCELLED'}

        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        start = time.time()
        
        with PakFileManager(folder_path) as pak_manager:
            brush_lst_bytes = pak_manager.read_file('brush.lst')
            
            if not brush_lst_bytes:
                self.report({'ERROR'}, "brush.lst not found")
                return {'CANCELLED'}

            try:
                brush_info, brush_entries = parse_brush_lst(io.BytesIO(brush_lst_bytes))
                
                unique_paths = set()
                for entry in brush_entries:
                    if entry['meshIdx'] < len(brush_info):
                        unique_paths.add(brush_info[entry['meshIdx']]['filename'])
                
                mesh_library = import_cgf_models_batch(unique_paths, prefs, context)
                count = create_brush_instances(brush_entries, brush_info, mesh_library, context)
                
                if props.remove_nodraw:
                    bpy.ops.aion.remove_nodraw()
                
                elapsed = time.time() - start
                print(f"[Timer] Brush import: {elapsed:.2f}s ({count} instances)")
                self.report({'INFO'}, f"Imported {count} brush instances in {elapsed:.1f}s")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

class AION_OT_ImportMissionObjects(Operator, ImportHelper):
    bl_idname = "aion.import_mission_objects"
    bl_label = "Import Mission Objects"
    bl_description ="Import Aion dynamic objects from a folder"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        folder_path = self.filepath if os.path.isdir(self.filepath) else os.path.dirname(self.filepath)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Invalid directory")
            return {'CANCELLED'}

        props = context.scene.aion_importer_props
        prefs = context.preferences.addons[__name__].preferences
        start = time.time()
        
        with PakFileManager(folder_path) as pak_manager:
            mission_file = None
            for fname in pak_manager.get_file_list():
                if fname.startswith('mission_') and fname.endswith('.xml'):
                    mission_file = fname
                    break
            
            if not mission_file:
                self.report({'ERROR'}, "Mission XML not found")
                return {'CANCELLED'}

            mission_bytes = pak_manager.read_file(mission_file)
            if not mission_bytes:
                self.report({'ERROR'}, "Could not read mission XML")
                return {'CANCELLED'}

            try:
                mission_objects = parse_mission_xml(io.BytesIO(mission_bytes))
                
                unique_paths = {obj.model_path for obj in mission_objects if obj.model_path}
                mesh_library = import_cgf_models_batch(unique_paths, prefs, context)
                count = create_mission_instances(mission_objects, mesh_library, context)
                
                if props.remove_nodraw:
                    bpy.ops.aion.remove_nodraw()
                
                elapsed = time.time() - start
                print(f"[Timer] Mission import: {elapsed:.2f}s ({count} instances)")
                self.report({'INFO'}, f"Imported {count} mission objects in {elapsed:.1f}s")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

class AION_OT_ImportLights(Operator, ImportHelper):
    bl_idname = "aion.import_lights"
    bl_label = "Import Lights"
    bl_description ="Import Aion deferred lights from a folder"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ""
    filter_glob: StringProperty(default="", options={'HIDDEN'})

    def execute(self, context):
        folder_path = self.filepath if os.path.isdir(self.filepath) else os.path.dirname(self.filepath)
        if not os.path.isdir(folder_path):
            self.report({'ERROR'}, "Invalid directory")
            return {'CANCELLED'}

        props = context.scene.aion_importer_props
        start = time.time()
        
        with PakFileManager(folder_path) as pak_manager:
            mission_file = None
            for fname in pak_manager.get_file_list():
                if fname.startswith('mission_') and fname.endswith('.xml'):
                    mission_file = fname
                    break
            
            if not mission_file:
                self.report({'ERROR'}, "Mission XML not found")
                return {'CANCELLED'}

            mission_bytes = pak_manager.read_file(mission_file)
            if not mission_bytes:
                self.report({'ERROR'}, "Could not read mission XML")
                return {'CANCELLED'}

            try:
                light_objects = parse_lights_xml(io.BytesIO(mission_bytes))
                count = create_light_instances(
                    light_objects, 
                    context,
                    base_power=props.light_base_power,
                    radius_multiplier=props.light_radius_multiplier
                )
                
                elapsed = time.time() - start
                print(f"[Timer] Lights import: {elapsed:.2f}s ({count} lights)")
                self.report({'INFO'}, f"Imported {count} lights in {elapsed:.1f}s")
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}

class AION_PT_ImporterPanel(Panel):
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
        box.prop(props, "remove_nodraw")
        box.prop(props, "light_base_power")
        box.prop(props, "light_radius_multiplier")
        box.operator(AION_OT_ImportMapFolder.bl_idname)
        
        box = layout.box()
        box.label(text="Individual Imports", icon='IMPORT')
        box.prop(props, "remove_nodraw")
        box.operator(AION_OT_ImportHeightmap.bl_idname)
        box.operator(AION_OT_ImportVegetation.bl_idname)
        box.operator(AION_OT_ImportBrushes.bl_idname)
        box.operator(AION_OT_ImportMissionObjects.bl_idname)
       
        box.operator(AION_OT_ImportLights.bl_idname)
        
        box = layout.box()
        box.label(text="Utilities", icon='TOOL_SETTINGS')
        box.operator(AION_OT_RemoveNoDraw.bl_idname)

classes = (
    AionImporterPreferences,
    AionImporterProperties,
    AION_OT_RemoveNoDraw,
    AION_OT_ImportMapFolder,
    AION_OT_ImportHeightmap,
    AION_OT_ImportVegetation,
    AION_OT_ImportBrushes,
    AION_OT_ImportMissionObjects,
    AION_OT_ImportLights,
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