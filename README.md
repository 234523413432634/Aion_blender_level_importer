This is a blender addon to import AION online levels into the blender.

# Extracting assets

1. Download the batch file that extracts the needed assets from your aion directory. The .bat file is in plain text, the "pak2zip.exe" is taken from AION Encdec, and "7z.exe" is an official 7zip binary. 
If you still feel uncomfortable running this bat, you can read what it does and do these steps manually.
2. Put the "createRes.bat" along with "pak2zip.exe" and "7z.exe" into your aion directory (folder with bin64, Data, L10N etc.) and run the bat file. After running the bat, there should be "AionResources" folder created (5 - 14 GB in size based on the game version).

# Preparing blender

1. Download and install [blender 3.6](https://download.blender.org/release/Blender3.6/blender-3.6.0-windows-x64.zip). Other versions of blender will not work.
2. Download and install this fork of [io_scene_cgf](https://github.com/234523413432634/io_scene_cgf) and Aion_blender_level_importer as blender addons.
3. In the aion map importer preferences, set the path to AionResources, which we created in the previous steps.

# importing a level & tips

Now that everything is set up, press "n" to open the side panel, select the "Aion importer" tab in it.

Here, you can import a full level (slow, only use for small levels or dungeons) or individual parts of the level.

After pressing the import button, select the path to the map folder in the "AionResources" folder. Example: D:\AionResources\Levels\lf1. After selecting this folder, the import process will begin.

The import time depends on your CPU and the map size. Importing dungeons should take a minute or two, while the large open maps could easily take 10 - 15 minutes.

If you want to import larger maps, open multiple blender instances and import individual parts of the level in each of them, then copy the imported objects into one blender instance.

For example, I launch blender.exe 4 times:

In the first blender instance - import heightmap

2nd instance - import brushes

3rd instance - import objects

4th instance import mission objects

Once all imports are complete, I copy the objects from each instance into the first one.

To improve viewport performance, I also join all vegetation into one mesh (A, Ctrl+J in the second blender instance) and then import that one mesh into the main blender instance. It's an optional step.

You CAN use the import full level button for large maps, but that would take 2 - 5 longer to import compared to the approach I've mentioned above.

# Limitations & bugs

1. Blender 3.6 only. This is an io_scene_cgf limitation. If the author of the addon decides to update it, then the map importer could easily be brought to newer versions as well. Though nothing stops you from importing a level into blender 3.6, saving the scene, and then opening that scene in newer versions.
2. Some (VERY few) objects are imported in a visually broken state. Once again, likely a io_scene_cgf issue. CGF format is very complex, so it is understandable that there will be a few wonky meshes.
3. The terrain material blending and tiling are a bit different from what they are in game. It's not that noticeable  and might be fixed in the future (or not).
