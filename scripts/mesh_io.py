import mitsuba as mi

from typing import List

mi.set_variant("cuda_ad_rgb")

scene = mi.load_file("data/matpreview/scene_v3.xml")
meshes : List[mi.Mesh] = scene.shapes()
for i, mesh in enumerate(meshes):
    mesh.write_ply(f"mesh_{i}.ply")