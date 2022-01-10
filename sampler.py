import numpy as np
from common_tools.geometry import TriangularMesh
from common_tools.utils import draw_colored_points_to_obj
from pysdf import SDF
from new_dataset import ShapeSample3D
from output_geometry import marching_cube_np

if __name__ == "__main__":
    # mesh = TriangularMesh("./data/3dmodel/skirt_2.obj")
    # mesh.normalize()
    # mesh.save_mesh("normalized_skirt_2.obj")

    dataset = ShapeSample3D("./data/3dmodel/normalized_skirt_2.obj", num_sample_pts=200000)
    
    # draw_colored_points_to_obj("output.obj", vertices=dataset.samples, scalars_for_color=dataset.sdf)

    # mesh = TriangularMesh("./data/3dmodel/normalized_skirt_2.obj")
    # marching_cube_np("mc_mesh_skirt_2.obj", mesh.sdf, 512)

