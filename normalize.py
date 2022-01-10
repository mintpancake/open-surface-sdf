import numpy as np
from common_tools.geometry import TriangularMesh
from common_tools.utils import draw_colored_points_to_obj
from pysdf import SDF
from new_dataset import ShapeSample3D
from output_geometry import marching_cube_np

if __name__ == "__main__":
    mesh = TriangularMesh("./data/3dmodel/TShirtNoCoat.obj")
    mesh.normalize()
    mesh.save_mesh("./data/3dmodel/normalized_TShirtNoCoat.obj")
