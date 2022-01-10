import igl
from numpy.lib.function_base import delete
import scipy as sp
import numpy as np


def points_on_line(p, l):
    res = np.zeros(len(p))
    ab = np.linalg.norm(l[0]-l[1])
    ap = np.linalg.norm(p-l[0], axis=1)
    bp = np.linalg.norm(p-l[1], axis=1)
    apb = ap+bp
    diff = np.abs(apb-ab)
    res[diff < 1e-5] = 1
    return res


class Mesh(object):
    def __init__(self, file):
        self.v, self.f = igl.read_triangle_mesh(file)
        self.boundary = igl.boundary_facets(self.f)


class Trimer(object):
    def __init__(self, tr_mesh_file, gt_mesh_file, save_name):
        self.tr_mesh = Mesh(tr_mesh_file)
        self.gt_mesh = Mesh(gt_mesh_file)
        self.save_name = save_name
        self.run()

    def run(self):
        sqrD, closest_faces, closest_points = igl.point_mesh_squared_distance(
            self.tr_mesh.v, self.gt_mesh.v, self.gt_mesh.f)
        outlier_pos = self.if_on_edge(closest_points, self.gt_mesh.boundary)
        outlier_pos[sqrD > 1e-4] = 1
        self.filter(outlier_pos)
        self.save()

    def if_on_edge(self, points, edges):
        res = np.zeros(len(points))
        for i in range(len(edges)):
            line = np.array(
                [self.gt_mesh.v[edges[i, 0]], self.gt_mesh.v[edges[i, 1]]])
            res[points_on_line(points, line) == 1] = 1
        return res

    def filter(self, v_pos):
        removed_v_idx = np.where(v_pos == 1)[0]
        v_idx_new2old = np.arange(len(self.tr_mesh.v))
        v_idx_new2old = v_idx_new2old[v_pos == 0]
        v_idx_old2new = -np.ones(len(self.tr_mesh.v))
        for new, old in enumerate(v_idx_new2old):
            v_idx_old2new[old] = new
        self.tr_mesh.v = self.tr_mesh.v[v_pos == 0]
        f_pos = np.zeros(len(self.tr_mesh.f))
        for i, face in enumerate(self.tr_mesh.f):
            if v_idx_old2new[face[0]] == -1 or v_idx_old2new[face[1]] == -1 or v_idx_old2new[face[2]] == -1:
                f_pos[i] = 1
            else:
                self.tr_mesh.f[i][0] = v_idx_old2new[face[0]]
                self.tr_mesh.f[i][1] = v_idx_old2new[face[1]]
                self.tr_mesh.f[i][2] = v_idx_old2new[face[2]]
        self.tr_mesh.f = self.tr_mesh.f[f_pos == 0]

    def save(self):
        igl.write_triangle_mesh(
            self.save_name, self.tr_mesh.v, self.tr_mesh.f)


if __name__ == "__main__":
    Trimer('saved/ShirtNoCoat1_surface=0/output3dshape_990_0.obj',
           'data/3dmodel/normalized_ShirtNoCoat1.obj',
           'trimed/trimed_ShirtNoCoat1.obj')
