from os import write
from matplotlib.pyplot import axis, bar
import numpy as np
import cv2
import torch
import trimesh
import matplotlib
from matplotlib import cm
from pysdf import SDF as sdf_func

##
class Geometry:
    EPS = 1e-12
    def distance_from_segment_to_point(a, b, p):
        ans = min(np.linalg.norm(a - p), np.linalg.norm(b - p))
        if (np.linalg.norm(a - b) > Geometry.EPS 
            and np.dot(p - a, b - a) > Geometry.EPS 
            and np.dot(p - b, a - b) > Geometry.EPS):
            ans = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
        return ans

    def distance_from_segment_to_points(a, b, ps):

        a, b = a.reshape(1,2), b.reshape(1,2)

        ## distance from a point to either endpoints
        stack_ab = np.stack([a, b], axis=0)
        dist_ps_to_end = np.sqrt( ((stack_ab.reshape(2,1,2) - ps.reshape(1,-1,2))**2).sum(axis=-1) )
        dist_ps_to_end = np.min(dist_ps_to_end, axis=0)

        ## find root
        t = np.matmul((ps-a), (b-a).transpose()) / np.linalg.norm(b-a)**2
        roots = a + t*(b-a) ## root
        dist_ps_to_seg = np.sum(((ps-roots)**2), axis=-1)
        dist_ps_to_seg = np.sqrt(dist_ps_to_seg)

        ## find those roots not in the segment and use their distances to endpoint instead
        mask = np.bitwise_or(t<Geometry.EPS, t>1.0).squeeze()
        dist_ps_to_seg[mask] = dist_ps_to_end[mask]
        return dist_ps_to_seg



## parent class of other following shapes
class Shape:
    ## negative inside; positive outside
    def sdf(self, p):
        pass
    
    
class Circle(Shape):
    def __init__(self, c, r):
        self.c = c
        self.r = r
    
    def sdf(self, p):
        if len(p.shape) == 2: ## 2d array
            c = self.c.reshape(1, 2)
            return np.linalg.norm(p - c, axis=-1) - self.r
        elif len(p.shape) == 1: ## a point
            c = self.c
            return np.linalg.norm(p - c) - self.r
        else:
            raise NotImplementedError
    
    
class Polygon(Shape):
    
    def __init__(self, v):
        self.v = v
    
    def sdf(self, p):
        if len(p.shape) == 1:
            return -self.distance(p) if self.point_is_inside(p) else self.distance(p)
        else:
            dist = self.array_distance(p)
            sign = np.ones(dist.shape)
            bool_ind = self.points_are_inside(p)
            sign[bool_ind] = -1.0
            return sign*dist

    def point_is_inside(self, p):
        angle_sum = 0
        L = len(self.v)
        for i in range(L):
            a = self.v[i]
            b = self.v[(i + 1) % L]
            angle_sum += np.arctan2(np.cross(a - p, b - p), np.dot(a - p, b - p))
        return abs(angle_sum) > 1

    def points_are_inside(self, p):
        L = len(self.v)
        for i in range(L):
            a = self.v[i].reshape(1,2)
            b = self.v[(i + 1) % L].reshape(1,2)

            cp_res = np.cross((a-p), (b-p))
            dp_res = np.sum(((a-p)*(b-p)), axis=-1)

            if i == 0:
                angle_sum = np.arctan2(cp_res, dp_res)
            else:
                angle_sum += np.arctan2(cp_res, dp_res)
        return abs(angle_sum) > 1

    ## return all segments in the polygon to a point p             
    def distance(self, p):
        ans = Geometry.distance_from_segment_to_point(self.v[-1], self.v[0], p)
        for i in range(len(self.v) - 1):
            ans = min(ans, Geometry.distance_from_segment_to_point(self.v[i], self.v[i + 1], p))
        return ans


    def array_distance(self, p):
        dist = Geometry.distance_from_segment_to_points(self.v[-1], self.v[0], p)
        for i in range(len(self.v) - 1):
            ans = Geometry.distance_from_segment_to_points(self.v[i], self.v[i+1], p)
            stacked = np.stack([ans, dist], axis=0)
            dist = np.min(stacked, axis=0)
        return dist


def plot_sdf_using_opencv(sdf_func, device, filename=None, is_net=False, res=100):
    # See https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    
    ## this is the rasterization step that samples the 2D domain as a regular grid
    COORDINATES_LINSPACE = np.linspace(-7, 7, res)
    y, x = np.meshgrid(COORDINATES_LINSPACE, COORDINATES_LINSPACE)
    if not is_net:
        z = [[sdf_func(np.float_([x_, y_])) 
                for y_ in  COORDINATES_LINSPACE] 
                for x_ in COORDINATES_LINSPACE]
    else:
        ## convert []
        z = [[sdf_func(torch.Tensor([x_, y_]).to(device)).detach().cpu().numpy() 
                for y_ in  COORDINATES_LINSPACE] 
                for x_ in COORDINATES_LINSPACE]
    z = np.float_(z)
        
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    # z_min = -np.abs(z).max()
    # z = np.clip(z, a_max=0.0, a_min=z_min)
    # z_min = -1.0
    # z_max = 1.0
    z = (z - z_min) / (z_max - z_min) * 255
    z = np.uint8(z)
    z = cv2.applyColorMap(z, cv2.COLORMAP_JET)
    if filename is None:
        filename = "tmp_res.png"
    print(filename)
    cv2.imwrite(filename, z)


###################################################
### 3D data
###################################################

def read_obj_file(filename):
    ## read obj file
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]

    vertices = []
    faces = []
    for l in lines:
        ## vertices
        if len(l) > 0 and l[0] == 'v':
            vertices.append([float(l[1]), float(l[2]), float(l[3])])
        ## faces
        elif len(l) > 0 and l[0] == 'f':
            faces.append([int(l[1]), int(l[2]), int(l[3])])

    vertices = np.array(vertices)
    faces = np.array(faces) - 1

    return vertices, faces


def write_obj_file(filename, V, F=None, C=None, vid_start=1):
    with open(filename, 'w') as f:
        if C is not None:
            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
                
        if F is not None:
            for Fi in F:
                f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start}\n")



class TriangularMesh(Shape):
    def __init__(self, obj_mesh_file):
        self.vertices = []
        self.faces = []
        self.vertices, self.faces = read_obj_file(filename=obj_mesh_file)
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        self.sdf_func = sdf_func(self.mesh.vertices, self.mesh.faces, False)

    def sdf(self, p):
        return self.sdf_points_to_triangles(p)

    def sdf_points_to_triangles(self, ps):
        # sdf = trimesh.proximity.signed_distance(self.mesh, ps)
        sdf = self.sdf_func(ps)
        return sdf


    def areas_of_triangles(self, normalized=True):
        face_vertices = self.prepare_face_vertices()
        areas = 0.5*np.cross(face_vertices[:,2] - face_vertices[:,0], face_vertices[:,1] - face_vertices[:,0])
        areas = np.sqrt(np.sum(areas * areas, axis=1))        
        if normalized:
            areas = areas / areas.max()
        return areas


    def sample_triangles(self, num_samples):
        # assert isinstance(num_per_face, int)
        # face_vertices = self.prepare_face_vertices()
        # num_faces = face_vertices.shape[0]

        # baryc_rand = np.random.rand(num_faces, num_per_face, 3)
        # baryc_rand = baryc_rand / np.sum(baryc_rand, axis=-1, keepdims=True)

        # ## [F, N, B3] * [F, V3, C3] -> [F, N, C3]
        # samples = np.matmul(baryc_rand,face_vertices)
        # return samples.reshape(-1, 3) ## [FN, C3]
        # # write_obj_file("samples.obj", samples.reshape(-1, 3))
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        samples, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
        return samples
        
    def translate(self, t):
        self.vertices += t
        self.__update()

    def scale(self, scale_factor=1.0):
        self.vertices *= scale_factor
        self.__update()


    def prepare_face_vertices(self):
        face_vertices = self.vertices[self.faces.reshape(-1),:]
        face_vertices = face_vertices.reshape(-1, 3, 3)
        return face_vertices


    def normalize(self):
        max = np.max(self.vertices, axis=0)
        min = np.min(self.vertices, axis=0)
        diag_len = np.linalg.norm(max-min)
        center = np.mean(self.vertices, axis=0)
        self.vertices = (self.vertices - center)/diag_len
        self.__update()
        
    
    def save_mesh(self, fname):
        self.mesh.export(fname)

    def __update(self):
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        self.sdf_func = sdf_func(self.mesh.vertices, self.mesh.faces)


if __name__ == "__main__":
    pObj = TriangularMesh("../data/rocker-arm.obj")
    pObj.normalize()

    # pObj.areas_of_triangles()
    # samples = pObj.sample_triangles(5)

    # noise = np.random.randn(int(samples.shape[0]), 3) ## standard normal dist
    # samples += noise*0.1

    # sdf = pObj.sdf_points_to_triangles(samples)

    # norm = matplotlib.colors.Normalize(vmin=min(sdf), vmax=max(sdf), clip=True)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    # colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(sdf)]
    # write_obj_file("colored_samples.obj", samples.reshape(-1, 3), C=colors)




