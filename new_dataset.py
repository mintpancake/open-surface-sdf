import numpy as np
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from common_tools.geometry import TriangularMesh
from common_tools.utils import draw_colored_points_to_obj
import time
from multiprocessing import Process, Pool
import os
from pysdf import SDF

def save_data_to_txt(filename, data):
    np.savetxt(filename, data, fmt='%f', delimiter=' ')
    print(f"save to {filename}")


class ShapeSample3D():
    def __init__(self, filepath, num_sample_pts, sample_noise=0.05, uniform_sampling=True):
        
        off_surface_ratio = 2.0 ## #off_surface / #on_surface
        uniform_ratio = 0.1 ## #uniform / on_surface

        print(filepath)
        self.mesh = TriangularMesh(filepath)
        self.vertices = self.mesh.vertices

        ## surface samples
        surface_samples = self.mesh.sample_triangles(int(num_sample_pts))

        ## off surface samples
        noise = np.random.randn(int(off_surface_ratio), num_sample_pts, 3) ## standard normal dist
        spatial_samples = surface_samples[None, ...] + noise*sample_noise
        spatial_samples = spatial_samples.reshape(-1, 3)

        self.samples = np.concatenate([surface_samples, spatial_samples], axis=0)

        if uniform_sampling:
            print("has uniform background")
            uniform_samples = np.random.randn(int(num_sample_pts*uniform_ratio), 3)
            self.samples = np.concatenate([self.samples, uniform_samples], axis=0)
        
        self.sdf = self.mesh.sdf_func(self.samples)
        self.sdf[:num_sample_pts] = 0 
        self.sdf = np.clip(self.sdf, -1.0, 1.0)       

        ##
        self.samples = torch.FloatTensor(self.samples)
        self.sdf = torch.FloatTensor(self.sdf)

        on_surface_1 = torch.ones(int(num_sample_pts))
        on_surface_0 = torch.zeros(int(num_sample_pts*(off_surface_ratio+uniform_ratio)))
        self.on_surface = torch.cat([on_surface_1, on_surface_0])
        self.data = torch.cat([self.samples, self.sdf.unsqueeze(1), self.on_surface.unsqueeze(1)], axis=-1)


    def get_data(self):
        return self.data


    def __len__(self):
        return int(self.samples.shape[0])


    def __getitem__(self, idx):
        out = {"samples": self.samples[idx, :], "sdfs": self.sdf[idx], "on_surface": self.on_surface[idx]}
        return out


if __name__ == "__main__":

    dataset = ShapeSample3D("./data/3dmodel/normalized_ShirtNoCoat1.obj", 50000)
    data = dataset.get_data().numpy()

    np.savetxt("./data/ShirtNoCoat1.txt", data)