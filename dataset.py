from numpy.random import rand
import torch
import numpy as np
from common_tools.geometry import Circle, TriangularMesh, plot_sdf_using_opencv

"""
circle function:
(x-cx)^2 + (y-cy)^2 = r^2
"""
class CircleSample(torch.utils.data.Dataset):
    def __init__(
        self,
        center_x:float,
        center_y:float,
        radius:float,
        num_base_pts=1e3,
        num_sample_pts=1e5,
        train_per_shape = False
    ):
        self.shape = Circle(np.float_([center_x, center_y]), radius)
        self.train_per_shape = train_per_shape
        
        ## get surface points
        rands = np.random.rand(int(num_base_pts))*2.0*3.142 ## map uniform dist [0, 1) to [0, 2Pi)
        x = center_x + radius*np.cos(rands)
        y = center_y + radius*np.sin(rands)
        self.surfpts = np.stack([x, y], axis=0) ## [2, N]
        
        ## generate sample points
        noise = np.random.randn(2, int(num_sample_pts)) ## standard normal dist
        random_indices = np.random.choice(range(0,int(num_base_pts)), int(num_sample_pts))
        self.samples = self.surfpts[:,random_indices] + noise*0.1
        self.sdf = self.shape.sdf(self.samples)
        self.sdf = np.clip(self.sdf, -1.0, 1.0)
        
        ## if we need this one
        self.data = np.concatenate([self.samples, np.expand_dims(self.sdf, axis=0)], axis=0)

    def draw(self):
        plot_sdf_using_opencv(self.shape.sdf, device=None, filename=self.filename)

    def __len__(self):
        if self.train_per_shape:
            return 1
        else:
            return self.data.shape[-1]

    def __getitem__(self, idx):
        if self.train_per_shape:
            return self.samples, self.sdf
        else:
            pt = torch.from_numpy(self.samples[:, idx]).to(torch.float)
            sdf = torch.Tensor([self.sdf[idx]]).to(torch.float)
            return pt, sdf



## TODO: use farthest sampling to uniformly sample the shapes.
class Polygon3DSample():
    def __init__(
        self,
        filepath,
        num_sample_pts,
        sample_noise
    ):
        self.shape = TriangularMesh(filepath)
        ## samples are of shape [N, 3]
        self.samples = self.shape.sample_triangles(int(num_sample_pts))
        ##TODO: sample the polygonal mesh
        noise = np.random.randn(int(self.samples.shape[0]), 3) ## standard normal dist
        self.samples += noise*sample_noise
        ## generate sample points
        self.sdf = self.shape.sdf(self.samples)
        self.sdf = np.clip(self.sdf, -1.0, 1.0)        
        ## 
        self.data = np.concatenate([self.samples, np.expand_dims(self.sdf, axis=-1)], axis=-1)
        self.vertices = self.shape.vertices

        self.samples = torch.FloatTensor(self.samples)
        self.sdf = torch.FloatTensor(self.sdf)


    def __len__(self):
        return int(self.samples.shape[0])


    def __getitem__(self, idx):
        return self.samples[idx, :], self.sdf[idx]



class SDFData():
    def __init__(
        self,
        filepath,
        use_surf_grad=True, 
    ):
        self.use_surf_grad = use_surf_grad        

        data = self.load(filepath)

        ## point coordinates
        self.samples = data[:,0:3]

        ## gt sdf
        self.sdfs = data[:,3:4] ## keep the dim

        ## gradient
        if use_surf_grad:
            assert data.shape[1] > 4
            self.gradients = data[:,4:7]

        ## on_surface label: true on, false off
        self.on_surface = data[:,-1:]
        self.on_surface=(self.on_surface==1)


    def __len__(self):
        return int(self.samples.shape[0])

    def __getitem__(self, idx):
        
        out = {
            "samples": self.samples[idx, :],
            "sdfs": self.sdfs[idx, :],
            "on_surface": self.on_surface[idx, :],
        }
        
        if self.use_surf_grad:
            return {**out, "gradients":self.gradients[idx, :]}
        else:
           return out

    def load(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]
        data = np.array(lines, dtype=np.float32)
        return torch.from_numpy(data)
