from common_tools.geometry import write_obj_file
import os
import json
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

# 参数设定模仿IGR
# inputs：要对什么求偏导，此处为网络输入，（N，3）的点序列
# outputs：被求导，此处为网络输出，（N，1）
# grad_outputs：外部梯度
# create/retain_graph：生成、保留计算图（用于后续bp，而不是单纯输出梯度“值”）
# only_inputs：只保留和inputs相关的grad，其他的梯度不会保留
def myGradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device='cpu').cuda()
    ori_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,  # 对于多维输出，传入与output shape一致的外部梯度
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )  # 输出为tuple，需要取出其中的tensor
    points_grad = ori_grad[0][:, -2:]  # points_grad的shape和网络input一致
    return points_grad

# # put points on cuda 
# # input points are tensor in the shape of [x,2] (x is the points number)
# # currently 2D form: compute the average distance from surface points to their nearest points
# def compute_chamfer_distance(surface_points,zero_points):
    
#     surface_points=surface_points.reshape([surface_points.shape[0],1,2])
#     surface_points=surface_points.repeat(1,zero_points.shape[0],1)
    
#     zero_points=zero_points.reshape(1,zero_points.shape[0],2)
#     zero_pointsb=zero_points.repeat(surface_points.shape[0],1,1)
    
#     # distance matrix for all surface points
#     dis=surface_points-zero_points
#     dis=torch.abs(dis)
#     dis=torch.norm(dis,dim=-1)
#     #print(dis)
    
#     # get min distance
#     min_dis,_=torch.min(dis,dim=1)
#     total_dis=min_dis.sum()
#     return total_dis/surface_points.shape[0]

def print_shape(x):
    print(x.shape)

def scatter_plot_pts_with_mask(pts, mask=None):
    plt.scatter(pts[:,0], pts[:,1], c=mask)
    plt.colorbar()
    plt.show()


def read_json(fname):
    with open(fname) as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname) as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

##TODO: generalize the logger to any data format
class SimpleLogger(object):
    def __init__(self, res_dir, kwarg_list, make_file=True):
        self.make_file = make_file
        self.res_dir = res_dir
        filename = "logger.txt" ## add datetime stamp
        if make_file:
            if not os.path.exists(self.res_dir):
                os.makedirs(self.res_dir)
            self.filename = os.path.join(self.res_dir, filename)
            with open(self.filename, 'w') as f:
                f.write("simple logger\n")

        self.kwarg_list = kwarg_list
        self.content = {}
        for kwarg in kwarg_list:
            self.content[kwarg] = []

    def logging(self, content):
        with open(self.filename, 'a') as f:
            if content is OrderedDict:
                for k, v in content.items():
                    f.write(k, v)
            else:
                f.write(content)

    def flush_out(self):
        with open(self.filename, 'a') as f:
            line = str()
            for k, v in self.content.items():
                line += (f"{k}: {v[-1]}; ")
            line += "\n"
            print(line)
            f.write(line)

    def print_out(self, line):
        with open(self.filename, 'a') as f:
            f.write(line)

    def update(self, loss, kwarg:str):
        assert kwarg in self.kwarg_list
        self.content[kwarg].append(loss)

    def get_best(self, kwarg, best='min'):
        assert kwarg in self.kwarg_list
        if best == 'min':
            best_val = min(self.content[kwarg])
        else:
            best_val = max(self.content[kwarg])

        return self.content[kwarg].index(best_val)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_colored_points_to_obj(filename, vertices, scalars_for_color):
    print(f"draw colored points to obj file :{filename}")
    assert len(vertices.shape) == 2
    assert vertices.shape[-1] == 3
    norm = matplotlib.colors.Normalize(vmin=min(scalars_for_color), vmax=max(scalars_for_color), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(scalars_for_color)]
    write_obj_file(filename, vertices.reshape(-1, 3), C=colors)



def catch_nan(tensor_data, name=None):
    if (torch.isnan(tensor_data)).any():
        if name is not None:
            print(f"catch nan in {name}")
        return True
    else:
        return False