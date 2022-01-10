import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from torch.autograd import grad

class Net(nn.Module):

    def __init__(self, skip_link_at = 4):
        super(Net, self).__init__()
        self.skip_link_at = skip_link_at
        num_neurons = 100
        self.mlp_list = nn.ModuleList()
        self.actv_list = nn.ModuleList()
        self.mlp_list.append(nn.Linear(2, num_neurons))
        self.actv_list.append(nn.ReLU(inplace=False))
        for i in range(1,7):
            if i == self.skip_link_at:
                self.mlp_list.append(nn.Linear(num_neurons+2, num_neurons))
            else:
                self.mlp_list.append(nn.Linear(num_neurons, num_neurons))
            self.actv_list.append(nn.ReLU(inplace=False))
        self.mlp_list.append(nn.Linear(num_neurons, 1))
        self.actv_list.append(nn.Tanh())
        assert(len(self.mlp_list)==len(self.actv_list))
        print(self)

    def forward(self, x):
        pt = deepcopy(x)
        for idx, layer in enumerate(zip(self.mlp_list, self.actv_list)):
            fc, actv = layer
            if idx == self.skip_link_at:
                x = torch.cat([x, pt], dim=-1)
            x = actv(fc(x))
        return x


class ImplicitNet(nn.Module):
    activation_list = ['relu', 'elu', 'leaky', 'sp']
    def __init__(self, dim, num_neurons=100, num_layers = 8, weight_norm=False, skip_link=False, activation='relu'):
        super(ImplicitNet, self).__init__()

        ## parameters
        self.num_layers = num_layers
        self.skip_link = skip_link
        self.num_neurons = num_neurons
        self.dim = dim

        self.mlp_list = nn.ModuleList()
        self.actv_list = nn.ModuleList()
        self.mlp_list.append(nn.Linear(dim, num_neurons))

        assert activation in self.activation_list
        if activation == 'elu':
            actv_fn = nn.ELU(inplace=False)
        elif activation == 'leaky':
            actv_fn = nn.LeakyReLU(inplace=False)
        elif activation == 'sp':
            actv_fn = nn.Softplus(beta=100)
        else:
            actv_fn = nn.ReLU(inplace=False)
        
        self.actv_list.append(actv_fn)

        ## hidden layers
        for i in range(0, num_layers-1):

            ## skip link
            if i == num_layers//2 and self.skip_link:
                self.actv_list.append(actv_fn)
                lin = nn.Linear(num_neurons+2, num_neurons)
            elif i == num_layers - 2:
                self.actv_list.append(nn.Tanh())
                lin = nn.Linear(num_neurons, 1)
            else:
                self.actv_list.append(actv_fn)
                lin = nn.Linear(num_neurons, num_neurons)

            self.init_weights(lin)
            # self.geo_init_weights(lin, num_neurons, last=False)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            self.mlp_list.append(lin)

        ## output layer
        print(self)
        assert(len(self.mlp_list)==len(self.actv_list))

    def forward(self, x):
        pt = torch.clone(x)
        for idx, layer in enumerate(zip(self.mlp_list, self.actv_list)):
            fc, actv = layer
            ## skip link
            if idx == self.num_layers//2 and self.skip_link:
                x = torch.cat([x, pt], dim=-1)
            x = actv(fc(x))
        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def grad_compute(self, x, outputs=None, detach=False):
        if outputs is None:
            outputs = self(x)
        d_points = torch.ones_like(outputs, requires_grad=False, device=x.device)
        ori_grad = grad(
            outputs=outputs,
            inputs=x,
            grad_outputs=d_points,  
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )
        if detach:
            return ori_grad[0].detach()
        return ori_grad[0]


class LossFunction():
    def __call__(self, pred_sdf, gt_sdf):
        reg_loss = F.l1_loss(pred_sdf.reshape(-1), gt_sdf.reshape(-1), reduction='mean')
        return reg_loss