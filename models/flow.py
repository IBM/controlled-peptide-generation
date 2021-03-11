import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_flow(flow_type, flow_layers, z_dim):
    if flow_type == 'planar':
        return PlanarFlow(flow_layers, z_dim)
    elif flow_type == 'radial':
        return RadialFlow(flow_layers, z_dim)
    elif flow_type == 'alternating':
        return AlternatingFlow(flow_layers, z_dim)
    else:
        raise ValueError('Please use either planar, radial, or alternating flow.')


class Flow(nn.Module):
    def __init__(self, flow_layers):
        super(Flow, self).__init__()
        self.flow = flow_layers

    def forward(self, z, train=True):
        return self._run_flow(z, train)

    def _run_flow(self, z, train=True):
        raise NotImplementedError


class PlanarFlow(Flow):
    def __init__(self, flow_layers, z_dim):
        super(PlanarFlow, self).__init__(flow_layers)
        self.z_dim = z_dim
        self.planar_weight = nn.ParameterList([nn.Parameter(torch.Tensor(1, z_dim)) for i in range(flow_layers)])
        self.planar_bias = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(flow_layers)])
        self.planar_scale = nn.ParameterList([nn.Parameter(torch.Tensor(1, z_dim)) for i in range(flow_layers)])
        for i in range(flow_layers):
            self.planar_weight[i].data.uniform_(-0.01, 0.01)
            self.planar_scale[i].data.uniform_(-0.01, 0.01)
            self.planar_bias[i].data.uniform_(-0.01, 0.01)

    def _run_flow(self, z, train=True):
        loss = torch.zeros(z.size(0)).to(z.device)
        for i in range(self.flow):
            # Maintain invertibility
            margin = torch.mm(self.planar_scale[i], self.planar_weight[i].t()).item()
            if margin < -1:
                component = torch.Tensor([-1 + np.log(1 + np.e ** margin) - margin]).to(z.device)
                self.planar_scale[i].data += component * self.planar_weight[i] / self.planar_weight[i].norm(2)

            activation = F.linear(z, self.planar_weight[i], self.planar_bias[i])
            z = z + self.planar_scale[i] * F.tanh(activation)
            if train:
                psi = (1 - F.tanh(activation) ** 2) * self.planar_weight[i]
                det_grad = 1 + torch.mm(psi, self.planar_scale[i].t())
                loss += torch.log(det_grad.abs().squeeze(1) + 1e-7)  # for numerical stability
        z.flowed = True
        if train:
            loss = loss.mean(0)  # batch averaging
            return z, loss
        else:
            return z


class RadialFlow(Flow):
    def __init__(self, flow_layers, z_dim):
        super(RadialFlow, self).__init__(flow_layers)
        self.z_dim = z_dim
        self.radial_initial = nn.ParameterList([nn.Parameter(torch.Tensor(1, z_dim)) for i in range(flow_layers)])
        self.radial_alpha = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(flow_layers)])
        self.radial_beta = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(flow_layers)])
        for i in range(flow_layers):
            self.radial_initial[i].data.uniform_(-0.01, 0.01)
            self.radial_alpha[i].data.uniform_(0.01, 1.0)
            self.radial_beta[i].data.uniform_(-0.01, 0.01)

    def _run_flow(self, z, train=True):
        loss = torch.zeros(z.size(0)).to(z.device)
        for i in range(self.flow):
            # Maintain invertibility
            if self.radial_beta[i].item() < -self.radial_alpha[i].item():
                self.radial_beta[i].data = -self.radial_alpha[i] + torch.log(1 + np.e ** self.radial_beta[i])

            radius = z - self.radial_initial[i]
            activation = (1 / (self.radial_alpha[i] + radius.norm(p=2, dim=1))).unsqueeze(1)
            z = z + self.radial_beta[i] * activation * radius
            if train:
                diagonal = (1 + self.radial_beta[i] * activation) ** (self.z_dim - 1)
                det_grad = (diagonal * (1 + self.radial_beta[i] * activation + self.radial_beta[i] * (
                    -activation ** 2) * radius.norm(2)))
                loss += torch.log(det_grad.abs().squeeze(1) + 1e-7)  # for numerical stability
        z.flowed = True
        if train:
            loss = loss.mean(0)  # batch averaging
            return z, loss
        else:
            return z


class AlternatingFlow(Flow):
    def __init__(self, flow_layers, z_dim):
        '''
        Note: There will be parameters not used, because we are initializing self.flow number of planar and radial flow variables
              This makes it programmatically easier to change the ordering of alternation
        '''
        super(AlternatingFlow, self).__init__(flow_layers)
        self.z_dim = z_dim
        self.planar_weight = nn.ParameterList([nn.Parameter(torch.Tensor(1, z_dim)) for i in range(flow_layers)])
        self.planar_bias = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(flow_layers)])
        self.planar_scale = nn.ParameterList([nn.Parameter(torch.Tensor(1, z_dim)) for i in range(flow_layers)])
        for i in range(self.flow):
            self.planar_weight[i].data.uniform_(-0.01, 0.01)
            self.planar_scale[i].data.uniform_(-0.01, 0.01)
            self.planar_bias[i].data.uniform_(-0.01, 0.01)
        # Radial flow variables
        self.radial_initial = nn.ParameterList([nn.Parameter(torch.Tensor(1, z_dim)) for i in range(flow_layers)])
        self.radial_alpha = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(flow_layers)])
        self.radial_beta = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for i in range(flow_layers)])
        for i in range(self.flow):
            self.radial_initial[i].data.uniform_(-0.01, 0.01)
            self.radial_alpha[i].data.uniform_(0.01, 1.0)
            self.radial_beta[i].data.uniform_(-0.01, 0.01)

    def _run_flow(self, z, train=True):
        loss = torch.zeros(z.size(0)).to(z.device)
        for i in range(self.flow):
            # Even step of flow
            if i % 2 == 0:
                # Maintain invertibility
                margin = torch.mm(self.planar_scale[i], self.planar_weight[i].t()).item()
                if margin < -1:
                    component = torch.Tensor([-1 + np.log(1 + np.e ** margin) - margin]).to(z.device)
                    self.planar_scale[i].data += component * self.planar_weight[i] / self.planar_weight[i].norm(2)

                activation = F.linear(z, self.planar_weight[i], self.planar_bias[i])
                z = z + self.planar_scale[i] * F.tanh(activation)
                if train:
                    psi = (1 - F.tanh(activation) ** 2) * self.planar_weight[i]
                    det_grad = 1 + torch.mm(psi, self.planar_scale[i].t())
                    loss += torch.log(det_grad.abs().squeeze(1) + 1e-7)  # for numerical stability
            # Odd step of flow
            else:
                # Maintain invertibility
                if self.radial_beta[i].item() < -self.radial_alpha[i].item():
                    self.radial_beta[i].data = -self.radial_alpha[i] + torch.log(1 + np.e ** self.radial_beta[i])

                radius = z - self.radial_initial[i]
                activation = (1 / (self.radial_alpha[i] + radius.norm(p=2, dim=1))).unsqueeze(1)
                z = z + self.radial_beta[i] * activation * radius
                if train:
                    diagonal = (1 + self.radial_beta[i] * activation) ** (self.z_dim - 1)
                    det_grad = (diagonal * (1 + self.radial_beta[i] * activation + self.radial_beta[i] * (
                        -activation ** 2) * radius.norm(2)))
                    loss += torch.log(det_grad.abs().squeeze(1) + 1e-7)  # for numerical stability
        z.flowed = True
        if train:
            loss = loss.mean(0)  # batch averaging
            return z, loss
        else:
            return z
