import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta

def quadratic_formula(a,b,c):
    r = torch.sqrt(torch.square(b) - 4*a*c)
    x1 = (-b + r)/(2*a)
    return x1.view(-1, 1)

def hyperbolic_cdist(a, b):
    # TODO
    return torch.log(x + z)


class PredicateTransform(nn.Module):
    
    def __init__(self, dim, depth, internal_width=64):
        super(PredicateTransform, self).__init__()
        self.internal_width = internal_width
        self.inn = Ff.SequenceINN(dim)
        for k in range(depth):
            self.inn.append(Fm.AllInOneBlock, subnet_constructor=lambda x, y: self.subnet_fc(x,y), permute_soft=True)
        self.inn.apply(PredicateTransform.init_weights)
        
    def subnet_fc(self, c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, self.internal_width), nn.GELU(),
                        nn.Linear(self.internal_width,  c_out))

    def forward(self, x):
        y, _ = self.inn(x,jac=False)
        return y
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            m.bias.data.fill_(0.01)
    
    def inverse(self, x):
        y, _ = self.inn(x, rev=True, jac=False)
        return y
    
            
def sample_points_on_unit_sphere(n, dim):
    y = torch.normal(0, 1, size=(n, dim))
    y /= torch.norm(y, dim=-1).view(n, -1)
    return y


def sample_within_unit_ball(n, dim, center, dist):
    assert dim == 2
    theta = torch.rand(n)*2*3.141592653589
    d = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)
    a = d.matmul(d.T).diagonal()
    b = 2*d.matmul(center)
    c = center.dot(center) - 1
    t1 = quadratic_formula(a, b, c)
    p = d*t1 + center
    r = dist.rsample((n,)) * torch.norm(p - center, dim=-1)
    return (d*r.view(-1, 1)) + center

class EPredicate(nn.Module):
    def __init__(self, dim, depth=5, width=5):
        super(EPredicate, self).__init__()
        self.dim = dim
        self.beta = torch.nn.Parameter(torch.tensor(0.))
        self.scale = 10.0
        self.center_length_ = torch.nn.Parameter(torch.rand(1)*0.5)
        self.center_theta_ = torch.nn.Parameter(torch.rand(1)*2*3.14159265358979323)
        self.transform = PredicateTransform(self.dim, depth=depth, internal_width=width)
    
    @property 
    def center(self):
        direction = torch.cat((torch.sin(self.center_theta_), torch.cos(self.center_theta_)))
        return direction*0.75*torch.tanh(self.center_length_)
    
    def get_intersection_with_unit_sphere(self, y):
        d = y - self.center
        a = d.matmul(d.T).diagonal()
        b = 2*d.matmul(y.T).diagonal()
        c = y.matmul(y.T).diagonal() - 1
        t1 = quadratic_formula(a, b, c)
        p = d*t1 + y
        return p
    
    def get_draw_paths(self, x, y, resolution=25):
        z = torch.vstack((x, y))
        z = self.transform(z)
        f_x = z[:len(x), :]
        f_y = z[len(x):, :]
        diffs = (f_y.view(1, -1, 2) -  f_x.view(-1, 1, 2))
        t = torch.linspace(0, 1, resolution)
        pairs = (diffs.unsqueeze(-1).expand(-1, -1, -1, resolution)*t + f_x.view(-1, 1, 2, 1)).view(-1, 2, resolution).permute(0, 2, 1)
        pairs = pairs.contiguous()
        pairs = self.inverse(pairs.view(-1, 2)).view(*pairs.shape)
        return pairs

    def distance(self, x, y, resolution=25):
        z = torch.vstack((x, y))
        z = self.transform(z)
        f_x = z[:len(x), :]
        f_y = z[len(x):, :]
        return torch.cdist(f_x, f_y)
    
    def forward(self, X):
        if not self.training:
            return self.get_value(X)
        y = self.transform(X) 
        p = self.get_intersection_with_unit_sphere(y)

        return torch.sigmoid(self.scale*((p-self.center).norm(dim=-1) - (y-self.center).norm(dim=-1)))
        
    def get_value(self, X):
        y = self.transform(X)
        p = self.get_intersection_with_unit_sphere(y)
        return ((p-self.center).norm(dim=-1) >= (y-self.center).norm(dim=-1)).to(dtype=torch.float)
    
    def inverse(self, x):
        return self.transform.inverse(x)
    
    def surface_given_sphere(self, sphere):
        return self.inverse(sphere)# - self.origin
    
    def surface(self, n=10):
        sphere = sample_points_on_unit_sphere(n, self.dim)
        return self.inverse(sphere)# - self.origin
    
    def sample(self, n=10):
        sphere = sample_within_unit_ball(n, self.dim, self.center, Beta(1, torch.tanh(self.beta) + 2))
        return self.inverse(sphere)
    
    def sample_uniform(self, n=10):
        sphere = sample_points_on_unit_sphere(n, self.dim) * 0.95
        sphere *= torch.pow(torch.rand((n, 1)), 1/np.sqrt(2)).detach()
        return self.inverse(sphere)

class EAdjective(nn.Module):
    def __init__(self, dim, depth=5, width=5, in_dim=None):
        super(EAdjective, self).__init__()
        self.dim = dim
        self.width = width
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = self.dim


        l = [
                nn.Linear(self.dim, self.width),
                nn.ReLU()
            ] + \
            max(0, depth-2)*[
                nn.Linear(self.width, self.width),
                nn.ReLU()
            ]+ \
            [
                nn.Linear(self.width, self.in_dim),
                nn.ReLU()
            ]
        self.transform = nn.Sequential(*l)

    def forward(self, x, y, relative=lambda x: x):
        x = self.transform(x)
        y = self.transform(y)
        z = F.relu(y.view(1, -1, self.in_dim) - x.view(-1, 1, self.in_dim))
        z = torch.norm(z, dim=-1) ** 2
        if not self.training:
            return z == 0
        return z

    def inverse(self, x):
        return self.transform.inverse(x)
