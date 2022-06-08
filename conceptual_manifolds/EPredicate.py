

import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.beta import Beta
def quadratic_formula(a,b,c):
    r = torch.sqrt(torch.square(b) - 4*a*c)
    x1 = (-b + r)/(2*a)
    #x2 = (-b - r)/(2*a)
    return x1.view(-1, 1)#, x2.view(-1, 1)


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


def rejection_sampling(n, dim, center, dist):
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
        self.scale_ = torch.nn.Parameter(torch.tensor(1.0))
        self.center_length_ = torch.nn.Parameter(torch.rand(1))
        self.center_theta_ = torch.nn.Parameter(torch.rand(1)*2*3.14159265358979323)
        self.transform = PredicateTransform(self.dim, depth=depth, internal_width=width)
    
    @property 
    def center(self):
        direction = torch.cat((torch.sin(self.center_theta_), torch.cos(self.center_theta_)))
        return direction*0.75*torch.tanh(self.center_length_)
    
    @property 
    def scale(self):
        return 10.0
        
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
        #pairs = self.get_draw_paths(x, y, resolution=resolution)
        #a, b = pairs[:, :-1, :].contiguous(), pairs[:, 1:, :].contiguous()
        #return F.pairwise_distance(a,b).sum(dim=-1)
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
        #return torch.exp(((p-self.center).norm(dim=-1) - (y-self.center).norm(dim=-1)))
        
        
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
        sphere = rejection_sampling(n, self.dim, self.center, Beta(1, torch.tanh(self.beta) + 2))
        return self.inverse(sphere)
    
    def sample_uniform(self, n=10):
        sphere = sample_points_on_unit_sphere(n, self.dim) * 0.95
        sphere *= torch.pow(torch.rand((n, 1)), 1/np.sqrt(2)).detach()
        return self.inverse(sphere)


Or = lambda *x: torch.max(torch.stack(x, dim=0), dim=0).values
And = lambda *x: torch.min(torch.stack(x,dim=0), dim=0).values

def Or(*x):
    x = torch.stack(x, dim=0)
    return torch.mean(torch.topk(x, max(1, int(len(x)*0.25)), dim=0).values, dim=0)

def And(*x):
    x = torch.stack(x, dim=0)
    return torch.mean(torch.topk(x, max(1, int(len(x)*0.25)), dim=0, largest=False).values, dim=0)

Gen = lambda x, P: torch.mean(P(x))
Not = lambda x: 1 - x

samples = 200

def Every(x, P, n=samples):
    x = x.sample_uniform(n)
    Q = P(x)
    return And(*Q)

def Each(x, P, n=samples):
    x = x.sample_uniform(n)
    Q = Or(P(x), Not(stuff['exists'](x)))
    return And(*Q)

def No(x, P, n=samples):
    x = x.sample_uniform(n)
    Q = Or(Not(P(x)), Not(stuff['exists'](x)))
    return And(*Q)

def ThereAre(P, n=samples):
    return Or(*stuff['exists'](P.sample_uniform(n)))

def Is(P, Q, n=samples):
    sphere = sample_points_on_unit_sphere(n, 2)
    ps = P.surface_given_sphere(sphere)
    qs = Q.surface_given_sphere(sphere)
    d = torch.norm(ps-qs, dim=-1)
    return torch.mean(torch.exp(-d))

def Similar(P, Q, R=None, n=samples, alpha=1.0):
    x = P.sample(int(np.sqrt(n)))
    y = Q.sample(int(np.sqrt(n)))
    if R is None:
        return torch.exp(-torch.cdist(x.view(1, -1, 2), y.view(1, -1, 2))).mean()
    else:
        return torch.exp(-R.distance(x, y)*alpha).mean()

class EAdjective(nn.Module):
    def __init__(self, dim, depth=5, width=5, in_dim=None):
        super(EAdjective, self).__init__()
        self.dim = dim
        self.width = width
        self.in_dim = in_dim
        if self.in_dim is None:
            self.in_dim = self.dim 
            
        l = [
                nn.Linear(self.dim * 2, self.width),
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
        x = self.transform(torch.cat((x, relative(torch.rand_like(x))), dim=-1))
        y = self.transform(torch.cat((y, relative(torch.rand_like(y))), dim=-1))
        z = F.relu(y.view(1, -1, self.in_dim) - x.view(-1, 1, self.in_dim)) ** 2
        if not self.training:
            return z == 0
        return torch.exp(-z)
    
    def inverse(self, x):
        return self.transform.inverse(x)


