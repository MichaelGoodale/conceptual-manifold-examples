import torch
import torch.nn.functional as F
import numpy as np

from conceptual_manifolds.EModels import sample_points_on_unit_sphere

DEFAULT_SAMPLES = 200

Mu = lambda x, P: torch.mean(P(x))
Not = lambda x: 1 - x

Or = lambda *x: torch.max(torch.stack(x, dim=0), dim=0).values
And = lambda *x: torch.min(torch.stack(x,dim=0), dim=0).values

def MuSample(x, P, n=DEFAULT_SAMPLES):
    x = x.sample(n)
    return Mu(x, P)

def All(x, P, n=DEFAULT_SAMPLES):
    x = x.sample(n)
    Q = P(x)
    return And(*P(x))

def No(x, P, n=DEFAULT_SAMPLES):
    x = x.sample(n)
    Q = Or(Not(P(x)), Not(stuff['exists'](x)))
    return And(*Q)

def Is(P, Q, n=DEFAULT_SAMPLES):
    sphere = sample_points_on_unit_sphere(n, 2)
    ps = P.surface_given_sphere(sphere)
    qs = Q.surface_given_sphere(sphere)
    d = torch.norm(ps-qs, dim=-1)
    return torch.mean(torch.exp(-d))


def MoreThan(R, a, b, relative=lambda x: x, samples=DEFAULT_SAMPLES, cutoff=0.75):
    if not torch.is_tensor(a):
        a_samples = a.sample(samples)
    else:
        a_samples = a
    b_samples = b.sample(samples)
    d = R(b_samples, a_samples, relative)
    return torch.exp(-d).mean(dim=0)

def Pos(R, a, scale, relative, samples=DEFAULT_SAMPLES, cutoff=0.75):
    return MoreThan(R, a, scale, lambda x: relative.transform(x), samples, cutoff)

def Similar(P, Q, R, n=DEFAULT_SAMPLES):
    x = P.sample(int(np.sqrt(n)))
    y = Q.sample(int(np.sqrt(n)))
    return torch.exp(-R.distance(x, y)).mean()
