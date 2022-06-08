Mu = lambda x, P: torch.mean(P(x))
Not = lambda x: 1 - x

Or = lambda *x: torch.max(torch.stack(x, dim=0), dim=0).values
And = lambda *x: torch.min(torch.stack(x,dim=0), dim=0).values


# Less `s
def Or(*x):
    x = torch.stack(x, dim=0)
    return torch.mean(torch.topk(x, max(1, int(len(x)*0.25)), dim=0).values, dim=0)

def And(*x):
    x = torch.stack(x, dim=0)
    return torch.mean(torch.topk(x, max(1, int(len(x)*0.25)), dim=0, largest=False).values, dim=0

samples = 200

def Every(x, P, n=samples):
    x = x.sample_uniform(n)
    Q = P(x)
    return And(*Q)

def Each(x, P, n=samples):
    x = x.sample_uniform(n)
    return And(*Q)

def No(x, P, n=samples):
    x = x.sample_uniform(n)
    Q = Or(Not(P(x)), Not(stuff['exists'](x)))
    return And(*Q)

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
