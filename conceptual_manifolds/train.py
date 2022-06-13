from itertools import chain

import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

import conceptual_manifolds.EModels 
import conceptual_manifolds.language 

def train_model(stuff, adjs, props_function, N=1000, lr=0.003):
    params = sum([list(x.parameters()) for _, x in chain(stuff.items(), adjs.items())], [])
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    avg_loss = 0
    with tqdm(range(N)) as pbar:
        for i in pbar:
            prop = props_function(stuff, adjs)
            for _, s in stuff.items():
                prop.append(F.relu(s.sample_uniform().norm(dim=-1) - 10).mean() * 0.75)
                
            p = torch.stack(prop)
            p[:-len(stuff)] = -torch.log(p[:-len(stuff)]+1e-5)
            loss = p.mean()
            optimizer.zero_grad()
            
            loss.backward()
            avg_loss += (p[:-len(stuff)].mean()).item()
            for _, s in stuff.items():
                torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
            optimizer.step()
            if i % 10 == 0 and i > 0:
                pbar.set_description(f"Loss {avg_loss/10:.3f}")
                avg_loss = 0
    return stuff, adjs
