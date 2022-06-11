
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm

import conceptual_manifolds.EModels as EModels
import conceptual_manifolds.language as lg
from conceptual_manifolds.train import train_model
from conceptual_manifolds.plots import generate_plot

D = 2
W = 4

stuff = {
    'superman': EModels.EPredicate(2, depth=D, width=W),
    'clark_kent': EModels.EPredicate(2, depth=D, width=W),
    'flies': EModels.EPredicate(2, depth=D, width=W),
    'works_at_the_daily_planet': EModels.EPredicate(2, depth=D, width=W),
}
adjs = {
}
                 
def get_props(stuff, adjs):
    prop = []    
    prop.append(lg.Is(stuff['superman'], stuff['clark_kent']))
    prop.append(lg.MuSample(stuff['superman'], stuff['flies']))
    prop.append(lg.Not(lg.MuSample(stuff['clark_kent'], stuff['flies'])))
    prop.append(lg.MuSample(stuff['clark_kent'], stuff['works_at_the_daily_planet']))
    prop.append(lg.Not(lg.MuSample(stuff['superman'], stuff['works_at_the_daily_planet'])))
    return prop

stuff, adjs = train_model(stuff, adjs, get_props, N=3000)
generate_plot(stuff, adjs, draw_distribution=['clark_kent', 'superman'])
